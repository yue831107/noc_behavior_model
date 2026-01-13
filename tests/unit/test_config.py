"""
Tests for configuration module.

Tests cover:
1. TransferMode enum
2. TrafficPattern enum
3. TimingConfig factory methods
4. TransferConfig validation and serialization
5. NoCTrafficConfig
6. YAML config loading functions
"""

import pytest
from pathlib import Path
from tempfile import TemporaryDirectory

from src.config import (
    TransferMode,
    TrafficPattern,
    TimingConfig,
    TransferConfig,
    NodeTransferConfig,
    NoCTrafficConfig,
    load_transfer_config,
    load_transfer_configs,
    load_noc_traffic_config,
    get_default_transfer_config,
    _parse_transfer_config,
    _parse_single_transfer,
    _parse_noc_traffic_config,
)


class TestTransferMode:
    """Test TransferMode enum."""

    def test_is_read_for_read_modes(self):
        """is_read should return True for read modes."""
        assert TransferMode.BROADCAST_READ.is_read is True
        assert TransferMode.GATHER.is_read is True

    def test_is_read_for_write_modes(self):
        """is_read should return False for write modes."""
        assert TransferMode.BROADCAST.is_read is False
        assert TransferMode.SCATTER.is_read is False

    def test_is_write_for_write_modes(self):
        """is_write should return True for write modes."""
        assert TransferMode.BROADCAST.is_write is True
        assert TransferMode.SCATTER.is_write is True

    def test_is_write_for_read_modes(self):
        """is_write should return False for read modes."""
        assert TransferMode.BROADCAST_READ.is_write is False
        assert TransferMode.GATHER.is_write is False


class TestTrafficPattern:
    """Test TrafficPattern enum."""

    def test_all_patterns_exist(self):
        """All expected patterns should exist."""
        assert TrafficPattern.NEIGHBOR.value == "neighbor"
        assert TrafficPattern.SHUFFLE.value == "shuffle"
        assert TrafficPattern.BIT_REVERSE.value == "bit_reverse"
        assert TrafficPattern.RANDOM.value == "random"
        assert TrafficPattern.TRANSPOSE.value == "transpose"


class TestTimingConfig:
    """Test TimingConfig dataclass."""

    def test_default_values(self):
        """Default config should have hardware-accurate values."""
        config = TimingConfig()
        assert config.aw_per_cycle == 1
        assert config.w_per_cycle == 1
        assert config.ar_per_cycle == 1
        assert config.flits_per_cycle == 1
        assert config.txn_per_cycle == 1

    def test_fast_mode(self):
        """fast() should return unlimited injection config."""
        config = TimingConfig.fast()
        assert config.aw_per_cycle == 999
        assert config.w_per_cycle == 999
        assert config.ar_per_cycle == 999
        assert config.flits_per_cycle == 999
        assert config.txn_per_cycle == 999

    def test_hardware_mode(self):
        """hardware() should return default config."""
        config = TimingConfig.hardware()
        assert config.aw_per_cycle == 1
        assert config.w_per_cycle == 1


class TestTransferConfig:
    """Test TransferConfig dataclass."""

    def test_is_read_property(self):
        """is_read should check transfer_mode."""
        config = TransferConfig(transfer_mode=TransferMode.BROADCAST_READ)
        assert config.is_read is True

        config = TransferConfig(transfer_mode=TransferMode.BROADCAST)
        assert config.is_read is False

    def test_is_write_property(self):
        """is_write should check transfer_mode."""
        config = TransferConfig(transfer_mode=TransferMode.SCATTER)
        assert config.is_write is True

        config = TransferConfig(transfer_mode=TransferMode.GATHER)
        assert config.is_write is False

    def test_effective_read_size_with_zero(self):
        """effective_read_size should use src_size when read_size is 0."""
        config = TransferConfig(src_size=512, read_size=0)
        assert config.effective_read_size == 512

    def test_effective_read_size_with_value(self):
        """effective_read_size should use read_size when non-zero."""
        config = TransferConfig(src_size=512, read_size=256)
        assert config.effective_read_size == 256

    def test_get_target_node_list_all(self):
        """get_target_node_list should return all nodes for 'all'."""
        config = TransferConfig(target_nodes="all")
        nodes = config.get_target_node_list(total_nodes=16)
        assert nodes == list(range(16))

    def test_get_target_node_list_explicit_list(self):
        """get_target_node_list should return explicit list as-is."""
        config = TransferConfig(target_nodes=[1, 3, 5])
        nodes = config.get_target_node_list()
        assert nodes == [1, 3, 5]

    def test_get_target_node_list_range(self):
        """get_target_node_list should parse range format."""
        config = TransferConfig(target_nodes="range:0-7")
        nodes = config.get_target_node_list()
        assert nodes == list(range(8))

    def test_get_target_node_list_invalid_raises(self):
        """get_target_node_list should raise for invalid format."""
        config = TransferConfig(target_nodes="invalid")
        with pytest.raises(ValueError, match="Invalid target_nodes format"):
            config.get_target_node_list()

    def test_validate_src_size_positive(self):
        """validate should raise for non-positive src_size."""
        config = TransferConfig(src_size=0)
        with pytest.raises(ValueError, match="src_size must be positive"):
            config.validate()

    def test_validate_max_burst_len_range(self):
        """validate should raise for invalid max_burst_len."""
        config = TransferConfig(max_burst_len=0)
        with pytest.raises(ValueError, match="max_burst_len must be 1-256"):
            config.validate()

        config = TransferConfig(max_burst_len=300)
        with pytest.raises(ValueError, match="max_burst_len must be 1-256"):
            config.validate()

    def test_validate_beat_size(self):
        """validate should raise for non-power-of-2 beat_size."""
        config = TransferConfig(beat_size=3)
        with pytest.raises(ValueError, match="beat_size must be power of 2"):
            config.validate()

    def test_validate_max_outstanding(self):
        """validate should raise for invalid max_outstanding."""
        config = TransferConfig(max_outstanding=0)
        with pytest.raises(ValueError, match="max_outstanding must be at least 1"):
            config.validate()

    def test_to_dict(self):
        """to_dict should return serializable dict."""
        config = TransferConfig(
            src_addr=0x1000,
            src_size=256,
            dst_addr=0x2000,
            target_nodes=[1, 2],
        )
        d = config.to_dict()

        assert 'transfer' in d
        assert d['transfer']['src_addr'] == 0x1000
        assert d['transfer']['src_size'] == 256
        assert d['transfer']['dst_addr'] == 0x2000
        assert d['transfer']['target_nodes'] == [1, 2]

    def test_save_and_load(self):
        """save should create loadable YAML file."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_config.yaml"

            config = TransferConfig(
                src_addr=0x1000,
                src_size=512,
                target_nodes=[0, 1, 2],
            )
            config.save(path)

            loaded = load_transfer_config(path)

            assert loaded.src_addr == config.src_addr
            assert loaded.src_size == config.src_size
            assert loaded.target_nodes == config.target_nodes


class TestLoadTransferConfig:
    """Test load_transfer_config function."""

    def test_load_not_found_raises(self):
        """load_transfer_config should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_transfer_config("nonexistent.yaml")

    def test_load_nested_format(self):
        """load_transfer_config should handle nested format."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "config.yaml"
            path.write_text("""
transfer:
  src_addr: 0x1000
  src_size: 256
  dst_addr: 0x2000
  target_nodes: [1, 2, 3]
  transfer_mode: scatter
""")
            config = load_transfer_config(path)

            assert config.src_addr == 0x1000
            assert config.src_size == 256
            assert config.transfer_mode == TransferMode.SCATTER


class TestLoadTransferConfigs:
    """Test load_transfer_configs function for multi-transfer."""

    def test_load_multi_transfer_format(self):
        """load_transfer_configs should handle multiple transfers."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "multi.yaml"
            path.write_text("""
transfers:
  - src_addr: 0x0000
    src_size: 256
    target_nodes: [0, 1]
  - src_addr: 0x1000
    src_size: 512
    target_nodes: [2, 3]
""")
            configs = load_transfer_configs(path)

            assert len(configs) == 2
            assert configs[0].src_addr == 0
            assert configs[0].src_size == 256
            assert configs[1].src_addr == 0x1000
            assert configs[1].src_size == 512

    def test_load_single_transfer_backward_compatible(self):
        """load_transfer_configs should handle single transfer format."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "single.yaml"
            path.write_text("""
transfer:
  src_addr: 0x2000
  src_size: 128
""")
            configs = load_transfer_configs(path)

            assert len(configs) == 1
            assert configs[0].src_addr == 0x2000

    def test_load_not_found_raises(self):
        """load_transfer_configs should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_transfer_configs("nonexistent.yaml")


class TestParseSingleTransfer:
    """Test _parse_single_transfer function."""

    def test_parse_with_hex_string_addr(self):
        """Should parse hex string addresses."""
        data = {
            "src_addr": "0x1000",
            "dst_addr": "0x2000",
            "src_size": 256,
        }
        config = _parse_single_transfer(data)

        assert config.src_addr == 0x1000
        assert config.dst_addr == 0x2000

    def test_parse_with_int_addr(self):
        """Should parse integer addresses."""
        data = {
            "src_addr": 4096,
            "dst_addr": 8192,
            "src_size": 128,
        }
        config = _parse_single_transfer(data)

        assert config.src_addr == 4096
        assert config.dst_addr == 8192


class TestGetDefaultTransferConfig:
    """Test get_default_transfer_config function."""

    def test_returns_default_config(self):
        """Should return default TransferConfig."""
        config = get_default_transfer_config()

        assert isinstance(config, TransferConfig)
        assert config.src_size == 1024
        assert config.transfer_mode == TransferMode.BROADCAST


class TestNodeTransferConfig:
    """Test NodeTransferConfig dataclass."""

    def test_encode_user_signal(self):
        """encode_user_signal should pack coordinates."""
        config = NodeTransferConfig(
            src_node_id=0,
            dest_coord=(3, 2),  # x=3, y=2
        )
        signal = config.encode_user_signal()

        # Signal format: (y << 8) | x
        assert signal == (2 << 8) | 3
        assert signal == 0x0203


class TestNoCTrafficConfig:
    """Test NoCTrafficConfig dataclass."""

    def test_default_values(self):
        """Default config should have expected values."""
        config = NoCTrafficConfig()

        assert config.pattern == TrafficPattern.NEIGHBOR
        assert config.mesh_cols == 5
        assert config.mesh_rows == 4
        assert config.transfer_size == 256

    def test_num_nodes_property(self):
        """num_nodes should compute correctly."""
        config = NoCTrafficConfig(mesh_cols=5, mesh_rows=4)
        # (5-1) * 4 = 16 compute nodes
        assert config.num_nodes == 16

    def test_node_configs_initialized(self):
        """node_configs should be empty list by default."""
        config = NoCTrafficConfig()
        assert config.node_configs == []

    def test_to_dict(self):
        """to_dict should return serializable dict."""
        config = NoCTrafficConfig(
            pattern=TrafficPattern.SHUFFLE,
            transfer_size=512,
        )
        d = config.to_dict()

        assert 'traffic' in d
        assert d['traffic']['pattern'] == 'shuffle'
        assert d['traffic']['transfer_size'] == 512

    def test_save_and_load(self):
        """save should create loadable YAML file."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "traffic.yaml"

            config = NoCTrafficConfig(
                pattern=TrafficPattern.BIT_REVERSE,
                transfer_size=1024,
            )
            config.save(path)

            loaded = load_noc_traffic_config(path)

            assert loaded.pattern == TrafficPattern.BIT_REVERSE
            assert loaded.transfer_size == 1024


class TestLoadNocTrafficConfig:
    """Test load_noc_traffic_config function."""

    def test_load_not_found_raises(self):
        """load_noc_traffic_config should raise FileNotFoundError."""
        with pytest.raises(FileNotFoundError, match="not found"):
            load_noc_traffic_config("nonexistent.yaml")

    def test_load_nested_format(self):
        """Should handle nested traffic format."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "traffic.yaml"
            path.write_text("""
traffic:
  pattern: transpose
  mesh_cols: 6
  mesh_rows: 5
  transfer_size: 512
""")
            config = load_noc_traffic_config(path)

            assert config.pattern == TrafficPattern.TRANSPOSE
            assert config.mesh_cols == 6
            assert config.mesh_rows == 5
            assert config.transfer_size == 512

    def test_load_flat_format(self):
        """Should handle flat format without 'traffic' wrapper."""
        with TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "flat.yaml"
            path.write_text("""
pattern: random
mesh_cols: 4
mesh_rows: 3
""")
            config = load_noc_traffic_config(path)

            assert config.pattern == TrafficPattern.RANDOM
            assert config.mesh_cols == 4
            assert config.mesh_rows == 3
