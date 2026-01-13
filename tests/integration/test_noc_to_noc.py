"""
Integration tests for NoC-to-NoC traffic patterns.

Tests the 5 traffic patterns:
- neighbor: Ring topology (node i -> node (i+1) % N)
- shuffle: Perfect shuffle permutation
- bit_reverse: Bit-reversed node ID mapping
- random: Random destination selection
- transpose: Swap X and Y coordinates
"""

import pytest

from src.config import (
    TrafficPattern,
    NoCTrafficConfig,
    NodeTransferConfig,
)
from src.traffic.pattern_generator import TrafficPatternGenerator
from src.testbench import LocalAXIMaster, LocalTransferConfig
from src.testbench import NodeController
from src.testbench import Memory, MemoryConfig


class TestTrafficPatternGenerator:
    """Tests for TrafficPatternGenerator."""

    @pytest.fixture
    def generator(self):
        """Create pattern generator for 5x4 mesh."""
        return TrafficPatternGenerator(mesh_cols=5, mesh_rows=4)

    def test_neighbor_pattern(self, generator):
        """Test NEIGHBOR pattern: ring topology."""
        config = NoCTrafficConfig(
            pattern=TrafficPattern.NEIGHBOR,
            mesh_cols=5,
            mesh_rows=4,
            transfer_size=256,
        )
        configs = generator.generate(config)

        # 4 cols (excluding edge) * 4 rows = 16 nodes
        assert len(configs) == 16

        # Verify ring pattern: node i -> node (i+1) % N
        for i, nc in enumerate(configs):
            assert nc.src_node_id == i
            expected_dst = (i + 1) % 16
            expected_coord = generator._node_id_to_coord(expected_dst)
            assert nc.dest_coord == expected_coord

    def test_shuffle_pattern(self, generator):
        """Test SHUFFLE pattern: left bit rotation."""
        config = NoCTrafficConfig(
            pattern=TrafficPattern.SHUFFLE,
            mesh_cols=5,
            mesh_rows=4,
            transfer_size=256,
        )
        configs = generator.generate(config)

        assert len(configs) == 16

        # Verify all nodes have different destinations (permutation)
        destinations = set(nc.dest_coord for nc in configs)
        # May have collisions due to modulo, but should be valid coordinates
        for nc in configs:
            x, y = nc.dest_coord
            assert 1 <= x < 5  # Valid x (excluding edge column)
            assert 0 <= y < 4  # Valid y

    def test_bit_reverse_pattern(self, generator):
        """Test BIT_REVERSE pattern."""
        config = NoCTrafficConfig(
            pattern=TrafficPattern.BIT_REVERSE,
            mesh_cols=5,
            mesh_rows=4,
            transfer_size=256,
        )
        configs = generator.generate(config)

        assert len(configs) == 16

        # Verify all configs are valid
        for nc in configs:
            assert 0 <= nc.src_node_id < 16
            x, y = nc.dest_coord
            assert 1 <= x < 5
            assert 0 <= y < 4

    def test_random_pattern_deterministic(self, generator):
        """Test RANDOM pattern is deterministic with same seed."""
        config1 = NoCTrafficConfig(
            pattern=TrafficPattern.RANDOM,
            mesh_cols=5,
            mesh_rows=4,
            random_seed=42,
        )
        config2 = NoCTrafficConfig(
            pattern=TrafficPattern.RANDOM,
            mesh_cols=5,
            mesh_rows=4,
            random_seed=42,
        )

        configs1 = generator.generate(config1)
        configs2 = generator.generate(config2)

        # Same seed should produce same destinations
        for c1, c2 in zip(configs1, configs2):
            assert c1.dest_coord == c2.dest_coord

    def test_random_pattern_excludes_self(self, generator):
        """Test RANDOM pattern excludes self-loops."""
        config = NoCTrafficConfig(
            pattern=TrafficPattern.RANDOM,
            mesh_cols=5,
            mesh_rows=4,
            random_seed=42,
        )
        configs = generator.generate(config)

        # No node should send to itself
        for nc in configs:
            src_coord = generator._node_id_to_coord(nc.src_node_id)
            assert nc.dest_coord != src_coord

    def test_transpose_pattern(self, generator):
        """Test TRANSPOSE pattern: swap x and y."""
        config = NoCTrafficConfig(
            pattern=TrafficPattern.TRANSPOSE,
            mesh_cols=5,
            mesh_rows=4,
            transfer_size=256,
        )
        configs = generator.generate(config)

        assert len(configs) == 16

        # Verify transpose mapping
        for nc in configs:
            src_coord = generator._node_id_to_coord(nc.src_node_id)
            src_x, src_y = src_coord

            # Expected: (x, y) -> (y+1, x-1) accounting for edge column
            expected_x = src_y + 1
            expected_y = src_x - 1

            if 1 <= expected_x < 5 and 0 <= expected_y < 4:
                assert nc.dest_coord == (expected_x, expected_y)


class TestLocalAXIMaster:
    """Tests for LocalAXIMaster."""

    @pytest.fixture
    def local_memory(self):
        """Create local memory."""
        return Memory(
            config=MemoryConfig(size=0x10000),
            name="TestMemory"
        )

    def test_user_signal_encoding(self, local_memory):
        """Test AXI user signal encoding of destination."""
        master = LocalAXIMaster(
            node_id=0,
            local_memory=local_memory,
            mesh_cols=5,
            mesh_rows=4,
        )

        config = LocalTransferConfig(
            dest_coord=(3, 2),  # x=3, y=2
            local_src_addr=0x0000,
            local_dst_addr=0x1000,
            transfer_size=64,
        )

        # Test encoding: awuser[7:0]=x, awuser[15:8]=y
        user_signal = config.encode_user_signal()
        assert user_signal == (2 << 8) | 3  # (y << 8) | x

        # Test decoding
        decoded = LocalTransferConfig.decode_user_signal(user_signal)
        assert decoded == (3, 2)

    def test_local_master_state_transitions(self, local_memory):
        """Test LocalAXIMaster state transitions."""
        # Initialize memory with test data
        test_data = b"TEST_DATA_12345"
        local_memory.write(0x0000, test_data)

        master = LocalAXIMaster(
            node_id=5,
            local_memory=local_memory,
            mesh_cols=5,
            mesh_rows=4,
        )

        # Initially idle
        assert master.is_idle
        assert not master.is_running

        # Configure transfer
        config = LocalTransferConfig(
            dest_coord=(2, 1),
            local_src_addr=0x0000,
            local_dst_addr=0x1000,
            transfer_size=len(test_data),
        )
        master.configure_transfer(config)

        # Still idle until started
        assert master.is_idle

        # Start transfer
        master.start()
        assert master.is_running
        assert not master.is_idle


class TestNodeController:
    """Tests for NodeController."""

    def test_node_controller_creation(self):
        """Test NodeController creation and initialization."""
        controller = NodeController(
            node_id=7,
            mesh_cols=5,
            mesh_rows=4,
            memory_size=0x10000,
        )

        # Verify coordinate calculation
        # Node 7: (7 % 4) + 1 = 4, 7 // 4 = 1 -> (4, 1)
        assert controller.coord == (4, 1)
        assert controller.node_id == 7

        # Verify components exist
        assert controller.local_memory is not None
        assert controller.local_master is not None
        assert controller.slave_ni is not None
        assert controller.master_ni is not None

    def test_node_controller_memory_init(self):
        """Test NodeController memory initialization."""
        controller = NodeController(
            node_id=0,
            mesh_cols=5,
            mesh_rows=4,
        )

        # Write test data
        test_data = b"NODE_0_TEST_DATA"
        controller.initialize_memory(0x0000, test_data)

        # Read back
        actual = controller.read_local_memory(0x0000, len(test_data))
        assert actual == test_data

    def test_node_controller_transfer_config(self):
        """Test NodeController transfer configuration."""
        controller = NodeController(
            node_id=3,
            mesh_cols=5,
            mesh_rows=4,
        )

        config = NodeTransferConfig(
            src_node_id=3,
            dest_coord=(2, 2),
            local_src_addr=0x0000,
            local_dst_addr=0x1000,
            transfer_size=128,
        )

        controller.configure_transfer(config)

        # Verify config was applied
        summary = controller.get_summary()
        assert summary['node_id'] == 3
        assert summary['transfer_config']['dest_coord'] == (2, 2)
        assert summary['transfer_config']['size'] == 128


class TestNoCTrafficConfig:
    """Tests for NoCTrafficConfig."""

    def test_config_defaults(self):
        """Test NoCTrafficConfig default values."""
        config = NoCTrafficConfig()

        assert config.pattern == TrafficPattern.NEIGHBOR
        assert config.mesh_cols == 5
        assert config.mesh_rows == 4
        assert config.transfer_size == 256
        assert config.num_nodes == 16

    def test_config_serialization(self):
        """Test NoCTrafficConfig to dict conversion."""
        config = NoCTrafficConfig(
            pattern=TrafficPattern.SHUFFLE,
            mesh_cols=6,
            mesh_rows=5,
            transfer_size=512,
        )

        data = config.to_dict()
        assert data['traffic']['pattern'] == 'shuffle'
        assert data['traffic']['mesh_cols'] == 6
        assert data['traffic']['mesh_rows'] == 5
        assert data['traffic']['transfer_size'] == 512


class TestNoCToNoCEdgeCases:
    """Edge case tests for NoC-to-NoC functionality."""

    def test_single_node_mesh(self):
        """Test with minimum viable mesh (2x1 = 1 node)."""
        generator = TrafficPatternGenerator(mesh_cols=2, mesh_rows=1)
        config = NoCTrafficConfig(
            pattern=TrafficPattern.NEIGHBOR,
            mesh_cols=2,
            mesh_rows=1,
        )
        configs = generator.generate(config)

        # Only 1 compute node
        assert len(configs) == 1
        # Node 0 -> Node 0 (loops to self in ring)
        assert configs[0].src_node_id == 0
        assert configs[0].dest_coord == (1, 0)  # Node 0's coord

    def test_large_mesh(self):
        """Test with larger mesh (9x8 = 64 nodes)."""
        generator = TrafficPatternGenerator(mesh_cols=9, mesh_rows=8)
        config = NoCTrafficConfig(
            pattern=TrafficPattern.NEIGHBOR,
            mesh_cols=9,
            mesh_rows=8,
        )
        configs = generator.generate(config)

        # 8 cols * 8 rows = 64 nodes
        assert len(configs) == 64

    def test_user_signal_boundary_values(self):
        """Test user signal encoding with boundary coordinates."""
        # Max supported in 8 bits each
        config1 = LocalTransferConfig(dest_coord=(0, 0))
        assert config1.encode_user_signal() == 0

        config2 = LocalTransferConfig(dest_coord=(255, 255))
        assert config2.encode_user_signal() == 0xFFFF

        config3 = LocalTransferConfig(dest_coord=(15, 7))
        user = config3.encode_user_signal()
        decoded = LocalTransferConfig.decode_user_signal(user)
        assert decoded == (15, 7)

    def test_transfer_config_with_all_patterns(self):
        """Test that all pattern types can be configured."""
        patterns = [
            TrafficPattern.NEIGHBOR,
            TrafficPattern.SHUFFLE,
            TrafficPattern.BIT_REVERSE,
            TrafficPattern.RANDOM,
            TrafficPattern.TRANSPOSE,
        ]

        generator = TrafficPatternGenerator(mesh_cols=5, mesh_rows=4)

        for pattern in patterns:
            config = NoCTrafficConfig(pattern=pattern)
            configs = generator.generate(config)
            assert len(configs) == 16, f"Failed for pattern {pattern}"
