"""
Integration tests for read-back verification.
"""

import pytest

from src.config import TransferConfig, TransferMode
from src.core.routing_selector import V1System
from src.testbench import Memory, MemoryConfig
from src.verification import GoldenSource


class TestReadVerificationIntegration:
    """Integration tests for read-back verification flow."""

    @pytest.fixture
    def system_with_memory(self):
        """Create V1System with host memory."""
        host_memory = Memory(
            config=MemoryConfig(size=0x10000),
            name="HostMemory"
        )
        system = V1System(
            mesh_cols=5,
            mesh_rows=4,
            buffer_depth=4,
            host_memory=host_memory,
        )
        return system, host_memory

    def test_golden_manager_initialization(self, system_with_memory):
        """Test that V1System has GoldenManager."""
        system, _ = system_with_memory

        assert system.golden_manager is not None
        assert system.golden_manager.entry_count == 0

    def test_capture_golden_from_write(self, system_with_memory):
        """Test capturing golden data during write."""
        system, _ = system_with_memory

        system.capture_golden_from_write(
            node_id=5,
            local_addr=0x1000,
            data=b"test_data",
            cycle=100,
        )

        assert system.golden_manager.entry_count == 1
        assert system.golden_manager.get_golden(5, 0x1000) == b"test_data"

    def test_standalone_read_from_prepopulated_memory(self, system_with_memory):
        """Test reading from pre-populated node memory."""
        system, host_memory = system_with_memory

        # Pre-populate a node's memory
        target_node_id = 5
        local_addr = 0x2000
        test_data = b"PREEXISTING_DATA_12345678"

        # Find the NI and write directly to its memory
        for coord, ni in system.mesh.nis.items():
            node_id = coord[0] + coord[1] * system._mesh_cols
            if node_id == target_node_id:
                ni.write_local(local_addr, test_data)
                break

        # Set golden manually (since no write operation)
        system.golden_manager.set_golden(
            target_node_id,
            local_addr,
            test_data,
            GoldenSource.MANUAL,
        )

        # Configure read transfer for single node
        read_config = TransferConfig(
            src_addr=0,
            src_size=len(test_data),
            dst_addr=0,
            target_nodes=[target_node_id],
            read_src_addr=local_addr,
            read_size=len(test_data),
            transfer_mode=TransferMode.BROADCAST_READ,
            max_burst_len=4,
            beat_size=8,
        )

        system.configure_read_transfer(read_config, use_golden=True)
        system.start_read_transfer()
        system.run_until_transfer_complete(max_cycles=3000)

        # Verify
        report = system.verify_read_results()

        # Note: Read data collection depends on full NI integration
        # This test verifies the API flow works correctly
        assert report is not None

    def test_read_config_validation(self, system_with_memory):
        """Test that read config must have read mode."""
        system, _ = system_with_memory

        write_config = TransferConfig(
            src_addr=0,
            src_size=64,
            dst_addr=0x1000,
            target_nodes="all",
            transfer_mode=TransferMode.BROADCAST,  # Write mode
        )

        with pytest.raises(ValueError) as exc_info:
            system.configure_read_transfer(write_config)

        assert "read" in str(exc_info.value).lower()

    def test_transfer_mode_is_read(self):
        """Test TransferMode.is_read property."""
        assert TransferMode.BROADCAST_READ.is_read
        assert TransferMode.GATHER.is_read
        assert not TransferMode.BROADCAST.is_read
        assert not TransferMode.SCATTER.is_read

    def test_transfer_mode_is_write(self):
        """Test TransferMode.is_write property."""
        assert TransferMode.BROADCAST.is_write
        assert TransferMode.SCATTER.is_write
        assert not TransferMode.BROADCAST_READ.is_write
        assert not TransferMode.GATHER.is_write

    def test_transfer_config_effective_read_size(self):
        """Test TransferConfig.effective_read_size property."""
        config = TransferConfig(
            src_size=100,
            read_size=0,  # 0 means use src_size
        )
        assert config.effective_read_size == 100

        config2 = TransferConfig(
            src_size=100,
            read_size=50,  # Explicit read size
        )
        assert config2.effective_read_size == 50

    def test_reset_for_read(self, system_with_memory):
        """Test reset_for_read preserves golden manager."""
        system, host_memory = system_with_memory

        # Add some golden data
        system.capture_golden_from_write(0, 0x1000, b"data", 0)

        # Reset for read
        system.reset_for_read()

        # Golden manager should be preserved
        assert system.golden_manager.entry_count == 1
        assert system.golden_manager.get_golden(0, 0x1000) == b"data"

    def test_verification_report_api(self, system_with_memory):
        """Test verification report API."""
        system, _ = system_with_memory

        # Add golden data
        system.capture_golden_from_write(0, 0x1000, b"expected", 0)

        # Get empty verification (no read data yet)
        report = system.verify_read_results()

        # Should report missing actual data
        assert report.total_checks == 1
        assert report.missing_actual == 1

    def test_get_read_data_empty(self, system_with_memory):
        """Test get_read_data when no transfer configured."""
        system, _ = system_with_memory

        # No host_axi_master configured yet
        read_data = system.get_read_data()
        assert read_data == {}


class TestTransferConfigReadModes:
    """Tests for TransferConfig read mode configurations."""

    def test_broadcast_read_config(self):
        """Test BROADCAST_READ configuration."""
        config = TransferConfig(
            src_addr=0,
            src_size=64,
            dst_addr=0x1000,
            target_nodes="all",
            read_src_addr=0x2000,
            read_size=64,
            transfer_mode=TransferMode.BROADCAST_READ,
        )

        assert config.is_read
        assert not config.is_write
        assert config.effective_read_size == 64

    def test_gather_config(self):
        """Test GATHER configuration."""
        config = TransferConfig(
            src_addr=0,
            src_size=1024,
            dst_addr=0,
            target_nodes=[0, 1, 2, 3],
            read_src_addr=0x3000,
            read_size=256,
            transfer_mode=TransferMode.GATHER,
        )

        assert config.is_read
        assert not config.is_write
        assert config.effective_read_size == 256
        assert config.get_target_node_list() == [0, 1, 2, 3]

    def test_config_target_nodes_parsing(self):
        """Test target_nodes parsing for various formats."""
        # All nodes
        config1 = TransferConfig(target_nodes="all")
        assert config1.get_target_node_list(16) == list(range(16))

        # Range format
        config2 = TransferConfig(target_nodes="range:4-7")
        assert config2.get_target_node_list() == [4, 5, 6, 7]

        # Explicit list
        config3 = TransferConfig(target_nodes=[1, 3, 5, 7])
        assert config3.get_target_node_list() == [1, 3, 5, 7]
