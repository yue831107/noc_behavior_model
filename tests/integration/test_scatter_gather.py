"""
Integration tests for SCATTER and GATHER transfer modes.

Golden data verification follows these principles:
- SCATTER: Golden captured per-node, key = (node_id, dst_addr)
- GATHER: Golden captured as concatenated result, key = ("host", dst_addr)
"""

import pytest

from src.config import TransferConfig, TransferMode
from src.core.routing_selector import V1System
from src.testbench import Memory, MemoryConfig
from src.verification import GoldenSource, GoldenKey


class TestScatterWrite:
    """Integration tests for SCATTER write mode."""

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

    def test_scatter_config_is_write_mode(self):
        """Test SCATTER is classified as write mode."""
        config = TransferConfig(
            src_addr=0,
            src_size=1024,
            dst_addr=0x1000,
            target_nodes=[0, 1, 2, 3],
            transfer_mode=TransferMode.SCATTER,
        )
        assert config.is_write
        assert not config.is_read

    def test_scatter_data_distribution_calculation(self):
        """Test SCATTER distributes data evenly across nodes."""
        config = TransferConfig(
            src_addr=0,
            src_size=1024,
            dst_addr=0x1000,
            target_nodes=[0, 1, 2, 3],
            transfer_mode=TransferMode.SCATTER,
        )
        target_nodes = config.get_target_node_list()
        portion_size = config.src_size // len(target_nodes)

        assert len(target_nodes) == 4
        assert portion_size == 256  # 1024 / 4 = 256 bytes per node

    def test_scatter_golden_captures_per_node_chunks(self, system_with_memory):
        """Test that SCATTER captures different golden data per node."""
        system, host_memory = system_with_memory

        # Create distinct test data for each portion
        total_size = 256
        test_data = bytes(range(total_size))
        host_memory.write(0x0000, test_data)

        target_nodes = [0, 1, 2, 3]
        portion_size = total_size // len(target_nodes)  # 64 bytes each

        # Capture golden for each node's portion (simulating SCATTER)
        for i, node_id in enumerate(target_nodes):
            offset = i * portion_size
            portion_data = test_data[offset:offset + portion_size]
            system.capture_golden_from_write(
                node_id=node_id,
                local_addr=0x1000,
                data=portion_data,
                cycle=i * 10,
            )

        # Verify each node has different golden data
        assert system.golden_manager.entry_count == 4

        golden_0 = system.golden_manager.get_golden(0, 0x1000)
        golden_1 = system.golden_manager.get_golden(1, 0x1000)
        golden_2 = system.golden_manager.get_golden(2, 0x1000)
        golden_3 = system.golden_manager.get_golden(3, 0x1000)

        # Each node should have different data
        assert golden_0 == test_data[0:64]
        assert golden_1 == test_data[64:128]
        assert golden_2 == test_data[128:192]
        assert golden_3 == test_data[192:256]

        # All portions should be different
        assert golden_0 != golden_1
        assert golden_1 != golden_2
        assert golden_2 != golden_3

    def test_scatter_single_node_gets_all_data(self, system_with_memory):
        """Test SCATTER with single node receives all data."""
        system, host_memory = system_with_memory

        test_data = b"ALL_DATA_FOR_SINGLE_NODE"
        host_memory.write(0x0000, test_data)

        config = TransferConfig(
            src_addr=0,
            src_size=len(test_data),
            dst_addr=0x1000,
            target_nodes=[5],  # Single node
            transfer_mode=TransferMode.SCATTER,
        )

        target_nodes = config.get_target_node_list()
        assert len(target_nodes) == 1

        # Single node gets all data
        portion_size = config.src_size // len(target_nodes)
        assert portion_size == len(test_data)

        # Capture golden
        system.capture_golden_from_write(
            node_id=5,
            local_addr=0x1000,
            data=test_data,
            cycle=0,
        )

        assert system.golden_manager.get_golden(5, 0x1000) == test_data


class TestGatherRead:
    """Integration tests for GATHER read mode."""

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

    def test_gather_config_is_read_mode(self):
        """Test GATHER is classified as read mode."""
        config = TransferConfig(
            src_addr=0,
            src_size=1024,
            target_nodes=[0, 1, 2, 3],
            read_src_addr=0x1000,
            read_size=1024,
            transfer_mode=TransferMode.GATHER,
        )
        assert config.is_read
        assert not config.is_write

    def test_gather_portion_calculation(self):
        """Test GATHER portion size calculation."""
        config = TransferConfig(
            target_nodes=[0, 1, 2, 3],
            read_src_addr=0x1000,
            read_size=1024,
            transfer_mode=TransferMode.GATHER,
        )
        target_nodes = config.get_target_node_list()
        portion_size = config.effective_read_size // len(target_nodes)

        assert len(target_nodes) == 4
        assert portion_size == 256  # 1024 / 4 = 256 bytes from each node

    def test_gather_golden_captures_concatenated_result(self, system_with_memory):
        """Test GATHER golden captures concatenated data for HostMemory."""
        system, _ = system_with_memory

        target_nodes = [0, 1, 2, 3]
        host_dst_addr = 0x2000

        # Create different data portions for each node
        data_portions = []
        for i in range(len(target_nodes)):
            portion_data = bytes([i * 16 + j for j in range(64)])
            data_portions.append(portion_data)

        # Capture golden using GATHER method (concatenated to HostMemory)
        system.golden_manager.capture_gather(
            host_addr=host_dst_addr,
            data_portions=data_portions,
            cycle=0,
        )

        # Verify single concatenated golden entry for HostMemory
        assert system.golden_manager.entry_count == 1

        # Get golden using ("host", addr) key
        golden = system.golden_manager.get_host_golden(host_dst_addr)
        assert golden is not None
        assert len(golden) == 64 * 4  # 256 bytes total

        # Verify concatenated content
        expected = b"".join(data_portions)
        assert golden == expected

    def test_gather_single_node(self, system_with_memory):
        """Test GATHER from single node."""
        system, _ = system_with_memory

        config = TransferConfig(
            target_nodes=[7],  # Single node
            read_src_addr=0x3000,
            read_size=128,
            transfer_mode=TransferMode.GATHER,
        )

        target_nodes = config.get_target_node_list()
        assert len(target_nodes) == 1

        # Single node provides all data
        portion_size = config.effective_read_size // len(target_nodes)
        assert portion_size == 128

    def test_gather_config_validation(self, system_with_memory):
        """Test GATHER config is accepted for read transfer."""
        system, _ = system_with_memory

        config = TransferConfig(
            target_nodes=[0, 1],
            read_src_addr=0x1000,
            read_size=64,
            transfer_mode=TransferMode.GATHER,
        )

        # Should not raise - GATHER is a valid read mode
        system.configure_read_transfer(config, use_golden=False)


class TestScatterGatherFlow:
    """Integration tests for SCATTER → GATHER round-trip flow."""

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

    def test_scatter_then_gather_config_sequence(self, system_with_memory):
        """Test SCATTER write followed by GATHER read config sequence."""
        system, host_memory = system_with_memory

        target_nodes = [0, 1, 2, 3]
        total_size = 256
        host_dst_addr = 0x2000

        # Phase 1: SCATTER write config
        scatter_config = TransferConfig(
            src_addr=0x0000,
            src_size=total_size,
            dst_addr=0x1000,
            target_nodes=target_nodes,
            transfer_mode=TransferMode.SCATTER,
        )

        assert scatter_config.is_write
        assert scatter_config.get_target_node_list() == target_nodes

        # Phase 2: Capture SCATTER golden (per-node for LocalMemory verification)
        test_data = bytes(range(total_size))
        host_memory.write(0x0000, test_data)

        portion_size = total_size // len(target_nodes)
        data_portions = []
        for i, node_id in enumerate(target_nodes):
            offset = i * portion_size
            portion_data = test_data[offset:offset + portion_size]
            data_portions.append(portion_data)
            # SCATTER: per-node golden for LocalMemory verification
            system.capture_golden_from_write(node_id, 0x1000, portion_data, i)

        # Phase 3: Capture GATHER golden (concatenated for HostMemory verification)
        system.golden_manager.capture_gather(
            host_addr=host_dst_addr,
            data_portions=data_portions,
            cycle=0,
        )

        # Phase 4: Reset for read
        system.reset_for_read()

        # SCATTER golden (4 entries) + GATHER golden (1 entry) = 5 entries
        assert system.golden_manager.entry_count == 5

        # Verify both golden types exist
        for node_id in target_nodes:
            assert system.golden_manager.get_golden(node_id, 0x1000) is not None
        assert system.golden_manager.get_host_golden(host_dst_addr) is not None

        # Phase 5: GATHER read config
        gather_config = TransferConfig(
            target_nodes=target_nodes,
            read_src_addr=0x1000,
            read_size=total_size,
            transfer_mode=TransferMode.GATHER,
        )

        assert gather_config.is_read
        system.configure_read_transfer(gather_config, use_golden=True)

    def test_scatter_gather_golden_integrity(self, system_with_memory):
        """Test golden data integrity through SCATTER → GATHER flow."""
        system, host_memory = system_with_memory

        target_nodes = [2, 5, 8, 11]  # Non-consecutive nodes
        total_size = 128
        test_data = bytes([i ^ 0xAA for i in range(total_size)])

        host_memory.write(0x0000, test_data)

        # Capture golden for each node (SCATTER)
        portion_size = total_size // len(target_nodes)
        for i, node_id in enumerate(target_nodes):
            offset = i * portion_size
            portion_data = test_data[offset:offset + portion_size]
            system.capture_golden_from_write(node_id, 0x2000, portion_data, i)

        # Verify golden integrity
        for i, node_id in enumerate(target_nodes):
            offset = i * portion_size
            expected = test_data[offset:offset + portion_size]
            actual = system.golden_manager.get_golden(node_id, 0x2000)
            assert actual == expected, f"Node {node_id} golden mismatch"

    def test_scatter_verification_report(self, system_with_memory):
        """Test verification report for SCATTER (per-node golden)."""
        system, host_memory = system_with_memory

        target_nodes = [0, 1]
        test_data = b"ABCD1234"

        # Capture golden for each node (SCATTER uses per-node golden)
        portion_size = len(test_data) // len(target_nodes)
        for i, node_id in enumerate(target_nodes):
            offset = i * portion_size
            portion_data = test_data[offset:offset + portion_size]
            system.capture_golden_from_write(node_id, 0x1000, portion_data, i)

        # Get verification report (no read data yet)
        report = system.verify_read_results()

        # Should report missing actual data for both nodes
        assert report.total_checks == 2
        assert report.missing_actual == 2

    def test_gather_verification_with_host_golden(self, system_with_memory):
        """Test GATHER verification using concatenated HostMemory golden."""
        system, host_memory = system_with_memory

        target_nodes = [0, 1, 2, 3]
        host_dst_addr = 0x2000
        total_size = 256
        portion_size = total_size // len(target_nodes)

        # Create source data portions (what each node would have)
        data_portions = []
        for i in range(len(target_nodes)):
            portion = bytes([i * 16 + j for j in range(portion_size)])
            data_portions.append(portion)

        # Capture GATHER golden (concatenated result expected in HostMemory)
        system.golden_manager.capture_gather(
            host_addr=host_dst_addr,
            data_portions=data_portions,
            cycle=0,
        )

        # Verify golden stored with ("host", addr) key
        assert system.golden_manager.entry_count == 1
        golden = system.golden_manager.get_host_golden(host_dst_addr)
        assert golden == b"".join(data_portions)

        # Simulate HostMemory receiving gathered data
        gathered_data = b"".join(data_portions)
        read_results: dict[GoldenKey, bytes] = {
            ("host", host_dst_addr): gathered_data
        }

        # Verify against golden
        report = system.golden_manager.verify(read_results)
        assert report.all_passed
        assert report.passed == 1


class TestScatterGatherEdgeCases:
    """Edge case tests for SCATTER and GATHER modes."""

    def test_uneven_scatter_division(self):
        """Test SCATTER with data size not evenly divisible by node count."""
        config = TransferConfig(
            src_addr=0,
            src_size=100,  # Not divisible by 3
            dst_addr=0x1000,
            target_nodes=[0, 1, 2],
            transfer_mode=TransferMode.SCATTER,
        )

        target_nodes = config.get_target_node_list()
        portion_size = config.src_size // len(target_nodes)

        # Integer division: 100 // 3 = 33 bytes per node
        # Total: 33 * 3 = 99 bytes (1 byte not distributed)
        assert portion_size == 33
        assert portion_size * len(target_nodes) == 99

    def test_minimum_scatter_size(self):
        """Test SCATTER with minimum viable size."""
        config = TransferConfig(
            src_addr=0,
            src_size=4,  # 1 byte per node
            dst_addr=0x1000,
            target_nodes=[0, 1, 2, 3],
            transfer_mode=TransferMode.SCATTER,
        )

        target_nodes = config.get_target_node_list()
        portion_size = config.src_size // len(target_nodes)

        assert portion_size == 1  # Minimum 1 byte per node

    def test_gather_from_non_consecutive_nodes(self):
        """Test GATHER from non-consecutive node IDs."""
        config = TransferConfig(
            target_nodes=[1, 4, 7, 15],  # Sparse node selection
            read_src_addr=0x1000,
            read_size=256,
            transfer_mode=TransferMode.GATHER,
        )

        target_nodes = config.get_target_node_list()
        assert target_nodes == [1, 4, 7, 15]
        assert len(target_nodes) == 4

    def test_scatter_to_all_nodes(self):
        """Test SCATTER to all 16 nodes."""
        config = TransferConfig(
            src_addr=0,
            src_size=1024,
            dst_addr=0x1000,
            target_nodes="all",
            transfer_mode=TransferMode.SCATTER,
        )

        target_nodes = config.get_target_node_list(total_nodes=16)
        portion_size = config.src_size // len(target_nodes)

        assert len(target_nodes) == 16
        assert portion_size == 64  # 1024 / 16 = 64 bytes per node
