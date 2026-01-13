"""
Edge case tests for transfer modes.

Golden data verification follows these principles:
- SCATTER/BROADCAST: Golden captured per-node, key = (node_id, dst_addr)
- GATHER: Golden captured as concatenated result, key = ("host", dst_addr)
"""

import pytest

from src.config import TransferConfig, TransferMode
from src.core.routing_selector import V1System
from src.testbench import Memory, MemoryConfig
from src.verification import GoldenKey


class TestSingleNodeTransfers:
    """Tests for single-node transfer scenarios."""

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

    def test_single_node_scatter(self, system_with_memory):
        """Test SCATTER with single node - gets all data."""
        system, host_memory = system_with_memory

        config = TransferConfig(
            src_addr=0,
            src_size=128,
            dst_addr=0x1000,
            target_nodes=[5],
            transfer_mode=TransferMode.SCATTER,
        )

        target_nodes = config.get_target_node_list()
        assert len(target_nodes) == 1

        # Single node gets entire src_size
        portion_size = config.src_size // len(target_nodes)
        assert portion_size == 128

    def test_single_node_gather(self, system_with_memory):
        """Test GATHER with single node - provides all data to HostMemory."""
        system, _ = system_with_memory

        config = TransferConfig(
            target_nodes=[7],
            read_src_addr=0x2000,
            read_size=256,
            transfer_mode=TransferMode.GATHER,
        )

        target_nodes = config.get_target_node_list()
        assert len(target_nodes) == 1

        # Single node provides entire read_size
        portion_size = config.effective_read_size // len(target_nodes)
        assert portion_size == 256

        # GATHER golden: single node's data goes to HostMemory
        host_dst_addr = 0x3000
        single_portion = bytes(range(256))
        system.golden_manager.capture_gather(
            host_addr=host_dst_addr,
            data_portions=[single_portion],
            cycle=0,
        )

        # Verify golden stored with ("host", addr) key
        golden = system.golden_manager.get_host_golden(host_dst_addr)
        assert golden == single_portion

    def test_single_node_broadcast_write(self, system_with_memory):
        """Test BROADCAST write to single node."""
        system, host_memory = system_with_memory

        test_data = b"SINGLE_NODE_DATA"
        host_memory.write(0x0000, test_data)

        config = TransferConfig(
            src_addr=0,
            src_size=len(test_data),
            dst_addr=0x1000,
            target_nodes=[0],
            transfer_mode=TransferMode.BROADCAST,
        )

        target_nodes = config.get_target_node_list()
        assert len(target_nodes) == 1
        assert target_nodes == [0]

    def test_single_node_broadcast_read(self, system_with_memory):
        """Test BROADCAST_READ from single node."""
        system, _ = system_with_memory

        config = TransferConfig(
            target_nodes=[15],  # Last node
            read_src_addr=0x3000,
            read_size=64,
            transfer_mode=TransferMode.BROADCAST_READ,
        )

        target_nodes = config.get_target_node_list()
        assert len(target_nodes) == 1
        assert target_nodes == [15]


class TestUnevenDataDistribution:
    """Tests for uneven data distribution scenarios."""

    def test_scatter_uneven_division(self):
        """Test SCATTER with data size not evenly divisible."""
        config = TransferConfig(
            src_addr=0,
            src_size=100,  # Not divisible by 3
            dst_addr=0x1000,
            target_nodes=[0, 1, 2],
            transfer_mode=TransferMode.SCATTER,
        )

        target_nodes = config.get_target_node_list()
        portion_size = config.src_size // len(target_nodes)

        # Integer division: 100 // 3 = 33
        assert portion_size == 33
        # Total distributed: 33 * 3 = 99 (1 byte remainder)
        assert portion_size * len(target_nodes) == 99

    def test_gather_uneven_division(self):
        """Test GATHER with read size not evenly divisible."""
        config = TransferConfig(
            target_nodes=[0, 1, 2, 3, 4],  # 5 nodes
            read_src_addr=0x1000,
            read_size=103,  # Not divisible by 5
            transfer_mode=TransferMode.GATHER,
        )

        target_nodes = config.get_target_node_list()
        portion_size = config.effective_read_size // len(target_nodes)

        # Integer division: 103 // 5 = 20
        assert portion_size == 20
        # Total gathered: 20 * 5 = 100 (3 bytes not gathered)
        total_gathered = portion_size * len(target_nodes)
        assert total_gathered == 100

        # GATHER golden would be 100 bytes concatenated in HostMemory
        # (remaining 3 bytes are truncated due to integer division)

    def test_scatter_large_node_count(self):
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
        assert portion_size == 64  # 1024 / 16 = 64


class TestMinimumDataSizes:
    """Tests for minimum viable data sizes."""

    def test_minimum_scatter_1_byte_per_node(self):
        """Test SCATTER with minimum 1 byte per node."""
        config = TransferConfig(
            src_addr=0,
            src_size=4,
            dst_addr=0x1000,
            target_nodes=[0, 1, 2, 3],
            transfer_mode=TransferMode.SCATTER,
        )

        target_nodes = config.get_target_node_list()
        portion_size = config.src_size // len(target_nodes)

        assert portion_size == 1  # Minimum 1 byte per node

    def test_minimum_gather_1_byte_per_node(self):
        """Test GATHER with minimum 1 byte from each node."""
        config = TransferConfig(
            target_nodes=[0, 1],
            read_src_addr=0x1000,
            read_size=2,
            transfer_mode=TransferMode.GATHER,
        )

        target_nodes = config.get_target_node_list()
        portion_size = config.effective_read_size // len(target_nodes)

        assert portion_size == 1  # Minimum 1 byte from each

    def test_scatter_zero_portion_size(self):
        """Test SCATTER where portion size would be 0."""
        config = TransferConfig(
            src_addr=0,
            src_size=3,  # Less than node count
            dst_addr=0x1000,
            target_nodes=[0, 1, 2, 3, 4],  # 5 nodes
            transfer_mode=TransferMode.SCATTER,
        )

        target_nodes = config.get_target_node_list()
        portion_size = config.src_size // len(target_nodes)

        # 3 // 5 = 0 - this is an edge case
        assert portion_size == 0


class TestNonConsecutiveNodes:
    """Tests for non-consecutive node selections."""

    def test_scatter_sparse_nodes(self):
        """Test SCATTER to sparse (non-consecutive) nodes."""
        config = TransferConfig(
            src_addr=0,
            src_size=256,
            dst_addr=0x1000,
            target_nodes=[1, 4, 7, 15],
            transfer_mode=TransferMode.SCATTER,
        )

        target_nodes = config.get_target_node_list()
        assert target_nodes == [1, 4, 7, 15]
        assert len(target_nodes) == 4

        portion_size = config.src_size // len(target_nodes)
        assert portion_size == 64

    def test_gather_sparse_nodes(self):
        """Test GATHER from sparse (non-consecutive) nodes."""
        config = TransferConfig(
            target_nodes=[0, 5, 10, 15],
            read_src_addr=0x2000,
            read_size=128,
            transfer_mode=TransferMode.GATHER,
        )

        target_nodes = config.get_target_node_list()
        assert target_nodes == [0, 5, 10, 15]

    def test_range_format_nodes(self):
        """Test range format for target nodes."""
        config = TransferConfig(
            target_nodes="range:4-7",
            read_src_addr=0x1000,
            read_size=64,
            transfer_mode=TransferMode.GATHER,
        )

        target_nodes = config.get_target_node_list()
        assert target_nodes == [4, 5, 6, 7]


class TestGoldenDataEdgeCases:
    """Tests for golden data edge cases."""

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

    def test_golden_overwrite_same_address(self, system_with_memory):
        """Test golden data overwrite at same address."""
        system, _ = system_with_memory

        # First write
        system.capture_golden_from_write(0, 0x1000, b"FIRST", 0)
        assert system.golden_manager.get_golden(0, 0x1000) == b"FIRST"

        # Overwrite with new data
        system.capture_golden_from_write(0, 0x1000, b"SECOND", 100)
        assert system.golden_manager.get_golden(0, 0x1000) == b"SECOND"

    def test_golden_different_addresses_same_node(self, system_with_memory):
        """Test golden data at different addresses on same node."""
        system, _ = system_with_memory

        system.capture_golden_from_write(5, 0x1000, b"ADDR_1000", 0)
        system.capture_golden_from_write(5, 0x2000, b"ADDR_2000", 10)

        assert system.golden_manager.get_golden(5, 0x1000) == b"ADDR_1000"
        assert system.golden_manager.get_golden(5, 0x2000) == b"ADDR_2000"
        assert system.golden_manager.entry_count == 2

    def test_golden_same_address_different_nodes(self, system_with_memory):
        """Test golden data at same address on different nodes."""
        system, _ = system_with_memory

        system.capture_golden_from_write(0, 0x1000, b"NODE_0", 0)
        system.capture_golden_from_write(1, 0x1000, b"NODE_1", 10)
        system.capture_golden_from_write(2, 0x1000, b"NODE_2", 20)

        assert system.golden_manager.get_golden(0, 0x1000) == b"NODE_0"
        assert system.golden_manager.get_golden(1, 0x1000) == b"NODE_1"
        assert system.golden_manager.get_golden(2, 0x1000) == b"NODE_2"
        assert system.golden_manager.entry_count == 3

    def test_golden_preserved_after_reset_for_read(self, system_with_memory):
        """Test golden data is preserved after reset_for_read."""
        system, _ = system_with_memory

        # Add golden entries
        system.capture_golden_from_write(0, 0x1000, b"DATA_0", 0)
        system.capture_golden_from_write(1, 0x2000, b"DATA_1", 10)

        initial_count = system.golden_manager.entry_count

        # Reset for read
        system.reset_for_read()

        # Golden should be preserved
        assert system.golden_manager.entry_count == initial_count
        assert system.golden_manager.get_golden(0, 0x1000) == b"DATA_0"
        assert system.golden_manager.get_golden(1, 0x2000) == b"DATA_1"
