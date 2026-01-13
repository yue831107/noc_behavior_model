"""
Unit tests for NoC-to-NoC golden data generation and verification.
"""

import pytest

from src.verification import GoldenManager, GoldenSource
from src.testbench import Memory, MemoryConfig
from src.config import NodeTransferConfig, TrafficPattern
from src.traffic.pattern_generator import TrafficPatternGenerator


class TestNoCGoldenGeneration:
    """Tests for NoC-to-NoC golden generation."""

    @pytest.fixture
    def setup_memories(self):
        """Create test memories for 16 nodes."""
        memories = {}
        for i in range(16):
            mem = Memory(
                config=MemoryConfig(size=0x10000),
                name=f"Node{i}_Memory"
            )
            # Initialize with unique data per node
            data = bytes((i * 16 + j) & 0xFF for j in range(256))
            mem.write(0x0000, data)
            memories[i] = mem
        return memories

    @pytest.fixture
    def generator(self):
        """Create pattern generator for 5x4 mesh."""
        return TrafficPatternGenerator(mesh_cols=5, mesh_rows=4)

    def test_generate_noc_golden_neighbor(self, setup_memories, generator):
        """Test golden generation for neighbor pattern."""
        from src.config import NoCTrafficConfig

        manager = GoldenManager()

        # Generate neighbor pattern configs
        config = NoCTrafficConfig(
            pattern=TrafficPattern.NEIGHBOR,
            mesh_cols=5,
            mesh_rows=4,
            transfer_size=256,
            local_src_addr=0x0000,
            local_dst_addr=0x1000,
        )
        node_configs = generator.generate(config)

        # Generate golden
        count = manager.generate_noc_golden(
            node_configs=node_configs,
            get_node_memory=lambda x: setup_memories[x],
            mesh_cols=5,
        )

        assert count == 16

        # Verify golden stored correctly for neighbor pattern
        # Node 0 sends to Node 1, so golden[(1, 0x1000)] = Node0's data
        expected_data = setup_memories[0].read(0x0000, 256)[0]
        actual_golden = manager.get_golden(1, 0x1000)
        assert actual_golden == expected_data

        # Node 1 sends to Node 2
        expected_data = setup_memories[1].read(0x0000, 256)[0]
        actual_golden = manager.get_golden(2, 0x1000)
        assert actual_golden == expected_data

    def test_generate_noc_golden_shuffle(self, setup_memories, generator):
        """Test golden generation for shuffle pattern."""
        from src.config import NoCTrafficConfig

        manager = GoldenManager()

        config = NoCTrafficConfig(
            pattern=TrafficPattern.SHUFFLE,
            mesh_cols=5,
            mesh_rows=4,
            transfer_size=256,
        )
        node_configs = generator.generate(config)

        count = manager.generate_noc_golden(
            node_configs=node_configs,
            get_node_memory=lambda x: setup_memories[x],
            mesh_cols=5,
        )

        # Should have valid golden entries
        assert count > 0
        assert manager.entry_count > 0

    def test_golden_collision_last_write_wins(self, setup_memories):
        """Test that collision uses last-write-wins semantics."""
        manager = GoldenManager()

        # Create two configs that write to same destination
        configs = [
            NodeTransferConfig(
                src_node_id=0,
                dest_coord=(2, 0),  # Node 1
                local_src_addr=0x0000,
                local_dst_addr=0x1000,
                transfer_size=256,
            ),
            NodeTransferConfig(
                src_node_id=2,
                dest_coord=(2, 0),  # Also Node 1 (collision!)
                local_src_addr=0x0000,
                local_dst_addr=0x1000,
                transfer_size=256,
            ),
        ]

        count = manager.generate_noc_golden(
            node_configs=configs,
            get_node_memory=lambda x: setup_memories[x],
            mesh_cols=5,
        )

        # Should have 2 writes, but only 1 entry (collision)
        assert count == 2
        assert manager.entry_count == 1

        # Last write wins: Node 2's data should be the golden
        expected_data = setup_memories[2].read(0x0000, 256)[0]
        actual_golden = manager.get_golden(1, 0x1000)
        assert actual_golden == expected_data

    def test_verify_with_matching_data(self, setup_memories, generator):
        """Test verification when actual matches expected."""
        from src.config import NoCTrafficConfig

        manager = GoldenManager()

        config = NoCTrafficConfig(
            pattern=TrafficPattern.NEIGHBOR,
            mesh_cols=5,
            mesh_rows=4,
            transfer_size=256,
            local_src_addr=0x0000,
            local_dst_addr=0x1000,
        )
        node_configs = generator.generate(config)

        # Generate golden
        manager.generate_noc_golden(
            node_configs=node_configs,
            get_node_memory=lambda x: setup_memories[x],
            mesh_cols=5,
        )

        # Simulate perfect transfer: copy data to destinations
        for nc in node_configs:
            src_data = setup_memories[nc.src_node_id].read(0x0000, 256)[0]
            dest_x, dest_y = nc.dest_coord
            if dest_x >= 1:
                dst_node_id = (dest_x - 1) + dest_y * 4
                if dst_node_id < 16:
                    setup_memories[dst_node_id].write(0x1000, src_data)

        # Build read results
        read_results = {}
        for nc in node_configs:
            dest_x, dest_y = nc.dest_coord
            if dest_x >= 1:
                dst_node_id = (dest_x - 1) + dest_y * 4
                if dst_node_id < 16:
                    data = setup_memories[dst_node_id].read(0x1000, 256)[0]
                    read_results[(dst_node_id, 0x1000)] = data

        # Verify
        report = manager.verify(read_results)
        assert report.all_passed
        assert report.passed == 16
        assert report.failed == 0

    def test_verify_with_mismatch(self, setup_memories, generator):
        """Test verification when actual doesn't match expected."""
        from src.config import NoCTrafficConfig

        manager = GoldenManager()

        config = NoCTrafficConfig(
            pattern=TrafficPattern.NEIGHBOR,
            mesh_cols=5,
            mesh_rows=4,
            transfer_size=256,
        )
        node_configs = generator.generate(config)

        # Generate golden
        manager.generate_noc_golden(
            node_configs=node_configs,
            get_node_memory=lambda x: setup_memories[x],
            mesh_cols=5,
        )

        # Don't copy data - just read whatever is at dst_addr (will be different)
        read_results = {}
        for nc in node_configs:
            dest_x, dest_y = nc.dest_coord
            if dest_x >= 1:
                dst_node_id = (dest_x - 1) + dest_y * 4
                if dst_node_id < 16:
                    # Read from dst_addr (not transferred, so different data)
                    data = setup_memories[dst_node_id].read(0x1000, 256)[0]
                    read_results[(dst_node_id, 0x1000)] = data

        # Verify - should fail since data wasn't transferred
        report = manager.verify(read_results)
        assert not report.all_passed
        assert report.failed > 0

    def test_golden_entry_count(self, setup_memories, generator):
        """Test that entry count is correct after generation."""
        from src.config import NoCTrafficConfig

        manager = GoldenManager()

        config = NoCTrafficConfig(
            pattern=TrafficPattern.NEIGHBOR,
            mesh_cols=5,
            mesh_rows=4,
            transfer_size=256,
        )
        node_configs = generator.generate(config)

        assert manager.entry_count == 0

        count = manager.generate_noc_golden(
            node_configs=node_configs,
            get_node_memory=lambda x: setup_memories[x],
            mesh_cols=5,
        )

        assert manager.entry_count == count
        assert manager.entry_count == 16

    def test_golden_clear(self, setup_memories, generator):
        """Test clearing golden data."""
        from src.config import NoCTrafficConfig

        manager = GoldenManager()

        config = NoCTrafficConfig(
            pattern=TrafficPattern.NEIGHBOR,
            mesh_cols=5,
            mesh_rows=4,
        )
        node_configs = generator.generate(config)

        manager.generate_noc_golden(
            node_configs=node_configs,
            get_node_memory=lambda x: setup_memories[x],
            mesh_cols=5,
        )

        assert manager.entry_count > 0
        manager.clear()
        assert manager.entry_count == 0


class TestNoCGoldenEdgeCases:
    """Edge case tests for NoC-to-NoC golden."""

    def test_empty_node_configs(self):
        """Test with empty node configs list."""
        manager = GoldenManager()

        count = manager.generate_noc_golden(
            node_configs=[],
            get_node_memory=lambda x: None,
            mesh_cols=5,
        )

        assert count == 0
        assert manager.entry_count == 0

    def test_invalid_destination_skipped(self):
        """Test that destinations in edge column are skipped."""
        manager = GoldenManager()
        mem = Memory(config=MemoryConfig(size=0x10000))
        mem.write(0x0000, bytes(256))

        configs = [
            NodeTransferConfig(
                src_node_id=0,
                dest_coord=(0, 0),  # Edge column - invalid!
                local_src_addr=0x0000,
                local_dst_addr=0x1000,
                transfer_size=256,
            ),
        ]

        count = manager.generate_noc_golden(
            node_configs=configs,
            get_node_memory=lambda x: mem,
            mesh_cols=5,
        )

        assert count == 0  # Skipped invalid destination

    def test_all_patterns_generate_golden(self):
        """Test that all 5 patterns can generate golden."""
        patterns = [
            TrafficPattern.NEIGHBOR,
            TrafficPattern.SHUFFLE,
            TrafficPattern.BIT_REVERSE,
            TrafficPattern.RANDOM,
            TrafficPattern.TRANSPOSE,
        ]

        from src.config import NoCTrafficConfig

        generator = TrafficPatternGenerator(mesh_cols=5, mesh_rows=4)

        # Create memories
        memories = {}
        for i in range(16):
            mem = Memory(config=MemoryConfig(size=0x10000))
            mem.write(0x0000, bytes([i] * 256))
            memories[i] = mem

        for pattern in patterns:
            manager = GoldenManager()
            config = NoCTrafficConfig(
                pattern=pattern,
                mesh_cols=5,
                mesh_rows=4,
                transfer_size=256,
                random_seed=42,
            )
            node_configs = generator.generate(config)

            count = manager.generate_noc_golden(
                node_configs=node_configs,
                get_node_memory=lambda x: memories[x],
                mesh_cols=5,
            )

            assert count > 0, f"Pattern {pattern.value} should generate golden"
