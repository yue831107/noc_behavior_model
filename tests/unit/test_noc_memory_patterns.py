"""
Unit tests for NoC-to-NoC memory initialization patterns.
"""

import pytest

from src.core.routing_selector import NoCSystem
from src.config import NoCTrafficConfig, TrafficPattern


class TestNoCMemoryPatterns:
    """Tests for NoCSystem.initialize_node_memory() patterns."""

    @pytest.fixture
    def system(self):
        """Create a NoCSystem with traffic configured."""
        system = NoCSystem(
            mesh_cols=5,
            mesh_rows=4,
            buffer_depth=4,
            memory_size=0x10000,
        )
        config = NoCTrafficConfig(
            pattern=TrafficPattern.NEIGHBOR,
            mesh_cols=5,
            mesh_rows=4,
            transfer_size=256,
            local_src_addr=0x0000,
        )
        system.configure_traffic(config)
        return system

    def test_pattern_sequential(self, system):
        """Test sequential pattern with node_id offset."""
        system.initialize_node_memory(pattern="sequential")

        # Node 0: 0x00, 0x01, 0x02, ...
        data_0 = system.node_controllers[0].read_local_memory(0x0000, 16)
        assert data_0[0] == 0x00
        assert data_0[1] == 0x01
        assert data_0[15] == 0x0F

        # Node 1: 0x10, 0x11, 0x12, ...
        data_1 = system.node_controllers[1].read_local_memory(0x0000, 16)
        assert data_1[0] == 0x10
        assert data_1[1] == 0x11

        # Node 15: 0xF0, 0xF1, ... (wraps)
        data_15 = system.node_controllers[15].read_local_memory(0x0000, 16)
        assert data_15[0] == 0xF0
        assert data_15[15] == 0xFF

    def test_pattern_random(self, system):
        """Test random pattern with deterministic seed."""
        system.initialize_node_memory(pattern="random", seed=42)

        # Read data from two nodes
        data_0 = system.node_controllers[0].read_local_memory(0x0000, 16)
        data_1 = system.node_controllers[1].read_local_memory(0x0000, 16)

        # Data should be different between nodes
        assert data_0 != data_1

        # Re-initialize with same seed should produce same data
        system.initialize_node_memory(pattern="random", seed=42)
        data_0_again = system.node_controllers[0].read_local_memory(0x0000, 16)
        assert data_0 == data_0_again

    def test_pattern_constant(self, system):
        """Test constant pattern with node_id prefix."""
        system.initialize_node_memory(pattern="constant", value=0xAB)

        # Node 0: [0x00, 0xAB, 0xAB, ...]
        data_0 = system.node_controllers[0].read_local_memory(0x0000, 16)
        assert data_0[0] == 0x00  # node_id
        assert data_0[1] == 0xAB
        assert data_0[15] == 0xAB

        # Node 5: [0x05, 0xAB, 0xAB, ...]
        data_5 = system.node_controllers[5].read_local_memory(0x0000, 16)
        assert data_5[0] == 0x05  # node_id
        assert data_5[1] == 0xAB

    def test_pattern_node_id(self, system):
        """Test node_id pattern (legacy constant mode)."""
        system.initialize_node_memory(pattern="node_id")

        # Node 3: [0x03, 0x03, 0x03, ...]
        data_3 = system.node_controllers[3].read_local_memory(0x0000, 16)
        assert data_3[0] == 0x03
        assert data_3[1] == 0x03
        assert all(b == 0x03 for b in data_3)

    def test_pattern_address(self, system):
        """Test address pattern with node_id offset."""
        system.initialize_node_memory(pattern="address")

        # Node 0: base_addr = 0x0000 + (0 << 16) = 0x00000000
        # First 4 bytes should be 0x00000000 in little-endian
        data_0 = system.node_controllers[0].read_local_memory(0x0000, 8)
        assert data_0[0:4] == bytes([0x00, 0x00, 0x00, 0x00])
        # Next 4 bytes: 0x00000004
        assert data_0[4:8] == bytes([0x04, 0x00, 0x00, 0x00])

        # Node 1: base_addr = 0x0000 + (1 << 16) = 0x00010000
        data_1 = system.node_controllers[1].read_local_memory(0x0000, 8)
        assert data_1[0:4] == bytes([0x00, 0x00, 0x01, 0x00])  # 0x00010000 LE

    def test_pattern_walking_ones(self, system):
        """Test walking ones pattern with node_id prefix."""
        system.initialize_node_memory(pattern="walking_ones")

        # Node 0: [node_id=0x00, 0x01, 0x02, 0x04, 0x08, 0x10, 0x20, 0x40, ...]
        data_0 = system.node_controllers[0].read_local_memory(0x0000, 9)
        assert data_0[0] == 0x00  # node_id
        assert data_0[1] == 0x01
        assert data_0[2] == 0x02
        assert data_0[3] == 0x04
        assert data_0[8] == 0x80

        # Node 1: [node_id=0x01, 0x02, 0x04, 0x08, ...]  (rotated)
        data_1 = system.node_controllers[1].read_local_memory(0x0000, 9)
        assert data_1[0] == 0x01  # node_id
        assert data_1[1] == 0x02
        assert data_1[2] == 0x04
        assert data_1[7] == 0x80
        assert data_1[8] == 0x01  # wraps

    def test_pattern_walking_zeros(self, system):
        """Test walking zeros pattern with node_id prefix."""
        system.initialize_node_memory(pattern="walking_zeros")

        # Node 0: [node_id=0x00, 0xFE, 0xFD, 0xFB, 0xF7, 0xEF, 0xDF, 0xBF, 0x7F, ...]
        data_0 = system.node_controllers[0].read_local_memory(0x0000, 9)
        assert data_0[0] == 0x00  # node_id
        assert data_0[1] == 0xFE  # ~0x01
        assert data_0[2] == 0xFD  # ~0x02
        assert data_0[3] == 0xFB  # ~0x04
        assert data_0[8] == 0x7F  # ~0x80

        # Node 1: [node_id=0x01, 0xFD, 0xFB, ...] (rotated)
        data_1 = system.node_controllers[1].read_local_memory(0x0000, 9)
        assert data_1[0] == 0x01  # node_id
        assert data_1[1] == 0xFD
        assert data_1[7] == 0x7F
        assert data_1[8] == 0xFE  # wraps

    def test_pattern_checkerboard(self, system):
        """Test checkerboard pattern with node_id prefix."""
        system.initialize_node_memory(pattern="checkerboard")

        # Even node (0): [node_id=0x00, 0xAA, 0x55, 0xAA, 0x55, ...]
        data_0 = system.node_controllers[0].read_local_memory(0x0000, 5)
        assert data_0[0] == 0x00  # node_id
        assert data_0[1] == 0xAA
        assert data_0[2] == 0x55
        assert data_0[3] == 0xAA
        assert data_0[4] == 0x55

        # Odd node (1): [node_id=0x01, 0x55, 0xAA, 0x55, 0xAA, ...] (inverted)
        data_1 = system.node_controllers[1].read_local_memory(0x0000, 5)
        assert data_1[0] == 0x01  # node_id
        assert data_1[1] == 0x55
        assert data_1[2] == 0xAA
        assert data_1[3] == 0x55
        assert data_1[4] == 0xAA

    def test_pattern_unknown_defaults_to_zeros(self, system):
        """Test unknown pattern defaults to zeros with node_id prefix."""
        system.initialize_node_memory(pattern="unknown_pattern")

        # Node 5: [0x05, 0x00, 0x00, ...]
        data_5 = system.node_controllers[5].read_local_memory(0x0000, 16)
        assert data_5[0] == 0x05  # node_id
        assert all(b == 0x00 for b in data_5[1:])

    def test_all_patterns_produce_unique_data(self, system):
        """Test that all patterns produce unique data per node."""
        patterns = [
            "sequential",
            "random",
            "constant",
            "address",
            "walking_ones",
            "walking_zeros",
            "checkerboard",
        ]

        for pattern in patterns:
            system.initialize_node_memory(pattern=pattern, seed=42, value=0xCC)

            # Collect data from all nodes
            all_data = []
            for node_id in range(16):
                data = system.node_controllers[node_id].read_local_memory(0x0000, 256)
                all_data.append(data)

            # Verify all nodes have different data
            unique_data = set(all_data)
            assert len(unique_data) == 16, f"Pattern '{pattern}' should produce unique data per node"


class TestNoCMemoryPatternsWithGolden:
    """Integration tests for patterns with golden verification."""

    def test_pattern_sequential_with_golden(self):
        """Test sequential pattern generates correct golden."""
        system = NoCSystem(mesh_cols=5, mesh_rows=4)
        config = NoCTrafficConfig(
            pattern=TrafficPattern.NEIGHBOR,
            transfer_size=64,
        )
        system.configure_traffic(config)
        system.initialize_node_memory(pattern="sequential")

        golden_count = system.generate_golden()
        assert golden_count == 16

        # Golden for node 1 should be node 0's data (neighbor pattern)
        golden_data = system.golden_manager.get_golden(1, 0x1000)
        assert golden_data is not None
        assert golden_data[0] == 0x00  # Node 0's data starts with 0x00

    def test_pattern_random_with_golden(self):
        """Test random pattern generates correct golden."""
        system = NoCSystem(mesh_cols=5, mesh_rows=4)
        config = NoCTrafficConfig(
            pattern=TrafficPattern.NEIGHBOR,
            transfer_size=64,
        )
        system.configure_traffic(config)
        system.initialize_node_memory(pattern="random", seed=123)

        golden_count = system.generate_golden()
        assert golden_count == 16

        # Golden should be deterministic
        golden_data = system.golden_manager.get_golden(1, 0x1000)

        # Re-create and verify same golden
        system2 = NoCSystem(mesh_cols=5, mesh_rows=4)
        system2.configure_traffic(config)
        system2.initialize_node_memory(pattern="random", seed=123)
        system2.generate_golden()

        golden_data_2 = system2.golden_manager.get_golden(1, 0x1000)
        assert golden_data == golden_data_2
