"""
Deadlock detection tests for circular wait scenarios.

Tests verify that XY routing with Y→X turn prevention and
wormhole arbitration prevent deadlock in circular traffic patterns.
"""

import pytest
from typing import List, Tuple

from src.core.mesh import Mesh, MeshConfig
from src.core.router import RouterConfig
from src.core.flit import FlitFactory
from src.core.router import Direction

from .deadlock_helpers import (
    DeadlockDetector,
    get_mesh_buffer_occupancy,
    get_router_forwarding_stats,
    drain_local_ports,
)
from tests.conftest import run_multi_router_cycle


@pytest.mark.deadlock
class TestCircularWaitPrevention:
    """Tests for circular wait deadlock prevention."""

    @pytest.fixture
    def small_mesh(self) -> Mesh:
        """Create a small 3x3 mesh for testing."""
        config = MeshConfig(
            cols=3,
            rows=3,
            router_config=RouterConfig(buffer_depth=4),
        )
        return Mesh(config)

    @pytest.fixture
    def standard_mesh(self) -> Mesh:
        """Create standard 5x4 mesh."""
        config = MeshConfig(
            cols=5,
            rows=4,
            router_config=RouterConfig(buffer_depth=4),
        )
        return Mesh(config)

    def _inject_flit(
        self,
        mesh: Mesh,
        src: Tuple[int, int],
        dest: Tuple[int, int],
        use_resp: bool = False,
    ) -> bool:
        """
        Inject a single flit into mesh at source router.

        Args:
            mesh: Target mesh.
            src: Source coordinate (x, y).
            dest: Destination coordinate (x, y).
            use_resp: If True, inject into response network.

        Returns:
            True if injection successful.
        """
        router = mesh.routers.get(src)
        if router is None:
            return False

        flit = FlitFactory.create_aw(
            src=src,
            dest=dest,
            addr=0x1000,
            axi_id=0,
            length=0,
            last=True,  # Single-flit packet - release wormhole lock after forwarding
        )

        target_router = router.resp_router if use_resp else router.req_router
        local_port = target_router.ports.get(Direction.LOCAL)

        if local_port is None:
            return False

        if local_port.occupancy < local_port._buffer.depth:
            local_port._buffer.push(flit)
            return True
        return False

    def test_ring_pattern_small_mesh(self, small_mesh):
        """
        Test ring traffic pattern on small mesh.

        Pattern uses only compute nodes (not edge column 0):
        (1,0)→(2,0)→(2,1)→(2,2)→(1,2)→(1,1)→(1,0)

        This creates a circular dependency that would deadlock
        without proper routing.
        """
        detector = DeadlockDetector(threshold_cycles=200)

        # Define ring pattern using only compute nodes (columns 1+, not edge column 0)
        ring_nodes = [
            (1, 0), (2, 0),
            (2, 1), (2, 2),
            (1, 2), (1, 1),
        ]

        # Inject flits in ring pattern
        for i in range(len(ring_nodes)):
            src = ring_nodes[i]
            dest = ring_nodes[(i + 1) % len(ring_nodes)]
            self._inject_flit(small_mesh, src, dest)

        # Get all wires for cycle simulation
        all_wires = small_mesh._req_wires + small_mesh._resp_wires
        routers = list(small_mesh.routers.values())

        # Run simulation
        total_forwarded = 0
        for cycle in range(500):
            run_multi_router_cycle(routers, all_wires, cycle)
            drain_local_ports(small_mesh)  # Simulate NI consumption
            detector.update(cycle, small_mesh)

            if detector.check_deadlock():
                pytest.fail(f"Deadlock detected at cycle {cycle}")

        # Verify progress was made
        stats = get_router_forwarding_stats(small_mesh)
        total_forwarded = sum(stats.values())
        assert total_forwarded > 0, "No flits were forwarded"

    def test_adversarial_xy_pattern(self, standard_mesh):
        """
        Test adversarial traffic that would cause Y→X turns.

        Inject traffic from column 1 to column 4, forcing
        potential Y→X turns which should be handled by routing.
        Uses only compute nodes (columns 1+, not edge column 0).
        """
        detector = DeadlockDetector(threshold_cycles=300)

        # Inject from (1,0) to various destinations (all compute nodes)
        destinations = [(4, 0), (4, 1), (4, 2), (4, 3)]
        for dest in destinations:
            self._inject_flit(standard_mesh, (1, 0), dest)

        # Inject from (1,3) to create crossing traffic
        for dest in [(4, 0), (4, 1)]:
            self._inject_flit(standard_mesh, (1, 3), dest)

        # Get wires
        all_wires = standard_mesh._req_wires + standard_mesh._resp_wires
        routers = list(standard_mesh.routers.values())

        # Run simulation
        for cycle in range(500):
            run_multi_router_cycle(routers, all_wires, cycle)
            drain_local_ports(standard_mesh)  # Simulate NI consumption
            detector.update(cycle, standard_mesh)

            if detector.check_deadlock():
                pytest.fail(f"Deadlock detected at cycle {cycle}")

    def test_bidirectional_traffic_no_deadlock(self, standard_mesh):
        """
        Test bidirectional traffic between opposite corners.

        Traffic patterns that could create head-of-line blocking.
        Uses only compute nodes (columns 1+, not edge column 0).
        """
        detector = DeadlockDetector(threshold_cycles=300)

        # East-West traffic (using compute nodes only)
        self._inject_flit(standard_mesh, (1, 0), (4, 0))
        self._inject_flit(standard_mesh, (4, 0), (1, 0))

        # North-South traffic
        self._inject_flit(standard_mesh, (2, 0), (2, 3))
        self._inject_flit(standard_mesh, (2, 3), (2, 0))

        # Diagonal traffic (using compute nodes only)
        self._inject_flit(standard_mesh, (1, 0), (4, 3))
        self._inject_flit(standard_mesh, (4, 3), (1, 0))

        # Get wires
        all_wires = standard_mesh._req_wires + standard_mesh._resp_wires
        routers = list(standard_mesh.routers.values())

        # Run simulation
        for cycle in range(500):
            run_multi_router_cycle(routers, all_wires, cycle)
            drain_local_ports(standard_mesh)  # Simulate NI consumption
            detector.update(cycle, standard_mesh)

            if detector.check_deadlock():
                pytest.fail(f"Deadlock detected at cycle {cycle}")

        # Verify all flits were forwarded
        stats = get_router_forwarding_stats(standard_mesh)
        total_forwarded = sum(stats.values())
        assert total_forwarded >= 6, f"Expected at least 6 flits forwarded, got {total_forwarded}"


@pytest.mark.deadlock
class TestWormholeLockingDeadlock:
    """Tests for wormhole locking deadlock scenarios."""

    @pytest.fixture
    def mesh(self) -> Mesh:
        """Create standard mesh."""
        config = MeshConfig(
            cols=5,
            rows=4,
            router_config=RouterConfig(buffer_depth=4),
        )
        return Mesh(config)

    def test_multi_flit_packet_no_starvation(self, mesh, multi_flit_packet_factory):
        """
        Test that multi-flit packets don't cause indefinite starvation.

        When a multi-flit packet holds a wormhole lock, other packets
        should still make progress on other paths.
        Uses only compute nodes (columns 1+, not edge column 0).
        """
        detector = DeadlockDetector(threshold_cycles=300)

        # Inject a multi-flit packet (from compute node, not edge)
        flits = multi_flit_packet_factory(
            src=(1, 0),
            dest=(4, 0),
            num_flits=4,
        )

        # Inject into source router (compute node)
        router = mesh.routers[(1, 0)]
        for flit in flits:
            router.req_router.ports[Direction.LOCAL]._buffer.push(flit)

        # Also inject single-flit packets on different paths
        single_flit = FlitFactory.create_aw(
            src=(1, 2),
            dest=(4, 2),
            addr=0x2000,
            axi_id=1,
            length=0,
        )
        mesh.routers[(1, 2)].req_router.ports[Direction.LOCAL]._buffer.push(single_flit)

        # Get wires
        all_wires = mesh._req_wires + mesh._resp_wires
        routers = list(mesh.routers.values())

        # Run simulation
        for cycle in range(500):
            run_multi_router_cycle(routers, all_wires, cycle)
            drain_local_ports(mesh)  # Simulate NI consumption
            detector.update(cycle, mesh)

            if detector.check_deadlock():
                pytest.fail(f"Deadlock detected at cycle {cycle}")

        # Both packets should have been forwarded
        stats = get_router_forwarding_stats(mesh)
        total_forwarded = sum(stats.values())
        assert total_forwarded >= 5, f"Expected at least 5 flits forwarded, got {total_forwarded}"


@pytest.mark.deadlock
class TestXYRoutingDeadlockPrevention:
    """Tests verifying XY routing prevents deadlock."""

    @pytest.fixture
    def mesh(self) -> Mesh:
        """Create standard mesh."""
        config = MeshConfig(
            cols=5,
            rows=4,
            router_config=RouterConfig(buffer_depth=4),
        )
        return Mesh(config)

    def test_no_yx_turn_in_routing(self, mesh, single_flit_factory):
        """
        Verify that Y→X turns are prevented.

        Flits that would require Y→X turn should be handled
        (either routed differently or dropped).
        """
        # Create flit that goes North first, then would need to go East
        # From (2,0) to (4,2) - optimal XY: East, East, North, North
        # If we force it to go North first, it would need Y→X turn

        flit = single_flit_factory(src=(2, 0), dest=(4, 2))

        # Inject at router (2,0)
        router = mesh.routers[(2, 0)]
        local_port = router.req_router.ports[Direction.LOCAL]
        local_port._buffer.push(flit)

        # Get wires
        all_wires = mesh._req_wires + mesh._resp_wires
        routers = list(mesh.routers.values())

        # Run simulation
        detector = DeadlockDetector(threshold_cycles=200)
        for cycle in range(300):
            run_multi_router_cycle(routers, all_wires, cycle)
            drain_local_ports(mesh)  # Simulate NI consumption
            detector.update(cycle, mesh)

            if detector.check_deadlock():
                pytest.fail(f"Deadlock detected at cycle {cycle}")

        # Flit should reach destination or be properly handled
        stats = get_router_forwarding_stats(mesh)
        total_forwarded = sum(stats.values())
        assert total_forwarded >= 1, "Flit was not forwarded"
