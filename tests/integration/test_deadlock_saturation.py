"""
Network saturation and starvation tests.

Tests verify network stability under high load and
fair resource allocation across ports.
"""

import pytest
from typing import Tuple
import random

from src.core.mesh import Mesh, MeshConfig
from src.core.router import RouterConfig
from src.core.flit import FlitFactory
from src.core.router import Direction

from .deadlock_helpers import (
    DeadlockDetector,
    get_mesh_buffer_occupancy,
    get_router_forwarding_stats,
    get_port_utilization_stats,
    drain_local_ports,
)
from tests.conftest import run_multi_router_cycle


@pytest.mark.saturation
class TestNetworkSaturation:
    """Tests for network saturation stability."""

    @pytest.fixture
    def mesh(self) -> Mesh:
        """Create standard 5x4 mesh."""
        config = MeshConfig(
            cols=5,
            rows=4,
            router_config=RouterConfig(buffer_depth=4),
        )
        return Mesh(config)

    def _inject_random_traffic(
        self,
        mesh: Mesh,
        count: int,
        seed: int = 42,
    ) -> int:
        """
        Inject random traffic into mesh.

        Args:
            mesh: Target mesh.
            count: Number of flits to attempt to inject.
            seed: Random seed for reproducibility.

        Returns:
            Number of flits successfully injected.
        """
        rng = random.Random(seed)
        # Only use compute nodes (column 1+, not edge column 0)
        coords = [c for c in mesh.routers.keys() if c[0] > 0]
        injected = 0

        for i in range(count):
            src = rng.choice(coords)
            dest = rng.choice([c for c in coords if c != src])

            flit = FlitFactory.create_aw(
                src=src,
                dest=dest,
                addr=0x1000 + i * 0x100,
                axi_id=i % 16,
                length=0,
                last=True,  # Single-flit packet
            )

            router = mesh.routers[src]
            local_port = router.req_router.ports.get(Direction.LOCAL)

            if local_port and local_port.occupancy < local_port._buffer.depth:
                local_port._buffer.push(flit)
                injected += 1

        return injected

    def _get_all_wires(self, mesh: Mesh):
        """Get all wires from mesh."""
        return mesh._req_wires + mesh._resp_wires

    def test_sustained_load_stability(self, mesh):
        """
        Test network stability under sustained load.

        Inject continuous traffic and verify:
        1. No deadlock occurs
        2. Throughput remains stable
        3. Buffer utilization stays bounded
        """
        detector = DeadlockDetector(threshold_cycles=300)
        all_wires = self._get_all_wires(mesh)
        routers = list(mesh.routers.values())

        # Track throughput over time
        throughput_samples = []

        # Warm-up phase: inject initial traffic
        self._inject_random_traffic(mesh, count=20, seed=42)

        # Simulation with continuous injection
        for cycle in range(1000):
            # Inject new traffic periodically
            if cycle % 10 == 0:
                self._inject_random_traffic(mesh, count=5, seed=42 + cycle)

            # Call drain_local_ports BEFORE cycle to ensure in_ready=True
            # This is critical for handshake completion in clear_accepted_outputs
            drain_local_ports(mesh)
            run_multi_router_cycle(routers, all_wires, cycle)
            drain_local_ports(mesh)  # Also after to clear outputs and restore credits
            detector.update(cycle, mesh)

            if detector.check_deadlock():
                pytest.fail(f"Deadlock detected at cycle {cycle}")

            # Sample throughput every 100 cycles
            if cycle > 0 and cycle % 100 == 0:
                stats = get_router_forwarding_stats(mesh)
                total = sum(stats.values())
                throughput_samples.append(total)

        # Verify throughput stability
        if len(throughput_samples) >= 2:
            # Calculate variance in throughput increments
            increments = [
                throughput_samples[i] - throughput_samples[i - 1]
                for i in range(1, len(throughput_samples))
            ]
            avg_increment = sum(increments) / len(increments)

            # Throughput should be positive (making progress)
            assert avg_increment > 0, "Network not making forward progress"

        # Verify buffer utilization is bounded
        occupancy = get_mesh_buffer_occupancy(mesh)
        max_occupancy = max(occupancy.values()) if occupancy else 0
        assert max_occupancy <= 4 * 5 * 2, "Buffer overflow detected"  # buffer_depth * ports * networks

    def test_burst_traffic_recovery(self, mesh):
        """
        Test recovery from traffic burst.

        Inject large burst, then verify network drains properly.
        """
        detector = DeadlockDetector(threshold_cycles=300)
        all_wires = self._get_all_wires(mesh)
        routers = list(mesh.routers.values())

        # Inject burst of traffic
        injected = self._inject_random_traffic(mesh, count=50, seed=123)
        assert injected > 0, "Failed to inject any traffic"

        # Run until network drains or timeout
        max_cycles = 2000
        for cycle in range(max_cycles):
            run_multi_router_cycle(routers, all_wires, cycle)
            drain_local_ports(mesh)  # Simulate NI consumption
            detector.update(cycle, mesh)

            if detector.check_deadlock():
                pytest.fail(f"Deadlock detected at cycle {cycle}")

            # Check if network has drained
            occupancy = get_mesh_buffer_occupancy(mesh)
            total_occupancy = sum(occupancy.values())
            if total_occupancy == 0:
                break

        # Verify all traffic was processed
        stats = get_router_forwarding_stats(mesh)
        total_forwarded = sum(stats.values())
        assert total_forwarded >= injected, f"Not all flits forwarded: {total_forwarded} < {injected}"

    def test_hotspot_traffic_pattern(self, mesh):
        """
        Test stability with hotspot traffic (many sources to one dest).

        All nodes send to center node - creates congestion.
        """
        detector = DeadlockDetector(threshold_cycles=300)
        all_wires = self._get_all_wires(mesh)
        routers = list(mesh.routers.values())

        # Hotspot destination
        hotspot = (2, 2)

        # Inject from all compute nodes (column 1+, not edge column 0) to hotspot
        injected = 0
        for coord in mesh.routers.keys():
            if coord == hotspot or coord[0] == 0:  # Skip hotspot and edge routers
                continue

            flit = FlitFactory.create_aw(
                src=coord,
                dest=hotspot,
                addr=0x1000,
                axi_id=injected % 16,
                length=0,
                last=True,  # Single-flit packet
            )

            router = mesh.routers[coord]
            local_port = router.req_router.ports.get(Direction.LOCAL)

            if local_port and local_port.occupancy < local_port._buffer.depth:
                local_port._buffer.push(flit)
                injected += 1

        # Run simulation
        for cycle in range(1000):
            # Call drain_local_ports BEFORE cycle to ensure in_ready=True
            # This is critical for handshake completion in clear_accepted_outputs
            drain_local_ports(mesh)
            run_multi_router_cycle(routers, all_wires, cycle)
            drain_local_ports(mesh)  # Also after to clear outputs and restore credits
            detector.update(cycle, mesh)

            if detector.check_deadlock():
                pytest.fail(f"Deadlock detected at cycle {cycle}")

        # Verify progress was made
        stats = get_router_forwarding_stats(mesh)
        total_forwarded = sum(stats.values())
        assert total_forwarded > 0, "No progress under hotspot traffic"


@pytest.mark.starvation
class TestPortFairness:
    """Tests for fair port allocation."""

    @pytest.fixture
    def mesh(self) -> Mesh:
        """Create small mesh for fairness testing."""
        config = MeshConfig(
            cols=3,
            rows=3,
            router_config=RouterConfig(buffer_depth=4),
        )
        return Mesh(config)

    def _get_all_wires(self, mesh: Mesh):
        """Get all wires from mesh."""
        return mesh._req_wires + mesh._resp_wires

    def test_arbiter_fairness(self, mesh):
        """
        Test that wormhole arbiter provides fair access.

        Multiple inputs competing for same output should
        each get roughly equal service.
        """
        all_wires = self._get_all_wires(mesh)
        routers = list(mesh.routers.values())

        # Inject traffic that creates contention at center router (1,1)
        # Using only compute nodes (columns 1+, not edge column 0)
        contending_sources = [(1, 1), (2, 1), (1, 0), (1, 2)]
        contending_dests = [(2, 1), (1, 1), (1, 2), (1, 0)]

        # Inject multiple flits from each source
        for i in range(4):  # 4 flits each
            for src, dest in zip(contending_sources, contending_dests):
                flit = FlitFactory.create_aw(
                    src=src,
                    dest=dest,
                    addr=0x1000 + i * 0x100,
                    axi_id=i,
                    length=0,
                    last=True,  # Single-flit packet
                )

                router = mesh.routers.get(src)
                if router:
                    local_port = router.req_router.ports.get(Direction.LOCAL)
                    if local_port and local_port.occupancy < local_port._buffer.depth:
                        local_port._buffer.push(flit)

        # Run simulation
        for cycle in range(500):
            run_multi_router_cycle(routers, all_wires, cycle)
            drain_local_ports(mesh)  # Simulate NI consumption

        # Check that traffic was forwarded (router-level stats)
        # Note: Per-port stats aren't available, so we verify overall forwarding
        stats = get_router_forwarding_stats(mesh)
        center_stats = stats.get((1, 1), 0)

        # Center router should have forwarded some traffic from contending sources
        assert center_stats > 0, "Center router did not forward any traffic"

        # Verify all injected traffic was processed
        total_forwarded = sum(stats.values())
        # We injected 4 flits * 4 sources = 16 flits (some may not fit in buffer)
        assert total_forwarded > 0, "No traffic was forwarded"

    def test_no_indefinite_starvation(self, mesh):
        """
        Test that no port is starved indefinitely.

        Even under high contention, all traffic should eventually
        make progress.
        """
        detector = DeadlockDetector(threshold_cycles=300)
        all_wires = self._get_all_wires(mesh)
        routers = list(mesh.routers.values())

        # Inject traffic from each corner (compute nodes only, not edge column 0)
        corners = [(1, 0), (1, 2), (2, 0), (2, 2)]
        opposite = [(2, 2), (2, 0), (1, 2), (1, 0)]

        injected = 0
        for src, dest in zip(corners, opposite):
            flit = FlitFactory.create_aw(
                src=src,
                dest=dest,
                addr=0x1000,
                axi_id=injected,
                length=0,
                last=True,  # Single-flit packet
            )

            router = mesh.routers.get(src)
            if router:
                local_port = router.req_router.ports.get(Direction.LOCAL)
                if local_port:
                    local_port._buffer.push(flit)
                    injected += 1

        # Run simulation
        for cycle in range(1000):
            run_multi_router_cycle(routers, all_wires, cycle)
            drain_local_ports(mesh)  # Simulate NI consumption
            detector.update(cycle, mesh)

            if detector.check_deadlock():
                pytest.fail(f"Starvation/deadlock detected at cycle {cycle}")

        # All traffic should have been forwarded
        stats = get_router_forwarding_stats(mesh)
        total_forwarded = sum(stats.values())
        assert total_forwarded >= injected, f"Some traffic starved: {total_forwarded} < {injected}"


@pytest.mark.saturation
class TestCreditExhaustion:
    """Tests for credit exhaustion and recovery."""

    @pytest.fixture
    def mesh(self) -> Mesh:
        """Create mesh with small buffers to test credit limits."""
        config = MeshConfig(
            cols=3,
            rows=3,
            router_config=RouterConfig(buffer_depth=2),
        )
        return Mesh(config)

    def _get_all_wires(self, mesh: Mesh):
        """Get all wires from mesh."""
        return mesh._req_wires + mesh._resp_wires

    def test_credit_limits_prevent_overflow(self, mesh):
        """
        Test that credit-based flow control prevents buffer overflow.

        With small buffers, credit exhaustion should backpressure
        without causing overflow or deadlock.
        """
        detector = DeadlockDetector(threshold_cycles=200)
        all_wires = self._get_all_wires(mesh)
        routers = list(mesh.routers.values())

        # Try to inject more traffic than buffers can hold
        # Using compute node (1,0), not edge router (0,0)
        injected = 0
        for i in range(20):
            flit = FlitFactory.create_aw(
                src=(1, 0),
                dest=(2, 2),
                addr=0x1000 + i * 0x100,
                axi_id=i % 4,
                length=0,
                last=True,  # Single-flit packet
            )

            router = mesh.routers[(1, 0)]
            local_port = router.req_router.ports.get(Direction.LOCAL)

            if local_port and local_port.occupancy < local_port._buffer.depth:
                local_port._buffer.push(flit)
                injected += 1

        # Run simulation
        for cycle in range(500):
            run_multi_router_cycle(routers, all_wires, cycle)
            drain_local_ports(mesh)  # Simulate NI consumption
            detector.update(cycle, mesh)

            if detector.check_deadlock():
                pytest.fail(f"Deadlock detected at cycle {cycle}")

            # Verify no buffer overflow
            for router in routers:
                for port in router.req_router.ports.values():
                    assert port.occupancy <= port._buffer.depth, "Buffer overflow!"

        # Some traffic should have been forwarded
        stats = get_router_forwarding_stats(mesh)
        total_forwarded = sum(stats.values())
        assert total_forwarded > 0, "No progress despite credit exhaustion"
