"""
Deadlock detection helpers for integration tests.

Provides:
- DeadlockDetector: Monitor for detecting stuck packets
- NetworkHealthMetrics: Aggregated network health statistics
- Helper functions for deadlock testing
"""

from dataclasses import dataclass, field
from typing import Dict, List, Set, Optional, Tuple
from collections import defaultdict

from src.core.mesh import Mesh
from src.core.router import Direction


@dataclass
class NetworkHealthMetrics:
    """Network health state metrics."""

    total_injected: int = 0
    total_delivered: int = 0
    stuck_packets: int = 0
    avg_latency: float = 0.0
    max_latency: int = 0
    buffer_utilization: float = 0.0
    throughput: float = 0.0


class DeadlockDetector:
    """
    Monitor and detect deadlock conditions during simulation.

    Tracks packet progress and detects when packets are stuck
    without making forward progress for extended periods.
    """

    def __init__(self, threshold_cycles: int = 500):
        """
        Initialize deadlock detector.

        Args:
            threshold_cycles: Number of cycles without progress to consider deadlock.
        """
        self.threshold = threshold_cycles
        self._current_cycle: int = 0

        # Track buffer occupancy over time
        self._occupancy_history: List[int] = []
        self._last_total_occupancy: int = 0

        # Track total flits forwarded
        self._total_forwarded: int = 0
        self._forwarded_history: List[int] = []

        # Detect sustained high occupancy
        self._high_occupancy_streak: int = 0
        self._max_occupancy_streak: int = 0

    def update(self, cycle: int, mesh: Mesh) -> None:
        """
        Update tracker with current cycle state.

        Args:
            cycle: Current simulation cycle.
            mesh: Mesh to monitor.
        """
        self._current_cycle = cycle

        # Get total buffer occupancy (including output signals)
        total_occupancy = 0
        total_capacity = 0

        for router in mesh.routers.values():
            for port in router.req_router.ports.values():
                total_occupancy += port.occupancy
                total_capacity += port._buffer.depth
                # Also count flits on output signals
                if port.out_valid:
                    total_occupancy += 1
            for port in router.resp_router.ports.values():
                total_occupancy += port.occupancy
                total_capacity += port._buffer.depth
                if port.out_valid:
                    total_occupancy += 1

        self._occupancy_history.append(total_occupancy)
        self._last_total_occupancy = total_occupancy

        # Track forwarding progress
        total_forwarded = sum(
            r.req_router.stats.flits_forwarded + r.resp_router.stats.flits_forwarded
            for r in mesh.routers.values()
        )
        self._forwarded_history.append(total_forwarded)
        self._total_forwarded = total_forwarded

        # Detect sustained high occupancy (>80% full)
        if total_capacity > 0:
            utilization = total_occupancy / total_capacity
            if utilization > 0.8:
                self._high_occupancy_streak += 1
                self._max_occupancy_streak = max(
                    self._max_occupancy_streak, self._high_occupancy_streak
                )
            else:
                self._high_occupancy_streak = 0

    def check_deadlock(self) -> bool:
        """
        Check if deadlock condition is detected.

        Returns:
            True if deadlock detected, False otherwise.
        """
        # If no/minimal traffic in the network, no deadlock
        # Use threshold > 2 to avoid false positives from transient residual flits
        if self._last_total_occupancy <= 2:
            return False

        # Check 1: Sustained high occupancy without progress
        if self._high_occupancy_streak >= self.threshold:
            # Also check if forwarding has stalled
            if len(self._forwarded_history) >= self.threshold:
                recent_progress = (
                    self._forwarded_history[-1]
                    - self._forwarded_history[-self.threshold]
                )
                if recent_progress == 0:
                    return True

        # Check 2: No forwarding progress for threshold cycles (with substantial traffic still present)
        if len(self._forwarded_history) >= self.threshold:
            start_forwarded = self._forwarded_history[-self.threshold]
            end_forwarded = self._forwarded_history[-1]
            if start_forwarded == end_forwarded and start_forwarded > 0:
                # Had some activity but now stalled - only deadlock if substantial traffic still present
                if self._last_total_occupancy > 2:
                    return True

        return False

    def get_health_metrics(self, mesh: Mesh) -> NetworkHealthMetrics:
        """
        Get current network health metrics.

        Args:
            mesh: Mesh to analyze.

        Returns:
            NetworkHealthMetrics with current state.
        """
        total_occupancy = 0
        total_capacity = 0
        total_forwarded = 0

        for router in mesh.routers.values():
            for port in router.req_router.ports.values():
                total_occupancy += port.occupancy
                total_capacity += port._buffer.depth
            for port in router.resp_router.ports.values():
                total_occupancy += port.occupancy
                total_capacity += port._buffer.depth
            total_forwarded += router.req_router.stats.flits_forwarded
            total_forwarded += router.resp_router.stats.flits_forwarded

        utilization = total_occupancy / total_capacity if total_capacity > 0 else 0.0

        # Calculate throughput (flits per cycle)
        throughput = 0.0
        if self._current_cycle > 0:
            throughput = total_forwarded / self._current_cycle

        return NetworkHealthMetrics(
            total_injected=0,  # Would need external tracking
            total_delivered=0,  # Would need external tracking
            stuck_packets=self._high_occupancy_streak if self.check_deadlock() else 0,
            avg_latency=0.0,  # Would need latency tracking
            max_latency=0,
            buffer_utilization=utilization,
            throughput=throughput,
        )

    def get_progress_rate(self, window: int = 100) -> float:
        """
        Get recent forwarding progress rate.

        Args:
            window: Number of cycles to consider.

        Returns:
            Flits forwarded per cycle in recent window.
        """
        if len(self._forwarded_history) < window:
            window = len(self._forwarded_history)

        if window < 2:
            return 0.0

        start = self._forwarded_history[-window]
        end = self._forwarded_history[-1]
        return (end - start) / window


def get_mesh_buffer_occupancy(mesh: Mesh) -> Dict[Tuple[int, int], int]:
    """
    Get buffer occupancy for each router.

    Args:
        mesh: Mesh to analyze.

    Returns:
        Dict of (x, y) -> total buffer occupancy.
    """
    occupancy = {}
    for coord, router in mesh.routers.items():
        total = 0
        for port in router.req_router.ports.values():
            total += port.occupancy
        for port in router.resp_router.ports.values():
            total += port.occupancy
        occupancy[coord] = total
    return occupancy


def get_router_forwarding_stats(mesh: Mesh) -> Dict[Tuple[int, int], int]:
    """
    Get forwarding statistics for each router.

    Args:
        mesh: Mesh to analyze.

    Returns:
        Dict of (x, y) -> total flits forwarded.
    """
    stats = {}
    for coord, router in mesh.routers.items():
        total = router.req_router.stats.flits_forwarded
        total += router.resp_router.stats.flits_forwarded
        stats[coord] = total
    return stats


def assert_no_deadlock(mesh: Mesh, cycles: int = 2000) -> None:
    """
    Run simulation and assert no deadlock occurs.

    Args:
        mesh: Mesh to simulate.
        cycles: Number of cycles to run.

    Raises:
        AssertionError: If deadlock detected.
    """
    detector = DeadlockDetector(threshold_cycles=500)

    for cycle in range(cycles):
        mesh.process_cycle(cycle)
        detector.update(cycle, mesh)

        if detector.check_deadlock():
            metrics = detector.get_health_metrics(mesh)
            raise AssertionError(
                f"Deadlock detected at cycle {cycle}. "
                f"Buffer utilization: {metrics.buffer_utilization:.2%}, "
                f"Progress rate: {detector.get_progress_rate():.2f} flits/cycle"
            )


def get_port_utilization_stats(mesh: Mesh) -> Dict[Direction, float]:
    """
    Get utilization statistics per port direction.

    Args:
        mesh: Mesh to analyze.

    Returns:
        Dict of Direction -> average utilization ratio.
    """
    direction_counts: Dict[Direction, List[int]] = defaultdict(list)

    for router in mesh.routers.values():
        for direction, port in router.req_router.ports.items():
            direction_counts[direction].append(port.stats.flits_received)

    utilization = {}
    total_flits = sum(sum(counts) for counts in direction_counts.values())

    if total_flits > 0:
        for direction, counts in direction_counts.items():
            utilization[direction] = sum(counts) / total_flits

    return utilization


def drain_local_ports(mesh: Mesh) -> int:
    """
    Drain flits from LOCAL ports that have reached their destinations.

    This simulates NI consumption of incoming flits, preventing LOCAL
    port backpressure during testing. It clears both the output signal
    (delivered flits waiting for NI) and any flits in the input buffer
    that have reached their destination.

    IMPORTANT: Also sets in_ready=True on LOCAL ports to simulate NI being
    ready to accept flits. Without this, the router cannot forward new flits
    to LOCAL port because in_ready would remain False.

    Args:
        mesh: Mesh to drain.

    Returns:
        Number of flits drained.
    """
    from src.core.flit import decode_node_id

    drained = 0
    for coord, router in mesh.routers.items():
        # Only drain at compute nodes (not edge routers)
        if coord[0] == 0:
            continue

        # Drain from request network LOCAL port
        local_port = router.req_router.ports.get(Direction.LOCAL)
        if local_port:
            # Set in_ready to True to simulate NI being ready to accept
            # This is critical - without it, router cannot forward to LOCAL
            local_port.in_ready = True

            # Restore credits to full capacity to simulate NI always ready
            # Credits are consumed by clear_accepted_outputs during handshake,
            # so we need to restore them to allow continuous forwarding
            initial_credits = local_port._buffer.depth
            local_port._output_credit.credits = initial_credits

            # Clear output signal (flit waiting for NI)
            if local_port.out_valid and local_port.out_flit:
                dst_coord = decode_node_id(local_port.out_flit.hdr.dst_id)
                if dst_coord == coord:
                    local_port.out_valid = False
                    local_port.out_flit = None
                    drained += 1

            # Also drain input buffer (flits delivered but not yet forwarded to output)
            while not local_port._buffer.is_empty():
                flit = local_port._buffer.peek()
                if flit:
                    dst_coord = decode_node_id(flit.hdr.dst_id)
                    if dst_coord == coord:
                        local_port._buffer.pop()
                        drained += 1
                    else:
                        break
                else:
                    break

        # Drain from response network LOCAL port
        local_port = router.resp_router.ports.get(Direction.LOCAL)
        if local_port:
            # Set in_ready to True to simulate NI being ready to accept
            local_port.in_ready = True

            # Restore credits to full capacity
            initial_credits = local_port._buffer.depth
            local_port._output_credit.credits = initial_credits

            # Clear output signal
            if local_port.out_valid and local_port.out_flit:
                dst_coord = decode_node_id(local_port.out_flit.hdr.dst_id)
                if dst_coord == coord:
                    local_port.out_valid = False
                    local_port.out_flit = None
                    drained += 1

            # Drain input buffer
            while not local_port._buffer.is_empty():
                flit = local_port._buffer.peek()
                if flit:
                    dst_coord = decode_node_id(flit.hdr.dst_id)
                    if dst_coord == coord:
                        local_port._buffer.pop()
                        drained += 1
                    else:
                        break
                else:
                    break

    return drained
