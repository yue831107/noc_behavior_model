"""
Theory-based Performance Validator.

Validates performance metrics against theoretical bounds using
Monitor-based approach (no core modification required).
"""

from typing import Tuple, Dict, Optional
from dataclasses import dataclass


@dataclass
class MeshConfig:
    """Mesh configuration for theoretical calculations."""
    cols: int = 5
    rows: int = 4
    edge_column: int = 0
    
    @property
    def compute_nodes(self) -> int:
        """Number of compute nodes (excluding edge column)."""
        return (self.cols - 1) * self.rows
    
    @property
    def total_routers(self) -> int:
        """Total number of routers including edge routers."""
        return self.cols * self.rows


@dataclass
class RouterConfig:
    """Router configuration for theoretical calculations."""
    flit_data_bytes: int = 32  # 256 bits = 32 bytes (FlooNoC W/R payload)
    pipeline_depth: int = 1    # Fast mode: 1-cycle
    buffer_depth: int = 4
    switching: str = "wormhole"


class TheoryValidator:
    """
    Validates performance metrics against theoretical bounds.
    
    Monitor-based validator: reads metrics from system without
    modifying core implementation.
    
    Ensures that simulated performance metrics do not violate
    fundamental physical and logical constraints.
    """
    
    def __init__(
        self, 
        mesh_config: Optional[MeshConfig] = None,
        router_config: Optional[RouterConfig] = None
    ):
        """
        Initialize validator with system configuration.
        
        Args:
            mesh_config: Mesh topology configuration
            router_config: Router hardware configuration
        """
        self.mesh_config = mesh_config or MeshConfig()
        self.router_config = router_config or RouterConfig()
    
    def calculate_max_throughput(self, bidirectional: bool = False) -> float:
        """
        Calculate theoretical maximum throughput for V1 Host-to-NoC.

        For V1 architecture, the bottleneck is the **single SlaveNI**,
        NOT the 4 Edge Routers. SlaveNI can inject at most 1 flit/cycle.

        Formula: T_max = N_NI × W_data (user data per flit)

        For V1 (5×4 mesh) with FlooNoC format:
        - SlaveNI count = 1
        - W flit data payload = 256 bits = 32 bytes
        - Single direction: 1 × 32 = 32 bytes/cycle

        Note: Bidirectional mode doubles this, but for write-only or
        read-only transfers, use single direction (default).

        Args:
            bidirectional: If True, count both Request and Response paths.
                          Default False for single-direction transfers.

        Returns:
            Maximum throughput in bytes/cycle (SlaveNI-limited)
        """
        # V1: SlaveNI bottleneck (1 NI, 1 flit/cycle per direction)
        num_slave_ni = 1  # V1 architecture has 1 Host SlaveNI

        # Each SlaveNI can inject 1 flit/cycle, W/R payload = 32 bytes
        single_direction = num_slave_ni * self.router_config.flit_data_bytes

        if bidirectional:
            # Request + Response both active (rare case)
            return single_direction * 2  # 64 bytes/cycle
        else:
            return single_direction  # 32 bytes/cycle
    
    def calculate_min_latency(
        self,
        src: Tuple[int, int],
        dest: Tuple[int, int],
        include_overhead: bool = True
    ) -> int:
        """
        Calculate theoretical minimum latency for a transfer.

        Min latency = manhattan_distance(src, dest) × router_pipeline_depth + overhead

        Overhead (when include_overhead=True):
        - NI packetization: ~1 cycle
        - Routing Selector: ~1 cycle

        Args:
            src: Source coordinate (x, y)
            dest: Destination coordinate (x, y)
            include_overhead: Include NI/Selector overhead (default True)

        Returns:
            Minimum latency in cycles
        """
        # Manhattan distance (XY routing)
        hops = abs(dest[0] - src[0]) + abs(dest[1] - src[1])

        # Pipeline latency per hop
        router_latency = hops * self.router_config.pipeline_depth

        if include_overhead:
            # NI + Selector overhead (~2 cycles)
            return router_latency + 2

        return router_latency
    
    def calculate_max_latency(
        self,
        src: Tuple[int, int],
        dest: Tuple[int, int],
        packet_size_flits: int = 1
    ) -> int:
        """
        Calculate theoretical maximum latency under congestion.

        Max latency = min_latency + buffer_queuing + serialization

        Components:
        - min_latency: hop count × pipeline depth + overhead
        - buffer_queuing: worst case full buffer at each hop
        - serialization: packet_size_flits - 1

        Args:
            src: Source coordinate
            dest: Destination coordinate
            packet_size_flits: Number of flits in packet

        Returns:
            Maximum reasonable latency in cycles
        """
        min_lat = self.calculate_min_latency(src, dest)
        hops = abs(dest[0] - src[0]) + abs(dest[1] - src[1])

        # Worst case: full buffer at each hop
        buffer_delay = hops * self.router_config.buffer_depth
        serialization_delay = packet_size_flits - 1

        # Max latency without arbitrary multiplier
        max_lat = min_lat + buffer_delay + serialization_delay

        return max_lat
    
    # =========================================================================
    # Validation Methods (Monitor-based)
    # =========================================================================
    
    def validate_throughput(
        self, 
        actual_throughput: float,
        tolerance: float = 0.05
    ) -> Tuple[bool, str]:
        """
        Validate throughput against theoretical maximum.
        
        Monitor-based: reads throughput metric from system.
        
        Args:
            actual_throughput: Measured throughput in bytes/cycle
            tolerance: Allowed tolerance above theoretical max (default 5%)
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        max_throughput = self.calculate_max_throughput()
        upper_bound = max_throughput * (1 + tolerance)
        
        if actual_throughput > upper_bound:
            error_msg = (
                f"Throughput violation: {actual_throughput:.2f} bytes/cycle "
                f"exceeds theoretical max {max_throughput:.2f} "
                f"(+{tolerance*100}% tolerance = {upper_bound:.2f})"
            )
            return False, error_msg
        
        if actual_throughput < 0:
            return False, f"Invalid negative throughput: {actual_throughput}"
        
        return True, "OK"
    
    def validate_latency(
        self,
        actual_latency: float,
        src: Tuple[int, int],
        dest: Tuple[int, int],
        packet_size_flits: int = 1,
        tolerance: float = 0.05
    ) -> Tuple[bool, str]:
        """
        Validate latency is within theoretical bounds.
        
        Monitor-based: reads latency metric from system.
        
        Args:
            actual_latency: Measured latency in cycles
            src: Source coordinate
            dest: Destination coordinate
            packet_size_flits: Packet size in flits
            tolerance: Allowed tolerance (default 5%)
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        min_lat = self.calculate_min_latency(src, dest)
        max_lat = self.calculate_max_latency(src, dest, packet_size_flits)
        
        lower_bound = min_lat * (1 - tolerance)
        
        if actual_latency < lower_bound:
            error_msg = (
                f"Latency violation: {actual_latency:.2f} cycles "
                f"is below theoretical minimum {min_lat} "
                f"(-{tolerance*100}% tolerance = {lower_bound:.2f})"
            )
            return False, error_msg
        
        if actual_latency > max_lat:
            error_msg = (
                f"Latency warning: {actual_latency:.2f} cycles "
                f"exceeds expected maximum {max_lat} "
                f"(may indicate deadlock or severe congestion)"
            )
            return False, error_msg
        
        return True, "OK"
    
    def validate_buffer_utilization(
        self,
        actual_utilization: float
    ) -> Tuple[bool, str]:
        """
        Validate buffer utilization is in valid range [0, 1].
        
        Monitor-based: reads buffer utilization from system.
        
        Args:
            actual_utilization: Measured buffer utilization (0.0 to 1.0)
        
        Returns:
            Tuple of (is_valid, error_message)
        """
        if actual_utilization < 0.0:
            return False, f"Invalid negative buffer utilization: {actual_utilization}"
        
        if actual_utilization > 1.0:
            error_msg = (
                f"Buffer utilization {actual_utilization:.2%} exceeds 100% "
                f"(indicates measurement error or overflow)"
            )
            return False, error_msg
        
        return True, "OK"
    
    def validate_all(self, metrics: Dict) -> Dict[str, Tuple[bool, str]]:
        """
        Validate all metrics in a single call.
        
        Monitor-based: reads all metrics from system via dictionary.
        
        Args:
            metrics: Dictionary containing performance metrics
                Expected keys: throughput, avg_latency, buffer_utilization
                Optional keys: src, dest, packet_size_flits
        
        Returns:
            Dictionary of validation results for each metric
        """
        results = {}
        
        # Validate throughput
        if 'throughput' in metrics:
            results['throughput'] = self.validate_throughput(
                metrics['throughput']
            )
        
        # Validate latency (if src/dest available)
        if all(k in metrics for k in ['avg_latency', 'src', 'dest']):
            results['latency'] = self.validate_latency(
                metrics['avg_latency'],
                metrics['src'],
                metrics['dest'],
                metrics.get('packet_size_flits', 1)
            )
        
        # Validate buffer utilization
        if 'buffer_utilization' in metrics:
            results['buffer_utilization'] = self.validate_buffer_utilization(
                metrics['buffer_utilization']
            )
        
        return results


def print_validation_results(results: Dict[str, Tuple[bool, str]]) -> None:
    """
    Pretty-print validation results.
    
    Args:
        results: Dictionary of validation results from validate_all()
    """
    print("=" * 70)
    print("Theory Validation Results")
    print("=" * 70)
    
    all_passed = True
    for metric, (is_valid, message) in results.items():
        status = "✓ PASS" if is_valid else "✗ FAIL"
        print(f"{metric:25s} {status:8s} {message}")
        all_passed = all_passed and is_valid
    
    print("=" * 70)
    print(f"Overall: {'PASS' if all_passed else 'FAIL'}")
    print("=" * 70)
