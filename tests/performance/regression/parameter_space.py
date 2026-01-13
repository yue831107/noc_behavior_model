"""
Parameter space definition for hardware configuration search.

ParameterSpace defines the searchable hardware parameters and provides
methods to generate combinations and prefilter infeasible ones using
theoretical bounds.
"""

from dataclasses import dataclass, field
from typing import List, Dict, Any, Iterator, Optional
import itertools

from .target import PerformanceTarget


@dataclass
class ParameterSpace:
    """
    Searchable hardware parameter space.

    Attributes:
        mesh_rows: List of mesh row counts to try.
        mesh_cols: List of mesh column counts to try.
        buffer_depth: List of router buffer depths to try.
        max_outstanding: List of NI max outstanding transactions.
        flit_width_bytes: Flit width in bytes (usually fixed at 8).
    """

    mesh_rows: List[int] = field(default_factory=lambda: [2, 4, 6, 8])
    mesh_cols: List[int] = field(default_factory=lambda: [3, 5, 7, 9])
    buffer_depth: List[int] = field(default_factory=lambda: [4, 8, 16, 32])
    max_outstanding: List[int] = field(default_factory=lambda: [8, 16, 32])
    flit_width_bytes: int = 8  # Fixed for V1 architecture

    def total_combinations(self) -> int:
        """Calculate total number of parameter combinations."""
        return (
            len(self.mesh_rows)
            * len(self.mesh_cols)
            * len(self.buffer_depth)
            * len(self.max_outstanding)
        )

    def generate_combinations(self) -> Iterator[Dict[str, Any]]:
        """
        Generate all parameter combinations.

        Yields:
            Dictionary with parameter values for each combination.
        """
        for rows, cols, buf, outstanding in itertools.product(
            self.mesh_rows,
            self.mesh_cols,
            self.buffer_depth,
            self.max_outstanding,
        ):
            yield {
                "mesh_rows": rows,
                "mesh_cols": cols,
                "buffer_depth": buf,
                "max_outstanding": outstanding,
            }

    def prefilter(
        self,
        target: PerformanceTarget,
        verbose: bool = False,
    ) -> List[Dict[str, Any]]:
        """
        Prefilter parameter combinations using theoretical bounds.

        Uses TheoryValidator formulas to eliminate combinations that
        cannot possibly meet the target:
        - T_max = rows × flit_width: Skip if T_max < min_throughput
        - L_min = max_hops + overhead: Skip if L_min > max_latency

        Args:
            target: Performance target to check against.
            verbose: Print filtering statistics.

        Returns:
            List of feasible parameter combinations.
        """
        feasible = []
        total = 0
        filtered_throughput = 0
        filtered_latency = 0

        for params in self.generate_combinations():
            total += 1
            rows = params["mesh_rows"]
            cols = params["mesh_cols"]

            # Throughput upper bound: T_max = rows × flit_width
            # For V1 architecture, edge routers limit throughput
            t_max = rows * self.flit_width_bytes

            if t_max < target.min_throughput:
                filtered_throughput += 1
                continue

            # Latency lower bound: L_min = max_hops + overhead
            # Max hops = (cols - 1) + (rows - 1) for XY routing
            # Overhead = 2 cycles (NI + Selector)
            max_hops = (cols - 1) + (rows - 1)
            l_min = max_hops + 2

            if l_min > target.max_latency:
                filtered_latency += 1
                continue

            feasible.append(params)

        if verbose:
            print(f"Parameter space prefiltering:")
            print(f"  Total combinations: {total}")
            print(f"  Filtered (throughput): {filtered_throughput}")
            print(f"  Filtered (latency): {filtered_latency}")
            print(f"  Feasible: {len(feasible)}")

        return feasible

    def estimate_theoretical_bounds(
        self, params: Dict[str, Any]
    ) -> Dict[str, float]:
        """
        Estimate theoretical performance bounds for a parameter set.

        Args:
            params: Parameter dictionary with mesh_rows, mesh_cols, etc.

        Returns:
            Dictionary with t_max (throughput upper bound) and
            l_min (latency lower bound).
        """
        rows = params["mesh_rows"]
        cols = params["mesh_cols"]

        # Throughput upper bound
        t_max = rows * self.flit_width_bytes

        # Latency lower bound (worst case: corner to corner)
        max_hops = (cols - 1) + (rows - 1)
        l_min = max_hops + 2  # +2 for NI/Selector overhead

        # Latency upper bound (worst case: full buffer queuing)
        buffer_depth = params.get("buffer_depth", 4)
        l_max = l_min + max_hops * buffer_depth

        return {
            "t_max": t_max,
            "l_min": l_min,
            "l_max": l_max,
            "max_hops": max_hops,
        }

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "mesh_rows": self.mesh_rows,
            "mesh_cols": self.mesh_cols,
            "buffer_depth": self.buffer_depth,
            "max_outstanding": self.max_outstanding,
            "flit_width_bytes": self.flit_width_bytes,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "ParameterSpace":
        """Create from dictionary."""
        return cls(
            mesh_rows=data.get("mesh_rows", [2, 4, 6, 8]),
            mesh_cols=data.get("mesh_cols", [3, 5, 7, 9]),
            buffer_depth=data.get("buffer_depth", [4, 8, 16, 32]),
            max_outstanding=data.get("max_outstanding", [8, 16, 32]),
            flit_width_bytes=data.get("flit_width_bytes", 8),
        )

    def __repr__(self) -> str:
        return (
            f"ParameterSpace("
            f"rows={self.mesh_rows}, "
            f"cols={self.mesh_cols}, "
            f"buffer={self.buffer_depth}, "
            f"outstanding={self.max_outstanding}, "
            f"total={self.total_combinations()})"
        )
