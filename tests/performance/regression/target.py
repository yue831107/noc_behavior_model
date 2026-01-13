"""
Performance target definition for regression testing.

PerformanceTarget specifies the optimization goals:
- max_latency: Maximum acceptable latency in cycles
- min_throughput: Minimum acceptable throughput in bytes/cycle
- Weights for multi-objective scoring
"""

from dataclasses import dataclass
from typing import Dict, Any


@dataclass
class PerformanceTarget:
    """
    Performance target specification.

    Attributes:
        max_latency: Maximum acceptable average latency (cycles).
        min_throughput: Minimum acceptable throughput (bytes/cycle).
        max_buffer_utilization: Maximum buffer utilization ratio (0-1).
        latency_weight: Weight for latency in scoring (0-1).
        throughput_weight: Weight for throughput in scoring (0-1).
    """

    max_latency: float
    min_throughput: float
    max_buffer_utilization: float = 0.9
    latency_weight: float = 0.5
    throughput_weight: float = 0.5

    def __post_init__(self):
        """Validate weights sum to 1."""
        total = self.latency_weight + self.throughput_weight
        if abs(total - 1.0) > 0.001:
            # Normalize weights
            self.latency_weight = self.latency_weight / total
            self.throughput_weight = self.throughput_weight / total

    def is_satisfied(self, metrics: Dict[str, float]) -> bool:
        """
        Check if metrics satisfy the target.

        Args:
            metrics: Dictionary with 'avg_latency', 'throughput',
                     and optionally 'buffer_utilization'.

        Returns:
            True if all constraints are satisfied.
        """
        avg_latency = metrics.get("avg_latency", float("inf"))
        throughput = metrics.get("throughput", 0.0)
        buffer_util = metrics.get("buffer_utilization", 0.0)

        latency_ok = avg_latency <= self.max_latency
        throughput_ok = throughput >= self.min_throughput
        buffer_ok = buffer_util <= self.max_buffer_utilization

        return latency_ok and throughput_ok and buffer_ok

    def score(self, metrics: Dict[str, float]) -> float:
        """
        Calculate weighted score for ranking solutions.

        Higher score is better. Score is normalized to [0, 1] range
        based on how well the solution meets the targets.

        Args:
            metrics: Dictionary with 'avg_latency' and 'throughput'.

        Returns:
            Weighted score (higher is better).
        """
        avg_latency = metrics.get("avg_latency", float("inf"))
        throughput = metrics.get("throughput", 0.0)

        # Latency score: 1.0 at target, higher when below target
        # Score drops linearly as latency exceeds target
        if avg_latency <= 0:
            latency_score = 0.0
        elif avg_latency <= self.max_latency:
            # Better than target: score > 1.0
            latency_score = self.max_latency / avg_latency
        else:
            # Worse than target: score < 1.0, approaches 0
            latency_score = self.max_latency / avg_latency

        # Throughput score: 1.0 at target, higher when above target
        if self.min_throughput <= 0:
            throughput_score = 1.0 if throughput > 0 else 0.0
        elif throughput >= self.min_throughput:
            # Better than target: score >= 1.0
            throughput_score = throughput / self.min_throughput
        else:
            # Worse than target: score < 1.0
            throughput_score = throughput / self.min_throughput

        # Weighted combination
        score = (
            self.latency_weight * latency_score
            + self.throughput_weight * throughput_score
        )

        return score

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return {
            "max_latency": self.max_latency,
            "min_throughput": self.min_throughput,
            "max_buffer_utilization": self.max_buffer_utilization,
            "latency_weight": self.latency_weight,
            "throughput_weight": self.throughput_weight,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "PerformanceTarget":
        """Create from dictionary."""
        return cls(
            max_latency=data["max_latency"],
            min_throughput=data["min_throughput"],
            max_buffer_utilization=data.get("max_buffer_utilization", 0.9),
            latency_weight=data.get("latency_weight", 0.5),
            throughput_weight=data.get("throughput_weight", 0.5),
        )

    def __repr__(self) -> str:
        return (
            f"PerformanceTarget(max_latency={self.max_latency}, "
            f"min_throughput={self.min_throughput})"
        )
