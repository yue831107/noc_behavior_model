"""
Optimization result container for regression testing.

OptimizationResult stores the best parameters found, metrics,
and search statistics.
"""

from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional
from pathlib import Path
import json

from .target import PerformanceTarget


@dataclass
class OptimizationResult:
    """
    Optimization result container.

    Attributes:
        best_params: Best parameter combination found.
        best_metrics: Metrics for the best parameters.
        best_score: Weighted score of the best solution.
        target_satisfied: Whether the target was met.
        target: The performance target used.
        total_combinations: Total parameter combinations in space.
        feasible_count: Combinations passing prefilter.
        tested_count: Combinations actually tested.
        satisfied_count: Combinations meeting the target.
        all_results: List of all tested {params, metrics, score}.
    """

    best_params: Dict[str, Any]
    best_metrics: Dict[str, float]
    best_score: float
    target_satisfied: bool
    target: PerformanceTarget
    total_combinations: int
    feasible_count: int
    tested_count: int
    satisfied_count: int
    all_results: List[Dict[str, Any]] = field(default_factory=list)

    def get_top_n(self, n: int = 5) -> List[Dict[str, Any]]:
        """
        Get top N solutions by score.

        Args:
            n: Number of top solutions to return.

        Returns:
            List of {params, metrics, score} dictionaries.
        """
        sorted_results = sorted(
            self.all_results,
            key=lambda x: x.get("score", 0),
            reverse=True,
        )
        return sorted_results[:n]

    def get_satisfied_solutions(self) -> List[Dict[str, Any]]:
        """
        Get all solutions that satisfy the target.

        Returns:
            List of {params, metrics, score} for satisfied solutions.
        """
        return [r for r in self.all_results if r.get("satisfied", False)]

    def summary(self) -> str:
        """
        Generate a text summary of the results.

        Returns:
            Multi-line summary string.
        """
        lines = [
            "=" * 60,
            "Regression Test Results",
            "=" * 60,
            "",
            f"Target: latency <= {self.target.max_latency} cycles, "
            f"throughput >= {self.target.min_throughput} B/cycle",
            "",
            "Search Statistics:",
            f"  Total combinations: {self.total_combinations}",
            f"  Feasible (prefilter): {self.feasible_count}",
            f"  Tested: {self.tested_count}",
            f"  Satisfied: {self.satisfied_count}",
            "",
        ]

        if self.target_satisfied:
            lines.append("Result: TARGET SATISFIED")
        else:
            lines.append("Result: TARGET NOT SATISFIED (showing best effort)")

        lines.extend([
            "",
            "Best Parameters:",
        ])
        for key, value in self.best_params.items():
            lines.append(f"  {key}: {value}")

        lines.extend([
            "",
            "Best Metrics:",
            f"  Latency: {self.best_metrics.get('avg_latency', 'N/A'):.2f} cycles",
            f"  Throughput: {self.best_metrics.get('throughput', 'N/A'):.2f} B/cycle",
            f"  Score: {self.best_score:.4f}",
            "",
            "=" * 60,
        ])

        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        return {
            "best_params": self.best_params,
            "best_metrics": self.best_metrics,
            "best_score": self.best_score,
            "target_satisfied": self.target_satisfied,
            "target": self.target.to_dict(),
            "statistics": {
                "total_combinations": self.total_combinations,
                "feasible_count": self.feasible_count,
                "tested_count": self.tested_count,
                "satisfied_count": self.satisfied_count,
            },
            "all_results": self.all_results,
        }

    def save(self, path: Path) -> None:
        """
        Save results to JSON file.

        Args:
            path: Output file path.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        with open(path, "w", encoding="utf-8") as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, path: Path) -> "OptimizationResult":
        """
        Load results from JSON file.

        Args:
            path: Input file path.

        Returns:
            OptimizationResult instance.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        target = PerformanceTarget.from_dict(data["target"])
        stats = data["statistics"]

        return cls(
            best_params=data["best_params"],
            best_metrics=data["best_metrics"],
            best_score=data["best_score"],
            target_satisfied=data["target_satisfied"],
            target=target,
            total_combinations=stats["total_combinations"],
            feasible_count=stats["feasible_count"],
            tested_count=stats["tested_count"],
            satisfied_count=stats["satisfied_count"],
            all_results=data.get("all_results", []),
        )

    def __repr__(self) -> str:
        status = "SATISFIED" if self.target_satisfied else "NOT SATISFIED"
        return (
            f"OptimizationResult({status}, "
            f"score={self.best_score:.4f}, "
            f"tested={self.tested_count}/{self.feasible_count})"
        )
