"""
Report generation for regression test results.

RegressionReport generates text summaries and charts
from optimization results.
"""

from typing import Optional, List, Dict, Any
from pathlib import Path
import json

from .result import OptimizationResult


class RegressionReport:
    """
    Report generator for regression test results.

    Generates text summaries, JSON exports, and optional charts.
    """

    def __init__(self, result: OptimizationResult):
        """
        Initialize report generator.

        Args:
            result: Optimization result to report on.
        """
        self.result = result

    def generate_summary(self) -> str:
        """
        Generate a text summary of the results.

        Returns:
            Multi-line summary string.
        """
        return self.result.summary()

    def generate_top_n_table(self, n: int = 5) -> str:
        """
        Generate a table of top N solutions.

        Args:
            n: Number of top solutions to show.

        Returns:
            Formatted table string.
        """
        top_n = self.result.get_top_n(n)

        if not top_n:
            return "No solutions found."

        lines = [
            "Top {} Solutions:".format(min(n, len(top_n))),
            "-" * 80,
            "{:<5} {:<8} {:<8} {:<8} {:<10} {:<10} {:<10}".format(
                "Rank", "Rows", "Cols", "Buffer", "Latency", "Throughput", "Score"
            ),
            "-" * 80,
        ]

        for i, entry in enumerate(top_n, 1):
            params = entry["params"]
            metrics = entry["metrics"]
            score = entry["score"]

            lines.append(
                "{:<5} {:<8} {:<8} {:<8} {:<10.2f} {:<10.2f} {:<10.4f}".format(
                    i,
                    params.get("mesh_rows", "?"),
                    params.get("mesh_cols", "?"),
                    params.get("buffer_depth", "?"),
                    metrics.get("avg_latency", 0),
                    metrics.get("throughput", 0),
                    score,
                )
            )

        lines.append("-" * 80)
        return "\n".join(lines)

    def generate_parameter_analysis(self) -> str:
        """
        Analyze which parameter values tend to produce better results.

        Returns:
            Analysis text.
        """
        if not self.result.all_results:
            return "No results to analyze."

        # Group by each parameter
        param_scores: Dict[str, Dict[Any, List[float]]] = {}

        for entry in self.result.all_results:
            params = entry["params"]
            score = entry["score"]

            for key, value in params.items():
                if key not in param_scores:
                    param_scores[key] = {}
                if value not in param_scores[key]:
                    param_scores[key][value] = []
                param_scores[key][value].append(score)

        lines = ["Parameter Analysis:", "-" * 60]

        for param_name, value_scores in param_scores.items():
            lines.append(f"\n{param_name}:")

            # Calculate average score for each value
            value_avgs = []
            for value, scores in value_scores.items():
                avg = sum(scores) / len(scores)
                value_avgs.append((value, avg, len(scores)))

            # Sort by average score
            value_avgs.sort(key=lambda x: x[1], reverse=True)

            for value, avg, count in value_avgs:
                lines.append(f"  {value}: avg_score={avg:.4f} (n={count})")

        return "\n".join(lines)

    def save_json(self, path: Path) -> None:
        """
        Save results to JSON file.

        Args:
            path: Output file path.
        """
        self.result.save(path)

    def save_full_report(self, output_dir: Path) -> None:
        """
        Save complete report (JSON + text summary).

        Args:
            output_dir: Output directory.
        """
        output_dir = Path(output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save JSON
        json_path = output_dir / "regression_result.json"
        self.save_json(json_path)

        # Save text summary
        summary_path = output_dir / "summary.txt"
        with open(summary_path, "w", encoding="utf-8") as f:
            f.write(self.generate_summary())
            f.write("\n\n")
            f.write(self.generate_top_n_table())
            f.write("\n\n")
            f.write(self.generate_parameter_analysis())

        print(f"Report saved to {output_dir}")
        print(f"  - {json_path.name}")
        print(f"  - {summary_path.name}")

    def plot_score_distribution(self, save_path: Optional[Path] = None):
        """
        Plot distribution of scores across all tested combinations.

        Args:
            save_path: Optional path to save the figure.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for plotting")
            return

        scores = [r["score"] for r in self.result.all_results]

        if not scores:
            print("No scores to plot")
            return

        fig, ax = plt.subplots(figsize=(10, 6))

        ax.hist(scores, bins=20, edgecolor="black", alpha=0.7)
        ax.axvline(
            self.result.best_score,
            color="red",
            linestyle="--",
            label=f"Best: {self.result.best_score:.4f}",
        )
        ax.axvline(
            1.0,
            color="green",
            linestyle=":",
            label="Target threshold",
        )

        ax.set_xlabel("Score")
        ax.set_ylabel("Count")
        ax.set_title("Score Distribution")
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

        plt.close()

    def plot_latency_vs_throughput(self, save_path: Optional[Path] = None):
        """
        Plot latency vs throughput for all tested combinations.

        Args:
            save_path: Optional path to save the figure.
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("matplotlib not available for plotting")
            return

        latencies = []
        throughputs = []
        satisfied = []

        for r in self.result.all_results:
            metrics = r["metrics"]
            latencies.append(metrics.get("avg_latency", 0))
            throughputs.append(metrics.get("throughput", 0))
            satisfied.append(r.get("satisfied", False))

        if not latencies:
            print("No data to plot")
            return

        fig, ax = plt.subplots(figsize=(10, 8))

        # Plot all points
        colors = ["green" if s else "gray" for s in satisfied]
        ax.scatter(latencies, throughputs, c=colors, alpha=0.6, s=50)

        # Highlight best
        best_lat = self.result.best_metrics.get("avg_latency", 0)
        best_tp = self.result.best_metrics.get("throughput", 0)
        ax.scatter(
            [best_lat], [best_tp],
            c="red", s=200, marker="*",
            label="Best",
            zorder=10,
        )

        # Target lines
        ax.axvline(
            self.result.target.max_latency,
            color="blue", linestyle="--", alpha=0.5,
            label=f"Max latency: {self.result.target.max_latency}",
        )
        ax.axhline(
            self.result.target.min_throughput,
            color="orange", linestyle="--", alpha=0.5,
            label=f"Min throughput: {self.result.target.min_throughput}",
        )

        ax.set_xlabel("Average Latency (cycles)")
        ax.set_ylabel("Throughput (bytes/cycle)")
        ax.set_title("Latency vs Throughput Trade-off")
        ax.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=150)
            print(f"Plot saved to {save_path}")
        else:
            plt.show()

        plt.close()
