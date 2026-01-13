"""
Parameter Sweep Results.

Data structure for storing and analyzing sweep results.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Tuple, Optional
from pathlib import Path
import json


@dataclass
class SweepResults:
    """
    Results from a parameter sweep.

    Stores all data points from sweep runs and provides
    methods for analysis and export.
    """
    # Raw data: list of dicts, each containing params + metrics
    raw_data: List[Dict[str, Any]] = field(default_factory=list)

    # Metadata
    parameters: List[str] = field(default_factory=list)
    metrics: List[str] = field(default_factory=list)
    system_type: str = "host_to_noc"

    def add_result(
        self,
        params: Dict[str, Any],
        metrics: Dict[str, float]
    ) -> None:
        """
        Add a single result to the dataset.

        Args:
            params: Parameter values for this run
            metrics: Measured metrics for this run
        """
        entry = {**params, **metrics}
        self.raw_data.append(entry)

        # Update parameter and metric lists
        for key in params:
            if key not in self.parameters:
                self.parameters.append(key)

        for key in metrics:
            if key not in self.metrics:
                self.metrics.append(key)

    def get_series(
        self,
        param: str,
        metric: str,
        aggregate: str = "mean"
    ) -> Tuple[List[Any], List[float]]:
        """
        Get (parameter_values, metric_values) series for plotting.

        Args:
            param: Parameter name for X-axis
            metric: Metric name for Y-axis
            aggregate: How to aggregate repeated values ("mean", "min", "max")

        Returns:
            Tuple of (x_values, y_values) sorted by x
        """
        if not self.raw_data:
            return [], []

        # Group by parameter value
        grouped: Dict[Any, List[float]] = {}
        for entry in self.raw_data:
            if param in entry and metric in entry:
                key = entry[param]
                value = entry[metric]
                if key not in grouped:
                    grouped[key] = []
                grouped[key].append(value)

        # Aggregate
        x_values = []
        y_values = []

        for key in sorted(grouped.keys()):
            values = grouped[key]
            x_values.append(key)

            if aggregate == "mean":
                y_values.append(sum(values) / len(values))
            elif aggregate == "min":
                y_values.append(min(values))
            elif aggregate == "max":
                y_values.append(max(values))
            else:
                y_values.append(sum(values) / len(values))

        return x_values, y_values

    def get_grouped_series(
        self,
        x_param: str,
        metric: str,
        group_param: str,
        aggregate: str = "mean"
    ) -> Dict[Any, Tuple[List[Any], List[float]]]:
        """
        Get series grouped by a secondary parameter.

        Args:
            x_param: Parameter for X-axis
            metric: Metric for Y-axis
            group_param: Parameter to group by (different curves)
            aggregate: Aggregation method

        Returns:
            Dict mapping group values to (x, y) tuples
        """
        if not self.raw_data:
            return {}

        # Group by group_param first, then by x_param
        groups: Dict[Any, Dict[Any, List[float]]] = {}

        for entry in self.raw_data:
            if all(k in entry for k in [x_param, metric, group_param]):
                group_key = entry[group_param]
                x_key = entry[x_param]
                value = entry[metric]

                if group_key not in groups:
                    groups[group_key] = {}
                if x_key not in groups[group_key]:
                    groups[group_key][x_key] = []
                groups[group_key][x_key].append(value)

        # Convert to series
        result = {}
        for group_key, x_data in groups.items():
            x_values = []
            y_values = []

            for x_key in sorted(x_data.keys()):
                values = x_data[x_key]
                x_values.append(x_key)

                if aggregate == "mean":
                    y_values.append(sum(values) / len(values))
                elif aggregate == "min":
                    y_values.append(min(values))
                elif aggregate == "max":
                    y_values.append(max(values))
                else:
                    y_values.append(sum(values) / len(values))

            result[group_key] = (x_values, y_values)

        return result

    def get_all_values(self, key: str) -> List[Any]:
        """Get all unique values for a parameter or metric."""
        values = set()
        for entry in self.raw_data:
            if key in entry:
                values.add(entry[key])
        return sorted(values)

    def filter(self, **conditions) -> "SweepResults":
        """
        Filter results by conditions.

        Args:
            **conditions: key=value pairs to filter by

        Returns:
            New SweepResults with filtered data
        """
        filtered_data = []
        for entry in self.raw_data:
            match = all(
                entry.get(k) == v for k, v in conditions.items()
            )
            if match:
                filtered_data.append(entry.copy())

        result = SweepResults(
            raw_data=filtered_data,
            parameters=self.parameters.copy(),
            metrics=self.metrics.copy(),
            system_type=self.system_type,
        )
        return result

    def summary(self) -> Dict[str, Dict[str, float]]:
        """
        Get summary statistics for all metrics.

        Returns:
            Dict mapping metric names to stats (min, max, mean, std)
        """
        summary = {}

        for metric in self.metrics:
            values = [e[metric] for e in self.raw_data if metric in e]
            if values:
                mean = sum(values) / len(values)
                variance = sum((v - mean) ** 2 for v in values) / len(values)
                std = variance ** 0.5

                summary[metric] = {
                    "min": min(values),
                    "max": max(values),
                    "mean": mean,
                    "std": std,
                    "count": len(values),
                }

        return summary

    def save(self, path: Path) -> None:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = {
            "raw_data": self.raw_data,
            "parameters": self.parameters,
            "metrics": self.metrics,
            "system_type": self.system_type,
        }

        with open(path, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

    @classmethod
    def load(cls, path: Path) -> "SweepResults":
        """Load results from JSON file."""
        with open(path, 'r', encoding='utf-8') as f:
            data = json.load(f)

        return cls(
            raw_data=data["raw_data"],
            parameters=data["parameters"],
            metrics=data["metrics"],
            system_type=data.get("system_type", "host_to_noc"),
        )

    def __len__(self) -> int:
        return len(self.raw_data)

    def __repr__(self) -> str:
        return (
            f"SweepResults({len(self.raw_data)} entries, "
            f"params={self.parameters}, metrics={self.metrics})"
        )
