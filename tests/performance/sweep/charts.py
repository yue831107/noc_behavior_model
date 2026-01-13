"""
Parameter Sweep Visualization Charts.

Provides visualization for comparing performance across parameter values.
"""

from typing import Dict, List, Optional, Tuple, Any
from pathlib import Path

import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

from .results import SweepResults


# Color palette for multiple curves
COLORS = list(mcolors.TABLEAU_COLORS.values())


class SweepCharts:
    """
    Visualization charts for parameter sweep results.

    Provides various chart types for analyzing how parameters
    affect performance metrics.
    """

    @staticmethod
    def plot_metric_vs_param(
        results: SweepResults,
        param: str,
        metric: str,
        ax: Optional[plt.Axes] = None,
        title: Optional[str] = None,
        xlabel: Optional[str] = None,
        ylabel: Optional[str] = None,
        marker: str = 'o',
        color: Optional[str] = None,
        **kwargs
    ) -> plt.Figure:
        """
        Plot single metric vs single parameter.

        Args:
            results: Sweep results
            param: Parameter name for X-axis
            metric: Metric name for Y-axis
            ax: Existing axes (creates new figure if None)
            title: Chart title
            xlabel: X-axis label
            ylabel: Y-axis label
            marker: Point marker style
            color: Line color
            **kwargs: Additional plot arguments

        Returns:
            Matplotlib Figure
        """
        if ax is None:
            fig, ax = plt.subplots(figsize=(10, 6))
        else:
            fig = ax.get_figure()

        x, y = results.get_series(param, metric)

        if not x:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                    transform=ax.transAxes)
            return fig

        ax.plot(x, y, marker=marker, color=color or COLORS[0],
                linewidth=2, markersize=8, **kwargs)

        ax.set_xlabel(xlabel or param, fontsize=12)
        ax.set_ylabel(ylabel or metric, fontsize=12)
        ax.set_title(title or f"{metric} vs {param}", fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)

        plt.tight_layout()
        return fig

    @staticmethod
    def plot_multi_metric(
        results: SweepResults,
        param: str,
        metrics: List[str],
        normalize: bool = False,
        figsize: Tuple[int, int] = (12, 8),
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot multiple metrics vs single parameter (stacked subplots).

        Args:
            results: Sweep results
            param: Parameter for X-axis
            metrics: List of metrics to plot
            normalize: Normalize each metric to [0, 1]
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Matplotlib Figure
        """
        n_metrics = len(metrics)
        fig, axes = plt.subplots(n_metrics, 1, figsize=figsize, sharex=True)

        if n_metrics == 1:
            axes = [axes]

        for i, (ax, metric) in enumerate(zip(axes, metrics)):
            x, y = results.get_series(param, metric)

            if not x:
                ax.text(0.5, 0.5, "No data", ha='center', va='center',
                        transform=ax.transAxes)
                continue

            if normalize and max(y) > 0:
                y = [v / max(y) for v in y]

            ax.plot(x, y, marker='o', color=COLORS[i % len(COLORS)],
                    linewidth=2, markersize=6, label=metric)
            ax.set_ylabel(metric, fontsize=11)
            ax.grid(True, alpha=0.3)
            ax.legend(loc='upper right')

        axes[-1].set_xlabel(param, fontsize=12)
        fig.suptitle(f"Performance Metrics vs {param}", fontsize=14, fontweight='bold')
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_multi_param_comparison(
        results: SweepResults,
        x_param: str,
        metric: str,
        group_param: str,
        figsize: Tuple[int, int] = (12, 7),
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot metric with multiple curves for different group parameter values.

        Example: throughput vs transfer_size, with different curves for
        each buffer_depth value.

        Args:
            results: Sweep results
            x_param: Parameter for X-axis
            metric: Metric for Y-axis
            group_param: Parameter to group by (different curves)
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        grouped = results.get_grouped_series(x_param, metric, group_param)

        if not grouped:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                    transform=ax.transAxes)
            return fig

        for i, (group_val, (x, y)) in enumerate(sorted(grouped.items())):
            color = COLORS[i % len(COLORS)]
            ax.plot(x, y, marker='o', color=color, linewidth=2, markersize=6,
                    label=f"{group_param}={group_val}")

        ax.set_xlabel(x_param, fontsize=12)
        ax.set_ylabel(metric, fontsize=12)
        ax.set_title(f"{metric} vs {x_param} (grouped by {group_param})",
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.legend(loc='best', fontsize=10)

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_heatmap_2d(
        results: SweepResults,
        x_param: str,
        y_param: str,
        metric: str,
        figsize: Tuple[int, int] = (10, 8),
        cmap: str = 'viridis',
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot 2D heatmap for two parameters vs one metric.

        Args:
            results: Sweep results
            x_param: Parameter for X-axis
            y_param: Parameter for Y-axis
            metric: Metric for color values
            figsize: Figure size
            cmap: Colormap name
            save_path: Path to save figure

        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        # Get unique values for each parameter
        x_values = sorted(results.get_all_values(x_param))
        y_values = sorted(results.get_all_values(y_param))

        if not x_values or not y_values:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                    transform=ax.transAxes)
            return fig

        # Create 2D array
        data = np.zeros((len(y_values), len(x_values)))
        data[:] = np.nan

        for entry in results.raw_data:
            if all(k in entry for k in [x_param, y_param, metric]):
                xi = x_values.index(entry[x_param])
                yi = y_values.index(entry[y_param])
                data[yi, xi] = entry[metric]

        # Plot heatmap
        im = ax.imshow(data, cmap=cmap, aspect='auto', origin='lower')

        # Set ticks
        ax.set_xticks(range(len(x_values)))
        ax.set_xticklabels([str(v) for v in x_values])
        ax.set_yticks(range(len(y_values)))
        ax.set_yticklabels([str(v) for v in y_values])

        ax.set_xlabel(x_param, fontsize=12)
        ax.set_ylabel(y_param, fontsize=12)
        ax.set_title(f"{metric} Heatmap", fontsize=14, fontweight='bold')

        # Colorbar
        cbar = plt.colorbar(im, ax=ax)
        cbar.set_label(metric, fontsize=11)

        # Add value annotations
        for i in range(len(y_values)):
            for j in range(len(x_values)):
                if not np.isnan(data[i, j]):
                    text = ax.text(j, i, f"{data[i, j]:.1f}",
                                   ha="center", va="center", color="white",
                                   fontsize=9, fontweight='bold')

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_sweep_dashboard(
        results: SweepResults,
        param: str,
        metrics: Optional[List[str]] = None,
        figsize: Tuple[int, int] = (14, 10),
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Create comprehensive dashboard with multiple charts.

        Layout:
        ┌────────────────┬────────────────┐
        │  Throughput    │   Latency      │
        ├────────────────┼────────────────┤
        │  Total Cycles  │   Summary      │
        └────────────────┴────────────────┘

        Args:
            results: Sweep results
            param: Main parameter being swept
            metrics: Metrics to show (defaults to common ones)
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Matplotlib Figure
        """
        if metrics is None:
            metrics = ['throughput', 'avg_latency', 'total_cycles', 'buffer_utilization']

        # Filter to available metrics
        available_metrics = [m for m in metrics if m in results.metrics]
        if not available_metrics:
            available_metrics = results.metrics[:4]

        n_plots = min(4, len(available_metrics))
        fig, axes = plt.subplots(2, 2, figsize=figsize)
        axes = axes.flatten()

        for i, metric in enumerate(available_metrics[:4]):
            ax = axes[i]
            x, y = results.get_series(param, metric)

            if x and y:
                ax.plot(x, y, marker='o', color=COLORS[i], linewidth=2, markersize=8)
                ax.fill_between(x, y, alpha=0.2, color=COLORS[i])

                # Add min/max annotations
                if len(y) > 1:
                    max_idx = y.index(max(y))
                    min_idx = y.index(min(y))
                    ax.annotate(f"max: {max(y):.2f}", (x[max_idx], y[max_idx]),
                                textcoords="offset points", xytext=(0, 10),
                                ha='center', fontsize=9)

            ax.set_xlabel(param, fontsize=11)
            ax.set_ylabel(metric, fontsize=11)
            ax.set_title(f"{metric.replace('_', ' ').title()}", fontsize=12, fontweight='bold')
            ax.grid(True, alpha=0.3)

        # Fill remaining subplots with summary if less than 4 metrics
        for i in range(n_plots, 4):
            ax = axes[i]
            summary = results.summary()
            if summary:
                text = "Summary Statistics\n" + "=" * 30 + "\n"
                for metric, stats in list(summary.items())[:5]:
                    text += f"\n{metric}:\n"
                    text += f"  min: {stats['min']:.2f}\n"
                    text += f"  max: {stats['max']:.2f}\n"
                    text += f"  mean: {stats['mean']:.2f}\n"
                ax.text(0.1, 0.9, text, transform=ax.transAxes, fontsize=10,
                        verticalalignment='top', fontfamily='monospace')
                ax.set_title("Summary", fontsize=12, fontweight='bold')
            ax.axis('off')

        fig.suptitle(f"Parameter Sweep Dashboard: {param}",
                     fontsize=16, fontweight='bold', y=1.02)
        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig

    @staticmethod
    def plot_bar_comparison(
        results: SweepResults,
        param: str,
        metrics: List[str],
        figsize: Tuple[int, int] = (12, 6),
        save_path: Optional[Path] = None,
    ) -> plt.Figure:
        """
        Plot bar chart comparing multiple metrics across parameter values.

        Args:
            results: Sweep results
            param: Parameter for X-axis categories
            metrics: Metrics to compare
            figsize: Figure size
            save_path: Path to save figure

        Returns:
            Matplotlib Figure
        """
        fig, ax = plt.subplots(figsize=figsize)

        param_values = results.get_all_values(param)
        if not param_values:
            ax.text(0.5, 0.5, "No data available", ha='center', va='center',
                    transform=ax.transAxes)
            return fig

        n_params = len(param_values)
        n_metrics = len(metrics)
        bar_width = 0.8 / n_metrics
        x = np.arange(n_params)

        for i, metric in enumerate(metrics):
            _, y = results.get_series(param, metric)
            if y:
                # Normalize for comparison
                max_y = max(y) if max(y) > 0 else 1
                y_norm = [v / max_y for v in y]
                offset = (i - n_metrics / 2 + 0.5) * bar_width
                ax.bar(x + offset, y_norm, bar_width, label=metric,
                       color=COLORS[i % len(COLORS)], alpha=0.8)

        ax.set_xlabel(param, fontsize=12)
        ax.set_ylabel("Normalized Value", fontsize=12)
        ax.set_title(f"Metrics Comparison by {param}", fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([str(v) for v in param_values])
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3, axis='y')

        plt.tight_layout()

        if save_path:
            Path(save_path).parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(save_path, dpi=150, bbox_inches='tight')

        return fig


def quick_plot(
    results: SweepResults,
    param: str,
    metric: str = "throughput",
    save_path: Optional[Path] = None,
) -> plt.Figure:
    """
    Quick convenience function for simple plots.

    Args:
        results: Sweep results
        param: Parameter for X-axis
        metric: Metric for Y-axis
        save_path: Optional save path

    Returns:
        Matplotlib Figure
    """
    fig = SweepCharts.plot_metric_vs_param(results, param, metric)

    if save_path:
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')

    return fig
