"""
Throughput Curve Visualization.

Displays throughput (bytes/cycle) over simulation time.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple, List, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from .metrics_collector import MetricsCollector


@dataclass
class ThroughputConfig:
    """Configuration for throughput curve visualization."""
    
    title: str = "Throughput Over Time"
    window_size: int = 10  # Sliding window for averaging
    figsize: Tuple[int, int] = (12, 5)
    line_color: str = 'steelblue'
    line_width: float = 1.5
    # Secondary metrics
    show_average_line: bool = True
    avg_line_color: str = 'red'
    avg_line_style: str = '--'
    # Axis labels
    xlabel: str = "Cycle"
    ylabel: str = "Throughput (bytes/cycle)"
    # Grid
    show_grid: bool = True
    grid_alpha: float = 0.3


def plot_throughput_curve(
    collector: "MetricsCollector",
    config: Optional[ThroughputConfig] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot throughput over time curve.
    
    Args:
        collector: MetricsCollector with captured data.
        config: Visualization configuration.
        save_path: Optional path to save the figure.
    
    Returns:
        Matplotlib Figure object.
    """
    if config is None:
        config = ThroughputConfig()
    
    # Get throughput data
    cycles, throughputs = collector.get_throughput_over_time(
        window_size=config.window_size
    )
    
    if not cycles:
        # No data - create empty figure with message
        fig, ax = plt.subplots(figsize=config.figsize)
        ax.text(
            0.5, 0.5, "No throughput data available",
            ha='center', va='center', fontsize=12,
            transform=ax.transAxes
        )
        ax.set_title(config.title)
        return fig
    
    # Create figure
    fig, ax = plt.subplots(figsize=config.figsize)
    
    # Plot throughput curve
    ax.plot(
        cycles, throughputs,
        color=config.line_color,
        linewidth=config.line_width,
        label='Throughput',
    )
    
    # Add average line
    if config.show_average_line and throughputs:
        avg = np.mean(throughputs)
        ax.axhline(
            y=avg,
            color=config.avg_line_color,
            linestyle=config.avg_line_style,
            linewidth=1.5,
            label=f'Average: {avg:.2f} B/cycle',
        )
    
    # Labels and title
    ax.set_xlabel(config.xlabel)
    ax.set_ylabel(config.ylabel)
    ax.set_title(config.title, fontsize=14, fontweight='bold')
    
    # Grid
    if config.show_grid:
        ax.grid(True, alpha=config.grid_alpha)
    
    # Legend
    ax.legend(loc='upper right')
    
    # Add statistics text
    if throughputs:
        stats_text = (
            f"Max: {max(throughputs):.2f} B/cycle\n"
            f"Min: {min(throughputs):.2f} B/cycle\n"
            f"Avg: {np.mean(throughputs):.2f} B/cycle"
        )
        ax.text(
            0.02, 0.98, stats_text,
            transform=ax.transAxes,
            verticalalignment='top',
            fontsize=9,
            bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5),
        )
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_multi_throughput_comparison(
    data: List[Tuple[str, List[int], List[float]]],
    config: Optional[ThroughputConfig] = None,
    save_path: Optional[str] = None,
) -> Figure:
    """
    Plot multiple throughput curves for comparison.
    
    Args:
        data: List of (label, cycles, throughputs) tuples.
        config: Visualization configuration.
        save_path: Optional path to save the figure.
    
    Returns:
        Matplotlib Figure object.
    """
    if config is None:
        config = ThroughputConfig()
    
    fig, ax = plt.subplots(figsize=config.figsize)
    
    colors = plt.cm.tab10.colors
    
    for i, (label, cycles, throughputs) in enumerate(data):
        color = colors[i % len(colors)]
        ax.plot(
            cycles, throughputs,
            color=color,
            linewidth=config.line_width,
            label=label,
        )
    
    ax.set_xlabel(config.xlabel)
    ax.set_ylabel(config.ylabel)
    ax.set_title(config.title, fontsize=14, fontweight='bold')
    
    if config.show_grid:
        ax.grid(True, alpha=config.grid_alpha)
    
    ax.legend(loc='upper right')
    
    plt.tight_layout()
    
    if save_path:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig
