"""
Latency Distribution Histogram Visualization.

Displays transaction latency distribution with optional CDF curve.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Tuple, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from .metrics_collector import MetricsCollector


@dataclass
class LatencyHistogramConfig:
    """Configuration for latency histogram visualization."""
    
    title: str = "Latency Distribution"
    bins: int = 50
    # CDF overlay
    show_cdf: bool = True
    cdf_color: str = 'red'
    # Percentile lines
    show_percentiles: bool = True
    percentiles: List[int] = field(default_factory=lambda: [50, 90, 99])
    percentile_colors: List[str] = field(
        default_factory=lambda: ['green', 'orange', 'red']
    )
    # Styling
    figsize: Tuple[int, int] = (10, 6)
    hist_color: str = 'steelblue'
    hist_alpha: float = 0.7
    xlabel: str = "Latency (cycles)"
    ylabel: str = "Count"


def plot_latency_histogram(
    collector: "MetricsCollector",
    config: Optional[LatencyHistogramConfig] = None,
    save_path: Optional[str] = None,
    latencies: Optional[List[int]] = None,
) -> Figure:
    """
    Plot latency distribution histogram with optional CDF.
    
    Args:
        collector: MetricsCollector with captured data.
        config: Visualization configuration.
        save_path: Optional path to save the figure.
        latencies: Optional override for latency data.
    
    Returns:
        Matplotlib Figure object.
    """
    if config is None:
        config = LatencyHistogramConfig()
    
    # Get latency data
    if latencies is None:
        latencies = collector.get_all_latencies()
    
    if not latencies:
        # No data - create empty figure with message
        fig, ax = plt.subplots(figsize=config.figsize)
        ax.text(
            0.5, 0.5, "No latency data available",
            ha='center', va='center', fontsize=12,
            transform=ax.transAxes
        )
        ax.set_title(config.title)
        return fig
    
    latencies = np.array(latencies)
    
    # Create figure with two y-axes if showing CDF
    fig, ax1 = plt.subplots(figsize=config.figsize)
    
    # Plot histogram
    n, bins, patches = ax1.hist(
        latencies,
        bins=config.bins,
        color=config.hist_color,
        alpha=config.hist_alpha,
        edgecolor='black',
        label='Histogram',
    )
    
    ax1.set_xlabel(config.xlabel)
    ax1.set_ylabel(config.ylabel, color=config.hist_color)
    ax1.tick_params(axis='y', labelcolor=config.hist_color)
    
    # Plot CDF on secondary y-axis
    if config.show_cdf:
        ax2 = ax1.twinx()
        
        # Calculate CDF
        sorted_latencies = np.sort(latencies)
        cdf = np.arange(1, len(sorted_latencies) + 1) / len(sorted_latencies)
        
        ax2.plot(
            sorted_latencies, cdf,
            color=config.cdf_color,
            linewidth=2,
            label='CDF',
        )
        ax2.set_ylabel('CDF', color=config.cdf_color)
        ax2.tick_params(axis='y', labelcolor=config.cdf_color)
        ax2.set_ylim(0, 1.05)
    
    # Add percentile lines
    if config.show_percentiles and config.percentiles:
        for i, p in enumerate(config.percentiles):
            pval = np.percentile(latencies, p)
            color = config.percentile_colors[i % len(config.percentile_colors)]
            ax1.axvline(
                x=pval,
                color=color,
                linestyle='--',
                linewidth=1.5,
                label=f'P{p}: {pval:.0f}',
            )
    
    # Title and legend
    ax1.set_title(config.title, fontsize=14, fontweight='bold')
    
    # Combine legends from both axes
    lines1, labels1 = ax1.get_legend_handles_labels()
    if config.show_cdf:
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax1.legend(lines1 + lines2, labels1 + labels2, loc='upper right')
    else:
        ax1.legend(loc='upper right')
    
    # Add statistics text box
    stats_text = (
        f"Count: {len(latencies)}\n"
        f"Mean: {np.mean(latencies):.1f}\n"
        f"Std: {np.std(latencies):.1f}\n"
        f"Min: {np.min(latencies):.0f}\n"
        f"Max: {np.max(latencies):.0f}"
    )
    ax1.text(
        0.02, 0.98, stats_text,
        transform=ax1.transAxes,
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


def calculate_latency_stats(latencies: List[int]) -> dict:
    """
    Calculate latency statistics.
    
    Args:
        latencies: List of latency values.
    
    Returns:
        Dictionary with statistics.
    """
    if not latencies:
        return {
            'count': 0,
            'mean': 0,
            'std': 0,
            'min': 0,
            'max': 0,
            'p50': 0,
            'p90': 0,
            'p99': 0,
        }
    
    arr = np.array(latencies)
    return {
        'count': len(arr),
        'mean': float(np.mean(arr)),
        'std': float(np.std(arr)),
        'min': int(np.min(arr)),
        'max': int(np.max(arr)),
        'p50': float(np.percentile(arr, 50)),
        'p90': float(np.percentile(arr, 90)),
        'p99': float(np.percentile(arr, 99)),
    }
