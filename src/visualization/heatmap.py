"""
Buffer Occupancy Heatmap Visualization.

Displays mesh router buffer occupancy as a color-coded heatmap.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Tuple, TYPE_CHECKING

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.figure import Figure

if TYPE_CHECKING:
    from .metrics_collector import MetricsCollector


@dataclass
class BufferHeatmapConfig:
    """Configuration for buffer heatmap visualization."""
    
    title: str = "Buffer Occupancy Heatmap"
    cmap: str = "YlOrRd"  # Yellow -> Orange -> Red
    show_values: bool = True
    show_colorbar: bool = True
    figsize: Tuple[int, int] = (10, 8)
    # Value format string for annotations
    value_format: str = "{:.0f}"
    # Edge column marking
    mark_edge_column: bool = True
    edge_column: int = 0
    # Axis labels
    xlabel: str = "Column"
    ylabel: str = "Row"


def plot_buffer_heatmap(
    collector: "MetricsCollector",
    config: Optional[BufferHeatmapConfig] = None,
    save_path: Optional[str] = None,
    snapshot_index: int = -1,
) -> Figure:
    """
    Plot buffer occupancy heatmap.
    
    Args:
        collector: MetricsCollector with captured data.
        config: Visualization configuration.
        save_path: Optional path to save the figure.
        snapshot_index: Which snapshot to visualize (-1 = latest).
    
    Returns:
        Matplotlib Figure object.
    """
    if config is None:
        config = BufferHeatmapConfig()
    
    # Get buffer occupancy matrix
    matrix = collector.get_buffer_occupancy_matrix(snapshot_index)
    
    # Get cycle number for title
    if collector.snapshots:
        cycle = collector.snapshots[snapshot_index].cycle
        title = f"{config.title} (Cycle {cycle})"
    else:
        title = config.title
    
    # Create figure
    fig, ax = plt.subplots(figsize=config.figsize)
    
    # Determine vmax for consistent color scaling
    # Use fixed max for consistent appearance across visualizations
    vmax = max(8, int(np.ceil(matrix.max()))) if matrix.max() > 0 else 8
    
    # Plot heatmap with fixed scale for smooth color gradients
    im = ax.imshow(
        matrix,
        cmap=config.cmap,
        aspect='auto',
        origin='lower',  # Row 0 at bottom
        vmin=0,
        vmax=vmax,
        interpolation='nearest',  # Sharp edges for grid cells
    )
    
    # Add colorbar with more ticks for granularity
    if config.show_colorbar:
        cbar = fig.colorbar(im, ax=ax, label='Buffer Occupancy')
        cbar.set_ticks(np.arange(0, vmax + 1, max(1, vmax // 8)))
    
    # Add value annotations
    if config.show_values:
        for y in range(matrix.shape[0]):
            for x in range(matrix.shape[1]):
                value = matrix[y, x]
                # Skip edge column if marked
                if config.mark_edge_column and x == config.edge_column:
                    text = "--"
                    color = 'gray'
                else:
                    text = config.value_format.format(value)
                    # Choose text color based on background
                    color = 'white' if value > matrix.max() * 0.6 else 'black'
                
                ax.text(
                    x, y, text,
                    ha='center', va='center',
                    color=color, fontsize=10,
                )
    
    # Set labels and title
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xlabel(config.xlabel)
    ax.set_ylabel(config.ylabel)
    
    # Set tick labels
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    
    # Mark edge column in label
    if config.mark_edge_column:
        xlabels = [
            f"Edge" if i == config.edge_column else str(i)
            for i in range(matrix.shape[1])
        ]
        ax.set_xticklabels(xlabels)
    
    plt.tight_layout()
    
    # Save if path provided
    if save_path:
        from pathlib import Path
        Path(save_path).parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(save_path, dpi=150, bbox_inches='tight')
    
    return fig


def plot_buffer_occupancy_animation_frame(
    ax,
    matrix: np.ndarray,
    config: BufferHeatmapConfig,
    cycle: int,
) -> None:
    """
    Plot a single frame for animation (used by animation.py).
    
    Args:
        ax: Matplotlib axes to plot on.
        matrix: Buffer occupancy matrix.
        config: Visualization configuration.
        cycle: Current cycle number.
    """
    ax.clear()
    
    im = ax.imshow(
        matrix,
        cmap=config.cmap,
        aspect='auto',
        origin='lower',
        vmin=0,
        vmax=16,  # Assume max buffer depth
    )
    
    ax.set_title(f"{config.title} (Cycle {cycle})")
    ax.set_xlabel(config.xlabel)
    ax.set_ylabel(config.ylabel)
    
    # Set tick labels
    ax.set_xticks(range(matrix.shape[1]))
    ax.set_yticks(range(matrix.shape[0]))
    
    return im
