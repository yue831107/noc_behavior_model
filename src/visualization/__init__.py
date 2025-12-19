"""
NoC Visualization Module.

Provides metrics collection, data persistence, and visualization
for NoC simulation analysis.

Phase 1: Static Charts
- MetricsCollector: Collect simulation snapshots
- MetricsStore: Save/load metrics to/from files
- plot_buffer_heatmap: Buffer occupancy heatmap
- plot_latency_histogram: Latency distribution histogram
- plot_throughput_curve: Throughput over time

Phase 2: Additional Charts (charts.py)
Phase 3: Animation (animation.py)
Phase 4: Dashboard (dashboard/)
"""

from .metrics_collector import (
    SimulationSnapshot,
    MetricsCollector,
)
from .metrics_store import (
    MetricsStore,
)
from .heatmap import (
    BufferHeatmapConfig,
    plot_buffer_heatmap,
)
from .histogram import (
    LatencyHistogramConfig,
    plot_latency_histogram,
)
from .throughput import (
    ThroughputConfig,
    plot_throughput_curve,
)

__all__ = [
    # Metrics Collection
    "SimulationSnapshot",
    "MetricsCollector",
    # Storage
    "MetricsStore",
    # Heatmap
    "BufferHeatmapConfig",
    "plot_buffer_heatmap",
    # Histogram
    "LatencyHistogramConfig",
    "plot_latency_histogram",
    # Throughput
    "ThroughputConfig",
    "plot_throughput_curve",
]
