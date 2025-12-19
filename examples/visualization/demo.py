#!/usr/bin/env python3
"""
NoC Visualization Demo Script.

Usage:
    py -3 examples/visualization/demo.py heatmap     # Buffer heatmap
    py -3 examples/visualization/demo.py latency     # Latency histogram
    py -3 examples/visualization/demo.py throughput  # Throughput curve
    py -3 examples/visualization/demo.py curves      # Additional curves
    py -3 examples/visualization/demo.py dashboard   # Combined dashboard
    py -3 examples/visualization/demo.py all         # All visualizations
    py -3 examples/visualization/demo.py save        # Save metrics to file

Options:
    --pattern PATTERN    Traffic pattern (default: neighbor)
    --cycles CYCLES      Simulation cycles (default: 500)
    --save-dir DIR       Output directory (default: output/charts)
    --show               Show plots interactively
"""

import sys
import argparse
from pathlib import Path
from datetime import datetime

# Add project root to path
project_root = Path(__file__).parent.parent.parent
sys.path.insert(0, str(project_root))


def run_simulation(pattern: str = "neighbor", cycles: int = 500, transfer_size: int = 256):
    """
    Run simulation and collect metrics.
    
    Returns:
        Tuple of (system, collector).
    """
    from src.core import NoCSystem
    from src.config import NoCTrafficConfig, TrafficPattern
    from src.visualization import MetricsCollector
    
    print(f"Creating NoCSystem (5x4 mesh)...")
    system = NoCSystem(
        mesh_cols=5,
        mesh_rows=4,
        buffer_depth=4,
        memory_size=0x10000,
    )
    
    # Configure traffic
    pattern_enum = TrafficPattern(pattern)
    config = NoCTrafficConfig(
        pattern=pattern_enum,
        transfer_size=transfer_size,
    )
    print(f"Configuring traffic pattern: {pattern}")
    system.configure_traffic(config)
    
    # Initialize memory
    system.initialize_node_memory(pattern="sequential")
    
    # Create collector with small interval for detailed data
    collector = MetricsCollector(system, capture_interval=1)
    
    # Run simulation until transfer complete (or max cycles)
    print(f"Running simulation (max {cycles} cycles)...")
    system.start_all_transfers()
    
    actual_cycles = 0
    for _ in range(cycles):
        system.process_cycle()
        collector.capture()
        actual_cycles += 1
        if system.all_transfers_complete:
            break
    
    print(f"Completed in {actual_cycles} cycles, collected {len(collector)} snapshots")
    
    return system, collector


def demo_heatmap(collector, save_dir: Path, show: bool = False):
    """Generate buffer heatmap using peak occupancy snapshot."""
    from src.visualization import plot_buffer_heatmap, BufferHeatmapConfig
    import matplotlib.pyplot as plt
    
    print("Generating buffer heatmap...")
    
    # Find snapshot with maximum buffer occupancy
    max_idx = -1
    max_occ = 0
    for i, snap in enumerate(collector.snapshots):
        total = sum(snap.buffer_occupancy.values())
        if total > max_occ:
            max_occ = total
            max_idx = i
    
    config = BufferHeatmapConfig(title=f"NoC Buffer Occupancy (Peak)")
    
    save_path = save_dir / "buffer_heatmap.png"
    fig = plot_buffer_heatmap(collector, config, save_path=str(save_path), snapshot_index=max_idx)
    print(f"  Saved: {save_path} (cycle {collector.snapshots[max_idx].cycle if max_idx >= 0 else 'N/A'}, total={max_occ})")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def demo_latency(collector, save_dir: Path, show: bool = False):
    """Generate latency histogram."""
    from src.visualization import plot_latency_histogram, LatencyHistogramConfig
    import matplotlib.pyplot as plt
    
    print("Generating latency histogram...")
    
    # Generate synthetic latency data since real latencies require
    # transaction completion callbacks
    import numpy as np
    np.random.seed(42)
    synthetic_latencies = np.random.exponential(scale=50, size=200).astype(int) + 10
    
    config = LatencyHistogramConfig(title="Transaction Latency Distribution")
    
    save_path = save_dir / "latency_histogram.png"
    fig = plot_latency_histogram(
        collector, config,
        save_path=str(save_path),
        latencies=list(synthetic_latencies),
    )
    print(f"  Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def demo_throughput(collector, save_dir: Path, show: bool = False):
    """Generate throughput curve."""
    from src.visualization import plot_throughput_curve, ThroughputConfig
    import matplotlib.pyplot as plt
    
    print("Generating throughput curve...")
    config = ThroughputConfig(title="NoC Throughput Over Time")
    
    save_path = save_dir / "throughput_curve.png"
    fig = plot_throughput_curve(collector, config, save_path=str(save_path))
    print(f"  Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def demo_curves(collector, save_dir: Path, show: bool = False):
    """Generate additional curve charts."""
    from src.visualization.charts import (
        plot_flit_count_curve,
        plot_buffer_utilization_curve,
        plot_transaction_progress,
        plot_router_comparison,
    )
    import matplotlib.pyplot as plt
    
    print("Generating additional curves...")
    
    # Flit count
    save_path = save_dir / "flit_count_curve.png"
    fig = plot_flit_count_curve(collector, save_path=str(save_path))
    print(f"  Saved: {save_path}")
    if not show:
        plt.close(fig)
    
    # Buffer utilization
    save_path = save_dir / "buffer_utilization_curve.png"
    fig = plot_buffer_utilization_curve(collector, save_path=str(save_path))
    print(f"  Saved: {save_path}")
    if not show:
        plt.close(fig)
    
    # Transaction progress
    save_path = save_dir / "transaction_progress.png"
    fig = plot_transaction_progress(collector, save_path=str(save_path))
    print(f"  Saved: {save_path}")
    if not show:
        plt.close(fig)
    
    # Router comparison
    save_path = save_dir / "router_comparison.png"
    fig = plot_router_comparison(collector, metric="flits", save_path=str(save_path))
    print(f"  Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def demo_dashboard(collector, save_dir: Path, show: bool = False):
    """Generate combined dashboard."""
    from src.visualization.charts import plot_combined_dashboard
    import matplotlib.pyplot as plt
    
    print("Generating combined dashboard...")
    
    save_path = save_dir / "dashboard.png"
    fig = plot_combined_dashboard(collector, save_path=str(save_path))
    print(f"  Saved: {save_path}")
    
    if show:
        plt.show()
    else:
        plt.close(fig)


def demo_save_metrics(collector, save_dir: Path):
    """Save metrics to files."""
    from src.visualization import MetricsStore
    
    print("Saving metrics...")
    
    # Create metrics store
    metrics_dir = save_dir.parent / "metrics"
    store = MetricsStore(base_dir=metrics_dir)
    
    # Save in all formats
    json_path = store.save_json(collector, "demo")
    print(f"  JSON: {json_path}")
    
    csv_path = store.save_csv(collector, "demo")
    print(f"  CSV:  {csv_path}")
    
    npz_path = store.save_npz(collector, "demo")
    print(f"  NPZ:  {npz_path}")


def main():
    parser = argparse.ArgumentParser(
        description="NoC Visualization Demo",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument(
        'command',
        choices=['heatmap', 'latency', 'throughput', 'curves', 'dashboard', 'all', 'save'],
        default='all',
        nargs='?',
        help='Visualization to generate (default: all)',
    )
    parser.add_argument(
        '--pattern', '-p',
        default='neighbor',
        choices=['neighbor', 'shuffle', 'bit_reverse', 'random', 'transpose'],
        help='Traffic pattern (default: neighbor)',
    )
    parser.add_argument(
        '--cycles', '-c',
        type=int,
        default=500,
        help='Simulation cycles (default: 500)',
    )
    parser.add_argument(
        '--save-dir', '-o',
        default='output/charts',
        help='Output directory (default: output/charts)',
    )
    parser.add_argument(
        '--show', '-s',
        action='store_true',
        help='Show plots interactively',
    )
    parser.add_argument(
        '--from-metrics', '-f',
        default=None,
        help='Load config from metrics JSON file (e.g., output/metrics/latest.json)',
    )
    
    args = parser.parse_args()
    
    # Load config from metrics file if provided
    if args.from_metrics:
        import json
        metrics_path = Path(args.from_metrics)
        if metrics_path.exists():
            with open(metrics_path) as f:
                metrics = json.load(f)
            args.pattern = metrics.get('pattern', args.pattern)
            # Use stored cycles for accurate visualization
            args.cycles = max(metrics.get('cycles', args.cycles) * 2, 100)
            # Load transfer_size from metrics
            args.transfer_size = metrics.get('transfer_size', 256)
            print(f"Loaded config from: {metrics_path}")
            print(f"  Pattern: {args.pattern}, Cycles: {args.cycles}, Transfer Size: {args.transfer_size}")
        else:
            print(f"Warning: Metrics file not found: {metrics_path}")
            args.transfer_size = 256
    else:
        args.transfer_size = 256
    
    # Setup output directory
    save_dir = Path(args.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    
    print("=" * 60)
    print(" NoC Visualization Demo")
    print("=" * 60)
    print(f"Timestamp: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"Pattern:   {args.pattern}")
    print(f"Cycles:    {args.cycles}")
    print(f"Output:    {save_dir}")
    print()
    
    # Check if this is a Host-to-NoC pattern (not a NoC-to-NoC traffic pattern)
    host_to_noc_patterns = ['multi_transfer', 'broadcast_write', 'broadcast_read', 'scatter_write', 'gather_read']
    
    if args.pattern in host_to_noc_patterns:
        # Host-to-NoC mode: use dummy simulation for chart generation
        print(f"Host-to-NoC mode detected: {args.pattern}")
        print("Using metrics from file for visualization (no re-simulation)...")
        
        # Create a minimal NoCSystem just for visualization structure
        from src.core import NoCSystem
        from src.visualization import MetricsCollector
        
        system = NoCSystem(
            mesh_cols=5,
            mesh_rows=4,
            buffer_depth=4,
            memory_size=0x10000,
        )
        collector = MetricsCollector(system, capture_interval=1)
        
        # Generate synthetic data from saved metrics
        if args.from_metrics:
            import json
            with open(args.from_metrics) as f:
                saved_metrics = json.load(f)
            # Create synthetic collector data for charts
            collector._generate_host_to_noc_data(saved_metrics)
    else:
        # NoC-to-NoC mode: run actual simulation
        system, collector = run_simulation(args.pattern, args.cycles, args.transfer_size)
    print()
    
    # Generate requested visualizations
    if args.command in ('heatmap', 'all'):
        demo_heatmap(collector, save_dir, args.show)
    
    if args.command in ('latency', 'all'):
        demo_latency(collector, save_dir, args.show)
    
    if args.command in ('throughput', 'all'):
        demo_throughput(collector, save_dir, args.show)
    
    if args.command in ('curves', 'all'):
        demo_curves(collector, save_dir, args.show)
    
    if args.command in ('dashboard', 'all'):
        demo_dashboard(collector, save_dir, args.show)
    
    if args.command in ('save', 'all'):
        demo_save_metrics(collector, save_dir)
    
    print()
    print("=" * 60)
    print(" Done!")
    print("=" * 60)
    
    return 0


if __name__ == '__main__':
    sys.exit(main())
