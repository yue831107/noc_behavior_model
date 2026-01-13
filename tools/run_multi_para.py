#!/usr/bin/env python3
"""
Multi-Parameter Simulation Runner.

Runs simulations with multiple parameter combinations and generates comparison charts.
Integrates with the existing run.py workflow.

Usage:
    python tools/run_multi_para.py --config examples/Host_to_NoC/config/generated.yaml --bin payload.bin
"""

import argparse
import json
import os
import subprocess
import sys
from itertools import product
from pathlib import Path
from typing import Dict, List, Any

# Fix for matplotlib home directory issue on Windows
if 'HOME' not in os.environ:
    os.environ['HOME'] = os.environ.get('USERPROFILE', 'C:\\Users\\Default')
if 'MPLCONFIGDIR' not in os.environ:
    os.environ['MPLCONFIGDIR'] = os.environ.get('TEMP', 'C:\\Temp')

import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Add project root to path
project_root = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(project_root))

from tests.performance.sweep import SweepResults, SweepCharts


def load_config(config_path: Path) -> dict:
    """Load transfer config YAML."""
    with open(config_path, 'r', encoding='utf-8') as f:
        return yaml.safe_load(f)


def generate_combinations(sweep_params: Dict[str, List]) -> List[Dict[str, Any]]:
    """Generate all parameter combinations from sweep config."""
    if not sweep_params:
        return [{}]

    keys = list(sweep_params.keys())
    values = [sweep_params[k] for k in keys]

    combinations = []
    for combo in product(*values):
        combinations.append(dict(zip(keys, combo)))

    return combinations


def run_simulation(
    config_path: Path,
    bin_file: Path,
    params: Dict[str, Any],
    output_json: Path,
    verbose: bool = False,
) -> Dict[str, float]:
    """
    Run a single simulation with given parameters.

    Calls examples/Host_to_NoC/run.py with appropriate arguments.
    """
    cmd = [
        sys.executable,
        str(project_root / 'examples' / 'Host_to_NoC' / 'run.py'),
        'multi_transfer',
        '--config', str(config_path),
        '--bin', str(bin_file),
        '--json-output', str(output_json),
    ]

    # Add system parameters
    if 'buffer_depth' in params:
        cmd.extend(['--buffer-depth', str(params['buffer_depth'])])
    if 'mesh_cols' in params:
        cmd.extend(['--mesh-cols', str(params['mesh_cols'])])
    if 'mesh_rows' in params:
        cmd.extend(['--mesh-rows', str(params['mesh_rows'])])

    if not verbose:
        cmd.append('-q')

    # Run simulation
    result = subprocess.run(cmd, capture_output=True, text=True)

    # Read metrics from JSON (even if validation failed with exit code 1)
    if output_json.exists():
        with open(output_json, 'r') as f:
            return json.load(f)

    # Only warn if no metrics were produced
    if result.returncode != 0:
        print(f"  Warning: Simulation failed (no metrics)")
        if verbose:
            print(result.stderr)
    return {}


def generate_charts(
    results: SweepResults,
    sweep_params: Dict[str, List],
    output_dir: Path,
) -> List[str]:
    """Generate comparison charts."""
    generated = []
    param_names = list(sweep_params.keys())

    if not param_names:
        return generated

    # Determine primary sweep parameter (the one with most values)
    primary_param = max(param_names, key=lambda k: len(sweep_params[k]))

    # Metrics to plot (total_cycles is most important for seeing buffer effects)
    metrics = ['total_cycles', 'throughput', 'avg_latency']
    available_metrics = [m for m in metrics if m in results.metrics]

    # If only one parameter, simple line charts
    if len(param_names) == 1:
        for metric in available_metrics:
            filename = f"{metric}_vs_{primary_param}.png"
            fig = SweepCharts.plot_metric_vs_param(
                results, primary_param, metric
            )
            fig.savefig(output_dir / filename, dpi=150, bbox_inches='tight')
            plt.close(fig)
            generated.append(filename)
    else:
        # Multiple parameters - use grouped comparison
        group_param = [p for p in param_names if p != primary_param][0]

        for metric in available_metrics:
            filename = f"{metric}_by_{group_param}.png"
            fig = SweepCharts.plot_multi_param_comparison(
                results,
                x_param=primary_param,
                metric=metric,
                group_param=group_param,
                save_path=output_dir / filename,
            )
            plt.close(fig)
            generated.append(filename)

        # 2D heatmap for throughput
        if 'throughput' in available_metrics and len(param_names) >= 2:
            filename = "throughput_heatmap.png"
            fig = SweepCharts.plot_heatmap_2d(
                results,
                x_param=primary_param,
                y_param=group_param,
                metric='throughput',
                save_path=output_dir / filename,
            )
            plt.close(fig)
            generated.append(filename)

    return generated


def main():
    parser = argparse.ArgumentParser(
        description='Run multi-parameter simulation sweep'
    )
    parser.add_argument(
        '--config', '-c',
        type=Path,
        required=True,
        help='Transfer config YAML file (with sweep section)'
    )
    parser.add_argument(
        '--bin', '-b',
        type=Path,
        required=True,
        help='BIN file for source data'
    )
    parser.add_argument(
        '-o', '--output',
        type=Path,
        default=Path('output/multi_para'),
        help='Output directory (default: output/multi_para)'
    )
    parser.add_argument(
        '-q', '--quiet',
        action='store_true',
        help='Quiet mode'
    )

    args = parser.parse_args()

    # Load config
    if not args.config.exists():
        print(f"Error: Config not found: {args.config}")
        return 1

    if not args.bin.exists():
        print(f"Error: BIN file not found: {args.bin}")
        return 1

    config = load_config(args.config)
    sweep_params = config.get('sweep', {})

    if not sweep_params:
        print("Error: No sweep parameters in config")
        print("Add 'sweep:' section with parameters like:")
        print("  sweep:")
        print("    buffer_depth: [2, 4, 8, 16]")
        return 1

    # Generate combinations
    combinations = generate_combinations(sweep_params)
    total = len(combinations)

    # Create output directory
    args.output.mkdir(parents=True, exist_ok=True)
    temp_dir = args.output / 'temp'
    temp_dir.mkdir(exist_ok=True)

    # Print header
    verbose = not args.quiet
    if verbose:
        print("=" * 60)
        print("Multi-Parameter Simulation")
        print("=" * 60)
        print(f"Config:       {args.config}")
        print(f"Parameters:   {list(sweep_params.keys())}")
        print(f"Combinations: {total}")
        print(f"Output:       {args.output}")
        print("=" * 60)
        print()

    # Run simulations
    results = SweepResults()

    for i, params in enumerate(combinations):
        param_str = ", ".join(f"{k}={v}" for k, v in params.items())
        if verbose:
            print(f"[{i+1}/{total}] Running: {param_str}")

        # Run simulation
        json_output = temp_dir / f"metrics_{i}.json"
        metrics = run_simulation(
            args.config,
            args.bin,
            params,
            json_output,
            verbose=False,
        )

        if metrics:
            results.add_result(params, metrics)
        else:
            if verbose:
                print(f"  Warning: No metrics collected")

    # Clean up temp files
    for f in temp_dir.glob('*.json'):
        f.unlink()
    temp_dir.rmdir()

    if verbose:
        print()
        print("=" * 60)
        print(f"Simulations complete: {len(results)} results")
        print("=" * 60)

    if len(results) == 0:
        print("Error: No results collected")
        return 1

    # Save results
    results.save(args.output / 'results.json')
    if verbose:
        print(f"\nResults saved: {args.output / 'results.json'}")

    # Generate charts
    if verbose:
        print("\nGenerating charts...")

    generated = generate_charts(results, sweep_params, args.output)

    if verbose:
        for filename in generated:
            print(f"  {filename}")

    # Print summary
    if verbose:
        print()
        print("=" * 60)
        print("Summary")
        print("=" * 60)

        summary = results.summary()
        for metric, stats in list(summary.items())[:4]:
            print(f"  {metric}: min={stats['min']:.2f}, max={stats['max']:.2f}, mean={stats['mean']:.2f}")

        print(f"\nOutput: {args.output.absolute()}")

    return 0


if __name__ == '__main__':
    sys.exit(main())
