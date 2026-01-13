"""
Parameter Sweep Framework for NoC Performance Analysis.

Provides tools for running parameter sweeps and visualizing results.

Usage:
    from tests.performance.sweep import (
        SweepConfig,
        SweepRunner,
        SweepResults,
        SweepCharts,
    )

    # Define sweep
    config = SweepConfig(
        parameters={"transfer_size": [64, 128, 256, 512, 1024]},
    )

    # Run sweep
    runner = SweepRunner()
    results = runner.run_sweep(config)

    # Visualize
    SweepCharts.plot_sweep_dashboard(results, "transfer_size", save_path="sweep.png")
"""

from .config import (
    SweepConfig,
    SweepMode,
    create_transfer_size_sweep,
    create_delay_sweep,
    create_transfer_count_sweep,
    create_pattern_sweep,
)

from .results import SweepResults

from .runner import (
    SweepRunner,
    run_quick_sweep,
)

from .charts import (
    SweepCharts,
    quick_plot,
)


__all__ = [
    # Config
    'SweepConfig',
    'SweepMode',
    'create_transfer_size_sweep',
    'create_delay_sweep',
    'create_transfer_count_sweep',
    'create_pattern_sweep',
    # Results
    'SweepResults',
    # Runner
    'SweepRunner',
    'run_quick_sweep',
    # Charts
    'SweepCharts',
    'quick_plot',
]
