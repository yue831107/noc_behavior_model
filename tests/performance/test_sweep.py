"""
Tests for Parameter Sweep Framework.

Tests configuration, results handling, and visualization components.
"""

import pytest
from pathlib import Path
import tempfile

from tests.performance.sweep import (
    SweepConfig,
    SweepMode,
    SweepResults,
    SweepRunner,
    SweepCharts,
    create_transfer_size_sweep,
    run_quick_sweep,
)


class TestSweepConfig:
    """Tests for SweepConfig."""

    def test_single_parameter_config(self):
        """Test configuration with single parameter."""
        config = SweepConfig(
            parameters={"transfer_size": [64, 128, 256, 512]}
        )

        assert config.num_combinations == 4
        assert config.parameter_names == ["transfer_size"]

    def test_multi_parameter_grid_mode(self):
        """Test grid mode generates cartesian product."""
        config = SweepConfig(
            parameters={
                "transfer_size": [64, 256],
                "num_transfers": [5, 10],
            },
            mode=SweepMode.GRID,
        )

        # 2 × 2 = 4 combinations
        assert config.num_combinations == 4

        combos = list(config.generate_combinations())
        assert len(combos) == 4

        # Check all combinations present
        sizes = {c["transfer_size"] for c in combos}
        counts = {c["num_transfers"] for c in combos}
        assert sizes == {64, 256}
        assert counts == {5, 10}

    def test_linear_mode(self):
        """Test linear mode varies one parameter at a time."""
        config = SweepConfig(
            parameters={
                "transfer_size": [64, 256],
                "num_transfers": [5, 10, 20],
            },
            mode=SweepMode.LINEAR,
        )

        # 2 + 3 = 5 combinations
        assert config.num_combinations == 5

    def test_fixed_params_included(self):
        """Test fixed parameters are included in combinations."""
        config = SweepConfig(
            parameters={"transfer_size": [64, 128]},
            fixed_params={"num_transfers": 10, "pattern": "neighbor"},
        )

        combos = list(config.generate_combinations())
        for combo in combos:
            assert combo["num_transfers"] == 10
            assert combo["pattern"] == "neighbor"

    def test_repetitions(self):
        """Test repetitions multiply combinations."""
        config = SweepConfig(
            parameters={"transfer_size": [64, 128]},
            repetitions=3,
        )

        # 2 values × 3 repetitions = 6
        assert config.num_combinations == 6

    def test_empty_parameters_raises(self):
        """Test that empty parameters raises error."""
        with pytest.raises(ValueError):
            SweepConfig(parameters={})


class TestSweepResults:
    """Tests for SweepResults."""

    def test_add_result(self):
        """Test adding results."""
        results = SweepResults()

        results.add_result(
            params={"transfer_size": 64},
            metrics={"throughput": 8.0, "latency": 10.0}
        )

        assert len(results) == 1
        assert "transfer_size" in results.parameters
        assert "throughput" in results.metrics

    def test_get_series(self):
        """Test extracting series for plotting."""
        results = SweepResults()

        # Add data points
        for size, tp in [(64, 8.0), (128, 15.0), (256, 28.0)]:
            results.add_result(
                params={"transfer_size": size},
                metrics={"throughput": tp}
            )

        x, y = results.get_series("transfer_size", "throughput")

        assert x == [64, 128, 256]
        assert y == [8.0, 15.0, 28.0]

    def test_get_series_with_aggregation(self):
        """Test aggregation of repeated values."""
        results = SweepResults()

        # Add multiple runs for same parameter
        for _ in range(3):
            results.add_result({"size": 100}, {"value": 10.0})
            results.add_result({"size": 100}, {"value": 20.0})
            results.add_result({"size": 100}, {"value": 30.0})

        x, y = results.get_series("size", "value", aggregate="mean")
        assert y[0] == 20.0  # Mean of [10, 20, 30]

        x, y = results.get_series("size", "value", aggregate="min")
        assert y[0] == 10.0

        x, y = results.get_series("size", "value", aggregate="max")
        assert y[0] == 30.0

    def test_get_grouped_series(self):
        """Test grouping by secondary parameter."""
        results = SweepResults()

        # Add data for two groups
        for size, tp in [(64, 8.0), (128, 15.0)]:
            results.add_result(
                {"transfer_size": size, "buffer_depth": 4},
                {"throughput": tp}
            )
        for size, tp in [(64, 10.0), (128, 18.0)]:
            results.add_result(
                {"transfer_size": size, "buffer_depth": 8},
                {"throughput": tp}
            )

        grouped = results.get_grouped_series(
            "transfer_size", "throughput", "buffer_depth"
        )

        assert 4 in grouped
        assert 8 in grouped
        assert grouped[4][1] == [8.0, 15.0]
        assert grouped[8][1] == [10.0, 18.0]

    def test_filter_results(self):
        """Test filtering results."""
        results = SweepResults()

        results.add_result({"a": 1, "b": 2}, {"m": 10})
        results.add_result({"a": 1, "b": 3}, {"m": 20})
        results.add_result({"a": 2, "b": 2}, {"m": 30})

        filtered = results.filter(a=1)
        assert len(filtered) == 2

        filtered = results.filter(a=1, b=2)
        assert len(filtered) == 1

    def test_save_and_load(self):
        """Test saving and loading results."""
        results = SweepResults()
        results.add_result({"x": 1}, {"y": 10.0})
        results.add_result({"x": 2}, {"y": 20.0})

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "results.json"
            results.save(path)

            loaded = SweepResults.load(path)
            assert len(loaded) == 2
            assert loaded.parameters == results.parameters
            assert loaded.metrics == results.metrics

    def test_summary_statistics(self):
        """Test summary statistics."""
        results = SweepResults()
        for v in [10, 20, 30]:
            results.add_result({"x": v}, {"metric": float(v)})

        summary = results.summary()
        assert "metric" in summary
        assert summary["metric"]["min"] == 10.0
        assert summary["metric"]["max"] == 30.0
        assert summary["metric"]["mean"] == 20.0


class TestSweepRunner:
    """Tests for SweepRunner."""

    def test_run_sweep_with_mock(self):
        """Test running sweep with mock data."""
        config = SweepConfig(
            parameters={"transfer_size": [64, 128, 256]}
        )

        runner = SweepRunner(verbose=False)
        results = runner.run_sweep(config)

        assert len(results) == 3
        assert "throughput" in results.metrics

    def test_run_sweep_with_custom_func(self):
        """Test running sweep with custom simulation function."""
        config = SweepConfig(
            parameters={"x": [1, 2, 3]}
        )

        def custom_sim(params):
            return {"y": params["x"] * 10}

        runner = SweepRunner(verbose=False)
        results = runner.run_sweep(config, simulation_func=custom_sim)

        x, y = results.get_series("x", "y")
        assert y == [10, 20, 30]


class TestSweepCharts:
    """Tests for SweepCharts visualization."""

    @pytest.fixture
    def sample_results(self):
        """Create sample results for testing."""
        results = SweepResults()
        for size in [64, 128, 256, 512]:
            results.add_result(
                params={"transfer_size": size},
                metrics={
                    "throughput": size / 10,
                    "avg_latency": 5 + size / 100,
                    "total_cycles": size * 2,
                }
            )
        return results

    def test_plot_metric_vs_param(self, sample_results):
        """Test basic metric vs parameter plot."""
        import matplotlib
        matplotlib.use('Agg')  # Non-interactive backend

        fig = SweepCharts.plot_metric_vs_param(
            sample_results, "transfer_size", "throughput"
        )

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_multi_metric(self, sample_results):
        """Test multi-metric plot."""
        import matplotlib
        matplotlib.use('Agg')

        fig = SweepCharts.plot_multi_metric(
            sample_results,
            "transfer_size",
            ["throughput", "avg_latency"]
        )

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_plot_sweep_dashboard(self, sample_results):
        """Test dashboard plot."""
        import matplotlib
        matplotlib.use('Agg')

        fig = SweepCharts.plot_sweep_dashboard(
            sample_results, "transfer_size"
        )

        assert fig is not None
        import matplotlib.pyplot as plt
        plt.close(fig)

    def test_save_plot(self, sample_results):
        """Test saving plot to file."""
        import matplotlib
        matplotlib.use('Agg')

        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "test_plot.png"

            fig = SweepCharts.plot_metric_vs_param(
                sample_results, "transfer_size", "throughput"
            )
            fig.savefig(path)

            assert path.exists()

            import matplotlib.pyplot as plt
            plt.close(fig)


class TestConvenienceFunctions:
    """Test convenience functions."""

    def test_create_transfer_size_sweep(self):
        """Test preset configuration factory."""
        config = create_transfer_size_sweep(sizes=[64, 128])

        assert "transfer_size" in config.parameters
        assert config.parameters["transfer_size"] == [64, 128]

    def test_run_quick_sweep(self):
        """Test quick sweep function."""
        results = run_quick_sweep(
            parameter="transfer_size",
            values=[64, 128, 256],
        )

        assert len(results) == 3


def test_integration_full_workflow():
    """
    Integration test: full sweep workflow.

    1. Create config
    2. Run sweep
    3. Generate visualization
    """
    import matplotlib
    matplotlib.use('Agg')

    # 1. Create config
    config = SweepConfig(
        parameters={"transfer_size": [64, 128, 256, 512]},
        fixed_params={"num_transfers": 5},
    )

    # 2. Run sweep
    runner = SweepRunner(verbose=False)
    results = runner.run_sweep(config)

    # 3. Generate visualization
    fig = SweepCharts.plot_sweep_dashboard(results, "transfer_size")
    assert fig is not None

    # Verify data was collected
    assert len(results) == 4
    assert "throughput" in results.metrics
    assert "total_cycles" in results.metrics

    import matplotlib.pyplot as plt
    plt.close(fig)
