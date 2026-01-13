"""
Tests for Regression Test Suite.

Tests parameter optimization, target matching, and result handling.
"""

import pytest
from pathlib import Path
import tempfile

from tests.performance.regression import (
    PerformanceTarget,
    ParameterSpace,
    GridSearch,
    ParameterOptimizer,
    OptimizationResult,
    RegressionReport,
)


class TestPerformanceTarget:
    """Tests for PerformanceTarget."""

    def test_create_target(self):
        """Test creating a performance target."""
        target = PerformanceTarget(
            max_latency=20,
            min_throughput=16,
        )

        assert target.max_latency == 20
        assert target.min_throughput == 16
        assert target.latency_weight == 0.5
        assert target.throughput_weight == 0.5

    def test_is_satisfied_both_met(self):
        """Test satisfaction when both constraints are met."""
        target = PerformanceTarget(max_latency=20, min_throughput=16)

        metrics = {"avg_latency": 15, "throughput": 20}
        assert target.is_satisfied(metrics) is True

    def test_is_satisfied_latency_exceeded(self):
        """Test non-satisfaction when latency exceeds target."""
        target = PerformanceTarget(max_latency=20, min_throughput=16)

        metrics = {"avg_latency": 25, "throughput": 20}
        assert target.is_satisfied(metrics) is False

    def test_is_satisfied_throughput_too_low(self):
        """Test non-satisfaction when throughput is below target."""
        target = PerformanceTarget(max_latency=20, min_throughput=16)

        metrics = {"avg_latency": 15, "throughput": 10}
        assert target.is_satisfied(metrics) is False

    def test_score_better_is_higher(self):
        """Test that better metrics produce higher scores."""
        target = PerformanceTarget(max_latency=20, min_throughput=16)

        good_metrics = {"avg_latency": 10, "throughput": 25}
        poor_metrics = {"avg_latency": 18, "throughput": 17}

        assert target.score(good_metrics) > target.score(poor_metrics)

    def test_score_weights(self):
        """Test that weights affect scoring."""
        # Latency-focused target
        lat_target = PerformanceTarget(
            max_latency=20, min_throughput=16,
            latency_weight=0.8, throughput_weight=0.2
        )

        # Throughput-focused target
        tp_target = PerformanceTarget(
            max_latency=20, min_throughput=16,
            latency_weight=0.2, throughput_weight=0.8
        )

        # Low latency, low throughput
        low_lat = {"avg_latency": 5, "throughput": 18}
        # High latency, high throughput
        high_tp = {"avg_latency": 18, "throughput": 30}

        # Latency-focused should prefer low_lat
        assert lat_target.score(low_lat) > lat_target.score(high_tp)
        # Throughput-focused should prefer high_tp
        assert tp_target.score(high_tp) > tp_target.score(low_lat)

    def test_serialization(self):
        """Test to_dict and from_dict."""
        target = PerformanceTarget(
            max_latency=15,
            min_throughput=25,
            latency_weight=0.7,
            throughput_weight=0.3,
        )

        data = target.to_dict()
        restored = PerformanceTarget.from_dict(data)

        assert restored.max_latency == target.max_latency
        assert restored.min_throughput == target.min_throughput
        assert restored.latency_weight == target.latency_weight
        assert restored.throughput_weight == target.throughput_weight


class TestParameterSpace:
    """Tests for ParameterSpace."""

    def test_create_space(self):
        """Test creating a parameter space."""
        space = ParameterSpace(
            mesh_rows=[2, 4],
            mesh_cols=[3, 5],
            buffer_depth=[4, 8],
            max_outstanding=[8],
        )

        assert space.mesh_rows == [2, 4]
        assert space.mesh_cols == [3, 5]
        assert space.buffer_depth == [4, 8]
        assert space.max_outstanding == [8]

    def test_total_combinations(self):
        """Test counting total combinations."""
        space = ParameterSpace(
            mesh_rows=[2, 4],         # 2 values
            mesh_cols=[3, 5, 7],      # 3 values
            buffer_depth=[4, 8],      # 2 values
            max_outstanding=[8, 16],  # 2 values
        )

        # 2 × 3 × 2 × 2 = 24
        assert space.total_combinations() == 24

    def test_generate_combinations(self):
        """Test generating all combinations."""
        space = ParameterSpace(
            mesh_rows=[2, 4],
            mesh_cols=[3],
            buffer_depth=[8],
            max_outstanding=[16],
        )

        combos = list(space.generate_combinations())
        assert len(combos) == 2

        # Check all expected keys present
        for combo in combos:
            assert "mesh_rows" in combo
            assert "mesh_cols" in combo
            assert "buffer_depth" in combo
            assert "max_outstanding" in combo

    def test_prefilter_throughput(self):
        """Test prefiltering removes low-throughput configs."""
        space = ParameterSpace(
            mesh_rows=[2, 4, 8],  # T_max = rows × 8
            mesh_cols=[5],
            buffer_depth=[8],
            max_outstanding=[16],
        )

        # Target requires 20 B/cycle
        # T_max(rows=2) = 16 < 20, should be filtered
        # T_max(rows=4) = 32 >= 20, should pass
        # T_max(rows=8) = 64 >= 20, should pass
        target = PerformanceTarget(max_latency=50, min_throughput=20)
        feasible = space.prefilter(target)

        assert len(feasible) == 2
        rows = {f["mesh_rows"] for f in feasible}
        assert 2 not in rows
        assert 4 in rows
        assert 8 in rows

    def test_prefilter_latency(self):
        """Test prefiltering removes high-latency configs."""
        space = ParameterSpace(
            mesh_rows=[4],
            mesh_cols=[3, 5, 9],  # L_min = (cols-1)+(rows-1)+2
            buffer_depth=[8],
            max_outstanding=[16],
        )

        # Target requires latency <= 10 cycles
        # L_min(cols=3) = 2+3+2 = 7 <= 10, should pass
        # L_min(cols=5) = 4+3+2 = 9 <= 10, should pass
        # L_min(cols=9) = 8+3+2 = 13 > 10, should be filtered
        target = PerformanceTarget(max_latency=10, min_throughput=1)
        feasible = space.prefilter(target)

        assert len(feasible) == 2
        cols = {f["mesh_cols"] for f in feasible}
        assert 3 in cols
        assert 5 in cols
        assert 9 not in cols

    def test_estimate_bounds(self):
        """Test theoretical bounds estimation."""
        space = ParameterSpace()
        params = {"mesh_rows": 4, "mesh_cols": 5, "buffer_depth": 8}

        bounds = space.estimate_theoretical_bounds(params)

        assert bounds["t_max"] == 4 * 8  # rows × flit_width
        assert bounds["l_min"] == (5 - 1) + (4 - 1) + 2  # max_hops + overhead
        assert bounds["max_hops"] == (5 - 1) + (4 - 1)

    def test_serialization(self):
        """Test to_dict and from_dict."""
        space = ParameterSpace(
            mesh_rows=[2, 4],
            mesh_cols=[3, 5],
            buffer_depth=[4, 8],
            max_outstanding=[8],
        )

        data = space.to_dict()
        restored = ParameterSpace.from_dict(data)

        assert restored.mesh_rows == space.mesh_rows
        assert restored.mesh_cols == space.mesh_cols
        assert restored.buffer_depth == space.buffer_depth
        assert restored.max_outstanding == space.max_outstanding


class TestGridSearch:
    """Tests for GridSearch strategy."""

    def test_initialize_and_iterate(self):
        """Test initialization and iteration."""
        strategy = GridSearch()

        candidates = [
            {"a": 1, "b": 2},
            {"a": 1, "b": 3},
            {"a": 2, "b": 2},
        ]

        strategy.initialize(candidates)
        assert not strategy.is_complete()
        assert strategy.total_candidates == 3

        # Get all batches
        results = []
        while not strategy.is_complete():
            batch = strategy.next_batch(batch_size=1)
            results.extend(batch)

        assert len(results) == 3
        assert strategy.is_complete()

    def test_batch_size(self):
        """Test batch retrieval with different sizes."""
        strategy = GridSearch()

        candidates = [{"x": i} for i in range(10)]
        strategy.initialize(candidates)

        batch1 = strategy.next_batch(batch_size=3)
        assert len(batch1) == 3

        batch2 = strategy.next_batch(batch_size=5)
        assert len(batch2) == 5

        batch3 = strategy.next_batch(batch_size=5)
        assert len(batch3) == 2  # Only 2 remaining

    def test_progress(self):
        """Test progress tracking."""
        strategy = GridSearch()
        strategy.initialize([{"x": 1}, {"x": 2}, {"x": 3}, {"x": 4}])

        assert strategy.progress() == 0.0

        strategy.next_batch(2)
        assert strategy.progress() == 0.5

        strategy.next_batch(2)
        assert strategy.progress() == 1.0

    def test_empty_candidates(self):
        """Test handling of empty candidates."""
        strategy = GridSearch()
        strategy.initialize([])

        assert strategy.is_complete()
        assert strategy.progress() == 1.0
        assert strategy.next_batch(1) == []


class TestParameterOptimizer:
    """Tests for ParameterOptimizer."""

    def test_optimize_finds_best(self):
        """Test that optimizer finds best solution."""
        target = PerformanceTarget(max_latency=20, min_throughput=20)
        space = ParameterSpace(
            mesh_rows=[4, 8],
            mesh_cols=[5],
            buffer_depth=[8, 16],
            max_outstanding=[16],
        )

        optimizer = ParameterOptimizer(
            target=target,
            parameter_space=space,
            verbose=False,
        )

        result = optimizer.optimize()

        assert result is not None
        assert result.best_params is not None
        assert result.best_metrics is not None
        assert result.tested_count > 0

    def test_optimize_early_stop(self):
        """Test early stopping when solution found."""
        target = PerformanceTarget(max_latency=100, min_throughput=1)
        space = ParameterSpace(
            mesh_rows=[2, 4, 6, 8],
            mesh_cols=[3, 5, 7],
            buffer_depth=[4, 8, 16],
            max_outstanding=[8, 16],
        )

        optimizer = ParameterOptimizer(
            target=target,
            parameter_space=space,
            verbose=False,
        )

        result = optimizer.optimize(early_stop=True)

        # With early stop, should find solution before testing all
        assert result.target_satisfied
        assert result.tested_count < result.feasible_count or result.feasible_count <= 1

    def test_optimize_tracks_all_results(self):
        """Test that all results are tracked."""
        target = PerformanceTarget(max_latency=50, min_throughput=10)
        space = ParameterSpace(
            mesh_rows=[4],
            mesh_cols=[5],
            buffer_depth=[8],
            max_outstanding=[16],
        )

        optimizer = ParameterOptimizer(
            target=target,
            parameter_space=space,
            verbose=False,
        )

        result = optimizer.optimize()

        assert len(result.all_results) == result.tested_count
        for entry in result.all_results:
            assert "params" in entry
            assert "metrics" in entry
            assert "score" in entry


class TestOptimizationResult:
    """Tests for OptimizationResult."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample result for testing."""
        target = PerformanceTarget(max_latency=20, min_throughput=16)

        all_results = [
            {"params": {"rows": 4}, "metrics": {"avg_latency": 10, "throughput": 20}, "score": 0.8, "satisfied": True},
            {"params": {"rows": 6}, "metrics": {"avg_latency": 12, "throughput": 25}, "score": 0.9, "satisfied": True},
            {"params": {"rows": 2}, "metrics": {"avg_latency": 8, "throughput": 12}, "score": 0.5, "satisfied": False},
        ]

        return OptimizationResult(
            best_params={"rows": 6},
            best_metrics={"avg_latency": 12, "throughput": 25},
            best_score=0.9,
            target_satisfied=True,
            target=target,
            total_combinations=10,
            feasible_count=5,
            tested_count=3,
            satisfied_count=2,
            all_results=all_results,
        )

    def test_get_top_n(self, sample_result):
        """Test getting top N solutions."""
        top2 = sample_result.get_top_n(2)

        assert len(top2) == 2
        assert top2[0]["score"] >= top2[1]["score"]

    def test_get_satisfied_solutions(self, sample_result):
        """Test getting satisfied solutions."""
        satisfied = sample_result.get_satisfied_solutions()

        assert len(satisfied) == 2
        for s in satisfied:
            assert s["satisfied"] is True

    def test_summary(self, sample_result):
        """Test generating summary text."""
        summary = sample_result.summary()

        assert "TARGET SATISFIED" in summary
        assert "Best Parameters" in summary
        assert "Best Metrics" in summary

    def test_save_and_load(self, sample_result):
        """Test saving and loading results."""
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "result.json"
            sample_result.save(path)

            loaded = OptimizationResult.load(path)

            assert loaded.best_score == sample_result.best_score
            assert loaded.target_satisfied == sample_result.target_satisfied
            assert loaded.tested_count == sample_result.tested_count


class TestRegressionReport:
    """Tests for RegressionReport."""

    @pytest.fixture
    def sample_result(self):
        """Create a sample result for report testing."""
        target = PerformanceTarget(max_latency=20, min_throughput=16)

        all_results = [
            {
                "params": {"mesh_rows": 4, "mesh_cols": 5, "buffer_depth": 8},
                "metrics": {"avg_latency": 10, "throughput": 20},
                "score": 0.8,
                "satisfied": True,
            },
            {
                "params": {"mesh_rows": 6, "mesh_cols": 5, "buffer_depth": 8},
                "metrics": {"avg_latency": 12, "throughput": 25},
                "score": 0.9,
                "satisfied": True,
            },
        ]

        return OptimizationResult(
            best_params={"mesh_rows": 6, "mesh_cols": 5, "buffer_depth": 8},
            best_metrics={"avg_latency": 12, "throughput": 25},
            best_score=0.9,
            target_satisfied=True,
            target=target,
            total_combinations=10,
            feasible_count=5,
            tested_count=2,
            satisfied_count=2,
            all_results=all_results,
        )

    def test_generate_summary(self, sample_result):
        """Test summary generation."""
        report = RegressionReport(sample_result)
        summary = report.generate_summary()

        assert "Regression Test Results" in summary
        assert "TARGET SATISFIED" in summary

    def test_generate_top_n_table(self, sample_result):
        """Test top N table generation."""
        report = RegressionReport(sample_result)
        table = report.generate_top_n_table(n=5)

        assert "Top" in table
        assert "Rank" in table
        assert "Score" in table

    def test_generate_parameter_analysis(self, sample_result):
        """Test parameter analysis generation."""
        report = RegressionReport(sample_result)
        analysis = report.generate_parameter_analysis()

        assert "Parameter Analysis" in analysis
        assert "mesh_rows" in analysis

    def test_save_full_report(self, sample_result):
        """Test saving full report."""
        with tempfile.TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir)

            report = RegressionReport(sample_result)
            report.save_full_report(output_dir)

            assert (output_dir / "regression_result.json").exists()
            assert (output_dir / "summary.txt").exists()


def test_integration_full_workflow():
    """
    Integration test: full regression workflow.

    1. Define target
    2. Create parameter space
    3. Run optimization
    4. Generate report
    """
    # 1. Define target
    target = PerformanceTarget(
        max_latency=30,
        min_throughput=20,
        latency_weight=0.6,
        throughput_weight=0.4,
    )

    # 2. Create parameter space
    space = ParameterSpace(
        mesh_rows=[4, 6],
        mesh_cols=[5, 7],
        buffer_depth=[8, 16],
        max_outstanding=[16],
    )

    # 3. Run optimization
    optimizer = ParameterOptimizer(
        target=target,
        parameter_space=space,
        verbose=False,
    )

    result = optimizer.optimize()

    # 4. Generate report
    report = RegressionReport(result)
    summary = report.generate_summary()
    table = report.generate_top_n_table()

    # Verify workflow completed
    assert result.tested_count > 0
    assert result.best_params is not None
    assert "Regression Test Results" in summary
    assert "Top" in table

    # Test saving
    with tempfile.TemporaryDirectory() as tmpdir:
        report.save_full_report(Path(tmpdir))
        assert (Path(tmpdir) / "regression_result.json").exists()
