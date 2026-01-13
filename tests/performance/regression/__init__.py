"""
Regression Test Suite for NoC Behavior Model.

This module provides automated parameter optimization to find hardware
configurations that meet specified performance targets (latency, throughput).

Main components:
- PerformanceTarget: Define optimization goals
- ParameterSpace: Define searchable hardware parameters
- GridSearch: Search strategy (exhaustive grid search)
- ParameterOptimizer: Main optimization controller
- OptimizationResult: Results with best parameters and statistics
- RegressionReport: Report generation (text + charts)

Usage:
    from tests.performance.regression import (
        PerformanceTarget,
        ParameterSpace,
        GridSearch,
        ParameterOptimizer,
    )

    target = PerformanceTarget(max_latency=15, min_throughput=25)
    space = ParameterSpace(mesh_rows=[2, 4], buffer_depth=[4, 8, 16])
    optimizer = ParameterOptimizer(target, space, GridSearch())
    result = optimizer.optimize()
"""

from .target import PerformanceTarget
from .parameter_space import ParameterSpace
from .search_strategy import SearchStrategy, GridSearch
from .optimizer import ParameterOptimizer
from .result import OptimizationResult
from .report import RegressionReport

__all__ = [
    "PerformanceTarget",
    "ParameterSpace",
    "SearchStrategy",
    "GridSearch",
    "ParameterOptimizer",
    "OptimizationResult",
    "RegressionReport",
]
