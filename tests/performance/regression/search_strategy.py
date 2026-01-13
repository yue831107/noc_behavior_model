"""
Search strategies for parameter optimization.

Currently implements GridSearch (exhaustive search).
Future: RandomSearch, BayesianOptimizer.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any


class SearchStrategy(ABC):
    """
    Abstract base class for search strategies.

    Search strategies control the order in which parameter
    combinations are tested.
    """

    @abstractmethod
    def initialize(self, candidates: List[Dict[str, Any]]) -> None:
        """
        Initialize the strategy with candidate parameters.

        Args:
            candidates: List of parameter combinations to search.
        """
        pass

    @abstractmethod
    def next_batch(self, batch_size: int = 1) -> List[Dict[str, Any]]:
        """
        Get the next batch of parameters to test.

        Args:
            batch_size: Number of parameter sets to return.

        Returns:
            List of parameter dictionaries.
        """
        pass

    @abstractmethod
    def is_complete(self) -> bool:
        """
        Check if all candidates have been tested.

        Returns:
            True if search is complete.
        """
        pass

    @abstractmethod
    def progress(self) -> float:
        """
        Get search progress as a fraction.

        Returns:
            Progress from 0.0 to 1.0.
        """
        pass


class GridSearch(SearchStrategy):
    """
    Exhaustive grid search strategy.

    Tests all parameter combinations in order.
    """

    def __init__(self):
        """Initialize grid search."""
        self._candidates: List[Dict[str, Any]] = []
        self._index: int = 0

    def initialize(self, candidates: List[Dict[str, Any]]) -> None:
        """
        Initialize with candidate parameters.

        Args:
            candidates: List of parameter combinations.
        """
        self._candidates = list(candidates)
        self._index = 0

    def next_batch(self, batch_size: int = 1) -> List[Dict[str, Any]]:
        """
        Get next batch of parameters.

        Args:
            batch_size: Number of parameters to return.

        Returns:
            List of parameter dictionaries.
        """
        if self._index >= len(self._candidates):
            return []

        batch = self._candidates[self._index : self._index + batch_size]
        self._index += len(batch)
        return batch

    def is_complete(self) -> bool:
        """Check if all candidates tested."""
        return self._index >= len(self._candidates)

    def progress(self) -> float:
        """Get search progress."""
        if len(self._candidates) == 0:
            return 1.0
        return self._index / len(self._candidates)

    @property
    def total_candidates(self) -> int:
        """Total number of candidates."""
        return len(self._candidates)

    @property
    def tested_count(self) -> int:
        """Number of candidates tested so far."""
        return self._index

    def __repr__(self) -> str:
        return (
            f"GridSearch(progress={self._index}/{len(self._candidates)})"
        )
