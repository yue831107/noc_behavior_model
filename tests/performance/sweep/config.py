"""
Parameter Sweep Configuration.

Defines sweep parameters and combinations for NoC performance analysis.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Any, Iterator, Optional
from enum import Enum
import itertools


class SweepMode(Enum):
    """Sweep mode for parameter combinations."""
    GRID = "grid"      # All combinations (cartesian product)
    LINEAR = "linear"  # One parameter at a time


@dataclass
class SweepConfig:
    """
    Configuration for parameter sweep.

    Attributes:
        parameters: Dict mapping parameter names to lists of values
        fixed_params: Parameters that remain constant across sweep
        mode: How to combine parameters (grid or linear)
        repetitions: Number of times to repeat each configuration
        system_type: Type of simulation system ("host_to_noc" or "noc_to_noc")
    """
    parameters: Dict[str, List[Any]]
    fixed_params: Dict[str, Any] = field(default_factory=dict)
    mode: SweepMode = SweepMode.GRID
    repetitions: int = 1
    system_type: str = "host_to_noc"

    def __post_init__(self):
        """Validate configuration."""
        if not self.parameters:
            raise ValueError("At least one parameter must be specified")

        for name, values in self.parameters.items():
            if not values:
                raise ValueError(f"Parameter '{name}' has no values")

    @property
    def num_combinations(self) -> int:
        """Total number of parameter combinations."""
        if self.mode == SweepMode.GRID:
            total = 1
            for values in self.parameters.values():
                total *= len(values)
            return total * self.repetitions
        else:  # LINEAR
            return sum(len(v) for v in self.parameters.values()) * self.repetitions

    @property
    def parameter_names(self) -> List[str]:
        """List of parameter names being swept."""
        return list(self.parameters.keys())

    def generate_combinations(self) -> Iterator[Dict[str, Any]]:
        """
        Generate parameter combinations based on mode.

        Yields:
            Dict with parameter name -> value for each combination
        """
        if self.mode == SweepMode.GRID:
            # Cartesian product of all parameters
            keys = list(self.parameters.keys())
            value_lists = [self.parameters[k] for k in keys]

            for values in itertools.product(*value_lists):
                combo = dict(zip(keys, values))
                combo.update(self.fixed_params)
                for _ in range(self.repetitions):
                    yield combo.copy()

        else:  # LINEAR
            # One parameter at a time, others at first value
            base_values = {k: v[0] for k, v in self.parameters.items()}
            base_values.update(self.fixed_params)

            for param_name, param_values in self.parameters.items():
                for value in param_values:
                    combo = base_values.copy()
                    combo[param_name] = value
                    for _ in range(self.repetitions):
                        yield combo.copy()


# Predefined sweep configurations for common scenarios
def create_transfer_size_sweep(
    sizes: Optional[List[int]] = None,
    **fixed_params
) -> SweepConfig:
    """Create sweep for transfer_size parameter."""
    return SweepConfig(
        parameters={
            "transfer_size": sizes or [64, 128, 256, 512, 1024, 2048]
        },
        fixed_params=fixed_params,
    )


def create_delay_sweep(
    delays: Optional[List[int]] = None,
    **fixed_params
) -> SweepConfig:
    """Create sweep for inter_txn_delay parameter."""
    return SweepConfig(
        parameters={
            "inter_txn_delay": delays or [0, 1, 2, 5, 10, 20]
        },
        fixed_params=fixed_params,
    )


def create_transfer_count_sweep(
    counts: Optional[List[int]] = None,
    **fixed_params
) -> SweepConfig:
    """Create sweep for num_transfers parameter."""
    return SweepConfig(
        parameters={
            "num_transfers": counts or [5, 10, 20, 50, 100]
        },
        fixed_params=fixed_params,
    )


def create_pattern_sweep(**fixed_params) -> SweepConfig:
    """Create sweep for traffic_pattern parameter."""
    return SweepConfig(
        parameters={
            "traffic_pattern": ["neighbor", "shuffle", "bit_reverse", "transpose", "random"]
        },
        fixed_params=fixed_params,
        system_type="noc_to_noc",
    )
