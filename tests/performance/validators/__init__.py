"""
Performance validators for NoC metrics validation.

All validators use Monitor-based approach - they observe metrics
from the system without modifying core implementation.
"""

from .theory_validator import TheoryValidator, MeshConfig, RouterConfig
from .consistency_validator import ConsistencyValidator

__all__ = [
    'TheoryValidator',
    'MeshConfig',
    'RouterConfig',
    'ConsistencyValidator',
]
