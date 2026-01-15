"""
Routing Selector module.

Provides the Routing Selector for V1 Architecture (single entry/exit point)
and the NoC-to-NoC System for multi-node traffic simulation.
"""

from .config import RoutingSelectorConfig, SelectorStats
from .edge_port import EdgeRouterPort, AXIModeEdgeRouterPort
from .selector import RoutingSelector
from .v1_system import V1System
from .noc_system import NoCSystem

__all__ = [
    # Configuration
    "RoutingSelectorConfig",
    "SelectorStats",
    # Edge Router Ports
    "EdgeRouterPort",
    "AXIModeEdgeRouterPort",
    # Core Selector
    "RoutingSelector",
    # System Classes
    "V1System",
    "NoCSystem",
]
