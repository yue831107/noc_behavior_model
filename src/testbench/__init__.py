"""Testbench components: Memory, AXI Masters, Node Controllers."""

from .memory import (
    MemoryConfig,
    Memory,
    MemoryStats,
    HostMemory,
    LocalMemory,
    MemoryCopyDescriptor,
)
from .axi_master import (
    AXIIdConfig,
    AXIIdGenerator,
    AXIMasterState,
    PendingTransaction,
    AXIMasterStats,
    AXIMasterController,
)
from .host_axi_master import (
    HostAXIMasterState,
    AXIChannelPort,
    AXIResponsePort,
    HostAXIMasterStats,
    HostAXIMaster,
)
from .local_axi_master import (
    LocalAXIMasterState,
    LocalTransferConfig,
    LocalAXIMaster,
)
from .node_controller import (
    NodeController,
    NodeControllerStats,
)

__all__ = [
    # Memory
    "MemoryConfig",
    "Memory",
    "MemoryStats",
    "HostMemory",
    "LocalMemory",
    "MemoryCopyDescriptor",
    # AXI Master Controller
    "AXIIdConfig",
    "AXIIdGenerator",
    "AXIMasterState",
    "PendingTransaction",
    "AXIMasterStats",
    "AXIMasterController",
    # Host AXI Master
    "HostAXIMasterState",
    "AXIChannelPort",
    "AXIResponsePort",
    "HostAXIMasterStats",
    "HostAXIMaster",
    # Local AXI Master (NoC-to-NoC)
    "LocalAXIMasterState",
    "LocalTransferConfig",
    "LocalAXIMaster",
    # Node Controller (NoC-to-NoC)
    "NodeController",
    "NodeControllerStats",
]
