"""Core NoC components: Flit, Buffer, Router, NI, Mesh."""

from .flit import (
    Flit,
    FlitType,
    FlitHeader,
    FlitFactory,
    create_response_flit,
)
from .buffer import (
    Buffer,
    FlitBuffer,
    BufferStats,
    CreditFlowControl,
    PortBuffer,
)
from .packet import (
    Packet,
    PacketType,
    PacketAssembler,
    PacketDisassembler,
    PacketFactory,
)
from .router import (
    Direction,
    PipelineConfig,
    RouterConfig,
    RouterStats,
    RouterPort,
    XYRouter,
    ReqRouter,
    RespRouter,
    Router,
    EdgeRouter,
    create_router,
)
from .ni import (
    NIConfig,
    NIStats,
    SlaveNI,
    MasterNI,
    AXISlave,
    LocalMemoryUnit,
    # Backward compatibility aliases
    NetworkInterface,  # Alias for SlaveNI
    ReqNI,             # Alias for _SlaveNI_ReqPath (deprecated)
    RespNI,            # Alias for _SlaveNI_RspPath (deprecated)
)
from .mesh import (
    MeshConfig,
    MeshStats,
    Mesh,
    create_mesh,
)
from .routing_selector import (
    RoutingSelectorConfig,
    SelectorStats,
    RoutingSelector,
    V1System,
    NoCSystem,
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
from .golden_manager import (
    GoldenKey,
    GoldenSource,
    GoldenManager,
    GoldenEntry,
    VerificationResult,
    VerificationReport,
)

__all__ = [
    # Flit
    "Flit",
    "FlitType",
    "FlitHeader",
    "FlitFactory",
    "create_response_flit",
    # Buffer
    "Buffer",
    "FlitBuffer",
    "BufferStats",
    "CreditFlowControl",
    "PortBuffer",
    # Packet
    "Packet",
    "PacketType",
    "PacketAssembler",
    "PacketDisassembler",
    "PacketFactory",
    # Router
    "Direction",
    "PipelineConfig",
    "RouterConfig",
    "RouterStats",
    "RouterPort",
    "XYRouter",
    "ReqRouter",
    "RespRouter",
    "Router",
    "EdgeRouter",
    "create_router",
    # NI
    "NIConfig",
    "NIStats",
    "SlaveNI",
    "MasterNI",
    "AXISlave",
    "NetworkInterface",  # Backward compatibility alias for SlaveNI
    "ReqNI",             # Deprecated alias
    "RespNI",            # Deprecated alias
    # Mesh
    "MeshConfig",
    "MeshStats",
    "Mesh",
    "create_mesh",
    # Routing Selector
    "RoutingSelectorConfig",
    "SelectorStats",
    "RoutingSelector",
    "V1System",
    "NoCSystem",
    # Local AXI Master (NoC-to-NoC)
    "LocalAXIMasterState",
    "LocalTransferConfig",
    "LocalAXIMaster",
    # Node Controller (NoC-to-NoC)
    "NodeController",
    "NodeControllerStats",
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
    # Golden Manager
    "GoldenKey",
    "GoldenSource",
    "GoldenManager",
    "GoldenEntry",
    "VerificationResult",
    "VerificationReport",
]
