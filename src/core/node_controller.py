"""
Node Controller for NoC-to-NoC simulation.

Each node has a NodeController that manages:
- LocalAXIMaster: Initiates transfers to other nodes
- SlaveNI: Converts AXI requests to NoC flits (with user signal routing)
- MasterNI: Receives NoC flits and writes to local memory
- LocalMemory: Storage for this node

This enables bidirectional communication between nodes.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple, TYPE_CHECKING

from .local_axi_master import LocalAXIMaster, LocalTransferConfig
from .ni import SlaveNI, MasterNI, NIConfig
from .memory import Memory, MemoryConfig, LocalMemory
from ..config import NodeTransferConfig
from ..address.address_map import SystemAddressMap, create_address_map

if TYPE_CHECKING:
    from .router import Router


@dataclass
class NodeControllerStats:
    """Statistics for NodeController."""
    transfers_initiated: int = 0
    transfers_completed: int = 0
    bytes_sent: int = 0
    bytes_received: int = 0


class NodeController:
    """
    Node Controller for bidirectional NoC-to-NoC communication.

    Each compute node has one NodeController that manages:
    - Outgoing transfers: LocalAXIMaster -> SlaveNI -> Mesh
    - Incoming transfers: Mesh -> MasterNI -> LocalMemory

    This is the "per-node" abstraction for NoC-to-NoC testing.
    """

    def __init__(
        self,
        node_id: int,
        mesh_cols: int = 5,
        mesh_rows: int = 4,
        memory_size: int = 0x100000000,
        ni_config: Optional[NIConfig] = None,
    ):
        """
        Initialize NodeController.

        Args:
            node_id: This node's ID.
            mesh_cols: Mesh columns.
            mesh_rows: Mesh rows.
            memory_size: Local memory size in bytes.
            ni_config: NI configuration (defaults with user_signal_routing=True).
        """
        self.node_id = node_id
        self.mesh_cols = mesh_cols
        self.mesh_rows = mesh_rows

        # Calculate coordinate
        compute_cols = mesh_cols - 1
        self.coord = (
            (node_id % compute_cols) + 1,  # x (skip edge column)
            node_id // compute_cols         # y
        )

        # Create address map (for SlaveNI address translation fallback)
        self.address_map = create_address_map(mesh_cols, mesh_rows)

        # Create NI config with user signal routing enabled
        self.ni_config = ni_config or NIConfig(
            use_user_signal_routing=True,  # Enable NoC-to-NoC mode
        )

        # === Local Memory (AXI Slave side of MasterNI) ===
        self.local_memory = Memory(
            config=MemoryConfig(size=memory_size),
            name=f"Node{node_id}_LocalMemory"
        )

        # === SlaveNI (receives from LocalAXIMaster, sends to mesh) ===
        self.slave_ni = SlaveNI(
            coord=self.coord,
            address_map=self.address_map,
            config=self.ni_config,
            ni_id=node_id,
        )

        # === MasterNI (receives from mesh, sends to local memory) ===
        self.master_ni = MasterNI(
            coord=self.coord,
            config=self.ni_config,
            ni_id=node_id,
            node_id=node_id,
            memory_size=memory_size,
        )

        # === LocalAXIMaster (initiates transfers) ===
        self.local_master = LocalAXIMaster(
            node_id=node_id,
            local_memory=self.local_memory,
            mesh_cols=mesh_cols,
            mesh_rows=mesh_rows,
        )

        # Connect LocalAXIMaster to SlaveNI
        self.local_master.connect_to_slave_ni(self.slave_ni)

        # Transfer configuration
        self._transfer_config: Optional[NodeTransferConfig] = None

        # Statistics
        self.stats = NodeControllerStats()

    def configure_transfer(self, config: NodeTransferConfig) -> None:
        """
        Configure this node's outgoing transfer.

        Args:
            config: Transfer configuration.
        """
        self._transfer_config = config

        # Convert to LocalTransferConfig for LocalAXIMaster
        local_config = LocalTransferConfig(
            dest_coord=config.dest_coord,
            local_src_addr=config.local_src_addr,
            local_dst_addr=config.local_dst_addr,
            transfer_size=config.transfer_size,
        )
        self.local_master.configure_transfer(local_config)

    def initialize_memory(self, addr: int, data: bytes) -> None:
        """
        Initialize local memory with data.

        Args:
            addr: Memory address.
            data: Data to write.
        """
        self.local_memory.write(addr, data)

    def start_transfer(self) -> None:
        """Start the configured transfer."""
        if self._transfer_config is None:
            return

        self.local_master.start()
        self.stats.transfers_initiated += 1
        self.stats.bytes_sent += self._transfer_config.transfer_size

    def process_cycle(self, cycle: int) -> None:
        """
        Process one simulation cycle.

        Args:
            cycle: Current simulation cycle.
        """
        # Track completion state before processing
        was_complete = self.local_master.is_complete
        
        # Process LocalAXIMaster (generates AW/W, receives B)
        self.local_master.process_cycle(cycle)

        # Process SlaveNI (converts AW/W to flits)
        self.slave_ni.process_cycle(cycle)

        # Process MasterNI (converts flits to AXI, writes to memory)
        self.master_ni.process_cycle(cycle)
        
        # Check if transfer just completed this cycle
        if not was_complete and self.local_master.is_complete:
            self.stats.transfers_completed += 1

    @property
    def is_transfer_complete(self) -> bool:
        """Check if this node's transfer is complete."""
        return self.local_master.is_complete

    @property
    def is_idle(self) -> bool:
        """Check if node is idle (not transferring)."""
        return self.local_master.is_idle

    def get_outgoing_flit(self):
        """
        Get outgoing request flit from SlaveNI.

        Called by mesh to route flits.
        """
        return self.slave_ni.get_req_flit()

    def receive_incoming_flit(self, flit) -> bool:
        """
        Receive incoming request flit at MasterNI.

        Called by mesh when routing flits to this node.
        """
        return self.master_ni.receive_req_flit(flit)

    def get_response_flit(self):
        """
        Get outgoing response flit from MasterNI.

        Called by mesh to route response flits back.
        """
        return self.master_ni.get_resp_flit()

    def receive_response_flit(self, flit) -> bool:
        """
        Receive incoming response flit at SlaveNI.

        Called by mesh when routing response flits.
        """
        return self.slave_ni.receive_resp_flit(flit)

    def read_local_memory(self, addr: int, size: int) -> bytes:
        """Read from local memory (for verification)."""
        data, _ = self.local_memory.read(addr, size)
        return data

    def verify_memory(self, addr: int, expected: bytes) -> bool:
        """Verify local memory contents match expected data."""
        actual, _ = self.local_memory.read(addr, len(expected))
        return actual == expected

    def get_summary(self) -> Dict:
        """Get node summary."""
        return {
            "node_id": self.node_id,
            "coord": self.coord,
            "transfer_state": self.local_master._state.value,
            "transfer_config": {
                "dest_coord": self._transfer_config.dest_coord if self._transfer_config else None,
                "size": self._transfer_config.transfer_size if self._transfer_config else 0,
            },
            "stats": {
                "transfers_initiated": self.stats.transfers_initiated,
                "transfers_completed": self.stats.transfers_completed,
                "bytes_sent": self.stats.bytes_sent,
                "bytes_received": self.stats.bytes_received,
            },
        }

    def __repr__(self) -> str:
        return (
            f"NodeController(id={self.node_id}, coord={self.coord}, "
            f"state={self.local_master._state.value})"
        )
