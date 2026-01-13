"""
Local AXI Master for NoC-to-NoC transfers.

Unlike HostAXIMaster which uses address encoding (node_id << 32 | local_addr),
LocalAXIMaster uses AXI user signal for destination routing.

Key Differences from HostAXIMaster:
- Address: 32-bit local address (not 64-bit global)
- Destination: Encoded in awuser[15:0] as (dest_y << 8) | dest_x
- Simpler: No DMA-like burst splitting (single transfer per node)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, List, Dict, Tuple, TYPE_CHECKING
from enum import Enum

from ..axi.interface import (
    AXI_AW, AXI_W, AXI_B, AXI_AR, AXI_R,
    AXISize, AXIBurst, AXIResp,
)

if TYPE_CHECKING:
    from .memory import Memory
    from .ni import SlaveNI


class LocalAXIMasterState(Enum):
    """Local AXI Master state."""
    IDLE = "idle"
    RUNNING = "running"
    COMPLETE = "complete"


@dataclass
class LocalTransferConfig:
    """Configuration for a single local AXI transfer."""
    dest_coord: Tuple[int, int]     # (x, y) destination coordinate
    local_src_addr: int = 0x0000    # Source address in local memory
    local_dst_addr: int = 0x1000    # Destination address at target
    transfer_size: int = 256        # Bytes to transfer

    def encode_user_signal(self) -> int:
        """Encode destination coordinate into AXI user signal.

        Format: awuser[7:0] = dest_x, awuser[15:8] = dest_y
        """
        dest_x, dest_y = self.dest_coord
        return (dest_y << 8) | dest_x

    @staticmethod
    def decode_user_signal(user_signal: int) -> Tuple[int, int]:
        """Decode AXI user signal to destination coordinate.

        Args:
            user_signal: AXI user signal value.

        Returns:
            Tuple of (dest_x, dest_y).
        """
        dest_x = user_signal & 0xFF
        dest_y = (user_signal >> 8) & 0xFF
        return (dest_x, dest_y)


@dataclass
class LocalAXIMasterStats:
    """Statistics for Local AXI Master."""
    aw_sent: int = 0
    w_sent: int = 0
    b_received: int = 0
    total_cycles: int = 0
    first_aw_cycle: int = 0
    last_b_cycle: int = 0


class LocalAXIMaster:
    """
    Local AXI Master for NoC-to-NoC transfers.

    Each node has one LocalAXIMaster for initiating transfers to other nodes.
    Uses AXI user signal for destination routing instead of address encoding.

    Signal Flow:
        LocalMemory -> LocalAXIMaster -> SlaveNI -> Mesh -> MasterNI -> LocalMemory
                                             |                              |
                                             <-------- Response ------------->
    """

    def __init__(
        self,
        node_id: int,
        local_memory: "Memory",
        mesh_cols: int = 5,
        mesh_rows: int = 4,
    ):
        """
        Initialize Local AXI Master.

        Args:
            node_id: This node's ID.
            local_memory: Local memory to read source data from.
            mesh_cols: Mesh columns (for coordinate calculation).
            mesh_rows: Mesh rows (for coordinate calculation).
        """
        self.node_id = node_id
        self.local_memory = local_memory
        self.mesh_cols = mesh_cols
        self.mesh_rows = mesh_rows

        # Calculate this node's coordinate
        self.src_coord = self._node_id_to_coord(node_id)

        # Transfer configuration
        self._transfer_config: Optional[LocalTransferConfig] = None

        # Connected SlaveNI
        self._slave_ni: Optional["SlaveNI"] = None

        # State
        self._state = LocalAXIMasterState.IDLE
        self._current_cycle = 0

        # Pending transaction tracking
        self._pending_axi_id: Optional[int] = None
        self._pending_data: Optional[bytes] = None
        self._aw_sent: bool = False
        self._w_sent: bool = False

        # AXI ID counter
        self._next_axi_id = 0

        # Statistics
        self.stats = LocalAXIMasterStats()

    def _node_id_to_coord(self, node_id: int) -> Tuple[int, int]:
        """Convert node ID to (x, y) coordinate.

        Note: Assumes edge_column=0, so compute nodes start at x=1.
        """
        compute_cols = self.mesh_cols - 1  # Exclude edge column
        x = (node_id % compute_cols) + 1   # +1 to skip edge column
        y = node_id // compute_cols
        return (x, y)

    def _coord_to_node_id(self, coord: Tuple[int, int]) -> int:
        """Convert (x, y) coordinate to node ID."""
        x, y = coord
        compute_cols = self.mesh_cols - 1
        return y * compute_cols + (x - 1)

    def connect_to_slave_ni(self, slave_ni: "SlaveNI") -> None:
        """Connect to local SlaveNI."""
        self._slave_ni = slave_ni

    def configure_transfer(self, config: LocalTransferConfig) -> None:
        """
        Configure transfer parameters.

        Args:
            config: Transfer configuration.
        """
        self._transfer_config = config

    def reset(self) -> None:
        """Reset master to IDLE state."""
        self._state = LocalAXIMasterState.IDLE
        self._current_cycle = 0
        self._pending_axi_id = None
        self._pending_data = None
        self._aw_sent = False
        self._w_sent = False
        self.stats = LocalAXIMasterStats()

    def start(self) -> None:
        """Start the configured transfer."""
        if self._state != LocalAXIMasterState.IDLE:
            return
        if self._transfer_config is None:
            return

        # Read data from local memory
        config = self._transfer_config
        data, _ = self.local_memory.read(config.local_src_addr, config.transfer_size)
        self._pending_data = data

        # Allocate AXI ID
        self._pending_axi_id = self._next_axi_id
        self._next_axi_id = (self._next_axi_id + 1) % 16

        # Reset state
        self._aw_sent = False
        self._w_sent = False
        self._state = LocalAXIMasterState.RUNNING
        self._current_cycle = 0
        self.stats = LocalAXIMasterStats()

    def process_cycle(self, cycle: int) -> None:
        """
        Process one simulation cycle.

        Args:
            cycle: Current simulation cycle.
        """
        if self._state != LocalAXIMasterState.RUNNING:
            return

        self._current_cycle = cycle
        self.stats.total_cycles = cycle + 1

        # Phase 1: Send AW (Write Address)
        if not self._aw_sent:
            self._try_send_aw(cycle)

        # Phase 2: Send W (Write Data)
        if self._aw_sent and not self._w_sent:
            self._try_send_w(cycle)

        # Phase 3: Receive B (Write Response)
        self._try_receive_b(cycle)

    def _try_send_aw(self, cycle: int) -> None:
        """Try to send AW channel."""
        if self._slave_ni is None or self._transfer_config is None:
            return

        config = self._transfer_config

        # Create AW with destination in user signal
        aw = AXI_AW(
            awid=self._pending_axi_id,
            awaddr=config.local_dst_addr,  # 32-bit local address at destination
            awlen=0,  # Single beat
            awsize=AXISize.SIZE_8,
            awburst=AXIBurst.INCR,
            awuser=config.encode_user_signal(),  # Destination coordinate!
        )

        if self._slave_ni.process_aw(aw, cycle):
            self._aw_sent = True
            self.stats.aw_sent += 1
            if self.stats.first_aw_cycle == 0:
                self.stats.first_aw_cycle = cycle

    def _try_send_w(self, cycle: int) -> None:
        """Try to send W channel."""
        if self._slave_ni is None or self._pending_data is None:
            return

        # Create W beat with all data
        w = AXI_W(
            wdata=self._pending_data,
            wstrb=(1 << len(self._pending_data)) - 1,  # All bytes valid
            wlast=True,
        )

        if self._slave_ni.process_w(w, self._pending_axi_id, cycle):
            self._w_sent = True
            self.stats.w_sent += 1

    def _try_receive_b(self, cycle: int) -> None:
        """Try to receive B response."""
        if self._slave_ni is None:
            return

        b_resp = self._slave_ni.get_b_response()
        if b_resp is not None and b_resp.bid == self._pending_axi_id:
            self.stats.b_received += 1
            self.stats.last_b_cycle = cycle
            self._state = LocalAXIMasterState.COMPLETE

    @property
    def is_complete(self) -> bool:
        """Check if transfer is complete."""
        return self._state == LocalAXIMasterState.COMPLETE

    @property
    def is_idle(self) -> bool:
        """Check if master is idle."""
        return self._state == LocalAXIMasterState.IDLE

    @property
    def is_running(self) -> bool:
        """Check if master is running."""
        return self._state == LocalAXIMasterState.RUNNING

    def get_summary(self) -> Dict:
        """Get transfer summary."""
        return {
            "node_id": self.node_id,
            "src_coord": self.src_coord,
            "state": self._state.value,
            "dest_coord": self._transfer_config.dest_coord if self._transfer_config else None,
            "timing": {
                "total_cycles": self.stats.total_cycles,
                "first_aw_cycle": self.stats.first_aw_cycle,
                "last_b_cycle": self.stats.last_b_cycle,
            },
            "axi_channels": {
                "aw_sent": self.stats.aw_sent,
                "w_sent": self.stats.w_sent,
                "b_received": self.stats.b_received,
            },
        }

    def __repr__(self) -> str:
        return (
            f"LocalAXIMaster(node={self.node_id}, "
            f"coord={self.src_coord}, state={self._state.value})"
        )
