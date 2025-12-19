"""
AXI4 Interface definitions.

Implements the five AXI4 channels:
- AW (Write Address)
- W (Write Data)
- B (Write Response)
- AR (Read Address)
- R (Read Data)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import IntEnum
from typing import Optional, List


class AXIBurst(IntEnum):
    """AXI burst type."""
    FIXED = 0   # Fixed address burst
    INCR = 1    # Incrementing address burst
    WRAP = 2    # Wrapping burst


class AXIResp(IntEnum):
    """AXI response type."""
    OKAY = 0      # Normal access success
    EXOKAY = 1    # Exclusive access okay
    SLVERR = 2    # Slave error
    DECERR = 3    # Decode error


class AXISize(IntEnum):
    """AXI transfer size (bytes per beat)."""
    SIZE_1 = 0    # 1 byte
    SIZE_2 = 1    # 2 bytes
    SIZE_4 = 2    # 4 bytes
    SIZE_8 = 3    # 8 bytes
    SIZE_16 = 4   # 16 bytes
    SIZE_32 = 5   # 32 bytes
    SIZE_64 = 6   # 64 bytes
    SIZE_128 = 7  # 128 bytes

    @property
    def bytes(self) -> int:
        """Get size in bytes."""
        return 1 << self.value


# =============================================================================
# Write Address Channel (AW)
# =============================================================================

@dataclass
class AXI_AW:
    """
    AXI Write Address Channel.

    Attributes:
        awid: Write address ID (transaction identifier).
        awaddr: Write address (64-bit).
        awlen: Burst length (0-255 → 1-256 beats).
        awsize: Burst size (bytes per beat).
        awburst: Burst type.
        awlock: Lock type (not used in AXI4).
        awcache: Memory type.
        awprot: Protection type.
        awqos: Quality of Service.
        awregion: Region identifier.
        awuser: User-defined signal.
    """
    awid: int
    awaddr: int
    awlen: int = 0          # 0 = 1 beat
    awsize: AXISize = AXISize.SIZE_8
    awburst: AXIBurst = AXIBurst.INCR
    awlock: int = 0
    awcache: int = 0
    awprot: int = 0
    awqos: int = 0
    awregion: int = 0
    awuser: int = 0

    @property
    def burst_length(self) -> int:
        """Actual burst length (1-256)."""
        return self.awlen + 1

    @property
    def transfer_size(self) -> int:
        """Total transfer size in bytes."""
        return self.burst_length * self.awsize.bytes

    def __repr__(self) -> str:
        return (
            f"AXI_AW(id={self.awid}, addr=0x{self.awaddr:016X}, "
            f"len={self.burst_length}, size={self.awsize.bytes}B)"
        )


# =============================================================================
# Write Data Channel (W)
# =============================================================================

@dataclass
class AXI_W:
    """
    AXI Write Data Channel.

    Attributes:
        wdata: Write data.
        wstrb: Write strobes (byte enable mask).
        wlast: Last transfer in burst.
        wuser: User-defined signal.
    """
    wdata: bytes
    wstrb: int = 0xFF       # Default: all bytes valid
    wlast: bool = False
    wuser: int = 0

    @property
    def data_size(self) -> int:
        """Data size in bytes."""
        return len(self.wdata)

    def __repr__(self) -> str:
        last_str = "LAST" if self.wlast else ""
        return f"AXI_W({self.data_size}B, strb=0x{self.wstrb:02X} {last_str})"


# =============================================================================
# Write Response Channel (B)
# =============================================================================

@dataclass
class AXI_B:
    """
    AXI Write Response Channel.

    Attributes:
        bid: Response ID (matches awid).
        bresp: Write response.
        buser: User-defined signal.
    """
    bid: int
    bresp: AXIResp = AXIResp.OKAY
    buser: int = 0

    @property
    def is_okay(self) -> bool:
        """Check if response is OKAY."""
        return self.bresp in (AXIResp.OKAY, AXIResp.EXOKAY)

    @property
    def is_error(self) -> bool:
        """Check if response is error."""
        return self.bresp in (AXIResp.SLVERR, AXIResp.DECERR)

    def __repr__(self) -> str:
        return f"AXI_B(id={self.bid}, resp={self.bresp.name})"


# =============================================================================
# Read Address Channel (AR)
# =============================================================================

@dataclass
class AXI_AR:
    """
    AXI Read Address Channel.

    Attributes:
        arid: Read address ID (transaction identifier).
        araddr: Read address (64-bit).
        arlen: Burst length (0-255 → 1-256 beats).
        arsize: Burst size (bytes per beat).
        arburst: Burst type.
        arlock: Lock type.
        arcache: Memory type.
        arprot: Protection type.
        arqos: Quality of Service.
        arregion: Region identifier.
        aruser: User-defined signal.
    """
    arid: int
    araddr: int
    arlen: int = 0          # 0 = 1 beat
    arsize: AXISize = AXISize.SIZE_8
    arburst: AXIBurst = AXIBurst.INCR
    arlock: int = 0
    arcache: int = 0
    arprot: int = 0
    arqos: int = 0
    arregion: int = 0
    aruser: int = 0

    @property
    def burst_length(self) -> int:
        """Actual burst length (1-256)."""
        return self.arlen + 1

    @property
    def transfer_size(self) -> int:
        """Total transfer size in bytes."""
        return self.burst_length * self.arsize.bytes

    def __repr__(self) -> str:
        return (
            f"AXI_AR(id={self.arid}, addr=0x{self.araddr:016X}, "
            f"len={self.burst_length}, size={self.arsize.bytes}B)"
        )


# =============================================================================
# Read Data Channel (R)
# =============================================================================

@dataclass
class AXI_R:
    """
    AXI Read Data Channel.

    Attributes:
        rid: Read ID (matches arid).
        rdata: Read data.
        rresp: Read response.
        rlast: Last transfer in burst.
        ruser: User-defined signal.
    """
    rid: int
    rdata: bytes
    rresp: AXIResp = AXIResp.OKAY
    rlast: bool = False
    ruser: int = 0

    @property
    def data_size(self) -> int:
        """Data size in bytes."""
        return len(self.rdata)

    @property
    def is_okay(self) -> bool:
        """Check if response is OKAY."""
        return self.rresp in (AXIResp.OKAY, AXIResp.EXOKAY)

    def __repr__(self) -> str:
        last_str = "LAST" if self.rlast else ""
        return f"AXI_R(id={self.rid}, {self.data_size}B, {self.rresp.name} {last_str})"


# =============================================================================
# AXI Transaction (combines channels)
# =============================================================================

@dataclass
class AXIWriteTransaction:
    """
    Complete AXI Write Transaction.

    Combines AW + W[] + B for a complete write operation.
    """
    aw: AXI_AW
    w_beats: List[AXI_W] = field(default_factory=list)
    b: Optional[AXI_B] = None
    timestamp_start: int = 0
    timestamp_end: int = 0

    @property
    def is_complete(self) -> bool:
        """Check if transaction is complete (has response)."""
        return self.b is not None

    @property
    def total_data(self) -> bytes:
        """Get all write data concatenated."""
        return b"".join(w.wdata for w in self.w_beats)

    @property
    def latency(self) -> int:
        """Transaction latency."""
        if self.timestamp_end == 0:
            return 0
        return self.timestamp_end - self.timestamp_start


@dataclass
class AXIReadTransaction:
    """
    Complete AXI Read Transaction.

    Combines AR + R[] for a complete read operation.
    """
    ar: AXI_AR
    r_beats: List[AXI_R] = field(default_factory=list)
    timestamp_start: int = 0
    timestamp_end: int = 0

    @property
    def is_complete(self) -> bool:
        """Check if transaction is complete (has all R beats)."""
        if not self.r_beats:
            return False
        return self.r_beats[-1].rlast

    @property
    def total_data(self) -> bytes:
        """Get all read data concatenated."""
        return b"".join(r.rdata for r in self.r_beats)

    @property
    def latency(self) -> int:
        """Transaction latency."""
        if self.timestamp_end == 0:
            return 0
        return self.timestamp_end - self.timestamp_start


# =============================================================================
# Helper functions
# =============================================================================

def create_write_transaction(
    addr: int,
    data: bytes,
    axi_id: int = 0,
    burst_size: AXISize = AXISize.SIZE_8,
) -> AXIWriteTransaction:
    """
    Create a complete write transaction.

    Args:
        addr: Target address.
        data: Data to write.
        axi_id: Transaction ID.
        burst_size: Bytes per beat.

    Returns:
        AXIWriteTransaction with AW and W beats populated.
    """
    beat_size = burst_size.bytes
    num_beats = (len(data) + beat_size - 1) // beat_size

    aw = AXI_AW(
        awid=axi_id,
        awaddr=addr,
        awlen=num_beats - 1,
        awsize=burst_size,
        awburst=AXIBurst.INCR,
    )

    w_beats = []
    for i in range(num_beats):
        start = i * beat_size
        end = min(start + beat_size, len(data))
        beat_data = data[start:end]
        # Pad if needed
        if len(beat_data) < beat_size:
            beat_data = beat_data + bytes(beat_size - len(beat_data))

        w = AXI_W(
            wdata=beat_data,
            wstrb=(1 << len(data[start:end])) - 1,
            wlast=(i == num_beats - 1),
        )
        w_beats.append(w)

    return AXIWriteTransaction(aw=aw, w_beats=w_beats)


def create_read_transaction(
    addr: int,
    size: int,
    axi_id: int = 0,
    burst_size: AXISize = AXISize.SIZE_8,
) -> AXIReadTransaction:
    """
    Create a read transaction (without response data).

    Args:
        addr: Target address.
        size: Total bytes to read.
        axi_id: Transaction ID.
        burst_size: Bytes per beat.

    Returns:
        AXIReadTransaction with AR populated.
    """
    beat_size = burst_size.bytes
    num_beats = (size + beat_size - 1) // beat_size

    ar = AXI_AR(
        arid=axi_id,
        araddr=addr,
        arlen=num_beats - 1,
        arsize=burst_size,
        arburst=AXIBurst.INCR,
    )

    return AXIReadTransaction(ar=ar)
