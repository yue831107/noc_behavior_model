"""
Flit (Flow Control Unit) data structure.

Flit is the basic unit of data transfer in NoC. A packet consists of
multiple flits: HEAD (routing info), BODY (data), TAIL (end marker).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import Optional
import struct


class FlitType(Enum):
    """Flit type enumeration."""
    HEAD = auto()       # Packet header with routing info
    BODY = auto()       # Packet body with data
    TAIL = auto()       # Packet tail (last flit)
    HEAD_TAIL = auto()  # Single-flit packet (header + tail)


@dataclass
class Flit:
    """
    Flit data structure.

    Attributes:
        flit_type: Type of flit (HEAD/BODY/TAIL/HEAD_TAIL).
        src: Source coordinate (x, y).
        dest: Destination coordinate (x, y).
        src_ni_id: Source NI ID (for V2 multi-NI, used in response routing).
        vc_id: Virtual Channel ID (reserved, currently unused).
        packet_id: Packet identifier for tracking.
        seq_num: Sequence number within packet.
        payload: Data payload (bytes).
        timestamp: Creation timestamp (simulation time).
    """
    flit_type: FlitType
    src: tuple[int, int]
    dest: tuple[int, int]
    src_ni_id: int = 0
    vc_id: int = 0
    packet_id: int = 0
    seq_num: int = 0
    payload: bytes = field(default_factory=bytes)
    timestamp: int = 0

    # For response routing
    is_request: bool = True  # True = Request, False = Response

    def is_head(self) -> bool:
        """Check if this flit is a header."""
        return self.flit_type in (FlitType.HEAD, FlitType.HEAD_TAIL)

    def is_tail(self) -> bool:
        """Check if this flit is a tail."""
        return self.flit_type in (FlitType.TAIL, FlitType.HEAD_TAIL)

    def is_single_flit(self) -> bool:
        """Check if this is a single-flit packet."""
        return self.flit_type == FlitType.HEAD_TAIL

    @property
    def payload_size(self) -> int:
        """Get payload size in bytes."""
        return len(self.payload)

    def __repr__(self) -> str:
        req_resp = "REQ" if self.is_request else "RSP"
        return (
            f"Flit({self.flit_type.name}, "
            f"pkt={self.packet_id}, seq={self.seq_num}, "
            f"{self.src}→{self.dest}, {req_resp})"
        )


@dataclass
class FlitHeader:
    """
    Flit header fields for HEAD/HEAD_TAIL flits.

    This represents the routing and control information
    embedded in the head flit of a packet.
    """
    src_x: int
    src_y: int
    dest_x: int
    dest_y: int
    src_ni_id: int
    packet_id: int
    packet_length: int  # Total flits in packet
    is_request: bool
    # AXI-related fields
    axi_id: int = 0
    local_addr: int = 0  # 32-bit local address
    burst_len: int = 0   # AXI burst length

    # V2 Smart Crossbar fields
    entry_edge_router: int = 0  # Edge router used for entry (V2)

    def to_bytes(self) -> bytes:
        """Serialize header to bytes."""
        flags = (1 if self.is_request else 0)
        return struct.pack(
            "<BBBBBBHHBIII",
            self.src_x,
            self.src_y,
            self.dest_x,
            self.dest_y,
            self.src_ni_id,
            self.packet_length,
            self.packet_id,
            self.axi_id,
            flags,
            self.local_addr,
            self.burst_len,
            0  # reserved
        )

    @classmethod
    def from_bytes(cls, data: bytes) -> FlitHeader:
        """Deserialize header from bytes."""
        unpacked = struct.unpack("<BBBBBBHHBIII", data[:23])
        return cls(
            src_x=unpacked[0],
            src_y=unpacked[1],
            dest_x=unpacked[2],
            dest_y=unpacked[3],
            src_ni_id=unpacked[4],
            packet_length=unpacked[5],
            packet_id=unpacked[6],
            axi_id=unpacked[7],
            is_request=bool(unpacked[8] & 1),
            local_addr=unpacked[9],
            burst_len=unpacked[10],
        )


class FlitFactory:
    """Factory for creating flits."""

    _packet_id_counter: int = 0

    @classmethod
    def _next_packet_id(cls) -> int:
        """Generate next packet ID."""
        cls._packet_id_counter += 1
        return cls._packet_id_counter

    @classmethod
    def reset_packet_id(cls) -> None:
        """Reset packet ID counter (for testing)."""
        cls._packet_id_counter = 0

    @classmethod
    def create_head(
        cls,
        src: tuple[int, int],
        dest: tuple[int, int],
        packet_id: Optional[int] = None,
        src_ni_id: int = 0,
        is_request: bool = True,
        payload: bytes = b"",
        timestamp: int = 0,
    ) -> Flit:
        """Create a HEAD flit."""
        if packet_id is None:
            packet_id = cls._next_packet_id()
        return Flit(
            flit_type=FlitType.HEAD,
            src=src,
            dest=dest,
            src_ni_id=src_ni_id,
            packet_id=packet_id,
            seq_num=0,
            is_request=is_request,
            payload=payload,
            timestamp=timestamp,
        )

    @classmethod
    def create_body(
        cls,
        src: tuple[int, int],
        dest: tuple[int, int],
        packet_id: int,
        seq_num: int,
        is_request: bool = True,
        payload: bytes = b"",
        timestamp: int = 0,
    ) -> Flit:
        """Create a BODY flit."""
        return Flit(
            flit_type=FlitType.BODY,
            src=src,
            dest=dest,
            packet_id=packet_id,
            seq_num=seq_num,
            is_request=is_request,
            payload=payload,
            timestamp=timestamp,
        )

    @classmethod
    def create_tail(
        cls,
        src: tuple[int, int],
        dest: tuple[int, int],
        packet_id: int,
        seq_num: int,
        is_request: bool = True,
        payload: bytes = b"",
        timestamp: int = 0,
    ) -> Flit:
        """Create a TAIL flit."""
        return Flit(
            flit_type=FlitType.TAIL,
            src=src,
            dest=dest,
            packet_id=packet_id,
            seq_num=seq_num,
            is_request=is_request,
            payload=payload,
            timestamp=timestamp,
        )

    @classmethod
    def create_single(
        cls,
        src: tuple[int, int],
        dest: tuple[int, int],
        packet_id: Optional[int] = None,
        src_ni_id: int = 0,
        is_request: bool = True,
        payload: bytes = b"",
        timestamp: int = 0,
    ) -> Flit:
        """Create a HEAD_TAIL (single flit) packet."""
        if packet_id is None:
            packet_id = cls._next_packet_id()
        return Flit(
            flit_type=FlitType.HEAD_TAIL,
            src=src,
            dest=dest,
            src_ni_id=src_ni_id,
            packet_id=packet_id,
            seq_num=0,
            is_request=is_request,
            payload=payload,
            timestamp=timestamp,
        )


def create_response_flit(request_flit: Flit, payload: bytes = b"") -> Flit:
    """
    Create a response flit from a request flit.

    Swaps src and dest, sets is_request=False.
    """
    return Flit(
        flit_type=request_flit.flit_type,
        src=request_flit.dest,      # Response comes from original dest
        dest=request_flit.src,      # Response goes to original src
        src_ni_id=request_flit.src_ni_id,
        vc_id=request_flit.vc_id,
        packet_id=request_flit.packet_id,
        seq_num=request_flit.seq_num,
        payload=payload,
        timestamp=0,  # Will be set by simulation
        is_request=False,
    )
