"""
Packet encapsulation and flit assembly/disassembly.

A Packet represents a complete NoC transaction, consisting of
multiple Flits (HEAD + BODY* + TAIL or single HEAD_TAIL).
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Iterator
from enum import Enum, auto
import struct

from .flit import Flit, FlitType, FlitFactory


# Packet header format (12 bytes):
# - packet_type: 1 byte
# - axi_id: 1 byte
# - reserved: 2 bytes
# - local_addr: 4 bytes
# - payload_len: 4 bytes
PACKET_HEADER_FORMAT = "<BBHI I"
PACKET_HEADER_SIZE = 12


class PacketType(Enum):
    """Packet type based on AXI transaction."""
    WRITE_REQ = auto()   # AXI Write Request (AW + W)
    WRITE_RESP = auto()  # AXI Write Response (B)
    READ_REQ = auto()    # AXI Read Request (AR)
    READ_RESP = auto()   # AXI Read Response (R)


@dataclass
class Packet:
    """
    Packet data structure.

    A packet is a complete unit of NoC transaction, composed of
    one or more flits.

    Attributes:
        packet_id: Unique packet identifier.
        packet_type: Type of packet (WRITE_REQ, READ_REQ, etc.).
        src: Source coordinate (x, y).
        dest: Destination coordinate (x, y).
        src_ni_id: Source NI ID (for multi-NI routing).
        axi_id: AXI transaction ID.
        local_addr: 32-bit local address at destination.
        payload: Complete payload data.
        timestamp: Creation timestamp.
    """
    packet_id: int
    packet_type: PacketType
    src: tuple[int, int]
    dest: tuple[int, int]
    src_ni_id: int = 0
    axi_id: int = 0
    local_addr: int = 0
    payload: bytes = field(default_factory=bytes)
    timestamp: int = 0
    read_size: int = 0  # For READ_REQ: number of bytes to read

    # Tracking
    flits: List[Flit] = field(default_factory=list)
    _assembled: bool = field(default=False, repr=False)

    @property
    def is_request(self) -> bool:
        """Check if this is a request packet."""
        return self.packet_type in (PacketType.WRITE_REQ, PacketType.READ_REQ)

    def serialize_header(self) -> bytes:
        """Serialize packet header to bytes."""
        # For READ_REQ, store read_size in the reserved field (16-bit)
        reserved = self.read_size & 0xFFFF if self.packet_type == PacketType.READ_REQ else 0
        return struct.pack(
            PACKET_HEADER_FORMAT,
            self.packet_type.value,
            self.axi_id & 0xFF,
            reserved,  # Contains read_size for READ_REQ
            self.local_addr,
            len(self.payload),
        )

    @classmethod
    def deserialize_header(cls, data: bytes) -> tuple:
        """
        Deserialize packet header from bytes.

        Returns:
            Tuple of (packet_type, axi_id, local_addr, payload_len, read_size).
        """
        unpacked = struct.unpack(PACKET_HEADER_FORMAT, data[:PACKET_HEADER_SIZE])
        packet_type = PacketType(unpacked[0])
        axi_id = unpacked[1]
        reserved = unpacked[2]  # Contains read_size for READ_REQ
        local_addr = unpacked[3]
        payload_len = unpacked[4]
        read_size = reserved if packet_type == PacketType.READ_REQ else 0
        return packet_type, axi_id, local_addr, payload_len, read_size

    def get_serialized_payload(self) -> bytes:
        """Get payload with header prepended."""
        return self.serialize_header() + self.payload

    @property
    def is_response(self) -> bool:
        """Check if this is a response packet."""
        return self.packet_type in (PacketType.WRITE_RESP, PacketType.READ_RESP)

    @property
    def flit_count(self) -> int:
        """Number of flits in this packet."""
        return len(self.flits)

    @property
    def payload_size(self) -> int:
        """Total payload size in bytes."""
        return len(self.payload)

    def __repr__(self) -> str:
        return (
            f"Packet(id={self.packet_id}, {self.packet_type.name}, "
            f"{self.src}→{self.dest}, flits={self.flit_count})"
        )


class PacketAssembler:
    """
    Assembles a Packet into Flits for transmission.

    Handles splitting payload into multiple flits based on
    flit payload capacity.
    """

    def __init__(self, flit_payload_size: int = 8):
        """
        Initialize assembler.

        Args:
            flit_payload_size: Maximum payload bytes per flit.
        """
        self.flit_payload_size = flit_payload_size

    def assemble(self, packet: Packet, timestamp: int = 0) -> List[Flit]:
        """
        Assemble a packet into flits.

        The packet header (type, axi_id, local_addr, payload_len) is
        prepended to the payload before splitting into flits.

        Args:
            packet: Packet to assemble.
            timestamp: Current simulation timestamp.

        Returns:
            List of flits representing the packet.
        """
        if packet._assembled:
            return packet.flits

        flits = []
        # Include packet header in the serialized data
        payload = packet.get_serialized_payload()
        total_payload_size = len(payload)

        # Calculate number of flits needed
        if total_payload_size == 0:
            # Single flit packet (e.g., read request)
            flit = FlitFactory.create_single(
                src=packet.src,
                dest=packet.dest,
                packet_id=packet.packet_id,
                src_ni_id=packet.src_ni_id,
                is_request=packet.is_request,
                payload=b"",
                timestamp=timestamp,
            )
            flits.append(flit)
        elif total_payload_size <= self.flit_payload_size:
            # Single flit with payload
            flit = FlitFactory.create_single(
                src=packet.src,
                dest=packet.dest,
                packet_id=packet.packet_id,
                src_ni_id=packet.src_ni_id,
                is_request=packet.is_request,
                payload=payload,
                timestamp=timestamp,
            )
            flits.append(flit)
        else:
            # Multi-flit packet
            offset = 0
            seq_num = 0

            # HEAD flit
            head_payload = payload[offset:offset + self.flit_payload_size]
            offset += self.flit_payload_size
            head = FlitFactory.create_head(
                src=packet.src,
                dest=packet.dest,
                packet_id=packet.packet_id,
                src_ni_id=packet.src_ni_id,
                is_request=packet.is_request,
                payload=head_payload,
                timestamp=timestamp,
            )
            flits.append(head)
            seq_num += 1

            # BODY flits
            while offset + self.flit_payload_size < total_payload_size:
                body_payload = payload[offset:offset + self.flit_payload_size]
                offset += self.flit_payload_size
                body = FlitFactory.create_body(
                    src=packet.src,
                    dest=packet.dest,
                    packet_id=packet.packet_id,
                    seq_num=seq_num,
                    is_request=packet.is_request,
                    payload=body_payload,
                    timestamp=timestamp,
                )
                flits.append(body)
                seq_num += 1

            # TAIL flit
            tail_payload = payload[offset:]
            tail = FlitFactory.create_tail(
                src=packet.src,
                dest=packet.dest,
                packet_id=packet.packet_id,
                seq_num=seq_num,
                is_request=packet.is_request,
                payload=tail_payload,
                timestamp=timestamp,
            )
            flits.append(tail)

        packet.flits = flits
        packet._assembled = True
        return flits

    def calculate_flit_count(self, payload_size: int) -> int:
        """
        Calculate number of flits needed for given payload size.

        Args:
            payload_size: Payload size in bytes.

        Returns:
            Number of flits required.
        """
        if payload_size == 0:
            return 1
        if payload_size <= self.flit_payload_size:
            return 1
        # HEAD + BODY* + TAIL
        # HEAD and TAIL each carry flit_payload_size
        # Remaining goes into BODY flits
        return (payload_size + self.flit_payload_size - 1) // self.flit_payload_size


class PacketDisassembler:
    """
    Disassembles Flits back into a Packet.

    Collects flits with the same packet_id and reconstructs
    the original packet.
    """

    def __init__(self):
        """Initialize disassembler."""
        self._pending: dict[int, List[Flit]] = {}  # packet_id -> flits

    def receive_flit(self, flit: Flit) -> Optional[Packet]:
        """
        Receive a flit and attempt to reconstruct packet.

        Args:
            flit: Received flit.

        Returns:
            Reconstructed Packet if complete, None otherwise.
        """
        packet_id = flit.packet_id

        # Single flit packet
        if flit.is_single_flit():
            return self._create_packet_from_flits([flit])

        # Multi-flit packet
        if packet_id not in self._pending:
            if not flit.is_head():
                # Received non-HEAD as first flit - error state
                # In production, this would need error handling
                return None
            self._pending[packet_id] = []

        self._pending[packet_id].append(flit)

        # Check if packet is complete
        if flit.is_tail():
            flits = self._pending.pop(packet_id)
            return self._create_packet_from_flits(flits)

        return None

    def _create_packet_from_flits(self, flits: List[Flit]) -> Packet:
        """
        Create a Packet from a list of flits.

        The flit payload contains: [packet_header (12 bytes)][actual_payload]

        Args:
            flits: List of flits (HEAD...TAIL or HEAD_TAIL).

        Returns:
            Reconstructed Packet.
        """
        if not flits:
            raise ValueError("Cannot create packet from empty flit list")

        head = flits[0]

        # Reconstruct full serialized payload
        serialized = b"".join(f.payload for f in flits)

        # Parse packet header
        if len(serialized) >= PACKET_HEADER_SIZE:
            packet_type, axi_id, local_addr, payload_len, read_size = \
                Packet.deserialize_header(serialized)
            actual_payload = serialized[PACKET_HEADER_SIZE:]
        else:
            # Fallback for malformed packets
            if head.is_request:
                packet_type = PacketType.WRITE_REQ
            else:
                packet_type = PacketType.WRITE_RESP
            axi_id = 0
            local_addr = 0
            actual_payload = serialized
            read_size = 0

        packet = Packet(
            packet_id=head.packet_id,
            packet_type=packet_type,
            src=head.src,
            dest=head.dest,
            src_ni_id=head.src_ni_id,
            axi_id=axi_id,
            local_addr=local_addr,
            payload=actual_payload,
            timestamp=head.timestamp,
        )
        packet.read_size = read_size
        packet.flits = flits
        packet._assembled = True

        return packet

    def clear(self) -> None:
        """Clear all pending flits."""
        self._pending.clear()

    @property
    def pending_count(self) -> int:
        """Number of packets currently being assembled."""
        return len(self._pending)


class PacketFactory:
    """Factory for creating packets."""

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
    def create_write_request(
        cls,
        src: tuple[int, int],
        dest: tuple[int, int],
        local_addr: int,
        data: bytes,
        axi_id: int = 0,
        src_ni_id: int = 0,
        timestamp: int = 0,
    ) -> Packet:
        """Create a write request packet."""
        return Packet(
            packet_id=cls._next_packet_id(),
            packet_type=PacketType.WRITE_REQ,
            src=src,
            dest=dest,
            src_ni_id=src_ni_id,
            axi_id=axi_id,
            local_addr=local_addr,
            payload=data,
            timestamp=timestamp,
        )

    @classmethod
    def create_read_request(
        cls,
        src: tuple[int, int],
        dest: tuple[int, int],
        local_addr: int,
        read_size: int = 0,
        axi_id: int = 0,
        src_ni_id: int = 0,
        timestamp: int = 0,
    ) -> Packet:
        """Create a read request packet."""
        pkt = Packet(
            packet_id=cls._next_packet_id(),
            packet_type=PacketType.READ_REQ,
            src=src,
            dest=dest,
            src_ni_id=src_ni_id,
            axi_id=axi_id,
            local_addr=local_addr,
            payload=b"",  # Read request has no data payload
            timestamp=timestamp,
        )
        pkt.read_size = read_size
        return pkt

    @classmethod
    def create_write_response(
        cls,
        request: Packet,
        timestamp: int = 0,
    ) -> Packet:
        """Create a write response packet from a request."""
        return Packet(
            packet_id=cls._next_packet_id(),
            packet_type=PacketType.WRITE_RESP,
            src=request.dest,  # Response from original destination
            dest=request.src,  # Response to original source
            src_ni_id=request.src_ni_id,
            axi_id=request.axi_id,
            local_addr=request.local_addr,
            payload=b"",  # Write response has no data
            timestamp=timestamp,
        )

    @classmethod
    def create_read_response(
        cls,
        request: Packet,
        data: bytes,
        timestamp: int = 0,
    ) -> Packet:
        """Create a read response packet from a request."""
        return Packet(
            packet_id=cls._next_packet_id(),
            packet_type=PacketType.READ_RESP,
            src=request.dest,  # Response from original destination
            dest=request.src,  # Response to original source
            src_ni_id=request.src_ni_id,
            axi_id=request.axi_id,
            local_addr=request.local_addr,
            payload=data,
            timestamp=timestamp,
        )

    @classmethod
    def create_write_response_from_info(
        cls,
        src: Tuple[int, int],
        dest: Tuple[int, int],
        axi_id: int,
        timestamp: int = 0,
    ) -> Packet:
        """
        Create a write response packet from routing info.

        Used by MasterNI when it doesn't have the original request packet,
        only the routing info from Per-ID FIFO.

        Args:
            src: Source coordinate (Master NI location).
            dest: Destination coordinate (original Slave NI location).
            axi_id: AXI transaction ID.
            timestamp: Creation timestamp.

        Returns:
            Write response packet.
        """
        return Packet(
            packet_id=cls._next_packet_id(),
            packet_type=PacketType.WRITE_RESP,
            src=src,
            dest=dest,
            src_ni_id=0,
            axi_id=axi_id,
            local_addr=0,
            payload=b"",
            timestamp=timestamp,
        )

    @classmethod
    def create_read_response_from_info(
        cls,
        src: Tuple[int, int],
        dest: Tuple[int, int],
        axi_id: int,
        data: bytes,
        timestamp: int = 0,
    ) -> Packet:
        """
        Create a read response packet from routing info.

        Used by MasterNI when it doesn't have the original request packet,
        only the routing info from Per-ID FIFO.

        Args:
            src: Source coordinate (Master NI location).
            dest: Destination coordinate (original Slave NI location).
            axi_id: AXI transaction ID.
            data: Read data.
            timestamp: Creation timestamp.

        Returns:
            Read response packet.
        """
        return Packet(
            packet_id=cls._next_packet_id(),
            packet_type=PacketType.READ_RESP,
            src=src,
            dest=dest,
            src_ni_id=0,
            axi_id=axi_id,
            local_addr=0,
            payload=data,
            timestamp=timestamp,
        )
