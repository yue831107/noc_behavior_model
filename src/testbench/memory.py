"""
Memory Models for NoC simulation.

Provides Host Memory and Local Memory models for memory copy scenarios.
Supports file I/O for File-In/File-Out hardware verification.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, List, Tuple
from pathlib import Path
import random


@dataclass
class MemoryConfig:
    """Memory configuration."""
    size: int = 0x100000000      # Memory size in bytes (default 4GB = 32-bit address space)
    latency_read: int = 1          # Read latency in cycles
    latency_write: int = 1         # Write latency in cycles
    bus_width: int = 8             # Bus width in bytes (64-bit)


class Memory:
    """
    Simple memory model.

    Supports read/write operations with configurable latency.
    """

    def __init__(
        self,
        config: Optional[MemoryConfig] = None,
        name: str = ""
    ):
        """
        Initialize memory.

        Args:
            config: Memory configuration.
            name: Memory name for identification.
        """
        self.config = config or MemoryConfig()
        self.name = name or "Memory"

        # Memory storage (sparse - only store written data)
        self._data: Dict[int, int] = {}  # address -> byte value

        # Statistics
        self.stats = MemoryStats()

    def write(self, address: int, data: bytes) -> int:
        """
        Write data to memory.

        Args:
            address: Start address.
            data: Data to write.

        Returns:
            Latency in cycles.
        """
        if address < 0 or address + len(data) > self.config.size:
            raise ValueError(f"Address out of range: {address}")

        for i, byte in enumerate(data):
            self._data[address + i] = byte

        self.stats.writes += 1
        self.stats.bytes_written += len(data)

        return self.config.latency_write

    def read(self, address: int, size: int) -> Tuple[bytes, int]:
        """
        Read data from memory.

        Args:
            address: Start address.
            size: Number of bytes to read.

        Returns:
            Tuple of (data bytes, latency in cycles).
        """
        if address < 0 or address + size > self.config.size:
            raise ValueError(f"Address out of range: {address}")

        data = bytearray(size)
        for i in range(size):
            data[i] = self._data.get(address + i, 0)

        self.stats.reads += 1
        self.stats.bytes_read += size

        return bytes(data), self.config.latency_read

    def fill(self, address: int, size: int, pattern: bytes = None) -> None:
        """
        Fill memory region with pattern.

        Args:
            address: Start address.
            size: Number of bytes to fill.
            pattern: Fill pattern (default: sequential bytes).
        """
        if pattern is None:
            # Sequential pattern
            for i in range(size):
                self._data[address + i] = i & 0xFF
        else:
            # Repeat pattern
            for i in range(size):
                self._data[address + i] = pattern[i % len(pattern)]

    def fill_random(self, address: int, size: int, seed: int = None) -> None:
        """
        Fill memory region with random data.

        Args:
            address: Start address.
            size: Number of bytes to fill.
            seed: Random seed.
        """
        rng = random.Random(seed)
        for i in range(size):
            self._data[address + i] = rng.randint(0, 255)

    def clear(self) -> None:
        """Clear all memory contents."""
        self._data.clear()
        self.stats = MemoryStats()

    def verify(self, address: int, expected: bytes) -> bool:
        """
        Verify memory contents.

        Args:
            address: Start address.
            expected: Expected data.

        Returns:
            True if memory matches expected data.
        """
        actual, _ = self.read(address, len(expected))
        return actual == expected

    def get_contents(self, address: int, size: int) -> bytes:
        """Get memory contents without updating stats."""
        data = bytearray(size)
        for i in range(size):
            data[i] = self._data.get(address + i, 0)
        return bytes(data)

    @property
    def used_bytes(self) -> int:
        """Number of bytes with data."""
        return len(self._data)

    # =========================================================================
    # File I/O Methods
    # =========================================================================

    def load_from_file(self, path: str | Path, address: int = 0) -> int:
        """
        Load binary data from file into memory.

        Args:
            path: Path to binary file.
            address: Start address to load data.

        Returns:
            Number of bytes loaded.

        Raises:
            FileNotFoundError: If file does not exist.
            ValueError: If data exceeds memory size.
        """
        path = Path(path)
        if not path.exists():
            raise FileNotFoundError(f"File not found: {path}")

        data = path.read_bytes()

        if address + len(data) > self.config.size:
            raise ValueError(
                f"Data size ({len(data)}) exceeds available memory "
                f"(address={address}, mem_size={self.config.size})"
            )

        for i, byte in enumerate(data):
            self._data[address + i] = byte

        return len(data)

    def dump_to_file(
        self,
        path: str | Path,
        address: int = 0,
        size: Optional[int] = None,
    ) -> int:
        """
        Dump memory contents to binary file.

        Args:
            path: Output file path.
            address: Start address to dump from.
            size: Number of bytes to dump (None = entire memory).

        Returns:
            Number of bytes written.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if size is None:
            size = self.config.size

        # Clamp to valid range
        if address + size > self.config.size:
            size = self.config.size - address

        data = self.get_contents(address, size)
        path.write_bytes(data)

        return len(data)

    def dump_to_hex(
        self,
        path: str | Path,
        address: int = 0,
        size: Optional[int] = None,
        bytes_per_line: int = 16,
    ) -> None:
        """
        Dump memory contents to hex text file.

        Args:
            path: Output file path.
            address: Start address to dump from.
            size: Number of bytes to dump.
            bytes_per_line: Bytes per line in output.
        """
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        if size is None:
            size = self.config.size

        data = self.get_contents(address, size)
        lines = []

        for offset in range(0, len(data), bytes_per_line):
            chunk = data[offset:offset + bytes_per_line]
            hex_part = " ".join(f"{b:02X}" for b in chunk)
            ascii_part = "".join(
                chr(b) if 32 <= b < 127 else "." for b in chunk
            )
            abs_addr = address + offset
            lines.append(f"{abs_addr:08X}  {hex_part:<{bytes_per_line*3}}  {ascii_part}")

        path.write_text("\n".join(lines) + "\n")

    def dump_used_regions(self, path: str | Path) -> int:
        """
        Dump only regions with data (sparse dump).

        Args:
            path: Output file path.

        Returns:
            Number of bytes written.
        """
        if not self._data:
            return 0

        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        # Find min/max addresses with data
        min_addr = min(self._data.keys())
        max_addr = max(self._data.keys())
        size = max_addr - min_addr + 1

        data = self.get_contents(min_addr, size)
        path.write_bytes(data)

        return len(data)


@dataclass
class MemoryStats:
    """Memory statistics."""
    reads: int = 0
    writes: int = 0
    bytes_read: int = 0
    bytes_written: int = 0


class HostMemory(Memory):
    """
    Host Memory model (e.g., DRAM).

    Represents the source memory for memory copy operations.
    """

    def __init__(
        self,
        size: int = 1024 * 1024,    # 1MB default
        latency_read: int = 10,      # Higher latency for DRAM
        latency_write: int = 10,
    ):
        config = MemoryConfig(
            size=size,
            latency_read=latency_read,
            latency_write=latency_write,
        )
        super().__init__(config, "HostMemory")


class LocalMemory(Memory):
    """
    Local Memory model (e.g., SRAM).

    Represents the destination memory at each compute node.
    Uses sparse storage to support full 32-bit address space.
    """

    def __init__(
        self,
        node_id: int,
        size: int = 0x100000000,   # 4GB default (32-bit address space)
        latency_read: int = 1,       # Lower latency for SRAM
        latency_write: int = 1,
    ):
        config = MemoryConfig(
            size=size,
            latency_read=latency_read,
            latency_write=latency_write,
        )
        super().__init__(config, f"LocalMemory_{node_id}")
        self.node_id = node_id


@dataclass
class MemoryCopyDescriptor:
    """
    Describes a memory copy operation.

    Used to track memory copy progress and completion.
    """
    src_addr: int               # Source address (Host Memory)
    dst_node: int               # Destination node ID
    dst_addr: int               # Destination address (Local Memory)
    size: int                   # Copy size in bytes
    block_size: int = 64        # Transfer block size

    # Progress tracking
    bytes_sent: int = 0
    bytes_acked: int = 0
    start_cycle: int = 0
    end_cycle: int = 0

    @property
    def is_complete(self) -> bool:
        """Check if copy is complete."""
        return self.bytes_acked >= self.size

    @property
    def progress(self) -> float:
        """Get progress percentage."""
        return (self.bytes_acked / self.size * 100) if self.size > 0 else 100.0

    @property
    def latency(self) -> int:
        """Get total latency in cycles."""
        if self.end_cycle > 0:
            return self.end_cycle - self.start_cycle
        return 0
