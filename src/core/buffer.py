"""
Buffer and Credit-based Flow Control.

Implements FIFO buffers for router ports and credit-based
flow control mechanism for Virtual Cut-Through switching.
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, List, Deque, Generic, TypeVar
from collections import deque

from .flit import Flit


T = TypeVar('T')


@dataclass
class Link:
    """
    Single-direction valid/ready/flit connection point.

    Represents a unidirectional link in the valid/ready handshake protocol.
    A successful transfer occurs when valid=True and ready=True simultaneously.

    Attributes:
        valid: Upstream asserts to indicate flit is available.
        ready: Downstream asserts to indicate buffer space is available.
        flit: The flit being transferred (valid when valid=True).
    """
    valid: bool = False
    ready: bool = False
    flit: Optional[Flit] = None

    def handshake(self) -> bool:
        """Check if transfer can occur (valid && ready)."""
        return self.valid and self.ready

    def clear(self) -> None:
        """Clear the link signals."""
        self.valid = False
        self.ready = False
        self.flit = None

    def set_valid(self, flit: Flit) -> None:
        """Set valid with flit data."""
        self.valid = True
        self.flit = flit

    def clear_valid(self) -> None:
        """Clear valid and flit."""
        self.valid = False
        self.flit = None


@dataclass
class BufferStats:
    """Statistics for buffer usage."""
    total_writes: int = 0
    total_reads: int = 0
    max_occupancy: int = 0
    total_occupancy_samples: int = 0
    cumulative_occupancy: int = 0

    @property
    def avg_occupancy(self) -> float:
        """Average buffer occupancy."""
        if self.total_occupancy_samples == 0:
            return 0.0
        return self.cumulative_occupancy / self.total_occupancy_samples

    def sample(self, current_occupancy: int) -> None:
        """Record a sample of current occupancy."""
        self.total_occupancy_samples += 1
        self.cumulative_occupancy += current_occupancy
        if current_occupancy > self.max_occupancy:
            self.max_occupancy = current_occupancy


class Buffer(Generic[T]):
    """
    Generic FIFO buffer.

    Attributes:
        depth: Maximum number of items the buffer can hold.
        name: Optional name for debugging.
    """

    def __init__(self, depth: int, name: str = ""):
        """
        Initialize buffer.

        Args:
            depth: Maximum buffer capacity.
            name: Optional name for identification.
        """
        self.depth = depth
        self.name = name
        self._queue: Deque[T] = deque(maxlen=depth)
        self.stats = BufferStats()

    def push(self, item: T) -> bool:
        """
        Push an item to the buffer.

        Args:
            item: Item to push.

        Returns:
            True if successful, False if buffer is full.
        """
        if self.is_full():
            return False
        self._queue.append(item)
        self.stats.total_writes += 1
        return True

    def pop(self) -> Optional[T]:
        """
        Pop an item from the buffer.

        Returns:
            The item, or None if buffer is empty.
        """
        if self.is_empty():
            return None
        item = self._queue.popleft()
        self.stats.total_reads += 1
        return item

    def peek(self) -> Optional[T]:
        """
        Peek at the front item without removing it.

        Returns:
            The front item, or None if buffer is empty.
        """
        if self.is_empty():
            return None
        return self._queue[0]

    def is_empty(self) -> bool:
        """Check if buffer is empty."""
        return len(self._queue) == 0

    def is_full(self) -> bool:
        """Check if buffer is full."""
        return len(self._queue) >= self.depth

    @property
    def occupancy(self) -> int:
        """Current number of items in buffer."""
        return len(self._queue)

    @property
    def free_space(self) -> int:
        """Available space in buffer."""
        return self.depth - len(self._queue)

    def sample_stats(self) -> None:
        """Sample current occupancy for statistics."""
        self.stats.sample(self.occupancy)

    def clear(self) -> None:
        """Clear all items from buffer."""
        self._queue.clear()

    def __len__(self) -> int:
        return len(self._queue)

    def __repr__(self) -> str:
        return f"Buffer({self.name}, {self.occupancy}/{self.depth})"


class FlitBuffer(Buffer[Flit]):
    """
    Specialized buffer for Flits.

    Adds flit-specific functionality like packet tracking.
    """

    def __init__(self, depth: int, name: str = ""):
        super().__init__(depth, name)
        self._packet_flits: dict[int, int] = {}  # packet_id -> flit count

    def push(self, flit: Flit) -> bool:
        """Push a flit to the buffer."""
        if not super().push(flit):
            return False
        # Track packets
        self._packet_flits[flit.packet_id] = (
            self._packet_flits.get(flit.packet_id, 0) + 1
        )
        return True

    def pop(self) -> Optional[Flit]:
        """Pop a flit from the buffer."""
        flit = super().pop()
        if flit is not None:
            self._packet_flits[flit.packet_id] -= 1
            if self._packet_flits[flit.packet_id] == 0:
                del self._packet_flits[flit.packet_id]
        return flit

    def has_complete_packet(self) -> bool:
        """
        Check if buffer contains at least one complete packet.

        A complete packet has HEAD...TAIL sequence.
        """
        if self.is_empty():
            return False

        # Check if first flit is HEAD and we have its TAIL
        first = self.peek()
        if first is None:
            return False

        if first.is_single_flit():
            return True

        if not first.is_head():
            # Corrupted state - first flit should be HEAD
            return False

        # Look for TAIL with same packet_id
        for flit in self._queue:
            if flit.packet_id == first.packet_id and flit.is_tail():
                return True

        return False

    def get_packet_flit_count(self, packet_id: int) -> int:
        """Get number of flits for a specific packet in buffer."""
        return self._packet_flits.get(packet_id, 0)


@dataclass
class CreditFlowControl:
    """
    Credit-based flow control.

    Manages credits for downstream buffer availability.
    Used in Virtual Cut-Through switching.

    Attributes:
        initial_credits: Initial credit count (= downstream buffer depth).
        credits: Current available credits.
    """
    initial_credits: int
    credits: int = field(init=False)

    def __post_init__(self):
        self.credits = self.initial_credits

    def can_send(self, count: int = 1) -> bool:
        """
        Check if we have enough credits to send.

        Args:
            count: Number of flits to send.

        Returns:
            True if enough credits available.
        """
        return self.credits >= count

    def consume(self, count: int = 1) -> bool:
        """
        Consume credits when sending flits.

        Args:
            count: Number of credits to consume.

        Returns:
            True if successful, False if insufficient credits.
        """
        if not self.can_send(count):
            return False
        self.credits -= count
        return True

    def release(self, count: int = 1) -> None:
        """
        Release credits (called when downstream frees buffer space).

        Args:
            count: Number of credits to release.
        """
        self.credits = min(self.credits + count, self.initial_credits)

    def reset(self) -> None:
        """Reset credits to initial value."""
        self.credits = self.initial_credits

    @property
    def available(self) -> int:
        """Available credits."""
        return self.credits

    @property
    def utilization(self) -> float:
        """Credit utilization (1.0 = all credits consumed)."""
        if self.initial_credits == 0:
            return 0.0
        return 1.0 - (self.credits / self.initial_credits)


class PortBuffer:
    """
    Buffer for a router port with credit flow control.

    Combines input buffer with credit tracking for the downstream.
    """

    def __init__(
        self,
        depth: int,
        name: str = "",
        downstream_depth: Optional[int] = None
    ):
        """
        Initialize port buffer.

        Args:
            depth: Input buffer depth.
            name: Port name for identification.
            downstream_depth: Downstream buffer depth for credit init.
                            If None, uses same as depth.
        """
        self.name = name
        self.input_buffer = FlitBuffer(depth, f"{name}_in")
        self.credit = CreditFlowControl(
            initial_credits=downstream_depth or depth
        )

    def can_accept(self) -> bool:
        """Check if port can accept incoming flit."""
        return not self.input_buffer.is_full()

    def can_send(self, packet_length: int = 1) -> bool:
        """
        Check if port can send (has credits for Virtual Cut-Through).

        For VCT, we need credits for the entire packet.
        """
        return self.credit.can_send(packet_length)

    def receive(self, flit: Flit) -> bool:
        """
        Receive a flit into input buffer.

        Args:
            flit: Flit to receive.

        Returns:
            True if accepted, False if buffer full.
        """
        return self.input_buffer.push(flit)

    def send(self) -> Optional[Flit]:
        """
        Send a flit (pop from buffer, consume credit).

        Returns:
            The flit, or None if buffer empty or no credits.
        """
        flit = self.input_buffer.peek()
        if flit is None:
            return None

        if not self.credit.consume(1):
            return None

        return self.input_buffer.pop()

    def release_credit(self) -> None:
        """Release one credit (called by upstream when we free space)."""
        self.credit.release(1)

    @property
    def occupancy(self) -> int:
        """Current buffer occupancy."""
        return self.input_buffer.occupancy

    @property
    def free_space(self) -> int:
        """Available buffer space."""
        return self.input_buffer.free_space

    def sample_stats(self) -> None:
        """Sample statistics."""
        self.input_buffer.sample_stats()

    def __repr__(self) -> str:
        return (
            f"PortBuffer({self.name}, "
            f"buf={self.input_buffer.occupancy}/{self.input_buffer.depth}, "
            f"credit={self.credit.credits}/{self.credit.initial_credits})"
        )
