"""
Tests for Buffer and Credit-based Flow Control.

Tests cover:
1. Link class - valid/ready handshake
2. BufferStats - statistics tracking
3. Buffer - generic FIFO buffer
4. FlitBuffer - flit-specific buffer with packet tracking
5. CreditFlowControl - credit-based flow control
6. PortBuffer - combined buffer with credits
"""

import pytest
from unittest.mock import Mock

from src.core.buffer import (
    Link,
    BufferStats,
    Buffer,
    FlitBuffer,
    CreditFlowControl,
    PortBuffer,
)
from src.core.flit import Flit, FlitHeader


class TestLink:
    """Test Link class for valid/ready handshake."""

    def test_initial_state(self):
        """Link should start with all signals clear."""
        link = Link()
        assert link.valid is False
        assert link.ready is False
        assert link.flit is None

    def test_handshake_success(self):
        """handshake should return True when valid && ready."""
        link = Link(valid=True, ready=True)
        assert link.handshake() is True

    def test_handshake_fail_no_valid(self):
        """handshake should return False without valid."""
        link = Link(valid=False, ready=True)
        assert link.handshake() is False

    def test_handshake_fail_no_ready(self):
        """handshake should return False without ready."""
        link = Link(valid=True, ready=False)
        assert link.handshake() is False

    def test_clear(self):
        """clear should reset all signals."""
        link = Link(valid=True, ready=True)
        link.flit = Mock()

        link.clear()

        assert link.valid is False
        assert link.ready is False
        assert link.flit is None

    def test_set_valid(self):
        """set_valid should set valid and flit."""
        link = Link()
        mock_flit = Mock()

        link.set_valid(mock_flit)

        assert link.valid is True
        assert link.flit is mock_flit

    def test_clear_valid(self):
        """clear_valid should clear valid and flit."""
        link = Link(valid=True)
        link.flit = Mock()

        link.clear_valid()

        assert link.valid is False
        assert link.flit is None


class TestBufferStats:
    """Test BufferStats class."""

    def test_initial_values(self):
        """Stats should start at zero."""
        stats = BufferStats()
        assert stats.total_writes == 0
        assert stats.total_reads == 0
        assert stats.max_occupancy == 0

    def test_avg_occupancy_zero_samples(self):
        """avg_occupancy should return 0 with no samples."""
        stats = BufferStats()
        assert stats.avg_occupancy == 0.0

    def test_avg_occupancy_with_samples(self):
        """avg_occupancy should calculate correctly."""
        stats = BufferStats()
        stats.sample(4)
        stats.sample(6)
        assert stats.avg_occupancy == 5.0

    def test_sample_updates_max(self):
        """sample should update max_occupancy."""
        stats = BufferStats()
        stats.sample(5)
        stats.sample(10)
        stats.sample(3)

        assert stats.max_occupancy == 10

    def test_sample_cumulative(self):
        """sample should track cumulative occupancy."""
        stats = BufferStats()
        stats.sample(2)
        stats.sample(3)
        stats.sample(4)

        assert stats.total_occupancy_samples == 3
        assert stats.cumulative_occupancy == 9


class TestBuffer:
    """Test generic Buffer class."""

    def test_init(self):
        """Buffer should initialize correctly."""
        buf = Buffer(depth=8, name="test_buf")
        assert buf.depth == 8
        assert buf.name == "test_buf"
        assert buf.occupancy == 0

    def test_push_and_pop(self):
        """push and pop should work correctly."""
        buf = Buffer(depth=4)

        assert buf.push("item1") is True
        assert buf.push("item2") is True

        assert buf.pop() == "item1"
        assert buf.pop() == "item2"

    def test_push_full_buffer(self):
        """push should return False when buffer is full."""
        buf = Buffer(depth=2)

        assert buf.push("item1") is True
        assert buf.push("item2") is True
        assert buf.push("item3") is False  # Full

    def test_pop_empty_buffer(self):
        """pop should return None when buffer is empty."""
        buf = Buffer(depth=4)
        assert buf.pop() is None

    def test_peek(self):
        """peek should return front item without removing."""
        buf = Buffer(depth=4)
        buf.push("item1")
        buf.push("item2")

        assert buf.peek() == "item1"
        assert buf.peek() == "item1"  # Still there
        assert buf.occupancy == 2

    def test_peek_empty(self):
        """peek should return None on empty buffer."""
        buf = Buffer(depth=4)
        assert buf.peek() is None

    def test_is_empty(self):
        """is_empty should check correctly."""
        buf = Buffer(depth=4)
        assert buf.is_empty() is True

        buf.push("item")
        assert buf.is_empty() is False

    def test_is_full(self):
        """is_full should check correctly."""
        buf = Buffer(depth=2)
        assert buf.is_full() is False

        buf.push("item1")
        buf.push("item2")
        assert buf.is_full() is True

    def test_occupancy_and_free_space(self):
        """occupancy and free_space should be correct."""
        buf = Buffer(depth=4)

        assert buf.occupancy == 0
        assert buf.free_space == 4

        buf.push("item1")
        buf.push("item2")

        assert buf.occupancy == 2
        assert buf.free_space == 2

    def test_sample_stats(self):
        """sample_stats should record current occupancy."""
        buf = Buffer(depth=4)
        buf.push("item1")
        buf.push("item2")

        buf.sample_stats()

        assert buf.stats.max_occupancy == 2
        assert buf.stats.total_occupancy_samples == 1

    def test_clear(self):
        """clear should remove all items."""
        buf = Buffer(depth=4)
        buf.push("item1")
        buf.push("item2")

        buf.clear()

        assert buf.is_empty() is True
        assert buf.occupancy == 0

    def test_len(self):
        """__len__ should return occupancy."""
        buf = Buffer(depth=4)
        buf.push("item1")
        buf.push("item2")

        assert len(buf) == 2

    def test_repr(self):
        """__repr__ should return informative string."""
        buf = Buffer(depth=4, name="test")
        buf.push("item")
        s = repr(buf)
        assert "test" in s
        assert "1/4" in s

    def test_stats_tracking(self):
        """Stats should track writes and reads."""
        buf = Buffer(depth=4)
        buf.push("item1")
        buf.push("item2")
        buf.pop()

        assert buf.stats.total_writes == 2
        assert buf.stats.total_reads == 1


class TestFlitBuffer:
    """Test FlitBuffer class with packet tracking."""

    def _make_flit(self, src_id=0, dst_id=1, rob_idx=0, last=False):
        """Create a test flit."""
        hdr = FlitHeader(
            src_id=src_id,
            dst_id=dst_id,
            rob_idx=rob_idx,
            rob_req=0,
            axi_ch=0,
            last=last,
        )
        return Flit(hdr=hdr, payload=b'\x00' * 8)

    def test_push_and_pop(self):
        """push and pop should track packet flits."""
        buf = FlitBuffer(depth=8)

        flit1 = self._make_flit(src_id=1, dst_id=2, rob_idx=0)
        flit2 = self._make_flit(src_id=1, dst_id=2, rob_idx=0, last=True)

        assert buf.push(flit1) is True
        assert buf.push(flit2) is True

        # Packet key = (src_id, dst_id, rob_idx) = (1, 2, 0)
        key = (1, 2, 0)
        assert buf.get_packet_flit_count(key) == 2

        buf.pop()
        assert buf.get_packet_flit_count(key) == 1

        buf.pop()
        assert buf.get_packet_flit_count(key) == 0

    def test_has_complete_packet_empty(self):
        """has_complete_packet should return False for empty buffer."""
        buf = FlitBuffer(depth=8)
        assert buf.has_complete_packet() is False

    def test_has_complete_packet_single_flit(self):
        """has_complete_packet should detect single-flit packet."""
        buf = FlitBuffer(depth=8)

        # Single flit packet (last=True on head)
        flit = self._make_flit(last=True)
        buf.push(flit)

        assert buf.has_complete_packet() is True

    def test_has_complete_packet_multi_flit(self):
        """has_complete_packet should detect multi-flit packet completion."""
        buf = FlitBuffer(depth=8)

        # Head flit (not last)
        head = self._make_flit(src_id=1, dst_id=2, rob_idx=5, last=False)
        buf.push(head)

        assert buf.has_complete_packet() is False

        # Tail flit (last=True, same packet key)
        tail = self._make_flit(src_id=1, dst_id=2, rob_idx=5, last=True)
        buf.push(tail)

        assert buf.has_complete_packet() is True

    def test_get_packet_flit_count_not_found(self):
        """get_packet_flit_count should return 0 for unknown key."""
        buf = FlitBuffer(depth=8)
        assert buf.get_packet_flit_count((99, 99, 99)) == 0


class TestCreditFlowControl:
    """Test CreditFlowControl class."""

    def test_init(self):
        """Credits should initialize to initial_credits."""
        cfc = CreditFlowControl(initial_credits=8)
        assert cfc.credits == 8
        assert cfc.initial_credits == 8

    def test_can_send(self):
        """can_send should check available credits."""
        cfc = CreditFlowControl(initial_credits=4)

        assert cfc.can_send(1) is True
        assert cfc.can_send(4) is True
        assert cfc.can_send(5) is False

    def test_consume(self):
        """consume should reduce credits."""
        cfc = CreditFlowControl(initial_credits=4)

        assert cfc.consume(2) is True
        assert cfc.credits == 2

        assert cfc.consume(3) is False  # Not enough
        assert cfc.credits == 2

    def test_release(self):
        """release should increase credits up to initial."""
        cfc = CreditFlowControl(initial_credits=4)
        cfc.consume(3)

        cfc.release(2)
        assert cfc.credits == 3

        cfc.release(10)  # Over-release should cap
        assert cfc.credits == 4

    def test_reset(self):
        """reset should restore initial credits."""
        cfc = CreditFlowControl(initial_credits=8)
        cfc.consume(5)

        cfc.reset()
        assert cfc.credits == 8

    def test_available_property(self):
        """available should return current credits."""
        cfc = CreditFlowControl(initial_credits=4)
        cfc.consume(1)

        assert cfc.available == 3

    def test_utilization_zero_initial(self):
        """utilization should handle zero initial_credits."""
        cfc = CreditFlowControl(initial_credits=0)
        assert cfc.utilization == 0.0

    def test_utilization_calculation(self):
        """utilization should calculate correctly."""
        cfc = CreditFlowControl(initial_credits=4)

        # All credits available = 0% utilization
        assert cfc.utilization == 0.0

        cfc.consume(2)
        # 2/4 consumed = 50% utilization
        assert cfc.utilization == 0.5

        cfc.consume(2)
        # All consumed = 100% utilization
        assert cfc.utilization == 1.0


class TestPortBuffer:
    """Test PortBuffer class."""

    def _make_flit(self, last=True):
        """Create a test flit."""
        hdr = FlitHeader(
            src_id=0, dst_id=1, rob_idx=0,
            rob_req=0, axi_ch=0, last=last,
        )
        return Flit(hdr=hdr, payload=b'\x00' * 8)

    def test_init(self):
        """PortBuffer should initialize correctly."""
        pb = PortBuffer(depth=4, name="test_port")

        assert pb.name == "test_port"
        assert pb.input_buffer.depth == 4
        assert pb.credit.initial_credits == 4

    def test_init_custom_downstream_depth(self):
        """PortBuffer should use custom downstream depth for credits."""
        pb = PortBuffer(depth=4, downstream_depth=8)

        assert pb.input_buffer.depth == 4
        assert pb.credit.initial_credits == 8

    def test_can_accept(self):
        """can_accept should check buffer space."""
        pb = PortBuffer(depth=2)

        assert pb.can_accept() is True

        pb.receive(self._make_flit())
        pb.receive(self._make_flit())

        assert pb.can_accept() is False

    def test_can_send(self):
        """can_send should check credits."""
        pb = PortBuffer(depth=4, downstream_depth=2)

        assert pb.can_send(1) is True
        assert pb.can_send(2) is True
        assert pb.can_send(3) is False

    def test_receive(self):
        """receive should push flit to buffer."""
        pb = PortBuffer(depth=4)

        assert pb.receive(self._make_flit()) is True
        assert pb.occupancy == 1

    def test_send(self):
        """send should pop flit and consume credit."""
        pb = PortBuffer(depth=4)
        flit = self._make_flit()
        pb.receive(flit)

        sent = pb.send()

        assert sent is flit
        assert pb.occupancy == 0
        assert pb.credit.credits == 3  # One consumed

    def test_send_empty_buffer(self):
        """send should return None on empty buffer."""
        pb = PortBuffer(depth=4)
        assert pb.send() is None

    def test_send_no_credits(self):
        """send should return None without credits."""
        pb = PortBuffer(depth=4, downstream_depth=1)
        pb.receive(self._make_flit())

        # Consume the only credit
        pb.credit.consume(1)

        # No credits available
        assert pb.send() is None

    def test_release_credit(self):
        """release_credit should increment credits."""
        pb = PortBuffer(depth=4)
        pb.receive(self._make_flit())
        pb.send()  # Consume one credit

        pb.release_credit()

        assert pb.credit.credits == 4

    def test_free_space_property(self):
        """free_space should return buffer space."""
        pb = PortBuffer(depth=4)
        pb.receive(self._make_flit())

        assert pb.free_space == 3

    def test_sample_stats(self):
        """sample_stats should sample buffer stats."""
        pb = PortBuffer(depth=4)
        pb.receive(self._make_flit())
        pb.receive(self._make_flit())

        pb.sample_stats()

        assert pb.input_buffer.stats.max_occupancy == 2

    def test_repr(self):
        """__repr__ should return informative string."""
        pb = PortBuffer(depth=4, name="test")
        pb.receive(self._make_flit())

        s = repr(pb)

        assert "test" in s
        assert "buf=" in s
        assert "credit=" in s
