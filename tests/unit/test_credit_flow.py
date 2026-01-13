"""
Tests for Credit-based Flow Control mechanism.

Tests verify:
1. CreditFlowControl consume/release operations
2. RouterPort consumed flag and credit release coordination
3. PortWire credit release propagation
4. Multi-cycle credit flow correctness
"""

import pytest
from src.core.router import Direction, RouterPort, PortWire
from src.core.buffer import CreditFlowControl


class TestCreditFlowControl:
    """Test CreditFlowControl basic operations."""

    def test_initial_credits(self):
        """Initial credits should match specified value."""
        credit = CreditFlowControl(initial_credits=4)
        assert credit.available == 4
        assert credit.can_send(1) is True
        assert credit.can_send(4) is True
        assert credit.can_send(5) is False

    def test_consume_decrements_credits(self):
        """consume() should decrement available credits."""
        credit = CreditFlowControl(initial_credits=4)

        assert credit.consume(1) is True
        assert credit.available == 3

        assert credit.consume(2) is True
        assert credit.available == 1

    def test_consume_fails_when_insufficient(self):
        """consume() should fail when insufficient credits."""
        credit = CreditFlowControl(initial_credits=2)

        assert credit.consume(3) is False
        assert credit.available == 2  # Unchanged

    def test_release_increments_credits(self):
        """release() should increment credits up to initial value."""
        credit = CreditFlowControl(initial_credits=4)
        credit.consume(3)
        assert credit.available == 1

        credit.release(1)
        assert credit.available == 2

        credit.release(2)
        assert credit.available == 4  # Back to initial

    def test_release_capped_at_initial(self):
        """release() should not exceed initial credits."""
        credit = CreditFlowControl(initial_credits=4)

        # Release without consuming
        credit.release(5)
        assert credit.available == 4  # Capped at initial

    def test_consume_all_then_release_all(self):
        """Full consume/release cycle should restore credits."""
        credit = CreditFlowControl(initial_credits=4)

        # Consume all
        for _ in range(4):
            assert credit.consume(1) is True
        assert credit.available == 0
        assert credit.can_send(1) is False

        # Release all
        for _ in range(4):
            credit.release(1)
        assert credit.available == 4
        assert credit.can_send(4) is True


class TestRouterPortConsumedFlag:
    """Test RouterPort _consumed_this_cycle flag behavior."""

    def test_consumed_flag_initially_false(self):
        """Consumed flag should start as False."""
        port = RouterPort(direction=Direction.EAST, buffer_depth=4)
        assert port.check_and_clear_consumed() is False

    def test_pop_for_routing_sets_consumed_flag(self, single_flit_factory):
        """pop_for_routing() should set consumed flag."""
        port = RouterPort(direction=Direction.EAST, buffer_depth=4)
        flit = single_flit_factory(src=(0, 0), dest=(2, 2))

        # Push flit to buffer
        port._buffer.push(flit)

        # Pop should set flag
        popped = port.pop_for_routing()
        assert popped is not None
        assert port.check_and_clear_consumed() is True

    def test_check_and_clear_returns_true_once(self, single_flit_factory):
        """check_and_clear_consumed() should clear flag after returning True."""
        port = RouterPort(direction=Direction.EAST, buffer_depth=4)
        flit = single_flit_factory(src=(0, 0), dest=(2, 2))

        port._buffer.push(flit)
        port.pop_for_routing()

        # First call returns True and clears
        assert port.check_and_clear_consumed() is True
        # Second call returns False
        assert port.check_and_clear_consumed() is False

    def test_multiple_pops_set_consumed_once(self, single_flit_factory):
        """Multiple pops should still result in single consumed flag."""
        port = RouterPort(direction=Direction.EAST, buffer_depth=4)
        flit1 = single_flit_factory(src=(0, 0), dest=(2, 2))
        flit2 = single_flit_factory(src=(0, 0), dest=(3, 3))

        port._buffer.push(flit1)
        port._buffer.push(flit2)

        # Pop twice
        port.pop_for_routing()
        port.pop_for_routing()

        # Flag is True (set by either pop)
        assert port.check_and_clear_consumed() is True
        # After clear, False
        assert port.check_and_clear_consumed() is False


class TestPortWireCreditRelease:
    """Test PortWire credit release propagation."""

    def test_credit_release_a_to_b(self, single_flit_factory):
        """When A consumes from buffer, B should release credit."""
        port_a = RouterPort(direction=Direction.EAST, buffer_depth=4)
        port_b = RouterPort(direction=Direction.WEST, buffer_depth=4)
        wire = PortWire(port_a, port_b)

        # Get initial credit count on B's output
        initial_credits = port_b._output_credit.available

        # Simulate A receiving and consuming a flit
        flit = single_flit_factory(src=(0, 0), dest=(2, 2))
        port_a._buffer.push(flit)
        port_a.pop_for_routing()

        # Simulate B having consumed a credit when sending
        port_b._output_credit.consume(1)
        credits_after_consume = port_b._output_credit.available
        assert credits_after_consume == initial_credits - 1

        # Propagate credit release
        wire.propagate_credit_release()

        # B should get credit back
        assert port_b._output_credit.available == initial_credits

    def test_credit_release_b_to_a(self, single_flit_factory):
        """When B consumes from buffer, A should release credit."""
        port_a = RouterPort(direction=Direction.EAST, buffer_depth=4)
        port_b = RouterPort(direction=Direction.WEST, buffer_depth=4)
        wire = PortWire(port_a, port_b)

        initial_credits = port_a._output_credit.available

        # Simulate B receiving and consuming a flit
        flit = single_flit_factory(src=(2, 2), dest=(0, 0))
        port_b._buffer.push(flit)
        port_b.pop_for_routing()

        # A consumed a credit when sending
        port_a._output_credit.consume(1)

        # Propagate credit release
        wire.propagate_credit_release()

        # A should get credit back
        assert port_a._output_credit.available == initial_credits

    def test_double_propagate_no_double_release(self, single_flit_factory):
        """Calling propagate_credit_release twice should not double-release."""
        port_a = RouterPort(direction=Direction.EAST, buffer_depth=4)
        port_b = RouterPort(direction=Direction.WEST, buffer_depth=4)
        wire = PortWire(port_a, port_b)

        initial_credits = port_b._output_credit.available

        # A consumes a flit
        flit = single_flit_factory(src=(0, 0), dest=(2, 2))
        port_a._buffer.push(flit)
        port_a.pop_for_routing()

        # B sent (consumed credit)
        port_b._output_credit.consume(1)

        # First propagate - releases credit
        wire.propagate_credit_release()
        assert port_b._output_credit.available == initial_credits

        # Second propagate - should NOT release again (flag already cleared)
        wire.propagate_credit_release()
        assert port_b._output_credit.available == initial_credits  # Still at initial, not over

    def test_no_consume_no_release(self):
        """No consumption should mean no credit release."""
        port_a = RouterPort(direction=Direction.EAST, buffer_depth=4)
        port_b = RouterPort(direction=Direction.WEST, buffer_depth=4)
        wire = PortWire(port_a, port_b)

        initial_credits = port_b._output_credit.available
        port_b._output_credit.consume(1)

        # No consumption on A's side
        wire.propagate_credit_release()

        # B should NOT get credit back (A didn't consume)
        assert port_b._output_credit.available == initial_credits - 1


class TestCreditFlowMultiCycle:
    """Test credit flow across multiple simulation cycles."""

    def test_sustained_flow_maintains_credits(self, single_flit_factory):
        """Sustained bidirectional flow should maintain credit balance."""
        port_a = RouterPort(direction=Direction.EAST, buffer_depth=4)
        port_b = RouterPort(direction=Direction.WEST, buffer_depth=4)
        wire = PortWire(port_a, port_b)

        initial_a_credits = port_a._output_credit.available
        initial_b_credits = port_b._output_credit.available

        # Simulate 10 cycles of bidirectional traffic
        for cycle in range(10):
            # A sends to B
            flit_ab = single_flit_factory(src=(0, 0), dest=(2, 2))
            if port_a._output_credit.can_send(1):
                port_a._output_credit.consume(1)
                port_b._buffer.push(flit_ab)

            # B sends to A
            flit_ba = single_flit_factory(src=(2, 2), dest=(0, 0))
            if port_b._output_credit.can_send(1):
                port_b._output_credit.consume(1)
                port_a._buffer.push(flit_ba)

            # Both sides consume from buffers
            if not port_a._buffer.is_empty():
                port_a.pop_for_routing()
            if not port_b._buffer.is_empty():
                port_b.pop_for_routing()

            # Propagate credit releases
            wire.propagate_credit_release()

        # Credits should be restored (all consumed flits were processed)
        assert port_a._output_credit.available == initial_a_credits
        assert port_b._output_credit.available == initial_b_credits

    def test_credit_exhaustion_and_recovery(self, single_flit_factory):
        """Credit exhaustion should block, then recover when released."""
        port_a = RouterPort(direction=Direction.EAST, buffer_depth=4)
        port_b = RouterPort(direction=Direction.WEST, buffer_depth=4)
        wire = PortWire(port_a, port_b)

        # Exhaust A's credits by sending 4 flits without B consuming
        flits_sent = 0
        for i in range(4):
            if port_a._output_credit.can_send(1):
                port_a._output_credit.consume(1)
                flit = single_flit_factory(src=(0, 0), dest=(2, 2))
                port_b._buffer.push(flit)
                flits_sent += 1

        assert flits_sent == 4
        assert port_a._output_credit.available == 0
        assert port_a._output_credit.can_send(1) is False

        # B starts consuming (releases credits back to A)
        for i in range(4):
            port_b.pop_for_routing()
            wire.propagate_credit_release()

            # A should get one credit back each time
            assert port_a._output_credit.available == i + 1

        # A can send again
        assert port_a._output_credit.can_send(4) is True


class TestCreditFlowEdgeCases:
    """Edge cases for credit flow."""

    def test_zero_buffer_depth(self):
        """Zero buffer depth should have 0 initial credits."""
        # Note: buffer_depth=0 might not be a valid config, but test the behavior
        credit = CreditFlowControl(initial_credits=0)
        assert credit.available == 0
        assert credit.can_send(1) is False

    def test_large_credit_operations(self):
        """Large credit values should work correctly."""
        credit = CreditFlowControl(initial_credits=100)

        assert credit.consume(50) is True
        assert credit.available == 50

        credit.release(50)
        assert credit.available == 100

    def test_concurrent_operations_simulation(self, single_flit_factory):
        """Simulate concurrent send/receive in single cycle."""
        port_a = RouterPort(direction=Direction.EAST, buffer_depth=4)
        port_b = RouterPort(direction=Direction.WEST, buffer_depth=4)
        wire = PortWire(port_a, port_b)

        # Pre-fill B's buffer with 2 flits
        for i in range(2):
            flit = single_flit_factory(src=(0, 0), dest=(2, 2))
            port_b._buffer.push(flit)

        # In one cycle:
        # 1. A sends 1 flit to B (consume credit)
        port_a._output_credit.consume(1)
        flit_new = single_flit_factory(src=(0, 0), dest=(2, 2))
        port_b._buffer.push(flit_new)

        # 2. B consumes 1 flit from buffer
        port_b.pop_for_routing()

        # 3. Propagate credit release
        wire.propagate_credit_release()

        # A should get credit back (B consumed)
        assert port_a._output_credit.available == 4  # Restored
