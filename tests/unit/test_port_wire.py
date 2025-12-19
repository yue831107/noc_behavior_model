"""
Tests for PortWire signal propagation.

Tests verify:
1. Bidirectional signal propagation (valid, ready, flit)
2. Credit release through wire
"""

import pytest
from src.core.router import Direction, RouterPort, PortWire
from src.core.flit import FlitFactory


class TestPortWireSignalPropagation:
    """Test bidirectional signal propagation."""

    def test_propagate_valid_a_to_b(self, single_flit_factory):
        """A.out_valid should propagate to B.in_valid."""
        port_a = RouterPort(direction=Direction.EAST, buffer_depth=4)
        port_b = RouterPort(direction=Direction.WEST, buffer_depth=4)
        wire = PortWire(port_a, port_b)

        flit = single_flit_factory(src=(0, 0), dest=(2, 2))
        port_a.out_valid = True
        port_a.out_flit = flit

        wire.propagate_signals()

        assert port_b.in_valid is True
        assert port_b.in_flit == flit

    def test_propagate_valid_b_to_a(self, single_flit_factory):
        """B.out_valid should propagate to A.in_valid."""
        port_a = RouterPort(direction=Direction.EAST, buffer_depth=4)
        port_b = RouterPort(direction=Direction.WEST, buffer_depth=4)
        wire = PortWire(port_a, port_b)

        flit = single_flit_factory(src=(2, 2), dest=(0, 0))
        port_b.out_valid = True
        port_b.out_flit = flit

        wire.propagate_signals()

        assert port_a.in_valid is True
        assert port_a.in_flit == flit

    def test_propagate_ready_b_to_a(self):
        """B.out_ready should propagate to A.in_ready."""
        port_a = RouterPort(direction=Direction.EAST, buffer_depth=4)
        port_b = RouterPort(direction=Direction.WEST, buffer_depth=4)
        wire = PortWire(port_a, port_b)

        port_b.update_ready()  # Sets out_ready based on buffer

        wire.propagate_signals()

        assert port_a.in_ready == port_b.out_ready

    def test_propagate_ready_a_to_b(self):
        """A.out_ready should propagate to B.in_ready."""
        port_a = RouterPort(direction=Direction.EAST, buffer_depth=4)
        port_b = RouterPort(direction=Direction.WEST, buffer_depth=4)
        wire = PortWire(port_a, port_b)

        port_a.update_ready()

        wire.propagate_signals()

        assert port_b.in_ready == port_a.out_ready

    def test_bidirectional_propagation(self, single_flit_factory):
        """Both directions should propagate simultaneously."""
        port_a = RouterPort(direction=Direction.EAST, buffer_depth=4)
        port_b = RouterPort(direction=Direction.WEST, buffer_depth=4)
        wire = PortWire(port_a, port_b)

        flit_a = single_flit_factory(src=(0, 0), dest=(2, 2))
        flit_b = single_flit_factory(src=(2, 2), dest=(0, 0))

        port_a.out_valid = True
        port_a.out_flit = flit_a

        port_b.out_valid = True
        port_b.out_flit = flit_b

        port_a.update_ready()
        port_b.update_ready()

        wire.propagate_signals()

        # A -> B
        assert port_b.in_valid is True
        assert port_b.in_flit == flit_a
        assert port_b.in_ready == port_a.out_ready

        # B -> A
        assert port_a.in_valid is True
        assert port_a.in_flit == flit_b
        assert port_a.in_ready == port_b.out_ready


class TestPortWireReadyPropagation:
    """Test ready signal propagation in various buffer states."""

    def test_ready_true_when_buffer_empty(self):
        """Ready should be True when downstream buffer is empty."""
        port_a = RouterPort(direction=Direction.EAST, buffer_depth=4)
        port_b = RouterPort(direction=Direction.WEST, buffer_depth=4)
        wire = PortWire(port_a, port_b)

        port_b.update_ready()
        wire.propagate_signals()

        assert port_a.in_ready is True

    def test_ready_false_when_buffer_full(self, single_flit_factory):
        """Ready should be False when downstream buffer is full."""
        port_a = RouterPort(direction=Direction.EAST, buffer_depth=4)
        port_b = RouterPort(direction=Direction.WEST, buffer_depth=2)  # Small buffer
        wire = PortWire(port_a, port_b)

        # Fill port_b's buffer
        for _ in range(2):
            flit = single_flit_factory(src=(0, 0), dest=(2, 2))
            port_b.receive(flit)

        port_b.update_ready()
        wire.propagate_signals()

        assert port_a.in_ready is False


class TestPortWireCreditRelease:
    """Test credit release through wire."""

    def test_credit_release_on_consume_a(self, single_flit_factory):
        """Credit should be released when A consumes a flit."""
        port_a = RouterPort(direction=Direction.EAST, buffer_depth=4)
        port_b = RouterPort(direction=Direction.WEST, buffer_depth=4)
        wire = PortWire(port_a, port_b)

        # B sends to A
        flit = single_flit_factory(src=(2, 2), dest=(0, 0))

        # Simulate B sending (consumes B's credit)
        port_b.set_output(flit)
        port_b.in_ready = True  # A is ready
        port_b.clear_output_if_accepted()

        initial_credits_b = port_b.credits_available

        # A receives and consumes the flit
        port_a.receive(flit)
        port_a._buffer.pop()  # Simulate A consuming
        port_a._consumed_this_cycle = True

        wire.propagate_credit_release()

        # B should get credit back
        assert port_b.credits_available == initial_credits_b + 1

    def test_credit_release_on_consume_b(self, single_flit_factory):
        """Credit should be released when B consumes a flit."""
        port_a = RouterPort(direction=Direction.EAST, buffer_depth=4)
        port_b = RouterPort(direction=Direction.WEST, buffer_depth=4)
        wire = PortWire(port_a, port_b)

        # A sends to B
        flit = single_flit_factory(src=(0, 0), dest=(2, 2))

        port_a.set_output(flit)
        port_a.in_ready = True
        port_a.clear_output_if_accepted()

        initial_credits_a = port_a.credits_available

        # B receives and consumes
        port_b.receive(flit)
        port_b._buffer.pop()
        port_b._consumed_this_cycle = True

        wire.propagate_credit_release()

        # A should get credit back
        assert port_a.credits_available == initial_credits_a + 1

    def test_no_credit_release_when_not_consumed(self, single_flit_factory):
        """Credit should not be released if flit not consumed."""
        port_a = RouterPort(direction=Direction.EAST, buffer_depth=4)
        port_b = RouterPort(direction=Direction.WEST, buffer_depth=4)
        wire = PortWire(port_a, port_b)

        # A sends to B
        flit = single_flit_factory(src=(0, 0), dest=(2, 2))
        port_a.set_output(flit)
        port_a.in_ready = True
        port_a.clear_output_if_accepted()

        credits_a = port_a.credits_available

        # B receives but doesn't consume (flit sits in buffer)
        port_b.receive(flit)
        # port_b._consumed_this_cycle is False

        wire.propagate_credit_release()

        # A should NOT get credit back
        assert port_a.credits_available == credits_a


class TestPortWireFullCycle:
    """Test complete transfer cycle through wire."""

    def test_complete_transfer_a_to_b(self, single_flit_factory):
        """Test complete flit transfer from A to B."""
        port_a = RouterPort(direction=Direction.EAST, buffer_depth=4)
        port_b = RouterPort(direction=Direction.WEST, buffer_depth=4)
        wire = PortWire(port_a, port_b)

        flit = single_flit_factory(src=(0, 0), dest=(2, 2))

        # Step 1: Update ready
        port_a.update_ready()
        port_b.update_ready()

        # Step 2: A sets output
        port_a.set_output(flit)

        # Step 3: Propagate signals
        wire.propagate_signals()

        # Verify B sees the flit
        assert port_b.in_valid is True
        assert port_b.in_flit == flit

        # Step 4: B samples input
        success = port_b.sample_input()

        assert success is True
        assert port_b.occupancy == 1

        # Step 5: A clears output (B was ready)
        port_a.clear_output_if_accepted()

        assert port_a.out_valid is False

    def test_blocked_transfer_when_not_ready(self, single_flit_factory):
        """Test that transfer is blocked when downstream is not ready."""
        port_a = RouterPort(direction=Direction.EAST, buffer_depth=4)
        port_b = RouterPort(direction=Direction.WEST, buffer_depth=1)
        wire = PortWire(port_a, port_b)

        # Fill B's buffer
        flit_fill = single_flit_factory(src=(0, 0), dest=(2, 2))
        port_b.receive(flit_fill)

        port_a.update_ready()
        port_b.update_ready()

        # A tries to send
        flit = single_flit_factory(src=(0, 0), dest=(2, 2))
        port_a.set_output(flit)

        wire.propagate_signals()

        # B is not ready
        assert port_b.out_ready is False
        assert port_a.in_ready is False

        # B cannot sample (buffer full)
        success = port_b.sample_input()
        assert success is False

        # A output not cleared (not accepted)
        cleared = port_a.clear_output_if_accepted()
        assert cleared is False
        assert port_a.out_valid is True
