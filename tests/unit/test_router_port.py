"""
Tests for RouterPort valid/ready interface.

Tests verify:
1. Initial signal states
2. Ready signal reflects buffer availability
3. Handshake (valid && ready) mechanics
4. Credit-based flow control
5. Output buffer support
"""

import pytest
from src.core.router import Direction, RouterPort, RouterConfig
from src.core.flit import FlitFactory


class TestRouterPortInitialState:
    """Test RouterPort initial state."""

    def test_initial_signal_states(self, router_config):
        """Check initial signal states."""
        port = RouterPort(
            direction=Direction.NORTH,
            buffer_depth=router_config.buffer_depth,
        )

        # Ingress signals
        assert port.in_valid is False
        assert port.in_flit is None
        assert port.out_ready is True  # Buffer empty = ready

        # Egress signals
        assert port.out_valid is False
        assert port.out_flit is None
        assert port.in_ready is False  # No downstream connected

    def test_initial_buffer_empty(self, router_config):
        """Buffer should be empty initially."""
        port = RouterPort(
            direction=Direction.EAST,
            buffer_depth=router_config.buffer_depth,
        )

        assert port.occupancy == 0
        assert port.peek() is None

    def test_initial_credits_full(self, router_config):
        """Port should start with full credits."""
        port = RouterPort(
            direction=Direction.SOUTH,
            buffer_depth=4,
        )

        assert port.credits_available == 4


class TestRouterPortReadySignal:
    """Test out_ready signal behavior."""

    def test_update_ready_when_buffer_empty(self, router_config):
        """out_ready should be True when buffer is empty."""
        port = RouterPort(
            direction=Direction.NORTH,
            buffer_depth=4,
        )

        port.update_ready()

        assert port.out_ready is True

    def test_update_ready_when_buffer_has_space(self, router_config, single_flit_factory):
        """out_ready should be True when buffer has space."""
        port = RouterPort(
            direction=Direction.NORTH,
            buffer_depth=4,
        )

        # Add one flit
        flit = single_flit_factory(src=(0, 0), dest=(2, 2))
        port.receive(flit)

        port.update_ready()

        assert port.out_ready is True  # Still has space

    def test_update_ready_when_buffer_full(self, router_config, single_flit_factory):
        """out_ready should be False when buffer is full."""
        port = RouterPort(
            direction=Direction.NORTH,
            buffer_depth=2,  # Small buffer for testing
        )

        # Fill buffer
        for _ in range(2):
            flit = single_flit_factory(src=(0, 0), dest=(2, 2))
            port.receive(flit)

        port.update_ready()

        assert port.out_ready is False


class TestRouterPortHandshake:
    """Test valid/ready handshake mechanics."""

    def test_sample_input_on_handshake(self, router_config, single_flit_factory):
        """sample_input should receive flit when valid && ready."""
        port = RouterPort(
            direction=Direction.NORTH,
            buffer_depth=4,
        )

        flit = single_flit_factory(src=(0, 0), dest=(2, 2))

        # Simulate incoming valid signal
        port.in_valid = True
        port.in_flit = flit
        port.update_ready()  # out_ready = True (buffer empty)

        success = port.sample_input()

        assert success is True
        assert port.occupancy == 1
        assert port.peek() == flit

    def test_sample_input_blocked_when_not_valid(self, router_config, single_flit_factory):
        """sample_input should fail when in_valid is False."""
        port = RouterPort(
            direction=Direction.NORTH,
            buffer_depth=4,
        )

        flit = single_flit_factory(src=(0, 0), dest=(2, 2))

        port.in_valid = False  # Not valid
        port.in_flit = flit
        port.update_ready()

        success = port.sample_input()

        assert success is False
        assert port.occupancy == 0

    def test_sample_input_blocked_when_not_ready(self, router_config, single_flit_factory):
        """sample_input should fail when buffer is full (not ready)."""
        port = RouterPort(
            direction=Direction.NORTH,
            buffer_depth=1,  # Tiny buffer
        )

        flit1 = single_flit_factory(src=(0, 0), dest=(2, 2))
        flit2 = single_flit_factory(src=(0, 0), dest=(2, 2))

        port.receive(flit1)  # Fill buffer
        port.update_ready()  # out_ready = False

        port.in_valid = True
        port.in_flit = flit2

        success = port.sample_input()

        assert success is False
        assert port.occupancy == 1  # Still only 1 flit

    def test_clear_input_signals(self, router_config, single_flit_factory):
        """clear_input_signals should reset in_valid and in_flit."""
        port = RouterPort(
            direction=Direction.NORTH,
            buffer_depth=4,
        )

        flit = single_flit_factory(src=(0, 0), dest=(2, 2))
        port.in_valid = True
        port.in_flit = flit

        port.clear_input_signals()

        assert port.in_valid is False
        assert port.in_flit is None


class TestRouterPortOutput:
    """Test output signal handling."""

    def test_set_output_success(self, router_config, single_flit_factory):
        """set_output should set out_valid and out_flit."""
        port = RouterPort(
            direction=Direction.NORTH,
            buffer_depth=4,
        )

        flit = single_flit_factory(src=(0, 0), dest=(2, 2))

        success = port.set_output(flit)

        assert success is True
        assert port.out_valid is True
        assert port.out_flit == flit

    def test_set_output_blocked_when_pending(self, router_config, single_flit_factory):
        """set_output should fail if there's already a pending output."""
        port = RouterPort(
            direction=Direction.NORTH,
            buffer_depth=4,
        )

        flit1 = single_flit_factory(src=(0, 0), dest=(2, 2))
        flit2 = single_flit_factory(src=(0, 0), dest=(2, 2))

        port.set_output(flit1)

        success = port.set_output(flit2)

        assert success is False
        assert port.out_flit == flit1  # Original flit unchanged

    def test_clear_output_if_accepted(self, router_config, single_flit_factory):
        """clear_output_if_accepted should clear when in_ready is True."""
        port = RouterPort(
            direction=Direction.NORTH,
            buffer_depth=4,
        )

        flit = single_flit_factory(src=(0, 0), dest=(2, 2))
        port.set_output(flit)

        # Simulate downstream acceptance
        port.in_ready = True

        cleared = port.clear_output_if_accepted()

        assert cleared is True
        assert port.out_valid is False
        assert port.out_flit is None

    def test_clear_output_not_accepted(self, router_config, single_flit_factory):
        """clear_output_if_accepted should not clear when in_ready is False."""
        port = RouterPort(
            direction=Direction.NORTH,
            buffer_depth=4,
        )

        flit = single_flit_factory(src=(0, 0), dest=(2, 2))
        port.set_output(flit)

        port.in_ready = False  # Downstream not ready

        cleared = port.clear_output_if_accepted()

        assert cleared is False
        assert port.out_valid is True
        assert port.out_flit == flit


class TestRouterPortCreditFlow:
    """Test credit-based flow control."""

    def test_credit_consumed_on_send(self, router_config, single_flit_factory):
        """Sending should consume credits."""
        port = RouterPort(
            direction=Direction.NORTH,
            buffer_depth=4,
        )

        flit = single_flit_factory(src=(0, 0), dest=(2, 2))

        initial_credits = port.credits_available

        port.set_output(flit)
        port.in_ready = True
        port.clear_output_if_accepted()

        assert port.credits_available == initial_credits - 1

    def test_cannot_send_without_credits(self, router_config, single_flit_factory):
        """set_output should fail without credits."""
        port = RouterPort(
            direction=Direction.NORTH,
            buffer_depth=1,
        )

        # Exhaust credits
        flit1 = single_flit_factory(src=(0, 0), dest=(2, 2))
        port.set_output(flit1)
        port.in_ready = True
        port.clear_output_if_accepted()

        assert port.credits_available == 0

        # Try to send another
        flit2 = single_flit_factory(src=(0, 0), dest=(2, 2))
        success = port.set_output(flit2)

        assert success is False

    def test_credit_release(self, router_config, single_flit_factory):
        """release_credit should restore sending ability."""
        port = RouterPort(
            direction=Direction.NORTH,
            buffer_depth=1,
        )

        flit1 = single_flit_factory(src=(0, 0), dest=(2, 2))
        port.set_output(flit1)
        port.in_ready = True
        port.clear_output_if_accepted()

        assert port.credits_available == 0

        port.release_credit()

        assert port.credits_available == 1

    def test_can_send_checks_credits(self, router_config, single_flit_factory):
        """can_send should return False when no credits available."""
        port = RouterPort(
            direction=Direction.NORTH,
            buffer_depth=1,
        )

        assert port.can_send() is True

        # Exhaust credits
        flit = single_flit_factory(src=(0, 0), dest=(2, 2))
        port.set_output(flit)
        port.in_ready = True
        port.clear_output_if_accepted()

        assert port.can_send() is False


class TestRouterPortOutputBuffer:
    """Test output buffer functionality."""

    def test_output_buffer_creation(self, router_config_with_output_buffer):
        """Port with output_buffer_depth > 0 should have output buffer."""
        port = RouterPort(
            direction=Direction.NORTH,
            buffer_depth=4,
            output_buffer_depth=2,
        )

        assert port._output_buffer is not None
        assert port._output_buffer_depth == 2

    def test_output_buffer_not_created_when_zero(self, router_config):
        """Port with output_buffer_depth = 0 should not have output buffer."""
        port = RouterPort(
            direction=Direction.NORTH,
            buffer_depth=4,
            output_buffer_depth=0,
        )

        assert port._output_buffer is None

    def test_set_output_to_buffer(self, single_flit_factory):
        """set_output should push to output buffer when enabled."""
        port = RouterPort(
            direction=Direction.NORTH,
            buffer_depth=4,
            output_buffer_depth=2,
        )

        flit1 = single_flit_factory(src=(0, 0), dest=(2, 2))
        flit2 = single_flit_factory(src=(0, 0), dest=(2, 2))

        success1 = port.set_output(flit1)
        success2 = port.set_output(flit2)

        assert success1 is True
        assert success2 is True
        assert port._output_buffer.occupancy == 2

    def test_output_buffer_full_blocks_set_output(self, single_flit_factory):
        """set_output should fail when output buffer is full."""
        port = RouterPort(
            direction=Direction.NORTH,
            buffer_depth=4,
            output_buffer_depth=2,
        )

        # Fill output buffer
        flit1 = single_flit_factory(src=(0, 0), dest=(2, 2))
        flit2 = single_flit_factory(src=(0, 0), dest=(2, 2))
        flit3 = single_flit_factory(src=(0, 0), dest=(2, 2))

        port.set_output(flit1)
        port.set_output(flit2)

        success = port.set_output(flit3)

        assert success is False

    def test_update_output_from_buffer(self, single_flit_factory):
        """update_output_from_buffer should set out_valid from buffer."""
        port = RouterPort(
            direction=Direction.NORTH,
            buffer_depth=4,
            output_buffer_depth=2,
        )

        flit = single_flit_factory(src=(0, 0), dest=(2, 2))
        port.set_output(flit)

        assert port.out_valid is False  # Not set yet

        port.update_output_from_buffer()

        assert port.out_valid is True
        assert port.out_flit == flit

    def test_clear_output_pops_from_buffer(self, single_flit_factory):
        """clear_output_if_accepted should pop from output buffer."""
        port = RouterPort(
            direction=Direction.NORTH,
            buffer_depth=4,
            output_buffer_depth=2,
        )

        flit1 = single_flit_factory(src=(0, 0), dest=(2, 2))
        flit2 = single_flit_factory(src=(0, 0), dest=(2, 2))

        port.set_output(flit1)
        port.set_output(flit2)

        # Update and accept first flit
        port.update_output_from_buffer()
        port.in_ready = True
        port.clear_output_if_accepted()

        assert port._output_buffer.occupancy == 1

        # Next flit should be available
        port.update_output_from_buffer()
        assert port.out_flit == flit2
