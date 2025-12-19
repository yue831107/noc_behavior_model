"""
Unit tests for EdgeRouterPort signal-based interface.

Tests verify:
1. RouterPort-based _req_port and _resp_port creation
2. Request signal methods (set_req_output, can_send_request_signal)
3. Response signal methods (sample_resp_input, update_resp_ready)
4. PortWire connection and signal propagation
5. Credit release behavior
"""

import pytest
from src.core.routing_selector import EdgeRouterPort
from src.core.router import Direction, RouterPort, PortWire, EdgeRouter, RouterConfig
from src.core.flit import Flit, FlitType, FlitFactory


@pytest.fixture(autouse=True)
def reset_flit_counters():
    """Reset packet/flit ID counters before each test."""
    FlitFactory.reset_packet_id()
    yield


@pytest.fixture
def edge_port() -> EdgeRouterPort:
    """Create an EdgeRouterPort for testing."""
    return EdgeRouterPort(row=1, buffer_depth=4)


@pytest.fixture
def router_config() -> RouterConfig:
    """Default router configuration."""
    return RouterConfig(
        buffer_depth=4,
        output_buffer_depth=0,
        flit_width=64,
        routing_algorithm="XY",
        arbitration="wormhole",
    )


@pytest.fixture
def single_flit() -> Flit:
    """Create a single flit for testing."""
    return FlitFactory.create_single(
        src=(0, 1),
        dest=(2, 1),
        is_request=True,
        payload=b"test",
        timestamp=0,
    )


@pytest.fixture
def response_flit() -> Flit:
    """Create a response flit for testing."""
    return FlitFactory.create_single(
        src=(2, 1),
        dest=(0, 1),
        is_request=False,
        payload=b"resp",
        timestamp=0,
    )


class TestEdgeRouterPortCreation:
    """Test EdgeRouterPort initialization."""

    def test_req_port_creation(self, edge_port):
        """EdgeRouterPort should have _req_port."""
        assert hasattr(edge_port, '_req_port')
        assert isinstance(edge_port._req_port, RouterPort)
        assert edge_port._req_port.direction == Direction.LOCAL

    def test_resp_port_creation(self, edge_port):
        """EdgeRouterPort should have _resp_port."""
        assert hasattr(edge_port, '_resp_port')
        assert isinstance(edge_port._resp_port, RouterPort)
        assert edge_port._resp_port.direction == Direction.LOCAL

    def test_wire_initially_none(self, edge_port):
        """Wire connections should be None initially."""
        assert edge_port._req_wire is None
        assert edge_port._resp_wire is None

    def test_coord_set_correctly(self, edge_port):
        """EdgeRouterPort coord should be (0, row)."""
        assert edge_port.coord == (0, 1)
        assert edge_port.row == 1



class TestEdgeRouterPortRequestSignals:
    """Test request signal methods."""

    def test_can_send_request_initially_true(self, edge_port):
        """Should be able to send initially (has credits)."""
        assert edge_port.can_send_request() is True

    def test_set_req_output_success(self, edge_port, single_flit):
        """set_req_output should set output signals."""
        success = edge_port.set_req_output(single_flit)

        assert success is True
        assert edge_port._req_port.out_valid is True
        assert edge_port._req_port.out_flit == single_flit

    def test_set_req_output_blocked_when_pending(self, edge_port, single_flit):
        """set_req_output should fail when output already pending."""
        edge_port.set_req_output(single_flit)

        # Try to set another output
        flit2 = FlitFactory.create_single(
            src=(0, 1), dest=(3, 1), is_request=True,
            payload=b"test2", timestamp=0
        )
        success = edge_port.set_req_output(flit2)

        assert success is False

    def test_clear_req_if_accepted_success(self, edge_port, single_flit):
        """clear_req_if_accepted should clear when ready."""
        edge_port.set_req_output(single_flit)
        edge_port._req_port.in_ready = True

        cleared = edge_port.clear_req_if_accepted()

        assert cleared is True
        assert edge_port._req_port.out_valid is False
        assert edge_port._req_port.out_flit is None

    def test_clear_req_if_accepted_fail_not_ready(self, edge_port, single_flit):
        """clear_req_if_accepted should not clear when not ready."""
        edge_port.set_req_output(single_flit)
        edge_port._req_port.in_ready = False

        cleared = edge_port.clear_req_if_accepted()

        assert cleared is False
        assert edge_port._req_port.out_valid is True

    def test_can_send_blocked_after_credit_exhaustion(self, edge_port, single_flit):
        """Should block after all credits consumed."""
        # Set credit limit low
        edge_port._req_port._output_credit.credits = 1

        # First send consumes the credit (set_output consumes on clear)
        edge_port.set_req_output(single_flit)
        edge_port._req_port.in_ready = True
        edge_port.clear_req_if_accepted()  # Consumes credit

        # No more credits
        assert edge_port.can_send_request() is False


class TestEdgeRouterPortResponseSignals:
    """Test response signal methods."""

    def test_update_resp_ready_when_empty(self, edge_port):
        """update_resp_ready should set ready when buffer empty."""
        edge_port.update_resp_ready()
        assert edge_port._resp_port.out_ready is True

    def test_update_resp_ready_when_full(self, edge_port, response_flit):
        """update_resp_ready should clear ready when buffer full."""
        # Fill the buffer
        for _ in range(edge_port._buffer_depth):
            flit = FlitFactory.create_single(
                src=(2, 1), dest=(0, 1), is_request=False,
                payload=b"resp", timestamp=0
            )
            edge_port._resp_port._buffer.push(flit)

        edge_port.update_resp_ready()
        assert edge_port._resp_port.out_ready is False

    def test_sample_resp_input_success(self, edge_port, response_flit):
        """sample_resp_input should receive flit when valid and ready."""
        edge_port._resp_port.in_valid = True
        edge_port._resp_port.in_flit = response_flit
        edge_port.update_resp_ready()

        success = edge_port.sample_resp_input()

        assert success is True
        assert edge_port._resp_port._buffer.occupancy == 1

    def test_sample_resp_input_fail_not_valid(self, edge_port):
        """sample_resp_input should fail when not valid."""
        edge_port._resp_port.in_valid = False
        edge_port.update_resp_ready()

        success = edge_port.sample_resp_input()

        assert success is False
        assert edge_port._resp_port._buffer.occupancy == 0

    def test_clear_resp_input_signals(self, edge_port, response_flit):
        """clear_resp_input_signals should reset input signals."""
        edge_port._resp_port.in_valid = True
        edge_port._resp_port.in_flit = response_flit

        edge_port.clear_resp_input_signals()

        assert edge_port._resp_port.in_valid is False
        assert edge_port._resp_port.in_flit is None

    def test_get_response_returns_flit(self, edge_port, response_flit):
        """get_response should return flit from buffer."""
        edge_port._resp_port._buffer.push(response_flit)

        flit = edge_port.get_response()

        assert flit == response_flit
        assert edge_port._resp_port._buffer.occupancy == 0

    def test_get_response_returns_none_when_empty(self, edge_port):
        """get_response should return None when empty."""
        flit = edge_port.get_response()
        assert flit is None

    def test_resp_occupancy_property(self, edge_port, response_flit):
        """resp_occupancy should reflect buffer state."""
        assert edge_port.resp_occupancy == 0

        edge_port._resp_port._buffer.push(response_flit)
        assert edge_port.resp_occupancy == 1


class TestEdgeRouterPortWireConnection:
    """Test PortWire connection behavior."""

    def test_req_wire_creation(self, edge_port, router_config):
        """Request wire should connect _req_port to EdgeRouter LOCAL."""
        edge_router = EdgeRouter(coord=(0, 1), config=router_config)
        edge_port.connect_edge_router(edge_router)

        req_local = edge_router.req_router.ports[Direction.LOCAL]
        edge_port._req_wire = PortWire(edge_port._req_port, req_local)

        assert edge_port._req_wire is not None
        assert edge_port._req_wire.port_a is edge_port._req_port
        assert edge_port._req_wire.port_b is req_local

    def test_req_signal_propagation(self, edge_port, router_config, single_flit):
        """Request signal should propagate via wire."""
        edge_router = EdgeRouter(coord=(0, 1), config=router_config)
        edge_port.connect_edge_router(edge_router)

        req_local = edge_router.req_router.ports[Direction.LOCAL]
        edge_port._req_wire = PortWire(edge_port._req_port, req_local)

        # Set output on edge_port._req_port
        edge_port.set_req_output(single_flit)

        # Propagate via wire
        edge_port._req_wire.propagate_signals()

        # EdgeRouter should see the signal
        assert req_local.in_valid is True
        assert req_local.in_flit == single_flit

    def test_resp_wire_creation(self, edge_port, router_config):
        """Response wire should connect EdgeRouter LOCAL to _resp_port."""
        edge_router = EdgeRouter(coord=(0, 1), config=router_config)
        edge_port.connect_edge_router(edge_router)

        resp_local = edge_router.resp_router.ports[Direction.LOCAL]
        edge_port._resp_wire = PortWire(resp_local, edge_port._resp_port)

        assert edge_port._resp_wire is not None
        assert edge_port._resp_wire.port_a is resp_local
        assert edge_port._resp_wire.port_b is edge_port._resp_port

    def test_resp_signal_propagation(self, edge_port, router_config, response_flit):
        """Response signal should propagate via wire."""
        edge_router = EdgeRouter(coord=(0, 1), config=router_config)
        edge_port.connect_edge_router(edge_router)

        resp_local = edge_router.resp_router.ports[Direction.LOCAL]
        edge_port._resp_wire = PortWire(resp_local, edge_port._resp_port)

        # Set output on EdgeRouter resp LOCAL
        resp_local.out_valid = True
        resp_local.out_flit = response_flit

        # Propagate via wire
        edge_port._resp_wire.propagate_signals()

        # EdgeRouterPort should see the signal
        assert edge_port._resp_port.in_valid is True
        assert edge_port._resp_port.in_flit == response_flit

    def test_ready_backpressure_propagation(self, edge_port, router_config):
        """Ready signal should propagate back via wire."""
        edge_router = EdgeRouter(coord=(0, 1), config=router_config)
        edge_port.connect_edge_router(edge_router)

        req_local = edge_router.req_router.ports[Direction.LOCAL]
        edge_port._req_wire = PortWire(edge_port._req_port, req_local)

        # EdgeRouter sets ready
        req_local.update_ready()  # Should be True when empty

        # Propagate via wire
        edge_port._req_wire.propagate_signals()

        # EdgeRouterPort should see ready
        assert edge_port._req_port.in_ready is True


class TestEdgeRouterPortCreditRelease:
    """Test credit release behavior."""

    def test_credit_release_on_consume(self, edge_port, router_config, single_flit):
        """Credit should be released when EdgeRouter consumes flit."""
        edge_router = EdgeRouter(coord=(0, 1), config=router_config)
        edge_port.connect_edge_router(edge_router)

        req_local = edge_router.req_router.ports[Direction.LOCAL]
        edge_port._req_wire = PortWire(edge_port._req_port, req_local)

        # Initialize credits
        edge_port._req_port._output_credit.credits = 4

        # Set output and propagate
        edge_port.set_req_output(single_flit)
        edge_port._req_wire.propagate_signals()

        # EdgeRouter samples and consumes
        req_local.update_ready()
        req_local.sample_input()

        # Clear output (consumes credit)
        edge_port._req_port.in_ready = req_local.out_ready
        edge_port._req_wire.propagate_signals()  # Get ready signal
        edge_port.clear_req_if_accepted()

        initial_credits = edge_port.available_credits  # Should be 3 after consume

        # EdgeRouter pops the flit (consumes from its buffer)
        req_local.pop_for_routing()

        # Simulate credit release via wire
        edge_port._req_wire.propagate_credit_release()

        # Credit should be released
        assert edge_port.available_credits == initial_credits + 1

    def test_available_credits_property(self, edge_port):
        """available_credits should reflect _req_port credit state."""
        edge_port._req_port._output_credit.credits = 3
        assert edge_port.available_credits == 3


class TestEdgeRouterPortLegacyCompatibility:
    """Test legacy method compatibility."""

    def test_legacy_can_send_request(self, edge_port):
        """Legacy can_send_request should work."""
        assert edge_port.can_send_request() is True

    def test_legacy_get_response(self, edge_port, response_flit):
        """Legacy get_response should return flit."""
        edge_port._resp_port._buffer.push(response_flit)

        flit = edge_port.get_response()

        assert flit == response_flit
        assert edge_port._resp_port._buffer.is_empty()

    def test_repr(self, edge_port):
        """__repr__ should return meaningful string."""
        repr_str = repr(edge_port)
        assert "EdgeRouterPort" in repr_str
        assert "row=1" in repr_str
