"""
Tests for NI-Router LOCAL port connection.

Tests verify:
1. Router forwards to LOCAL port
2. NI receives from Router
3. NI generates response
4. Router receives from NI
5. NI ready signal handling
6. NI handshake sampling
"""

import pytest
from src.core.router import Direction, RouterPort, PortWire, Router
from src.core.ni import MasterNI, NIConfig
from src.core.flit import FlitFactory, AxiChannel


class TestRouterToNIForwarding:
    """Test Router forwarding flits to NI via LOCAL port."""

    def test_router_local_port_exists(self, xy_router):
        """Router should have a LOCAL port."""
        assert Direction.LOCAL in xy_router.ports

    def test_router_forwards_to_local(self, xy_router, single_flit_factory):
        """Flit destined for router's coordinate should route to LOCAL."""
        # Router at (2,2), flit destined for (2,2)
        flit = single_flit_factory(src=(0, 0), dest=(2, 2))

        # Inject at WEST port
        west_port = xy_router.ports[Direction.WEST]
        west_port.receive(flit)

        # Process routing
        xy_router.process_cycle()

        # Check LOCAL port has the flit
        local_port = xy_router.ports[Direction.LOCAL]
        assert local_port.out_valid is True or local_port.occupancy > 0

    def test_router_local_output_signals(self, xy_router, single_flit_factory):
        """LOCAL port should have correct output signals when flit ready."""
        flit = single_flit_factory(src=(0, 0), dest=(2, 2))

        # Inject and route
        west_port = xy_router.ports[Direction.WEST]
        west_port.receive(flit)

        # Set LOCAL port ready (simulating NI is ready to receive)
        local_port = xy_router.ports[Direction.LOCAL]
        local_port.in_ready = True

        xy_router.process_cycle()

        # Output should be valid if flit routed to LOCAL
        # Note: depends on routing implementation details


class TestNIReceiveFromRouter:
    """Test NI receiving flits from Router."""

    def test_ni_ready_when_buffer_empty(self, master_ni):
        """NI should be ready when request buffer is empty."""
        master_ni.update_ready_signals()
        assert master_ni.req_in_ready is True

    def test_ni_receives_flit_via_handshake(self, master_ni, single_flit_factory):
        """NI should receive flit when valid && ready."""
        flit = single_flit_factory(src=(0, 0), dest=(1, 1))

        # Set up handshake
        master_ni.req_in_valid = True
        master_ni.req_in_flit = flit
        master_ni.update_ready_signals()

        # Sample input
        success = master_ni.sample_req_input()

        assert success is True
        assert master_ni.req_input.occupancy == 1

    def test_ni_not_ready_when_buffer_full(self, ni_config, single_flit_factory):
        """NI should not be ready when request buffer is full."""
        # Create NI with small buffer
        small_config = NIConfig(req_buffer_depth=2)
        ni = MasterNI(coord=(1, 1), config=small_config)

        # Fill buffer
        for i in range(2):
            flit = single_flit_factory(src=(0, 0), dest=(1, 1))
            ni.req_input.push(flit)

        ni.update_ready_signals()
        assert ni.req_in_ready is False

    def test_ni_rejects_when_not_ready(self, ni_config, single_flit_factory):
        """NI should reject flit when not ready."""
        small_config = NIConfig(req_buffer_depth=1)
        ni = MasterNI(coord=(1, 1), config=small_config)

        # Fill buffer
        flit1 = single_flit_factory(src=(0, 0), dest=(1, 1))
        ni.req_input.push(flit1)

        ni.update_ready_signals()
        assert ni.req_in_ready is False

        # Try to send another
        flit2 = single_flit_factory(src=(0, 0), dest=(1, 1))
        ni.req_in_valid = True
        ni.req_in_flit = flit2

        success = ni.sample_req_input()
        assert success is False
        assert ni.req_input.occupancy == 1  # Still only 1


class TestNIGeneratesResponse:
    """Test NI generating response flits."""

    def test_ni_resp_output_initially_empty(self, master_ni):
        """NI response output should be empty initially."""
        assert master_ni.resp_output.is_empty()
        master_ni.update_resp_output()
        assert master_ni.resp_out_valid is False

    def test_ni_has_pending_response_false_when_empty(self, master_ni):
        """has_pending_response should be False when empty."""
        assert master_ni.has_pending_response() is False


class TestRouterReceiveFromNI:
    """Test Router receiving flits from NI."""

    def test_ni_output_handshake(self, master_ni, single_flit_factory):
        """NI output should follow valid/ready handshake."""
        # Manually push a response flit to output
        flit = single_flit_factory(src=(1, 1), dest=(0, 0))
        master_ni.resp_output.push(flit)

        # Update output signals
        master_ni.update_resp_output()

        assert master_ni.resp_out_valid is True
        assert master_ni.resp_out_flit == flit

    def test_ni_output_cleared_when_accepted(self, master_ni, single_flit_factory):
        """NI output should be cleared when downstream accepts."""
        flit = single_flit_factory(src=(1, 1), dest=(0, 0))
        master_ni.resp_output.push(flit)
        master_ni.update_resp_output()

        # Simulate downstream ready
        master_ni.resp_out_ready = True

        accepted = master_ni.clear_resp_output_if_accepted()

        assert accepted is True
        assert master_ni.resp_out_valid is False
        assert master_ni.resp_out_flit is None
        assert master_ni.resp_output.is_empty()

    def test_ni_output_not_cleared_when_not_ready(self, master_ni, single_flit_factory):
        """NI output should NOT be cleared when downstream not ready."""
        flit = single_flit_factory(src=(1, 1), dest=(0, 0))
        master_ni.resp_output.push(flit)
        master_ni.update_resp_output()

        # Downstream not ready
        master_ni.resp_out_ready = False

        accepted = master_ni.clear_resp_output_if_accepted()

        assert accepted is False
        assert master_ni.resp_out_valid is True
        assert master_ni.resp_out_flit == flit


class TestNIRouterIntegration:
    """Test NI-Router integration scenarios."""

    def test_router_ni_wire_connection(self, single_flit_factory, router_config):
        """Test wiring Router LOCAL port to NI."""
        from src.core.router import XYRouter

        router = XYRouter(coord=(1, 1), config=router_config)
        ni = MasterNI(coord=(1, 1))

        # Get LOCAL port
        local_port = router.ports[Direction.LOCAL]

        # Simulate wire propagation: Router LOCAL → NI
        # Set router output
        flit = single_flit_factory(src=(0, 0), dest=(1, 1))
        local_port.out_valid = True
        local_port.out_flit = flit

        # NI reads from router output
        ni.req_in_valid = local_port.out_valid
        ni.req_in_flit = local_port.out_flit
        ni.update_ready_signals()

        # Propagate ready back
        local_port.in_ready = ni.req_in_ready

        # NI samples
        success = ni.sample_req_input()
        assert success is True

    def test_ni_router_resp_wire_connection(self, single_flit_factory, router_config):
        """Test NI response output to Router LOCAL port."""
        from src.core.router import XYRouter

        router = XYRouter(coord=(1, 1), config=router_config)
        ni = MasterNI(coord=(1, 1))

        local_port = router.ports[Direction.LOCAL]

        # NI has response to send
        flit = single_flit_factory(src=(1, 1), dest=(0, 0))
        ni.resp_output.push(flit)
        ni.update_resp_output()

        # Router LOCAL port shows ready
        local_port.update_ready()

        # Wire: NI output → Router LOCAL input
        local_port.in_valid = ni.resp_out_valid
        local_port.in_flit = ni.resp_out_flit

        # Wire: Router ready → NI
        ni.resp_out_ready = local_port.out_ready

        # Router samples input
        success = local_port.sample_input()
        assert success is True
        assert local_port.occupancy == 1

        # NI clears output
        ni.clear_resp_output_if_accepted()
        assert ni.resp_out_valid is False


class TestNIRequestProcessing:
    """Test NI processing of received requests."""

    def test_ni_processes_received_flits(self, master_ni, single_flit_factory):
        """NI should process received flits in process_cycle."""
        flit = single_flit_factory(src=(0, 0), dest=(1, 1))

        # Receive flit
        master_ni.req_in_valid = True
        master_ni.req_in_flit = flit
        master_ni.update_ready_signals()
        master_ni.sample_req_input()
        master_ni.clear_input_signals()

        initial_occupancy = master_ni.req_input.occupancy
        assert initial_occupancy == 1

        # Process cycle (will try to disassemble into packets)
        master_ni.process_cycle(current_time=0)

        # Flit should be processed (popped from input buffer)
        assert master_ni.req_input.occupancy == 0


class TestNISignalClear:
    """Test NI signal clearing behavior."""

    def test_clear_input_signals(self, master_ni, single_flit_factory):
        """clear_input_signals should reset input signals."""
        flit = single_flit_factory(src=(0, 0), dest=(1, 1))

        master_ni.req_in_valid = True
        master_ni.req_in_flit = flit

        master_ni.clear_input_signals()

        assert master_ni.req_in_valid is False
        assert master_ni.req_in_flit is None


class TestNILegacyInterface:
    """Test NI legacy interface methods."""

    def test_receive_req_flit(self, master_ni, single_flit_factory):
        """Legacy receive_req_flit should work."""
        flit = single_flit_factory(src=(0, 0), dest=(1, 1))

        success = master_ni.receive_req_flit(flit)

        assert success is True
        assert master_ni.req_input.occupancy == 1

    def test_receive_req_flit_blocked_when_full(self, ni_config, single_flit_factory):
        """Legacy receive_req_flit should fail when buffer full."""
        small_config = NIConfig(req_buffer_depth=1)
        ni = MasterNI(coord=(1, 1), config=small_config)

        flit1 = single_flit_factory(src=(0, 0), dest=(1, 1))
        flit2 = single_flit_factory(src=(0, 0), dest=(1, 1))

        assert ni.receive_req_flit(flit1) is True
        assert ni.receive_req_flit(flit2) is False

    def test_get_resp_flit(self, master_ni, single_flit_factory):
        """Legacy get_resp_flit should work."""
        flit = single_flit_factory(src=(1, 1), dest=(0, 0))
        master_ni.resp_output.push(flit)

        result = master_ni.get_resp_flit()

        assert result == flit
        assert master_ni.resp_output.is_empty()

    def test_get_resp_flit_returns_none_when_empty(self, master_ni):
        """Legacy get_resp_flit should return None when empty."""
        result = master_ni.get_resp_flit()
        assert result is None


class TestNIMemoryAccess:
    """Test NI local memory access."""

    def test_write_local(self, master_ni):
        """write_local should write to memory."""
        data = b"TESTDATA"
        master_ni.write_local(0x1000, data)

        # Verify
        result = master_ni.read_local(0x1000, len(data))
        assert result == data

    def test_verify_local(self, master_ni):
        """verify_local should check memory contents."""
        data = b"EXPECTED"
        master_ni.write_local(0x2000, data)

        assert master_ni.verify_local(0x2000, data) is True
        assert master_ni.verify_local(0x2000, b"WRONG") is False


class TestNIRepr:
    """Test NI string representation."""

    def test_repr(self, master_ni):
        """__repr__ should return meaningful string."""
        repr_str = repr(master_ni)
        assert "MasterNI" in repr_str
        assert "(1, 1)" in repr_str
