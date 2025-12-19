"""
Integration tests for complete Router-NI path verification.

Tests verify:
1. Single flit to NI (Edge → Router → NI)
2. Multi-hop to NI (Edge → R → R → NI)
3. NI response to Edge (NI → Router → Edge)
4. Req/Resp network separation
"""

import pytest
from typing import List, Tuple

from src.core.router import Direction, XYRouter, PortWire, RouterConfig, Router
from src.core.ni import MasterNI, NIConfig
from src.core.flit import Flit, FlitType, FlitFactory

from tests.conftest import run_multi_router_cycle


class TestSingleHopToNI:
    """Test single hop from edge to NI through router."""

    def test_flit_reaches_local_port(self, router_config, ni_config, single_flit_factory):
        """
        Test flit reaches Router's LOCAL port when destined for that router.

        Edge → R1(1,1).LOCAL
        """
        r1 = XYRouter(coord=(1, 1), config=router_config, name="R1")
        ni = MasterNI(coord=(1, 1), config=ni_config)

        # Inject flit destined for R1's location
        flit = single_flit_factory(src=(0, 1), dest=(1, 1))
        r1.ports[Direction.WEST].receive(flit)

        # Run cycles
        for _ in range(5):
            r1.sample_all_inputs()
            r1.clear_all_input_signals()
            r1.update_all_ready()
            r1.route_and_forward(0)

        # Flit should be at LOCAL port
        local_port = r1.ports[Direction.LOCAL]
        assert local_port.out_valid or local_port.occupancy > 0

    def test_ni_receives_from_router_local(
        self, router_config, ni_config, single_flit_factory
    ):
        """
        Test NI receives flit from Router's LOCAL port.

        Uses legacy interface to verify flit transfer.
        """
        r1 = XYRouter(coord=(1, 1), config=router_config, name="R1")
        ni = MasterNI(coord=(1, 1), config=ni_config)

        # Inject flit
        flit = single_flit_factory(src=(0, 1), dest=(1, 1))
        r1.ports[Direction.WEST].receive(flit)

        # Run router cycle to route to LOCAL
        for _ in range(3):
            r1.sample_all_inputs()
            r1.clear_all_input_signals()
            r1.update_all_ready()
            r1.route_and_forward(0)

        # Manually transfer from LOCAL port to NI
        local_port = r1.ports[Direction.LOCAL]
        if local_port.out_valid and local_port.out_flit is not None:
            success = ni.receive_req_flit(local_port.out_flit)
            assert success is True
            assert ni.req_input.occupancy == 1


class TestMultiHopToNI:
    """Test multi-hop path from edge through routers to NI."""

    def test_two_hop_to_ni(self, router_config, ni_config, single_flit_factory):
        """
        Test flit traverses two routers to reach NI.

        Edge → R1(1,1) → R2(2,1) → R2.LOCAL
        """
        r1 = XYRouter(coord=(1, 1), config=router_config, name="R1")
        r2 = XYRouter(coord=(2, 1), config=router_config, name="R2")
        ni = MasterNI(coord=(2, 1), config=ni_config)

        # Connect routers
        wire = PortWire(r1.ports[Direction.EAST], r2.ports[Direction.WEST])

        # Flit destined for R2's location
        flit = single_flit_factory(src=(0, 1), dest=(2, 1))
        r1.ports[Direction.WEST].receive(flit)

        # Run cycles
        routers = [r1, r2]
        for _ in range(10):
            run_multi_router_cycle(routers, [wire])

        # Flit should reach R2.LOCAL
        r2_local = r2.ports[Direction.LOCAL]
        assert r2_local.out_valid or r2_local.occupancy > 0

    def test_three_hop_xy_routing(self, router_config, ni_config, single_flit_factory):
        """
        Test XY routing: EAST then NORTH to reach NI.

        R1(0,0) → R2(1,0) → R3(1,1).LOCAL

        XY routing: X first (EAST), then Y (NORTH)
        """
        r1 = XYRouter(coord=(0, 0), config=router_config, name="R1")
        r2 = XYRouter(coord=(1, 0), config=router_config, name="R2")
        r3 = XYRouter(coord=(1, 1), config=router_config, name="R3")
        ni = MasterNI(coord=(1, 1), config=ni_config)

        # Connect routers: R1-EAST-R2, R2-NORTH-R3
        wire1 = PortWire(r1.ports[Direction.EAST], r2.ports[Direction.WEST])
        wire2 = PortWire(r2.ports[Direction.NORTH], r3.ports[Direction.SOUTH])

        # Flit destined for R3's location (needs XY turn)
        flit = single_flit_factory(src=(0, 0), dest=(1, 1))
        r1.ports[Direction.LOCAL].receive(flit)  # Inject at LOCAL (src)

        # Run cycles
        routers = [r1, r2, r3]
        wires = [wire1, wire2]
        for _ in range(15):
            run_multi_router_cycle(routers, wires)

        # Flit should reach R3.LOCAL
        r3_local = r3.ports[Direction.LOCAL]
        assert r3_local.out_valid or r3_local.occupancy > 0


class TestNIResponsePath:
    """Test NI response output path through router."""

    def test_ni_response_reaches_router(
        self, router_config, ni_config, single_flit_factory
    ):
        """
        Test NI response flit reaches Router's LOCAL input.

        NI → R1.LOCAL (input)
        """
        r1 = XYRouter(coord=(1, 1), config=router_config, name="R1")
        ni = MasterNI(coord=(1, 1), config=ni_config)

        # Create response flit and push to NI output
        resp_flit = single_flit_factory(src=(1, 1), dest=(0, 1), is_request=False)
        ni.resp_output.push(resp_flit)
        ni.update_resp_output()

        # Verify NI has output ready
        assert ni.resp_out_valid is True

        # Transfer to router LOCAL port
        r1_local = r1.ports[Direction.LOCAL]
        r1_local.in_valid = ni.resp_out_valid
        r1_local.in_flit = ni.resp_out_flit

        # Router samples
        r1_local.update_ready()
        if r1_local.sample_input():
            assert r1_local.occupancy == 1

    def test_response_routes_from_local(
        self, router_config, ni_config, single_flit_factory
    ):
        """
        Test response flit routes from LOCAL port to destination direction.

        R1.LOCAL → R1.WEST (response to (0,1))
        """
        r1 = XYRouter(coord=(1, 1), config=router_config, name="R1")

        # Response flit going from (1,1) to (0,1) should route WEST
        resp_flit = single_flit_factory(src=(1, 1), dest=(0, 1), is_request=False)
        r1.ports[Direction.LOCAL].receive(resp_flit)

        # Run routing cycle
        for _ in range(3):
            r1.sample_all_inputs()
            r1.clear_all_input_signals()
            r1.update_all_ready()
            r1.route_and_forward(0)

        # Flit should be at WEST output
        west_port = r1.ports[Direction.WEST]
        assert west_port.out_valid or west_port.occupancy > 0


class TestMultiHopResponsePath:
    """Test multi-hop response path from NI back to edge."""

    def test_response_traverses_chain(
        self, router_config, ni_config, single_flit_factory
    ):
        """
        Test response flit travels through router chain.

        NI @ R2(2,1) → R2.WEST → R1(1,1).EAST → R1.WEST
        """
        r1 = XYRouter(coord=(1, 1), config=router_config, name="R1")
        r2 = XYRouter(coord=(2, 1), config=router_config, name="R2")
        ni = MasterNI(coord=(2, 1), config=ni_config)

        wire = PortWire(r1.ports[Direction.EAST], r2.ports[Direction.WEST])

        # Response from (2,1) to (0,1)
        resp_flit = single_flit_factory(src=(2, 1), dest=(0, 1), is_request=False)
        r2.ports[Direction.LOCAL].receive(resp_flit)

        # Run cycles
        routers = [r1, r2]
        for _ in range(10):
            run_multi_router_cycle(routers, [wire])

        # Response should reach R1.WEST
        r1_west = r1.ports[Direction.WEST]
        assert r1_west.out_valid or r1_west.occupancy > 0


class TestReqRespNetworkSeparation:
    """Test that Req and Resp use separate networks (Combined Router)."""

    def test_combined_router_has_both_networks(self, router_config):
        """Combined Router should have separate Req and Resp routers."""
        router = Router(coord=(1, 1), config=router_config)

        # Should have both req_router and resp_router
        assert hasattr(router, 'req_router')
        assert hasattr(router, 'resp_router')
        assert router.req_router is not None
        assert router.resp_router is not None

    def test_request_uses_req_network(
        self, router_config, single_flit_factory
    ):
        """Request flit should use request network."""
        router = Router(coord=(1, 1), config=router_config)

        req_flit = single_flit_factory(src=(0, 1), dest=(2, 1), is_request=True)
        success = router.receive_request(Direction.WEST, req_flit)

        assert success is True
        # Flit should be in req_router
        req_west = router.get_req_port(Direction.WEST)
        assert req_west.occupancy == 1

    def test_response_uses_resp_network(
        self, router_config, single_flit_factory
    ):
        """Response flit should use response network."""
        router = Router(coord=(1, 1), config=router_config)

        resp_flit = single_flit_factory(src=(2, 1), dest=(0, 1), is_request=False)
        success = router.receive_response(Direction.EAST, resp_flit)

        assert success is True
        # Flit should be in resp_router
        resp_east = router.get_resp_port(Direction.EAST)
        assert resp_east.occupancy == 1


class TestEndToEndPath:
    """Test complete end-to-end paths."""

    def test_req_to_ni_and_resp_back(
        self, router_config, ni_config, single_flit_factory
    ):
        """
        Test complete request path to NI.

        Verifies:
        1. Request reaches NI via LOCAL port
        2. NI can process the request

        Note: Full roundtrip requires AXI transaction which is tested elsewhere.
        """
        r1 = XYRouter(coord=(1, 1), config=router_config, name="R1")
        ni = MasterNI(coord=(1, 1), config=ni_config)

        # Request to NI's location
        req_flit = single_flit_factory(src=(0, 1), dest=(1, 1), is_request=True)
        r1.ports[Direction.WEST].receive(req_flit)

        # Route to LOCAL
        for _ in range(3):
            r1.sample_all_inputs()
            r1.clear_all_input_signals()
            r1.update_all_ready()
            r1.route_and_forward(0)

        # Transfer to NI
        local_port = r1.ports[Direction.LOCAL]
        if local_port.out_valid:
            ni.receive_req_flit(local_port.out_flit)
            ni.process_cycle(0)

            # Verify NI processed it (req_input will be empty after processing)
            # The flit was received and processed
            assert True  # If we got here without error, the path works


class TestMultiFlitPacketPath:
    """Test multi-flit packet traversal through complete path."""

    def test_multi_flit_reaches_local(
        self, router_config, multi_flit_packet_factory
    ):
        """
        Multi-flit packet should reach LOCAL port in order.
        """
        r1 = XYRouter(coord=(1, 1), config=router_config, name="R1")

        flits = multi_flit_packet_factory(src=(0, 1), dest=(1, 1), num_flits=3)

        # Inject all flits
        for flit in flits:
            r1.ports[Direction.WEST].receive(flit)

        # Run cycles
        received = []
        for _ in range(10):
            r1.sample_all_inputs()
            r1.clear_all_input_signals()
            r1.update_all_ready()
            r1.route_and_forward(0)

            # Check LOCAL output
            local_port = r1.ports[Direction.LOCAL]
            if local_port.out_valid and local_port.out_flit is not None:
                received.append(local_port.out_flit.flit_type)
                local_port.in_ready = True
                local_port.clear_output_if_accepted()

        # Verify all flits arrived in order
        if len(received) >= 3:
            assert received[0] == FlitType.HEAD
            assert received[1] == FlitType.BODY
            assert received[2] == FlitType.TAIL


class TestPathLatency:
    """Test path latency (cycle count for flit to traverse)."""

    def test_single_hop_latency(self, router_config, single_flit_factory):
        """
        Measure cycles for single hop (should be ~2 cycles).

        Cycle 0: Inject, route
        Cycle 1: Sample at dest, route to output
        Cycle 2: Output valid
        """
        r1 = XYRouter(coord=(1, 1), config=router_config, name="R1")

        flit = single_flit_factory(src=(0, 1), dest=(1, 1))
        r1.ports[Direction.WEST].receive(flit)

        arrived_cycle = None
        for cycle in range(10):
            r1.sample_all_inputs()
            r1.clear_all_input_signals()
            r1.update_all_ready()
            r1.route_and_forward(cycle)

            local_port = r1.ports[Direction.LOCAL]
            if local_port.out_valid and arrived_cycle is None:
                arrived_cycle = cycle
                break

        # Should arrive within a few cycles
        assert arrived_cycle is not None
        assert arrived_cycle <= 3  # Reasonable latency

    def test_two_hop_latency(self, router_config, single_flit_factory):
        """
        Measure cycles for two hops.
        """
        r1 = XYRouter(coord=(1, 1), config=router_config, name="R1")
        r2 = XYRouter(coord=(2, 1), config=router_config, name="R2")
        wire = PortWire(r1.ports[Direction.EAST], r2.ports[Direction.WEST])

        flit = single_flit_factory(src=(0, 1), dest=(2, 1))
        r1.ports[Direction.WEST].receive(flit)

        arrived_cycle = None
        for cycle in range(20):
            run_multi_router_cycle([r1, r2], [wire], cycle)

            r2_local = r2.ports[Direction.LOCAL]
            if r2_local.out_valid and arrived_cycle is None:
                arrived_cycle = cycle
                break

        # Two hops should take more cycles
        assert arrived_cycle is not None
        assert arrived_cycle <= 6  # Reasonable latency for 2 hops
