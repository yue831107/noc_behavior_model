"""
Tests for MetricsProvider protocol and utility functions.

Tests cover:
1. MetricsProvider Protocol interface
2. get_metrics_from_system() with Protocol-compliant systems
3. _extract_metrics_legacy() fallback paths
"""

import pytest
from unittest.mock import Mock, MagicMock
from typing import Dict, Tuple

from src.verification.metrics_provider import (
    MetricsProvider,
    get_metrics_from_system,
    _extract_metrics_legacy,
)


class TestMetricsProviderProtocol:
    """Test MetricsProvider Protocol definition."""

    def test_protocol_is_runtime_checkable(self):
        """Protocol should be runtime checkable."""
        assert hasattr(MetricsProvider, '__runtime_checkable__') or \
               hasattr(MetricsProvider, '_is_runtime_protocol')

    def test_compliant_class_passes_isinstance(self):
        """A class implementing all methods should pass isinstance check."""
        class CompliantSystem:
            @property
            def current_cycle(self) -> int:
                return 100

            @property
            def mesh_dimensions(self) -> Tuple[int, int]:
                return (5, 4)

            def get_buffer_occupancy(self) -> Dict[Tuple[int, int], int]:
                return {(1, 1): 2}

            def get_flit_stats(self) -> Dict[Tuple[int, int], int]:
                return {(1, 1): 10}

            def get_transfer_stats(self) -> Tuple[int, int, int]:
                return (5, 1024, 256)

        system = CompliantSystem()
        assert isinstance(system, MetricsProvider)

    def test_non_compliant_class_fails_isinstance(self):
        """A class missing methods should fail isinstance check."""
        class IncompleteSystem:
            @property
            def current_cycle(self) -> int:
                return 100
            # Missing other required methods

        system = IncompleteSystem()
        assert not isinstance(system, MetricsProvider)


class TestGetMetricsFromSystem:
    """Test get_metrics_from_system() function."""

    def test_with_protocol_compliant_system(self):
        """Should use Protocol methods when available."""
        class ProtocolSystem:
            @property
            def current_cycle(self) -> int:
                return 200

            @property
            def mesh_dimensions(self) -> Tuple[int, int]:
                return (5, 4)

            def get_buffer_occupancy(self) -> Dict[Tuple[int, int], int]:
                return {(1, 1): 3, (2, 2): 5}

            def get_flit_stats(self) -> Dict[Tuple[int, int], int]:
                return {(1, 1): 15, (2, 2): 20}

            def get_transfer_stats(self) -> Tuple[int, int, int]:
                return (10, 2048, 256)

        system = ProtocolSystem()
        metrics = get_metrics_from_system(system)

        assert metrics['cycle'] == 200
        assert metrics['mesh_cols'] == 5
        assert metrics['mesh_rows'] == 4
        assert metrics['buffer_occupancy'] == {(1, 1): 3, (2, 2): 5}
        assert metrics['flit_stats'] == {(1, 1): 15, (2, 2): 20}
        assert metrics['completed_transactions'] == 10
        assert metrics['bytes_transferred'] == 2048
        assert metrics['transfer_size'] == 256

    def test_falls_back_to_legacy_extraction(self):
        """Should use legacy extraction for non-Protocol systems."""
        # Create a mock that doesn't implement Protocol
        system = Mock(spec=[])
        system.current_cycle = 50

        metrics = get_metrics_from_system(system)

        # Should have used legacy path
        assert metrics['cycle'] == 50


class TestExtractMetricsLegacy:
    """Test _extract_metrics_legacy() fallback function."""

    def test_extracts_cycle_from_current_cycle(self):
        """Should extract cycle from current_cycle attribute."""
        system = Mock(spec=['current_cycle'])
        system.current_cycle = 150

        metrics = _extract_metrics_legacy(system)

        assert metrics['cycle'] == 150

    def test_extracts_cycle_from_current_time_fallback(self):
        """Should fall back to current_time if current_cycle missing."""
        system = Mock(spec=['current_time'])
        system.current_time = 250

        metrics = _extract_metrics_legacy(system)

        assert metrics['cycle'] == 250

    def test_cycle_defaults_to_zero(self):
        """Should default cycle to 0 if no time attribute."""
        system = Mock(spec=[])

        metrics = _extract_metrics_legacy(system)

        assert metrics['cycle'] == 0

    def test_extracts_mesh_dimensions_from_public_attrs(self):
        """Should extract mesh dimensions from public attributes."""
        system = Mock(spec=['mesh_cols', 'mesh_rows'])
        system.mesh_cols = 6
        system.mesh_rows = 5

        metrics = _extract_metrics_legacy(system)

        assert metrics['mesh_cols'] == 6
        assert metrics['mesh_rows'] == 5

    def test_extracts_mesh_dimensions_from_private_attrs(self):
        """Should fall back to private mesh dimension attributes."""
        system = Mock(spec=['_mesh_cols', '_mesh_rows'])
        system._mesh_cols = 4
        system._mesh_rows = 3

        metrics = _extract_metrics_legacy(system)

        assert metrics['mesh_cols'] == 4
        assert metrics['mesh_rows'] == 3

    def test_mesh_dimensions_default_to_5x4(self):
        """Should default mesh dimensions to 5x4."""
        system = Mock(spec=[])

        metrics = _extract_metrics_legacy(system)

        assert metrics['mesh_cols'] == 5
        assert metrics['mesh_rows'] == 4

    def test_extracts_buffer_occupancy_from_mesh(self):
        """Should extract buffer occupancy from mesh routers."""
        # Create mock router with ports
        mock_port = Mock()
        mock_port.input_buffer = Mock()
        mock_port.input_buffer.occupancy = 2
        mock_port.out_valid = True
        mock_port.out_flit = Mock()

        mock_req_router = Mock(spec=['ports', 'stats', 'flits_in_pipeline'])
        mock_req_router.ports = {'NORTH': mock_port}
        mock_req_router.stats = Mock()
        mock_req_router.stats.flits_forwarded = 10
        mock_req_router.flits_in_pipeline = 0

        mock_resp_router = Mock(spec=['ports', 'stats', 'flits_in_pipeline'])
        mock_resp_router.ports = {}
        mock_resp_router.stats = Mock()
        mock_resp_router.stats.flits_forwarded = 5
        mock_resp_router.flits_in_pipeline = 0

        mock_router = Mock()
        mock_router.req_router = mock_req_router
        mock_router.resp_router = mock_resp_router

        mock_mesh = Mock()
        mock_mesh.routers = {(1, 1): mock_router}

        system = Mock(spec=['mesh'])
        system.mesh = mock_mesh

        metrics = _extract_metrics_legacy(system)

        # Should have counted occupancy (2 from buffer + 1 from out_valid)
        assert (1, 1) in metrics['buffer_occupancy']
        assert metrics['buffer_occupancy'][(1, 1)] == 3
        assert metrics['flit_stats'][(1, 1)] == 15

    def test_extracts_transfer_stats_from_host_axi_master(self):
        """Should extract transfer stats from host_axi_master."""
        mock_controller_stats = Mock()
        mock_controller_stats.completed_transactions = 20
        mock_controller_stats.read_completed = 5
        mock_controller_stats.completed_bytes = 4096
        mock_controller_stats.read_bytes_received = 1024

        mock_master = Mock()
        mock_master.controller_stats = mock_controller_stats

        system = Mock(spec=['host_axi_master'])
        system.host_axi_master = mock_master

        metrics = _extract_metrics_legacy(system)

        assert metrics['completed_transactions'] == 25  # 20 + 5
        assert metrics['bytes_transferred'] == 5120  # 4096 + 1024

    def test_extracts_transfer_stats_from_host_axi_master_fallback(self):
        """Should fall back to HostAXIMasterStats if controller_stats missing."""
        mock_stats = Mock()
        mock_stats.completed_transactions = 15
        mock_stats.completed_bytes = 2048

        mock_master = Mock(spec=['stats'])
        mock_master.stats = mock_stats

        system = Mock(spec=['host_axi_master'])
        system.host_axi_master = mock_master

        metrics = _extract_metrics_legacy(system)

        assert metrics['completed_transactions'] == 15
        assert metrics['bytes_transferred'] == 2048

    def test_extracts_transfer_stats_from_node_controllers(self):
        """Should extract transfer stats from node_controllers for NoC-to-NoC."""
        mock_stats1 = Mock()
        mock_stats1.transfers_completed = 10
        mock_stats2 = Mock()
        mock_stats2.transfers_completed = 8

        mock_controller1 = Mock()
        mock_controller1.stats = mock_stats1
        mock_controller2 = Mock()
        mock_controller2.stats = mock_stats2

        mock_traffic_config = Mock()
        mock_traffic_config.transfer_size = 256

        system = Mock(spec=['node_controllers', '_traffic_config'])
        system.node_controllers = {0: mock_controller1, 1: mock_controller2}
        system._traffic_config = mock_traffic_config
        system.host_axi_master = None

        metrics = _extract_metrics_legacy(system)

        assert metrics['completed_transactions'] == 18  # 10 + 8
        assert metrics['bytes_transferred'] == 4608  # 18 * 256
        assert metrics['transfer_size'] == 256

    def test_handles_missing_host_axi_master(self):
        """Should handle None host_axi_master gracefully."""
        system = Mock(spec=['host_axi_master'])
        system.host_axi_master = None

        metrics = _extract_metrics_legacy(system)

        assert metrics['completed_transactions'] == 0
        assert metrics['bytes_transferred'] == 0

    def test_handles_empty_mesh(self):
        """Should handle mesh with no routers."""
        mock_mesh = Mock()
        mock_mesh.routers = {}

        system = Mock(spec=['mesh'])
        system.mesh = mock_mesh

        metrics = _extract_metrics_legacy(system)

        assert metrics['buffer_occupancy'] == {}
        assert metrics['flit_stats'] == {}

    def test_handles_router_without_stats(self):
        """Should handle routers without stats attribute."""
        mock_req_router = Mock(spec=['ports'])
        mock_req_router.ports = {}

        mock_resp_router = Mock(spec=['ports'])
        mock_resp_router.ports = {}

        mock_router = Mock()
        mock_router.req_router = mock_req_router
        mock_router.resp_router = mock_resp_router

        mock_mesh = Mock()
        mock_mesh.routers = {(1, 1): mock_router}

        system = Mock(spec=['mesh'])
        system.mesh = mock_mesh

        metrics = _extract_metrics_legacy(system)

        assert (1, 1) in metrics['buffer_occupancy']

    def test_counts_flits_in_pipeline(self):
        """Should count flits in pipeline stages."""
        mock_req_router = Mock(spec=['ports', 'flits_in_pipeline', 'stats'])
        mock_req_router.ports = {}
        mock_req_router.flits_in_pipeline = 3
        mock_req_router.stats = Mock()
        mock_req_router.stats.flits_forwarded = 0

        mock_resp_router = Mock(spec=['ports', 'flits_in_pipeline', 'stats'])
        mock_resp_router.ports = {}
        mock_resp_router.flits_in_pipeline = 2
        mock_resp_router.stats = Mock()
        mock_resp_router.stats.flits_forwarded = 0

        mock_router = Mock()
        mock_router.req_router = mock_req_router
        mock_router.resp_router = mock_resp_router

        mock_mesh = Mock()
        mock_mesh.routers = {(1, 1): mock_router}

        system = Mock(spec=['mesh'])
        system.mesh = mock_mesh

        metrics = _extract_metrics_legacy(system)

        # Should count pipeline flits: 3 + 2 = 5
        assert metrics['buffer_occupancy'][(1, 1)] == 5

    def test_extracts_buffer_occupancy_from_resp_router_ports(self):
        """Should count resp_router port buffers and out_valid."""
        mock_req_router = Mock(spec=['ports', 'stats'])
        mock_req_router.ports = {}
        mock_req_router.stats = Mock()
        mock_req_router.stats.flits_forwarded = 5

        mock_resp_port = Mock()
        mock_resp_port.input_buffer = Mock()
        mock_resp_port.input_buffer.occupancy = 4
        mock_resp_port.out_valid = True
        mock_resp_port.out_flit = Mock()

        mock_resp_router = Mock(spec=['ports', 'stats'])
        mock_resp_router.ports = {'SOUTH': mock_resp_port}
        mock_resp_router.stats = Mock()
        mock_resp_router.stats.flits_forwarded = 3

        mock_router = Mock()
        mock_router.req_router = mock_req_router
        mock_router.resp_router = mock_resp_router

        mock_mesh = Mock()
        mock_mesh.routers = {(2, 2): mock_router}

        system = Mock(spec=['mesh'])
        system.mesh = mock_mesh

        metrics = _extract_metrics_legacy(system)

        # resp_router: buffer(4) + out_valid(1) = 5
        assert metrics['buffer_occupancy'][(2, 2)] == 5
        assert metrics['flit_stats'][(2, 2)] == 8  # 5 + 3

    def test_extracts_transfer_stats_using_b_r_received_fallback(self):
        """Should fall back to b_received + r_received when completed_transactions is 0."""
        mock_stats = Mock(spec=['completed_transactions', 'completed_bytes',
                                'b_received', 'r_received'])
        mock_stats.completed_transactions = 0  # Forces fallback
        mock_stats.completed_bytes = 0
        mock_stats.b_received = 10
        mock_stats.r_received = 5

        mock_master = Mock(spec=['stats'])
        mock_master.stats = mock_stats

        system = Mock(spec=['host_axi_master'])
        system.host_axi_master = mock_master

        metrics = _extract_metrics_legacy(system)

        assert metrics['completed_transactions'] == 15  # 10 + 5


class TestMetricsReturnStructure:
    """Test that metrics return correct structure."""

    def test_returns_all_required_keys(self):
        """Should return all required keys in metrics dict."""
        system = Mock(spec=[])

        metrics = _extract_metrics_legacy(system)

        required_keys = [
            'cycle',
            'mesh_cols',
            'mesh_rows',
            'buffer_occupancy',
            'flit_stats',
            'completed_transactions',
            'bytes_transferred',
            'transfer_size',
        ]

        for key in required_keys:
            assert key in metrics, f"Missing required key: {key}"

    def test_buffer_occupancy_is_dict(self):
        """buffer_occupancy should be a dict."""
        system = Mock(spec=[])
        metrics = _extract_metrics_legacy(system)
        assert isinstance(metrics['buffer_occupancy'], dict)

    def test_flit_stats_is_dict(self):
        """flit_stats should be a dict."""
        system = Mock(spec=[])
        metrics = _extract_metrics_legacy(system)
        assert isinstance(metrics['flit_stats'], dict)
