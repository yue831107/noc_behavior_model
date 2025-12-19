"""
Metrics Provider Protocol.

Defines a standard interface for systems to expose metrics data.
This decouples visualization from core implementation details.
"""

from __future__ import annotations

from typing import Protocol, Dict, Tuple, List, runtime_checkable


@runtime_checkable
class MetricsProvider(Protocol):
    """
    Protocol for systems that provide metrics for visualization.
    
    Systems (V1System, NoCSystem) should implement these methods
    to support MetricsCollector without tight coupling.
    """
    
    @property
    def current_cycle(self) -> int:
        """Current simulation cycle."""
        ...
    
    @property
    def mesh_dimensions(self) -> Tuple[int, int]:
        """Return (cols, rows) of the mesh."""
        ...
    
    def get_buffer_occupancy(self) -> Dict[Tuple[int, int], int]:
        """
        Get buffer occupancy for each router.
        
        Returns:
            Dict of (x, y) coordinate -> total buffer occupancy.
        """
        ...
    
    def get_flit_stats(self) -> Dict[Tuple[int, int], int]:
        """
        Get flit forwarding stats for each router.
        
        Returns:
            Dict of (x, y) coordinate -> flits forwarded count.
        """
        ...
    
    def get_transfer_stats(self) -> Tuple[int, int, int]:
        """
        Get transfer completion statistics.
        
        Returns:
            Tuple of (completed_transactions, bytes_transferred, transfer_size).
            transfer_size is per-transaction size (0 if unknown).
        """
        ...


def get_metrics_from_system(system) -> Dict:
    """
    Safely extract metrics from a system, regardless of implementation.
    
    Works with both Protocol-implementing systems and legacy systems.
    
    Args:
        system: V1System, NoCSystem, or any MetricsProvider.
    
    Returns:
        Dict with keys: cycle, buffer_occupancy, flit_stats, transfer_stats
    """
    # Try Protocol methods first
    if isinstance(system, MetricsProvider):
        cols, rows = system.mesh_dimensions
        completed, bytes_transferred, transfer_size = system.get_transfer_stats()
        return {
            'cycle': system.current_cycle,
            'mesh_cols': cols,
            'mesh_rows': rows,
            'buffer_occupancy': system.get_buffer_occupancy(),
            'flit_stats': system.get_flit_stats(),
            'completed_transactions': completed,
            'bytes_transferred': bytes_transferred,
            'transfer_size': transfer_size,
        }
    
    # Fallback for legacy systems
    return _extract_metrics_legacy(system)


def _extract_metrics_legacy(system) -> Dict:
    """
    Extract metrics from legacy systems without Protocol.
    
    This is the fallback path for backward compatibility.
    """
    # Get cycle
    if hasattr(system, 'current_cycle'):
        cycle = system.current_cycle
    elif hasattr(system, 'current_time'):
        cycle = system.current_time
    else:
        cycle = 0
    
    # Get mesh dimensions
    if hasattr(system, 'mesh_cols'):
        mesh_cols = system.mesh_cols
    elif hasattr(system, '_mesh_cols'):
        mesh_cols = system._mesh_cols
    else:
        mesh_cols = 5
    
    if hasattr(system, 'mesh_rows'):
        mesh_rows = system.mesh_rows
    elif hasattr(system, '_mesh_rows'):
        mesh_rows = system._mesh_rows
    else:
        mesh_rows = 4
    
    # Get buffer occupancy
    buffer_occupancy = {}
    flit_stats = {}
    
    if hasattr(system, 'mesh') and hasattr(system.mesh, 'routers'):
        for coord, router in system.mesh.routers.items():
            occupancy = 0
            flit_count = 0
            
            if hasattr(router, 'req_router'):
                for port in router.req_router.ports.values():
                    if hasattr(port, 'input_buffer'):
                        occupancy += port.input_buffer.occupancy
                if hasattr(router.req_router, 'stats'):
                    flit_count = router.req_router.stats.flits_forwarded
            
            if hasattr(router, 'resp_router'):
                for port in router.resp_router.ports.values():
                    if hasattr(port, 'input_buffer'):
                        occupancy += port.input_buffer.occupancy
            
            buffer_occupancy[coord] = occupancy
            flit_stats[coord] = flit_count
    
    # Get transfer stats
    completed = 0
    bytes_transferred = 0
    transfer_size = 0
    
    if hasattr(system, 'host_axi_master') and system.host_axi_master:
        stats = system.host_axi_master.stats
        completed = stats.completed_transactions
        bytes_transferred = stats.completed_bytes
    elif hasattr(system, 'node_controllers'):
        for controller in system.node_controllers.values():
            if hasattr(controller, 'stats'):
                completed += controller.stats.transfers_completed
        
        if hasattr(system, '_traffic_config') and system._traffic_config:
            transfer_size = system._traffic_config.transfer_size
            bytes_transferred = completed * transfer_size
    
    return {
        'cycle': cycle,
        'mesh_cols': mesh_cols,
        'mesh_rows': mesh_rows,
        'buffer_occupancy': buffer_occupancy,
        'flit_stats': flit_stats,
        'completed_transactions': completed,
        'bytes_transferred': bytes_transferred,
        'transfer_size': transfer_size,
    }
