"""
System Address Map for NoC.

Handles translation between 64-bit AXI addresses and
32-bit local addresses + NoC coordinates.

Address Format (64-bit):
    [63:40] Reserved (should be 0)
    [39:32] Node ID (8-bit, 0-255)
    [31:0]  Local Address (32-bit)
"""

from __future__ import annotations
from dataclasses import dataclass, field
from typing import Optional, Dict, Tuple


@dataclass
class AddressMapConfig:
    """Configuration for address map."""
    mesh_cols: int = 5          # Total columns (including edge)
    mesh_rows: int = 4          # Total rows
    edge_column: int = 0        # Edge router column (no NI)
    node_id_bits: int = 8       # Bits for node ID field
    local_addr_bits: int = 32   # Bits for local address


class SystemAddressMap:
    """
    System Address Map for 64-bit to 32-bit + coordinate translation.

    Maps Node IDs to NoC coordinates for compute nodes (nodes with NI).
    Edge routers (column 0) do not have Node IDs as they don't have NI.

    Example for 5x4 mesh with edge_column=0:
        Node ID 0  → (1, 0)    Node ID 4  → (1, 1)
        Node ID 1  → (2, 0)    Node ID 5  → (2, 1)
        Node ID 2  → (3, 0)    Node ID 6  → (3, 1)
        Node ID 3  → (4, 0)    Node ID 7  → (4, 1)
        ...
    """

    def __init__(self, config: Optional[AddressMapConfig] = None):
        """
        Initialize address map.

        Args:
            config: Address map configuration.
        """
        self.config = config or AddressMapConfig()
        self._node_to_coord: Dict[int, Tuple[int, int]] = {}
        self._coord_to_node: Dict[Tuple[int, int], int] = {}
        self._build_map()

    def _build_map(self) -> None:
        """Build Node ID to coordinate mapping."""
        node_id = 0
        for y in range(self.config.mesh_rows):
            for x in range(self.config.edge_column + 1, self.config.mesh_cols):
                self._node_to_coord[node_id] = (x, y)
                self._coord_to_node[(x, y)] = node_id
                node_id += 1

        self._max_node_id = node_id - 1
        self._num_nodes = node_id

    @property
    def num_nodes(self) -> int:
        """Total number of compute nodes."""
        return self._num_nodes

    @property
    def max_node_id(self) -> int:
        """Maximum valid node ID."""
        return self._max_node_id

    def translate(self, axi_addr: int) -> Tuple[Tuple[int, int], int]:
        """
        Translate 64-bit AXI address to (coordinate, local_address).

        Args:
            axi_addr: 64-bit AXI address.

        Returns:
            Tuple of (dest_coord, local_addr).

        Raises:
            ValueError: If node ID is invalid.
        """
        node_id = self.extract_node_id(axi_addr)
        local_addr = self.extract_local_addr(axi_addr)

        if node_id not in self._node_to_coord:
            raise ValueError(
                f"Invalid Node ID: {node_id} "
                f"(valid range: 0-{self._max_node_id})"
            )

        dest_coord = self._node_to_coord[node_id]
        return dest_coord, local_addr

    def extract_node_id(self, axi_addr: int) -> int:
        """
        Extract Node ID from 64-bit AXI address.

        Args:
            axi_addr: 64-bit AXI address.

        Returns:
            Node ID (8-bit).
        """
        mask = (1 << self.config.node_id_bits) - 1
        return (axi_addr >> self.config.local_addr_bits) & mask

    def extract_local_addr(self, axi_addr: int) -> int:
        """
        Extract local address from 64-bit AXI address.

        Args:
            axi_addr: 64-bit AXI address.

        Returns:
            Local address (32-bit).
        """
        mask = (1 << self.config.local_addr_bits) - 1
        return axi_addr & mask

    def build_axi_addr(self, node_id: int, local_addr: int) -> int:
        """
        Build 64-bit AXI address from node ID and local address.

        Args:
            node_id: Target node ID.
            local_addr: Local address at target.

        Returns:
            64-bit AXI address.
        """
        return (node_id << self.config.local_addr_bits) | (
            local_addr & ((1 << self.config.local_addr_bits) - 1)
        )

    def coord_to_axi_addr(
        self, coord: Tuple[int, int], local_addr: int
    ) -> int:
        """
        Build 64-bit AXI address from coordinate and local address.

        Args:
            coord: Target coordinate (x, y).
            local_addr: Local address at target.

        Returns:
            64-bit AXI address.

        Raises:
            ValueError: If coordinate is not a valid compute node.
        """
        if coord not in self._coord_to_node:
            raise ValueError(f"Invalid compute node coordinate: {coord}")

        node_id = self._coord_to_node[coord]
        return self.build_axi_addr(node_id, local_addr)

    def get_coord(self, node_id: int) -> Tuple[int, int]:
        """
        Get coordinate for a node ID.

        Args:
            node_id: Node ID.

        Returns:
            Coordinate (x, y).

        Raises:
            ValueError: If node ID is invalid.
        """
        if node_id not in self._node_to_coord:
            raise ValueError(f"Invalid Node ID: {node_id}")
        return self._node_to_coord[node_id]

    def get_node_id(self, coord: Tuple[int, int]) -> int:
        """
        Get node ID for a coordinate.

        Args:
            coord: Coordinate (x, y).

        Returns:
            Node ID.

        Raises:
            ValueError: If coordinate is not a valid compute node.
        """
        if coord not in self._coord_to_node:
            raise ValueError(f"Invalid compute node coordinate: {coord}")
        return self._coord_to_node[coord]

    def is_valid_node_id(self, node_id: int) -> bool:
        """Check if node ID is valid."""
        return node_id in self._node_to_coord

    def is_compute_node(self, coord: Tuple[int, int]) -> bool:
        """Check if coordinate is a compute node (has NI)."""
        return coord in self._coord_to_node

    def is_edge_router(self, coord: Tuple[int, int]) -> bool:
        """Check if coordinate is an edge router (no NI)."""
        x, y = coord
        return (
            x == self.config.edge_column
            and 0 <= y < self.config.mesh_rows
        )

    def get_all_compute_nodes(self) -> list[Tuple[int, int]]:
        """Get all compute node coordinates."""
        return list(self._coord_to_node.keys())

    def get_all_edge_routers(self) -> list[Tuple[int, int]]:
        """Get all edge router coordinates."""
        return [
            (self.config.edge_column, y)
            for y in range(self.config.mesh_rows)
        ]

    def __repr__(self) -> str:
        return (
            f"SystemAddressMap("
            f"nodes={self._num_nodes}, "
            f"mesh={self.config.mesh_cols}x{self.config.mesh_rows})"
        )

    def print_map(self) -> None:
        """Print the address map for debugging."""
        print(f"System Address Map ({self.config.mesh_cols}x{self.config.mesh_rows} mesh)")
        print(f"Edge column: {self.config.edge_column}")
        print(f"Compute nodes: {self._num_nodes}")
        print()
        print("Node ID → Coordinate mapping:")
        for node_id in sorted(self._node_to_coord.keys()):
            coord = self._node_to_coord[node_id]
            axi_addr = self.build_axi_addr(node_id, 0)
            print(f"  Node {node_id:3d} → {coord} (base addr: 0x{axi_addr:016X})")


@dataclass
class AddressTranslator:
    """
    Address translator for NI.

    Wraps SystemAddressMap with caching and validation.
    """
    address_map: SystemAddressMap
    _cache: Dict[int, Tuple[Tuple[int, int], int]] = field(
        default_factory=dict, repr=False
    )

    def translate(self, axi_addr: int) -> Tuple[Tuple[int, int], int]:
        """
        Translate AXI address with caching.

        Args:
            axi_addr: 64-bit AXI address.

        Returns:
            Tuple of (dest_coord, local_addr).
        """
        # Check cache (only cache by node_id portion)
        node_portion = axi_addr >> self.address_map.config.local_addr_bits

        if node_portion not in self._cache:
            coord, _ = self.address_map.translate(axi_addr)
            self._cache[node_portion] = (coord, 0)

        coord = self._cache[node_portion][0]
        local_addr = self.address_map.extract_local_addr(axi_addr)
        return coord, local_addr

    def clear_cache(self) -> None:
        """Clear translation cache."""
        self._cache.clear()


# =============================================================================
# Convenience functions
# =============================================================================

def create_default_address_map() -> SystemAddressMap:
    """Create address map with default configuration (5x4 mesh)."""
    return SystemAddressMap(AddressMapConfig())


def create_address_map(
    mesh_cols: int,
    mesh_rows: int,
    edge_column: int = 0
) -> SystemAddressMap:
    """
    Create address map with custom configuration.

    Args:
        mesh_cols: Number of columns.
        mesh_rows: Number of rows.
        edge_column: Edge router column.

    Returns:
        Configured SystemAddressMap.
    """
    config = AddressMapConfig(
        mesh_cols=mesh_cols,
        mesh_rows=mesh_rows,
        edge_column=edge_column,
    )
    return SystemAddressMap(config)
