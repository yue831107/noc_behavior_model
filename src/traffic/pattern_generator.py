"""
Traffic Pattern Generator for NoC-to-NoC simulation.

Generates per-node transfer configurations based on traffic patterns.
Supports 5 patterns: neighbor, shuffle, bit_reverse, random, transpose.
"""

from __future__ import annotations

from typing import List, Tuple
import random
import math

from ..config import (
    TrafficPattern,
    NoCTrafficConfig,
    NodeTransferConfig,
)


class TrafficPatternGenerator:
    """
    Generate per-node configurations based on traffic pattern.

    Traffic Patterns:
    - NEIGHBOR: Ring pattern, node i -> node (i+1) % N
    - SHUFFLE: Left rotate bits of node ID
    - BIT_REVERSE: Reverse bits of node ID
    - RANDOM: Random destination (deterministic with seed)
    - TRANSPOSE: Swap X and Y coordinates
    """

    def __init__(self, mesh_cols: int = 5, mesh_rows: int = 4):
        """
        Initialize pattern generator.

        Args:
            mesh_cols: Total mesh columns (including edge column).
            mesh_rows: Total mesh rows.
        """
        self.mesh_cols = mesh_cols
        self.mesh_rows = mesh_rows
        self.compute_cols = mesh_cols - 1  # Exclude edge column
        self.num_nodes = self.compute_cols * mesh_rows

    def _node_id_to_coord(self, node_id: int) -> Tuple[int, int]:
        """Convert node ID to (x, y) coordinate."""
        x = (node_id % self.compute_cols) + 1  # +1 for edge column
        y = node_id // self.compute_cols
        return (x, y)

    def _coord_to_node_id(self, coord: Tuple[int, int]) -> int:
        """Convert (x, y) coordinate to node ID."""
        x, y = coord
        if x < 1:  # Edge column has no node ID
            return -1
        return y * self.compute_cols + (x - 1)

    def generate(self, config: NoCTrafficConfig) -> List[NodeTransferConfig]:
        """
        Generate per-node configurations based on traffic pattern.

        Args:
            config: Traffic configuration with pattern and parameters.

        Returns:
            List of NodeTransferConfig for each node.
        """
        # Update mesh dimensions from config
        self.mesh_cols = config.mesh_cols
        self.mesh_rows = config.mesh_rows
        self.compute_cols = config.mesh_cols - 1
        self.num_nodes = self.compute_cols * config.mesh_rows

        pattern = config.pattern

        if pattern == TrafficPattern.NEIGHBOR:
            return self._gen_neighbor(config)
        elif pattern == TrafficPattern.SHUFFLE:
            return self._gen_shuffle(config)
        elif pattern == TrafficPattern.BIT_REVERSE:
            return self._gen_bit_reverse(config)
        elif pattern == TrafficPattern.RANDOM:
            return self._gen_random(config)
        elif pattern == TrafficPattern.TRANSPOSE:
            return self._gen_transpose(config)
        else:
            raise ValueError(f"Unknown traffic pattern: {pattern}")

    def _gen_neighbor(self, config: NoCTrafficConfig) -> List[NodeTransferConfig]:
        """
        Generate NEIGHBOR pattern: ring topology.

        Formula: dst = (src + 1) % N
        Each node sends to the next node in a ring.
        """
        configs = []
        for src_id in range(self.num_nodes):
            dst_id = (src_id + 1) % self.num_nodes
            dst_coord = self._node_id_to_coord(dst_id)

            configs.append(NodeTransferConfig(
                src_node_id=src_id,
                dest_coord=dst_coord,
                local_src_addr=config.local_src_addr,
                local_dst_addr=config.local_dst_addr,
                transfer_size=config.transfer_size,
            ))

        return configs

    def _gen_shuffle(self, config: NoCTrafficConfig) -> List[NodeTransferConfig]:
        """
        Generate SHUFFLE pattern: perfect shuffle permutation.

        Formula: dst = shuffle(src)
        Left rotate bits of source node ID.
        shuffle(abcd) = bcda (for 4-bit ID)
        
        Note: Shuffle is a bijection (1-to-1) when num_nodes is a power of 2.
        """
        configs = []
        # Use exact log2 for power-of-2 node counts to ensure bijection
        if self.num_nodes > 0 and (self.num_nodes & (self.num_nodes - 1)) == 0:
            # Power of 2: use exact bit count
            bits = int(math.log2(self.num_nodes))
        else:
            # Non-power of 2: use ceil
            bits = max(1, math.ceil(math.log2(self.num_nodes + 1)))

        for src_id in range(self.num_nodes):
            # Left rotate by 1 bit within the required bits
            msb = (src_id >> (bits - 1)) & 1
            dst_id = ((src_id << 1) | msb) & ((1 << bits) - 1)

            # Ensure dst_id is within valid range
            dst_id = dst_id % self.num_nodes

            dst_coord = self._node_id_to_coord(dst_id)

            configs.append(NodeTransferConfig(
                src_node_id=src_id,
                dest_coord=dst_coord,
                local_src_addr=config.local_src_addr,
                local_dst_addr=config.local_dst_addr,
                transfer_size=config.transfer_size,
            ))

        return configs

    def _gen_bit_reverse(self, config: NoCTrafficConfig) -> List[NodeTransferConfig]:
        """
        Generate BIT_REVERSE pattern.

        Formula: dst = reverse_bits(src)
        Reverse the bits of the source node ID.
        """
        configs = []
        # Number of bits needed to represent node IDs
        bits = max(1, math.ceil(math.log2(self.num_nodes + 1)))

        for src_id in range(self.num_nodes):
            # Reverse bits
            reversed_id = 0
            temp = src_id
            for _ in range(bits):
                reversed_id = (reversed_id << 1) | (temp & 1)
                temp >>= 1

            # Ensure dst_id is within valid range
            dst_id = reversed_id % self.num_nodes

            dst_coord = self._node_id_to_coord(dst_id)

            configs.append(NodeTransferConfig(
                src_node_id=src_id,
                dest_coord=dst_coord,
                local_src_addr=config.local_src_addr,
                local_dst_addr=config.local_dst_addr,
                transfer_size=config.transfer_size,
            ))

        return configs

    def _gen_random(self, config: NoCTrafficConfig) -> List[NodeTransferConfig]:
        """
        Generate RANDOM pattern.

        Formula: dst = random()
        Each node sends to a random destination (excluding self).
        Uses seed for reproducibility.
        """
        configs = []
        rng = random.Random(config.random_seed)

        for src_id in range(self.num_nodes):
            # Pick random destination (excluding self)
            candidates = [i for i in range(self.num_nodes) if i != src_id]
            dst_id = rng.choice(candidates) if candidates else src_id

            dst_coord = self._node_id_to_coord(dst_id)

            configs.append(NodeTransferConfig(
                src_node_id=src_id,
                dest_coord=dst_coord,
                local_src_addr=config.local_src_addr,
                local_dst_addr=config.local_dst_addr,
                transfer_size=config.transfer_size,
            ))

        return configs

    def _gen_transpose(self, config: NoCTrafficConfig) -> List[NodeTransferConfig]:
        """
        Generate TRANSPOSE pattern.

        Formula: (x, y) -> (y, x)
        Swap X and Y coordinates.

        Note: For non-square meshes, some nodes may map to invalid coordinates.
        In such cases, they send to themselves (self-loop).
        """
        configs = []

        for src_id in range(self.num_nodes):
            src_coord = self._node_id_to_coord(src_id)
            src_x, src_y = src_coord

            # Transpose: (x, y) -> (y+1, x)
            # Note: x starts at 1 due to edge column, so we adjust
            transposed_x = src_y + 1  # y -> x (with edge offset)
            transposed_y = src_x - 1  # x -> y (remove edge offset)

            # Check if transposed coordinate is valid
            if (1 <= transposed_x < self.mesh_cols and
                0 <= transposed_y < self.mesh_rows):
                dst_coord = (transposed_x, transposed_y)
            else:
                # Invalid coordinate - send to self (or could skip)
                dst_coord = src_coord

            configs.append(NodeTransferConfig(
                src_node_id=src_id,
                dest_coord=dst_coord,
                local_src_addr=config.local_src_addr,
                local_dst_addr=config.local_dst_addr,
                transfer_size=config.transfer_size,
            ))

        return configs

    def print_pattern(self, configs: List[NodeTransferConfig]) -> None:
        """Print traffic pattern for debugging."""
        print(f"Traffic Pattern ({len(configs)} nodes):")
        print("-" * 50)
        for cfg in configs:
            src_coord = self._node_id_to_coord(cfg.src_node_id)
            print(f"  Node {cfg.src_node_id:2d} {src_coord} -> {cfg.dest_coord}")
        print("-" * 50)
