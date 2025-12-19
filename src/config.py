"""
Configuration loader for NoC simulation.

Loads YAML configuration files for transfer operations.
"""

from __future__ import annotations
from dataclasses import dataclass
from typing import Dict, Any, List, Union, Tuple
from pathlib import Path
from enum import Enum
import yaml


class TransferMode(Enum):
    """Transfer mode for memory copy operations."""
    # Write modes (Host → NoC nodes)
    BROADCAST = "broadcast"  # Same data to all nodes
    SCATTER = "scatter"      # Distribute data across nodes

    # Read modes (NoC nodes → Host)
    BROADCAST_READ = "broadcast_read"  # Read same addr from all nodes
    GATHER = "gather"                   # Collect different data from each node

    @property
    def is_read(self) -> bool:
        """Check if this is a read mode."""
        return self in (TransferMode.BROADCAST_READ, TransferMode.GATHER)

    @property
    def is_write(self) -> bool:
        """Check if this is a write mode."""
        return self in (TransferMode.BROADCAST, TransferMode.SCATTER)


class TrafficPattern(Enum):
    """
    NoC-to-NoC traffic patterns.

    These patterns define how nodes communicate with each other.
    Each pattern maps source node to destination node.
    """
    # Ring pattern: node i -> node (i+1) % N
    NEIGHBOR = "neighbor"

    # Shuffle pattern: left rotate node ID bits
    SHUFFLE = "shuffle"

    # Bit reverse: reverse bits of node ID
    BIT_REVERSE = "bit_reverse"

    # Random: random destination for each source
    RANDOM = "random"

    # Transpose: swap (x, y) coordinates -> (y, x)
    TRANSPOSE = "transpose"


@dataclass
class TimingConfig:
    """
    Timing configuration for cycle-accurate simulation.
    
    Controls the injection rate of various components to model
    hardware-realistic timing behavior.
    
    Default values (1 per cycle) match typical hardware constraints.
    Set to higher values to speed up simulation at cost of accuracy.
    """
    # AXI Master rates (per cycle)
    aw_per_cycle: int = 1      # AW (Write Address) channels per cycle
    w_per_cycle: int = 1       # W (Write Data) beats per cycle
    ar_per_cycle: int = 1      # AR (Read Address) channels per cycle
    
    # NI rates (per cycle)
    flits_per_cycle: int = 1   # Flit injection/ejection rate
    
    # Transaction generation rates
    txn_per_cycle: int = 1     # Transactions generated per cycle
    
    @classmethod
    def fast(cls) -> "TimingConfig":
        """Fast mode: unlimited injection (original behavior for quick testing)."""
        return cls(
            aw_per_cycle=999,
            w_per_cycle=999,
            ar_per_cycle=999,
            flits_per_cycle=999,
            txn_per_cycle=999,
        )
    
    @classmethod
    def hardware(cls) -> "TimingConfig":
        """Hardware mode: 1 per cycle (cycle-accurate)."""
        return cls()  # Defaults are already hardware-accurate


@dataclass
class TransferConfig:
    """
    Transfer configuration for AXI Master (DMA-like) operations.

    Used to configure file-driven simulation for hardware verification.
    Supports both write (BROADCAST, SCATTER) and read (BROADCAST_READ, GATHER) modes.
    """
    # Source settings (Host Memory) - for write operations
    src_addr: int = 0x0000          # Start address in Host Memory
    src_size: int = 1024            # Total data size in bytes

    # Destination settings (Local Memory) - for write operations
    dst_addr: int = 0x1000          # Destination address in Local Memory
    target_nodes: Union[str, List[int]] = "all"  # "all" | [0,1,2] | "range:0-7"

    # Read settings (Local Memory → Host Memory) - for read operations
    read_src_addr: int = 0x1000     # Address to read from in Local Memory
    read_size: int = 0              # Size to read (0 = same as src_size)

    # AXI Burst settings
    max_burst_len: int = 16         # Max burst length (1-256, AXI spec)
    beat_size: int = 8              # Bytes per beat (1/2/4/8/16/32/64/128)

    # Flow control
    max_outstanding: int = 8        # Max outstanding transactions

    # Transfer mode
    transfer_mode: TransferMode = TransferMode.BROADCAST

    # Advanced options
    interleave_nodes: int = 1       # Node interleaving (1=sequential)
    inter_txn_delay: int = 0        # Delay cycles between transactions

    @property
    def is_read(self) -> bool:
        """Check if this is a read transfer."""
        return self.transfer_mode.is_read

    @property
    def is_write(self) -> bool:
        """Check if this is a write transfer."""
        return self.transfer_mode.is_write

    @property
    def effective_read_size(self) -> int:
        """Get effective read size (uses src_size if read_size is 0)."""
        return self.read_size if self.read_size > 0 else self.src_size

    def get_target_node_list(self, total_nodes: int = 16) -> List[int]:
        """
        Parse target_nodes into a list of node IDs.

        Args:
            total_nodes: Total number of nodes in the mesh.

        Returns:
            List of target node IDs.
        """
        if isinstance(self.target_nodes, list):
            return self.target_nodes

        if self.target_nodes == "all":
            return list(range(total_nodes))

        if isinstance(self.target_nodes, str):
            if self.target_nodes.startswith("range:"):
                # Parse "range:0-7" format
                range_str = self.target_nodes[6:]
                start, end = map(int, range_str.split("-"))
                return list(range(start, end + 1))

        raise ValueError(f"Invalid target_nodes format: {self.target_nodes}")

    def validate(self) -> None:
        """Validate configuration values."""
        if self.src_size <= 0:
            raise ValueError("src_size must be positive")
        if self.max_burst_len < 1 or self.max_burst_len > 256:
            raise ValueError("max_burst_len must be 1-256")
        if self.beat_size not in (1, 2, 4, 8, 16, 32, 64, 128):
            raise ValueError("beat_size must be power of 2 (1-128)")
        if self.max_outstanding < 1:
            raise ValueError("max_outstanding must be at least 1")

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            "transfer": {
                "src_addr": self.src_addr,
                "src_size": self.src_size,
                "dst_addr": self.dst_addr,
                "target_nodes": self.target_nodes,
                "read_src_addr": self.read_src_addr,
                "read_size": self.read_size,
                "max_burst_len": self.max_burst_len,
                "beat_size": self.beat_size,
                "max_outstanding": self.max_outstanding,
                "transfer_mode": self.transfer_mode.value,
                "interleave_nodes": self.interleave_nodes,
                "inter_txn_delay": self.inter_txn_delay,
            }
        }

    def save(self, path: Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


def load_transfer_config(config_path: str | Path) -> TransferConfig:
    """
    Load transfer configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        TransferConfig instance.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Transfer config file not found: {config_path}")

    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    return _parse_transfer_config(data)


def _parse_transfer_config(data: Dict[str, Any]) -> TransferConfig:
    """Parse YAML data into TransferConfig."""
    transfer = data.get("transfer", data)  # Support both nested and flat format

    config = TransferConfig(
        src_addr=transfer.get("src_addr", 0x0000),
        src_size=transfer.get("src_size", 1024),
        dst_addr=transfer.get("dst_addr", 0x1000),
        target_nodes=transfer.get("target_nodes", "all"),
        read_src_addr=transfer.get("read_src_addr", 0x1000),
        read_size=transfer.get("read_size", 0),
        max_burst_len=transfer.get("max_burst_len", 16),
        beat_size=transfer.get("beat_size", 8),
        max_outstanding=transfer.get("max_outstanding", 8),
        transfer_mode=TransferMode(
            transfer.get("transfer_mode", "broadcast")
        ),
        interleave_nodes=transfer.get("interleave_nodes", 1),
        inter_txn_delay=transfer.get("inter_txn_delay", 0),
    )

    config.validate()
    return config


def get_default_transfer_config() -> TransferConfig:
    """Get default transfer configuration."""
    return TransferConfig()


def load_transfer_configs(config_path: str | Path) -> List[TransferConfig]:
    """
    Load single or multiple transfer configurations from YAML file.

    Supports two formats:
    1. Single transfer (backward compatible):
       ```yaml
       transfer:
         src_addr: 0x0000
         src_size: 1024
       ```

    2. Multiple transfers:
       ```yaml
       transfers:
         - src_addr: 0x0000
           src_size: 1024
           target_nodes: [0, 1, 2, 3]
         - src_addr: 0x1000
           src_size: 512
           target_nodes: [4, 5, 6, 7]
       ```

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        List of TransferConfig instances.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Transfer config file not found: {config_path}")

    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    # Check for multi-transfer format
    if 'transfers' in data:
        configs = []
        for transfer_data in data['transfers']:
            config = _parse_single_transfer(transfer_data)
            configs.append(config)
        return configs

    # Single transfer format (backward compatible)
    return [_parse_transfer_config(data)]


def _parse_single_transfer(transfer: Dict[str, Any]) -> TransferConfig:
    """Parse a single transfer dict (no 'transfer' wrapper)."""
    
    def parse_addr(val, default: int) -> int:
        """Parse address value (int or hex string like '0x0000')."""
        if val is None:
            return default
        if isinstance(val, int):
            return val
        if isinstance(val, str):
            return int(val, 0)  # Auto-detect base (0x for hex)
        return default
    
    config = TransferConfig(
        src_addr=parse_addr(transfer.get("src_addr"), 0x0000),
        src_size=transfer.get("src_size", 1024),
        dst_addr=parse_addr(transfer.get("dst_addr"), 0x1000),
        target_nodes=transfer.get("target_nodes", "all"),
        read_src_addr=parse_addr(transfer.get("read_src_addr"), 0x1000),
        read_size=transfer.get("read_size", 0),
        max_burst_len=transfer.get("max_burst_len", 16),
        beat_size=transfer.get("beat_size", 8),
        max_outstanding=transfer.get("max_outstanding", 8),
        transfer_mode=TransferMode(
            transfer.get("transfer_mode", "broadcast")
        ),
        interleave_nodes=transfer.get("interleave_nodes", 1),
        inter_txn_delay=transfer.get("inter_txn_delay", 0),
    )
    config.validate()
    return config


# =============================================================================
# NoC-to-NoC Traffic Configuration
# =============================================================================

@dataclass
class NodeTransferConfig:
    """
    Per-node transfer configuration for NoC-to-NoC traffic.

    Each node has independent configuration for its outgoing transfer.
    """
    src_node_id: int                    # Source node ID
    dest_coord: Tuple[int, int]         # Destination (x, y) coordinate
    local_src_addr: int = 0x0000        # Source address in local memory
    local_dst_addr: int = 0x1000        # Destination address at target
    transfer_size: int = 256            # Bytes to transfer

    def encode_user_signal(self) -> int:
        """Encode destination coordinate into AXI user signal."""
        dest_x, dest_y = self.dest_coord
        return (dest_y << 8) | dest_x


@dataclass
class NoCTrafficConfig:
    """
    Configuration for NoC-to-NoC traffic simulation.

    Supports traffic pattern-based configuration where each node's
    destination is computed from the pattern formula.
    """
    # Traffic pattern
    pattern: TrafficPattern = TrafficPattern.NEIGHBOR

    # Mesh dimensions
    mesh_cols: int = 5
    mesh_rows: int = 4

    # Transfer parameters (applied to all nodes)
    transfer_size: int = 256
    local_src_addr: int = 0x0000
    local_dst_addr: int = 0x1000

    # Per-node configs (auto-generated from pattern if None)
    node_configs: List[NodeTransferConfig] = None

    # Random seed (for RANDOM pattern reproducibility)
    random_seed: int = 42

    def __post_init__(self):
        """Initialize node_configs as empty list if None."""
        if self.node_configs is None:
            self.node_configs = []

    @property
    def num_nodes(self) -> int:
        """Number of compute nodes (excluding edge column)."""
        return (self.mesh_cols - 1) * self.mesh_rows

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for YAML serialization."""
        return {
            "traffic": {
                "pattern": self.pattern.value,
                "mesh_cols": self.mesh_cols,
                "mesh_rows": self.mesh_rows,
                "transfer_size": self.transfer_size,
                "local_src_addr": self.local_src_addr,
                "local_dst_addr": self.local_dst_addr,
                "random_seed": self.random_seed,
            }
        }

    def save(self, path: Path) -> None:
        """Save configuration to YAML file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, 'w', encoding='utf-8') as f:
            yaml.dump(self.to_dict(), f, default_flow_style=False, sort_keys=False)


def load_noc_traffic_config(config_path: str | Path) -> NoCTrafficConfig:
    """
    Load NoC traffic configuration from YAML file.

    Args:
        config_path: Path to YAML configuration file.

    Returns:
        NoCTrafficConfig instance.
    """
    path = Path(config_path)
    if not path.exists():
        raise FileNotFoundError(f"Traffic config file not found: {config_path}")

    with open(path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)

    return _parse_noc_traffic_config(data)


def _parse_noc_traffic_config(data: Dict[str, Any]) -> NoCTrafficConfig:
    """Parse YAML data into NoCTrafficConfig."""
    traffic = data.get("traffic", data)

    return NoCTrafficConfig(
        pattern=TrafficPattern(traffic.get("pattern", "neighbor")),
        mesh_cols=traffic.get("mesh_cols", 5),
        mesh_rows=traffic.get("mesh_rows", 4),
        transfer_size=traffic.get("transfer_size", 256),
        local_src_addr=traffic.get("local_src_addr", 0x0000),
        local_dst_addr=traffic.get("local_dst_addr", 0x1000),
        random_seed=traffic.get("random_seed", 42),
    )
