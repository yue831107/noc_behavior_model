# NoC-to-NoC Traffic Communication

This document describes the Node-to-Node (NoC-to-NoC) communication architecture and traffic patterns.

## Overview

NoC-to-NoC mode enables direct communication between compute nodes, where each node can initiate transfers to other nodes. This differs from Host-to-NoC mode where only the host can initiate transfers.

| Aspect | Host-to-NoC | NoC-to-NoC |
|--------|-------------|------------|
| Address Space | 64-bit (node_id << 32 \| local_addr) | 32-bit local address |
| Destination | Encoded in address bits [63:32] | Encoded in AXI user signal |
| Initiator | Single HostAXIMaster | Multiple LocalAXIMasters (one per node) |
| Configuration | Single TransferConfig | Per-node NodeTransferConfig |

---

## Architecture

### Node Controller

Each compute node is wrapped by a `NodeController` that provides bidirectional communication:

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                           NodeController                                     │
│  ┌─────────────────────────────────────────────────────────────────────────┐│
│  │                    Router (ReqRouter + RespRouter)                      ││
│  └────────────────────────────┬──────────────────────┬─────────────────────┘│
│                               │                      │                      │
│               ┌───────────────┘                      └───────────────┐      │
│               ▼                                                      ▼      │
│  ┌─────────────────────────┐                        ┌─────────────────────┐ │
│  │  MasterNI               │                        │  SlaveNI            │ │
│  │  (Receive from others)  │                        │  (Send to others)   │ │
│  └───────────┬─────────────┘                        └──────────┬──────────┘ │
│              │                                                  │           │
│              ▼                                                  │           │
│  ┌─────────────────────────────────────────────────────────────┐│           │
│  │                    LocalMemory (64KB)                       ││           │
│  └─────────────────────────────┬───────────────────────────────┘│           │
│                                │                                 │           │
│                                ▲                                 │           │
│  ┌─────────────────────────────┴─────────────────────────────────┘          │
│  │  LocalAXIMaster                                                          │
│  │  - Reads from LocalMemory[src_addr]                                      │
│  │  - Sends to destination via SlaveNI                                      │
│  │  - Destination encoded in awuser: [7:0]=x, [15:8]=y                      │
│  └──────────────────────────────────────────────────────────────────────────┘
└─────────────────────────────────────────────────────────────────────────────┘
```

### Key Components

| Component | File | Purpose |
|-----------|------|---------|
| LocalAXIMaster | `src/core/local_axi_master.py` | Initiates local transfers |
| NodeController | `src/core/node_controller.py` | Wraps node components |
| TrafficPatternGenerator | `src/traffic/pattern_generator.py` | Generates per-node configs |
| NoCSystem | `src/core/routing_selector.py` | Manages all node controllers |
| NoCTrafficConfig | `src/config.py` | Traffic pattern configuration |

---

## AXI User Signal Routing

In NoC-to-NoC mode, destination coordinates are encoded in the AXI user signal instead of the address:

```
awuser[7:0]  = dest_x (X coordinate)
awuser[15:8] = dest_y (Y coordinate)
```

### Encoding/Decoding

```python
# Encoding
def encode_user_signal(dest_coord: Tuple[int, int]) -> int:
    dest_x, dest_y = dest_coord
    return (dest_y << 8) | dest_x

# Decoding
def decode_user_signal(user_signal: int) -> Tuple[int, int]:
    dest_x = user_signal & 0xFF
    dest_y = (user_signal >> 8) & 0xFF
    return (dest_x, dest_y)
```

### SlaveNI Modification

SlaveNI uses `use_user_signal_routing` flag to select routing mode:

```python
class SlaveNI:
    def process_aw(self, aw: AWChannel) -> Flit:
        if self.config.use_user_signal_routing:
            # NoC-to-NoC: Extract destination from user signal
            awuser = getattr(aw, 'awuser', 0) or 0
            dest_x = awuser & 0xFF
            dest_y = (awuser >> 8) & 0xFF
            dest_coord = (dest_x, dest_y)
            local_addr = aw.awaddr & 0xFFFFFFFF
        else:
            # Host-to-NoC: Extract destination from address
            node_id = (aw.awaddr >> 32) & 0xFFFFFFFF
            dest_coord = node_id_to_coord(node_id)
            local_addr = aw.awaddr & 0xFFFFFFFF
```

---

## Traffic Patterns

Five standard traffic patterns are supported:

### 1. NEIGHBOR (Ring Topology)

Each node sends to its next neighbor in a ring:

```
Formula: dst = (src + 1) % N

Example (16 nodes):
  0 → 1 → 2 → 3 → 4 → ... → 15 → 0
```

### 2. SHUFFLE (Perfect Shuffle)

Left rotate the bits of node ID:

```
Formula: dst = ((src << 1) | (src >> (log2(N)-1))) % N

Example (16 nodes, 4-bit):
  0 (0000) → 0 (0000)
  1 (0001) → 2 (0010)
  2 (0010) → 4 (0100)
  3 (0011) → 6 (0110)
  7 (0111) → 14 (1110)
```

### 3. BIT_REVERSE

Reverse the bits of node ID:

```
Formula: dst = reverse_bits(src, log2(N))

Example (16 nodes, 4-bit):
  0 (0000) → 0 (0000)
  1 (0001) → 8 (1000)
  2 (0010) → 4 (0100)
  3 (0011) → 12 (1100)
  5 (0101) → 10 (1010)
```

### 4. RANDOM

Random destination excluding self:

```
Formula: dst = random() where dst != src

Properties:
- Deterministic with seed (reproducible)
- Self-loops excluded
```

### 5. TRANSPOSE

Swap X and Y coordinates:

```
Formula: (x, y) → (y, x)

Example (5x4 mesh):
  (1, 0) → (1, 0)  // x=1, y=0 → x=0+1, y=1-1
  (2, 1) → (2, 1)  // Diagonal stays same
  (3, 2) → (3, 2)  // After adjustment for edge column
```

---

## Configuration

### NoCTrafficConfig

```python
@dataclass
class NoCTrafficConfig:
    pattern: TrafficPattern = TrafficPattern.NEIGHBOR
    mesh_cols: int = 5
    mesh_rows: int = 4
    transfer_size: int = 256
    local_src_addr: int = 0x0000
    local_dst_addr: int = 0x1000
    random_seed: int = 42
    node_configs: Optional[List[NodeTransferConfig]] = None
```

### NodeTransferConfig

```python
@dataclass
class NodeTransferConfig:
    src_node_id: int
    dest_coord: Tuple[int, int]
    local_src_addr: int = 0x0000
    local_dst_addr: int = 0x1000
    transfer_size: int = 256
```

### YAML Configuration

```yaml
traffic:
  pattern: "neighbor"
  mesh_cols: 5
  mesh_rows: 4
  transfer_size: 256
  local_src_addr: 0x0000
  local_dst_addr: 0x1000
  random_seed: 42
```

---

## Data Flow

### Transfer Sequence

```
1. TrafficPatternGenerator creates NodeTransferConfig for each node
2. NoCSystem configures each NodeController with its config
3. Initialize each node's LocalMemory with unique data
4. All nodes start transfers simultaneously
5. LocalAXIMaster reads from local src_addr
6. Creates AXI Write with dest_coord in awuser
7. SlaveNI converts to flits with destination
8. Flits route through mesh to destination
9. Destination MasterNI writes to local dst_addr
```

### Golden Verification Flow

NoC-to-NoC uses `GoldenManager` for data verification with the following flow:

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    NoC-to-NoC Golden Verification Flow                   │
├─────────────────────────────────────────────────────────────────────────┤
│                                                                         │
│  1. configure_traffic(config)                                           │
│     └─→ Generate NodeTransferConfig for each node                       │
│                                                                         │
│  2. initialize_node_memory(pattern="sequential")                        │
│     └─→ Node[i].LocalMemory[src_addr] = unique_data(i)                  │
│                                                                         │
│  3. generate_golden()  ←─────────────────────────────────────┐          │
│     │                                                        │          │
│     │  For each NodeTransferConfig:                          │          │
│     │    src_node_id = nc.src_node_id                        │          │
│     │    dst_node_id = coord_to_id(nc.dest_coord)            │          │
│     │    expected = Node[src].LocalMemory[src_addr]  ────────┘          │
│     │                                                                   │
│     └─→ GoldenManager.store[(dst_node_id, dst_addr)] = expected         │
│                                                                         │
│  4. start_all_transfers() + run_until_complete()                        │
│     └─→ Data flows through mesh (if wired)                              │
│                                                                         │
│  5. verify_transfers()                                                  │
│     │                                                                   │
│     │  For each NodeTransferConfig:                                     │
│     │    actual = Node[dst].LocalMemory[dst_addr]                       │
│     │    expected = GoldenManager.get(dst_node_id, dst_addr)            │
│     │                                                                   │
│     └─→ Compare: expected == actual → PASS/FAIL                         │
│                                                                         │
└─────────────────────────────────────────────────────────────────────────┘
```

### Golden Key Format

| Aspect | Host-to-NoC | NoC-to-NoC |
|--------|-------------|------------|
| Golden Source | HostMemory[src_addr] | Node[src].LocalMemory[src_addr] |
| Golden Target | Node[dst].LocalMemory[dst_addr] | Node[dst].LocalMemory[dst_addr] |
| Key Format | `(dst_node_id, dst_addr)` | `(dst_node_id, dst_addr)` |
| Capture Timing | When HostAXIMaster initiates | Before simulation starts |
| Collision | Possible (BROADCAST to all) | Possible (multiple → same dst) |

### GoldenManager.generate_noc_golden()

```python
def generate_noc_golden(
    self,
    node_configs: List[NodeTransferConfig],
    get_node_memory: Callable[[int], Memory],
    mesh_cols: int = 5,
) -> int:
    """
    Generate golden data for NoC-to-NoC traffic.

    For each transfer config:
      1. Read source data from Node[src].LocalMemory[src_addr]
      2. Store as golden for Node[dst].LocalMemory[dst_addr]

    Collision handling: Last write wins (later config overwrites earlier).

    Args:
        node_configs: List of per-node transfer configurations.
        get_node_memory: Function to get node's LocalMemory by node_id.
        mesh_cols: Mesh columns (for coord_to_node_id conversion).

    Returns:
        Number of golden entries generated.
    """
    count = 0
    compute_cols = mesh_cols - 1

    for nc in node_configs:
        # Calculate destination node_id from coord
        dest_x, dest_y = nc.dest_coord
        if dest_x < 1 or dest_x >= mesh_cols:
            continue  # Edge column, skip

        dst_node_id = (dest_x - 1) + dest_y * compute_cols

        # Read expected data from source node's memory
        src_memory = get_node_memory(nc.src_node_id)
        expected_data, _ = src_memory.read(
            nc.local_src_addr, nc.transfer_size
        )

        # Store as golden for destination node
        self.capture_write(
            node_id=dst_node_id,
            addr=nc.local_dst_addr,
            data=expected_data,
            cycle=0,
        )
        count += 1

    return count
```

### Collision Handling

When multiple sources write to the same destination address:

```
Example: Node 0 and Node 2 both send to Node 1 @ 0x1000

Config order:
  [0] Node 0 → Node 1 @ 0x1000  (first write)
  [1] Node 2 → Node 1 @ 0x1000  (second write, overwrites)

Result:
  Golden[(1, 0x1000)] = Node 2's source data (last write wins)
```

**Policy**: Last write wins - later node_config entry overwrites earlier golden entry.

### Verification Report

```python
@dataclass
class VerificationReport:
    total_checks: int = 0
    passed: int = 0
    failed: int = 0
    missing_golden: int = 0
    missing_actual: int = 0
    results: List[VerificationResult] = field(default_factory=list)

    @property
    def all_passed(self) -> bool:
        return self.failed == 0 and self.missing_golden == 0 and self.missing_actual == 0

    @property
    def pass_rate(self) -> float:
        return self.passed / self.total_checks if self.total_checks > 0 else 0.0
```

### Unit Tests

Golden verification is tested in `tests/unit/test_noc_golden.py`:

| Test | Description |
|------|-------------|
| `test_generate_noc_golden_neighbor` | Golden generation for neighbor pattern |
| `test_generate_noc_golden_shuffle` | Golden generation for shuffle pattern |
| `test_golden_collision_last_write_wins` | Collision uses last-write-wins semantics |
| `test_verify_with_matching_data` | Verification passes with correct data |
| `test_verify_with_mismatch` | Verification fails with incorrect data |
| `test_golden_entry_count` | Entry count is correct after generation |
| `test_golden_clear` | Clear removes all golden data |
| `test_empty_node_configs` | Handle empty config list |
| `test_invalid_destination_skipped` | Edge column destinations are skipped |
| `test_all_patterns_generate_golden` | All 5 patterns can generate golden |

---

## NoCSystem API

### Creation

```python
system = NoCSystem(
    mesh_cols=5,
    mesh_rows=4,
    buffer_depth=4,
    memory_size=0x10000,
)
```

### Configuration

```python
config = NoCTrafficConfig(
    pattern=TrafficPattern.NEIGHBOR,
    transfer_size=256,
)
system.configure_traffic(config)
```

### Memory Initialization

Initialize each node's local memory with test data. All 7 patterns include `node_id` in the data for uniqueness.

```python
# Initialize with sequential data per node
system.initialize_node_memory(pattern="sequential")

# Or with other patterns
system.initialize_node_memory(pattern="random", seed=42)
system.initialize_node_memory(pattern="constant", value=0xAB)
system.initialize_node_memory(pattern="address")
system.initialize_node_memory(pattern="walking_ones")
system.initialize_node_memory(pattern="walking_zeros")
system.initialize_node_memory(pattern="checkerboard")
```

**Supported Patterns:**

| Pattern | Description | Node Data Format |
|---------|-------------|------------------|
| `sequential` | Sequential bytes | Node N: `[N*16, N*16+1, ...]` |
| `random` | Random bytes | Deterministic seed per node |
| `constant` | Fixed value fill | `[node_id, value, value, ...]` |
| `address` | Address values | 4-byte LE with node_id offset |
| `walking_ones` | Walking ones | `[node_id, 0x01, 0x02, ...]` |
| `walking_zeros` | Walking zeros | `[node_id, 0xFE, 0xFD, ...]` |
| `checkerboard` | Checkerboard | `[node_id, 0xAA, 0x55, ...]` |

### Execution

```python
system.start_all_transfers()
cycles = system.run_until_complete(max_cycles=10000)
```

### Access Node Data

```python
# Get node controller
controller = system.node_controllers[node_id]

# Read/write local memory
data = controller.read_local_memory(addr, size)
controller.initialize_memory(addr, data)
```

### Golden Verification

```python
# Generate golden data (BEFORE simulation)
golden_count = system.generate_golden()
print(f"Generated {golden_count} golden entries")

# Run simulation...
system.start_all_transfers()
cycles = system.run_until_complete(max_cycles=10000)

# Verify against golden (AFTER simulation)
report = system.verify_transfers()
print(f"Pass: {report.passed}, Fail: {report.failed}")

# Print detailed report if failures
if not report.all_passed:
    system.golden_manager.print_report(report)
```

| Method | Description |
|--------|-------------|
| `generate_golden()` | Read expected data from source nodes, return entry count |
| `verify_transfers()` | Compare actual vs golden, return `VerificationReport` |
| `golden_manager` | Access internal `GoldenManager` instance |

---

## Performance Metrics

The simulation outputs:

| Metric | Description |
|--------|-------------|
| Total Cycles | Simulation cycles to complete |
| Total Data | Total bytes transferred (transfer_size * num_nodes) |
| Throughput | Bytes per cycle (total_data / cycles) |
| Avg Latency | Average cycles per transfer (cycles / num_nodes) |
| Simulation Speed | Cycles per second (wall-clock) |

---

## Related Files

| File | Purpose |
|------|---------|
| `src/core/local_axi_master.py` | LocalAXIMaster and LocalTransferConfig |
| `src/core/node_controller.py` | NodeController class |
| `src/traffic/pattern_generator.py` | TrafficPatternGenerator (5 patterns) |
| `src/core/routing_selector.py` | NoCSystem class with `generate_golden()`, `verify_transfers()` |
| `src/core/golden_manager.py` | GoldenManager with `generate_noc_golden()` method |
| `src/core/ni.py` | SlaveNI with user signal routing |
| `src/config.py` | TrafficPattern, NoCTrafficConfig, NodeTransferConfig |
| `examples/NoC_to_NoC/run.py` | Test runner with golden verification |
| `examples/NoC_to_NoC/config/*.yaml` | Pattern config files |
| `tests/unit/test_noc_golden.py` | Unit tests for NoC-to-NoC golden generation |
| `tests/unit/test_noc_memory_patterns.py` | Unit tests for memory initialization patterns |
| `tests/integration/test_noc_to_noc.py` | Integration tests |
