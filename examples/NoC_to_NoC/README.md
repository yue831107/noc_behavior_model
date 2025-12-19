# NoC-to-NoC Traffic Simulation

Node-to-node communication testing within the mesh network.

## Quick Start

```bash
# From project root (use py -3 on Windows if python fails)
py -3 examples/NoC_to_NoC/run.py neighbor
```

## Usage

```bash
cd examples/NoC_to_NoC

# Basic usage
py -3 run.py [PATTERN] [-m MEMORY] [-P DIR] [--all] [-q]

# Dynamic data generation
py -3 run.py neighbor              # Default (sequential)
py -3 run.py neighbor -m random    # Random memory data
py -3 run.py --all -m walking_ones # All patterns with walking_ones

# Load from payload files
make gen_noc_payload               # Generate payload/node_XX.bin first
py -3 run.py neighbor -P payload/  # Load from payload files
py -3 run.py --all -P payload/     # All patterns with payload files
```

> **Windows 注意**：如果 `python` 指令出錯，請使用 `py -3`

## Traffic Patterns

| Pattern | Destination | Collision |
|---------|-------------|-----------|
| `neighbor` | `(i + 1) % N` | No |
| `transpose` | `swap(x, y)` | No |
| `shuffle` | `rotate_left(bits)` | Yes |
| `bit_reverse` | `reverse(bits)` | Yes |
| `random` | `random(seed)` | Yes |

## Memory Patterns

| Pattern | Description |
|---------|-------------|
| `sequential` | `(node * 16 + i) & 0xFF` |
| `random` | Deterministic random |
| `constant` | Fixed value |
| `node_id` | Fill with node ID |
| `address` | Address-based |
| `walking_ones` | `1 << (i % 8)` |
| `walking_zeros` | `~(1 << (i % 8))` |
| `checkerboard` | `0xAA/0x55` |

## Configuration

Edit `config/<pattern>.yaml`:

```yaml
traffic:
  pattern: "neighbor"
  mesh_cols: 5
  mesh_rows: 4
  transfer_size: 256
  local_src_addr: 0x0000
  local_dst_addr: 0x1000
```

## Custom Test

1. Create `config/mytest.yaml`
2. Run: `python run.py mytest`

## Output

```
--- Golden Data Verification ---
  Total Checks:      16
  PASSED:            16
  FAILED:            0
  Pass Rate:         100.0%
```

## Notes

- Patterns with collision may show partial failures (expected)
- Use `neighbor` or `transpose` for 100% deterministic results

See [user_guide.md](../../user_guide.md) Section 4 for complete reference.
