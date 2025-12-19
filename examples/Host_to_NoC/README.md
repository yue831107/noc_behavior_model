# Host-to-NoC Traffic Simulation

Host memory to compute nodes data transfer testing.

## Quick Start

```bash
# From project root
make gen_payload    # Generate test data
make sim_write      # Run broadcast write test
```

## Workflow

```bash
# 1. Generate payload (required first)
make gen_payload                    # Default: sequential, 1024B
make gen_payload PATTERN=random     # Random data
make gen_payload SIZE=2048          # Custom size

# 2. Run simulation
make sim_write      # Broadcast write
make sim_read       # Broadcast read
make sim_scatter    # Scatter write
make sim_gather     # Gather read
make sim_all        # All tests
```

## Test Modes

| Mode | Description |
|------|-------------|
| `broadcast_write` | Same data to all nodes |
| `broadcast_read` | Read same address from all |
| `scatter_write` | Different data per node |
| `gather_read` | Collect different addresses |

## Payload Patterns

| Pattern | Description |
|---------|-------------|
| `sequential` | 0x00, 0x01, 0x02... |
| `random` | Pseudo-random |
| `constant` | Fixed value |
| `address` | Based on address |
| `walking_ones` | Shifting 1-bit |
| `walking_zeros` | Shifting 0-bit |
| `checkerboard` | 0xAA, 0x55... |

## Configuration

Edit `config/<mode>.yaml`:

```yaml
topology:
  mesh_rows: 4
  mesh_cols: 5

transfer:
  src_addr: 0x0000
  src_size: 1024
  dst_addr: 0x1000
  target_nodes: "all"
  transfer_mode: "broadcast"

simulation:
  max_cycles: 10000
```

## Direct Python Usage

```bash
python examples/Host_to_NoC/run.py broadcast_write
python examples/Host_to_NoC/run.py scatter_write
```

## Files

```
Host_to_NoC/
  config/           # YAML configurations
  payload/          # Binary test data (generated)
  run.py            # Test runner
```

See [user_guide.md](../../user_guide.md) Section 3 for complete reference.
