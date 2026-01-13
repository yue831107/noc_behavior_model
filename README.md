# NoC Behavior Model

A parameterizable Network-on-Chip (NoC) behavioral model for performance analysis and hardware verification.

## Architecture

- **Topology**: 5x4 2D Mesh with physically separated Request and Response networks
- **Routing**: Deterministic XY Routing with Y→X turn prevention
- **Switching**: Wormhole switching with credit-based flow control
- **Simulation**: Cycle-accurate behavioral model

## Quick Start

### Installation

```bash
git clone <repo_url>
cd noc_behavior_model
pip install -r requirements.txt
```

### Run Demo

```bash
# Host-to-NoC: Generate payload and run simulation
make quick

# NoC-to-NoC: Generate per-node payloads and run neighbor pattern
make gen_noc_payload && make sim_noc_neighbor
```

---

## Testing

### Test Commands

| Command | Description | Time |
|---------|-------------|------|
| `make test` | Run all tests (404 tests) | ~2 min |
| `make test_smoke` | Quick smoke test (core functionality) | ~10 sec |
| `make test_fast` | Run all tests, stop on first failure | varies |
| `make test_unit` | Unit tests only | ~30 sec |
| `make test_integration` | Integration tests only | ~1 min |
| `make test_performance` | Performance validation tests | ~30 sec |
| `make test_coverage` | Run with coverage report | ~2 min |

### Direct pytest Commands

```bash
# Run all tests
python -m pytest tests/ -v

# Run specific test file
python -m pytest tests/unit/test_router_port.py -v

# Run specific test class
python -m pytest tests/unit/test_router_port.py::TestRouterPortInitialState -v

# Run specific test function
python -m pytest tests/unit/test_router_port.py::TestRouterPortInitialState::test_initial_signal_states -v

# Run tests matching keyword
python -m pytest tests/ -k "routing" -v

# Run with short traceback
python -m pytest tests/ --tb=short

# Run and stop on first failure
python -m pytest tests/ -x

# Run with coverage
python -m pytest tests/ --cov=src --cov-report=term-missing
```

### Test Categories

| Category | Path | Description |
|----------|------|-------------|
| Unit | `tests/unit/` | Component-level tests (Router, NI, Flit, Buffer) |
| Integration | `tests/integration/` | Multi-component tests (Mesh, Deadlock, Scatter/Gather) |
| Performance | `tests/performance/` | Validation tests (Theory, Consistency, Sweep, Regression) |

### Examples

```bash
# Example 1: Quick validation before commit
make test_smoke

# Example 2: Full test with verbose output
make test

# Example 3: Debug a specific failing test
python -m pytest tests/unit/test_xy_routing.py::TestXYPriority -v --tb=long

# Example 4: Run deadlock-related tests only
python -m pytest tests/integration/ -k "deadlock" -v

# Example 5: Run performance validation
make test_performance
```

---

## Simulation

### Host-to-NoC Workflow

```bash
# Step 1: Generate payload
make gen_payload PAYLOAD_SIZE=1024 PAYLOAD_PATTERN=sequential

# Step 2: Generate transfer config
make gen_config NUM_TRANSFERS=10

# Step 3: Run simulation
make sim
```

### NoC-to-NoC Traffic Patterns

| Command | Pattern | Description |
|---------|---------|-------------|
| `make sim_noc_neighbor` | Neighbor | Each node sends to adjacent node |
| `make sim_noc_shuffle` | Shuffle | Bit-shuffle permutation |
| `make sim_noc_bit_reverse` | Bit Reverse | Bit-reversal permutation |
| `make sim_noc_random` | Random | Random destination selection |
| `make sim_noc_transpose` | Transpose | (x,y) → (y,x) mapping |
| `make sim_noc_all` | All | Run all patterns sequentially |

### Payload Patterns

| Pattern | Description |
|---------|-------------|
| `sequential` | Incrementing byte values (0x00, 0x01, ...) |
| `random` | Random bytes (seeded) |
| `constant` | All zeros |
| `address` | Address-based pattern |
| `walking_ones` | Walking ones pattern |
| `walking_zeros` | Walking zeros pattern |
| `checkerboard` | Alternating 0xAA/0x55 |

---

## Development

### Setup Pre-commit Hooks

```bash
pip install pre-commit
pre-commit install
```

### Project Structure

```
src/
├── core/           # NoC hardware model (Router, NI, Mesh, Flit)
├── testbench/      # Test peripherals (Memory, AXI Master)
├── verification/   # Golden data and validators
└── visualization/  # Charts and metrics

tests/
├── unit/           # Component tests
├── integration/    # System tests
└── performance/    # Validation tests
```

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture Overview](docs/design/01_overview.md) | High-level system design |
| [Router Design](docs/design/02_router.md) | Router architecture and XY routing |
| [Network Interface](docs/design/03_network_interface.md) | NI and AXI protocol conversion |
| [Design Docs](docs/design/) | All component specifications |

## License

MIT License
