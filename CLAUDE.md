# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

This is a **Network-on-Chip (NoC) Behavior Model** project. The goal is to create a parameterizable, visualizable behavioral simulation for performance analysis.

## Key Documents

| File | Purpose |
|------|---------|
| `spec.md` | Implementation specification (V1 & V2 architecture) |
| `plan.md` | Implementation plan and progress tracking |
| `user_guide.md` | User guide (architecture, testing, configuration, patterns) |
| `docs/images/NI.jpg` | NI internal architecture |
| `docs/images/operation.jpg` | DMA/PIO operation modes |
| `docs/images/selector.jpg` | Routing Selector architecture |
| `docs/images/smart_crossbar.jpg` | Smart Crossbar architecture |
| `docs/images/sim_engine.jpg` | Simulation engine diagram |
| `docs/images/test_bench.jpg` | Test bench architecture |
| `docs/images/traffic_pattern.jpg` | Traffic patterns (5 patterns) |
| `docs/design/14_noc_to_noc.md` | NoC-to-NoC design document |

## Architecture Versions

- **Version 1**: Single Entry Routing Selector (current implementation)
- **Version 2**: Smart Crossbar with 4 NIs (planned)

## Key Architecture Decisions

| Decision | Detail |
|----------|--------|
| Req/Resp Separation | Physical separation (ReqRouter + RespRouter, ReqNI + RespNI) |
| Switching | Virtual Cut-Through |
| Flow Control | Credit-based |
| Routing | XY Deterministic |
| Topology | 5x4 Mesh (Column 0 = Edge Routers) |

## Implementation Goals

1. **Parameterizable**: Mesh size, buffer depth, latencies, traffic patterns
2. **Visualizable**: Real-time heatmap, latency histogram, throughput curves
3. **Metrics**: Latency, throughput, buffer utilization, bottleneck analysis

## Coding Standards

### Python Style (PEP 8)

- **Line length**: Max 88 characters (Black formatter compatible)
- **Indentation**: 4 spaces (no tabs)
- **Naming conventions**:
  - `snake_case` for functions, methods, variables
  - `PascalCase` for classes
  - `UPPER_CASE` for constants
- **Imports**:
  - Standard library first, then third-party, then local
  - One import per line for `from` imports
- **Docstrings**: Google style for all public classes and methods
- **Type hints**: Required for all function signatures

### Code Organization

- One class per file when class is substantial
- Group related small classes in single file
- Keep files under 500 lines when possible
- Use `__init__.py` to expose public API

### Documentation

- All public methods must have docstrings
- Complex algorithms should have inline comments
- Update `plan.md` Progress Tracking after completing each phase

## Development Workflow

1. **Incremental development**: Complete one component, review, then proceed
2. **Update progress**: Sync `plan.md` after each implementation step
3. **Test early**: Write unit tests alongside implementation

## Document Maintenance

When code is modified or updated, evaluate whether the following documents need to be updated:

| Document | Update When |
|----------|-------------|
| `plan.md` | Phase completion, progress changes, new decisions |
| `user_guide.md` | API changes, new features, configuration changes, new traffic patterns, architecture changes |

**Auto-sync rule**: After any code modification, check if `user_guide.md` sections are affected:
- Section 1 (Architecture): Core component changes
- Section 3-4 (Testing/Simulation): Test procedure or API changes
- Section 5 (Traffic Patterns): New patterns or pattern parameter changes
- Section 6 (Configuration): YAML structure or config parameter changes
- Section 7 (Custom Patterns): TrafficPattern API changes
- Section 8 (Metrics): New metrics or metric calculation changes

## Project Structure

```
noc_behavior_model/
├── src/
│   ├── core/           # Router, NI, Routing Selector, Flit, Mesh, Memory
│   │   ├── local_axi_master.py  # LocalAXIMaster (NoC-to-NoC)
│   │   └── node_controller.py   # NodeController (NoC-to-NoC)
│   ├── traffic/        # Traffic pattern generation
│   │   └── pattern_generator.py # 5 traffic patterns
│   ├── axi/            # AXI4 protocol definitions
│   ├── address/        # Address translation
│   └── config.py       # TransferConfig, NoCTrafficConfig
├── tests/              # Unit and integration tests
│   ├── unit/           # Unit tests
│   └── integration/    # Integration tests
│       └── test_noc_to_noc.py  # NoC-to-NoC tests
├── examples/           # Test cases (file-driven)
│   ├── Host_to_NoC/    # Host → NoC transfer tests
│   │   ├── config/     # Test configuration files (YAML)
│   │   ├── payload/    # Test data files (binary)
│   │   └── run.py      # Test runner
│   └── NoC_to_NoC/     # NoC → NoC traffic tests
│       ├── config/     # Traffic pattern configs (5 patterns)
│       └── run.py      # Test runner with metrics
├── tools/              # Utility scripts
│   └── pattern_gen.py  # Payload generator
├── docs/
│   ├── images/         # Architecture diagrams
│   └── design/         # Design documents
└── Makefile            # Build/test automation
```

## Test Case Structure

Each test scenario (e.g., Host_to_NoC, NoC_to_NoC) is organized as:
- `config/` - YAML configuration files for different test modes
- `payload/` - Binary test data files
- `run.py` - Test runner script
- Output logs are generated in `output/` (auto-created, in .gitignore)

## Common Commands

All commands are wrapped by Makefile for convenience:

```bash
# Show all available commands
make help

# Quick verification (generate payload + run write test)
make quick

# Full workflow (payload + all simulations + tests)
make all

# Host-to-NoC commands
make gen_payload                    # Generate payload (sequential, 1024B)
make gen_payload PATTERN=random SIZE=2048  # Custom payload
make sim_write                      # Broadcast write test
make sim_read                       # Broadcast read test
make sim_scatter                    # Scatter write test
make sim_gather                     # Gather read test
make sim_all                        # Run all Host-to-NoC simulations

# NoC-to-NoC commands
make sim_noc                        # Default pattern (neighbor)
make sim_noc_neighbor               # Neighbor pattern (ring topology)
make sim_noc_shuffle                # Shuffle pattern (bit rotation)
make sim_noc_bit_reverse            # Bit reverse pattern
make sim_noc_random                 # Random pattern
make sim_noc_transpose              # Transpose pattern (swap x,y)
make sim_noc_all                    # Run all 5 patterns

# Testing
make test                           # Run all pytest tests
make test_unit                      # Unit tests only
make test_integration               # Integration tests only
make clean                          # Clean generated files
```

Payload patterns: `sequential`, `random`, `constant`, `address`, `walking_ones`, `walking_zeros`, `checkerboard`

Traffic patterns (NoC-to-NoC): `neighbor`, `shuffle`, `bit_reverse`, `random`, `transpose`
