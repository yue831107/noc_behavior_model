# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [1.2] - 2025-01-13

### Added
- Performance validation framework (Layer 1-2)
  - Theory validator for throughput/latency verification
  - Consistency validator for cross-run stability
- Parameter sweep infrastructure (`tests/performance/sweep/`)
- Performance regression testing (`tests/performance/regression/`)
- Comprehensive unit tests (586 tests, 84% coverage)
  - `test_buffer.py`, `test_config.py`, `test_memory.py`
  - `test_metrics_provider.py`, `test_local_axi_master.py`
- Claude Code configuration
  - Agents: `code-reviewer`, `github-workflow`
  - Commands: `/test`, `/perf`, `/sim`
  - Skills: `flit-packet`, `noc-routing`, `testing`
- MIT LICENSE file
- `.coveragerc` for coverage configuration
- `pytest.ini` for pytest configuration
- `.pre-commit-config.yaml` for CI/CD readiness

### Changed
- Reorganized codebase structure
  - Moved `memory.py`, `axi_master.py`, `node_controller.py` to `src/testbench/`
  - Moved `golden_manager.py`, `metrics_provider.py`, validators to `src/verification/`
- Restructured documentation with numbered ordering (`docs/design/01_*.md`)
- Improved exception handling: replaced broad `except Exception` with specific types
- Replaced magic numbers with named constants

### Removed
- Obsolete `user_guide.md`
- Obsolete `requirements-web.txt`
- Old example configs (`test50.yaml`, `test_deadlock.yaml`, `test_gather.yaml`)

## [1.1] - 2024-12-25

### Added
- Initial NoC behavior model implementation
- 5x4 2D Mesh topology with XY routing
- Wormhole switching with credit-based flow control
- Host-to-NoC operation mode (V1System)
- NoC-to-NoC operation mode (NoCSystem)
- SlaveNI and MasterNI implementations
- Routing Selector with multi-path support
- Golden data verification (GoldenManager)
- Basic visualization and metrics collection
- Example simulations (`Host_to_NoC/`, `NoC_to_NoC/`)

### Features
- Transfer modes: BROADCAST, SCATTER, BROADCAST_READ, GATHER
- Traffic patterns: neighbor, shuffle, bit_reverse, random, transpose
- Cycle-accurate simulation model
- FlooNoC-style flit format with 20-bit header

---

[1.2]: https://github.com/yue831107/noc-behavior-model/compare/v1.1...v1.2
[1.1]: https://github.com/yue831107/noc-behavior-model/releases/tag/v1.1
