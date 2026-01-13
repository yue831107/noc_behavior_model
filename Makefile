# NoC Behavior Model Makefile
# Usage: make [target] [VARIABLE=value]

# Variables (can be overridden)
# Windows: use "py -3" to avoid GTKWave python.exe PATH conflict
PYTHON = py -3

# Host-to-NoC payload settings
PAYLOAD_DIR = examples/Host_to_NoC/payload
CONFIG_DIR = examples/Host_to_NoC/config
PAYLOAD_SIZE = 1024
PAYLOAD_PATTERN = sequential
PAYLOAD_FILE = $(PAYLOAD_DIR)/payload.bin

# NoC-to-NoC payload settings
NOC_PAYLOAD_DIR = examples/NoC_to_NoC/payload
NOC_NODES = 16
NOC_SIZE = 256
NOC_PATTERN = sequential
SEED = 42

# Multi-parameter sweep settings
SWEEP_BUFFER_DEPTH = 2,4,8,16

# Phony targets
.PHONY: help gen_payload gen_noc_payload gen_config gen_config_sweep \
        sim_write sim_read sim_scatter sim_gather sim_all sim \
        sim_noc sim_noc_neighbor sim_noc_shuffle sim_noc_bit_reverse sim_noc_random sim_noc_transpose sim_noc_all \
        test test_smoke test_fast test_unit test_integration test_coverage test_performance test_theory test_consistency test_performance_report \
        clean clean_payload clean_noc_payload all quick viz multi_para \
        regression regression_noc regression_quick test_regression

# Default target
help:
	@echo "NoC Behavior Model - Available Commands"
	@echo "========================================"
	@echo "[Host-to-NoC Workflow] (3-step)"
	@echo "  make gen_payload              Step 1: Generate payload BIN file"
	@echo "  make gen_config               Step 2: Generate transfer config YAML"
	@echo "  make sim                      Step 3: Run simulation"
	@echo ""
	@echo "  Options:"
	@echo "    gen_payload SIZE=N PATTERN=X    Payload size and pattern"
	@echo "    gen_config NUM_TRANSFERS=N      Number of transfers (default: 10)"
	@echo ""
	@echo "[NoC-to-NoC Workflow]"
	@echo "  make gen_noc_payload          Generate per-node payloads"
	@echo "  make sim_noc_neighbor         Run neighbor pattern"
	@echo "  make sim_noc_shuffle          Run shuffle pattern"
	@echo "  make sim_noc_bit_reverse      Run bit_reverse pattern"
	@echo "  make sim_noc_random           Run random pattern"
	@echo "  make sim_noc_transpose        Run transpose pattern"
	@echo "  make sim_noc_all              Run all patterns"
	@echo ""
	@echo "[Visualization]"
	@echo "  make viz                      Generate charts from latest sim"
	@echo ""
	@echo "[Multi-Parameter Sweep]"
	@echo "  make gen_config_sweep         Generate config with sweep params"
	@echo "  make multi_para               Run multi-parameter simulation"
	@echo "  Options: SWEEP_BUFFER_DEPTH=2,4,8,16"
	@echo ""
	@echo "[Regression Test - Hardware Optimization]"
	@echo "  make regression               Find optimal params (Host-to-NoC)"
	@echo "  make regression_noc           Find optimal params (NoC-to-NoC)"
	@echo "  make regression_quick         Quick search (early-stop)"
	@echo "  make test_regression          Run regression module tests"
	@echo ""
	@echo "[Testing]"
	@echo "  make test                     Run all pytest tests"
	@echo "  make test_smoke               Quick smoke test (~10s)"
	@echo "  make test_fast                Run all tests, stop on first failure"
	@echo "  make test_unit                Unit tests only"
	@echo "  make test_integration         Integration tests only"
	@echo "  make test_performance         Performance validation tests"
	@echo "  make test_theory              Theory-based validation only"
	@echo "  make test_consistency         Consistency validation only"
	@echo ""
	@echo "[Utilities]"
	@echo "  make clean                    Clean all generated files"
	@echo "----------------------------------------"
	@echo "Patterns: sequential random constant address walking_ones walking_zeros checkerboard"

# Host-to-NoC Payload generation
gen_payload:
	$(PYTHON) -c "from pathlib import Path; Path('$(PAYLOAD_DIR)').mkdir(parents=True, exist_ok=True)"
	$(PYTHON) tools/pattern_gen.py -p $(PAYLOAD_PATTERN) -s $(PAYLOAD_SIZE) -o $(PAYLOAD_FILE) --seed $(SEED) --hex-dump

# NoC-to-NoC Payload generation (per-node files)
gen_noc_payload:
	$(PYTHON) tools/pattern_gen.py --nodes $(NOC_NODES) -p $(NOC_PATTERN) -s $(NOC_SIZE) -o $(NOC_PAYLOAD_DIR) --seed $(SEED) --hex-dump

# Host-to-NoC Transfer Config generation
# Usage: make gen_config NUM_TRANSFERS=500
NUM_TRANSFERS = 10
TRANSFER_MODE = random
TRANSFER_MIN = 256
TRANSFER_MAX = 4096
TRANSFER_OUTPUT = examples/Host_to_NoC/config/generated.yaml

gen_config:
	$(PYTHON) tools/gen_transfer_config.py -n $(NUM_TRANSFERS) --mode $(TRANSFER_MODE) --min-size $(TRANSFER_MIN) --max-size $(TRANSFER_MAX) --seed $(SEED) -o $(TRANSFER_OUTPUT)

# Generate config with sweep parameters
gen_config_sweep:
	$(PYTHON) tools/gen_transfer_config.py -n $(NUM_TRANSFERS) --mode $(TRANSFER_MODE) --min-size $(TRANSFER_MIN) --max-size $(TRANSFER_MAX) --seed $(SEED) -o $(TRANSFER_OUTPUT) --sweep-buffer-depth $(SWEEP_BUFFER_DEPTH)

# Host-to-NoC Simulation (Specific Cases)
sim_write:
	$(PYTHON) examples/Host_to_NoC/run.py multi_transfer --config $(CONFIG_DIR)/broadcast_write.yaml --bin $(PAYLOAD_FILE)

sim_read:
	$(PYTHON) examples/Host_to_NoC/run.py multi_transfer --config $(CONFIG_DIR)/broadcast_read.yaml --bin $(PAYLOAD_FILE)

sim_scatter:
	$(PYTHON) examples/Host_to_NoC/run.py multi_transfer --config $(CONFIG_DIR)/scatter_write.yaml --bin $(PAYLOAD_FILE)

sim_gather:
	$(PYTHON) examples/Host_to_NoC/run.py multi_transfer --config $(CONFIG_DIR)/gather_read.yaml --bin $(PAYLOAD_FILE)

sim_all:
	$(PYTHON) examples/Host_to_NoC/run.py multi_transfer --config $(CONFIG_DIR)/multi_transfer.yaml --bin $(PAYLOAD_FILE)

# Host-to-NoC Simulation (Unified/Randomized)
# Requires: gen_payload, gen_config first
# Usage: make sim
sim:
	$(PYTHON) examples/Host_to_NoC/run.py multi_transfer --config $(TRANSFER_OUTPUT) --bin $(PAYLOAD_FILE)

# NoC-to-NoC Simulations (requires gen_noc_payload first)
sim_noc: sim_noc_neighbor

sim_noc_neighbor:
	$(PYTHON) examples/NoC_to_NoC/run.py neighbor -P $(NOC_PAYLOAD_DIR)

sim_noc_shuffle:
	$(PYTHON) examples/NoC_to_NoC/run.py shuffle -P $(NOC_PAYLOAD_DIR)

sim_noc_bit_reverse:
	$(PYTHON) examples/NoC_to_NoC/run.py bit_reverse -P $(NOC_PAYLOAD_DIR)

sim_noc_random:
	$(PYTHON) examples/NoC_to_NoC/run.py random -P $(NOC_PAYLOAD_DIR)

sim_noc_transpose:
	$(PYTHON) examples/NoC_to_NoC/run.py transpose -P $(NOC_PAYLOAD_DIR)

sim_noc_all:
	$(PYTHON) examples/NoC_to_NoC/run.py --all -P $(NOC_PAYLOAD_DIR)

# Testing
test:
	$(PYTHON) -m pytest tests/ -v

test_smoke:
	$(PYTHON) -m pytest tests/unit/test_router_port.py tests/unit/test_xy_routing.py tests/unit/test_flit.py -q

test_fast:
	$(PYTHON) -m pytest tests/ -x -q

test_unit:
	$(PYTHON) -m pytest tests/unit/ -v

test_integration:
	$(PYTHON) -m pytest tests/integration/ -v

test_coverage:
	$(PYTHON) -m pytest tests/ --cov=src --cov-report=term-missing

# Performance Validation
test_performance:
	$(PYTHON) -m pytest tests/performance/ -v

test_theory:
	$(PYTHON) -m pytest tests/performance/test_theory_validation.py -v

test_consistency:
	$(PYTHON) -m pytest tests/performance/test_consistency_validation.py -v

test_performance_report:
	$(PYTHON) -m pytest tests/performance/ -v --html=output/performance_report.html --self-contained-html

# Cleaning
clean_payload:
	$(PYTHON) -c "from pathlib import Path; [f.unlink() for f in Path('$(PAYLOAD_DIR)').glob('*.bin')] if Path('$(PAYLOAD_DIR)').exists() else None"
	$(PYTHON) -c "from pathlib import Path; [f.unlink() for f in Path('$(PAYLOAD_DIR)').glob('*.hex')] if Path('$(PAYLOAD_DIR)').exists() else None"

clean_noc_payload:
	$(PYTHON) -c "from pathlib import Path; [f.unlink() for f in Path('$(NOC_PAYLOAD_DIR)').glob('*.bin')] if Path('$(NOC_PAYLOAD_DIR)').exists() else None"
	$(PYTHON) -c "from pathlib import Path; [f.unlink() for f in Path('$(NOC_PAYLOAD_DIR)').glob('*.hex')] if Path('$(NOC_PAYLOAD_DIR)').exists() else None"

clean: clean_payload clean_noc_payload
	$(PYTHON) -c "import shutil; from pathlib import Path; [shutil.rmtree(d) for d in Path('.').rglob('__pycache__') if d.is_dir()]"
	$(PYTHON) -c "import shutil; from pathlib import Path; shutil.rmtree('.pytest_cache', ignore_errors=True)"
	$(PYTHON) -c "import shutil; from pathlib import Path; shutil.rmtree('output', ignore_errors=True)"
	@echo "Clean complete."

# Workflows
all: gen_payload sim_all test

quick: gen_payload sim_write

# Visualization (uses latest simulation results)
viz:
	$(PYTHON) -m src.visualization.report_generator all --from-metrics output/metrics/latest.json

# Multi-Parameter Simulation
# Workflow: make gen_payload && make gen_config_sweep && make multi_para
multi_para:
	$(PYTHON) tools/run_multi_para.py --config $(TRANSFER_OUTPUT) --bin $(PAYLOAD_FILE) -o output/multi_para

# Regression Test - Hardware Parameter Optimization
# Searches parameter space to find optimal configuration for given targets
REGRESSION_CONFIG = tools/regression_config.yaml
REGRESSION_CONFIG_NOC = tools/regression_config_noc.yaml
REGRESSION_OUTPUT = output/regression

regression:
	$(PYTHON) tools/run_regression.py --config $(REGRESSION_CONFIG) -o $(REGRESSION_OUTPUT)

regression_noc:
	$(PYTHON) tools/run_regression.py --config $(REGRESSION_CONFIG_NOC) -o $(REGRESSION_OUTPUT)/noc

regression_quick:
	$(PYTHON) tools/run_regression.py --config $(REGRESSION_CONFIG) -o $(REGRESSION_OUTPUT) --early-stop

test_regression:
	$(PYTHON) -m pytest tests/performance/test_regression.py -v
