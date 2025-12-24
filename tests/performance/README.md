# Performance Validation Framework

Monitor-based performance metrics validation for NoC Behavior Model.

## 概述

這個框架提供多層次的效能指標驗證，**完全不修改 Core 代碼**，採用外部監控方式讀取系統 metrics。

## 設計原則

✅ **Monitor-based** - 透過 `MetricsProvider` 介面觀察系統狀態  
✅ **零侵入** - Core 不需要知道驗證器存在  
✅ **單向依賴** - Validator → Core（只讀取，不修改）

## 驗證策略

### 1. 理論驗證（Theory-based Validation）

驗證效能指標是否在理論上下界內。

**驗證項目**：
- Throughput ≤ 理論最大值（flit_width × routers）
- Latency ≥ 理論最小值（hops × pipeline_depth）
- Buffer Utilization ∈ [0, 1]

**使用範例**：
```python
from tests.performance.validators.theory_validator import TheoryValidator

validator = TheoryValidator()
metrics = {
    'throughput': 248.5,
    'avg_latency': 2.8,
    'buffer_utilization': 0.35
}

results = validator.validate_all(metrics)
# {'throughput': (True, 'OK'), 'buffer_utilization': (True, 'OK')}
```

### 2-5. 其他驗證器（規劃中）

- **一致性驗證** - Little's Law、能量守恆
- **回歸測試** - 與 baseline 比對
- **漸進性測試** - 參數變化趨勢
- **極端測試** - 零負載、飽和測試

## 目錄結構

```
tests/performance/
├── __init__.py
├── validators/
│   ├── __init__.py
│   └── theory_validator.py         ✅ 已實作
├── baselines/
│   ├── neighbor_pattern.json       ✅ 已建立
│   └── shuffle_pattern.json        ✅ 已建立
├── test_theory_validation.py       ✅ 已實作（12 tests）
└── utils.py                         ✅ 已實作
```

## 使用方式

### 執行測試

```bash
# 執行所有效能驗證測試
make test_performance

# 只執行理論驗證
make test_theory

# 產生 HTML 報告
make test_performance_report
# 報告位於：output/performance_report.html
```

### 在程式中使用

```python
# 1. 執行模擬（Core 完全不知道有驗證器）
from examples.NoC_to_NoC.run import run_neighbor_pattern
system = run_neighbor_pattern()

# 2. 提取 metrics（Monitor-based，透過已存在的 MetricsProvider）
from tests.performance.utils import extract_metrics_from_simulation
metrics = extract_metrics_from_simulation(system)

# 3. 驗證（在測試層，與 Core 解耦）
from tests.performance.validators.theory_validator import TheoryValidator
validator = TheoryValidator()
results = validator.validate_all(metrics)

# 4. 檢查結果
assert all(is_valid for is_valid, _ in results.values())
```

## 測試結果

```bash
$ make test_performance
================== test session starts ==================
tests/performance/test_theory_validation.py::TestTheoryValidation::test_throughput_calculation PASSED
tests/performance/test_theory_validation.py::TestTheoryValidation::test_min_latency_calculation PASSED
tests/performance/test_theory_validation.py::TestTheoryValidation::test_validate_throughput_pass PASSED
tests/performance/test_theory_validation.py::TestTheoryValidation::test_validate_throughput_fail PASSED
tests/performance/test_theory_validation.py::TestTheoryValidation::test_validate_latency_pass PASSED
tests/performance/test_theory_validation.py::TestTheoryValidation::test_validate_latency_too_low PASSED
tests/performance/test_theory_validation.py::TestTheoryValidation::test_validate_buffer_utilization_pass PASSED
tests/performance/test_theory_validation.py::TestTheoryValidation::test_validate_buffer_utilization_fail PASSED
tests/performance/test_theory_validation.py::TestTheoryValidation::test_validate_all_metrics PASSED
tests/performance/test_theory_validation.py::TestTheoryValidation::test_validate_all_with_violations PASSED
tests/performance/test_theory_validation.py::TestTheoryValidation::test_print_validation_results PASSED
tests/performance/test_theory_validation.py::test_theory_validator_integration PASSED
================== 12 passed in 0.03s ===================
```

## Baseline 格式

```json
{
  "test_name": "neighbor_pattern_16nodes_256bytes",
  "config": {
    "pattern": "neighbor",
    "nodes": 16,
    "transfer_size": 256
  },
  "expected_metrics": {
    "total_cycles": {"min": 25, "max": 40, "target": 33},
    "avg_latency": {"min": 2.0, "max": 4.0, "target": 2.5},
    "throughput": {"min": 220, "max": 270, "target": 248}
  }
}
```

## 後續開發

- [ ] Consistency Validator - 驗證 Little's Law、能量守恆
- [ ] Regression Validator - 比對 baseline
- [ ] Progressive Validator - 驗證參數趨勢
- [ ] Boundary Validator - 極端測試
- [ ] 整合到 CI/CD

## 注意事項

> **完全不修改 Core**：所有驗證器都使用 Monitor-based 方式，透過 `MetricsProvider` 讀取系統狀態，Core 完全不知道驗證器存在。

> **容忍度**：所有驗證都允許一定誤差（預設 5%），因為模擬有隨機性。

> **Baseline 維護**：Baseline 應定期更新，並註明建立日期與環境。
