# Performance Tests

效能驗證測試。詳細說明請參閱 [效能驗證框架](../../docs/design/14_performance_validation.md)。

## 執行測試

```bash
# 所有效能測試
py -3 -m pytest tests/performance/ -v

# 理論驗證
py -3 -m pytest tests/performance/test_theory_validation.py -v

# 一致性驗證
py -3 -m pytest tests/performance/test_consistency_validation.py -v

# 參數掃描
py -3 -m pytest tests/performance/test_sweep.py -v
```

## 目錄結構

```
tests/performance/
├── regression/           # 回歸測試設定
├── sweep/                # 參數掃描設定
├── test_*.py             # 測試檔案
└── conftest.py           # 共用 fixtures

src/verification/
├── theory_validator.py       # 理論驗證器
├── consistency_validator.py  # 一致性驗證器
├── golden_manager.py         # Golden 資料管理
└── metrics_provider.py       # 指標提供介面
```
