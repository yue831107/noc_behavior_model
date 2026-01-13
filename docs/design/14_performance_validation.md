# 效能驗證框架

本文件說明 NoC Behavior Model 的效能驗證機制與公式。

---

## 1. 驗證架構

```
┌─────────────────────────────────────────────────────────────────┐
│                        驗證流程                                  │
│                                                                 │
│   ┌─────────┐      ┌─────────────┐      ┌─────────────┐        │
│   │  NoC    │ ───► │  Metrics    │ ───► │  Validator  │        │
│   │  Core   │      │  Provider   │      │  (外部)     │        │
│   └─────────┘      └─────────────┘      └─────────────┘        │
│        │                  │                    │                │
│   不被修改            讀取統計             驗證結果              │
│                                                                 │
│   ✓ Monitor-based    ✓ 零侵入    ✓ 單向依賴 (Validator → Core)  │
└─────────────────────────────────────────────────────────────────┘
```

---

## 2. 理論驗證 (Theory Validation)

### 2.1 Throughput 上界

**公式**：

```
T_max = N_edge × W_flit

其中：
  T_max   = 理論最大吞吐量 (bytes/cycle)
  N_edge  = Edge Router 數量 (= mesh_rows)
  W_flit  = Flit 寬度 (bytes)
```

**V1 架構範例** (5×4 mesh)：

```
T_max = 4 × 8 = 32 bytes/cycle

┌─────────────────────────────────────┐
│  Routing Selector (瓶頸)            │
│       ↓   ↓   ↓   ↓                 │
│     (0,3)(0,2)(0,1)(0,0)  ← 4 Edge Routers
│       │   │   │   │                 │
│       └───┴───┴───┘                 │
│    4 ports × 8 B = 32 B/cycle       │
└─────────────────────────────────────┘
```

**驗證條件**：

```
T_actual ≤ T_max × (1 + tolerance)

tolerance = 5% (預設)
```

---

### 2.2 Latency 下界

**公式**：

```
L_min = D_manhattan × P_depth + L_overhead

其中：
  L_min       = 理論最小延遲 (cycles)
  D_manhattan = |x_dst - x_src| + |y_dst - y_src|
  P_depth     = Router Pipeline 深度 (cycles/hop)
  L_overhead  = NI + Selector 額外延遲 (~2 cycles)
```

**範例**：(0,0) → (3,2)

```
D_manhattan = |3-0| + |2-0| = 5 hops

Pipeline Mode:
  fast:     P_depth = 1 → L_min = 5×1 + 2 = 7 cycles
  standard: P_depth = 2 → L_min = 5×2 + 2 = 12 cycles
  hardware: P_depth = 4 → L_min = 5×4 + 2 = 22 cycles
```

**路徑圖示**：

```
    (0,0) ─── (1,0) ─── (2,0) ─── (3,0)
                                   │
                                 (3,1)
                                   │
                                 (3,2) ← 目的地

    Total: 3 hops (X) + 2 hops (Y) = 5 hops
```

---

### 2.3 Latency 上界

**公式**：

```
L_max = L_min + D_buffer + D_serial

其中：
  L_max     = 理論最大延遲 (cycles)
  L_min     = 理論最小延遲
  D_buffer  = hops × buffer_depth  (最壞 buffer 排隊)
  D_serial  = packet_flits - 1     (封包序列化)
```

**範例**：(0,0) → (3,2), buffer_depth=4, 單 flit 封包

```
L_min    = 7 cycles
D_buffer = 5 × 4 = 20 cycles
D_serial = 1 - 1 = 0 cycles

L_max = 7 + 20 + 0 = 27 cycles
```

---

### 2.4 Buffer Utilization

**驗證條件**：

```
0 ≤ U_buffer ≤ 1

其中：
  U_buffer = 平均 buffer 佔用率
```

| 條件 | 結果 |
|------|------|
| U < 0 | ✗ 無效 (量測錯誤) |
| 0 ≤ U ≤ 1 | ✓ 有效 |
| U > 1 | ✗ 無效 (溢出) |

---

## 3. 一致性驗證 (Consistency Validation)

### 3.1 Little's Law

**公式**：

```
L = λ × W

其中：
  L = 平均佇列長度 (flits in network)
  λ = 到達率 (flits/cycle) = throughput / flit_width
  W = 平均等待時間 (cycles) = avg_latency
```

**驗證方式**：

```
L_expected = λ × W
L_actual   = avg_occupancy (from metrics)

deviation = |L_actual - L_expected| / L_expected

驗證通過條件: deviation ≤ tolerance (10%)
```

**範例**：

```
輸入:
  throughput     = 16 bytes/cycle
  flit_width     = 8 bytes
  avg_latency    = 5 cycles
  avg_occupancy  = 10 flits

計算:
  λ = 16 / 8 = 2 flits/cycle
  L_expected = 2 × 5 = 10 flits
  L_actual = 10 flits

結果:
  deviation = |10 - 10| / 10 = 0%
  ✓ PASS (< 10% tolerance)
```

---

### 3.2 Flit Conservation

**公式**：

```
F_sent = F_received   (無損系統)

其中：
  F_sent     = 所有來源發送的 flit 總數
  F_received = 所有目的地接收的 flit 總數
```

**異常檢測**：

| 狀況 | 診斷 |
|------|------|
| F_received < F_sent | Flit 遺失 (packet loss) |
| F_received > F_sent | Flit 重複 (duplication) |
| F_received = F_sent | ✓ 守恆成立 |

---

### 3.3 Bandwidth Conservation

**公式** (穩態)：

```
Σ(R_injection) = Σ(R_ejection)

其中：
  R_injection = 各節點注入率 (bytes/cycle)
  R_ejection  = 各節點彈出率 (bytes/cycle)
```

**驗證條件**：

```
deviation = |Σ(injection) - Σ(ejection)| / Σ(injection)

驗證通過條件: deviation ≤ tolerance (10%)
```

---

### 3.4 Router Logic

**公式**：

```
F_received = F_forwarded + F_consumed

其中：
  F_received  = Router 接收的 flit 數
  F_forwarded = 轉發到其他 Router 的 flit 數
  F_consumed  = 本地 NI 消耗的 flit 數
```

**圖示**：

```
         ┌──────────────┐
    ───► │              │ ───►
         │    Router    │         F_forwarded
    ───► │              │ ───►
         │              │
         └──────┬───────┘
                │
                ▼
              [NI]                F_consumed
```

---

## 4. 驗證公式總表

| 類別 | 驗證項目 | 公式 | 容忍度 |
|------|----------|------|--------|
| 理論 | Throughput 上界 | T ≤ N_edge × W_flit | 5% |
| 理論 | Latency 下界 | L ≥ hops × P_depth + 2 | 5% |
| 理論 | Buffer Utilization | 0 ≤ U ≤ 1 | - |
| 一致性 | Little's Law | L = λ × W | 10% |
| 一致性 | Flit Conservation | Sent = Received | 0% |
| 一致性 | Bandwidth Conservation | In = Out | 10% |
| 一致性 | Router Logic | Recv = Fwd + Consumed | 0% |

---

## 5. 使用範例

### 5.1 Theory Validator

```python
from tests.performance.validators.theory_validator import TheoryValidator

validator = TheoryValidator()

# 驗證 Throughput
result = validator.validate_throughput(actual_throughput=28.5)
# (True, 'OK') if 28.5 ≤ 32 × 1.05 = 33.6

# 驗證 Latency
result = validator.validate_latency(
    actual_latency=8.0,
    src=(0, 0),
    dest=(3, 2)
)
# (True, 'OK') if 8.0 ≥ 7 × 0.95 = 6.65

# 批次驗證
results = validator.validate_all({
    'throughput': 28.5,
    'avg_latency': 8.0,
    'src': (0, 0),
    'dest': (3, 2),
    'buffer_utilization': 0.35
})
```

### 5.2 Consistency Validator

```python
from tests.performance.validators.consistency_validator import ConsistencyValidator

validator = ConsistencyValidator(tolerance=0.10)

# 驗證 Little's Law
result = validator.validate_littles_law(
    throughput=16.0,      # bytes/cycle
    avg_latency=5.0,      # cycles
    avg_occupancy=10.0    # flits
)
# λ = 16/8 = 2, L_expected = 2×5 = 10, deviation = 0%
# (True, 'OK (deviation=0.0%)')

# 驗證 Flit Conservation
result = validator.validate_flit_conservation(
    total_sent=1000,
    total_received=1000
)
# (True, 'OK (conservation holds)')

# 批次驗證
results = validator.validate_all({
    'throughput': 16.0,
    'avg_latency': 5.0,
    'avg_occupancy': 10.0,
    'total_sent': 1000,
    'total_received': 1000
})
```

---

## 6. Host-to-NoC vs NoC-to-NoC 驗證差異

| 驗證項目 | Host-to-NoC | NoC-to-NoC | 原因 |
|----------|-------------|------------|------|
| Throughput 上界 | ✓ 驗證 T ≤ 32 B/cycle | ✗ 跳過 | NoC-to-NoC 無 Edge Router 瓶頸 |
| Little's Law | ✓ 驗證 | ✗ 跳過 | Burst 流量不滿足穩態假設 |
| Buffer Utilization | ✓ 驗證 | ✓ 驗證 | 兩種模式皆適用 |
| Flit Conservation | ✓ 驗證 | ✓ 驗證 | 兩種模式皆適用 |
| Data Verification | ✓ 驗證 | ✓ 驗證 | Golden 比對 |

### 6.1 NoC-to-NoC Little's Law 跳過說明

Little's Law 假設穩態到達率 (steady-state arrival rate)，但 NoC-to-NoC 為 Burst 流量：
- 所有節點同時開始傳輸
- 流量注入集中在開始階段
- 吞吐量指標量測的是完成率，非到達率

因此 `L = λ × W` 公式不適用於 NoC-to-NoC 模式。

---

## 7. 批量效能測試

### 7.1 批量測試工具

使用 `tools/run_batch_perf_test.py` 執行大規模效能驗證：

```bash
# 執行 500 個 Host-to-NoC 測試
py -3 tools/run_batch_perf_test.py --mode host_to_noc --count 500

# 執行 500 個 NoC-to-NoC 測試
py -3 tools/run_batch_perf_test.py --mode noc_to_noc --count 500

# 同時執行兩種模式
py -3 tools/run_batch_perf_test.py --mode both --count 500

# 自訂輸出目錄
py -3 tools/run_batch_perf_test.py --mode both -o output/my_tests
```

### 7.2 測試覆蓋範圍

**Host-to-NoC**:
| 參數 | 範圍 |
|------|------|
| Transfer Size | 64, 128, 256, 512, 1024, 2048, 4096, 8192 bytes |
| Target Nodes | 1, 2, 4, 8, 16 nodes |
| Transfer Mode | broadcast, scatter |

**NoC-to-NoC**:
| 參數 | 範圍 |
|------|------|
| Transfer Size | 64, 128, 256, 512, 1024, 2048, 4096, 8192 bytes |
| Traffic Pattern | neighbor, shuffle, bit_reverse, random, transpose |

### 7.3 輸出報告

```
output/batch_tests/
├── batch_host_to_noc_summary.json   # Host-to-NoC 摘要
├── batch_host_to_noc_details.json   # Host-to-NoC 詳細結果
├── batch_noc_to_noc_summary.json    # NoC-to-NoC 摘要
└── batch_noc_to_noc_details.json    # NoC-to-NoC 詳細結果
```

**Summary 格式**:
```json
{
  "mode": "host_to_noc",
  "total_tests": 500,
  "passed_tests": 500,
  "failed_tests": 0,
  "pass_rate": 100.0,
  "throughput": { "min": 8.0, "max": 31.89, "avg": 25.22 },
  "latency": { "min": 7, "max": 2185, "avg": 228.6 }
}
```

### 7.4 驗證項目

批量測試自動驗證：
1. **Throughput 上界** (Host-to-NoC): T ≤ 32 B/cycle
2. **Buffer Utilization**: 0 ≤ U ≤ 1
3. **Data Verification**: 傳輸資料正確性 (Golden 比對)

---

## 8. 測試執行

```bash
# 執行所有效能驗證測試
py -3 -m pytest tests/performance/ -v

# 理論驗證 (12 tests)
py -3 -m pytest tests/performance/test_theory_validation.py -v

# 一致性驗證 (6 tests)
py -3 -m pytest tests/performance/test_consistency_validation.py -v

# 批量效能測試 (1000 tests)
py -3 tools/run_batch_perf_test.py --mode both --count 500
```

---

## 相關文件

- [效能指標](12_metrics.md) - Stats 類別與 MetricsCollector
- [硬體參數指南](hardware_parameters_guide.md) - 參數調校
- [模擬參數](11_simulation.md) - 模擬配置
