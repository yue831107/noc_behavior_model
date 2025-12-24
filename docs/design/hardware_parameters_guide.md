# NoC 硬體可調參數設計指南

## 參數總覽

| 參數類別 | 參數名稱 | 預設值 | 影響 |
|---------|---------|--------|------|
| **網格** | `cols × rows` | 5 × 4 | 網路規模、總容量 |
| **Router** | `buffer_depth` | 4 | 壅塞、延遲 |
| **Router** | `pipeline` | fast | 時序精度 |
| **NI** | `max_outstanding` | 16 | 並行交易數 |
| **NI** | `flit_payload_size` | 32 B | 封包效率 |

---

## 1. 網格拓撲參數

### MeshConfig

| 參數 | 預設 | 範圍 | 效能影響 |
|------|------|------|----------|
| `cols` | 5 | 2-16 | 橫向規模 |
| `rows` | 4 | 1-16 | 縱向規模 |
| `edge_column` | 0 | 0 | Selector 連接位置 |

**公式**：
```
Max Nodes = (cols - 1) × rows
Max Throughput (Selector 瓶頸) = rows × 8 B/c
```

---

## 2. Router 參數

### RouterConfig

| 參數 | 預設 | 範圍 | 效能影響 |
|------|------|------|----------|
| `buffer_depth` | 4 | 2-32 | ↑ 減少壅塞，↑ 面積 |
| `flit_width` | 64 bits | 32-128 | 頻寬 |
| `output_buffer_depth` | 0 | 0-8 | 輸出緩衝 |

### PipelineConfig

| 模式 | RC | VA | SA | ST | 總延遲 | 用途 |
|------|----|----|----|----|--------|------|
| `fast` | 1 | 0 | 0 | 0 | 1 | 快速模擬 |
| `standard` | 1 | 0 | 1 | 0 | 2 | 平衡 |
| `hardware` | 1 | 1 | 1 | 1 | 4 | 精確時序 |

---

## 3. NI 參數

### NIConfig

| 參數 | 預設 | 範圍 | 效能影響 |
|------|------|------|----------|
| `max_outstanding` | 16 | 4-64 | 並行交易數，↑ 吞吐量 |
| `req_buffer_depth` | 8 | 4-32 | 請求緩衝 |
| `resp_buffer_depth` | 8 | 4-32 | 回應緩衝 |
| `flit_payload_size` | 32 B | 16-128 | ↑ 減少 flits/packet |
| `axi_data_width` | 64 bits | 32-128 | AXI 頻寬 |

---

## 4. 效能目標設定範例

### 目標：達到 30 B/cycle 吞吐量

**約束**：
- Selector 瓶頸 = rows × 8 = 32 B/c（需 rows ≥ 4）
- 需避免壅塞

**建議配置**：

```python
MeshConfig(
    cols=5,
    rows=4,  # 32 B/c capacity
)

RouterConfig(
    buffer_depth=8,  # 增加緩衝減少壅塞
    pipeline=PipelineConfig.fast(),
)

NIConfig(
    max_outstanding=32,  # 高並行
    req_buffer_depth=16,
)
```

**預期結果**：
- Max Throughput = 32 B/c
- 目標 30 B/c → 94% utilization → ✓ 可達成

---

### 目標：最小延遲（低負載）

**建議配置**：

```python
RouterConfig(
    buffer_depth=4,  # 最小即可
    pipeline=PipelineConfig.fast(),  # 1 cycle/hop
)
```

**預期 Latency**：
```
L_min = hops × 1 cycle
```

---

### 目標：支援 64 個並行交易

**建議配置**：

```python
NIConfig(
    max_outstanding=64,
    req_buffer_depth=32,
    resp_buffer_depth=32,
)
```

---

## 5. 參數與效能指標對應表

| 參數 | ↑ Throughput | ↓ Latency | ↓ 壅塞 | 備註 |
|------|-------------|-----------|--------|------|
| ↑ `buffer_depth` | ○ | ✗ | ✓ | 面積增加 |
| ↑ `max_outstanding` | ✓ | ○ | ○ | 記憶體增加 |
| ↑ `mesh size` | ✓ | ✗ | ○ | 總容量增加 |
| ↓ `pipeline` | ○ | ✓ | ○ | 模擬精度降低 |

圖例：✓ 正相關、✗ 負相關、○ 無明顯影響

---

## 6. 下一步：效能實驗設計

1. 定義效能目標（例：25 B/c, < 10 cycle latency）
2. 選擇初始參數配置
3. 執行模擬
4. 用驗證框架確認結果合理
5. 調整參數迭代

```bash
# 執行模擬後驗證
make test_performance
```
