# 指標與視覺化

本文件定義效能指標與視覺化需求。

---

## 1. 效能指標

效能指標透過各元件的 Stats 類別收集，而非獨立的 Metrics 模組。

### 1.1 元件層級統計

| 元件 | Stats 類別 | 主要指標 |
|------|-----------|---------|
| Buffer | `BufferStats` | 讀寫次數、平均/最大佔用率 |
| Router | `RouterStats` | flit 數量、延遲、阻塞事件 |
| NI | `NIStats` | AXI 交易數、flit 數量 |
| Mesh | `MeshStats` | 總週期、總 flit 數 |
| Selector | `SelectorStats` | 路徑選擇分佈、阻塞次數 |
| Memory | `MemoryStats` | 讀寫次數、bytes 傳輸量 |
| AXI Master | `AXIMasterStats` | 交易數、延遲、完成率 |

### 1.2 BufferStats

```python
@dataclass
class BufferStats:
    total_writes: int = 0
    total_reads: int = 0
    max_occupancy: int = 0
    total_occupancy_samples: int = 0
    cumulative_occupancy: int = 0

    @property
    def avg_occupancy(self) -> float:
        """平均 buffer 佔用率。"""
```

### 1.3 RouterStats

```python
@dataclass
class RouterStats:
    flits_received: int = 0
    flits_forwarded: int = 0
    flits_dropped: int = 0
    total_latency: int = 0
    arbitration_cycles: int = 0
    buffer_full_events: int = 0
    port_utilization: Dict[Direction, int]
```

### 1.4 SelectorStats

```python
@dataclass
class SelectorStats:
    # Ingress
    req_flits_received: int = 0
    req_flits_injected: int = 0
    req_blocked_no_credit: int = 0

    # Egress
    resp_flits_collected: int = 0
    resp_flits_sent: int = 0

    # 路徑選擇分佈
    path_selections: Dict[int, int]  # row -> count
```

### 1.5 AXIMasterStats

```python
@dataclass
class AXIMasterStats:
    # Write
    total_transactions: int = 0
    total_bytes: int = 0
    completed_transactions: int = 0
    completed_bytes: int = 0
    total_latency: int = 0

    # Read
    read_transactions: int = 0
    read_bytes_requested: int = 0
    read_completed: int = 0
    read_bytes_received: int = 0
    read_latency: int = 0
    read_matches: int = 0
```

---

## 2. 統計收集方式

統計資料透過各元件的 `.stats` 屬性存取：

```python
from src.core import V1System

system = V1System(mesh_cols=5, mesh_rows=4)

# 執行模擬後...
system.run(cycles=1000)

# 取得統計
selector_stats = system.selector.stats
mesh_stats = system.mesh.stats

# Router 統計
for pos, router in system.mesh.routers.items():
    print(f"Router {pos}: {router.req_router.stats.flits_forwarded} flits")

# NI 統計
for pos, ni in system.mesh.nis.items():
    print(f"NI {pos}: {ni.stats.req_flits_sent} req flits")
```

---

## 3. 視覺化需求 (規劃中)

視覺化功能目前為規劃階段，尚未實作。

### 3.1 即時 Heatmap (規劃)

```
顯示:
- 各 Router 的 Buffer 佔用率 (顏色深淺)
- 封包流向 (箭頭動畫)
- 阻塞位置 (紅色標記)
```

### 3.2 延遲分佈圖 (規劃)

```
圖表類型: Histogram + CDF
X 軸: 延遲 (cycles)
Y 軸: 封包數量 / 累積機率
```

### 3.3 Throughput vs. Injection Rate (規劃)

```
圖表類型: 折線圖
X 軸: 注入率
Y 軸: 吞吐量
標記: 飽和點
```

---

## 4. V1 vs V2 比較 (規劃中)

V2 Smart Crossbar 架構尚未實作，比較指標為規劃階段。

預計比較項目:
- Throughput 曲線 (不同注入率)
- Latency 分佈
- 瓶頸分析 (Selector blocking vs Crossbar contention)
- 公平性 (各 NI 的 Throughput)

---

## 相關文件

- [模擬參數](06_simulation.md)
- [V2 Smart Crossbar](10_v2_smart_crossbar.md) - V1 vs V2 效能比較
