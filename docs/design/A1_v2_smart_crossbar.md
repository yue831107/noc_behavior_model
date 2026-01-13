# 附錄 A1: V2 Smart Crossbar 架構 (未實作)

> **注意**: 本文件描述未來規劃的 V2 架構，目前尚未實作。僅供參考。

本文件定義使用 **Smart Crossbar** 取代 V1 的單一 Routing Selector，解決單點瓶頸問題。

---

## 1. 系統概述

### 1.1 架構

```
┌─────────────────────────────────────────────────────────────────────┐
│                         AXI Subsystem                               │
│  ┌───────┐    ┌─────────┐    ┌────────────────┐                    │
│  │  MCU  │    │ AXI DMA │    │ DRAM Controller│                    │
│  │(Host) │    │         │    │                │                    │
│  └───┬───┘    └────┬────┘    └───────┬────────┘                    │
│      │ AXI M       │ AXI M/S         │ AXI S                       │
│      └─────────────┴─────────────────┘                              │
│                      │                                              │
│               ┌──────┴──────┐                                       │
│               │  AXI XBAR   │                                       │
│               └──┬──┬──┬──┬─┘                                       │
│                  │  │  │  │  (4× AXI S)                             │
└──────────────────┼──┼──┼──┼─────────────────────────────────────────┘
                   │  │  │  │
            ┌──────┴──┴──┴──┴──────┐
            │    NI  NI  NI  NI    │  ← 4 個獨立 NI (各含 Address Translation)
            │    3   2   1   0     │
            └──────┬──┬──┬──┬──────┘
                   │  │  │  │
         ┌─────────┴──┴──┴──┴─────────┐
         │      Smart Crossbar        │  ← 4×4 全連接交換
         │         (4×4)              │
         └───┬─────┬─────┬─────┬──────┘
             │     │     │     │
           (0,3) (0,2) (0,1) (0,0)      ← Edge Routers (Local port)
             ║     ║     ║     │
             ║     ║     ║     │        ← N/S 互連
      ┌──────╨─────╨─────╨─────┴─────────────────────────────────┐
      │  (0,3)────(1,3)───(2,3)───(3,3)───(4,3)                 │
      │    ║   R    │  R     │  R    │  R    │  R                │
      │    ║       NI       NI      NI      NI                   │
      │  (0,2)────(1,2)───(2,2)───(3,2)───(4,2)                 │
      │    ║   R    │  R     │  R    │  R    │  R                │
      │    ║       NI       NI      NI      NI                   │
      │  (0,1)────(1,1)───(2,1)───(3,1)───(4,1)                 │
      │    ║   R    │  R     │  R    │  R    │  R                │
      │    ║       NI       NI      NI      NI                   │
      │  (0,0)────(1,0)───(2,0)───(3,0)───(4,0)                 │
      │       R    │  R     │  R    │  R    │  R                │
      │           NI       NI      NI      NI                   │
      └─────────────────────────────────────────────────────────┘
```

### 1.2 V1 vs V2 比較

| 特性 | V1: Routing Selector | V2: Smart Crossbar |
|------|---------------------|-------------------|
| **NI 數量** | 1 | 4 (獨立) |
| **入口結構** | 單一 Selector | 4×4 Crossbar |
| **AXI 連接** | 1 × AXI S | 4 × AXI S |
| **理論頻寬** | 1× | 4× |
| **單點瓶頸** | ❌ 有 | ✓ 無 |
| **冗餘性** | ❌ 無 | ✓ 部分 |
| **複雜度** | 低 | 中等 |
| **硬體成本** | 低 | 較高 (4 NI + Crossbar) |

### 1.3 設計優勢

1. **消除單點瓶頸**: 4 個獨立 NI 可並行處理
2. **負載分散**: Smart Crossbar 動態分配流量
3. **容錯能力**: 單一 NI 故障時其他繼續運作
4. **高吞吐**: 理論 4 倍頻寬提升

---

## 2. Smart Crossbar

### 2.1 架構

```
          NI 3      NI 2      NI 1      NI 0
           │         │         │         │
           ▼         ▼         ▼         ▼
    ┌──────────────────────────────────────────┐
    │            Smart Crossbar (4×4)          │
    │                                          │
    │    ┌───┐   ┌───┐   ┌───┐   ┌───┐        │
    │ ───┤ × ├───┤ × ├───┤ × ├───┤ × ├───     │
    │    └─┬─┘   └─┬─┘   └─┬─┘   └─┬─┘        │
    │      │   ╲   │   ╲   │   ╲   │          │
    │      │    ╲  │    ╲  │    ╲  │          │
    │    ┌─┴─┐   ╲─┴─┐   ╲─┴─┐   ╲─┴─┐        │
    │ ───┤ × ├───┤ × ├───┤ × ├───┤ × ├───     │
    │    └─┬─┘   └─┬─┘   └─┬─┘   └─┬─┘        │
    │      │       │       │       │          │
    │     ... (全連接網路) ...                  │
    │                                          │
    └──────┬───────┬───────┬───────┬──────────┘
           │       │       │       │
         (0,3)   (0,2)   (0,1)   (0,0)
```

**連接特性**:
- 任意 NI 可連接任意 Edge Router
- 支援多路徑並行傳輸 (Non-blocking)
- Request / Response 獨立路徑

### 2.2 Crossbar 參數

```python
class SmartCrossbarConfig:
    num_ni: int = 4                    # 輸入 NI 數
    num_edge_routers: int = 4          # 輸出 Edge Router 數
    crossbar_type: str = "full"        # "full" (Non-blocking) 或 "partial"

    # Arbitration
    arbitration: str = "round_robin"   # Arbitration Policy
    priority_scheme: str = "equal"     # "equal", "weighted", "strict"

    # Buffer
    input_buffer_depth: int = 4        # 輸入 Buffer 深度
    output_buffer_depth: int = 4       # 輸出 Buffer 深度

    # 路徑選擇
    selection_algorithm: str = "equivalence"  # 路徑選擇演算法
    hop_weight: float = 1.0
    credit_weight: float = 1.0
```

### 2.3 Routing 決策

**Request Path (NI → NoC)**

```python
def crossbar_route_request(ni_id: int, packet: Packet) -> int:
    """
    決定封包從哪個 Edge Router 進入 NoC。

    Returns: edge_router_y (0-3)
    """
    dest_x, dest_y = packet.dest_coord

    # 策略 1: Shortest Path (選擇對應 dest_y 的 Edge Router)
    if selection_algorithm == "shortest":
        return dest_y

    # 策略 2: Equivalence (考慮 hop + credit)
    if selection_algorithm == "equivalence":
        best_port = None
        min_cost = float('inf')

        for edge_y in range(num_edge_routers):
            if not is_port_available(ni_id, edge_y):
                continue

            hop = abs(dest_y - edge_y) + dest_x  # Manhattan Distance
            credit = get_edge_router_credit(edge_y)
            cost = hop_weight * hop - credit_weight * credit

            if cost < min_cost:
                min_cost = cost
                best_port = edge_y

        return best_port

    # 策略 3: Round-robin (負載平衡)
    if selection_algorithm == "round_robin":
        return (ni_id + packet.seq) % num_edge_routers
```

**Response Path (NoC → NI)**

Response 封包的 `dest` 欄位包含原始 NI ID，Crossbar 路由回正確 NI。

```python
def crossbar_route_response(edge_router_y: int, packet: Packet) -> int:
    """決定 Response 封包送往哪個 NI。"""
    # Response Header 包含原始 src_ni_id
    return packet.src_ni_id
```

### 2.4 並行能力

| 情境 | V1 行為 | V2 行為 |
|------|---------|---------|
| 4 個同時 Request | 串列 | ✓ 並行 (4 NI) |
| 同 NI → 同 Edge Router | 串列 | 串列 (資源衝突) |
| 同 NI → 不同 Edge Router | N/A | ✓ 並行 |
| 不同 NI → 同 Edge Router | N/A | 串列 (仲裁後) |
| 不同 NI → 不同 Edge Router | N/A | ✓ 並行 |

---

## 3. Multi-NI 配置

### 3.1 NI 分配

```python
class MultiNIConfig:
    """Multi-NI 配置。"""

    num_ni: int = 4

    # NI ID 分配 (對應 AXI XBAR port)
    ni_axi_port_map: dict = {
        0: "AXI_S0",
        1: "AXI_S1",
        2: "AXI_S2",
        3: "AXI_S3"
    }

    # 地址空間分配 (各 NI 負責的 Node ID 範圍)
    # 可選: 靜態分配或動態路由
    address_partition: str = "dynamic"  # "static" 或 "dynamic"
```

### 3.2 Address Routing (V2)

**靜態分配模式**:
```
NI 0: Node ID 0-3   → (1,0)~(4,0)  [Row 0]
NI 1: Node ID 4-7   → (1,1)~(4,1)  [Row 1]
NI 2: Node ID 8-11  → (1,2)~(4,2)  [Row 2]
NI 3: Node ID 12-15 → (1,3)~(4,3)  [Row 3]
```

**動態路由模式**:
- 任意 NI 可存取任意 Node
- Smart Crossbar 處理路由決策
- 需 AXI XBAR 層級地址解碼

---

## 4. 效能比較

### 4.1 預期指標差異

| 指標 | V1 (預期) | V2 (預期) | 改善幅度 |
|------|-----------|-----------|----------|
| Max Throughput | 1× | ~3.5-4× | 顯著 |
| Avg Latency (低負載) | L | L + δ | 略增 (Crossbar Latency) |
| Avg Latency (高負載) | 急遽上升 | 穩定 | 顯著 |
| 飽和點 | ~25% Injection | ~80%+ Injection | 顯著 |
| Buffer 使用率 | 高度不均 | 較均勻 | 改善 |

---

## 相關文件

- [系統概述](01_overview.md)
- [Routing Selector 規格](04_routing_selector.md) - V1 架構
- [指標與視覺化](12_metrics.md) - V1 vs V2 比較指標
