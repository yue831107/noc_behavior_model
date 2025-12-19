# 設計決策

本文件記錄重要的架構設計決策。

---

## 1. 模擬模型決策

| 項目 | 決策 | 說明 |
|------|------|------|
| **模擬精度** | Functional | 功能級模擬，延遲以週期計算 |
| **Switching** | Wormhole | HEAD Flit 不等待完整封包即可轉發 |
| **Virtual Channel** | 不需要 | Req/Resp 物理分離，無需 VC |
| **Req/Resp 分離** | **物理分離** | 雙軌架構 Router 與 NI (類似 FlooNoC) |
| **Flow Control** | Credit-based | 下游回報可用 Buffer 數 |
| **AXI Protocol** | Full AXI4 | 五通道完整支援: AW/W/AR/B/R |
| **實作順序** | V1 優先 | 先完成 Routing Selector 作為基準 |
| **視覺化** | 延後 | 先完成核心模擬，再做視覺化 |

---

## 2. Req/Resp 物理分離

**關鍵設計**: Request 與 Response 使用**物理分離網路**，類似 PULP FlooNoC 架構。

```
┌─────────────────────────────────────────────────────────────────┐
│                    Physical Separation                          │
│                                                                 │
│   ┌─────────────┐              ┌─────────────┐                 │
│   │ Req Router  │◄────────────►│ Req Router  │  ← Request 網路 │
│   └─────────────┘              └─────────────┘                 │
│                                                                 │
│   ┌─────────────┐              ┌─────────────┐                 │
│   │ Resp Router │◄────────────►│ Resp Router │  ← Response 網路│
│   └─────────────┘              └─────────────┘                 │
│                                                                 │
│   每個 "Router" 節點 = Req Router + Resp Router                 │
│   每個 "NI" = Req NI + Resp NI                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 優點

- Req/Resp 互不阻塞 (無 Head-of-line Blocking)
- 無需 Virtual Channel 防止 Deadlock
- 架構清晰，Req 與 Resp 路徑獨立

### 實作影響

| 元件 | 實際結構 |
|------|----------|
| Router | `ReqRouter` + `RespRouter` (獨立 Buffer、Routing) |
| NI | `ReqNI` + `RespNI` |
| Link | 每條 Link 實為 2 條 (Req + Resp) |
| Buffer 計算 | Req/Resp Buffer 分開計算 |

---

## 3. Wormhole Switching

```
封包到達 Router:
1. HEAD Flit 可立即轉發 (若下游有 Buffer 空間)
2. BODY/TAIL Flits 沿建立的路徑跟隨
3. 路徑保持直到 TAIL Flit 通過

特性:
- 低延遲: HEAD 不等待完整封包
- Buffer 效率高: 只需緩衝傳輸中的 Flits
- 路徑預留: 從 HEAD 到 TAIL 保持路徑
```

---

## 4. Credit-based Flow Control

```python
class CreditFlowControl:
    """追蹤下游 Buffer 的可用 Credits。"""

    def __init__(self, initial_credits: int):
        self.credits = initial_credits  # = 下游 Buffer 深度

    def can_send(self) -> bool:
        return self.credits > 0

    def send_flit(self):
        """發送 Flit 時消耗 Credit。"""
        assert self.credits > 0
        self.credits -= 1

    def receive_credit(self):
        """下游釋放 Buffer 時回收 Credit。"""
        self.credits += 1
```

---

## 5. AXI4 Interface Model

```python
class AXI4Interface:
    """完整 AXI4 五通道模型。"""

    # Write Address Channel (AW)
    class AW:
        awaddr: int      # 64-bit 地址
        awlen: int       # Burst 長度 (0-255 → 1-256 beats)
        awsize: int      # Burst 大小 (每 beat bytes)
        awburst: int     # Burst 類型 (FIXED, INCR, WRAP)
        awid: int        # Transaction ID

    # Write Data Channel (W)
    class W:
        wdata: bytes     # Write Data
        wstrb: int       # Byte Strobes
        wlast: bool      # Burst 最後一筆

    # Write Response Channel (B)
    class B:
        bresp: int       # Response (OKAY, EXOKAY, SLVERR, DECERR)
        bid: int         # Transaction ID

    # Read Address Channel (AR)
    class AR:
        araddr: int      # 64-bit 地址
        arlen: int       # Burst 長度
        arsize: int      # Burst 大小
        arburst: int     # Burst 類型
        arid: int        # Transaction ID

    # Read Data Channel (R)
    class R:
        rdata: bytes     # Read Data
        rresp: int       # Response
        rlast: bool      # Burst 最後一筆
        rid: int         # Transaction ID
```

---

## 相關文件

- [Router 規格](01_router.md)
- [Network Interface 規格](02_network_interface.md)
- [內部介面架構](11_internal_interface.md)
