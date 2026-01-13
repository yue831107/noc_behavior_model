# 系統概述

本文件說明 NoC Behavior Model 的整體系統架構。

---

## 1. V1 架構圖

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
│               └──────┬──────┘                                       │
│                      │ AXI S                                        │
└──────────────────────┼──────────────────────────────────────────────┘
                       │
                ┌──────┴──────┐
                │     NI      │  ← Protocol Conversion (AXI ↔ Flit)
                └──────┬──────┘
                       │ Req / Resp (獨立雙向)
              ┌────────┴────────┐
              │     Selector    │  ← Path Selection (Hop, Credit)
              └┬──┬──┬──┬───────┘
               │  │  │  │
         ┌─────┘  │  │  └─────┐
         │     ┌──┘  └──┐     │
         ▼     ▼        ▼     ▼
      ┌─────────────────────────────────────────────────────────┐
      │  Edge      │        Compute Nodes (with NI)             │
      │  Routers   │                                            │
      │  (N/S連接) │                                            │
      │     │      │                                            │
      │  (0,3)────(1,3)───(2,3)───(3,3)───(4,3)               │
      │    ║   R    │  R     │  R    │  R    │  R              │
      │    ║       NI       NI      NI      NI                 │
      │  (0,2)────(1,2)───(2,2)───(3,2)───(4,2)               │
      │    ║   R    │  R     │  R    │  R    │  R              │
      │    ║       NI       NI      NI      NI                 │
      │  (0,1)────(1,1)───(2,1)───(3,1)───(4,1)               │
      │    ║   R    │  R     │  R    │  R    │  R              │
      │    ║       NI       NI      NI      NI                 │
      │  (0,0)────(1,0)───(2,0)───(3,0)───(4,0)               │
      │       R    │  R     │  R    │  R    │  R              │
      │           NI       NI      NI      NI                 │
      └─────────────────────────────────────────────────────────┘

      Column 0: Edge Routers (僅 Router，無 NI，N/S 互連)
      Column 1-4: Compute Nodes (Router + NI)
      ║ = Edge Routers 間的 N/S 連接 (Response 路由必需)
```

---

## 2. 拓撲參數

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `mesh_cols` | 5 | 欄數 (含 Edge Column) |
| `mesh_rows` | 4 | 列數 |
| `edge_column` | 0 | Edge Routers 所在欄 |
| `compute_cols` | 1-4 | Compute Nodes 所在欄 |

---

## 3. V1 特性

| 特性 | 說明 |
|------|------|
| Single Entry Point | 所有流量經由單一 Routing Selector 進出 |
| Centralized Routing | 入口點決定路徑 |
| Separate Req/Resp Channels | 雙向平行傳輸 |

---

## 4. 已知限制

- **瓶頸**: 單一進出點限制 Throughput
- **無冗餘**: NI 故障時無備用路徑

---

## 相關文件

- [Router 規格](02_router.md)
- [Network Interface 規格](03_network_interface.md)
- [Routing Selector 規格](04_routing_selector.md)
- [V2 Smart Crossbar 架構](A1_v2_smart_crossbar.md)
