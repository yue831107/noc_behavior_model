# NoC Behavior Model - 設計文件

本目錄包含 NoC Behavior Model 的設計規格文件。

---

## 文件索引

### 核心元件

| 序號 | 文件 | 說明 |
|------|------|------|
| 01 | [系統概述](01_overview.md) | V1 架構圖、拓撲參數 |
| 02 | [Router 規格](02_router.md) | Ports、XY Routing、Wormhole Arbiter |
| 03 | [Network Interface 規格](03_network_interface.md) | SlaveNI、MasterNI、資料路徑 |
| 04 | [Routing Selector 規格](04_routing_selector.md) | 路徑選擇演算法、Edge Router 連接 |
| 05 | [Flit 格式](05_flit.md) | 類型、結構、大小限制 |
| 06 | [內部介面架構](06_internal_interface.md) | Credit-Based Flow Control、Port Interface |

### 系統行為

| 序號 | 文件 | 說明 |
|------|------|------|
| 07 | [操作模式](07_operation_modes.md) | DMA/PIO 模式、交易流程 |
| 08 | [Memory 操作](08_memory_operations.md) | Host-to-NoC 傳輸、交錯傳輸 |
| 09 | [NoC-to-NoC 通訊](09_noc_to_noc.md) | 節點間通訊、Traffic Patterns |
| 10 | [Golden 驗證機制](10_golden_verification.md) | 資料驗證、比對流程 |

### 模擬與分析

| 序號 | 文件 | 說明 |
|------|------|------|
| 11 | [模擬參數](11_simulation.md) | V1System 配置、TransferConfig |
| 12 | [效能指標](12_metrics.md) | Stats 類別、統計收集 |
| 13 | [設計決策](13_design_decisions.md) | 架構選擇、Flow Control 設計 |
| 14 | [效能驗證框架](14_performance_validation.md) | 驗證策略、Baseline 格式 |
| 15 | [Physical Channel 架構比較](15_physical_channel_modes.md) | General vs AXI Mode、Trade-off |

### 附錄

| 序號 | 文件 | 說明 |
|------|------|------|
| A1 | [V2 Smart Crossbar](A1_v2_smart_crossbar.md) | 未來規劃架構 (未實作) |
| - | [硬體參數指南](hardware_parameters_guide.md) | 參數調校參考 |

---

## 快速參考

### 依功能分類

**核心元件**
- [Router](02_router.md) - XY Routing、Wormhole Switching
- [Network Interface](03_network_interface.md) - AXI ↔ Flit 轉換
- [Routing Selector](04_routing_selector.md) - V1 入口/出口點
- [Flit 格式](05_flit.md) - 封包結構

**操作模式**
- [Host-to-NoC](08_memory_operations.md) - Host 對節點傳輸
- [NoC-to-NoC](09_noc_to_noc.md) - 節點間通訊

**驗證與分析**
- [Golden 驗證](10_golden_verification.md) - 資料正確性驗證
- [效能指標](12_metrics.md) - 統計收集
- [效能驗證框架](14_performance_validation.md) - 理論驗證、一致性驗證

---

## 圖片資源

| 檔案 | 內容 |
|------|------|
| `docs/images/NI.jpg` | NI 內部架構 |
| `docs/images/selector.jpg` | V1 Routing Selector 架構 |
| `docs/images/test_bench.jpg` | 測試架構圖 |
