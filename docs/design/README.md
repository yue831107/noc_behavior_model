# NoC Behavior Model - 設計文件

本目錄包含 NoC Behavior Model 的設計規格，從原始 `spec.md` 切分而來，便於每次專注於單一功能模組。

---

## 文件索引

| 文件 | 說明 | 主要內容 |
|------|------|----------|
| [00_overview.md](00_overview.md) | 系統概述 | V1 架構圖、拓撲參數、特性與限制 |
| [01_router.md](01_router.md) | Router 規格 | Ports、參數、Routing Algorithms |
| [02_network_interface.md](02_network_interface.md) | Network Interface 規格 | Address Translation、Slave/Master NI、資料路徑 |
| [03_routing_selector.md](03_routing_selector.md) | Routing Selector 規格 | 架構、Edge Router 連接、路徑選擇演算法 |
| [04_flit.md](04_flit.md) | Flit 格式 | 類型、結構、大小限制 |
| [05_operation_modes.md](05_operation_modes.md) | 操作模式 | DMA/PIO 模式、交易流程 |
| [06_simulation.md](06_simulation.md) | 模擬參數 | SimConfig、流量模式 |
| [07_metrics.md](07_metrics.md) | 指標與視覺化 | 效能指標、視覺化需求 |
| [08_design_decisions.md](08_design_decisions.md) | 設計決策 | 模擬模型、Req/Resp 分離、Flow Control、AXI4 |
| [09_implementation.md](09_implementation.md) | 實作注意事項 | 語言選擇、模擬架構、檔案結構 |
| [10_v2_smart_crossbar.md](10_v2_smart_crossbar.md) | V2 Smart Crossbar 架構 | Multi-NI、Crossbar 路由、效能比較 |
| [11_internal_interface.md](11_internal_interface.md) | 內部介面架構 | Credit-Based Flow Control、Port Interface |
| [12_memory_copy.md](12_memory_copy.md) | Memory Copy 操作 | 架構、V1 限制、交錯傳輸 |
| [13_golden_verification.md](13_golden_verification.md) | Golden 驗證機制 | Host Write/Read Golden 產生與比對 |

---

## 快速參考

### 依功能分類

**核心元件**
- [Router](01_router.md)
- [Network Interface](02_network_interface.md)
- [Routing Selector](03_routing_selector.md)
- [Flit](04_flit.md)

**系統行為**
- [操作模式](05_operation_modes.md)
- [Memory Copy](12_memory_copy.md)

**模擬與分析**
- [模擬參數](06_simulation.md)
- [指標與視覺化](07_metrics.md)
- [Golden 驗證機制](13_golden_verification.md)

**設計與實作**
- [設計決策](08_design_decisions.md)
- [實作注意事項](09_implementation.md)
- [內部介面架構](11_internal_interface.md)

**架構版本**
- [V1 系統概述](00_overview.md)
- [V2 Smart Crossbar](10_v2_smart_crossbar.md)

---

## 版本歷史

| 版本 | 說明 | 狀態 |
|------|------|------|
| **V1** | Single Entry Routing Selector | 實作完成 |
| **V2** | Smart Crossbar (Multi-NI) | 定義完成 |

---

## 參考圖片

| 檔案 | 內容 |
|------|------|
| `docs/images/NI.jpg` | NI 內部架構 |
| `docs/images/operation.jpg` | DMA/PIO 操作模式 |
| `docs/images/network_interface_slide.jpg` | NI 詳細架構 (Req/Resp 分離) |
| `docs/images/selector.jpg` | V1 Routing Selector 架構 |
| `docs/images/selector_slide.jpg` | V1 Routing Selector 詳細連接 |
| `docs/images/smart_crossbar.jpg` | V2 Smart Crossbar 架構 |
| `docs/images/smart_crossbar_slide.jpg` | V2 Smart Crossbar 詳細架構 |
| `docs/images/sim_engine.jpg` | 模擬引擎架構 |
| `docs/images/test_bench.jpg` | 測試架構圖 |
