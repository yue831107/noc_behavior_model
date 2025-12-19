# 實作注意事項

本文件說明實作相關的技術決策與指引。

---

## 1. 語言選擇

| 語言 | 優點 | 缺點 |
|------|------|------|
| **Python** | 開發快速，視覺化豐富 (matplotlib, plotly) | 效能較低 |
| **C/C++** | 高效能，適合大規模模擬 | 開發時間長 |
| **Python + C** | 核心 C，介面 Python | 整合複雜度 |

**建議**: 使用 Python 快速原型開發，效能瓶頸再以 C 優化。

---

## 2. 模擬架構

目前採用 cycle-accurate 模擬，以 `V1System` 作為頂層封裝：

```
┌─────────────────────────────────────────────┐
│                 V1System                    │
│  ┌────────────────────────────────────────┐ │
│  │          RoutingSelector               │ │
│  │  (Ingress/Egress 控制)                 │ │
│  └────────────────────────────────────────┘ │
│                     │                       │
│  ┌──────────┬───────┴───────┬────────────┐ │
│  │  Mesh    │ Edge Routers  │ Host AXI   │ │
│  │(Routers) │     (NI)      │  Master    │ │
│  └──────────┴───────────────┴────────────┘ │
│                     │                       │
│  ┌────────────────────────────────────────┐ │
│  │         Component Stats                │ │
│  │  (BufferStats, RouterStats, etc.)      │ │
│  └────────────────────────────────────────┘ │
└─────────────────────────────────────────────┘
```

視覺化功能 (Heatmap, Charts) 目前為規劃階段，尚未實作。

---

## 3. 檔案結構

```
noc_behavior_model/
├── CLAUDE.md              # Claude Code 指引
├── README.md              # 專案說明
├── Makefile               # Make 命令集
├── docs/
│   ├── design/            # 設計文件
│   │   ├── README.md      # 索引
│   │   ├── 00_overview.md
│   │   ├── 01_router.md
│   │   └── ...
│   └── images/            # 架構圖
│       ├── NI.jpg
│       ├── selector.jpg
│       └── ...
├── src/
│   ├── __init__.py
│   ├── config.py          # TransferConfig 定義
│   ├── core/              # NoC 核心元件
│   │   ├── __init__.py    # 公開 API 導出
│   │   ├── flit.py        # Flit 定義
│   │   ├── packet.py      # Packet 組裝/拆解
│   │   ├── buffer.py      # Buffer 與 Credit Flow
│   │   ├── router.py      # Router 模型
│   │   ├── ni.py          # Network Interface
│   │   ├── mesh.py        # Mesh 拓撲
│   │   ├── routing_selector.py  # V1 Selector + V1System
│   │   ├── memory.py      # Host/Local Memory
│   │   ├── axi_master.py  # AXI Master Controller
│   │   ├── host_axi_master.py   # Host 端 AXI Master
│   │   └── golden_manager.py    # 驗證用 Golden Model
│   ├── axi/               # AXI 協定定義
│   │   ├── __init__.py
│   │   └── interface.py   # AXI_AW, AXI_W, AXI_B, etc.
│   └── address/           # 位址映射
│       ├── __init__.py
│       └── address_map.py
├── tests/
│   ├── conftest.py        # pytest fixtures
│   ├── unit/              # 單元測試
│   │   ├── test_xy_routing.py
│   │   ├── test_router_port.py
│   │   ├── test_golden_manager.py
│   │   └── ...
│   └── integration/       # 整合測試
│       ├── test_router_chain.py
│       ├── test_selector_edge_path.py
│       └── ...
├── examples/
│   └── Host_to_NoC/       # DMA 傳輸範例
│       ├── run.py         # 主程式
│       ├── config/        # YAML 配置
│       │   ├── default.yaml
│       │   ├── broadcast_write.yaml
│       │   └── broadcast_read.yaml
│       ├── payload/       # 測試資料
│       └── output/        # 模擬結果
└── tools/
    └── pattern_gen.py     # Payload 產生器
```

---

## 4. 主要元件

| 元件 | 檔案 | 說明 |
|------|------|------|
| V1System | `routing_selector.py` | 頂層系統封裝 |
| RoutingSelector | `routing_selector.py` | 路徑選擇器 |
| Mesh | `mesh.py` | 5×4 2D Mesh |
| Router | `router.py` | XY Router (Req/Resp) |
| SlaveNI | `ni.py` | 節點端 Network Interface |
| MasterNI | `ni.py` | Host 端 NI |
| HostAXIMaster | `host_axi_master.py` | Host AXI Master |
| GoldenManager | `golden_manager.py` | 驗證用 Golden Model |

---

## 5. Make 命令

常用開發命令:

```bash
make test           # 執行所有測試
make test-unit      # 單元測試
make test-integration  # 整合測試
make sim_write      # Broadcast Write 模擬
make sim_read       # Broadcast Read 模擬
make clean          # 清理暫存檔
```

---

## 相關文件

- [系統概述](00_overview.md)
- [模擬參數](06_simulation.md)
