# 操作模式

本文件說明 NoC 的 DMA 與 PIO 操作模式。

---

## 1. 模式定義

| 模式 | 方向 | 觸發者 | 用途 |
|------|------|--------|------|
| DMA | LPDDR ↔ NoC | AXI DMA | 大量資料傳輸 |
| PIO | Host ↔ NoC | CPU | 控制/狀態存取 |

---

## 2. 交易流程

### 2.1 DMA Write (LPDDR → NoC)

```
1. CPU 設定 DMA: src_addr, dest_node, length
2. DMA 從 DRAM 讀取資料
3. DMA 經 AXI XBAR → NI 發送
4. NI 打包成 Flits，經 Routing Selector 進入 NoC
5. 封包路由至目的節點
6. 目的 NI 回傳 Response
```

### 2.2 PIO Read (Host ← NoC)

```
1. CPU 發起 AXI Read，地址映射到 NoC 節點
2. NI 轉換為 Request Flit
3. 經 NoC 路由到目的地
4. 目的地回傳 Data Flit
5. NI Reorder Buffer 確保順序
6. 回傳 AXI Read Response 給 CPU
```

---

## 相關文件

- [系統概述](01_overview.md)
- [Network Interface 規格](03_network_interface.md)
- [Memory Copy 操作](08_memory_operations.md)
