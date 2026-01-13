# Flit 格式

NoC 內的基本傳輸單元，參考 FlooNoC 格式設計（簡化版）。

---

## 1. Header 格式 (hdr_t) - 20 bits

| Bit | 欄位 | 寬度 | 說明 |
|-----|------|------|------|
| [0] | rob_req | 1 | Reorder Buffer 請求旗標 |
| [5:1] | rob_idx | 5 | RoB 索引 (支援 32 entries) |
| [10:6] | dst_id | 5 | 目標節點 {x[2:0], y[1:0]} |
| [15:11] | src_id | 5 | 來源節點 {x[2:0], y[1:0]} |
| [16] | last | 1 | Packet 結束標記 (最後一個 flit) |
| [19:17] | axi_ch | 3 | AXI Channel 類型 |

### Node ID 格式 (5 bits)

| Bit | 欄位 | 寬度 | 範圍 |
|-----|------|------|------|
| [4:2] | x | 3 | 0~4 |
| [1:0] | y | 2 | 0~3 |

### AXI Channel 類型 (axi_ch)

| 值 | 名稱 | Link | 說明 |
|----|------|------|------|
| 0 | AW | Req | Write Address |
| 1 | W | Req | Write Data |
| 2 | AR | Req | Read Address |
| 3 | B | Rsp | Write Response |
| 4 | R | Rsp | Read Response |

---

## 2. Payload 格式

### 2.1 AW Channel (53 bits)

| Bit | 欄位 | 寬度 |
|-----|------|------|
| [31:0] | awaddr | 32 |
| [39:32] | awid | 8 |
| [47:40] | awlen | 8 |
| [50:48] | awsize | 3 |
| [52:51] | awburst | 2 |

### 2.2 W Channel (288 bits)

| Bit | 欄位 | 寬度 |
|-----|------|------|
| [255:0] | wdata | 256 |
| [287:256] | wstrb | 32 |

### 2.3 AR Channel (53 bits)

| Bit | 欄位 | 寬度 |
|-----|------|------|
| [31:0] | araddr | 32 |
| [39:32] | arid | 8 |
| [47:40] | arlen | 8 |
| [50:48] | arsize | 3 |
| [52:51] | arburst | 2 |

### 2.4 B Channel (10 bits)

| Bit | 欄位 | 寬度 |
|-----|------|------|
| [7:0] | bid | 8 |
| [9:8] | bresp | 2 |

### 2.5 R Channel (266 bits)

| Bit | 欄位 | 寬度 |
|-----|------|------|
| [255:0] | rdata | 256 |
| [263:256] | rid | 8 |
| [265:264] | rresp | 2 |

---

## 3. Flit 格式

### 3.1 各 Channel Flit 大小

| Channel | Header | Payload | Flit Total |
|---------|--------|---------|------------|
| AW | 20 | 53 | 73 bits |
| W | 20 | 288 | **308 bits** |
| AR | 20 | 53 | 73 bits |
| B | 20 | 10 | 30 bits |
| R | 20 | 266 | **286 bits** |

### 3.2 Union 對齊

為了讓不同 channel 能用同一個 type 傳輸，使用 union 對齊到最大值：

#### Request Channel Union (308 bits)

```
typedef union packed {
    floo_aw_flit_t axi_aw;   // 73 bits  + 235 bits rsvd = 308 bits
    floo_w_flit_t  axi_w;    // 308 bits + 0 bits rsvd   = 308 bits
    floo_ar_flit_t axi_ar;   // 73 bits  + 235 bits rsvd = 308 bits
} floo_req_chan_t;
```

| Channel | 實際大小 | rsvd | Union 大小 |
|---------|----------|------|------------|
| AW | 73 bits | 235 bits | 308 bits |
| W | 308 bits | 0 bits | 308 bits |
| AR | 73 bits | 235 bits | 308 bits |

#### Response Channel Union (286 bits)

```
typedef union packed {
    floo_b_flit_t axi_b;     // 30 bits  + 256 bits rsvd = 286 bits
    floo_r_flit_t axi_r;     // 286 bits + 0 bits rsvd   = 286 bits
} floo_rsp_chan_t;
```

| Channel | 實際大小 | rsvd | Union 大小 |
|---------|----------|------|------------|
| B | 30 bits | 256 bits | 286 bits |
| R | 286 bits | 0 bits | 286 bits |

---

## 4. Physical Link 格式

Physical Link = valid + ready + flit data，一個 cycle 傳輸一個完整 flit。

### 4.1 Request Link (310 bits)

| Bit | 欄位 | 寬度 | 說明 |
|-----|------|------|------|
| [0] | valid | 1 | 資料有效 |
| [1] | ready | 1 | 接收端準備好 |
| [309:2] | req | 308 | Request flit (union) |

```
┌───────┬───────┬─────────────────────────────────────┐
│ valid │ ready │            req_chan (308 bits)      │
│ 1-bit │ 1-bit │     hdr(20) + W_payload(288)        │
└───────┴───────┴─────────────────────────────────────┘
                 Total: 310 bits

註：AW/AR 傳輸時，payload 區域包含 53 bits 資料 + 235 bits rsvd
```

### 4.2 Response Link (288 bits)

| Bit | 欄位 | 寬度 | 說明 |
|-----|------|------|------|
| [0] | valid | 1 | 資料有效 |
| [1] | ready | 1 | 接收端準備好 |
| [287:2] | rsp | 286 | Response flit (union) |

```
┌───────┬───────┬─────────────────────────────────────┐
│ valid │ ready │            rsp_chan (286 bits)      │
│ 1-bit │ 1-bit │     hdr(20) + R_payload(266)        │
└───────┴───────┴─────────────────────────────────────┘
                 Total: 288 bits

註：B 傳輸時，payload 區域包含 10 bits 資料 + 256 bits rsvd
```

### 4.3 Physical Link 總結

| Link | Flit 大小 | valid + ready | Physical Link 寬度 |
|------|-----------|---------------|-------------------|
| Req | 308 bits | 2 bits | **310 bits** |
| Rsp | 286 bits | 2 bits | **288 bits** |

---

## 5. RoB 設計分析

### 5.1 rob_idx 與 Outstanding 的關係

```
RoBSize = 總 outstanding 容量
rob_idx_bits = $clog2(RoBSize)
```

| RoBSize | rob_idx 位寬 | 最大 Outstanding |
|---------|--------------|------------------|
| 16 | 4 bits | 16 |
| 32 | 5 bits | 32 |
| 64 | 6 bits | 64 |
| 128 | 7 bits | 128 |

### 5.2 rob_idx 與 axi_id 的關係

兩者**獨立**，用途不同：

| 欄位 | 位置 | 用途 |
|------|------|------|
| axi_id | Payload 內 | 識別 transaction stream (per-ID ordering) |
| rob_idx | Header 內 | RoB entry 索引 (跨所有 ID 共享) |

### 5.3 FlooNoC RoB 架構

```
                    ┌─────────────────────────────────────┐
                    │         Reorder Buffer              │
                    │  ┌─────┬─────┬─────┬─────┬─────┐   │
                    │  │  0  │  1  │  2  │ ... │ N-1 │   │
                    │  └─────┴─────┴─────┴─────┴─────┘   │
                    │         ↑                           │
                    │     rob_idx                         │
                    └─────────────────────────────────────┘
                              ↑
┌─────────────────────────────┴─────────────────────────────┐
│                    Status Table                            │
│  ┌──────────┬──────────┬──────────┬──────────┐            │
│  │ axi_id=0 │ axi_id=1 │ axi_id=2 │   ...    │            │
│  │ rob_idx  │ rob_idx  │ rob_idx  │          │            │
│  │ rob_req  │ rob_req  │ rob_req  │          │            │
│  └──────────┴──────────┴──────────┴──────────┘            │
└───────────────────────────────────────────────────────────┘
```

- 每個 axi_id 在 Status Table 中追蹤其 outstanding transactions
- 所有 axi_id 共享同一個 Reorder Buffer
- rob_idx 指向 RoB 中的 entry

---

## 6. 設計參數

| 參數 | 值 | 說明 |
|------|-----|------|
| X_BITS | 3 | X 座標位寬 |
| Y_BITS | 2 | Y 座標位寬 |
| ROB_IDX_BITS | 5 | RoB 索引位寬 (32 entries) |
| AXI_ID_WIDTH | 8 | AXI ID 位寬 |
| AXI_ADDR_WIDTH | 32 | AXI 位址位寬 |
| AXI_DATA_WIDTH | 256 | AXI 資料位寬 |

---

## 7. 與舊格式對比

| 項目 | 舊格式 | 新格式 |
|------|--------|--------|
| Header 大小 | 18 bytes | 20 bits |
| Flit 類型 | HEAD/BODY/TAIL | last bit |
| packet_id | 16-bit | 移除 |
| rob_idx | 無 | 5-bit |
| AXI ID | 4-bit | 8-bit |
| Data Width | 64-bit | 256-bit |
| Req Flit | 可變 | 固定 308 bits |
| Rsp Flit | 可變 | 固定 286 bits |
| Req Link | N/A | 310 bits |
| Rsp Link | N/A | 288 bits |

---

## 相關文件

- [Router 規格](02_router.md)
- [Network Interface 規格](03_network_interface.md)
