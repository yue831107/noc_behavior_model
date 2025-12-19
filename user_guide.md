# NoC Behavior Model - 使用手冊

本文件說明如何使用 NoC Behavior Model 進行模擬與測試。

---

## 1. 快速開始

### 1.1 系統需求

- Python 3.10+
- PyYAML

### 1.2 安裝

```bash
cd noc_behavior_model
pip install pyyaml

# 驗證
make help
```

---

## 2. 指令總覽

執行 `make help` 可查看所有可用指令。

### 2.1 Payload 生成

| 指令 | 說明 |
|------|------|
| `make gen_payload` | 產生 Host-to-NoC payload (1024B) |
| `make gen_noc_payload` | 產生 NoC-to-NoC payload (256B × 16 nodes) |

**自訂選項：**
```bash
make gen_payload PATTERN=random SIZE=2048
make gen_noc_payload NOC_PATTERN=walking_ones NOC_SIZE=512 SEED=123
```

**支援的 Patterns：**
`sequential`, `random`, `constant`, `address`, `walking_ones`, `walking_zeros`, `checkerboard`

---

### 2.2 Host-to-NoC 模擬

Host 透過 NoC 對所有 Compute Node 進行讀寫操作。

#### 三步驟流程

```bash
# Step 1: 產生 Payload BIN 檔
make gen_payload PAYLOAD_SIZE=1048576 PAYLOAD_PATTERN=random

# Step 2: 產生 Transfer Config (YAML)
make gen_config NUM_TRANSFERS=100

# Step 3: 執行模擬
make sim
```

#### 可選參數

| 步驟 | 參數 | 說明 |
|------|------|------|
| `gen_payload` | `PAYLOAD_SIZE=N` | Payload 大小 (bytes) |
| `gen_payload` | `PAYLOAD_PATTERN=X` | Pattern 類型 |
| `gen_config` | `NUM_TRANSFERS=N` | 傳輸數量 (預設: 10) |
| `gen_config` | `TRANSFER_MODE=X` | random / broadcast / scatter |

#### 輸出檔案

```
examples/Host_to_NoC/
├── payload/payload.bin      ← gen_payload 產生
├── config/generated.yaml    ← gen_config 產生
output/
├── metrics/latest.json      ← sim 產生 (供 viz 使用)
```

---

### 2.3 NoC-to-NoC 模擬

Compute Node 之間直接通訊，支援 5 種 traffic patterns。

| 指令 | 說明 |
|------|------|
| `make sim_noc_neighbor` | Neighbor：環狀拓撲 (dst = src + 1) |
| `make sim_noc_shuffle` | Shuffle：位元旋轉 |
| `make sim_noc_bit_reverse` | Bit Reverse：位元反轉 |
| `make sim_noc_random` | Random：隨機目標 |
| `make sim_noc_transpose` | Transpose：座標互換 (x,y) → (y,x) |

**完整流程：**
```bash
make gen_noc_payload     # Step 1: 產生 16 個節點的 payload
make sim_noc_neighbor    # Step 2: 執行 neighbor pattern 測試
make viz                 # Step 3: 產生效能圖表（使用上次模擬結果）
```

---

### 2.4 視覺化

| 指令 | 說明 |
|------|------|
| `make viz` | 產生效能圖表（使用最近一次模擬結果） |

圖表輸出至 `output/charts/` 目錄：
- `buffer_heatmap.png` - Buffer 使用率熱圖
- `throughput_curve.png` - Throughput 曲線
- `latency_histogram.png` - Latency 分佈
- `dashboard.png` - 綜合儀表板

---

### 2.5 測試

| 指令 | 說明 |
|------|------|
| `make test` | 執行所有 pytest 測試 |
| `make test_unit` | 只執行 unit tests |
| `make test_integration` | 只執行 integration tests |

---

### 2.6 清理

| 指令 | 說明 |
|------|------|
| `make clean` | 清除所有產生的檔案 |
| `make clean_payload` | 只清除 Host-to-NoC payload |
| `make clean_noc_payload` | 只清除 NoC-to-NoC payload |

---

## 3. 自訂設定

### 3.1 Transfer Config（YAML 設定檔）

每個模擬都有對應的 YAML 設定檔，可自訂傳輸參數。

**Host-to-NoC 設定檔位置：** `examples/Host_to_NoC/config/`

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `src_addr` | Host Memory 來源位址 | `0x0000` |
| `src_size` | 傳輸大小 (bytes) | `1024` |
| `dst_addr` | Node Local Memory 目的位址 | `0x1000` |
| `target_nodes` | 目標節點 (`"all"` 或 `[0, 1, 2]`) | `"all"` |

**NoC-to-NoC 設定檔位置：** `examples/NoC_to_NoC/config/`

| 參數 | 說明 | 預設值 |
|------|------|--------|
| `pattern` | Traffic pattern 名稱 | `neighbor` |
| `mesh_cols` | Mesh 欄數（含 Edge Column） | `5` |
| `mesh_rows` | Mesh 列數 | `4` |
| `transfer_size` | 每節點傳輸大小 (bytes) | `256` |
| `local_src_addr` | 來源位址（Local Memory） | `0x0000` |
| `local_dst_addr` | 目的位址（Local Memory） | `0x1000` |

**自訂設定步驟：**

```bash
# 1. 複製現有設定
cp examples/NoC_to_NoC/config/neighbor.yaml examples/NoC_to_NoC/config/custom.yaml

# 2. 編輯 custom.yaml（可用任意文字編輯器）

# 3. 執行自訂設定
py -3 examples/NoC_to_NoC/run.py custom
```

---

### 3.2 範例目錄

專案提供兩個完整範例供參考：

| 範例 | 路徑 | 說明 |
|------|------|------|
| **Host-to-NoC** | `examples/Host_to_NoC/` | Host 對所有 Node 的讀寫操作 |
| **NoC-to-NoC** | `examples/NoC_to_NoC/` | Node 之間直接通訊 |

**目錄結構：**

```
examples/
├── Host_to_NoC/
│   ├── run.py              # 執行腳本
│   ├── config/             # YAML 設定檔
│   │   ├── broadcast_write.yaml
│   │   ├── scatter_write.yaml
│   │   └── ...
│   └── payload/            # Payload 檔案 (make gen_payload 產生)
│
└── NoC_to_NoC/
    ├── run.py              # 執行腳本
    ├── config/             # YAML 設定檔
    │   ├── neighbor.yaml
    │   ├── shuffle.yaml
    │   └── ...
    └── payload/            # Payload 檔案 (make gen_noc_payload 產生)
```

---

## 4. Traffic Pattern 說明

### 4.1 五種模式比較

| Pattern | 公式 | 用途 |
|---------|------|------|
| **NEIGHBOR** | `dst = (src + 1) % N` | 低延遲基準測試 |
| **SHUFFLE** | `rotate_left(src)` | 負載平衡測試 |
| **BIT_REVERSE** | `reverse_bits(src)` | 最壞情況延遲 |
| **RANDOM** | `random()` | 真實流量模擬 |
| **TRANSPOSE** | `(x,y) → (y,x)` | 矩陣運算模式 |

### 4.2 Golden 驗證

- 模擬前自動從 Source Node 擷取預期資料
- 模擬後比對 Destination Node 實際資料
- 多個 Source 寫同一 Dest 時，距離最長者贏

---

## 5. 常用工作流程

### 5.1 基本驗證流程

```bash
make gen_noc_payload    # 產生 payload
make sim_noc_neighbor   # 執行模擬（含 Golden 驗證）
make viz                # 產生效能圖表
```

### 5.2 完整測試流程

```bash
make clean              # 清理
make gen_payload        # Host-to-NoC payload
make gen_noc_payload    # NoC-to-NoC payload
make sim_all            # 所有 Host-to-NoC 測試
make test               # pytest 測試
```

---

## 6. 輸出說明

### 6.1 模擬輸出範例

```
======================================================================
 NoC-to-NoC Traffic Test: NEIGHBOR
======================================================================
--- Performance Metrics ---
  Total Cycles:      33
  Throughput:        248.24 bytes/cycle
  Avg Latency:       2.06 cycles/transfer

--- Golden Data Verification ---
  PASSED:            16
  FAILED:            0
  Pass Rate:         100.0%

--- Summary ---
  Status:            PASS
======================================================================
```

### 6.2 檔案結構

```
output/
├── charts/           # 視覺化圖表
│   ├── buffer_heatmap.png
│   ├── throughput_curve.png
│   └── dashboard.png
└── metrics/          # 模擬數據
    └── latest.json   # 最近一次模擬結果
```

---

## 7. 常見問題

### Q: 執行 sim_noc 時出現 "payload file not found"

**A:** 請先執行 `make gen_noc_payload` 產生 payload 檔案。

### Q: Shuffle/Bit-reverse 顯示部分 FAIL

**A:** 這些 pattern 可能有多個 Source 寫入同一 Dest 的情況，是正常行為。

### Q: Windows 上 python 指令無法執行

**A:** Makefile 已使用 `py -3` 避免 PATH 衝突，請直接使用 make 指令。
