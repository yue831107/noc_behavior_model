# 模擬參數

本文件定義模擬的參數設定。

---

## 1. 系統配置

系統參數透過 `V1System` 建構時設定：

```python
from src.core import V1System, HostMemory

system = V1System(
    mesh_cols=5,        # Mesh 欄數 (含 Edge Column)
    mesh_rows=4,        # Mesh 列數
    buffer_depth=4,     # Router buffer 深度
    host_memory=HostMemory(size=0x10000),  # 選用 Host Memory
)
```

| 參數 | 預設值 | 說明 |
|------|--------|------|
| `mesh_cols` | 5 | Mesh 欄數 (Column 0 為 Edge Router) |
| `mesh_rows` | 4 | Mesh 列數 |
| `buffer_depth` | 4 | Router/NI buffer 深度 |
| `host_memory` | None | DMA 傳輸用 Host Memory |

---

## 2. 傳輸配置 (TransferConfig)

File-Driven 模擬使用 `TransferConfig` 設定傳輸參數：

```python
from src.config import TransferConfig, TransferMode

config = TransferConfig(
    # Write 參數
    src_addr=0x0000,        # Host Memory 來源位址
    src_size=1024,          # 傳輸大小 (bytes)
    dst_addr=0x1000,        # Node Local Memory 目的位址

    # Read 參數
    read_src_addr=0x1000,   # Node 端讀取位址
    read_size=0,            # 讀取大小 (0 = 同 src_size)

    # 共用參數
    target_nodes="all",     # "all" | [0,1,2] | "range:0-7"
    transfer_mode=TransferMode.BROADCAST,

    # AXI Burst 設定
    max_burst_len=16,       # 最大 burst 長度 (1-256)
    beat_size=8,            # 每 beat 大小 (bytes)
    max_outstanding=8,      # 最大未完成交易數
)
```

### 傳輸模式

| 模式 | 方向 | 說明 |
|------|------|------|
| `BROADCAST` | Write | 相同資料寫入所有節點 |
| `SCATTER` | Write | 資料分割，每節點不同部分 |
| `BROADCAST_READ` | Read | 從所有節點讀取相同位址 |
| `GATHER` | Read | 從各節點收集不同資料 |

---

## 3. YAML 配置檔

配置檔位於 `examples/Host_to_NoC/config/`：

```yaml
# broadcast_write.yaml
transfer:
  src_addr: 0x0000
  src_size: 1024
  dst_addr: 0x1000
  target_nodes: "all"
  max_burst_len: 16
  beat_size: 8
  max_outstanding: 8
  transfer_mode: "broadcast"
```

---

## 4. 已知限制

| 限制 | 說明 |
|------|------|
| 單 beat 讀取 | `submit_read()` 目前僅支援 8 bytes 單 beat 讀取 |
| Write only | File-Driven 測試主要支援 Write 流程 |

---

## 相關文件

- [系統概述](00_overview.md)
- [操作模式](05_operation_modes.md)
- [實作注意事項](09_implementation.md)
