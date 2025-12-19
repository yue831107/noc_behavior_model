# Router 規格

每個 Router 負責 Mesh 網路內的封包轉發。

---

## 1. Ports

| Port | 方向 | 連接對象 |
|------|------|----------|
| N | 北 | (x, y+1) |
| E | 東 | (x+1, y) |
| S | 南 | (x, y-1) |
| W | 西 | (x-1, y) |
| L | 本地 | 本地 NI |

---

## 2. 參數

```python
@dataclass
class RouterConfig:
    buffer_depth: int = 4           # Input buffer 深度 (flits)
    output_buffer_depth: int = 0    # Output buffer 深度 (0 = 無 output buffer)
    flit_width: int = 64            # Flit 寬度 (bits)
    routing_algorithm: str = "XY"   # Routing: "XY" (禁止 Y→X)
    arbitration: str = "wormhole"   # Arbitration: "wormhole" (封包鎖定)
    switching: str = "wormhole"     # Switching: "wormhole"
```

---

## 3. XY Routing (含 Y→X 轉向禁止)

### 3.1 基本規則

XY Routing 是確定性路由演算法：
1. 先沿 X 軸移動（EAST/WEST）
2. 再沿 Y 軸移動（NORTH/SOUTH）
3. 到達目標後轉送至 LOCAL

### 3.2 Y→X 轉向禁止（Deadlock Prevention）

為避免 deadlock，一旦封包進入 Y 軸方向，禁止轉回 X 軸：

```
禁止的轉向 (Y→X):
  NORTH → EAST  ❌
  NORTH → WEST  ❌
  SOUTH → EAST  ❌
  SOUTH → WEST  ❌

允許的轉向:
  任何 → NORTH/SOUTH  ✓ (進入 Y 軸)
  EAST/WEST → 任何    ✓
  LOCAL → 任何        ✓
```

### 3.3 實作

```python
def compute_output_port(self, flit: Flit, input_dir: Direction) -> Optional[Direction]:
    """
    XY 路由計算，禁止 Y→X 轉向。
    """
    dest = flit.dest
    curr = self.coord

    # 計算期望輸出方向
    if dest[0] > curr[0]:
        desired = Direction.EAST
    elif dest[0] < curr[0]:
        desired = Direction.WEST
    elif dest[1] > curr[1]:
        desired = Direction.NORTH
    elif dest[1] < curr[1]:
        desired = Direction.SOUTH
    else:
        return Direction.LOCAL

    # Y→X 轉向檢查
    if input_dir in (Direction.NORTH, Direction.SOUTH):
        if desired in (Direction.EAST, Direction.WEST):
            return None  # 禁止 Y→X 轉向

    return desired
```

---

## 4. Wormhole Arbiter

### 4.1 概念

Wormhole Arbiter 確保封包完整性，避免 flit interleaving：

1. **HEAD flit** 獲得仲裁後，鎖定 input→output 路徑
2. **BODY flit** 無需重新仲裁，直接沿鎖定路徑轉發
3. **TAIL flit** 傳輸後釋放鎖定

### 4.2 狀態機

```
         HEAD flit              TAIL flit
    ┌───────────────►  LOCKED  ────────────────┐
    │                                          │
    │                BODY flit                 │
    │                  (繼續)                   │
    │                                          │
┌───┴───────────┐                      ┌───────▼───────┐
│     IDLE      │◄─────────────────────│   RELEASE     │
│   (可仲裁)     │                      │  (釋放鎖定)   │
└───────────────┘                      └───────────────┘
```

### 4.3 仲裁流程

```python
def arbitrate(self, requests) -> List[Tuple[Direction, Direction, Flit]]:
    """
    仲裁流程：
    1. 優先處理已鎖定的路徑（無需仲裁）
    2. 新請求使用 Round-robin 仲裁
    """
    grants = []

    # Phase 1: 處理已鎖定的路徑
    for input_dir, (flit, output_dir) in requests.items():
        if self.is_input_locked(input_dir):
            grants.append((input_dir, output_dir, flit))

    # Phase 2: Round-robin 處理新請求
    for input_dir in self._get_rr_order():
        if input_dir in requests and not self.is_input_locked(input_dir):
            flit, output_dir = requests[input_dir]
            if not self.is_output_locked(output_dir):
                grants.append((input_dir, output_dir, flit))

    return grants
```

---

## 5. Output Buffer

### 5.1 概念

Output Buffer 提供額外的緩衝，提高吞吐量：

```
Input Buffer → Crossbar → Output Buffer → Link
     ↑                         ↑
   接收 flit              暫存待發送 flit
```

### 5.2 配置

```python
# 啟用 Output Buffer (深度 = 2)
config = RouterConfig(
    buffer_depth=4,
    output_buffer_depth=2
)
```

### 5.3 運作流程

1. **set_output()**: 將 flit push 到 output buffer
2. **update_output_from_buffer()**: 從 buffer peek flit 設定 out_valid
3. **clear_output_if_accepted()**: 若下游接受，pop flit 並清除信號

---

## 6. Req/Resp 物理分離

每個邏輯 Router 由兩個獨立 Router 組成：

```python
class Router:
    """邏輯 Router = Req Router + Resp Router。"""

    def __init__(self, coord: Tuple[int, int], config: RouterConfig):
        self.coord = coord
        self.req_router = ReqRouter(coord, config)   # Request 網路
        self.resp_router = RespRouter(coord, config) # Response 網路

class ReqRouter(XYRouter):
    """Request 路徑 Router - 處理讀寫請求。"""

class RespRouter(XYRouter):
    """Response 路徑 Router - 處理讀寫回應。"""
```

---

## 7. RouterPort 介面

每個 Port 使用 Valid/Ready Handshake：

```python
class RouterPort:
    # Input buffer
    _buffer: FlitBuffer

    # Output buffer (可選)
    _output_buffer: Optional[FlitBuffer]

    # Ingress Interface (接收)
    in_valid: bool       # 上游設定
    in_flit: Flit        # 上游設定
    out_ready: bool      # 本 port 設定

    # Egress Interface (發送)
    out_valid: bool      # 本 port 設定
    out_flit: Flit       # 本 port 設定
    in_ready: bool       # 下游設定
```

---

## 相關文件

- [系統概述](00_overview.md)
- [Flit 格式](04_flit.md)
- [內部介面架構](11_internal_interface.md)
