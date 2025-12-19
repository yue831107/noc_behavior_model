# 內部介面架構

本文件說明元件介面與 Valid/Ready Handshake 機制。

---

## 1. Valid/Ready Handshake Protocol

本模型使用 **Valid/Ready Handshake** 作為外部介面：

```
┌──────────┐                    ┌──────────┐
│ Upstream │  ─── valid ────►  │Downstream│
│  Router  │  ─── flit ─────►  │  Router  │
│          │  ◄── ready ─────  │          │
└──────────┘                    └──────────┘
```

**Handshake 規則**:
1. Upstream 設定 `valid=True` 並提供 `flit`
2. Downstream 設定 `ready=True` 表示可接收
3. 當 `valid && ready` 同時為 True 時，傳輸成功
4. 傳輸完成後清除 valid 和 flit

**內部仍保留 Credit 機制**:
- RouterPort 內部使用 credit 追蹤下游 buffer 空間
- `can_send()` 檢查是否有足夠 credit
- 當 buffer 消耗 flit 時釋放 credit

---

## 2. RouterPort Interface

```python
class RouterPort:
    """Router port with valid/ready handshake interface."""

    # 內部 Buffer
    _buffer: FlitBuffer
    _output_credit: CreditFlowControl

    # Ingress Interface (接收來自上游)
    in_valid: bool       # 上游設定
    in_flit: Flit        # 上游設定
    out_ready: bool      # 本 port 設定 (= buffer 有空間)

    # Egress Interface (發送到下游)
    out_valid: bool      # 本 port 設定
    out_flit: Flit       # 本 port 設定
    in_ready: bool       # 下游設定

    # 方法
    def update_ready(self) -> None:
        """更新 out_ready (buffer 狀態)"""

    def sample_input(self) -> bool:
        """執行 handshake，接收 flit"""

    def set_output(self, flit: Flit) -> bool:
        """設定輸出 valid/flit"""

    def clear_output_if_accepted(self) -> bool:
        """若下游接受則清除輸出"""
```

---

## 3. PortWire - Router ↔ Router 連接

使用 **PortWire** 連接兩個 RouterPort，自動傳播信號：

```
Router A [EAST]                    Router B [WEST]
================                   ================
out_valid  ──────────────────────► in_valid
out_flit   ──────────────────────► in_flit
in_ready   ◄────────────────────── out_ready

          ◄────────────────────────
          ──────────────────────────►
          ──────────────────────────►
          (反向信號同理)
```

**PortWire 類別**:
```python
class PortWire:
    """連接兩個 RouterPort 的雙向線路"""
    port_a: RouterPort
    port_b: RouterPort

    def propagate_signals(self) -> None:
        """傳播 valid/ready/flit 信號"""
        # A → B
        port_b.in_valid = port_a.out_valid
        port_b.in_flit = port_a.out_flit
        port_a.in_ready = port_b.out_ready
        # B → A
        port_a.in_valid = port_b.out_valid
        port_a.in_flit = port_b.out_flit
        port_b.in_ready = port_a.out_ready

    def propagate_credit_release(self) -> None:
        """處理 credit 釋放"""
```

**連接程式碼** (`mesh.py`):
```python
# 建立 PortWire 而非設定 neighbor
req_wire = PortWire(
    router1.get_req_port(dir1),
    router2.get_req_port(dir2)
)
self._req_wires.append(req_wire)
```

---

## 4. 處理週期 (Phased Cycle)

Mesh 使用 8 階段週期處理：

> **重要**: Sample 必須在 Propagate **之前**執行。這是因為 sample 取樣的是**上一個 cycle** 結束時 propagate 設定的信號。若順序錯誤，當前 cycle 的 propagate 會覆蓋上一 cycle 的信號。

```python
def process_cycle(self, current_time: int) -> None:
    # Phase 1: Sample inputs (從上一 cycle 的 propagate 結果取樣)
    for router in self.routers.values():
        router.sample_all_inputs()

    # Phase 2: 清除 input 信號
    for router in self.routers.values():
        router.clear_all_input_signals()

    # Phase 3: 更新 ready 信號 (基於新的 buffer 狀態)
    for router in self.routers.values():
        router.update_all_ready()

    # Phase 4: 路由轉發
    for router in self.routers.values():
        router.route_and_forward(current_time)

    # Phase 5: 傳播信號 (供下一 cycle sample)
    self._propagate_all_wires()

    # Phase 6: 清除已接受的輸出
    for router in self.routers.values():
        router.clear_accepted_outputs()

    # Phase 7: Credit 釋放
    self._handle_credit_release()

    # Phase 8: NI 處理與傳輸
    for ni in self.nis.values():
        ni.process_cycle(current_time)
    self._transfer_ni_flits(current_time)
```

### Phase 順序圖

```
Cycle N-1                           Cycle N
─────────────────────────────────   ─────────────────────────────────
        ...                         Phase 1: sample_all_inputs()
        Phase 5: propagate() ──────────────────►  (取樣上一 cycle 的信號)
        ...                         Phase 2: clear_input_signals()
                                    Phase 3: update_ready()
                                    Phase 4: route_and_forward()
                                    Phase 5: propagate() ────────────► Cycle N+1
                                    Phase 6: clear_accepted_outputs()
                                    Phase 7: credit_release()
```

---

## 5. Selector ↔ EdgeRouter 連接 (PortWire)

Selector 使用 **PortWire** 連接 EdgeRouter，與 Router 間連接相同：

```
Selector                                              EdgeRouter (0, row)
┌──────────────────────────────────┐                 ┌─────────────────────┐
│         EdgeRouterPort           │                 │      req_router     │
│  ┌────────────────────────────┐  │                 │  ┌───────────────┐  │
│  │        _req_port           │  │    PortWire     │  │  LOCAL port   │  │
│  │  ┌──────────────────────┐  │  │                 │  │ ┌───────────┐ │  │
│  │  │ out_valid ───────────┼──┼──┼─────────────────┼──┼►│ in_valid  │ │  │
│  │  │ out_flit  ───────────┼──┼──┼─────────────────┼──┼►│ in_flit   │ │  │
│  │  │ in_ready  ◄──────────┼──┼──┼─────────────────┼──┼─│ out_ready │ │  │
│  │  │ _output_credit       │  │  │                 │  │ │ (buffer)  │ │  │
│  │  └──────────────────────┘  │  │                 │  │ └───────────┘ │  │
│  └────────────────────────────┘  │                 │  └───────────────┘  │
└──────────────────────────────────┘                 └─────────────────────┘

Response 方向 (EdgeRouter → Selector):
┌─────────────────────┐                 ┌──────────────────────────────────┐
│      resp_router    │                 │         EdgeRouterPort           │
│  ┌───────────────┐  │    PortWire     │  ┌────────────────────────────┐  │
│  │  LOCAL port   │  │                 │  │        _resp_port          │  │
│  │ ┌───────────┐ │  │                 │  │  ┌──────────────────────┐  │  │
│  │ │ out_valid ├─┼──┼─────────────────┼──┼──┼►│ in_valid           │  │  │
│  │ │ out_flit  ├─┼──┼─────────────────┼──┼──┼►│ in_flit            │  │  │
│  │ │ in_ready  │◄┼──┼─────────────────┼──┼──┼─│ out_ready          │  │  │
│  │ └───────────┘ │  │                 │  │  │ (buffer→egress)     │  │  │
│  └───────────────┘  │                 │  │  └──────────────────────┘  │  │
└─────────────────────┘                 │  └────────────────────────────┘  │
                                        └──────────────────────────────────┘
```

### 連接建立

```python
def connect_edge_routers(self, edge_routers: List[EdgeRouter]) -> None:
    for router in edge_routers:
        row = router.coord[1]
        port = self.edge_ports[row]

        # Request Wire: Selector._req_port <-> EdgeRouter.req.LOCAL
        req_local = router.req_router.ports[Direction.LOCAL]
        port._req_wire = PortWire(port._req_port, req_local)

        # Response Wire: EdgeRouter.resp.LOCAL <-> Selector._resp_port
        resp_local = router.resp_router.ports[Direction.LOCAL]
        port._resp_wire = PortWire(resp_local, port._resp_port)

        # 初始化 credit = EdgeRouter buffer depth
        port._req_port._output_credit = CreditFlowControl(
            initial_credits=req_local._buffer_depth
        )
```

### Credit 管理

| 項目 | 說明 |
|------|------|
| 初始值 | EdgeRouter.req.LOCAL buffer depth |
| 消耗 | `clear_req_if_accepted()` 時 (handshake 完成) |
| 釋放 | `handle_credit_release()` → `PortWire.propagate_credit_release()` |
| 查詢 | `edge_port.available_credits` property |

---

## 6. Router ↔ Master NI 連接

Router LOCAL port 連接 Master NI，使用信號介面 (但未使用 PortWire)：

### 現行架構

```
Router (col, row)                          Master NI
┌─────────────────────────────┐           ┌─────────────────┐
│    req_router.LOCAL port    │           │                 │
│  ┌───────────────────────┐  │           │                 │
│  │ out_valid ─────────────┼──┼── check ─►│ receive_req_   │
│  │ out_flit  ─────────────┼──┼── get ───►│  flit()        │
│  │ (clear after accept)   │  │           │                 │
│  └───────────────────────┘  │           │                 │
└─────────────────────────────┘           └─────────────────┘

    resp_router.LOCAL port                  Master NI
┌─────────────────────────────┐           ┌─────────────────┐
│  ┌───────────────────────┐  │           │                 │
│  │ in_valid  ◄────────────┼──┼── set ────│ get_resp_flit()│
│  │ in_flit   ◄────────────┼──┼── set ────│                │
│  └───────────────────────┘  │           │                 │
└─────────────────────────────┘           └─────────────────┘
```

### 傳輸邏輯 (`_transfer_ni_flits`)

```python
def _transfer_ni_flits(self, current_time: int) -> None:
    for coord, ni in self.nis.items():
        router = self.routers[coord]

        # Request: Router LOCAL out → NI
        req_local = router.get_req_port(Direction.LOCAL)
        if req_local.out_valid and req_local.out_flit is not None:
            flit = req_local.out_flit
            if ni.receive_req_flit(flit):
                # 模擬 handshake: 清除 output
                req_local.out_valid = False
                req_local.out_flit = None

        # Response: NI → Router LOCAL in
        resp_local = router.get_resp_port(Direction.LOCAL)
        if ni.has_pending_response() and not resp_local.in_valid:
            flit = ni.get_resp_flit()
            if flit is not None:
                resp_local.in_valid = True
                resp_local.in_flit = flit
```

> **注意**: Router-NI 連接目前使用直接信號設定而非 PortWire，因為 NI 內部處理邏輯不同於 Router。

---

## 7. Valid/Ready 與 Credit 的對應

| 舊機制 | 新機制 |
|--------|--------|
| `output_credit.can_send()` | 內部檢查 + `downstream.out_ready` |
| `output_credit.consume()` | handshake: `valid && ready` |
| `pop()` → `neighbor.output_credit.release()` | `out_ready` 自動反映 buffer 狀態 |

---

## 相關文件

- [Router 規格](01_router.md)
- [Routing Selector 規格](03_routing_selector.md)
- [設計決策](08_design_decisions.md) - Flow Control
