# Routing Selector 規格

入口 Router，決定封包進入 Mesh 的最佳路徑。

---

## 1. 架構

```
                                   ┌─────────────────────────────────────┐
                                   │            Selector                 │
                                   │  ┌───────────┐   ┌───────────┐     │
     From NI ─── Req (orange) ────→│  │  Ingress  │   │  Egress   │     │←── Resp from (0,3)
                                   │  │   Path    │   │   Path    │     │←── Resp from (0,2)
     To NI ←─── Resp (blue) ──────←│  │  Select   │   │  Select   │     │←── Resp from (0,1)
                                   │  └───────────┘   └───────────┘     │←── Resp from (0,0)
                                   │        │               ▲           │
                                   │        ▼               │           │
                                   │   ┌────┴────┬────┬────┴────┐      │
                                   └───┤    │    │    │    │    ├──────┘
                                       ▼    ▼    ▼    ▼    │
                                     (0,3) (0,2) (0,1) (0,0)
                                       │     │     │     │
                                       R     R     R     R   ← Edge Routers (Local port)
```

### 連接方式

- Selector 連接 Column 0 的 4 個 Edge Routers: (0,0), (0,1), (0,2), (0,3)
- 經由各 Router 的 **Local (L) port** 連接
- Request 與 Response 為獨立通道
- **Edge Routers 必須 N/S 互連** (見下文)

---

## 2. Edge Router 連接

```
                Selector
         ┌────────┴────────┐
         │  Req      Resp  │
         └─┬──┬──┬──┬──────┘
           │  │  │  │
           L  L  L  L        ← Local port
           │  │  │  │
         (0,3)─(0,2)─(0,1)─(0,0)    ← N/S 相連！
           │     │     │     │
           E     E     E     E      ← East to Column 1
```

### Edge Router N/S 連接必要性

使用 XY Routing:
1. Request 進入時 **src 座標設為入口 Edge Router** (例如 `(0,2)`)
2. Response 返回時先沿 X 軸移動到 Column 0
3. 到達 Column 0 後，需沿 Y 軸移動到 src 座標所在列
4. **若無 N/S 連接，Response 無法在 Column 0 內移動**

| 情境 | 無 N/S 連接 | 有 N/S 連接 |
|------|------------|------------|
| Response 到達錯誤列 | ❌ 卡住，無法轉發 | ✓ 可經 N/S 移動到正確 Router |
| Local port 檢查 | dest ≠ self → 拒絕 | 正常轉發到正確 Router |

---

## 3. 通道配置

| 層級 | Request 通道數 | Response 通道數 |
|------|----------------|-----------------|
| Per-NI | 1 | 1 |
| Per-Router | 1 (×4 Routers) | 1 (×4 Routers) |
| **總計** | 4 | 4 |

---

## 4. 參數

```python
class RoutingSelectorConfig:
    num_directions: int = 4        # 連接的 Router 方向數
    ingress_buffer: int = 8        # Ingress Buffer 深度
    egress_buffer: int = 8         # Egress Buffer 深度
    hop_weight: float = 1.0        # Hop 計數權重
    credit_weight: float = 1.0     # Credit 權重
```

---

## 5. EdgeRouterPort 結構

Selector 透過 **EdgeRouterPort** 連接每個 EdgeRouter，使用 PortWire 信號介面：

```
┌─────────────────────────────────────────────────────────────────────────┐
│                           RoutingSelector                                │
│  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐  ┌─────────────┐     │
│  │EdgeRouterPort│  │EdgeRouterPort│  │EdgeRouterPort│  │EdgeRouterPort│   │
│  │   row=0     │  │   row=1     │  │   row=2     │  │   row=3     │     │
│  │ _req_port   │  │ _req_port   │  │ _req_port   │  │ _req_port   │     │
│  │ _resp_port  │  │ _resp_port  │  │ _resp_port  │  │ _resp_port  │     │
│  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘  └──────┬──────┘     │
│      PortWire         PortWire         PortWire         PortWire        │
└─────────┼────────────────┼────────────────┼────────────────┼────────────┘
          ▼                ▼                ▼                ▼
    EdgeRouter(0,0)  EdgeRouter(0,1)  EdgeRouter(0,2)  EdgeRouter(0,3)
     req.LOCAL        req.LOCAL        req.LOCAL        req.LOCAL
```

### EdgeRouterPort 類別

```python
class EdgeRouterPort:
    row: int                          # 連接的 EdgeRouter row
    coord: Tuple[int, int]            # = (0, row)

    # 信號介面 (使用 RouterPort)
    _req_port: RouterPort             # Request 發送端
    _resp_port: RouterPort            # Response 接收端

    # PortWire 連接 (connect_edge_routers 時建立)
    _req_wire: PortWire               # → EdgeRouter.req.LOCAL
    _resp_wire: PortWire              # ← EdgeRouter.resp.LOCAL

    # Credit 管理
    @property
    def available_credits(self) -> int:
        return self._req_port._output_credit.credits

```
### 信號流向

```
Request Path (Selector → EdgeRouter):
┌─────────────────┐                      ┌─────────────────┐
│ EdgeRouterPort  │      PortWire        │   EdgeRouter    │
│   _req_port     │                      │ req.LOCAL port  │
│  ┌───────────┐  │                      │  ┌───────────┐  │
│  │ out_valid ├──┼──────────────────────┼─►│ in_valid  │  │
│  │ out_flit  ├──┼──────────────────────┼─►│ in_flit   │  │
│  │ in_ready  │◄─┼──────────────────────┼──┤ out_ready │  │
│  └───────────┘  │                      │  └───────────┘  │
└─────────────────┘                      └─────────────────┘

Response Path (EdgeRouter → Selector):
┌─────────────────┐                      ┌─────────────────┐
│   EdgeRouter    │      PortWire        │ EdgeRouterPort  │
│ resp.LOCAL port │                      │   _resp_port    │
│  ┌───────────┐  │                      │  ┌───────────┐  │
│  │ out_valid ├──┼──────────────────────┼─►│ in_valid  │  │
│  │ out_flit  ├──┼──────────────────────┼─►│ in_flit   │  │
│  │ in_ready  │◄─┼──────────────────────┼──┤ out_ready │  │
│  └───────────┘  │                      │  └───────────┘  │
└─────────────────┘                      └─────────────────┘
```

---

## 6. Credit 管理

Selector 對每個 EdgeRouter 維護獨立的 credit：

| 項目 | 說明 |
|------|------|
| Credit 來源 | EdgeRouter.req.LOCAL buffer depth |
| Credit 儲存 | `EdgeRouterPort._req_port._output_credit.credits` |
| 消耗時機 | `clear_req_if_accepted()` - handshake 完成時 |
| 釋放時機 | `handle_credit_release()` → `PortWire.propagate_credit_release()` |
| 選路影響 | Credit 高的路徑優先選擇 (避免 backpressure) |

---

## 7. 路徑選擇演算法

### 7.1 Ingress Path (進入 NoC)

```python
def select_ingress_path(packet, available_ports):
    """Equivalence function: 選擇最低成本路徑。"""
    best_port = None
    min_cost = float('inf')

    for port in available_ports:
        hop = calculate_hops(port, packet.dest)
        credit = get_downstream_credit(port)

        # Equivalence 公式
        cost = hop_weight * hop - credit_weight * credit

        if cost < min_cost:
            min_cost = cost
            best_port = port

    return best_port
```

### 7.2 Egress Path (離開 NoC)

```python
def select_egress_read():
    """依 Edge Router Buffer 佔用率決定讀取優先序。"""
    # 優先讀取佔用率最高的 Buffer，防止上游阻塞
    # 檢查 (0,0)~(0,3) Routers 的 Buffer 狀態
    return max(edge_routers, key=lambda r: r.buffer_occupancy)
```

---

## 8. Phased Cycle Processing

Selector 使用多階段週期處理，與 Mesh 協調：

| Phase | 方法 | 說明 |
|-------|------|------|
| 1 | `update_all_ready()` | 更新 response port ready 信號 |
| 2 | `propagate_all_wires()` | 傳播 PortWire 信號 |
| 3 | `sample_all_inputs()` | 取樣 response inputs |
| 4 | `clear_all_input_signals()` | 清除 input 信號 |
| 5 | `_process_ingress()` | 選擇路徑，設定 request outputs |
| 6 | `_process_egress()` | 移動 responses 到 egress buffer |
| 7 | `clear_accepted_outputs()` | 清除已被接受的 outputs |
| 8 | `handle_credit_release()` | 處理 credit 釋放 |

### V1System 整體協調

```python
def process_cycle(self):
    # 1. MasterNI → Selector.ingress_buffer
    while master_ni.has_pending_output():
        flit = master_ni.get_req_flit()
        selector.accept_request(flit)

    # 2. Selector 設定 request outputs 並 propagate
    selector.update_all_ready()
    selector.propagate_all_wires()

    # 3. Mesh samples (取樣 Selector 的 outputs)
    mesh.process_cycle()

    # 4. 清除已接受的 outputs (handshake 完成)
    selector.clear_accepted_outputs()
    selector.handle_credit_release()

    # 5. Propagate 取得 Mesh 的 response outputs
    selector.propagate_all_wires()

    # 6. Sample responses, process ingress/egress
    selector.sample_all_inputs()
    selector.clear_all_input_signals()
    selector._process_ingress()
    selector._process_egress()

    # 7. Propagate 新的 request outputs
    selector.propagate_all_wires()

    # 8. Responses → MasterNI
    while selector.has_pending_responses:
        flit = selector.get_response()
        master_ni.receive_resp_flit(flit)
```

---

## 相關文件

- [系統概述](00_overview.md)
- [內部介面架構](11_internal_interface.md) - Selector ↔ EdgeRouter 連接
- [V2 Smart Crossbar](10_v2_smart_crossbar.md) - V2 架構改用 Smart Crossbar
