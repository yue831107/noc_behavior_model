# Flit 格式

NoC 內的基本傳輸單元。

---

## 1. Flit 類型

| 類型 | 說明 |
|------|------|
| HEAD | 封包標頭，包含路由資訊 |
| BODY | 封包資料 |
| TAIL | 封包結尾 |
| HEAD_TAIL | 單 Flit 封包 |

---

## 2. Flit 結構

```python
class Flit:
    flit_type: FlitType      # HEAD, BODY, TAIL, HEAD_TAIL
    src: tuple[int, int]     # 來源座標 (x, y)
    dest: tuple[int, int]    # 目的座標 (x, y)
    vc_id: int               # Virtual Channel ID
    payload: bytes           # 資料內容
    seq_num: int             # 序號 (供 Reordering)
    is_request: bool         # True=Request, False=Response
```

---

## 3. Flit 大小限制

| 參數 | 值 |
|------|-----|
| Flit Payload 大小 | 32 bytes |
| Packet Header | 12 bytes |
| **單 Flit 最大資料量** | **20 bytes** |

當 `block_size > 20 bytes` 時，需多個 Flits 組成封包。使用 Wormhole Routing 時，HEAD Flit 不等待完整封包即可轉發。

**建議**: 使用 `block_size ≤ 20` 以維持 Single-Flit 傳輸。

---

## 4. V2 擴展欄位

```python
class FlitV2(Flit):
    # 繼承 V1 欄位
    # ...

    # V2 新增欄位
    src_ni_id: int         # 來源 NI ID (0-3)，供 Response 路由
    entry_edge_router: int # 入口 Edge Router (0-3)
```

---

## 相關文件

- [Router 規格](01_router.md)
- [Network Interface 規格](02_network_interface.md)
- [設計決策](08_design_decisions.md) - Wormhole Switching
