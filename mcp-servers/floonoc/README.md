# FlooNoC Reference MCP Server

A lightweight MCP server providing [FlooNoC](https://github.com/pulp-platform/FlooNoC) design specifications for reference when working with NoC Behavior Model.

This is a **knowledge-based** MCP that provides specification queries, not simulation execution.

## Features

- Query FlooNoC flit header format
- Get AXI channel specifications
- Encode/decode node IDs
- Compare NoC Behavior Model with FlooNoC RTL

## MCP Tools

| Tool | Description |
|------|-------------|
| `floonoc_get_flit_format` | Get 20-bit flit header format |
| `floonoc_get_axi_channels` | Get AXI channel specifications (AW, W, AR, B, R) |
| `floonoc_get_packet_types` | Get packet type specifications |
| `floonoc_get_physical_links` | Get physical link widths |
| `floonoc_encode_node_id` | Encode (x, y) coordinates to 5-bit node ID |
| `floonoc_decode_node_id` | Decode node ID to coordinates |
| `floonoc_compare_design` | Compare with FlooNoC RTL design |
| `floonoc_get_all_specs` | Get complete specifications |

## Installation

```bash
pip install -r requirements.txt
```

## Usage in Claude Code

```
# Query flit format
"What is the FlooNoC flit header format?"

# Get AXI channel details
"Show me the AW channel specification"

# Encode/decode node IDs
"Encode coordinates (3, 2) to node ID"
"Decode node ID 14"

# Compare designs
"Compare NoC Behavior Model with FlooNoC"
```

## Specifications Overview

### Flit Header (20 bits)
```
[19:17] axi_ch   - AXI channel type
[16]    last     - Last flit of packet
[15:11] src_id   - Source node ID
[10:6]  dst_id   - Destination node ID
[5:1]   rob_idx  - RoB index
[0]     rob_req  - RoB request flag
```

### AXI Channels
| Channel | Type | Payload |
|---------|------|---------|
| AW | Request | 53 bits |
| W | Request | 288 bits |
| AR | Request | 53 bits |
| B | Response | 10 bits |
| R | Response | 266 bits |

### Physical Links
- Request Network: 310 bits
- Response Network: 288 bits

## Reference

- [FlooNoC GitHub](https://github.com/pulp-platform/FlooNoC)
- [PULP Platform](https://pulp-platform.org/)
