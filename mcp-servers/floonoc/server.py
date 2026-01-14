#!/usr/bin/env python3
"""
FlooNoC Reference MCP Server.

A lightweight MCP server providing FlooNoC design specifications and
documentation for reference when working with NoC Behavior Model.

This is a "knowledge-based" MCP that provides specification queries,
not simulation execution.
"""

import json
from enum import Enum
from typing import Any, Dict, List, Optional

from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, ConfigDict, Field

# Initialize MCP server
mcp = FastMCP("floonoc_mcp")

# ============================================================================
# FlooNoC Specifications (Based on NoC Behavior Model Implementation)
# ============================================================================

FLIT_HEADER_SPEC = {
    "total_bits": 20,
    "fields": [
        {"name": "rob_req", "bits": "0", "width": 1, "description": "RoB request flag"},
        {"name": "rob_idx", "bits": "5:1", "width": 5, "description": "RoB index (32 entries max)"},
        {"name": "dst_id", "bits": "10:6", "width": 5, "description": "Destination node {x[2:0], y[1:0]}"},
        {"name": "src_id", "bits": "15:11", "width": 5, "description": "Source node {x[2:0], y[1:0]}"},
        {"name": "last", "bits": "16", "width": 1, "description": "Last flit of packet"},
        {"name": "axi_ch", "bits": "19:17", "width": 3, "description": "AXI channel type (0-4)"},
    ]
}

AXI_CHANNELS = {
    "AW": {
        "code": 0,
        "name": "Write Address",
        "type": "Request",
        "payload_bits": 53,
        "fields": [
            {"name": "addr", "width": 32, "description": "Write address"},
            {"name": "id", "width": 8, "description": "Transaction ID"},
            {"name": "len", "width": 8, "description": "Burst length (0-255)"},
            {"name": "size", "width": 3, "description": "Burst size (2^size bytes)"},
            {"name": "burst", "width": 2, "description": "Burst type (FIXED/INCR/WRAP)"},
        ]
    },
    "W": {
        "code": 1,
        "name": "Write Data",
        "type": "Request",
        "payload_bits": 288,
        "fields": [
            {"name": "data", "width": 256, "description": "Write data (32 bytes)"},
            {"name": "strb", "width": 32, "description": "Byte strobe mask"},
        ]
    },
    "AR": {
        "code": 2,
        "name": "Read Address",
        "type": "Request",
        "payload_bits": 53,
        "fields": [
            {"name": "addr", "width": 32, "description": "Read address"},
            {"name": "id", "width": 8, "description": "Transaction ID"},
            {"name": "len", "width": 8, "description": "Burst length (0-255)"},
            {"name": "size", "width": 3, "description": "Burst size (2^size bytes)"},
            {"name": "burst", "width": 2, "description": "Burst type (FIXED/INCR/WRAP)"},
        ]
    },
    "B": {
        "code": 3,
        "name": "Write Response",
        "type": "Response",
        "payload_bits": 10,
        "fields": [
            {"name": "id", "width": 8, "description": "Transaction ID"},
            {"name": "resp", "width": 2, "description": "Response status (OKAY/EXOKAY/SLVERR/DECERR)"},
        ]
    },
    "R": {
        "code": 4,
        "name": "Read Data",
        "type": "Response",
        "payload_bits": 266,
        "fields": [
            {"name": "data", "width": 256, "description": "Read data (32 bytes)"},
            {"name": "id", "width": 8, "description": "Transaction ID"},
            {"name": "resp", "width": 2, "description": "Response status"},
        ]
    },
}

PHYSICAL_LINKS = {
    "request": {
        "total_bits": 310,
        "breakdown": {
            "valid": 1,
            "ready": 1,
            "header": 20,
            "payload": 288,  # Max: W channel
        },
        "description": "Request network physical link (AW, W, AR channels)"
    },
    "response": {
        "total_bits": 288,
        "breakdown": {
            "valid": 1,
            "ready": 1,
            "header": 20,
            "payload": 266,  # Max: R channel
        },
        "description": "Response network physical link (B, R channels)"
    }
}

PACKET_TYPES = {
    "WRITE_REQ": {
        "description": "AXI Write Request",
        "flits": ["AW", "W*"],
        "note": "One AW flit followed by one or more W flits"
    },
    "WRITE_RESP": {
        "description": "AXI Write Response",
        "flits": ["B"],
        "note": "Single B flit"
    },
    "READ_REQ": {
        "description": "AXI Read Request",
        "flits": ["AR"],
        "note": "Single AR flit"
    },
    "READ_RESP": {
        "description": "AXI Read Response",
        "flits": ["R*"],
        "note": "One or more R flits based on burst length"
    }
}

NODE_ID_ENCODING = {
    "total_bits": 5,
    "format": "{x[2:0], y[1:0]}",
    "x_bits": 3,
    "y_bits": 2,
    "max_x": 7,
    "max_y": 3,
    "max_nodes": 32,
    "example": {
        "coord": "(3, 2)",
        "binary": "01110",
        "decimal": 14,
        "calculation": "(3 << 2) | 2 = 12 + 2 = 14"
    }
}

DESIGN_COMPARISON = {
    "NoC Behavior Model": {
        "language": "Python",
        "purpose": "Cycle-accurate behavioral simulation",
        "topology": "5x4 2D Mesh (non-square)",
        "routing": "XY Routing (dimension-order)",
        "flow_control": "Wormhole + Credit-based",
        "flit_format": "FlooNoC-style 20-bit header",
        "data_width": "256 bits (32 bytes)",
        "protocol": "AXI4 with full burst support"
    },
    "FlooNoC (RTL)": {
        "language": "SystemVerilog",
        "purpose": "Synthesizable hardware implementation",
        "topology": "Configurable (mesh, custom)",
        "routing": "XY Routing, ID-Table routing",
        "flow_control": "Credit-based with virtual channels",
        "flit_format": "Configurable header + payload",
        "data_width": "Configurable (64-512 bits)",
        "protocol": "AXI4 + AXI5 ATOPs"
    }
}

FLOONOC_GITHUB = "https://github.com/pulp-platform/FlooNoC"

# ============================================================================
# Enums
# ============================================================================

class ResponseFormat(str, Enum):
    """Output format for tool responses."""
    MARKDOWN = "markdown"
    JSON = "json"


class SpecCategory(str, Enum):
    """Specification category."""
    FLIT = "flit"
    AXI = "axi"
    PACKET = "packet"
    LINK = "link"
    NODE = "node"
    ALL = "all"


# ============================================================================
# Pydantic Models
# ============================================================================

class GetSpecInput(BaseModel):
    """Input for getting FlooNoC specifications."""
    model_config = ConfigDict(str_strip_whitespace=True)

    category: SpecCategory = Field(
        default=SpecCategory.ALL,
        description="Specification category (flit, axi, packet, link, node, all)"
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format"
    )


class GetAxiChannelInput(BaseModel):
    """Input for getting AXI channel details."""
    model_config = ConfigDict(str_strip_whitespace=True)

    channel: Optional[str] = Field(
        default=None,
        description="AXI channel name (AW, W, AR, B, R). None for all channels."
    )
    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format"
    )


class EncodeNodeIdInput(BaseModel):
    """Input for encoding node coordinates."""
    model_config = ConfigDict(str_strip_whitespace=True)

    x: int = Field(..., description="X coordinate", ge=0, le=7)
    y: int = Field(..., description="Y coordinate", ge=0, le=3)


class DecodeNodeIdInput(BaseModel):
    """Input for decoding node ID."""
    model_config = ConfigDict(str_strip_whitespace=True)

    node_id: int = Field(..., description="5-bit node ID", ge=0, le=31)


class CompareDesignInput(BaseModel):
    """Input for comparing design specifications."""
    model_config = ConfigDict(str_strip_whitespace=True)

    response_format: ResponseFormat = Field(
        default=ResponseFormat.MARKDOWN,
        description="Output format"
    )


# ============================================================================
# MCP Tools
# ============================================================================

@mcp.tool(
    name="floonoc_get_flit_format",
    annotations={
        "title": "Get FlooNoC Flit Format",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def floonoc_get_flit_format(response_format: ResponseFormat = ResponseFormat.MARKDOWN) -> str:
    """Get FlooNoC flit header format specification.

    Returns the 20-bit flit header format used in FlooNoC-style NoC designs,
    including field positions, widths, and descriptions.

    Args:
        response_format: Output format (markdown or json)

    Returns:
        str: Flit format specification
    """
    if response_format == ResponseFormat.JSON:
        return json.dumps(FLIT_HEADER_SPEC, indent=2)

    lines = [
        "# FlooNoC Flit Header Format (20 bits)",
        "",
        "```",
        "[19:17] axi_ch   - AXI channel type (3 bits)",
        "[16]    last     - Last flit of packet (1 bit)",
        "[15:11] src_id   - Source node ID (5 bits)",
        "[10:6]  dst_id   - Destination node ID (5 bits)",
        "[5:1]   rob_idx  - RoB index (5 bits)",
        "[0]     rob_req  - RoB request flag (1 bit)",
        "```",
        "",
        "## Field Details",
        "",
        "| Field | Bits | Width | Description |",
        "|-------|------|-------|-------------|",
    ]

    for field in FLIT_HEADER_SPEC["fields"]:
        lines.append(
            f"| `{field['name']}` | {field['bits']} | {field['width']} | {field['description']} |"
        )

    lines.extend([
        "",
        "## Node ID Encoding",
        f"- Format: `{{x[2:0], y[1:0]}}` = 5 bits",
        f"- Max X: 7, Max Y: 3 (32 nodes max)",
    ])

    return "\n".join(lines)


@mcp.tool(
    name="floonoc_get_axi_channels",
    annotations={
        "title": "Get AXI Channel Specifications",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def floonoc_get_axi_channels(params: GetAxiChannelInput) -> str:
    """Get AXI channel specifications used in FlooNoC.

    Returns detailed information about AXI channels including payload
    format, field widths, and descriptions.

    Args:
        params: Channel filter and format options

    Returns:
        str: AXI channel specifications
    """
    if params.channel:
        ch = params.channel.upper()
        if ch not in AXI_CHANNELS:
            return f"Unknown channel: {params.channel}. Valid: AW, W, AR, B, R"
        channels = {ch: AXI_CHANNELS[ch]}
    else:
        channels = AXI_CHANNELS

    if params.response_format == ResponseFormat.JSON:
        return json.dumps(channels, indent=2)

    lines = ["# FlooNoC AXI Channel Specifications", ""]

    for ch_name, ch_info in channels.items():
        lines.extend([
            f"## {ch_name} - {ch_info['name']} ({ch_info['type']})",
            "",
            f"**Payload Size**: {ch_info['payload_bits']} bits",
            "",
            "| Field | Width | Description |",
            "|-------|-------|-------------|",
        ])

        for field in ch_info["fields"]:
            lines.append(f"| `{field['name']}` | {field['width']} | {field['description']} |")

        lines.append("")

    # Add summary table
    lines.extend([
        "## Summary",
        "",
        "| Channel | Code | Type | Payload Bits |",
        "|---------|------|------|--------------|",
    ])
    for ch_name, ch_info in AXI_CHANNELS.items():
        lines.append(f"| {ch_name} | {ch_info['code']} | {ch_info['type']} | {ch_info['payload_bits']} |")

    return "\n".join(lines)


@mcp.tool(
    name="floonoc_get_packet_types",
    annotations={
        "title": "Get Packet Type Specifications",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def floonoc_get_packet_types(response_format: ResponseFormat = ResponseFormat.MARKDOWN) -> str:
    """Get FlooNoC packet type specifications.

    Returns information about how AXI transactions are encapsulated
    into NoC packets and flits.

    Args:
        response_format: Output format (markdown or json)

    Returns:
        str: Packet type specifications
    """
    if response_format == ResponseFormat.JSON:
        return json.dumps(PACKET_TYPES, indent=2)

    lines = [
        "# FlooNoC Packet Types",
        "",
        "| Type | Description | Flit Sequence | Note |",
        "|------|-------------|---------------|------|",
    ]

    for ptype, info in PACKET_TYPES.items():
        flits = " → ".join(info["flits"])
        lines.append(f"| `{ptype}` | {info['description']} | {flits} | {info['note']} |")

    lines.extend([
        "",
        "## Packet Assembly Rules",
        "",
        "1. **Write Request**: AW flit sets `last=0`, final W flit sets `last=1`",
        "2. **Read Request**: Single AR flit with `last=1`",
        "3. **Write Response**: Single B flit with `last=1`",
        "4. **Read Response**: Multiple R flits, final R sets `last=1`",
        "",
        "## Packet Identification",
        "",
        "Packets are identified by the tuple: `(src_id, dst_id, rob_idx)`",
    ])

    return "\n".join(lines)


@mcp.tool(
    name="floonoc_get_physical_links",
    annotations={
        "title": "Get Physical Link Specifications",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def floonoc_get_physical_links(response_format: ResponseFormat = ResponseFormat.MARKDOWN) -> str:
    """Get FlooNoC physical link specifications.

    Returns information about the request and response network
    physical link widths and signal breakdown.

    Args:
        response_format: Output format (markdown or json)

    Returns:
        str: Physical link specifications
    """
    if response_format == ResponseFormat.JSON:
        return json.dumps(PHYSICAL_LINKS, indent=2)

    lines = [
        "# FlooNoC Physical Links",
        "",
        "FlooNoC uses separate physical networks for requests and responses.",
        "",
        "## Request Network (310 bits)",
        "",
        "| Signal | Width | Description |",
        "|--------|-------|-------------|",
        "| valid | 1 | Flit valid signal |",
        "| ready | 1 | Credit/ready signal |",
        "| header | 20 | Flit header |",
        "| payload | 288 | Max payload (W channel) |",
        "",
        "Carries: AW, W, AR channels",
        "",
        "## Response Network (288 bits)",
        "",
        "| Signal | Width | Description |",
        "|--------|-------|-------------|",
        "| valid | 1 | Flit valid signal |",
        "| ready | 1 | Credit/ready signal |",
        "| header | 20 | Flit header |",
        "| payload | 266 | Max payload (R channel) |",
        "",
        "Carries: B, R channels",
    ]

    return "\n".join(lines)


@mcp.tool(
    name="floonoc_encode_node_id",
    annotations={
        "title": "Encode Node Coordinates to ID",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def floonoc_encode_node_id(params: EncodeNodeIdInput) -> str:
    """Encode (x, y) coordinates to FlooNoC 5-bit node ID.

    Args:
        params: X and Y coordinates

    Returns:
        str: Encoded node ID with calculation
    """
    node_id = ((params.x & 0x7) << 2) | (params.y & 0x3)

    return (
        f"## Node ID Encoding\n\n"
        f"- Coordinates: ({params.x}, {params.y})\n"
        f"- Formula: `(x << 2) | y`\n"
        f"- Calculation: `({params.x} << 2) | {params.y}` = `{params.x << 2} | {params.y}` = **{node_id}**\n"
        f"- Binary: `{node_id:05b}`\n"
        f"- Format: `{{x[2:0]={params.x:03b}, y[1:0]={params.y:02b}}}`"
    )


@mcp.tool(
    name="floonoc_decode_node_id",
    annotations={
        "title": "Decode Node ID to Coordinates",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def floonoc_decode_node_id(params: DecodeNodeIdInput) -> str:
    """Decode FlooNoC 5-bit node ID to (x, y) coordinates.

    Args:
        params: 5-bit node ID

    Returns:
        str: Decoded coordinates with calculation
    """
    x = (params.node_id >> 2) & 0x7
    y = params.node_id & 0x3

    return (
        f"## Node ID Decoding\n\n"
        f"- Node ID: {params.node_id} (binary: `{params.node_id:05b}`)\n"
        f"- X: `(node_id >> 2) & 0x7` = `({params.node_id} >> 2) & 7` = **{x}**\n"
        f"- Y: `node_id & 0x3` = `{params.node_id} & 3` = **{y}**\n"
        f"- Coordinates: **({x}, {y})**"
    )


@mcp.tool(
    name="floonoc_compare_design",
    annotations={
        "title": "Compare NoC Designs",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def floonoc_compare_design(params: CompareDesignInput) -> str:
    """Compare NoC Behavior Model with FlooNoC RTL design.

    Returns a detailed comparison table showing differences and
    similarities between the two implementations.

    Args:
        params: Format options

    Returns:
        str: Design comparison
    """
    if params.response_format == ResponseFormat.JSON:
        return json.dumps(DESIGN_COMPARISON, indent=2)

    noc = DESIGN_COMPARISON["NoC Behavior Model"]
    floo = DESIGN_COMPARISON["FlooNoC (RTL)"]

    lines = [
        "# NoC Behavior Model vs FlooNoC Comparison",
        "",
        "| Aspect | NoC Behavior Model | FlooNoC (RTL) |",
        "|--------|-------------------|---------------|",
    ]

    for key in noc.keys():
        lines.append(f"| {key.replace('_', ' ').title()} | {noc[key]} | {floo[key]} |")

    lines.extend([
        "",
        "## Key Differences",
        "",
        "1. **Purpose**: Behavioral model for performance analysis vs synthesizable RTL",
        "2. **Topology**: Fixed 5x4 mesh vs configurable arbitrary topology",
        "3. **Data Width**: Fixed 256-bit vs configurable",
        "4. **Flexibility**: Python-based rapid iteration vs hardware constraints",
        "",
        "## Compatibility",
        "",
        "- Both use compatible flit header format (20-bit)",
        "- Both support AXI4 protocol with burst transactions",
        "- Both use XY routing (dimension-order) by default",
        "- Both use credit-based flow control",
        "",
        f"**FlooNoC GitHub**: {FLOONOC_GITHUB}",
    ])

    return "\n".join(lines)


@mcp.tool(
    name="floonoc_get_all_specs",
    annotations={
        "title": "Get All Specifications",
        "readOnlyHint": True,
        "destructiveHint": False,
        "idempotentHint": True,
        "openWorldHint": False
    }
)
async def floonoc_get_all_specs(response_format: ResponseFormat = ResponseFormat.MARKDOWN) -> str:
    """Get complete FlooNoC specification reference.

    Returns all specifications in a single comprehensive document.

    Args:
        response_format: Output format (markdown or json)

    Returns:
        str: Complete specifications
    """
    if response_format == ResponseFormat.JSON:
        return json.dumps({
            "flit_header": FLIT_HEADER_SPEC,
            "axi_channels": AXI_CHANNELS,
            "physical_links": PHYSICAL_LINKS,
            "packet_types": PACKET_TYPES,
            "node_id_encoding": NODE_ID_ENCODING,
            "design_comparison": DESIGN_COMPARISON,
            "github": FLOONOC_GITHUB,
        }, indent=2)

    # Combine all specs into one document
    sections = [
        await floonoc_get_flit_format(ResponseFormat.MARKDOWN),
        "",
        "---",
        "",
        await floonoc_get_axi_channels(GetAxiChannelInput(response_format=ResponseFormat.MARKDOWN)),
        "",
        "---",
        "",
        await floonoc_get_packet_types(ResponseFormat.MARKDOWN),
        "",
        "---",
        "",
        await floonoc_get_physical_links(ResponseFormat.MARKDOWN),
        "",
        "---",
        "",
        await floonoc_compare_design(CompareDesignInput(response_format=ResponseFormat.MARKDOWN)),
    ]

    return "\n".join(sections)


# ============================================================================
# Main Entry Point
# ============================================================================

if __name__ == "__main__":
    mcp.run()
