#!/usr/bin/env python3
"""
Transfer Config Generator.

Generates random multi-transfer configurations for V1System testing.

Usage:
    python gen_transfer_config.py -n 10            # Generate 10 random transfers
    python gen_transfer_config.py -n 5 --seed 123  # Reproducible generation
    python gen_transfer_config.py -n 20 --max-size 8192 -o custom.yaml
"""

import argparse
import random
import yaml
from pathlib import Path
from dataclasses import dataclass, asdict
from typing import List, Union
from datetime import datetime


@dataclass
class TransferSpec:
    """Single transfer specification."""
    src_addr: int
    src_size: int
    dst_addr: int
    target_nodes: Union[str, List[int]]
    transfer_mode: str
    max_burst_len: int = 16
    max_outstanding: int = 8
    read_src_addr: int = 0x1000  # For read operations, address in node memory


def generate_random_transfers(
    count: int,
    num_nodes: int = 16,
    min_size: int = 256,
    max_size: int = 4096,
    mode: str = "random",
    seed: int = 42,
) -> List[TransferSpec]:
    """
    Generate random transfer configurations.
    
    Args:
        count: Number of transfers to generate.
        num_nodes: Total nodes in the system.
        min_size: Minimum transfer size in bytes.
        max_size: Maximum transfer size in bytes.
        mode: Transfer mode - "random", "broadcast", "scatter", or "mixed".
        seed: Random seed for reproducibility.
    
    Returns:
        List of TransferSpec objects.
    """
    random.seed(seed)
    transfers = []
    
    # Address space limits (cycle within these ranges)
    MAX_SRC_ADDR = 0x100000  # 1MB host memory address space
    MAX_DST_ADDR = 0x100000  # 1MB local memory address space (per node)
    
    # Track address usage
    next_src_addr = 0x0000
    next_dst_addr = 0x1000
    
    write_modes = ["broadcast", "scatter"]
    read_modes = ["gather"]
    all_modes = write_modes + read_modes
    
    for i in range(count):
        # Generate size (aligned to 256 bytes)
        size = random.randint(min_size // 256, max_size // 256) * 256
        
        # Generate addresses with cycling to prevent overflow
        src_addr = next_src_addr % MAX_SRC_ADDR
        dst_addr = next_dst_addr % MAX_DST_ADDR
        
        # Ensure addresses are aligned
        src_addr = (src_addr // 256) * 256
        dst_addr = (dst_addr // 256) * 256
        
        next_src_addr += size
        next_dst_addr += size
        
        # Generate target nodes
        if mode == "random" or mode == "mixed":
            # Random subset of nodes or all
            if random.random() < 0.3:
                target_nodes = "all"
            else:
                num_targets = random.randint(1, min(8, num_nodes))
                target_nodes = sorted(random.sample(range(num_nodes), num_targets))
        else:
            target_nodes = "all"
        
        # Generate transfer mode
        if mode == "mixed" or mode == "random":
            transfer_mode = random.choice(all_modes)
        elif mode == "gather":
            transfer_mode = "gather"
        else:
            transfer_mode = mode
        
        # Generate outstanding (higher for larger transfers)
        max_outstanding = min(16, max(4, size // 512))
        
        # For read (gather) mode, swap src/dst semantics
        if transfer_mode == "gather":
            # Read: src_addr is in node memory, dst_addr is in host memory
            spec = TransferSpec(
                src_addr=dst_addr,       # Read to host memory here
                src_size=size,
                dst_addr=src_addr,       # This is unused for reads
                read_src_addr=dst_addr,  # Read from node memory here
                target_nodes=target_nodes,
                transfer_mode=transfer_mode,
                max_burst_len=16,
                max_outstanding=max_outstanding,
            )
        else:
            # Write: src_addr is in host memory, dst_addr is in node memory
            spec = TransferSpec(
                src_addr=src_addr,
                src_size=size,
                dst_addr=dst_addr,
                target_nodes=target_nodes,
                transfer_mode=transfer_mode,
                max_burst_len=16,
                max_outstanding=max_outstanding,
            )
        transfers.append(spec)
    
    return transfers


def to_yaml(
    transfers: List[TransferSpec],
    seed: int,
    output_path: Path,
    sweep_params: dict = None,
) -> None:
    """
    Write transfers to YAML file.

    Args:
        transfers: List of TransferSpec objects.
        seed: Seed used for generation.
        output_path: Output file path.
        sweep_params: Optional sweep parameters for multi_para execution.
    """
    # Convert to dicts with hex addresses
    transfer_dicts = []
    for t in transfers:
        d = {
            "src_addr": f"0x{t.src_addr:04X}",
            "src_size": t.src_size,
            "dst_addr": f"0x{t.dst_addr:04X}",
            "target_nodes": t.target_nodes,
            "transfer_mode": t.transfer_mode,
            "max_burst_len": t.max_burst_len,
            "max_outstanding": t.max_outstanding,
        }
        # Add read_src_addr for gather mode
        if t.transfer_mode == "gather":
            d["read_src_addr"] = f"0x{t.read_src_addr:04X}"
        transfer_dicts.append(d)

    data = {"transfers": transfer_dicts}

    # Add sweep parameters if provided
    if sweep_params:
        data["sweep"] = sweep_params

    # Add header comment
    header = f"""# Auto-generated Transfer Configuration
# Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
# Seed: {seed}
# Count: {len(transfers)}
# Total data: {sum(t.src_size for t in transfers)} bytes

"""

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(header)
        yaml.dump(data, f, default_flow_style=False, sort_keys=False)

    print(f"Generated {len(transfers)} transfer configs")
    print(f"  Output: {output_path}")
    print(f"  Seed: {seed}")
    print(f"  Total data: {sum(t.src_size for t in transfers)} bytes")
    if sweep_params:
        print(f"  Sweep params: {list(sweep_params.keys())}")


def parse_sweep_param(value: str) -> List:
    """Parse sweep parameter value like '2,4,8,16' into list."""
    return [int(x) for x in value.split(',')]


def main():
    parser = argparse.ArgumentParser(
        description="Generate random multi-transfer configurations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python gen_transfer_config.py -n 10
  python gen_transfer_config.py -n 5 --seed 123 --mode broadcast
  python gen_transfer_config.py -n 20 --min-size 512 --max-size 8192

Multi-parameter sweep:
  python gen_transfer_config.py -n 10 --sweep-buffer-depth 2,4,8,16
  python gen_transfer_config.py -n 10 --sweep-buffer-depth 2,4,8 --sweep-mesh-cols 3,4,5
"""
    )

    parser.add_argument(
        '-n', '--num',
        type=int,
        default=5,
        help='Number of transfers to generate (default: 5)'
    )
    parser.add_argument(
        '--min-size',
        type=int,
        default=256,
        help='Minimum transfer size in bytes (default: 256)'
    )
    parser.add_argument(
        '--max-size',
        type=int,
        default=4096,
        help='Maximum transfer size in bytes (default: 4096)'
    )
    parser.add_argument(
        '--nodes',
        type=int,
        default=16,
        help='Number of nodes in system (default: 16)'
    )
    parser.add_argument(
        '--mode',
        choices=['random', 'broadcast', 'scatter', 'gather', 'mixed'],
        default='random',
        help='Transfer mode: random includes gather (default: random)'
    )
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    parser.add_argument(
        '-o', '--output',
        type=str,
        default='examples/Host_to_NoC/config/generated.yaml',
        help='Output YAML file path'
    )

    # Sweep parameters
    parser.add_argument(
        '--sweep-buffer-depth',
        type=str,
        default=None,
        help='Buffer depth values to sweep, comma-separated (e.g., 2,4,8,16)'
    )
    parser.add_argument(
        '--sweep-mesh-cols',
        type=str,
        default=None,
        help='Mesh columns to sweep, comma-separated (e.g., 3,4,5)'
    )
    parser.add_argument(
        '--sweep-mesh-rows',
        type=str,
        default=None,
        help='Mesh rows to sweep, comma-separated (e.g., 3,4,5)'
    )

    args = parser.parse_args()

    # Generate transfers
    transfers = generate_random_transfers(
        count=args.num,
        num_nodes=args.nodes,
        min_size=args.min_size,
        max_size=args.max_size,
        mode=args.mode,
        seed=args.seed,
    )

    # Build sweep parameters
    sweep_params = {}
    if args.sweep_buffer_depth:
        sweep_params['buffer_depth'] = parse_sweep_param(args.sweep_buffer_depth)
    if args.sweep_mesh_cols:
        sweep_params['mesh_cols'] = parse_sweep_param(args.sweep_mesh_cols)
    if args.sweep_mesh_rows:
        sweep_params['mesh_rows'] = parse_sweep_param(args.sweep_mesh_rows)

    # Write to YAML
    output_path = Path(args.output)
    to_yaml(
        transfers,
        args.seed,
        output_path,
        sweep_params=sweep_params if sweep_params else None,
    )

    # Print summary
    print()
    print("Transfer summary:")
    for i, t in enumerate(transfers):
        target_str = t.target_nodes if t.target_nodes == "all" else f"{len(t.target_nodes)} nodes"
        print(f"  [{i+1:2d}] {t.src_size:5d}B -> {target_str} ({t.transfer_mode})")


if __name__ == '__main__':
    main()
