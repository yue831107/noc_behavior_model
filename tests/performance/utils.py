"""
Utility functions for performance validation tests.

Provides helper functions to extract metrics from simulation systems
using Monitor-based approach.
"""

from typing import Dict, Any, Tuple
from pathlib import Path


def extract_metrics_from_simulation(system: Any) -> Dict[str, Any]:
    """
    Extract performance metrics from simulation system.
    
    Monitor-based: uses existing MetricsProvider interface.
    Does not modify core implementation.
    
    Args:
        system: Simulation system (V1System or NoCSystem)
    
    Returns:
        Dictionary containing performance metrics
    """
    # Import here to avoid circular dependency
    from src.core.metrics_provider import get_metrics_from_system
    
    # Get raw metrics from system
    raw_metrics = get_metrics_from_system(system)
    
    # Calculate derived metrics
    cycle = raw_metrics['cycle']
    bytes_transferred = raw_metrics['bytes_transferred']
    completed = raw_metrics['completed_transactions']
    
    # Throughput (bytes/cycle)
    throughput = bytes_transferred / cycle if cycle > 0 else 0.0
    
    # Buffer utilization (average across all routers)
    buffer_occupancy = raw_metrics['buffer_occupancy']
    if buffer_occupancy:
        total_occupancy = sum(buffer_occupancy.values())
        num_routers = len(buffer_occupancy)
        # Assume buffer depth = 4 (can be passed as parameter if needed)
        avg_buffer_util = total_occupancy / (num_routers * 4) if num_routers > 0 else 0.0
    else:
        avg_buffer_util = 0.0
    
    return {
        'cycle': cycle,
        'throughput': throughput,
        'bytes_transferred': bytes_transferred,
        'completed_transactions': completed,
        'buffer_utilization': avg_buffer_util,
        'mesh_cols': raw_metrics['mesh_cols'],
        'mesh_rows': raw_metrics['mesh_rows'],
    }


def calculate_average_latency(
    system: Any,
    pattern: str = "neighbor"
) -> Tuple[float, Tuple[int, int], Tuple[int, int]]:
    """
    Calculate average latency for a traffic pattern.
    
    Monitor-based: extracts latency from completed transactions.
    
    Args:
        system: Simulation system
        pattern: Traffic pattern name
    
    Returns:
        Tuple of (avg_latency, representative_src, representative_dest)
    """
    # For neighbor pattern: avg hop distance is 1
    # This is a simplified calculation
    # In real implementation, should track latency per transaction
    
    metrics = extract_metrics_from_simulation(system)
    cycle = metrics['cycle']
    completed = metrics['completed_transactions']
    
    # Simplified: assume latency ≈ cycle / completed
    # In reality, should aggregate from transaction trackers
    if completed > 0:
        avg_latency = cycle / completed
    else:
        avg_latency = 0.0
    
    # Representative src/dest for the pattern
    if pattern == "neighbor":
        src, dest = (1, 1), (2, 1)  # 1 hop east
    elif pattern == "transpose":
        src, dest = (1, 1), (1, 1)  # Transpose (x,y) to (y,x)
    else:
        src, dest = (1, 1), (3, 2)  # Average case
    
    return avg_latency, src, dest


def load_baseline(baseline_path: Path) -> Dict[str, Any]:
    """
    Load baseline metrics from JSON file.
    
    Args:
        baseline_path: Path to baseline JSON file
    
    Returns:
        Dictionary containing baseline metrics
    """
    import json
    
    with open(baseline_path, 'r', encoding='utf-8') as f:
        baseline = json.load(f)
    
    return baseline


def generate_validation_report(
    results: Dict[str, Tuple[bool, str]],
    output_path: Path
) -> None:
    """
    Generate HTML validation report.
    
    Args:
        results: Validation results from validators
        output_path: Path to save HTML report
    """
    html_content = """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Performance Validation Report</title>
        <style>
            body { font-family: Arial, sans-serif; margin: 20px; }
            h1 { color: #333; }
            table { border-collapse: collapse; width: 100%; margin-top: 20px; }
            th, td { border: 1px solid #ddd; padding: 12px; text-align: left; }
            th { background-color: #4CAF50; color: white; }
            .pass { color: green; font-weight: bold; }
            .fail { color: red; font-weight: bold; }
        </style>
    </head>
    <body>
        <h1>Performance Validation Report</h1>
        <table>
            <tr>
                <th>Metric</th>
                <th>Status</th>
                <th>Message</th>
            </tr>
    """
    
    all_passed = True
    for metric, (is_valid, message) in results.items():
        status_class = "pass" if is_valid else "fail"
        status_text = "PASS" if is_valid else "FAIL"
        all_passed = all_passed and is_valid
        
        html_content += f"""
            <tr>
                <td>{metric}</td>
                <td class="{status_class}">{status_text}</td>
                <td>{message}</td>
            </tr>
        """
    
    html_content += f"""
        </table>
        <h2>Overall: <span class="{'pass' if all_passed else 'fail'}">{
            'PASS' if all_passed else 'FAIL'
        }</span></h2>
    </body>
    </html>
    """
    
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write(html_content)
