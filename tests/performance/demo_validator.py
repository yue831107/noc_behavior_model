"""
Demo: How the validator detects violations.

This shows what happens when you feed invalid metrics to the validator.
"""

from src.verification.theory_validator import (
    TheoryValidator,
    print_validation_results
)

def demo_normal_metrics():
    """Demo: Normal metrics should PASS."""
    print("\n" + "="*70)
    print("Demo 1: Normal Metrics (應該全部通過)")
    print("="*70)
    
    validator = TheoryValidator()
    
    metrics = {
        'throughput': 28.0,         # OK: < 32 B/c
        'avg_latency': 2.5,         # OK: reasonable
        'src': (1, 1),
        'dest': (2, 1),             # 1 hop
        'buffer_utilization': 0.35, # OK: 35%
    }
    
    results = validator.validate_all(metrics)
    print_validation_results(results)


def demo_throughput_violation():
    """Demo: Throughput violation should be DETECTED."""
    print("\n" + "="*70)
    print("Demo 2: Throughput Violation (應該偵測到錯誤)")
    print("="*70)
    
    validator = TheoryValidator()
    
    metrics = {
        'throughput': 45.0,  # ❌ FAIL: > 32 B/c (超過理論上限)
        'avg_latency': 2.5,
        'src': (1, 1),
        'dest': (2, 1),
        'buffer_utilization': 0.35,
    }
    
    results = validator.validate_all(metrics)
    print_validation_results(results)


def demo_multiple_violations():
    """Demo: Multiple violations should be DETECTED."""
    print("\n" + "="*70)
    print("Demo 3: Multiple Violations (應該偵測到多個錯誤)")
    print("="*70)
    
    validator = TheoryValidator()
    
    metrics = {
        'throughput': 50.0,   # ❌ FAIL: > 32 B/c
        'avg_latency': 0.5,   # ❌ FAIL: < min latency for 1 hop
        'src': (1, 1),
        'dest': (2, 1),       # 1 hop, min = 1 cycle
        'buffer_utilization': 1.2,  # ❌ FAIL: > 100%
    }
    
    results = validator.validate_all(metrics)
    print_validation_results(results)


if __name__ == "__main__":
    # Run all demos
    demo_normal_metrics()
    demo_throughput_violation()
    demo_multiple_violations()
    
    print("\n" + "="*70)
    print("總結：驗證器能正確偵測異常值！")
    print("="*70)
