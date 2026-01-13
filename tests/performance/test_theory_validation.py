"""
Theory-based validation tests.

Tests that verify performance metrics are within theoretical bounds.
Uses Monitor-based approach - no core modification required.
"""

import pytest
from src.verification.theory_validator import (
    TheoryValidator,
    MeshConfig,
    RouterConfig,
    print_validation_results
)


class TestTheoryValidation:
    """Theory-based validation test cases."""
    
    def test_throughput_max_calculation(self):
        """
        驗證公式: T_max = N_NI × W_data

        V1 架構瓶頸在 SlaveNI (1 NI × 32 bytes = 32 B/cycle 單向)
        雙向 (Req + Resp) = 64 B/cycle
        """
        config = MeshConfig(cols=5, rows=4)
        router_config = RouterConfig(flit_data_bytes=32)
        validator = TheoryValidator(config, router_config)

        # Single direction (default for write-only or read-only)
        max_tp_single = validator.calculate_max_throughput(bidirectional=False)
        assert max_tp_single == 32.0

        # V1 architecture: SlaveNI bottleneck (bidirectional)
        max_tp = validator.calculate_max_throughput(bidirectional=True)
        assert max_tp == 64.0
    
    def test_latency_min_calculation(self):
        """
        驗證公式: L_min = D_Manhattan × d_pipeline + overhead

        Overhead = 2 cycles (NI + Selector)
        """
        validator = TheoryValidator()

        # 1 hop distance: 1 + 2 = 3
        assert validator.calculate_min_latency((1, 1), (2, 1)) == 3

        # 3 hops distance (2 east + 1 north): 3 + 2 = 5
        assert validator.calculate_min_latency((1, 1), (3, 2)) == 5

        # Without overhead: 1 hop = 1
        assert validator.calculate_min_latency((1, 1), (2, 1), include_overhead=False) == 1
    
    def test_validate_throughput(self):
        """
        驗證準則: T_actual ≤ T_max × (1 + ε)
        T_max = 32 B/cycle (single direction SlaveNI with 256-bit data)
        """
        validator = TheoryValidator()

        # PASS: 28.0 ≤ 32.0
        is_valid, msg = validator.validate_throughput(28.0)
        assert is_valid and msg == "OK"

        # FAIL: 40.0 > 32.0 × 1.05 = 33.6
        is_valid, msg = validator.validate_throughput(40.0)
        assert not is_valid and "violation" in msg.lower()
    
    def test_validate_latency(self):
        """
        驗證準則: L_min × 0.95 ≤ L_actual ≤ L_max

        For 1-hop path (1,1)->(2,1):
        - min_latency = 1 + 2 = 3 cycles
        - lower_bound = 3 × 0.95 = 2.85 cycles
        """
        validator = TheoryValidator()

        # PASS: latency in valid range (3.5 > 2.85)
        is_valid, _ = validator.validate_latency(3.5, (1, 1), (2, 1))
        assert is_valid

        # FAIL: latency below minimum (impossible)
        # For 4-hop path (1,1)->(3,3): min = 4 + 2 = 6
        is_valid, msg = validator.validate_latency(2.0, (1, 1), (3, 3))
        assert not is_valid and "below" in msg.lower()
    
    def test_validate_buffer_utilization(self):
        """
        驗證公式: 0 ≤ U_buffer ≤ 1
        """
        validator = TheoryValidator()
        
        # PASS: 0.35 in [0, 1]
        is_valid, _ = validator.validate_buffer_utilization(0.35)
        assert is_valid
        
        # FAIL: 1.5 > 1 (overflow)
        is_valid, msg = validator.validate_buffer_utilization(1.5)
        assert not is_valid and "exceeds" in msg.lower()
    
    def test_validate_all_metrics(self):
        """整合驗證：所有指標批次檢查"""
        validator = TheoryValidator()

        # All valid metrics
        # 1-hop path: min latency = 1 + 2 = 3, so 3.5 is valid
        results = validator.validate_all({
            'throughput': 28.0,  # ≤ 32.0
            'avg_latency': 3.5,
            'src': (1, 1),
            'dest': (2, 1),
            'buffer_utilization': 0.35,
        })
        assert all(is_valid for is_valid, _ in results.values())

        # All invalid metrics
        # 4-hop path: min latency = 4 + 2 = 6, so 0.1 is below
        results = validator.validate_all({
            'throughput': 40.0,   # > 32.0 × 1.05 = 33.6
            'avg_latency': 0.1,   # too low for 4-hop path
            'src': (1, 1),
            'dest': (3, 3),
            'buffer_utilization': 1.5,  # > 1
        })
        assert all(not is_valid for is_valid, _ in results.values())


def test_theory_validator_integration():
    """
    整合測試：驗證完整模擬流程
    """
    validator = TheoryValidator(
        mesh_config=MeshConfig(cols=5, rows=4),
        router_config=RouterConfig(flit_data_bytes=32, pipeline_depth=1)
    )

    # Mock simulation metrics
    # 2-hop path (1,1)->(2,2): min latency = 2 + 2 = 4
    # So 4.5 is a valid latency
    # Throughput must be ≤ 32.0 (SlaveNI limit with 256-bit data)
    results = validator.validate_all({
        'throughput': 28.0,  # ≤ 32.0 (FlooNoC limit)
        'avg_latency': 4.5,
        'src': (1, 1),
        'dest': (2, 2),
        'buffer_utilization': 0.42,
    })

    print_validation_results(results)
    assert all(is_valid for is_valid, _ in results.values())
