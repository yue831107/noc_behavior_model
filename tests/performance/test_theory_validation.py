"""
Theory-based validation tests.

Tests that verify performance metrics are within theoretical bounds.
Uses Monitor-based approach - no core modification required.
"""

import pytest
from tests.performance.validators.theory_validator import (
    TheoryValidator,
    MeshConfig,
    RouterConfig,
    print_validation_results
)


class TestTheoryValidation:
    """Theory-based validation test cases."""
    
    def test_throughput_max_calculation(self):
        """
        驗證公式: T_max = N_edge × W_flit = 4 × 8 = 32 B/c
        """
        config = MeshConfig(cols=5, rows=4)
        router_config = RouterConfig(flit_width_bytes=8)
        validator = TheoryValidator(config, router_config)
        
        # V1 architecture: Routing Selector bottleneck
        max_tp = validator.calculate_max_throughput()
        assert max_tp == 32.0
    
    def test_latency_min_calculation(self):
        """
        驗證公式: L_min = D_Manhattan × d_pipeline
        """
        validator = TheoryValidator()
        
        # 1 hop distance
        assert validator.calculate_min_latency((1, 1), (2, 1)) == 1
        
        # 3 hops distance (2 east + 1 north)
        assert validator.calculate_min_latency((1, 1), (3, 2)) == 3
    
    def test_validate_throughput(self):
        """
        驗證準則: T_actual ≤ T_max × (1 + ε)
        """
        validator = TheoryValidator()
        
        # PASS: 28.0 ≤ 32.0
        is_valid, msg = validator.validate_throughput(28.0)
        assert is_valid and msg == "OK"
        
        # FAIL: 40.0 > 32.0
        is_valid, msg = validator.validate_throughput(40.0)
        assert not is_valid and "violation" in msg.lower()
    
    def test_validate_latency(self):
        """
        驗證準則: L_min × 0.95 ≤ L_actual ≤ L_max
        """
        validator = TheoryValidator()
        
        # PASS: latency in valid range
        is_valid, _ = validator.validate_latency(2.5, (1, 1), (2, 1))
        assert is_valid
        
        # FAIL: latency below minimum (impossible)
        is_valid, msg = validator.validate_latency(0.5, (1, 1), (3, 3))
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
        results = validator.validate_all({
            'throughput': 28.0,
            'avg_latency': 2.5,
            'src': (1, 1),
            'dest': (2, 1),
            'buffer_utilization': 0.35,
        })
        assert all(is_valid for is_valid, _ in results.values())
        
        # All invalid metrics
        results = validator.validate_all({
            'throughput': 50.0,   # > 32
            'avg_latency': 0.1,   # too low
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
        router_config=RouterConfig(flit_width_bytes=8, pipeline_depth=1)
    )
    
    # Mock simulation metrics
    results = validator.validate_all({
        'throughput': 28.0,
        'avg_latency': 2.8,
        'src': (1, 1),
        'dest': (2, 2),
        'buffer_utilization': 0.42,
    })
    
    print_validation_results(results)
    assert all(is_valid for is_valid, _ in results.values())
