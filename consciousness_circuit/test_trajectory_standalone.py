#!/usr/bin/env python3
"""
Unit tests for trajectory analysis components (standalone version).
Tests helios_metrics and tame_metrics without importing through __init__.
"""

import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..'))

import numpy as np
import pytest

# Import directly to avoid the broken logging_config
from consciousness_circuit.helios_metrics import (
    SignalClass,
    compute_lyapunov_exponent,
    compute_hurst_exponent,
    compute_msd_from_trajectory,
    classify_trajectory,
    verify_signal,
    compute_attractor_score,
)
from consciousness_circuit.tame_metrics import (
    compute_agency_score,
    detect_attractor_convergence,
    compute_goal_directedness,
    compute_trajectory_coherence,
    TAMEMetrics,
)


class TestHeliosMetricsBasic:
    """Basic tests for helios trajectory analysis metrics."""
    
    def test_compute_lyapunov_exponent_stable(self):
        """Test Lyapunov exponent for converging trajectory."""
        t = np.linspace(0, 10, 100)
        trajectory = np.exp(-t).reshape(-1, 1)
        
        lyapunov = compute_lyapunov_exponent(trajectory)
        assert isinstance(lyapunov, float)
        assert not np.isnan(lyapunov)
        print(f"✓ Lyapunov for converging trajectory: {lyapunov:.4f}")
    
    def test_compute_hurst_exponent_random_walk(self):
        """Test Hurst exponent for random walk."""
        np.random.seed(42)
        trajectory = np.cumsum(np.random.randn(200))
        
        hurst = compute_hurst_exponent(trajectory)
        assert 0.0 <= hurst <= 1.0
        print(f"✓ Hurst for random walk: {hurst:.4f} (expected ~0.5)")
    
    def test_compute_msd_ballistic(self):
        """Test MSD for ballistic motion."""
        t = np.linspace(0, 10, 50)
        trajectory = np.column_stack([t, t, t])
        
        msd = compute_msd_from_trajectory(trajectory)
        
        assert len(msd) > 0
        assert msd[-1] > msd[0]
        print(f"✓ MSD for ballistic motion: starts at {msd[0]:.2f}, ends at {msd[-1]:.2f}")
    
    def test_classify_trajectory(self):
        """Test trajectory classification."""
        t = np.linspace(0, 10, 50)
        trajectory = np.column_stack([t**2, t**2])
        msd = compute_msd_from_trajectory(trajectory)
        classification = classify_trajectory(msd)
        
        assert isinstance(classification, str)
        assert classification in [c.value for c in SignalClass]
        print(f"✓ Trajectory classified as: {classification}")
    
    def test_verify_signal(self):
        """Test signal verification."""
        np.random.seed(42)
        trajectory = np.cumsum(np.random.randn(100, 5), axis=0)
        
        signal_class = verify_signal(trajectory)
        
        assert isinstance(signal_class, str)
        assert signal_class in [c.value for c in SignalClass]
        print(f"✓ Signal verified as: {signal_class}")


class TestTAMEMetricsBasic:
    """Basic tests for TAME metrics."""
    
    def test_compute_agency_score(self):
        """Test agency score computation."""
        t = np.linspace(0, 10, 50)
        trajectory = np.column_stack([t, t])
        
        agency = compute_agency_score(trajectory)
        
        assert 0.0 <= agency <= 1.0
        print(f"✓ Agency score for directed motion: {agency:.4f}")
    
    def test_detect_attractor_convergence(self):
        """Test attractor convergence detection."""
        t = np.linspace(0, 10, 100)
        trajectory = np.exp(-t).reshape(-1, 1)
        
        strength, is_converging = detect_attractor_convergence(trajectory)
        
        assert 0.0 <= strength <= 1.0
        assert isinstance(is_converging, bool)
        print(f"✓ Attractor: strength={strength:.4f}, converging={is_converging}")
    
    def test_compute_goal_directedness(self):
        """Test goal-directedness computation."""
        goal = np.array([10.0, 10.0])
        t = np.linspace(0, 1, 50)
        trajectory = np.column_stack([t * goal[0], t * goal[1]])
        
        goal_dir = compute_goal_directedness(trajectory, goal)
        
        assert 0.0 <= goal_dir <= 1.0
        print(f"✓ Goal-directedness toward target: {goal_dir:.4f}")
    
    def test_tame_metrics_class(self):
        """Test TAMEMetrics class."""
        np.random.seed(42)
        trajectory = np.cumsum(np.random.randn(100, 5), axis=0)
        
        tame = TAMEMetrics()
        results = tame.compute_all(trajectory)
        
        assert "agency_score" in results
        assert "overall_tame_score" in results
        assert 0.0 <= results["agency_score"] <= 1.0
        assert 0.0 <= results["overall_tame_score"] <= 1.0
        print(f"✓ TAME metrics: {tame}")


def run_manual_tests():
    """Run tests manually without pytest."""
    print("="*70)
    print("Running Trajectory Analysis Tests")
    print("="*70)
    
    test_helios = TestHeliosMetricsBasic()
    test_tame = TestTAMEMetricsBasic()
    
    print("\nHelios Metrics Tests:")
    print("-"*70)
    test_helios.test_compute_lyapunov_exponent_stable()
    test_helios.test_compute_hurst_exponent_random_walk()
    test_helios.test_compute_msd_ballistic()
    test_helios.test_classify_trajectory()
    test_helios.test_verify_signal()
    
    print("\nTAME Metrics Tests:")
    print("-"*70)
    test_tame.test_compute_agency_score()
    test_tame.test_detect_attractor_convergence()
    test_tame.test_compute_goal_directedness()
    test_tame.test_tame_metrics_class()
    
    print("\n" + "="*70)
    print("✓ All tests passed!")
    print("="*70)


if __name__ == "__main__":
    # Try pytest first, fall back to manual
    try:
        pytest.main([__file__, "-v", "-s"])
    except:
        run_manual_tests()
