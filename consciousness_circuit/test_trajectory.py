#!/usr/bin/env python3
"""
Unit tests for trajectory analysis components.
"""

import numpy as np
import pytest
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
from consciousness_circuit.plugins import (
    TrajectoryPlugin,
    ChaosPlugin,
    AgencyPlugin,
)


class TestHeliosMetrics:
    """Test helios trajectory analysis metrics."""
    
    def test_compute_lyapunov_exponent_stable(self):
        """Test Lyapunov exponent for converging trajectory."""
        # Create a trajectory that converges to zero
        t = np.linspace(0, 10, 100)
        trajectory = np.exp(-t).reshape(-1, 1)
        
        lyapunov = compute_lyapunov_exponent(trajectory)
        # Should be negative or near zero for stable/converging system
        assert lyapunov < 1.0, "Lyapunov should be small for converging trajectory"
    
    def test_compute_lyapunov_exponent_chaotic(self):
        """Test Lyapunov exponent for chaotic trajectory."""
        # Create a more chaotic trajectory
        np.random.seed(42)
        trajectory = np.cumsum(np.random.randn(100, 3), axis=0)
        
        lyapunov = compute_lyapunov_exponent(trajectory)
        # Should return a valid number
        assert isinstance(lyapunov, float)
        assert not np.isnan(lyapunov)
    
    def test_compute_hurst_exponent_random_walk(self):
        """Test Hurst exponent for random walk."""
        np.random.seed(42)
        trajectory = np.cumsum(np.random.randn(200))
        
        hurst = compute_hurst_exponent(trajectory)
        # Random walk should have H ≈ 0.5
        assert 0.0 <= hurst <= 1.0
        assert 0.3 < hurst < 0.7, f"Random walk Hurst={hurst:.3f} should be near 0.5"
    
    def test_compute_hurst_exponent_persistent(self):
        """Test Hurst exponent for persistent signal."""
        # Create persistent signal (trending)
        t = np.linspace(0, 10, 200)
        trajectory = t + 0.1 * np.random.randn(200)
        
        hurst = compute_hurst_exponent(trajectory)
        # Persistent trend should have H > 0.5
        assert hurst > 0.4, f"Persistent signal Hurst={hurst:.3f} should be > 0.4"
    
    def test_compute_msd_ballistic(self):
        """Test MSD for ballistic motion."""
        # Linear motion (ballistic)
        t = np.linspace(0, 10, 50)
        trajectory = np.column_stack([t, t, t])
        
        msd = compute_msd_from_trajectory(trajectory)
        
        # MSD should increase
        assert len(msd) > 0
        assert msd[-1] > msd[0], "MSD should increase over time"
        
        # For ballistic motion, MSD ~ t^2
        # Check that MSD is growing faster than linearly
        if len(msd) > 2:
            assert msd[-1] / msd[1] > 2, "Ballistic MSD should grow super-linearly"
    
    def test_compute_msd_confined(self):
        """Test MSD for confined motion."""
        # Oscillating motion (confined)
        t = np.linspace(0, 10, 100)
        trajectory = np.column_stack([np.sin(t), np.cos(t)])
        
        msd = compute_msd_from_trajectory(trajectory)
        
        # MSD should saturate for confined motion
        assert len(msd) > 0
        # Check that late MSD is not much larger than early MSD
        if len(msd) > 20:
            early_msd = np.mean(msd[:10])
            late_msd = np.mean(msd[-10:])
            assert late_msd / early_msd < 5, "Confined motion MSD should not grow much"
    
    def test_classify_trajectory(self):
        """Test trajectory classification."""
        # Ballistic trajectory
        t = np.linspace(0, 10, 50)
        trajectory = np.column_stack([t**2, t**2])
        msd = compute_msd_from_trajectory(trajectory)
        classification = classify_trajectory(msd)
        
        assert isinstance(classification, str)
        assert classification in [c.value for c in SignalClass]
    
    def test_verify_signal(self):
        """Test signal verification."""
        np.random.seed(42)
        trajectory = np.cumsum(np.random.randn(100, 5), axis=0)
        
        signal_class = verify_signal(trajectory)
        
        assert isinstance(signal_class, str)
        assert signal_class in [c.value for c in SignalClass]
    
    def test_compute_attractor_score(self):
        """Test attractor score computation."""
        # Converging trajectory
        t = np.linspace(0, 10, 100)
        trajectory = np.exp(-t).reshape(-1, 1)
        
        strength, is_converging = compute_attractor_score(trajectory)
        
        assert 0.0 <= strength <= 1.0
        assert isinstance(is_converging, bool)


class TestTAMEMetrics:
    """Test TAME (Trajectory Analysis for Meta-Emergence) metrics."""
    
    def test_compute_agency_score(self):
        """Test agency score computation."""
        # Directed motion (high agency)
        t = np.linspace(0, 10, 50)
        trajectory = np.column_stack([t, t])
        
        agency = compute_agency_score(trajectory)
        
        assert 0.0 <= agency <= 1.0
        assert agency > 0.3, "Directed motion should have moderate agency"
    
    def test_compute_agency_score_random(self):
        """Test agency score for random motion."""
        np.random.seed(42)
        trajectory = np.random.randn(50, 3)
        
        agency = compute_agency_score(trajectory)
        
        assert 0.0 <= agency <= 1.0
        # Random motion should have low agency
        assert agency < 0.7, "Random motion should have lower agency"
    
    def test_detect_attractor_convergence(self):
        """Test attractor convergence detection."""
        # Converging trajectory
        t = np.linspace(0, 10, 100)
        trajectory = np.exp(-t).reshape(-1, 1)
        
        strength, is_converging = detect_attractor_convergence(trajectory)
        
        assert 0.0 <= strength <= 1.0
        assert isinstance(is_converging, bool)
    
    def test_compute_goal_directedness(self):
        """Test goal-directedness computation."""
        # Moving toward a goal
        goal = np.array([10.0, 10.0])
        t = np.linspace(0, 1, 50)
        trajectory = np.column_stack([t * goal[0], t * goal[1]])
        
        goal_dir = compute_goal_directedness(trajectory, goal)
        
        assert 0.0 <= goal_dir <= 1.0
        assert goal_dir > 0.5, "Direct path to goal should have high goal-directedness"
    
    def test_compute_trajectory_coherence(self):
        """Test trajectory coherence computation."""
        # Structured trajectory
        t = np.linspace(0, 10, 100)
        trajectory = np.column_stack([np.sin(t), np.cos(t)])
        
        coherence = compute_trajectory_coherence(trajectory)
        
        assert 0.0 <= coherence <= 1.0
    
    def test_tame_metrics_class(self):
        """Test TAMEMetrics class."""
        np.random.seed(42)
        trajectory = np.cumsum(np.random.randn(100, 5), axis=0)
        
        tame = TAMEMetrics()
        results = tame.compute_all(trajectory)
        
        assert "agency_score" in results
        assert "attractor_strength" in results
        assert "is_converging" in results
        assert "goal_directedness" in results
        assert "trajectory_coherence" in results
        assert "overall_tame_score" in results
        
        # Check all values are in valid ranges
        assert 0.0 <= results["agency_score"] <= 1.0
        assert 0.0 <= results["attractor_strength"] <= 1.0
        assert 0.0 <= results["goal_directedness"] <= 1.0
        assert 0.0 <= results["trajectory_coherence"] <= 1.0
        assert 0.0 <= results["overall_tame_score"] <= 1.0


class TestPlugins:
    """Test analysis plugins."""
    
    def test_trajectory_plugin(self):
        """Test TrajectoryPlugin."""
        plugin = TrajectoryPlugin()
        
        # Create test trajectory
        t = np.linspace(0, 10, 50)
        trajectory = np.column_stack([t, t])
        
        result = plugin.analyze(trajectory)
        
        assert "msd" in result
        assert "trajectory_class" in result
        assert "diffusion_coefficient" in result
        assert isinstance(result["trajectory_class"], str)
    
    def test_chaos_plugin(self):
        """Test ChaosPlugin."""
        plugin = ChaosPlugin()
        
        np.random.seed(42)
        trajectory = np.cumsum(np.random.randn(100, 5), axis=0)
        
        result = plugin.analyze(trajectory)
        
        assert "lyapunov" in result
        assert "hurst" in result
        assert "signal_class" in result
        assert isinstance(result["lyapunov"], float)
        assert isinstance(result["hurst"], float)
        assert isinstance(result["signal_class"], str)
    
    def test_agency_plugin(self):
        """Test AgencyPlugin."""
        plugin = AgencyPlugin()
        
        # Directed trajectory
        t = np.linspace(0, 10, 50)
        trajectory = np.column_stack([t, t])
        
        result = plugin.analyze(trajectory)
        
        assert "agency_score" in result
        assert "attractor_strength" in result
        assert "is_converging" in result
        assert "goal_directedness" in result
        assert 0.0 <= result["agency_score"] <= 1.0
        assert 0.0 <= result["goal_directedness"] <= 1.0


class TestEdgeCases:
    """Test edge cases and error handling."""
    
    def test_short_trajectory(self):
        """Test handling of very short trajectories."""
        trajectory = np.array([[1.0], [2.0]])
        
        # Should not crash
        lyapunov = compute_lyapunov_exponent(trajectory)
        assert isinstance(lyapunov, float)
        
        hurst = compute_hurst_exponent(trajectory)
        assert isinstance(hurst, float)
    
    def test_single_dimension(self):
        """Test handling of 1D trajectories."""
        trajectory = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        
        hurst = compute_hurst_exponent(trajectory)
        assert isinstance(hurst, float)
        assert 0.0 <= hurst <= 1.0
    
    def test_constant_trajectory(self):
        """Test handling of constant trajectories."""
        trajectory = np.ones((50, 3))
        
        # Should handle gracefully
        msd = compute_msd_from_trajectory(trajectory)
        assert len(msd) > 0
        assert np.all(msd >= 0)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
