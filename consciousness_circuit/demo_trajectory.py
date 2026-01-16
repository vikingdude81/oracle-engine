#!/usr/bin/env python3
"""
Demo: Trajectory Analysis with Consciousness Measurement
=========================================================

Demonstrates the new trajectory analysis features without requiring a full model.
Shows how the components work with synthetic data.
"""

import numpy as np
import sys
import os

# Add parent directory to path
sys.path.insert(0, os.path.dirname(__file__))

from helios_metrics import (
    compute_lyapunov_exponent,
    compute_hurst_exponent,
    compute_msd_from_trajectory,
    verify_signal,
    SignalClass,
)
from tame_metrics import TAMEMetrics


def generate_test_trajectories():
    """Generate different types of trajectories for testing."""
    np.random.seed(42)
    
    # 1. Ballistic (directed motion)
    t = np.linspace(0, 10, 100)
    ballistic = np.column_stack([t * 2, t * 2, t * 2])
    
    # 2. Diffusive (random walk)
    diffusive = np.cumsum(np.random.randn(100, 3), axis=0)
    
    # 3. Confined (oscillatory)
    confined = np.column_stack([
        np.sin(t),
        np.cos(t),
        np.sin(2*t)
    ])
    
    # 4. Converging (to attractor)
    converging = np.exp(-t).reshape(-1, 1) * np.random.randn(1, 3) + np.random.randn(100, 3) * 0.1
    
    return {
        "Ballistic (Directed)": ballistic,
        "Diffusive (Random)": diffusive,
        "Confined (Oscillatory)": confined,
        "Converging (Attractor)": converging,
    }


def analyze_trajectory(name: str, trajectory: np.ndarray):
    """Analyze a single trajectory and print results."""
    print(f"\n{'='*70}")
    print(f"Analyzing: {name}")
    print(f"{'='*70}")
    print(f"Shape: {trajectory.shape}")
    
    # Helios metrics
    lyapunov = compute_lyapunov_exponent(trajectory)
    hurst = compute_hurst_exponent(trajectory)
    msd = compute_msd_from_trajectory(trajectory)
    signal_class = verify_signal(trajectory, lyapunov, hurst)
    
    print(f"\nChaos & Dynamics:")
    print(f"  Lyapunov exponent: {lyapunov:+.4f}")
    if lyapunov > 0.5:
        print(f"    → High chaos (unstable)")
    elif lyapunov < -0.3:
        print(f"    → Stable/converging")
    else:
        print(f"    → Neutral")
    
    print(f"  Hurst exponent: {hurst:.4f}")
    if hurst > 0.6:
        print(f"    → Persistent (trending)")
    elif hurst < 0.4:
        print(f"    → Anti-persistent (mean-reverting)")
    else:
        print(f"    → Random walk")
    
    print(f"  Signal class: {signal_class}")
    print(f"  MSD range: {msd[0]:.4f} → {msd[-1]:.4f}")
    
    # TAME metrics
    tame = TAMEMetrics()
    tame_results = tame.compute_all(trajectory)
    
    print(f"\nAgency & Goal-Directedness:")
    print(f"  Agency score: {tame_results['agency_score']:.4f}")
    print(f"  Goal-directedness: {tame_results['goal_directedness']:.4f}")
    print(f"  Attractor strength: {tame_results['attractor_strength']:.4f}")
    print(f"  Is converging: {tame_results['is_converging']}")
    print(f"  Trajectory coherence: {tame_results['trajectory_coherence']:.4f}")
    print(f"\n  Overall TAME score: {tame_results['overall_tame_score']:.4f}")


def main():
    """Run trajectory analysis demo."""
    print("="*70)
    print("TRAJECTORY ANALYSIS DEMO")
    print("="*70)
    print("\nThis demo shows how the trajectory analysis components work")
    print("with different types of motion patterns.\n")
    
    trajectories = generate_test_trajectories()
    
    for name, trajectory in trajectories.items():
        analyze_trajectory(name, trajectory)
    
    print(f"\n{'='*70}")
    print("Demo Complete!")
    print(f"{'='*70}")
    print("\nKey Insights:")
    print("  • Ballistic motion → high agency, directed")
    print("  • Diffusive motion → lower agency, exploratory")
    print("  • Confined motion → low Lyapunov, bounded")
    print("  • Converging motion → negative Lyapunov, attracting")
    print("\nThese metrics can be applied to LLM hidden states to understand")
    print("whether the model is:")
    print("  - Exploring vs exploiting (Lyapunov)")
    print("  - Maintaining context vs drifting (Hurst)")
    print("  - Goal-directed vs reactive (agency)")
    print("  - Converging to coherent thought (attractor)")


if __name__ == "__main__":
    main()
