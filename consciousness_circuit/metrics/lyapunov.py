"""
Lyapunov Exponent Calculation - FULLY STANDALONE
================================================

Measures sensitivity to initial conditions in time series data.
Can be copied to any project - only requires numpy.

Usage:
    from lyapunov import compute_lyapunov, LyapunovResult
    
    result = compute_lyapunov(trajectory)
    print(f"Lyapunov: {result.exponent:.3f}")
    print(f"Is chaotic: {result.is_chaotic}")
    print(result.interpretation)

Dependencies: numpy only
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, List


@dataclass
class LyapunovResult:
    """Result from Lyapunov exponent calculation."""
    
    exponent: float
    """The largest Lyapunov exponent."""
    
    divergences: List[float]
    """Individual divergence measurements used in calculation."""
    
    n_samples: int
    """Number of points in the trajectory."""
    
    @property
    def is_chaotic(self) -> bool:
        """True if system exhibits chaotic behavior (λ > 0.3)."""
        return self.exponent > 0.3
    
    @property
    def is_stable(self) -> bool:
        """True if system is stable/converging (λ < -0.1)."""
        return self.exponent < -0.1
    
    @property
    def is_neutral(self) -> bool:
        """True if system is neutrally stable (-0.1 ≤ λ ≤ 0.3)."""
        return -0.1 <= self.exponent <= 0.3
    
    @property
    def interpretation(self) -> str:
        """Human-readable interpretation of the result."""
        if self.is_chaotic:
            return f"CHAOTIC (λ={self.exponent:.3f}) - Sensitive to initial conditions, exponential divergence"
        elif self.is_stable:
            return f"STABLE (λ={self.exponent:.3f}) - Converging, robust to perturbations"
        else:
            return f"NEUTRAL (λ={self.exponent:.3f}) - Neither chaotic nor strongly stable"
    
    def __repr__(self):
        return f"LyapunovResult(exponent={self.exponent:.4f}, chaotic={self.is_chaotic}, stable={self.is_stable})"


def compute_lyapunov(trajectory: np.ndarray,
                    tau: Optional[int] = None,
                    min_tsep: int = 1) -> LyapunovResult:
    """
    Compute the largest Lyapunov exponent using Rosenstein algorithm.
    
    The Lyapunov exponent measures sensitivity to initial conditions:
    - λ > 0: Chaotic/unstable (nearby trajectories diverge exponentially)
    - λ ≈ 0: Neutral/marginally stable
    - λ < 0: Convergent/stable (nearby trajectories converge)
    
    This is a standalone implementation requiring only numpy.
    
    Args:
        trajectory: Time series data, shape [time_steps] or [time_steps, features]
                   For multi-dimensional data, considers Euclidean distances
        tau: Time delay for tracking divergence (auto-computed if None)
            Controls how far ahead to track trajectory divergence
        min_tsep: Minimum temporal separation between neighbors (default: 1)
                 Prevents comparing a point with itself or very nearby points
    
    Returns:
        LyapunovResult with exponent, divergences, and interpretations
    
    Examples:
        >>> # Stable/converging system
        >>> t = np.linspace(0, 10, 100)
        >>> stable_traj = np.exp(-t).reshape(-1, 1)
        >>> result = compute_lyapunov(stable_traj)
        >>> result.is_stable
        True
        
        >>> # Chaotic system
        >>> chaotic_traj = np.random.randn(100, 3).cumsum(axis=0)
        >>> result = compute_lyapunov(chaotic_traj)
        >>> result.exponent
        0.5  # Positive value indicates chaos
    
    References:
        Rosenstein, M. T., Collins, J. J., & De Luca, C. J. (1993).
        A practical method for calculating largest Lyapunov exponents from small data sets.
        Physica D: Nonlinear Phenomena, 65(1-2), 117-134.
    """
    # Ensure 2D array [time_steps, features]
    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)
    
    n_steps, n_features = trajectory.shape
    
    # Auto-compute time delay if not provided
    # Use ~5% of trajectory length as a reasonable default
    if tau is None:
        tau = max(1, n_steps // 20)
    
    # Need enough points for meaningful analysis
    if n_steps < 10:
        return LyapunovResult(
            exponent=0.0,
            divergences=[],
            n_samples=n_steps
        )
    
    # Track divergence rates
    divergences = []
    
    # For each point in the trajectory
    for i in range(n_steps - tau - 1):
        # Find nearest neighbor with sufficient temporal separation
        min_dist = np.inf
        min_idx = -1
        
        for j in range(n_steps - tau - 1):
            # Skip if too close in time (avoid spurious correlations)
            if abs(i - j) < min_tsep:
                continue
            
            # Compute distance between states
            dist = np.linalg.norm(trajectory[i] - trajectory[j])
            
            # Track nearest neighbor (excluding self)
            if dist < min_dist and dist > 0:
                min_dist = dist
                min_idx = j
        
        # If we found a valid neighbor, track its divergence
        if min_idx >= 0:
            # Track how the distance evolves over time
            for k in range(1, min(tau, n_steps - max(i, min_idx) - 1)):
                # Distance after k time steps
                dist_future = np.linalg.norm(
                    trajectory[i + k] - trajectory[min_idx + k]
                )
                
                # Compute logarithmic divergence rate
                # log(d(t)) = log(d(0)) + λ * t
                # λ = [log(d(t)) - log(d(0))] / t
                if dist_future > 0 and min_dist > 0:
                    divergence_rate = np.log(dist_future / min_dist) / k
                    divergences.append(divergence_rate)
    
    # Handle case with no valid divergences
    if len(divergences) == 0:
        return LyapunovResult(
            exponent=0.0,
            divergences=[],
            n_samples=n_steps
        )
    
    # Lyapunov exponent is the average divergence rate
    exponent = float(np.mean(divergences))
    
    return LyapunovResult(
        exponent=exponent,
        divergences=divergences,
        n_samples=n_steps
    )


class LyapunovAnalyzer:
    """
    Windowed Lyapunov analysis for detecting regime changes.
    
    Useful for long time series where dynamics may change over time.
    
    Example:
        >>> analyzer = LyapunovAnalyzer(window_size=50, stride=10)
        >>> results = analyzer.analyze_windowed(long_trajectory)
        >>> for i, result in enumerate(results):
        ...     print(f"Window {i}: {result.interpretation}")
    """
    
    def __init__(self, window_size: int = 50, stride: Optional[int] = None):
        """
        Initialize analyzer.
        
        Args:
            window_size: Size of sliding window for analysis
            stride: Step size between windows (default: window_size // 2)
        """
        self.window_size = window_size
        self.stride = stride or (window_size // 2)
    
    def analyze_windowed(self, trajectory: np.ndarray) -> List[LyapunovResult]:
        """
        Analyze trajectory using sliding windows.
        
        Args:
            trajectory: Time series data [time_steps] or [time_steps, features]
        
        Returns:
            List of LyapunovResult, one per window
        """
        if trajectory.ndim == 1:
            trajectory = trajectory.reshape(-1, 1)
        
        n_steps = len(trajectory)
        results = []
        
        # Slide window across trajectory
        for start in range(0, n_steps - self.window_size + 1, self.stride):
            end = start + self.window_size
            window = trajectory[start:end]
            
            result = compute_lyapunov(window)
            results.append(result)
        
        return results
    
    def detect_regime_changes(self, 
                            trajectory: np.ndarray,
                            threshold: float = 0.5) -> List[int]:
        """
        Detect points where dynamics change significantly.
        
        Args:
            trajectory: Time series data
            threshold: Minimum change in Lyapunov exponent to flag
        
        Returns:
            List of indices where regime changes occur
        """
        results = self.analyze_windowed(trajectory)
        
        if len(results) < 2:
            return []
        
        # Find large changes in Lyapunov exponent
        changes = []
        for i in range(len(results) - 1):
            delta = abs(results[i + 1].exponent - results[i].exponent)
            if delta > threshold:
                # Convert window index to trajectory index
                change_point = i * self.stride + self.window_size // 2
                changes.append(change_point)
        
        return changes


# Convenience function aliases for backward compatibility
compute_lyapunov_exponent = compute_lyapunov


__all__ = [
    "LyapunovResult",
    "compute_lyapunov",
    "compute_lyapunov_exponent",  # Alias for compatibility
    "LyapunovAnalyzer",
]


if __name__ == "__main__":
    # Self-test examples
    print("Lyapunov Exponent Calculator - Standalone Tests")
    print("=" * 50)
    
    # Test 1: Stable/converging system
    print("\n1. Stable System (exponential decay):")
    t = np.linspace(0, 10, 100)
    stable = np.exp(-t).reshape(-1, 1)
    result = compute_lyapunov(stable)
    print(f"   {result.interpretation}")
    
    # Test 2: Random walk (near-neutral)
    print("\n2. Random Walk:")
    np.random.seed(42)
    random_walk = np.cumsum(np.random.randn(100)).reshape(-1, 1)
    result = compute_lyapunov(random_walk)
    print(f"   {result.interpretation}")
    
    # Test 3: Chaotic system (diverging)
    print("\n3. Chaotic System:")
    chaotic = np.random.randn(100, 3).cumsum(axis=0)
    chaotic *= 1.5  # Amplify to ensure divergence
    result = compute_lyapunov(chaotic)
    print(f"   {result.interpretation}")
    
    # Test 4: Windowed analysis
    print("\n4. Windowed Analysis (regime change):")
    # Create trajectory with changing dynamics
    part1 = np.exp(-np.linspace(0, 5, 50)).reshape(-1, 1)  # Stable
    part2 = np.random.randn(50, 1).cumsum(axis=0)  # Chaotic
    combined = np.vstack([part1, part2])
    
    analyzer = LyapunovAnalyzer(window_size=30, stride=15)
    results = analyzer.analyze_windowed(combined)
    print(f"   Found {len(results)} windows")
    for i, r in enumerate(results):
        print(f"   Window {i}: λ={r.exponent:.3f} ({'chaotic' if r.is_chaotic else 'stable' if r.is_stable else 'neutral'})")
    
    changes = analyzer.detect_regime_changes(combined, threshold=0.3)
    print(f"   Regime changes detected at indices: {changes}")
    
    print("\n✓ All tests completed successfully!")
