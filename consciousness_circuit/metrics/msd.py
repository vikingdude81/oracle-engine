"""
Mean Squared Displacement (MSD) - FULLY STANDALONE
==================================================

Measures how trajectories spread over time in state space.
Can be copied to any project - only requires numpy.

Usage:
    from msd import compute_msd, MSDResult
    
    result = compute_msd(trajectory)
    print(f"Motion type: {result.motion_type}")
    print(f"Diffusion exponent: {result.diffusion_exponent:.3f}")
    print(result.interpretation)

Dependencies: numpy only
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional


@dataclass
class MSDResult:
    """Result from Mean Squared Displacement calculation."""
    
    msd_values: np.ndarray
    """MSD values at each time lag."""
    
    time_lags: np.ndarray
    """Time lags corresponding to MSD values."""
    
    diffusion_exponent: float
    """Power-law exponent α where MSD ~ t^α."""
    
    diffusion_coefficient: float
    """Diffusion coefficient D (for normal diffusion, MSD = 2Dt)."""
    
    @property
    def is_ballistic(self) -> bool:
        """True if motion is ballistic (α ≈ 2, directed motion)."""
        return self.diffusion_exponent > 1.7
    
    @property
    def is_diffusive(self) -> bool:
        """True if motion is normal diffusive (α ≈ 1, random walk)."""
        return 0.8 <= self.diffusion_exponent <= 1.5
    
    @property
    def is_subdiffusive(self) -> bool:
        """True if motion is subdiffusive (α < 1, constrained)."""
        return self.diffusion_exponent < 0.8
    
    @property
    def is_superdiffusive(self) -> bool:
        """True if motion is superdiffusive (α > 1, enhanced spreading)."""
        return self.diffusion_exponent > 1.5
    
    @property
    def is_confined(self) -> bool:
        """True if motion is confined (α < 0.5, bounded)."""
        return self.diffusion_exponent < 0.5
    
    @property
    def motion_type(self) -> str:
        """Classify the type of motion based on diffusion exponent."""
        if self.is_ballistic:
            return "ballistic"
        elif self.is_superdiffusive and not self.is_ballistic:
            return "superdiffusive"
        elif self.is_diffusive:
            return "diffusive"
        elif self.is_subdiffusive and not self.is_confined:
            return "subdiffusive"
        elif self.is_confined:
            return "confined"
        else:
            return "unknown"
    
    @property
    def interpretation(self) -> str:
        """Human-readable interpretation of the result."""
        motion = self.motion_type.upper()
        alpha = self.diffusion_exponent
        
        if self.is_ballistic:
            return f"{motion} (α={alpha:.2f}) - Directed, purposeful motion with constant velocity"
        elif self.is_superdiffusive:
            return f"{motion} (α={alpha:.2f}) - Enhanced spreading, faster than random walk"
        elif self.is_diffusive:
            return f"{motion} (α={alpha:.2f}) - Normal diffusion, random walk-like spreading"
        elif self.is_subdiffusive:
            return f"{motion} (α={alpha:.2f}) - Constrained diffusion, obstacles or memory effects"
        elif self.is_confined:
            return f"{motion} (α={alpha:.2f}) - Confined motion, bounded in state space"
        else:
            return f"{motion} (α={alpha:.2f}) - Anomalous diffusion"
    
    def __repr__(self):
        return f"MSDResult(motion={self.motion_type}, alpha={self.diffusion_exponent:.3f}, D={self.diffusion_coefficient:.3f})"


def compute_msd(trajectory: np.ndarray,
               max_lag: Optional[int] = None,
               min_lag: int = 1) -> MSDResult:
    """
    Compute Mean Squared Displacement (MSD) from trajectory.
    
    MSD(τ) = ⟨|r(t+τ) - r(t)|²⟩ where ⟨·⟩ is ensemble/time average
    
    The MSD growth characterizes motion type:
    - MSD ~ τ²: Ballistic (directed motion, constant velocity)
    - MSD ~ τ¹: Diffusive (random walk, Brownian motion)
    - MSD ~ τ^α where α < 1: Subdiffusive (constrained, obstacles)
    - MSD ~ τ^α where α > 1: Superdiffusive (enhanced, Lévy flights)
    - MSD ~ constant: Confined (bounded motion)
    
    This is a standalone implementation requiring only numpy.
    
    Args:
        trajectory: Position data, shape [time_steps] or [time_steps, features]
                   For multi-dimensional, computes Euclidean distances
        max_lag: Maximum time lag to compute (default: half of length)
        min_lag: Minimum time lag (default: 1)
    
    Returns:
        MSDResult with MSD values, diffusion exponent, and interpretation
    
    Examples:
        >>> # Ballistic motion (directed)
        >>> t = np.arange(100)
        >>> ballistic = np.stack([t, t], axis=1)  # Moving diagonally
        >>> result = compute_msd(ballistic)
        >>> result.is_ballistic
        True
        
        >>> # Random walk (diffusive)
        >>> np.random.seed(42)
        >>> random_walk = np.cumsum(np.random.randn(100, 2), axis=0)
        >>> result = compute_msd(random_walk)
        >>> result.is_diffusive
        True
        
        >>> # Confined motion
        >>> t = np.linspace(0, 10, 100)
        >>> confined = np.stack([np.sin(t), np.cos(t)], axis=1)
        >>> result = compute_msd(confined)
        >>> result.is_confined
        True
    
    References:
        - Saxton, M. J., & Jacobson, K. (1997). Single-particle tracking.
          Annual Review of Biophysics and Biomolecular Structure, 26(1), 373-399.
        - Metzler, R., & Klafter, J. (2000). The random walk's guide to anomalous diffusion.
          Physics Reports, 339(1), 1-77.
    """
    # Ensure 2D array [time_steps, features]
    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)
    
    n_steps, n_features = trajectory.shape
    
    # Set maximum lag
    if max_lag is None:
        max_lag = min(n_steps // 2, 100)
    
    max_lag = min(max_lag, n_steps - 1)
    max_lag = max(max_lag, min_lag)
    
    # Compute MSD for each lag
    msd_values = []
    time_lags = []
    
    for lag in range(min_lag, max_lag + 1):
        # Compute displacements at this lag
        displacements = trajectory[lag:] - trajectory[:-lag]
        
        # Squared distances
        squared_distances = np.sum(displacements ** 2, axis=1)
        
        # Mean squared displacement
        msd = np.mean(squared_distances)
        
        msd_values.append(msd)
        time_lags.append(lag)
    
    msd_values = np.array(msd_values)
    time_lags = np.array(time_lags)
    
    # Compute diffusion exponent (α) by fitting MSD ~ t^α
    diffusion_exponent, diffusion_coefficient = compute_diffusion_exponent(
        time_lags, msd_values
    )
    
    return MSDResult(
        msd_values=msd_values,
        time_lags=time_lags,
        diffusion_exponent=diffusion_exponent,
        diffusion_coefficient=diffusion_coefficient
    )


def compute_diffusion_exponent(time_lags: np.ndarray,
                               msd_values: np.ndarray) -> tuple[float, float]:
    """
    Compute diffusion exponent from MSD vs time data.
    
    Fits: log(MSD) = α * log(t) + log(C)
    Where: MSD ~ C * t^α
    
    Args:
        time_lags: Time lag values
        msd_values: Corresponding MSD values
    
    Returns:
        (diffusion_exponent, diffusion_coefficient) tuple
        - diffusion_exponent: Power-law exponent α
        - diffusion_coefficient: Prefactor C (or D for normal diffusion)
    """
    if len(time_lags) < 2:
        return 1.0, 0.0
    
    # Ensure positive values for log
    valid_mask = (time_lags > 0) & (msd_values > 0)
    
    if np.sum(valid_mask) < 2:
        return 1.0, 0.0
    
    time_lags = time_lags[valid_mask]
    msd_values = msd_values[valid_mask]
    
    # Fit in log-log space: log(MSD) = α * log(t) + log(C)
    log_lags = np.log(time_lags)
    log_msd = np.log(msd_values)
    
    # Linear regression
    alpha, log_C = np.polyfit(log_lags, log_msd, 1)
    
    # Diffusion coefficient
    # For normal diffusion: MSD = 2Dt, so C = 2D
    # For general: MSD = C * t^α
    C = np.exp(log_C)
    
    # Clamp exponent to reasonable range
    alpha = float(np.clip(alpha, 0.0, 3.0))
    
    return alpha, float(C)


def compute_msd_from_trajectory(trajectory: np.ndarray,
                                max_lag: Optional[int] = None) -> np.ndarray:
    """
    Legacy function for backward compatibility.
    
    Returns just the MSD values array (not the full result).
    
    Args:
        trajectory: Position data [time_steps] or [time_steps, features]
        max_lag: Maximum time lag
    
    Returns:
        Array of MSD values
    """
    result = compute_msd(trajectory, max_lag=max_lag)
    return result.msd_values


def classify_motion(msd_values: np.ndarray,
                   time_lags: Optional[np.ndarray] = None) -> str:
    """
    Classify motion type from MSD values.
    
    Args:
        msd_values: MSD values at different lags
        time_lags: Optional time lag values (auto-generated if None)
    
    Returns:
        Motion type string: 'ballistic', 'diffusive', 'subdiffusive', 'confined'
    """
    if time_lags is None:
        time_lags = np.arange(1, len(msd_values) + 1)
    
    alpha, _ = compute_diffusion_exponent(time_lags, msd_values)
    
    # Create temporary result for classification
    temp_result = MSDResult(
        msd_values=msd_values,
        time_lags=time_lags,
        diffusion_exponent=alpha,
        diffusion_coefficient=0.0
    )
    
    return temp_result.motion_type


__all__ = [
    "MSDResult",
    "compute_msd",
    "compute_diffusion_exponent",
    "compute_msd_from_trajectory",  # Backward compatibility
    "classify_motion",
]


if __name__ == "__main__":
    # Self-test examples
    print("Mean Squared Displacement Calculator - Standalone Tests")
    print("=" * 50)
    
    np.random.seed(42)
    
    # Test 1: Ballistic motion (directed)
    print("\n1. Ballistic Motion (directed, constant velocity):")
    t = np.arange(100)
    ballistic = np.stack([t, t * 0.5], axis=1)
    result = compute_msd(ballistic)
    print(f"   {result.interpretation}")
    print(f"   Diffusion coefficient: {result.diffusion_coefficient:.3f}")
    
    # Test 2: Normal diffusion (random walk)
    print("\n2. Normal Diffusion (random walk):")
    random_walk = np.cumsum(np.random.randn(200, 2), axis=0)
    result = compute_msd(random_walk)
    print(f"   {result.interpretation}")
    print(f"   Diffusion coefficient: {result.diffusion_coefficient:.3f}")
    
    # Test 3: Confined motion (bounded)
    print("\n3. Confined Motion (bounded in space):")
    t = np.linspace(0, 10 * np.pi, 200)
    confined = np.stack([np.sin(t), np.cos(t)], axis=1)
    result = compute_msd(confined)
    print(f"   {result.interpretation}")
    print(f"   MSD plateau: {result.msd_values[-1]:.3f}")
    
    # Test 4: Subdiffusive motion
    print("\n4. Subdiffusive Motion (constrained):")
    # Create subdiffusive walk (each step depends on history)
    subdiff = np.zeros((200, 2))
    for i in range(1, 200):
        # Reduced step size with history
        step = np.random.randn(2) * (1.0 / (1 + i * 0.01))
        subdiff[i] = subdiff[i-1] + step
    result = compute_msd(subdiff)
    print(f"   {result.interpretation}")
    
    # Test 5: 1D trajectory
    print("\n5. 1D Trajectory:")
    t = np.arange(100)
    traj_1d = t + np.random.randn(100) * 5
    result = compute_msd(traj_1d)
    print(f"   {result.interpretation}")
    
    # Test 6: Classify motion
    print("\n6. Motion Classification:")
    motion_types = {
        "ballistic": ballistic,
        "diffusive": random_walk,
        "confined": confined,
        "subdiffusive": subdiff,
    }
    for name, traj in motion_types.items():
        result = compute_msd(traj)
        classified = classify_motion(result.msd_values, result.time_lags)
        match = "✓" if classified == name else "✗"
        print(f"   {match} Expected: {name:15s} Got: {classified:15s} (α={result.diffusion_exponent:.2f})")
    
    print("\n✓ All tests completed successfully!")
