"""
Helios Trajectory Analysis Metrics
===================================

Mathematical tools for analyzing dynamical systems in hidden state trajectories.
Ported from helios-trajectory-analysis project.

Key Components:
- Lyapunov exponent: Measure of chaos/sensitivity to initial conditions
- Hurst exponent: Measure of long-term memory and persistence
- MSD (Mean Squared Displacement): Measure of diffusion/spreading
- Signal classification: NOISE, DRIFT, ATTRACTOR, PERIODIC, CHAOTIC
"""

import numpy as np
from enum import Enum
from typing import Tuple, List, Optional
from scipy import signal as scipy_signal


class SignalClass(Enum):
    """Classification of signal behavior."""
    NOISE = "noise"              # Random, no structure
    DRIFT = "drift"              # Slow wandering
    ATTRACTOR = "attractor"      # Converging to fixed point
    PERIODIC = "periodic"        # Regular oscillations
    CHAOTIC = "chaotic"          # Deterministic chaos
    BALLISTIC = "ballistic"      # Directed motion
    DIFFUSIVE = "diffusive"      # Random walk
    CONFINED = "confined"        # Bounded motion


def compute_lyapunov_exponent(trajectory: np.ndarray, 
                              tau: Optional[int] = None,
                              min_tsep: int = 1) -> float:
    """
    Compute the largest Lyapunov exponent using Rosenstein algorithm.
    
    The Lyapunov exponent measures sensitivity to initial conditions:
    - λ > 0: Chaotic/unstable (divergence)
    - λ ≈ 0: Neutral/stable
    - λ < 0: Convergent (attractor)
    
    Args:
        trajectory: Time series data [time_steps, features] or [time_steps]
        tau: Time delay for embedding (auto-computed if None)
        min_tsep: Minimum temporal separation between neighbors
        
    Returns:
        Largest Lyapunov exponent
    """
    # Ensure 2D array
    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)
    
    n_steps, n_features = trajectory.shape
    
    # Auto-compute time delay if not provided
    if tau is None:
        tau = max(1, n_steps // 20)
    
    # Need enough points for analysis
    if n_steps < 10:
        return 0.0
    
    # Compute distances between all pairs of points
    divergences = []
    
    for i in range(n_steps - tau - 1):
        # Find nearest neighbor with temporal separation
        min_dist = np.inf
        min_idx = -1
        
        for j in range(n_steps - tau - 1):
            if abs(i - j) < min_tsep:
                continue
            
            dist = np.linalg.norm(trajectory[i] - trajectory[j])
            if dist < min_dist and dist > 0:
                min_dist = dist
                min_idx = j
        
        if min_idx >= 0:
            # Track divergence over time
            for k in range(1, min(tau, n_steps - max(i, min_idx) - 1)):
                dist_future = np.linalg.norm(
                    trajectory[i + k] - trajectory[min_idx + k]
                )
                if dist_future > 0 and min_dist > 0:
                    divergences.append(np.log(dist_future / min_dist) / k)
    
    if len(divergences) == 0:
        return 0.0
    
    # Average divergence rate
    return float(np.mean(divergences))


def compute_hurst_exponent(trajectory: np.ndarray, 
                          min_window: int = 10) -> float:
    """
    Compute Hurst exponent using R/S (rescaled range) analysis.
    
    The Hurst exponent characterizes long-term memory:
    - H = 0.5: Random walk (no memory)
    - H > 0.5: Persistent (trending)
    - H < 0.5: Anti-persistent (mean-reverting)
    
    Args:
        trajectory: Time series data [time_steps, features] or [time_steps]
        min_window: Minimum window size for analysis
        
    Returns:
        Hurst exponent
    """
    # Flatten if multidimensional
    if trajectory.ndim > 1:
        trajectory = np.linalg.norm(trajectory, axis=1)
    
    n = len(trajectory)
    
    if n < min_window * 2:
        return 0.5  # Default to random walk
    
    # Try different window sizes
    window_sizes = []
    rs_values = []
    
    for window_size in range(min_window, n // 2, max(1, (n // 2 - min_window) // 10)):
        num_windows = n // window_size
        rs_window = []
        
        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            window = trajectory[start:end]
            
            # Mean-adjusted cumulative sum
            mean = np.mean(window)
            cumsum = np.cumsum(window - mean)
            
            # Range
            R = np.max(cumsum) - np.min(cumsum)
            
            # Standard deviation
            S = np.std(window, ddof=1)
            
            if S > 0 and R > 0:
                rs_window.append(R / S)
        
        if rs_window:
            window_sizes.append(window_size)
            rs_values.append(np.mean(rs_window))
    
    if len(window_sizes) < 2:
        return 0.5
    
    # Fit log(R/S) = H * log(n) + c
    log_sizes = np.log(window_sizes)
    log_rs = np.log(rs_values)
    
    # Linear regression
    hurst, _ = np.polyfit(log_sizes, log_rs, 1)
    
    # Clamp to reasonable range
    return float(np.clip(hurst, 0.0, 1.0))


def compute_msd_from_trajectory(trajectory: np.ndarray,
                                max_lag: Optional[int] = None) -> np.ndarray:
    """
    Compute Mean Squared Displacement (MSD) from trajectory.
    
    MSD(τ) = ⟨|r(t+τ) - r(t)|²⟩
    
    MSD behavior indicates:
    - MSD ~ τ¹: Ballistic (directed motion)
    - MSD ~ τ²: Diffusive (random walk)
    - MSD ~ const: Confined (bounded)
    
    Args:
        trajectory: Position data [time_steps, features]
        max_lag: Maximum time lag to compute (default: half length)
        
    Returns:
        MSD values for each time lag [max_lag]
    """
    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)
    
    n_steps = len(trajectory)
    
    if max_lag is None:
        max_lag = min(n_steps // 2, 100)
    
    max_lag = min(max_lag, n_steps - 1)
    
    msd = np.zeros(max_lag)
    
    for lag in range(1, max_lag + 1):
        displacements = trajectory[lag:] - trajectory[:-lag]
        squared_distances = np.sum(displacements ** 2, axis=1)
        msd[lag - 1] = np.mean(squared_distances)
    
    return msd


def classify_trajectory(msd: np.ndarray, threshold_ratio: float = 2.0) -> str:
    """
    Classify trajectory based on MSD behavior.
    
    Args:
        msd: MSD values from compute_msd_from_trajectory
        threshold_ratio: Ratio for classification thresholds
        
    Returns:
        Trajectory classification string
    """
    if len(msd) < 3:
        return SignalClass.NOISE.value
    
    # Fit MSD ~ t^alpha
    lags = np.arange(1, len(msd) + 1)
    log_lags = np.log(lags)
    log_msd = np.log(np.maximum(msd, 1e-10))
    
    # Linear fit in log-log space
    alpha, _ = np.polyfit(log_lags, log_msd, 1)
    
    # Classify based on exponent
    if alpha > 1.5:
        return SignalClass.BALLISTIC.value
    elif alpha > 0.8:
        return SignalClass.DIFFUSIVE.value
    elif alpha < 0.3:
        return SignalClass.CONFINED.value
    else:
        return SignalClass.DRIFT.value


def verify_signal(trajectory: np.ndarray,
                 lyapunov: Optional[float] = None,
                 hurst: Optional[float] = None) -> str:
    """
    Multi-test verification of signal classification.
    
    Combines multiple metrics for robust classification:
    - Lyapunov exponent (chaos)
    - Hurst exponent (memory)
    - MSD analysis (diffusion)
    - Spectral analysis (periodicity)
    
    Args:
        trajectory: Time series data
        lyapunov: Pre-computed Lyapunov exponent (computed if None)
        hurst: Pre-computed Hurst exponent (computed if None)
        
    Returns:
        Signal classification string
    """
    # Flatten if needed
    if trajectory.ndim > 1:
        traj_1d = np.linalg.norm(trajectory, axis=1)
    else:
        traj_1d = trajectory
    
    # Compute metrics if not provided
    if lyapunov is None:
        lyapunov = compute_lyapunov_exponent(trajectory)
    
    if hurst is None:
        hurst = compute_hurst_exponent(traj_1d)
    
    # MSD analysis
    msd = compute_msd_from_trajectory(trajectory)
    traj_class = classify_trajectory(msd)
    
    # Spectral analysis for periodicity
    if len(traj_1d) > 10:
        freqs, power = scipy_signal.periodogram(traj_1d)
        # Check if there's a dominant frequency
        if len(power) > 1:
            max_power = np.max(power[1:])  # Exclude DC
            mean_power = np.mean(power[1:])
            if max_power > 10 * mean_power:
                return SignalClass.PERIODIC.value
    
    # Decision tree based on metrics
    if lyapunov > 0.5:
        return SignalClass.CHAOTIC.value
    elif lyapunov < -0.5 and traj_class == SignalClass.CONFINED.value:
        return SignalClass.ATTRACTOR.value
    elif traj_class == SignalClass.BALLISTIC.value:
        return SignalClass.BALLISTIC.value
    elif hurst < 0.4:
        return SignalClass.NOISE.value
    else:
        return traj_class


def compute_attractor_score(trajectory: np.ndarray, 
                           window_size: int = 10) -> Tuple[float, bool]:
    """
    Compute how strongly trajectory is converging to an attractor.
    
    Args:
        trajectory: Hidden state trajectory
        window_size: Window for computing convergence
        
    Returns:
        (attractor_strength, is_converging) tuple
    """
    if len(trajectory) < window_size:
        return 0.0, False
    
    # Compute variance in windows
    variances = []
    for i in range(0, len(trajectory) - window_size + 1, window_size // 2):
        window = trajectory[i:i + window_size]
        if window.ndim > 1:
            var = np.mean(np.var(window, axis=0))
        else:
            var = np.var(window)
        variances.append(var)
    
    if len(variances) < 2:
        return 0.0, False
    
    # Check if variance is decreasing (converging)
    variance_trend = np.polyfit(range(len(variances)), variances, 1)[0]
    is_converging = variance_trend < -1e-6
    
    # Strength based on how quickly variance decreases
    if is_converging:
        strength = min(1.0, abs(variance_trend) * 100)
    else:
        strength = 0.0
    
    return float(strength), bool(is_converging)


__all__ = [
    "SignalClass",
    "compute_lyapunov_exponent",
    "compute_hurst_exponent",
    "compute_msd_from_trajectory",
    "classify_trajectory",
    "verify_signal",
    "compute_attractor_score",
]
