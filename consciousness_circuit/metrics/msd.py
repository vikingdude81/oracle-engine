"""
Mean Squared Displacement Calculator

Standalone module for diffusion analysis.

Usage:
    from consciousness_circuit.metrics.msd import compute_msd, compute_diffusion_exponent
    
    lags, msd_values = compute_msd(x, y)
    alpha = compute_diffusion_exponent(x, y)  # α=1 normal, α>1 super, α<1 sub
"""

import numpy as np
from typing import List, Tuple, Union, Optional
from dataclasses import dataclass


@dataclass
class MSDResult:
    """Result of mean squared displacement analysis."""
    lags: np.ndarray
    msd_values: np.ndarray
    diffusion_exponent: float  # alpha
    diffusion_coefficient: float  # D
    
    @property
    def is_ballistic(self) -> bool:
        """Returns True for ballistic motion (α > 1.5)."""
        return self.diffusion_exponent > 1.5
    
    @property
    def is_subdiffusive(self) -> bool:
        """Returns True for subdiffusive motion (α < 0.8)."""
        return self.diffusion_exponent < 0.8
    
    @property
    def is_normal_diffusion(self) -> bool:
        """Returns True for normal/Brownian diffusion (0.8 < α < 1.2)."""
        return 0.8 <= self.diffusion_exponent <= 1.2
    
    @property
    def motion_type(self) -> str:
        """Human-readable classification of motion type."""
        if self.diffusion_exponent > 1.8:
            return "ballistic"
        elif self.diffusion_exponent > 1.2:
            return "superdiffusive"
        elif self.diffusion_exponent > 0.8:
            return "normal"
        else:
            return "subdiffusive"


def compute_msd(
    x: Union[List[float], np.ndarray],
    y: Union[List[float], np.ndarray],
    max_lag: Optional[int] = None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Compute mean squared displacement at various lag times.
    
    MSD(τ) = ⟨|r(t+τ) - r(t)|²⟩
    
    Args:
        x: X coordinates of trajectory
        y: Y coordinates of trajectory
        max_lag: Maximum lag time (default: len(trajectory) // 4)
        
    Returns:
        Tuple of (lags, msd_values)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    if len(x) != len(y):
        raise ValueError(f"x and y must have same length, got {len(x)} and {len(y)}")
    
    n = len(x)
    
    if n < 10:
        raise ValueError(f"Need at least 10 points for MSD, got {n}")
    
    if max_lag is None:
        max_lag = min(n // 4, 100)
    
    max_lag = min(max_lag, n - 1)
    
    # Compute MSD for each lag
    lags = np.arange(1, max_lag + 1)
    msd_values = np.zeros(len(lags))
    
    for i, lag in enumerate(lags):
        # Squared displacements at this lag
        dx = x[lag:] - x[:-lag]
        dy = y[lag:] - y[:-lag]
        squared_displacements = dx**2 + dy**2
        
        # Mean over all time origins
        msd_values[i] = np.mean(squared_displacements)
    
    return lags, msd_values


def compute_diffusion_exponent(
    x: Union[List[float], np.ndarray],
    y: Union[List[float], np.ndarray],
    max_lag: Optional[int] = None
) -> float:
    """
    Compute diffusion exponent α from MSD ~ t^α.
    
    α = 1: Normal diffusion (Brownian motion)
    α > 1: Superdiffusion (e.g., ballistic motion with α=2)
    α < 1: Subdiffusion (e.g., confined motion)
    
    Args:
        x: X coordinates of trajectory
        y: Y coordinates of trajectory
        max_lag: Maximum lag time for fitting
        
    Returns:
        Diffusion exponent α
    """
    lags, msd_values = compute_msd(x, y, max_lag)
    
    # Fit log(MSD) ~ α * log(lag)
    # Use only first portion to avoid finite-size effects
    fit_range = min(len(lags), max(10, len(lags) // 2))
    
    log_lags = np.log(lags[:fit_range])
    log_msd = np.log(msd_values[:fit_range])
    
    # Remove invalid values
    valid = np.isfinite(log_msd) & (msd_values[:fit_range] > 0)
    
    if np.sum(valid) < 2:
        return 1.0  # Default to normal diffusion
    
    # Linear fit
    coeffs = np.polyfit(log_lags[valid], log_msd[valid], 1)
    alpha = coeffs[0]
    
    # Clamp to reasonable range
    return float(np.clip(alpha, 0.0, 3.0))


def compute_diffusion_coefficient(
    x: Union[List[float], np.ndarray],
    y: Union[List[float], np.ndarray]
) -> float:
    """
    Compute diffusion coefficient D from MSD = 4Dt (for 2D normal diffusion).
    
    Args:
        x: X coordinates of trajectory
        y: Y coordinates of trajectory
        
    Returns:
        Diffusion coefficient D
    """
    lags, msd_values = compute_msd(x, y)
    
    # Use early time points (first 10 lags or 20% of data)
    n_fit = min(10, max(3, len(lags) // 5))
    
    # Fit MSD = 4D * t
    coeffs = np.polyfit(lags[:n_fit], msd_values[:n_fit], 1)
    slope = coeffs[0]
    
    # D = slope / 4 (for 2D)
    D = slope / 4.0
    
    return float(max(0.0, D))


def analyze_msd(
    x: Union[List[float], np.ndarray],
    y: Union[List[float], np.ndarray],
    max_lag: Optional[int] = None
) -> MSDResult:
    """
    Complete MSD analysis with all metrics.
    
    Args:
        x: X coordinates of trajectory
        y: Y coordinates of trajectory
        max_lag: Maximum lag time
        
    Returns:
        MSDResult with lags, MSD values, and derived metrics
    """
    lags, msd_values = compute_msd(x, y, max_lag)
    alpha = compute_diffusion_exponent(x, y, max_lag)
    D = compute_diffusion_coefficient(x, y)
    
    return MSDResult(
        lags=lags,
        msd_values=msd_values,
        diffusion_exponent=alpha,
        diffusion_coefficient=D
    )


class MSDAnalyzer:
    """Analyzer for mean squared displacement with multiple utilities."""
    
    def __init__(self, max_lag: Optional[int] = None):
        """
        Initialize MSD analyzer.
        
        Args:
            max_lag: Maximum lag time (default: auto)
        """
        self.max_lag = max_lag
    
    def analyze(
        self,
        trajectory: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
    ) -> MSDResult:
        """
        Analyze trajectory MSD.
        
        Args:
            trajectory: Either (x, y) tuple or Nx2 array
            
        Returns:
            MSDResult with complete analysis
        """
        # Handle different input formats
        if isinstance(trajectory, tuple):
            x, y = trajectory
        else:
            trajectory = np.asarray(trajectory)
            if trajectory.ndim == 1:
                raise ValueError("Need 2D trajectory for MSD analysis")
            x, y = trajectory[:, 0], trajectory[:, 1]
        
        return analyze_msd(x, y, self.max_lag)
    
    def analyze_ensemble(
        self,
        trajectories: List[Tuple[np.ndarray, np.ndarray]]
    ) -> MSDResult:
        """
        Analyze ensemble of trajectories.
        
        Computes ensemble-averaged MSD.
        
        Args:
            trajectories: List of (x, y) trajectory tuples
            
        Returns:
            MSDResult with ensemble-averaged metrics
        """
        if not trajectories:
            raise ValueError("Need at least one trajectory")
        
        # Find common max_lag
        min_length = min(len(x) for x, _ in trajectories)
        max_lag = self.max_lag or min_length // 4
        max_lag = min(max_lag, min_length - 1)
        
        # Compute MSD for each trajectory
        all_lags = []
        all_msd = []
        
        for x, y in trajectories:
            lags, msd_values = compute_msd(x, y, max_lag)
            all_lags.append(lags)
            all_msd.append(msd_values)
        
        # Ensemble average
        lags = all_lags[0]  # All should be the same
        msd_ensemble = np.mean(all_msd, axis=0)
        
        # Fit for exponent and coefficient
        fit_range = min(len(lags), max(10, len(lags) // 2))
        log_lags = np.log(lags[:fit_range])
        log_msd = np.log(msd_ensemble[:fit_range])
        
        valid = np.isfinite(log_msd) & (msd_ensemble[:fit_range] > 0)
        
        if np.sum(valid) >= 2:
            coeffs = np.polyfit(log_lags[valid], log_msd[valid], 1)
            alpha = float(np.clip(coeffs[0], 0.0, 3.0))
        else:
            alpha = 1.0
        
        # Diffusion coefficient
        n_fit = min(10, max(3, len(lags) // 5))
        coeffs = np.polyfit(lags[:n_fit], msd_ensemble[:n_fit], 1)
        D = float(max(0.0, coeffs[0] / 4.0))
        
        return MSDResult(
            lags=lags,
            msd_values=msd_ensemble,
            diffusion_exponent=alpha,
            diffusion_coefficient=D
        )
    
    def classify_motion(self, result: MSDResult) -> str:
        """
        Classify motion type from MSD result.
        
        Args:
            result: MSDResult from analysis
            
        Returns:
            Motion type classification
        """
        return result.motion_type
