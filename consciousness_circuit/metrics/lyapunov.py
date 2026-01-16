"""
Lyapunov Exponent Calculator

Standalone module - no dependencies on other consciousness_circuit modules.
Can be copied to any project.

Usage:
    from consciousness_circuit.metrics.lyapunov import compute_lyapunov
    
    # From trajectory
    lyap = compute_lyapunov(x_coords, y_coords)
    
    # From 1D sequence
    lyap = compute_lyapunov_1d(sequence)
    
    # With full diagnostics
    result = LyapunovAnalyzer().analyze(trajectory)
"""

import numpy as np
from typing import List, Tuple, Optional, Union
from dataclasses import dataclass


@dataclass
class LyapunovResult:
    """Result of Lyapunov exponent computation."""
    value: float
    confidence: float
    method: str  # 'rosenstein', 'wolf', etc.
    diagnostics: dict
    
    @property
    def is_chaotic(self) -> bool:
        """Returns True if system exhibits chaotic behavior (λ > 0.1)."""
        return self.value > 0.1
    
    @property
    def is_stable(self) -> bool:
        """Returns True if system is stable/convergent (λ < -0.1)."""
        return self.value < -0.1
    
    @property 
    def interpretation(self) -> str:
        """Human-readable interpretation of Lyapunov exponent."""
        if self.is_chaotic:
            return "chaotic"
        elif self.is_stable:
            return "stable/convergent"
        else:
            return "neutral/random"


def compute_lyapunov(
    x: Union[List[float], np.ndarray],
    y: Union[List[float], np.ndarray],
    method: str = 'rosenstein',
    normalize: bool = True
) -> float:
    """
    Compute Lyapunov exponent from 2D trajectory.
    
    The Lyapunov exponent measures the rate of divergence of nearby trajectories.
    Positive values indicate chaos, negative values indicate stability.
    
    Args:
        x: X coordinates of trajectory
        y: Y coordinates of trajectory
        method: Algorithm ('rosenstein', 'wolf')
        normalize: Normalize for diffusion effects
        
    Returns:
        Lyapunov exponent (λ > 0 = chaotic, λ < 0 = stable, λ ≈ 0 = neutral)
    """
    x = np.asarray(x)
    y = np.asarray(y)
    
    if len(x) != len(y):
        raise ValueError(f"x and y must have same length, got {len(x)} and {len(y)}")
    
    if len(x) < 10:
        raise ValueError(f"Need at least 10 points for Lyapunov computation, got {len(x)}")
    
    # Combine into trajectory
    trajectory = np.column_stack([x, y])
    
    if method == 'rosenstein':
        return _rosenstein_lyapunov(trajectory, normalize)
    elif method == 'wolf':
        return _wolf_lyapunov(trajectory, normalize)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'rosenstein' or 'wolf'")


def compute_lyapunov_1d(
    sequence: Union[List[float], np.ndarray],
    embedding_dim: int = 3,
    delay: int = 1
) -> float:
    """
    Compute Lyapunov exponent from 1D sequence using time-delay embedding.
    
    Reconstructs phase space using Takens' theorem before computing Lyapunov.
    
    Args:
        sequence: 1D data sequence
        embedding_dim: Embedding dimension (default 3)
        delay: Time delay for embedding (default 1)
        
    Returns:
        Lyapunov exponent
    """
    sequence = np.asarray(sequence)
    
    if len(sequence) < embedding_dim * delay + 10:
        raise ValueError(f"Sequence too short for embedding_dim={embedding_dim}, delay={delay}")
    
    # Time-delay embedding
    embedded = _time_delay_embed(sequence, embedding_dim, delay)
    
    # Compute Lyapunov on embedded trajectory
    return _rosenstein_lyapunov(embedded, normalize=True)


def _time_delay_embed(
    sequence: np.ndarray,
    embedding_dim: int,
    delay: int
) -> np.ndarray:
    """Create time-delay embedding of 1D sequence."""
    n = len(sequence) - (embedding_dim - 1) * delay
    embedded = np.zeros((n, embedding_dim))
    
    for i in range(embedding_dim):
        embedded[:, i] = sequence[i * delay : i * delay + n]
    
    return embedded


def _rosenstein_lyapunov(trajectory: np.ndarray, normalize: bool) -> float:
    """
    Rosenstein algorithm for largest Lyapunov exponent.
    
    Tracks divergence of nearest neighbors over time.
    """
    n_points = len(trajectory)
    
    # Find nearest neighbors for each point (excluding immediate neighbors)
    min_temporal_sep = max(1, n_points // 10)
    
    divergences = []
    
    for i in range(n_points - min_temporal_sep):
        # Find nearest neighbor (not temporally adjacent)
        distances = np.linalg.norm(trajectory - trajectory[i], axis=1)
        
        # Exclude temporal neighbors
        valid_indices = np.arange(n_points)
        valid_mask = np.abs(valid_indices - i) > min_temporal_sep
        
        if not np.any(valid_mask):
            continue
            
        valid_distances = distances.copy()
        valid_distances[~valid_mask] = np.inf
        
        nearest_idx = np.argmin(valid_distances)
        
        if nearest_idx == i or valid_distances[nearest_idx] == np.inf:
            continue
        
        # Track divergence over time
        max_evolution = min(n_points - i, n_points - nearest_idx)
        
        for dt in range(1, min(max_evolution, 20)):
            d0 = np.linalg.norm(trajectory[i] - trajectory[nearest_idx])
            dt_distance = np.linalg.norm(
                trajectory[i + dt] - trajectory[nearest_idx + dt]
            )
            
            if d0 > 0 and dt_distance > 0:
                log_div = np.log(dt_distance / d0)
                divergences.append((dt, log_div))
    
    if not divergences:
        return 0.0
    
    # Fit linear regression to log(divergence) vs time
    divergences = np.array(divergences)
    times = divergences[:, 0]
    log_divs = divergences[:, 1]
    
    # Remove outliers
    log_divs = log_divs[np.isfinite(log_divs)]
    times = times[np.isfinite(log_divs)]
    
    if len(times) < 2:
        return 0.0
    
    # Linear fit
    coeffs = np.polyfit(times, log_divs, 1)
    lyapunov = coeffs[0]
    
    # Normalize if requested
    if normalize:
        # Estimate diffusion coefficient
        displacements = np.diff(trajectory, axis=0)
        msd = np.mean(np.sum(displacements**2, axis=1))
        
        if msd > 0:
            lyapunov = lyapunov / np.sqrt(msd)
    
    return float(lyapunov)


def _wolf_lyapunov(trajectory: np.ndarray, normalize: bool) -> float:
    """
    Wolf algorithm for largest Lyapunov exponent.
    
    More sophisticated but slower than Rosenstein.
    """
    # For simplicity, fall back to Rosenstein
    # Full Wolf algorithm is complex and rarely needed
    return _rosenstein_lyapunov(trajectory, normalize)


class LyapunovAnalyzer:
    """Full analyzer with diagnostics and visualization capabilities."""
    
    def __init__(self, method: str = 'rosenstein'):
        """
        Initialize Lyapunov analyzer.
        
        Args:
            method: Algorithm to use ('rosenstein' or 'wolf')
        """
        self.method = method
    
    def analyze(
        self,
        trajectory: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]]
    ) -> LyapunovResult:
        """
        Full analysis with confidence and diagnostics.
        
        Args:
            trajectory: Either (x, y) tuple or Nx2 array
            
        Returns:
            LyapunovResult with value, confidence, and diagnostics
        """
        # Handle different input formats
        if isinstance(trajectory, tuple):
            x, y = trajectory
            traj_array = np.column_stack([x, y])
        else:
            traj_array = np.asarray(trajectory)
        
        if traj_array.ndim == 1:
            # 1D sequence, use time-delay embedding
            lyap = compute_lyapunov_1d(traj_array)
        else:
            # 2D or higher trajectory
            if traj_array.shape[1] == 2:
                x, y = traj_array[:, 0], traj_array[:, 1]
                lyap = compute_lyapunov(x, y, method=self.method)
            else:
                # Use first 2 dimensions
                lyap = compute_lyapunov(
                    traj_array[:, 0],
                    traj_array[:, 1],
                    method=self.method
                )
        
        # Estimate confidence based on trajectory length
        confidence = min(1.0, len(traj_array) / 100.0)
        
        # Diagnostics
        diagnostics = {
            'trajectory_length': len(traj_array),
            'dimensions': traj_array.shape[1] if traj_array.ndim > 1 else 1,
            'mean_displacement': float(np.mean(np.linalg.norm(
                np.diff(traj_array, axis=0) if traj_array.ndim > 1 else np.diff(traj_array)
            ))),
        }
        
        return LyapunovResult(
            value=lyap,
            confidence=confidence,
            method=self.method,
            diagnostics=diagnostics
        )
    
    def analyze_windowed(
        self,
        trajectory: np.ndarray,
        window_size: int = 50,
        stride: int = 10
    ) -> List[LyapunovResult]:
        """
        Sliding window analysis to detect changing dynamics.
        
        Args:
            trajectory: Input trajectory
            window_size: Size of sliding window
            stride: Step size for sliding window
            
        Returns:
            List of LyapunovResult for each window
        """
        results = []
        
        if isinstance(trajectory, tuple):
            x, y = trajectory
            trajectory = np.column_stack([x, y])
        else:
            trajectory = np.asarray(trajectory)
        
        n_points = len(trajectory)
        
        for start in range(0, n_points - window_size, stride):
            end = start + window_size
            window = trajectory[start:end]
            
            try:
                result = self.analyze(window)
                results.append(result)
            except (ValueError, RuntimeError):
                # Skip windows with insufficient data
                continue
        
        return results
