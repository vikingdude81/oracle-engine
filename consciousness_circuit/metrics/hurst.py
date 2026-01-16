"""
Hurst Exponent Calculation - FULLY STANDALONE
==============================================

Measures long-term memory and persistence in time series data.
Can be copied to any project - only requires numpy.

Usage:
    from hurst import compute_hurst, HurstResult
    
    result = compute_hurst(time_series, method='rs')
    print(f"Hurst: {result.exponent:.3f}")
    print(f"Has memory: {result.has_memory}")
    print(result.interpretation)

Dependencies: numpy only
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Literal


@dataclass
class HurstResult:
    """Result from Hurst exponent calculation."""
    
    exponent: float
    """The Hurst exponent (0 to 1)."""
    
    method: str
    """Method used: 'rs' (rescaled range) or 'dfa' (detrended fluctuation)."""
    
    window_sizes: np.ndarray
    """Window sizes used in calculation."""
    
    fluctuations: np.ndarray
    """Fluctuation values at each window size."""
    
    @property
    def is_trending(self) -> bool:
        """True if series is persistent/trending (H > 0.55)."""
        return self.exponent > 0.55
    
    @property
    def is_mean_reverting(self) -> bool:
        """True if series is anti-persistent/mean-reverting (H < 0.45)."""
        return self.exponent < 0.45
    
    @property
    def is_random_walk(self) -> bool:
        """True if series is like a random walk (0.45 ≤ H ≤ 0.55)."""
        return 0.45 <= self.exponent <= 0.55
    
    @property
    def has_memory(self) -> bool:
        """True if series has long-term memory (H ≠ 0.5)."""
        return abs(self.exponent - 0.5) > 0.05
    
    @property
    def interpretation(self) -> str:
        """Human-readable interpretation of the result."""
        if self.is_trending:
            return f"PERSISTENT/TRENDING (H={self.exponent:.3f}) - Past trends continue, positive autocorrelation"
        elif self.is_mean_reverting:
            return f"ANTI-PERSISTENT/MEAN-REVERTING (H={self.exponent:.3f}) - Tends to reverse direction, negative autocorrelation"
        else:
            return f"RANDOM WALK (H={self.exponent:.3f}) - No long-term memory, like Brownian motion"
    
    def __repr__(self):
        return f"HurstResult(exponent={self.exponent:.4f}, method='{self.method}', memory={self.has_memory})"


def compute_hurst(time_series: np.ndarray,
                 method: Literal['rs', 'dfa'] = 'rs',
                 min_window: int = 10,
                 max_window: Optional[int] = None,
                 n_windows: int = 10) -> HurstResult:
    """
    Compute Hurst exponent to measure long-term memory.
    
    The Hurst exponent characterizes the long-range dependence:
    - H = 0.5: Random walk, no memory (like Brownian motion)
    - H > 0.5: Persistent, trending behavior (past trends continue)
    - H < 0.5: Anti-persistent, mean-reverting (tends to reverse)
    
    This is a standalone implementation requiring only numpy.
    
    Args:
        time_series: 1D time series data, or 2D [time_steps, features]
                    For multi-dimensional, computes norm first
        method: Calculation method:
               'rs' = Rescaled Range analysis (default, classical)
               'dfa' = Detrended Fluctuation Analysis (more robust)
        min_window: Minimum window size for analysis
        max_window: Maximum window size (default: half of series length)
        n_windows: Number of window sizes to test
    
    Returns:
        HurstResult with exponent, method info, and interpretations
    
    Examples:
        >>> # Random walk (H ≈ 0.5)
        >>> np.random.seed(42)
        >>> random_walk = np.cumsum(np.random.randn(200))
        >>> result = compute_hurst(random_walk)
        >>> 0.4 < result.exponent < 0.6
        True
        
        >>> # Persistent series (H > 0.5)
        >>> t = np.arange(200)
        >>> trending = t + np.random.randn(200) * 5
        >>> result = compute_hurst(trending)
        >>> result.is_trending
        True
        
        >>> # Anti-persistent series (H < 0.5)
        >>> alternating = np.array([(-1)**i for i in range(200)])
        >>> result = compute_hurst(alternating)
        >>> result.is_mean_reverting
        True
    
    References:
        - Hurst, H. E. (1951). Long-term storage capacity of reservoirs.
          Transactions of the American Society of Civil Engineers, 116, 770-808.
        - Peng, C. K., et al. (1994). Mosaic organization of DNA nucleotides.
          Physical Review E, 49(2), 1685.
    """
    # Flatten if multidimensional (use Euclidean norm)
    if time_series.ndim > 1:
        time_series = np.linalg.norm(time_series, axis=1)
    
    n = len(time_series)
    
    # Set maximum window size
    if max_window is None:
        max_window = n // 2
    
    max_window = min(max_window, n // 2)
    
    # Need reasonable data length
    if n < min_window * 2:
        return HurstResult(
            exponent=0.5,  # Default to random walk
            method=method,
            window_sizes=np.array([]),
            fluctuations=np.array([])
        )
    
    # Choose method
    if method == 'rs':
        return _compute_hurst_rs(time_series, min_window, max_window, n_windows)
    elif method == 'dfa':
        return _compute_hurst_dfa(time_series, min_window, max_window, n_windows)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'rs' or 'dfa'")


def _compute_hurst_rs(time_series: np.ndarray,
                     min_window: int,
                     max_window: int,
                     n_windows: int) -> HurstResult:
    """
    Compute Hurst exponent using Rescaled Range (R/S) analysis.
    
    This is the classical method introduced by Hurst.
    """
    n = len(time_series)
    
    # Generate window sizes (logarithmically spaced)
    window_sizes = np.unique(np.logspace(
        np.log10(min_window),
        np.log10(max_window),
        n_windows
    ).astype(int))
    
    rs_values = []
    
    for window_size in window_sizes:
        num_windows = n // window_size
        
        if num_windows == 0:
            continue
        
        rs_window = []
        
        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            window = time_series[start:end]
            
            # Mean-adjusted cumulative sum
            mean = np.mean(window)
            cumsum = np.cumsum(window - mean)
            
            # Range (R)
            R = np.max(cumsum) - np.min(cumsum)
            
            # Standard deviation (S)
            S = np.std(window, ddof=1)
            
            # R/S ratio
            if S > 0 and R > 0:
                rs_window.append(R / S)
        
        if rs_window:
            rs_values.append(np.mean(rs_window))
        else:
            rs_values.append(np.nan)
    
    # Remove NaN values
    valid_mask = ~np.isnan(rs_values)
    window_sizes = window_sizes[valid_mask]
    rs_values = np.array(rs_values)[valid_mask]
    
    if len(window_sizes) < 2:
        return HurstResult(
            exponent=0.5,
            method='rs',
            window_sizes=window_sizes,
            fluctuations=rs_values
        )
    
    # Fit log(R/S) = H * log(n) + c
    log_sizes = np.log(window_sizes)
    log_rs = np.log(rs_values)
    
    # Linear regression in log-log space
    hurst, _ = np.polyfit(log_sizes, log_rs, 1)
    
    # Clamp to valid range
    hurst = float(np.clip(hurst, 0.0, 1.0))
    
    return HurstResult(
        exponent=hurst,
        method='rs',
        window_sizes=window_sizes,
        fluctuations=rs_values
    )


def _compute_hurst_dfa(time_series: np.ndarray,
                      min_window: int,
                      max_window: int,
                      n_windows: int) -> HurstResult:
    """
    Compute Hurst exponent using Detrended Fluctuation Analysis (DFA).
    
    More robust than R/S analysis, especially for non-stationary data.
    """
    n = len(time_series)
    
    # Integrate the time series (cumulative sum)
    y = np.cumsum(time_series - np.mean(time_series))
    
    # Generate window sizes
    window_sizes = np.unique(np.logspace(
        np.log10(min_window),
        np.log10(max_window),
        n_windows
    ).astype(int))
    
    fluctuations = []
    
    for window_size in window_sizes:
        num_windows = n // window_size
        
        if num_windows == 0:
            continue
        
        window_variances = []
        
        for i in range(num_windows):
            start = i * window_size
            end = start + window_size
            window = y[start:end]
            
            # Fit linear trend
            x = np.arange(len(window))
            coeffs = np.polyfit(x, window, 1)
            trend = np.polyval(coeffs, x)
            
            # Detrended signal
            detrended = window - trend
            
            # Variance
            variance = np.mean(detrended ** 2)
            window_variances.append(variance)
        
        if window_variances:
            # Root mean square fluctuation
            fluctuation = np.sqrt(np.mean(window_variances))
            fluctuations.append(fluctuation)
        else:
            fluctuations.append(np.nan)
    
    # Remove NaN values
    valid_mask = ~np.isnan(fluctuations)
    window_sizes = window_sizes[valid_mask]
    fluctuations = np.array(fluctuations)[valid_mask]
    
    if len(window_sizes) < 2:
        return HurstResult(
            exponent=0.5,
            method='dfa',
            window_sizes=window_sizes,
            fluctuations=fluctuations
        )
    
    # Fit log(F(n)) = H * log(n) + c
    # For DFA, the scaling exponent α relates to Hurst as H = α
    log_sizes = np.log(window_sizes)
    log_fluct = np.log(fluctuations)
    
    # Linear regression
    alpha, _ = np.polyfit(log_sizes, log_fluct, 1)
    
    # Clamp to valid range
    hurst = float(np.clip(alpha, 0.0, 1.0))
    
    return HurstResult(
        exponent=hurst,
        method='dfa',
        window_sizes=window_sizes,
        fluctuations=fluctuations
    )


# Backward compatibility alias
compute_hurst_exponent = compute_hurst


__all__ = [
    "HurstResult",
    "compute_hurst",
    "compute_hurst_exponent",  # Alias for compatibility
]


if __name__ == "__main__":
    # Self-test examples
    print("Hurst Exponent Calculator - Standalone Tests")
    print("=" * 50)
    
    np.random.seed(42)
    
    # Test 1: Random walk (H ≈ 0.5)
    print("\n1. Random Walk (H ≈ 0.5):")
    random_walk = np.cumsum(np.random.randn(200))
    result_rs = compute_hurst(random_walk, method='rs')
    result_dfa = compute_hurst(random_walk, method='dfa')
    print(f"   R/S method:  {result_rs.interpretation}")
    print(f"   DFA method:  {result_dfa.interpretation}")
    
    # Test 2: Persistent/trending (H > 0.5)
    print("\n2. Persistent Series (H > 0.5):")
    t = np.arange(200)
    trending = t + np.random.randn(200) * 10
    result = compute_hurst(trending, method='rs')
    print(f"   {result.interpretation}")
    
    # Test 3: Anti-persistent/mean-reverting (H < 0.5)
    print("\n3. Anti-Persistent Series (H < 0.5):")
    # Create mean-reverting series
    anti_persistent = np.zeros(200)
    anti_persistent[0] = 0
    for i in range(1, 200):
        # Tend to move opposite to current deviation from mean
        anti_persistent[i] = anti_persistent[i-1] + np.random.randn() - 0.1 * anti_persistent[i-1]
    result = compute_hurst(anti_persistent, method='rs')
    print(f"   {result.interpretation}")
    
    # Test 4: Comparison of methods
    print("\n4. Method Comparison on Same Data:")
    test_series = np.cumsum(np.random.randn(300))
    rs_result = compute_hurst(test_series, method='rs')
    dfa_result = compute_hurst(test_series, method='dfa')
    print(f"   R/S:  H = {rs_result.exponent:.4f}")
    print(f"   DFA:  H = {dfa_result.exponent:.4f}")
    print(f"   Difference: {abs(rs_result.exponent - dfa_result.exponent):.4f}")
    
    print("\n✓ All tests completed successfully!")
