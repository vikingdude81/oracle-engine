"""
Hurst Exponent Calculator

Standalone module - no dependencies on other consciousness_circuit modules.

Usage:
    from consciousness_circuit.metrics.hurst import compute_hurst
    
    h = compute_hurst(sequence)  # H=0.5 random, H>0.5 trending, H<0.5 mean-reverting
"""

import numpy as np
from typing import List, Union, Optional
from dataclasses import dataclass


@dataclass
class HurstResult:
    """Result of Hurst exponent computation."""
    value: float
    confidence: float
    method: str  # 'rs', 'dfa', 'wavelet'
    
    @property
    def is_trending(self) -> bool:
        """Returns True if series shows trending/persistent behavior (H > 0.55)."""
        return self.value > 0.55
    
    @property
    def is_mean_reverting(self) -> bool:
        """Returns True if series is mean-reverting/anti-persistent (H < 0.45)."""
        return self.value < 0.45
    
    @property
    def has_memory(self) -> bool:
        """Returns True if series has long-term memory (H significantly != 0.5)."""
        return abs(self.value - 0.5) > 0.1
    
    @property
    def interpretation(self) -> str:
        """Human-readable interpretation of Hurst exponent."""
        if self.is_trending:
            return "trending/persistent"
        elif self.is_mean_reverting:
            return "mean-reverting/anti-persistent"
        else:
            return "random walk"


def compute_hurst(
    sequence: Union[List[float], np.ndarray],
    method: str = 'rs',
    min_window: int = 10
) -> float:
    """
    Compute Hurst exponent using specified method.
    
    The Hurst exponent measures long-term memory in time series:
    - H = 0.5: Random walk (no memory)
    - H > 0.5: Trending/persistent (positive autocorrelation)
    - H < 0.5: Mean-reverting (negative autocorrelation)
    
    Args:
        sequence: 1D data sequence
        method: 'rs' (rescaled range), 'dfa' (detrended fluctuation), or 'wavelet'
        min_window: Minimum window size for analysis
        
    Returns:
        Hurst exponent (0 to 1)
    """
    sequence = np.asarray(sequence)
    
    if len(sequence) < 2 * min_window:
        raise ValueError(f"Sequence too short: need at least {2 * min_window} points")
    
    if method == 'rs':
        return _rs_hurst(sequence, min_window)
    elif method == 'dfa':
        return _dfa_hurst(sequence, min_window)
    elif method == 'wavelet':
        return _wavelet_hurst(sequence, min_window)
    else:
        raise ValueError(f"Unknown method: {method}. Use 'rs', 'dfa', or 'wavelet'")


def _rs_hurst(sequence: np.ndarray, min_window: int) -> float:
    """
    R/S analysis (Rescaled Range) method for Hurst exponent.
    
    Classic method by Hurst (1951).
    """
    n = len(sequence)
    
    # Use logarithmically spaced window sizes
    max_window = n // 4
    n_windows = min(20, int(np.log2(max_window / min_window)) + 1)
    
    if n_windows < 2:
        # Not enough data for multiple windows
        return 0.5
    
    window_sizes = np.logspace(
        np.log10(min_window),
        np.log10(max_window),
        n_windows
    ).astype(int)
    
    window_sizes = np.unique(window_sizes)
    
    rs_values = []
    
    for window_size in window_sizes:
        if window_size >= n:
            continue
            
        # Split sequence into non-overlapping windows
        n_segments = n // window_size
        
        rs_segment = []
        
        for i in range(n_segments):
            segment = sequence[i * window_size : (i + 1) * window_size]
            
            # Mean-center the segment
            mean = np.mean(segment)
            y = segment - mean
            
            # Cumulative sum
            z = np.cumsum(y)
            
            # Range
            r = np.max(z) - np.min(z)
            
            # Standard deviation
            s = np.std(segment, ddof=1)
            
            if s > 0:
                rs_segment.append(r / s)
        
        if rs_segment:
            rs_values.append((window_size, np.mean(rs_segment)))
    
    if len(rs_values) < 2:
        return 0.5
    
    # Fit log(R/S) vs log(n) to get Hurst exponent
    window_sizes = np.array([w for w, _ in rs_values])
    rs_vals = np.array([rs for _, rs in rs_values])
    
    # Remove zeros and invalid values
    valid = (rs_vals > 0) & np.isfinite(rs_vals)
    
    if np.sum(valid) < 2:
        return 0.5
    
    log_sizes = np.log(window_sizes[valid])
    log_rs = np.log(rs_vals[valid])
    
    # Linear regression
    coeffs = np.polyfit(log_sizes, log_rs, 1)
    hurst = coeffs[0]
    
    # Clamp to valid range
    return float(np.clip(hurst, 0.0, 1.0))


def _dfa_hurst(sequence: np.ndarray, min_window: int) -> float:
    """
    Detrended Fluctuation Analysis (DFA) method for Hurst exponent.
    
    More robust than R/S for non-stationary data.
    """
    n = len(sequence)
    
    # Cumulative sum (integration)
    y = np.cumsum(sequence - np.mean(sequence))
    
    # Window sizes
    max_window = n // 4
    n_windows = min(15, int(np.log2(max_window / min_window)) + 1)
    
    if n_windows < 2:
        return 0.5
    
    window_sizes = np.logspace(
        np.log10(min_window),
        np.log10(max_window),
        n_windows
    ).astype(int)
    
    window_sizes = np.unique(window_sizes)
    
    fluctuations = []
    
    for window_size in window_sizes:
        if window_size >= n:
            continue
        
        # Split into non-overlapping segments
        n_segments = n // window_size
        
        segment_fluctuations = []
        
        for i in range(n_segments):
            segment = y[i * window_size : (i + 1) * window_size]
            
            # Fit polynomial trend (order 1 = linear detrending)
            x = np.arange(len(segment))
            coeffs = np.polyfit(x, segment, 1)
            trend = np.polyval(coeffs, x)
            
            # Detrended fluctuation
            detrended = segment - trend
            f = np.sqrt(np.mean(detrended ** 2))
            
            segment_fluctuations.append(f)
        
        if segment_fluctuations:
            fluctuations.append((window_size, np.mean(segment_fluctuations)))
    
    if len(fluctuations) < 2:
        return 0.5
    
    # Fit log(F) vs log(n) to get scaling exponent
    window_sizes = np.array([w for w, _ in fluctuations])
    f_vals = np.array([f for _, f in fluctuations])
    
    valid = (f_vals > 0) & np.isfinite(f_vals)
    
    if np.sum(valid) < 2:
        return 0.5
    
    log_sizes = np.log(window_sizes[valid])
    log_f = np.log(f_vals[valid])
    
    # Linear regression
    coeffs = np.polyfit(log_sizes, log_f, 1)
    alpha = coeffs[0]
    
    # DFA exponent alpha relates to Hurst exponent H
    # For fractional Brownian motion: H = alpha
    hurst = alpha
    
    return float(np.clip(hurst, 0.0, 1.0))


def _wavelet_hurst(sequence: np.ndarray, min_window: int) -> float:
    """
    Wavelet-based Hurst exponent estimation.
    
    Uses wavelet variance at different scales.
    For simplicity, falls back to DFA in this implementation.
    """
    # Full wavelet transform requires scipy.signal
    # For standalone version, use DFA as approximation
    return _dfa_hurst(sequence, min_window)


class HurstAnalyzer:
    """Full analyzer with multiple methods and diagnostics."""
    
    def __init__(self, method: str = 'rs'):
        """
        Initialize Hurst analyzer.
        
        Args:
            method: 'rs' (rescaled range), 'dfa', or 'wavelet'
        """
        self.method = method
    
    def analyze(
        self,
        sequence: Union[List[float], np.ndarray],
        min_window: Optional[int] = None
    ) -> HurstResult:
        """
        Full analysis with confidence estimation.
        
        Args:
            sequence: Input time series
            min_window: Minimum window size (default: auto)
            
        Returns:
            HurstResult with value, confidence, and method
        """
        sequence = np.asarray(sequence)
        
        if min_window is None:
            min_window = max(10, len(sequence) // 20)
        
        hurst = compute_hurst(sequence, method=self.method, min_window=min_window)
        
        # Estimate confidence based on sequence length
        confidence = min(1.0, len(sequence) / 200.0)
        
        return HurstResult(
            value=hurst,
            confidence=confidence,
            method=self.method
        )
    
    def analyze_multi_method(
        self,
        sequence: Union[List[float], np.ndarray]
    ) -> dict:
        """
        Analyze with multiple methods for robustness.
        
        Args:
            sequence: Input time series
            
        Returns:
            Dictionary with results from each method
        """
        results = {}
        
        for method in ['rs', 'dfa']:
            try:
                analyzer = HurstAnalyzer(method=method)
                result = analyzer.analyze(sequence)
                results[method] = result
            except (ValueError, RuntimeError):
                continue
        
        return results
    
    def analyze_windowed(
        self,
        sequence: Union[List[float], np.ndarray],
        window_size: int = 100,
        stride: int = 20
    ) -> List[HurstResult]:
        """
        Sliding window analysis to detect changing persistence.
        
        Args:
            sequence: Input time series
            window_size: Size of sliding window
            stride: Step size for sliding window
            
        Returns:
            List of HurstResult for each window
        """
        sequence = np.asarray(sequence)
        results = []
        
        n = len(sequence)
        
        for start in range(0, n - window_size, stride):
            end = start + window_size
            window = sequence[start:end]
            
            try:
                result = self.analyze(window)
                results.append(result)
            except (ValueError, RuntimeError):
                continue
        
        return results
