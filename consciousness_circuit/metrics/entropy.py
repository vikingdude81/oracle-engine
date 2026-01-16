"""
Entropy and Randomness Metrics

Standalone module for spectral entropy, runs test, and autocorrelation.

Usage:
    from consciousness_circuit.metrics.entropy import (
        compute_spectral_entropy,
        compute_runs_test,
        compute_autocorrelation
    )
    
    entropy = compute_spectral_entropy(sequence)
    is_random = compute_runs_test(sequence)
    acf = compute_autocorrelation(sequence, max_lag=20)
"""

import numpy as np
from typing import List, Union, Optional, Tuple
from dataclasses import dataclass


@dataclass
class EntropyResult:
    """Result of entropy computation."""
    spectral_entropy: float
    normalized_entropy: float  # 0-1 scale
    
    @property
    def is_random(self) -> bool:
        """High entropy indicates randomness."""
        return self.normalized_entropy > 0.8
    
    @property
    def is_structured(self) -> bool:
        """Low entropy indicates structure/patterns."""
        return self.normalized_entropy < 0.3


@dataclass
class RunsTestResult:
    """Result of runs test for randomness."""
    n_runs: int
    expected_runs: float
    z_score: float
    p_value: float
    
    @property
    def is_random(self) -> bool:
        """Returns True if sequence appears random (p > 0.05)."""
        return self.p_value > 0.05


def compute_spectral_entropy(
    sequence: Union[List[float], np.ndarray],
    normalize: bool = True
) -> float:
    """
    Compute spectral entropy of a sequence.
    
    Spectral entropy measures the "flatness" of the power spectrum.
    High entropy = whitenoise-like, Low entropy = structured/periodic.
    
    Args:
        sequence: Input time series
        normalize: Return normalized entropy (0-1)
        
    Returns:
        Spectral entropy (normalized to 0-1 if normalize=True)
    """
    sequence = np.asarray(sequence)
    
    if len(sequence) < 4:
        raise ValueError(f"Sequence too short: need at least 4 points, got {len(sequence)}")
    
    # Compute power spectral density using FFT
    fft = np.fft.fft(sequence)
    psd = np.abs(fft) ** 2
    
    # Use only positive frequencies (excluding DC)
    psd = psd[1:len(psd)//2 + 1]
    
    # Normalize to probability distribution
    psd = psd / np.sum(psd)
    
    # Shannon entropy
    # Remove zeros to avoid log(0)
    psd_nonzero = psd[psd > 0]
    
    if len(psd_nonzero) == 0:
        return 0.0
    
    entropy = -np.sum(psd_nonzero * np.log2(psd_nonzero))
    
    if normalize:
        # Normalize by maximum possible entropy
        max_entropy = np.log2(len(psd))
        entropy = entropy / max_entropy if max_entropy > 0 else 0.0
    
    return float(entropy)


def compute_runs_test(
    sequence: Union[List[float], np.ndarray],
    threshold: Optional[float] = None
) -> RunsTestResult:
    """
    Wald-Wolfowitz runs test for randomness.
    
    A "run" is a sequence of consecutive values above or below the threshold.
    Tests whether the number of runs is consistent with randomness.
    
    Args:
        sequence: Input sequence
        threshold: Threshold for binary conversion (default: median)
        
    Returns:
        RunsTestResult with test statistics
    """
    sequence = np.asarray(sequence)
    
    if len(sequence) < 2:
        raise ValueError("Need at least 2 points for runs test")
    
    # Convert to binary sequence
    if threshold is None:
        threshold = np.median(sequence)
    
    binary = (sequence > threshold).astype(int)
    
    # Count runs
    n_runs = 1
    for i in range(1, len(binary)):
        if binary[i] != binary[i-1]:
            n_runs += 1
    
    # Count values above and below threshold
    n1 = np.sum(binary)
    n0 = len(binary) - n1
    
    if n0 == 0 or n1 == 0:
        # All same value, not random
        return RunsTestResult(
            n_runs=n_runs,
            expected_runs=0,
            z_score=float('inf'),
            p_value=0.0
        )
    
    # Expected number of runs and standard deviation under null hypothesis
    n = len(binary)
    expected_runs = (2 * n0 * n1) / n + 1
    variance = (2 * n0 * n1 * (2 * n0 * n1 - n)) / (n**2 * (n - 1))
    std_runs = np.sqrt(variance)
    
    # Z-score
    if std_runs > 0:
        z_score = (n_runs - expected_runs) / std_runs
    else:
        z_score = 0.0
    
    # Two-tailed p-value (approximate using normal distribution)
    p_value = 2 * (1 - _standard_normal_cdf(abs(z_score)))
    
    return RunsTestResult(
        n_runs=n_runs,
        expected_runs=expected_runs,
        z_score=z_score,
        p_value=p_value
    )


def _standard_normal_cdf(x: float) -> float:
    """
    Approximate CDF of standard normal distribution.
    
    Uses error function approximation.
    """
    # erf approximation
    t = 1.0 / (1.0 + 0.2316419 * abs(x))
    d = 0.3989423 * np.exp(-x*x/2.0)
    p = d * t * (0.3193815 + t * (-0.3565638 + t * (1.781478 + t * (-1.821256 + t * 1.330274))))
    
    if x > 0:
        return 1.0 - p
    else:
        return p


def compute_autocorrelation(
    sequence: Union[List[float], np.ndarray],
    max_lag: Optional[int] = None,
    normalize: bool = True
) -> np.ndarray:
    """
    Compute autocorrelation function (ACF).
    
    Measures correlation of sequence with itself at different time lags.
    
    Args:
        sequence: Input time series
        max_lag: Maximum lag (default: len(sequence) // 4)
        normalize: Normalize ACF by variance (default: True)
        
    Returns:
        Array of autocorrelation values for lags 0 to max_lag
    """
    sequence = np.asarray(sequence)
    n = len(sequence)
    
    if n < 2:
        raise ValueError("Need at least 2 points for autocorrelation")
    
    if max_lag is None:
        max_lag = min(n // 4, 50)
    
    max_lag = min(max_lag, n - 1)
    
    # Center the sequence
    mean = np.mean(sequence)
    centered = sequence - mean
    
    # Compute ACF
    acf = np.zeros(max_lag + 1)
    
    if normalize:
        # Variance for normalization
        var = np.sum(centered ** 2) / n
        
        if var == 0:
            return acf
    
    for lag in range(max_lag + 1):
        # Correlation at this lag
        if lag == 0:
            acf[lag] = 1.0 if normalize else var * n
        else:
            c = np.sum(centered[:-lag] * centered[lag:])
            
            if normalize:
                acf[lag] = c / (var * n) if var > 0 else 0
            else:
                acf[lag] = c / n
    
    return acf


def compute_decorrelation_time(
    sequence: Union[List[float], np.ndarray],
    threshold: float = 1.0 / np.e
) -> int:
    """
    Compute decorrelation time (lag at which ACF drops below threshold).
    
    Default threshold is 1/e ≈ 0.368.
    
    Args:
        sequence: Input time series
        threshold: ACF threshold for decorrelation
        
    Returns:
        Decorrelation time (in units of sampling interval)
    """
    acf = compute_autocorrelation(sequence, normalize=True)
    
    # Find first lag where ACF drops below threshold
    for lag in range(1, len(acf)):
        if acf[lag] < threshold:
            return lag
    
    # If never drops below threshold, return max lag
    return len(acf) - 1


def analyze_entropy(
    sequence: Union[List[float], np.ndarray]
) -> EntropyResult:
    """
    Complete entropy analysis of sequence.
    
    Args:
        sequence: Input time series
        
    Returns:
        EntropyResult with spectral entropy metrics
    """
    spectral_entropy = compute_spectral_entropy(sequence, normalize=False)
    normalized_entropy = compute_spectral_entropy(sequence, normalize=True)
    
    return EntropyResult(
        spectral_entropy=spectral_entropy,
        normalized_entropy=normalized_entropy
    )


class EntropyAnalyzer:
    """Analyzer for entropy and randomness metrics."""
    
    def __init__(self):
        """Initialize entropy analyzer."""
        pass
    
    def analyze(self, sequence: Union[List[float], np.ndarray]) -> dict:
        """
        Complete analysis of sequence randomness/structure.
        
        Args:
            sequence: Input time series
            
        Returns:
            Dictionary with all entropy metrics
        """
        sequence = np.asarray(sequence)
        
        # Spectral entropy
        entropy_result = analyze_entropy(sequence)
        
        # Runs test
        try:
            runs_result = compute_runs_test(sequence)
        except ValueError:
            runs_result = None
        
        # Autocorrelation
        try:
            acf = compute_autocorrelation(sequence, max_lag=20)
            decorr_time = compute_decorrelation_time(sequence)
        except ValueError:
            acf = None
            decorr_time = 0
        
        return {
            'spectral_entropy': entropy_result.spectral_entropy,
            'normalized_entropy': entropy_result.normalized_entropy,
            'is_random_entropy': entropy_result.is_random,
            'is_structured': entropy_result.is_structured,
            'runs_test': runs_result,
            'autocorrelation': acf,
            'decorrelation_time': decorr_time,
        }
