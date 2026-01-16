"""
Entropy and Randomness Metrics - FULLY STANDALONE
=================================================

Spectral entropy, runs test, and autocorrelation for time series.
Can be copied to any project - only requires numpy.

Usage:
    from entropy import compute_spectral_entropy, compute_runs_test, compute_autocorrelation
    
    entropy = compute_spectral_entropy(signal)
    print(f"Spectral entropy: {entropy:.3f}")
    
    is_random = compute_runs_test(signal)
    print(f"Passes randomness test: {is_random}")

Dependencies: numpy only
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class SpectralEntropyResult:
    """Result from spectral entropy calculation."""
    
    entropy: float
    """Spectral entropy value (0 to 1)."""
    
    power_spectrum: np.ndarray
    """Power spectrum used in calculation."""
    
    frequencies: np.ndarray
    """Frequency values corresponding to power spectrum."""
    
    @property
    def is_deterministic(self) -> bool:
        """True if entropy is low (< 0.3), indicating deterministic signal."""
        return self.entropy < 0.3
    
    @property
    def is_random(self) -> bool:
        """True if entropy is high (> 0.7), indicating random signal."""
        return self.entropy > 0.7
    
    @property
    def interpretation(self) -> str:
        """Human-readable interpretation."""
        if self.is_deterministic:
            return f"DETERMINISTIC (entropy={self.entropy:.3f}) - Regular, predictable frequency content"
        elif self.is_random:
            return f"RANDOM (entropy={self.entropy:.3f}) - Flat, noisy frequency spectrum"
        else:
            return f"MIXED (entropy={self.entropy:.3f}) - Some structure with noise"
    
    def __repr__(self):
        return f"SpectralEntropyResult(entropy={self.entropy:.4f}, deterministic={self.is_deterministic})"


@dataclass
class RunsTestResult:
    """Result from Wald-Wolfowitz runs test."""
    
    n_runs: int
    """Number of runs (sequences of same sign)."""
    
    expected_runs: float
    """Expected number of runs for random sequence."""
    
    z_score: float
    """Z-score for the test statistic."""
    
    p_value: float
    """Two-tailed p-value."""
    
    is_random: bool
    """True if null hypothesis (randomness) is not rejected at α=0.05."""
    
    @property
    def interpretation(self) -> str:
        """Human-readable interpretation."""
        if self.is_random:
            return f"RANDOM (p={self.p_value:.3f}) - Consistent with random sequence"
        elif self.n_runs < self.expected_runs:
            return f"TOO FEW RUNS (p={self.p_value:.3f}) - Tends toward clustering"
        else:
            return f"TOO MANY RUNS (p={self.p_value:.3f}) - Tends toward alternation"
    
    def __repr__(self):
        return f"RunsTestResult(runs={self.n_runs}, expected={self.expected_runs:.1f}, random={self.is_random})"


@dataclass
class AutocorrelationResult:
    """Result from autocorrelation calculation."""
    
    acf: np.ndarray
    """Autocorrelation function values."""
    
    lags: np.ndarray
    """Lag values corresponding to ACF."""
    
    decay_rate: float
    """Exponential decay rate of ACF."""
    
    @property
    def has_memory(self) -> bool:
        """True if autocorrelation decays slowly (memory present)."""
        # Slow decay = decay_rate < 0.1
        return abs(self.decay_rate) < 0.1
    
    @property
    def memory_length(self) -> int:
        """Approximate memory length (lag where ACF drops below 0.2)."""
        threshold_mask = np.abs(self.acf) > 0.2
        if np.any(threshold_mask):
            return int(np.max(np.where(threshold_mask)[0])) + 1
        return 0
    
    @property
    def interpretation(self) -> str:
        """Human-readable interpretation."""
        if self.has_memory:
            return f"LONG MEMORY (decay={self.decay_rate:.3f}) - Memory length ~{self.memory_length} steps"
        else:
            return f"SHORT MEMORY (decay={self.decay_rate:.3f}) - Rapid decorrelation"
    
    def __repr__(self):
        return f"AutocorrelationResult(decay={self.decay_rate:.4f}, memory_length={self.memory_length})"


def compute_spectral_entropy(signal: np.ndarray,
                            normalize: bool = True) -> SpectralEntropyResult:
    """
    Compute spectral entropy to measure signal regularity.
    
    Spectral entropy quantifies the flatness of the power spectrum:
    - Low entropy (→0): Deterministic, few dominant frequencies (e.g., sine wave)
    - High entropy (→1): Random, flat spectrum (e.g., white noise)
    
    This is a standalone implementation requiring only numpy.
    
    Args:
        signal: 1D time series, or 2D [time_steps, features] (uses norm)
        normalize: If True, normalizes entropy to [0, 1] range
    
    Returns:
        SpectralEntropyResult with entropy, power spectrum, and interpretation
    
    Examples:
        >>> # Deterministic signal (sine wave)
        >>> t = np.linspace(0, 10, 1000)
        >>> sine = np.sin(2 * np.pi * 5 * t)
        >>> result = compute_spectral_entropy(sine)
        >>> result.is_deterministic
        True
        
        >>> # Random signal (white noise)
        >>> np.random.seed(42)
        >>> noise = np.random.randn(1000)
        >>> result = compute_spectral_entropy(noise)
        >>> result.is_random
        True
    
    References:
        - Inouye, T., et al. (1991). Quantification of EEG irregularity by use of
          the entropy of the power spectrum. Electroencephalography and Clinical
          Neurophysiology, 79(3), 204-210.
    """
    # Flatten if multidimensional
    if signal.ndim > 1:
        signal = np.linalg.norm(signal, axis=1)
    
    n = len(signal)
    
    # Compute power spectrum using FFT
    fft = np.fft.fft(signal)
    power = np.abs(fft[:n // 2]) ** 2
    
    # Normalize to probability distribution
    power = power / np.sum(power)
    
    # Remove zeros to avoid log(0)
    power = power[power > 0]
    
    # Compute Shannon entropy
    entropy = -np.sum(power * np.log(power))
    
    # Normalize to [0, 1] if requested
    if normalize:
        # Maximum entropy for n/2 bins
        max_entropy = np.log(len(power))
        if max_entropy > 0:
            entropy = entropy / max_entropy
    
    # Get frequencies for full spectrum
    frequencies = np.fft.fftfreq(n)[:n // 2]
    full_power = np.abs(np.fft.fft(signal)[:n // 2]) ** 2
    
    return SpectralEntropyResult(
        entropy=float(entropy),
        power_spectrum=full_power,
        frequencies=frequencies
    )


def compute_runs_test(sequence: np.ndarray,
                     threshold: Optional[float] = None) -> RunsTestResult:
    """
    Wald-Wolfowitz runs test for randomness.
    
    A "run" is a sequence of consecutive values above (or below) a threshold.
    Tests whether the number of runs is consistent with randomness.
    
    This is a standalone implementation requiring only numpy.
    
    Args:
        sequence: 1D time series, or 2D [time_steps, features] (uses norm)
        threshold: Threshold for binarizing (default: median)
    
    Returns:
        RunsTestResult with test statistics and interpretation
    
    Examples:
        >>> # Random sequence
        >>> np.random.seed(42)
        >>> random_seq = np.random.randn(100)
        >>> result = compute_runs_test(random_seq)
        >>> result.is_random
        True
        
        >>> # Clustered sequence (too few runs)
        >>> clustered = np.concatenate([np.ones(50), -np.ones(50)])
        >>> result = compute_runs_test(clustered)
        >>> result.is_random
        False
    
    References:
        - Wald, A., & Wolfowitz, J. (1940). On a test whether two samples are
          from the same population. The Annals of Mathematical Statistics, 11(2), 147-162.
    """
    # Flatten if multidimensional
    if sequence.ndim > 1:
        sequence = np.linalg.norm(sequence, axis=1)
    
    n = len(sequence)
    
    if n < 3:
        return RunsTestResult(
            n_runs=0,
            expected_runs=0.0,
            z_score=0.0,
            p_value=1.0,
            is_random=True
        )
    
    # Use median as threshold if not provided
    if threshold is None:
        threshold = np.median(sequence)
    
    # Binarize sequence (1 if above threshold, 0 if below)
    binary = (sequence > threshold).astype(int)
    
    # Count number of 1s and 0s
    n1 = np.sum(binary)
    n0 = n - n1
    
    # Handle edge cases
    if n1 == 0 or n0 == 0:
        return RunsTestResult(
            n_runs=1,
            expected_runs=1.0,
            z_score=0.0,
            p_value=1.0,
            is_random=True
        )
    
    # Count runs
    runs = 1
    for i in range(1, n):
        if binary[i] != binary[i - 1]:
            runs += 1
    
    # Expected number of runs for random sequence
    expected_runs = (2 * n1 * n0) / n + 1
    
    # Variance of number of runs
    variance = (2 * n1 * n0 * (2 * n1 * n0 - n)) / (n ** 2 * (n - 1))
    
    # Z-score
    if variance > 0:
        z_score = (runs - expected_runs) / np.sqrt(variance)
    else:
        z_score = 0.0
    
    # Two-tailed p-value (approximate using normal distribution)
    p_value = 2 * (1 - _normal_cdf(abs(z_score)))
    
    # Test at α = 0.05 significance level
    is_random = p_value > 0.05
    
    return RunsTestResult(
        n_runs=int(runs),
        expected_runs=float(expected_runs),
        z_score=float(z_score),
        p_value=float(p_value),
        is_random=bool(is_random)
    )


def compute_autocorrelation(signal: np.ndarray,
                           max_lag: Optional[int] = None) -> AutocorrelationResult:
    """
    Compute autocorrelation function (ACF).
    
    Measures correlation of signal with itself at different time lags.
    Useful for detecting periodicity and memory in time series.
    
    This is a standalone implementation requiring only numpy.
    
    Args:
        signal: 1D time series, or 2D [time_steps, features] (uses norm)
        max_lag: Maximum lag to compute (default: min(50, n//4))
    
    Returns:
        AutocorrelationResult with ACF values, decay rate, and interpretation
    
    Examples:
        >>> # Signal with memory
        >>> np.random.seed(42)
        >>> persistent = np.cumsum(np.random.randn(200))
        >>> result = compute_autocorrelation(persistent)
        >>> result.has_memory
        True
        
        >>> # White noise (no memory)
        >>> noise = np.random.randn(200)
        >>> result = compute_autocorrelation(noise)
        >>> result.has_memory
        False
    """
    # Flatten if multidimensional
    if signal.ndim > 1:
        signal = np.linalg.norm(signal, axis=1)
    
    n = len(signal)
    
    # Set default max_lag
    if max_lag is None:
        max_lag = min(50, n // 4)
    
    max_lag = min(max_lag, n - 1)
    
    # Normalize signal (zero mean)
    signal_norm = signal - np.mean(signal)
    variance = np.var(signal)
    
    if variance == 0:
        # Constant signal
        acf = np.ones(max_lag + 1)
        decay_rate = 0.0
    else:
        # Compute ACF
        acf = np.zeros(max_lag + 1)
        for lag in range(max_lag + 1):
            if lag == 0:
                acf[lag] = 1.0
            else:
                # Correlation at this lag
                correlation = np.mean(signal_norm[:-lag] * signal_norm[lag:])
                acf[lag] = correlation / variance
    
    # Estimate decay rate (fit exponential decay)
    # ACF ≈ exp(-decay_rate * lag)
    lags = np.arange(1, max_lag + 1)
    acf_nonzero = acf[1:]
    
    # Fit in log space (only for positive ACF values)
    positive_mask = acf_nonzero > 0
    if np.sum(positive_mask) > 1:
        log_acf = np.log(acf_nonzero[positive_mask])
        lags_positive = lags[positive_mask]
        # Linear fit: log(ACF) = -decay_rate * lag + const
        decay_rate, _ = np.polyfit(lags_positive, log_acf, 1)
        decay_rate = -decay_rate  # Negate to get positive decay rate
    else:
        decay_rate = 1.0  # Fast decay
    
    return AutocorrelationResult(
        acf=acf,
        lags=np.arange(max_lag + 1),
        decay_rate=float(decay_rate)
    )


def _normal_cdf(x: float) -> float:
    """
    Approximate cumulative distribution function for standard normal.
    
    Uses error function approximation via tanh.
    Coefficients are derived from Abramowitz & Stegun approximation.
    """
    # Constants for erf approximation
    SQRT_2_OVER_PI = 0.7978845608  # sqrt(2/pi)
    ERF_COEFFICIENT = 0.044715
    
    return 0.5 * (1 + np.tanh(SQRT_2_OVER_PI * (x + ERF_COEFFICIENT * x ** 3)))


__all__ = [
    "SpectralEntropyResult",
    "RunsTestResult",
    "AutocorrelationResult",
    "compute_spectral_entropy",
    "compute_runs_test",
    "compute_autocorrelation",
]


if __name__ == "__main__":
    # Self-test examples
    print("Entropy and Randomness Metrics - Standalone Tests")
    print("=" * 50)
    
    np.random.seed(42)
    
    # Test 1: Spectral entropy - deterministic signal
    print("\n1. Spectral Entropy - Sine Wave (deterministic):")
    t = np.linspace(0, 10, 1000)
    sine = np.sin(2 * np.pi * 5 * t)
    result = compute_spectral_entropy(sine)
    print(f"   {result.interpretation}")
    
    # Test 2: Spectral entropy - random signal
    print("\n2. Spectral Entropy - White Noise (random):")
    noise = np.random.randn(1000)
    result = compute_spectral_entropy(noise)
    print(f"   {result.interpretation}")
    
    # Test 3: Runs test - random sequence
    print("\n3. Runs Test - Random Sequence:")
    random_seq = np.random.randn(100)
    result = compute_runs_test(random_seq)
    print(f"   {result.interpretation}")
    print(f"   Runs: {result.n_runs}, Expected: {result.expected_runs:.1f}")
    
    # Test 4: Runs test - clustered sequence
    print("\n4. Runs Test - Clustered Sequence:")
    clustered = np.concatenate([np.ones(50), -np.ones(50)])
    result = compute_runs_test(clustered)
    print(f"   {result.interpretation}")
    print(f"   Runs: {result.n_runs}, Expected: {result.expected_runs:.1f}")
    
    # Test 5: Runs test - alternating sequence
    print("\n5. Runs Test - Alternating Sequence:")
    alternating = np.array([(-1)**i for i in range(100)])
    result = compute_runs_test(alternating)
    print(f"   {result.interpretation}")
    print(f"   Runs: {result.n_runs}, Expected: {result.expected_runs:.1f}")
    
    # Test 6: Autocorrelation - persistent signal
    print("\n6. Autocorrelation - Persistent Signal (random walk):")
    persistent = np.cumsum(np.random.randn(200))
    result = compute_autocorrelation(persistent)
    print(f"   {result.interpretation}")
    
    # Test 7: Autocorrelation - white noise
    print("\n7. Autocorrelation - White Noise:")
    noise = np.random.randn(200)
    result = compute_autocorrelation(noise)
    print(f"   {result.interpretation}")
    
    # Test 8: Autocorrelation - periodic signal
    print("\n8. Autocorrelation - Periodic Signal:")
    t = np.linspace(0, 20, 200)
    periodic = np.sin(2 * np.pi * 0.5 * t)
    result = compute_autocorrelation(periodic, max_lag=50)
    print(f"   {result.interpretation}")
    print(f"   First few ACF values: {result.acf[:5]}")
    
    print("\n✓ All tests completed successfully!")
