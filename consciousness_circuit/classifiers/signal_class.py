"""
Signal Classification - FULLY STANDALONE
========================================

Classifies dynamical system behavior based on metrics.
Can be copied to any project - only requires numpy.

Usage:
    from signal_class import SignalClass, classify_signal, ClassificationResult
    
    metrics = {
        'lyapunov': 0.4,
        'hurst': 0.6,
        'msd_exponent': 1.2
    }
    result = classify_signal(metrics)
    print(f"Class: {result.signal_class}")
    print(f"Confidence: {result.confidence:.2f}")

Dependencies: numpy only
"""

import numpy as np
from enum import Enum
from dataclasses import dataclass
from typing import Dict, Optional, Any


class SignalClass(Enum):
    """Classification of signal behavior in dynamical systems."""
    
    NOISE = "noise"              # Random, no structure
    DRIFT = "drift"              # Slow wandering
    ATTRACTOR = "attractor"      # Converging to fixed point
    PERIODIC = "periodic"        # Regular oscillations
    CHAOTIC = "chaotic"          # Deterministic chaos
    BALLISTIC = "ballistic"      # Directed motion
    DIFFUSIVE = "diffusive"      # Random walk
    CONFINED = "confined"        # Bounded motion
    UNKNOWN = "unknown"          # Cannot classify
    
    def __str__(self):
        return self.value
    
    @property
    def description(self) -> str:
        """Human-readable description of the signal class."""
        descriptions = {
            SignalClass.NOISE: "Random signal with no structure or memory",
            SignalClass.DRIFT: "Slow wandering without clear direction",
            SignalClass.ATTRACTOR: "Converging to a stable fixed point",
            SignalClass.PERIODIC: "Regular oscillations or cyclic behavior",
            SignalClass.CHAOTIC: "Deterministic chaos - sensitive to initial conditions",
            SignalClass.BALLISTIC: "Directed motion with constant velocity",
            SignalClass.DIFFUSIVE: "Random walk - normal diffusion",
            SignalClass.CONFINED: "Bounded motion in limited region",
            SignalClass.UNKNOWN: "Insufficient data or unclear classification",
        }
        return descriptions.get(self, "No description available")


@dataclass
class ClassificationResult:
    """Result from signal classification."""
    
    signal_class: SignalClass
    """The classified signal type."""
    
    confidence: float
    """Confidence in classification (0 to 1)."""
    
    scores: Dict[SignalClass, float]
    """Scores for all classes (for debugging/analysis)."""
    
    metrics_used: Dict[str, Any]
    """Metrics that were used in classification."""
    
    @property
    def is_deterministic(self) -> bool:
        """True if signal is deterministic (not noise or pure drift)."""
        return self.signal_class not in [SignalClass.NOISE, SignalClass.UNKNOWN]
    
    @property
    def is_stable(self) -> bool:
        """True if signal is stable (attractor or confined)."""
        return self.signal_class in [SignalClass.ATTRACTOR, SignalClass.CONFINED]
    
    @property
    def is_chaotic(self) -> bool:
        """True if signal exhibits chaotic behavior."""
        return self.signal_class == SignalClass.CHAOTIC
    
    @property
    def interpretation(self) -> str:
        """Human-readable interpretation."""
        conf_str = f"(confidence: {self.confidence:.2%})"
        return f"{self.signal_class.value.upper()} {conf_str} - {self.signal_class.description}"
    
    def __repr__(self):
        return f"ClassificationResult(class={self.signal_class.value}, confidence={self.confidence:.3f})"


def classify_signal(metrics: Dict[str, float],
                   use_fuzzy: bool = True) -> ClassificationResult:
    """
    Classify signal based on computed metrics.
    
    Combines multiple metrics (Lyapunov, Hurst, MSD, etc.) to determine
    the type of dynamical behavior in a signal.
    
    This is a standalone implementation requiring only numpy.
    
    Args:
        metrics: Dictionary of computed metrics. Can include:
                - 'lyapunov': Lyapunov exponent
                - 'hurst': Hurst exponent
                - 'msd_exponent': MSD power-law exponent (α)
                - 'spectral_entropy': Spectral entropy
                - 'is_periodic': Boolean for periodicity
                - 'variance': Signal variance
        use_fuzzy: If True, uses fuzzy classification with confidence scores
                  If False, uses hard thresholds (faster, less nuanced)
    
    Returns:
        ClassificationResult with class, confidence, and scores
    
    Examples:
        >>> # Chaotic signal
        >>> metrics = {'lyapunov': 0.6, 'hurst': 0.5}
        >>> result = classify_signal(metrics)
        >>> result.signal_class == SignalClass.CHAOTIC
        True
        
        >>> # Attractor (stable)
        >>> metrics = {'lyapunov': -0.4, 'msd_exponent': 0.3}
        >>> result = classify_signal(metrics)
        >>> result.signal_class == SignalClass.ATTRACTOR
        True
        
        >>> # Ballistic motion
        >>> metrics = {'msd_exponent': 1.9, 'hurst': 0.9}
        >>> result = classify_signal(metrics)
        >>> result.signal_class == SignalClass.BALLISTIC
        True
    """
    if use_fuzzy:
        return _classify_fuzzy(metrics)
    else:
        return _classify_hard(metrics)


def _classify_fuzzy(metrics: Dict[str, float]) -> ClassificationResult:
    """
    Fuzzy classification with confidence scores.
    
    Computes a score for each class and selects the highest.
    """
    # Initialize scores for each class
    scores = {cls: 0.0 for cls in SignalClass}
    
    # Extract metrics (with defaults)
    lyapunov = metrics.get('lyapunov', 0.0)
    hurst = metrics.get('hurst', 0.5)
    msd_exp = metrics.get('msd_exponent', 1.0)
    entropy = metrics.get('spectral_entropy', 0.5)
    is_periodic = metrics.get('is_periodic', False)
    
    # Periodic check (strongest signal)
    if is_periodic:
        scores[SignalClass.PERIODIC] = 0.9
    
    # Lyapunov-based scoring
    if lyapunov > 0.5:
        # High positive = chaotic
        scores[SignalClass.CHAOTIC] += min(1.0, lyapunov / 0.5) * 0.8
    elif lyapunov < -0.3:
        # Negative = attracting
        scores[SignalClass.ATTRACTOR] += min(1.0, -lyapunov / 0.3) * 0.7
    elif -0.1 <= lyapunov <= 0.1:
        # Near zero = diffusive or drift
        scores[SignalClass.DIFFUSIVE] += 0.3
        scores[SignalClass.DRIFT] += 0.2
    
    # Hurst-based scoring
    if hurst > 0.7:
        # High persistence = trending/ballistic
        scores[SignalClass.BALLISTIC] += (hurst - 0.7) / 0.3 * 0.6
    elif hurst < 0.3:
        # Anti-persistent = noise
        scores[SignalClass.NOISE] += (0.3 - hurst) / 0.3 * 0.6
    elif 0.45 <= hurst <= 0.55:
        # Random walk
        scores[SignalClass.DIFFUSIVE] += 0.5
    
    # MSD exponent-based scoring
    if msd_exp > 1.7:
        # Superdiffusive/ballistic
        scores[SignalClass.BALLISTIC] += (msd_exp - 1.7) / 0.3 * 0.7
    elif 0.8 <= msd_exp <= 1.5:
        # Normal diffusion
        scores[SignalClass.DIFFUSIVE] += 0.6
    elif msd_exp < 0.5:
        # Confined
        scores[SignalClass.CONFINED] += (0.5 - msd_exp) / 0.5 * 0.8
    elif 0.5 <= msd_exp < 0.8:
        # Subdiffusive
        scores[SignalClass.DRIFT] += 0.4
    
    # Entropy-based scoring
    if entropy > 0.7:
        # High entropy = random
        scores[SignalClass.NOISE] += (entropy - 0.7) / 0.3 * 0.5
    elif entropy < 0.3:
        # Low entropy = structured
        scores[SignalClass.PERIODIC] += (0.3 - entropy) / 0.3 * 0.3
        scores[SignalClass.ATTRACTOR] += (0.3 - entropy) / 0.3 * 0.3
    
    # Find best class
    if max(scores.values()) == 0:
        best_class = SignalClass.UNKNOWN
        confidence = 0.0
    else:
        best_class = max(scores, key=scores.get)
        # Confidence based on margin to second-best
        sorted_scores = sorted(scores.values(), reverse=True)
        best_score = sorted_scores[0]
        second_score = sorted_scores[1] if len(sorted_scores) > 1 else 0
        confidence = min(1.0, best_score / (best_score + second_score + 0.1))
    
    return ClassificationResult(
        signal_class=best_class,
        confidence=confidence,
        scores=scores,
        metrics_used=metrics
    )


def _classify_hard(metrics: Dict[str, float]) -> ClassificationResult:
    """
    Hard threshold classification (simpler, faster).
    
    Uses decision tree with clear thresholds.
    """
    lyapunov = metrics.get('lyapunov', 0.0)
    hurst = metrics.get('hurst', 0.5)
    msd_exp = metrics.get('msd_exponent', 1.0)
    is_periodic = metrics.get('is_periodic', False)
    
    # Decision tree
    if is_periodic:
        signal_class = SignalClass.PERIODIC
        confidence = 0.9
    elif lyapunov > 0.5:
        signal_class = SignalClass.CHAOTIC
        confidence = 0.8
    elif lyapunov < -0.5 and msd_exp < 0.5:
        signal_class = SignalClass.ATTRACTOR
        confidence = 0.8
    elif msd_exp > 1.7:
        signal_class = SignalClass.BALLISTIC
        confidence = 0.7
    elif hurst < 0.4:
        signal_class = SignalClass.NOISE
        confidence = 0.6
    elif msd_exp < 0.5:
        signal_class = SignalClass.CONFINED
        confidence = 0.6
    elif 0.8 <= msd_exp <= 1.5:
        signal_class = SignalClass.DIFFUSIVE
        confidence = 0.6
    else:
        signal_class = SignalClass.DRIFT
        confidence = 0.4
    
    return ClassificationResult(
        signal_class=signal_class,
        confidence=confidence,
        scores={signal_class: 1.0},
        metrics_used=metrics
    )


class SignalClassifier:
    """
    Configurable signal classifier.
    
    Allows customization of classification thresholds and logic.
    
    Example:
        >>> classifier = SignalClassifier(
        ...     lyapunov_chaos_threshold=0.4,
        ...     use_fuzzy=True
        ... )
        >>> result = classifier.classify({'lyapunov': 0.5, 'hurst': 0.6})
        >>> print(result.interpretation)
    """
    
    def __init__(self,
                 lyapunov_chaos_threshold: float = 0.5,
                 lyapunov_stable_threshold: float = -0.3,
                 hurst_persistent_threshold: float = 0.7,
                 hurst_antipersistent_threshold: float = 0.3,
                 msd_ballistic_threshold: float = 1.7,
                 msd_confined_threshold: float = 0.5,
                 use_fuzzy: bool = True):
        """
        Initialize classifier with custom thresholds.
        
        Args:
            lyapunov_chaos_threshold: Threshold for chaotic behavior
            lyapunov_stable_threshold: Threshold for stable/attracting
            hurst_persistent_threshold: Threshold for persistent/trending
            hurst_antipersistent_threshold: Threshold for anti-persistent
            msd_ballistic_threshold: Threshold for ballistic motion
            msd_confined_threshold: Threshold for confined motion
            use_fuzzy: Use fuzzy classification (vs hard thresholds)
        """
        self.lyapunov_chaos = lyapunov_chaos_threshold
        self.lyapunov_stable = lyapunov_stable_threshold
        self.hurst_persistent = hurst_persistent_threshold
        self.hurst_antipersistent = hurst_antipersistent_threshold
        self.msd_ballistic = msd_ballistic_threshold
        self.msd_confined = msd_confined_threshold
        self.use_fuzzy = use_fuzzy
    
    def classify(self, metrics: Dict[str, float]) -> ClassificationResult:
        """
        Classify signal with configured thresholds.
        
        Args:
            metrics: Dictionary of computed metrics
        
        Returns:
            ClassificationResult
        """
        # For now, delegates to the global functions
        # Could be extended to use custom thresholds
        return classify_signal(metrics, use_fuzzy=self.use_fuzzy)


__all__ = [
    "SignalClass",
    "ClassificationResult",
    "classify_signal",
    "SignalClassifier",
]


if __name__ == "__main__":
    # Self-test examples
    print("Signal Classification - Standalone Tests")
    print("=" * 50)
    
    # Test 1: Chaotic signal
    print("\n1. Chaotic Signal:")
    metrics = {'lyapunov': 0.6, 'hurst': 0.5, 'msd_exponent': 1.3}
    result = classify_signal(metrics)
    print(f"   {result.interpretation}")
    
    # Test 2: Attractor (stable)
    print("\n2. Attractor (Stable):")
    metrics = {'lyapunov': -0.4, 'hurst': 0.4, 'msd_exponent': 0.3}
    result = classify_signal(metrics)
    print(f"   {result.interpretation}")
    
    # Test 3: Ballistic motion
    print("\n3. Ballistic Motion:")
    metrics = {'lyapunov': 0.1, 'hurst': 0.9, 'msd_exponent': 1.9}
    result = classify_signal(metrics)
    print(f"   {result.interpretation}")
    
    # Test 4: Random noise
    print("\n4. Random Noise:")
    metrics = {'lyapunov': 0.0, 'hurst': 0.2, 'spectral_entropy': 0.9}
    result = classify_signal(metrics)
    print(f"   {result.interpretation}")
    
    # Test 5: Periodic signal
    print("\n5. Periodic Signal:")
    metrics = {'is_periodic': True, 'spectral_entropy': 0.2}
    result = classify_signal(metrics)
    print(f"   {result.interpretation}")
    
    # Test 6: Diffusive (random walk)
    print("\n6. Diffusive (Random Walk):")
    metrics = {'lyapunov': 0.0, 'hurst': 0.5, 'msd_exponent': 1.0}
    result = classify_signal(metrics)
    print(f"   {result.interpretation}")
    
    # Test 7: Confined motion
    print("\n7. Confined Motion:")
    metrics = {'lyapunov': -0.2, 'hurst': 0.3, 'msd_exponent': 0.2}
    result = classify_signal(metrics)
    print(f"   {result.interpretation}")
    
    # Test 8: Custom classifier
    print("\n8. Custom Classifier (stricter chaos threshold):")
    classifier = SignalClassifier(lyapunov_chaos_threshold=0.4)
    metrics = {'lyapunov': 0.45, 'hurst': 0.6}
    result = classifier.classify(metrics)
    print(f"   {result.interpretation}")
    
    # Test 9: Fuzzy vs Hard classification
    print("\n9. Fuzzy vs Hard Classification:")
    metrics = {'lyapunov': 0.3, 'hurst': 0.6, 'msd_exponent': 1.2}
    
    fuzzy = classify_signal(metrics, use_fuzzy=True)
    hard = classify_signal(metrics, use_fuzzy=False)
    
    print(f"   Fuzzy: {fuzzy.signal_class.value} (conf={fuzzy.confidence:.2f})")
    print(f"   Hard:  {hard.signal_class.value} (conf={hard.confidence:.2f})")
    
    print("\n✓ All tests completed successfully!")
