"""
Signal Classification System

Standalone enum and classifier for trajectory patterns.

Usage:
    from consciousness_circuit.classifiers.signal_class import SignalClass, classify_signal
    
    sig_class = classify_signal(metrics_dict)
    print(sig_class.name)  # ATTRACTOR, DRIFT, NOISE, etc.
"""

from enum import Enum, auto
from typing import Dict, Any, Optional
from dataclasses import dataclass


class SignalClass(Enum):
    """Classification of signal/trajectory patterns."""
    NOISE = auto()       # Pure random walk
    DRIFT = auto()       # Gradual bias/trend
    ATTRACTOR = auto()   # Convergent behavior
    PERIODIC = auto()    # Cyclic patterns
    CHAOTIC = auto()     # Deterministic chaos
    ANOMALOUS = auto()   # Unusual diffusion
    INFLUENCE = auto()   # Multi-indicator agreement (high consciousness)
    UNKNOWN = auto()     # Unclassifiable


@dataclass
class ClassificationResult:
    """Result of signal classification."""
    signal_class: SignalClass
    confidence: float
    evidence: Dict[str, Any]
    
    @property
    def is_structured(self) -> bool:
        """Returns True if signal shows structure (not noise or unknown)."""
        return self.signal_class not in (SignalClass.NOISE, SignalClass.UNKNOWN)
    
    @property
    def is_conscious_like(self) -> bool:
        """Returns True if signal shows consciousness-like patterns."""
        return self.signal_class in (SignalClass.INFLUENCE, SignalClass.ATTRACTOR)


def classify_signal(
    metrics: Dict[str, float],
    thresholds: Optional[Dict[str, float]] = None
) -> ClassificationResult:
    """
    Classify signal based on computed metrics.
    
    Expected metrics keys:
        - lyapunov: Lyapunov exponent (chaos indicator)
        - hurst: Hurst exponent (memory/persistence)
        - diffusion_exponent: MSD alpha (diffusion type)
        - spectral_entropy: (optional) Frequency domain randomness
        - autocorrelation: (optional) Time domain structure
        - agency_score: (optional) Goal-directedness
    
    Args:
        metrics: Dictionary of computed metrics
        thresholds: Optional custom thresholds for classification
        
    Returns:
        ClassificationResult with class, confidence, and evidence
    """
    classifier = SignalClassifier(thresholds)
    return classifier.classify(metrics)


class SignalClassifier:
    """Configurable classifier with custom thresholds."""
    
    def __init__(self, thresholds: Optional[Dict] = None):
        """
        Initialize classifier with thresholds.
        
        Args:
            thresholds: Custom thresholds (uses defaults if None)
        """
        self.thresholds = thresholds or self._default_thresholds()
    
    @staticmethod
    def _default_thresholds() -> Dict[str, float]:
        """Default classification thresholds."""
        return {
            # Lyapunov thresholds
            'lyapunov_chaotic': 0.3,      # λ > 0.3 = chaotic
            'lyapunov_stable': -0.1,       # λ < -0.1 = stable/attractor
            
            # Hurst thresholds
            'hurst_trending': 0.6,         # H > 0.6 = persistent/trending
            'hurst_mean_reverting': 0.4,   # H < 0.4 = mean-reverting
            
            # Diffusion exponent thresholds
            'diffusion_superdiffusive': 1.3,  # α > 1.3 = superdiffusive
            'diffusion_subdiffusive': 0.7,    # α < 0.7 = subdiffusive
            'diffusion_ballistic': 1.8,       # α > 1.8 = ballistic
            
            # Entropy thresholds
            'entropy_random': 0.8,            # High entropy = random
            'entropy_structured': 0.3,        # Low entropy = structured
            
            # Agency thresholds
            'agency_high': 0.6,               # High agency score
            
            # Confidence thresholds
            'min_confidence': 0.5,
        }
    
    def classify(self, metrics: Dict[str, float]) -> ClassificationResult:
        """
        Classify signal based on metrics.
        
        Args:
            metrics: Dictionary of computed metrics
            
        Returns:
            ClassificationResult
        """
        evidence = {}
        scores = {}
        
        # Extract metrics
        lyapunov = metrics.get('lyapunov', 0.0)
        hurst = metrics.get('hurst', 0.5)
        diffusion_exp = metrics.get('diffusion_exponent', 1.0)
        spectral_entropy = metrics.get('spectral_entropy', 0.5)
        agency = metrics.get('agency_score', 0.0)
        
        # Classification logic
        
        # 1. Check for CHAOTIC behavior
        if lyapunov > self.thresholds['lyapunov_chaotic']:
            scores['CHAOTIC'] = min(1.0, lyapunov / self.thresholds['lyapunov_chaotic'])
            evidence['chaotic_lyapunov'] = lyapunov
        
        # 2. Check for ATTRACTOR behavior
        if lyapunov < self.thresholds['lyapunov_stable']:
            attractor_score = abs(lyapunov) / abs(self.thresholds['lyapunov_stable'])
            scores['ATTRACTOR'] = min(1.0, attractor_score)
            evidence['stable_lyapunov'] = lyapunov
        
        # 3. Check for DRIFT (trending)
        if hurst > self.thresholds['hurst_trending']:
            drift_score = (hurst - 0.5) / (self.thresholds['hurst_trending'] - 0.5)
            scores['DRIFT'] = min(1.0, drift_score)
            evidence['trending_hurst'] = hurst
        
        # 4. Check for ANOMALOUS diffusion
        if diffusion_exp > self.thresholds['diffusion_superdiffusive']:
            anomaly_score = (diffusion_exp - 1.0) / (self.thresholds['diffusion_superdiffusive'] - 1.0)
            scores['ANOMALOUS'] = min(1.0, anomaly_score)
            evidence['anomalous_diffusion'] = diffusion_exp
        elif diffusion_exp < self.thresholds['diffusion_subdiffusive']:
            anomaly_score = (1.0 - diffusion_exp) / (1.0 - self.thresholds['diffusion_subdiffusive'])
            scores['ANOMALOUS'] = min(1.0, anomaly_score)
            evidence['subdiffusive'] = diffusion_exp
        
        # 5. Check for PERIODIC (would need autocorrelation analysis)
        # Skip for now as it requires additional analysis
        
        # 6. Check for INFLUENCE (consciousness-like multi-indicator agreement)
        influence_indicators = 0
        influence_score = 0.0
        
        if lyapunov < self.thresholds['lyapunov_stable']:
            influence_indicators += 1
            influence_score += 0.25
        
        if hurst > 0.55:  # Some memory
            influence_indicators += 1
            influence_score += 0.25
        
        if agency > self.thresholds['agency_high']:
            influence_indicators += 1
            influence_score += 0.3
        
        if spectral_entropy < self.thresholds['entropy_structured']:
            influence_indicators += 1
            influence_score += 0.2
        
        if influence_indicators >= 3:
            scores['INFLUENCE'] = influence_score
            evidence['influence_indicators'] = influence_indicators
        
        # 7. Check for NOISE (random walk)
        if (abs(lyapunov) < 0.1 and 
            abs(hurst - 0.5) < 0.1 and 
            abs(diffusion_exp - 1.0) < 0.2):
            noise_score = 1.0 - max(
                abs(lyapunov) / 0.1,
                abs(hurst - 0.5) / 0.1,
                abs(diffusion_exp - 1.0) / 0.2
            )
            scores['NOISE'] = noise_score
            evidence['random_walk_indicators'] = {
                'lyapunov': lyapunov,
                'hurst': hurst,
                'diffusion': diffusion_exp
            }
        
        # Select best classification
        if not scores:
            return ClassificationResult(
                signal_class=SignalClass.UNKNOWN,
                confidence=0.0,
                evidence={'metrics': metrics}
            )
        
        # Find highest score
        best_class_name = max(scores, key=scores.get)
        best_score = scores[best_class_name]
        
        # Convert to enum
        signal_class = SignalClass[best_class_name]
        
        return ClassificationResult(
            signal_class=signal_class,
            confidence=best_score,
            evidence=evidence
        )
    
    def classify_with_all_scores(
        self,
        metrics: Dict[str, float]
    ) -> Dict[SignalClass, float]:
        """
        Return scores for all signal classes.
        
        Useful for uncertainty quantification.
        
        Args:
            metrics: Dictionary of computed metrics
            
        Returns:
            Dictionary mapping each SignalClass to its score
        """
        result = self.classify(metrics)
        
        # Re-run classification to get all scores
        # (In production, would refactor to compute all scores once)
        
        all_scores = {}
        for signal_class in SignalClass:
            # This is a simplified version
            # Full implementation would compute all scores simultaneously
            all_scores[signal_class] = 0.0
        
        # Set the classified one
        all_scores[result.signal_class] = result.confidence
        
        return all_scores
