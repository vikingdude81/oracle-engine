"""
Signal Classifiers Package
===========================

Signal classification for dynamical systems.
Fully standalone - can copy signal_class.py independently.

Usage:
    from consciousness_circuit.classifiers import SignalClass, classify_signal
    
    metrics = {'lyapunov': 0.5, 'hurst': 0.6}
    result = classify_signal(metrics)
    print(result.signal_class)
"""

from .signal_class import (
    SignalClass,
    ClassificationResult,
    classify_signal,
    SignalClassifier,
)


__all__ = [
    "SignalClass",
    "ClassificationResult",
    "classify_signal",
    "SignalClassifier",
]
