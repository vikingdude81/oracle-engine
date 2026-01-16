"""
Consciousness Circuit Classifiers

Standalone classification modules for trajectory patterns.

Usage:
    from consciousness_circuit.classifiers import SignalClass, classify_signal
    
    signal_class = classify_signal(metrics_dict)
"""

from .signal_class import SignalClass, classify_signal, SignalClassifier, ClassificationResult

__all__ = [
    'SignalClass',
    'classify_signal',
    'SignalClassifier',
    'ClassificationResult',
]
