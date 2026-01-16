"""
Analyzers Package
=================

High-level analyzers that combine multiple metrics and classifiers.

Available Analyzers:
- ConsciousnessTrajectoryAnalyzer: Full trajectory + consciousness analysis

Usage:
    from consciousness_circuit.analyzers import ConsciousnessTrajectoryAnalyzer
    
    analyzer = ConsciousnessTrajectoryAnalyzer()
    analyzer.bind_model(model, tokenizer)
    result = analyzer.deep_analyze("Let me think...")
    print(result.interpretation())
"""

from .trajectory import (
    ConsciousnessTrajectoryAnalyzer,
    TrajectoryAnalysisResult,
)


__all__ = [
    "ConsciousnessTrajectoryAnalyzer",
    "TrajectoryAnalysisResult",
]
