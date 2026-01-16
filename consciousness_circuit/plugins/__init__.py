"""
Consciousness Circuit Plugins

Standalone plugin modules for analysis and intervention.

Usage:
    from consciousness_circuit.plugins import AttractorLockPlugin, CoherenceBoostPlugin
    
    attractor = AttractorLockPlugin()
    if attractor.should_intervene(metrics):
        modified = attractor.intervene(hidden_states, metrics)
"""

from .base import AnalysisPlugin, InterventionPlugin, TrainingPlugin, PluginResult
from .attractor_lock import AttractorLockPlugin

__all__ = [
    'AnalysisPlugin',
    'InterventionPlugin',
    'TrainingPlugin',
    'PluginResult',
    'AttractorLockPlugin',
]
