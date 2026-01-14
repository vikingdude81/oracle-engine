"""
Consciousness Circuit v3.0
==========================

Measure meta-cognitive signatures ("consciousness-like" activations) in transformer LLMs.
Detects how much a model's hidden states resemble reflective, uncertain, multi-perspective reasoning
versus quick, automatic responses.

Quick Start:
    from consciousness_circuit import measure_consciousness
    result = measure_consciousness(model, tokenizer, "What is consciousness?")
    print(f"Score: {result.score:.3f}")

Per-Token Analysis:
    from consciousness_circuit import ConsciousnessVisualizer
    viz = ConsciousnessVisualizer()
    trajectory = viz.measure_per_token(model, tokenizer, "Let me think about this...")
    trajectory.plot()  # Interactive visualization

Full API:
    from consciousness_circuit import UniversalCircuit, ValidationBasedDiscovery
    
    # Measure with auto-detection
    circuit = UniversalCircuit()
    result = circuit.measure(model, tokenizer, prompt)
    
    # Discover new circuit for a model
    discovery = ValidationBasedDiscovery(model, tokenizer)
    circuit = discovery.discover()

Author: VFD-Org
License: MIT
Version: 3.0.0
"""

from .circuit import ConsciousnessCircuit, CONSCIOUS_DIMS_V2_1, remap_dimensions
from .analysis import analyze_dimension_activations, compare_models
from .discover import DimensionDiscovery, DiscoveredCircuit, compare_architectures
from .universal import UniversalCircuit, UniversalResult, measure_consciousness
from .visualization import (
    TokenTrajectory,
    ComparisonResult, 
    ConsciousnessVisualizer,
    create_interactive_dashboard,
)

# Lazy import for discover_validated (requires model to be loaded)
def _get_validation_discovery():
    from .discover_validated import ValidationBasedDiscovery
    return ValidationBasedDiscovery

__version__ = "3.0.0"
__all__ = [
    # Universal API (recommended)
    "UniversalCircuit",
    "UniversalResult",
    "measure_consciousness",
    # Visualization
    "TokenTrajectory",
    "ComparisonResult",
    "ConsciousnessVisualizer",
    "create_interactive_dashboard",
    # Discovery tools
    "DimensionDiscovery",
    "DiscoveredCircuit",
    "compare_architectures",
    # Legacy/core circuit
    "ConsciousnessCircuit",
    "CONSCIOUS_DIMS_V2_1", 
    "remap_dimensions",
    # Analysis tools
    "analyze_dimension_activations",
    "compare_models",
]
