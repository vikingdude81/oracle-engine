"""
Consciousness Circuit v3.0 - Modular Consciousness Analysis Toolkit
===================================================================

Measure meta-cognitive signatures ("consciousness-like" activations) in transformer LLMs.
Detects how much a model's hidden states resemble reflective, uncertain, multi-perspective reasoning
versus quick, automatic responses.

Full Pipeline:
    from consciousness_circuit import measure_consciousness
    result = measure_consciousness(model, tokenizer, "What is consciousness?")
    print(f"Score: {result.score:.3f}")

Individual Components (Standalone):
    # Just metrics
    from consciousness_circuit.metrics import compute_lyapunov, compute_hurst
    lyap = compute_lyapunov(x, y)
    hurst = compute_hurst(sequence)
    
    # Just classification
    from consciousness_circuit.classifiers import SignalClass, classify_signal
    classification = classify_signal({'lyapunov': lyap, 'hurst': hurst})
    
    # Just plugins
    from consciousness_circuit.plugins import AttractorLockPlugin
    plugin = AttractorLockPlugin()
    
    # Just training
    from consciousness_circuit.training import ConsciousnessRewardModel
    reward = ConsciousnessRewardModel.compute_from_metrics(metrics)

Per-Token Analysis:
    from consciousness_circuit import ConsciousnessVisualizer
    viz = ConsciousnessVisualizer()
    trajectory = viz.measure_per_token(model, tokenizer, "Let me think about this...")
    trajectory.plot()  # Interactive visualization

Author: VFD-Org
License: MIT
Version: 3.0.0
"""

from .circuit import ConsciousnessCircuit, CONSCIOUS_DIMS_V2_1, remap_dimensions
from .analysis import analyze_dimension_activations, compare_models
from .discover import DimensionDiscovery, DiscoveredCircuit, compare_architectures
from .universal import (
    UniversalCircuit,
    UniversalResult,
    measure_consciousness,
    CachedUniversalCircuit,
    get_adaptive_layer_fraction,
    get_ensemble_layers,
)
from .visualization import (
    TokenTrajectory,
    ComparisonResult,
    ConsciousnessVisualizer,
    create_interactive_dashboard,
)
from .logging_config import get_logger, setup_logging, ExperimentLogger
from .model_adapters import (
    ModelAdapter,
    HuggingFaceAdapter,
    NanoGPTAdapter,
    UnslothAdapter,
    create_adapter,
    get_hidden_states,
)

# New modular components (v3.0)
from .metrics import (
    compute_lyapunov,
    compute_hurst,
    compute_msd,
    compute_agency_score,
)
from .classifiers import SignalClass, classify_signal
from .plugins import AttractorLockPlugin
from .training import ConsciousnessRewardModel

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
    "CachedUniversalCircuit",
    "get_adaptive_layer_fraction",
    "get_ensemble_layers",
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
    # Logging utilities
    "get_logger",
    "setup_logging",
    "ExperimentLogger",
    # Model adapters
    "ModelAdapter",
    "HuggingFaceAdapter",
    "NanoGPTAdapter",
    "UnslothAdapter",
    "create_adapter",
    "get_hidden_states",
    # Modular components (v3.0)
    "compute_lyapunov",
    "compute_hurst",
    "compute_msd",
    "compute_agency_score",
    "SignalClass",
    "classify_signal",
    "AttractorLockPlugin",
    "ConsciousnessRewardModel",
]
