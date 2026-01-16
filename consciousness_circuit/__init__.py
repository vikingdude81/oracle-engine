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
from .trajectory_wrapper import ConsciousnessTrajectoryAnalyzer, TrajectoryAnalysisResult
from .helios_metrics import (
    compute_lyapunov_exponent,
    compute_hurst_exponent,
    compute_msd_from_trajectory,
    SignalClass,
    verify_signal,
)
from .tame_metrics import TAMEMetrics, compute_agency_score, detect_attractor_convergence

# Modular metrics (standalone)
from .metrics import (
    # Lyapunov
    LyapunovResult,
    compute_lyapunov,
    LyapunovAnalyzer,
    # Hurst
    HurstResult,
    compute_hurst,
    # MSD
    MSDResult,
    compute_msd,
    compute_diffusion_exponent,
    classify_motion,
    # Entropy
    SpectralEntropyResult,
    RunsTestResult,
    AutocorrelationResult,
    compute_spectral_entropy,
    compute_runs_test,
    compute_autocorrelation,
    # Agency
    AgencyResult,
    compute_tame_metrics,
)

# Classifiers
from .classifiers import (
    ClassificationResult,
    classify_signal,
    SignalClassifier,
)

# Plugins (analysis and intervention)
from .plugins import (
    PluginResult,
    AnalysisPlugin,
    InterventionPlugin,
    PluginRegistry,
    # Intervention plugins
    AttractorLockPlugin,
    CoherenceBoostPlugin,
    GoalDirectorPlugin,
)

# Training utilities
from .training import (
    RewardConfig,
    RewardResult,
    ConsciousnessRewardModel,
    PreferencePair,
    generate_preference_pairs,
    rank_responses,
)

# Analyzers
from .analyzers import (
    TrajectoryAnalysisResult as ModularTrajectoryResult,
)

# Benchmarks
from .benchmarks import (
    ModelProfiler,
    ProfileResult,
    get_test_suite,
    get_full_benchmark,
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
    # Trajectory analysis
    "ConsciousnessTrajectoryAnalyzer",
    "TrajectoryAnalysisResult",
    # Helios metrics (legacy)
    "compute_lyapunov_exponent",
    "compute_hurst_exponent",
    "compute_msd_from_trajectory",
    "SignalClass",
    "verify_signal",
    # TAME metrics (legacy)
    "TAMEMetrics",
    "compute_agency_score",
    "detect_attractor_convergence",
    # Modular metrics (standalone)
    "LyapunovResult",
    "compute_lyapunov",
    "LyapunovAnalyzer",
    "HurstResult",
    "compute_hurst",
    "MSDResult",
    "compute_msd",
    "compute_diffusion_exponent",
    "classify_motion",
    "SpectralEntropyResult",
    "RunsTestResult",
    "AutocorrelationResult",
    "compute_spectral_entropy",
    "compute_runs_test",
    "compute_autocorrelation",
    "AgencyResult",
    "compute_tame_metrics",
    # Classifiers
    "ClassificationResult",
    "classify_signal",
    "SignalClassifier",
    # Plugins
    "PluginResult",
    "AnalysisPlugin",
    "InterventionPlugin",
    "PluginRegistry",
    "AttractorLockPlugin",
    "CoherenceBoostPlugin",
    "GoalDirectorPlugin",
    # Training utilities
    "RewardConfig",
    "RewardResult",
    "ConsciousnessRewardModel",
    "PreferencePair",
    "generate_preference_pairs",
    "rank_responses",
    # Analyzers
    "ModularTrajectoryResult",
    # Benchmarks
    "ModelProfiler",
    "ProfileResult",
    "get_test_suite",
    "get_full_benchmark",
]
