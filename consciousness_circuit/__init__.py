"""
Consciousness Circuit v3.5
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

Standalone Usage (numpy/scipy only — no torch/transformers required):
    from consciousness_circuit import compute_lyapunov, compute_hurst, classify_signal
    The metrics/, classifiers/, plugins/, training/, analyzers/, and benchmarks/
    subpackages import without the ML stack. The full measurement API
    (UniversalCircuit, measure_consciousness, ...) requires torch + transformers;
    importing those names without the ML stack raises an informative ImportError.

Author: VFD-Org
License: MIT
Version: 3.5.1
"""

# ==============================================================================
# Standalone layer (numpy/scipy only) — always available
# ==============================================================================

from .logging_config import get_logger, setup_logging, ExperimentLogger
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
    # Analysis plugins (standalone)
    CompressibilityPlugin,
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

# ==============================================================================
# Full-stack layer (requires torch + transformers) — optional
# ==============================================================================

FULL_STACK_AVAILABLE = True
_FULL_STACK_IMPORT_ERROR = None

try:
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
        # v3.2 scoring improvements
        length_normalization_factor,
        compute_dimension_diversity,
        detect_anomalies,
        compute_confidence,
        entropy_weight_tokens,
        # v3.4 per-dimension normalization
        normalize_dimensions_adaptive,
    )
    from .visualization import (
        TokenTrajectory,
        ComparisonResult,
        ConsciousnessVisualizer,
        create_interactive_dashboard,
    )
    from .model_adapters import (
        ModelAdapter,
        HuggingFaceAdapter,
        NanoGPTAdapter,
        UnslothAdapter,
        create_adapter,
        get_hidden_states,
    )
    from .trajectory_wrapper import ConsciousnessTrajectoryAnalyzer, TrajectoryAnalysisResult
except ImportError as _e:  # torch/transformers not installed
    FULL_STACK_AVAILABLE = False
    _FULL_STACK_IMPORT_ERROR = _e

    def __getattr__(name):
        """Give an informative error when full-stack names are accessed without torch/transformers."""
        _full_stack_names = {
            "ConsciousnessCircuit", "CONSCIOUS_DIMS_V2_1", "remap_dimensions",
            "analyze_dimension_activations", "compare_models",
            "DimensionDiscovery", "DiscoveredCircuit", "compare_architectures",
            "UniversalCircuit", "UniversalResult", "measure_consciousness",
            "CachedUniversalCircuit", "get_adaptive_layer_fraction", "get_ensemble_layers",
            "length_normalization_factor", "compute_dimension_diversity",
            "detect_anomalies", "compute_confidence", "entropy_weight_tokens",
            "normalize_dimensions_adaptive",
            "TokenTrajectory", "ComparisonResult", "ConsciousnessVisualizer",
            "create_interactive_dashboard",
            "ModelAdapter", "HuggingFaceAdapter", "NanoGPTAdapter", "UnslothAdapter",
            "create_adapter", "get_hidden_states",
            "ConsciousnessTrajectoryAnalyzer", "TrajectoryAnalysisResult",
        }
        if name in _full_stack_names:
            raise ImportError(
                f"consciousness_circuit.{name} requires the full ML stack "
                f"(torch + transformers). Install with: pip install torch transformers\n"
                f"Original error: {_FULL_STACK_IMPORT_ERROR}"
            )
        raise AttributeError(f"module 'consciousness_circuit' has no attribute {name!r}")


# Lazy import for discover_validated (requires model to be loaded)
def _get_validation_discovery():
    from .discover_validated import ValidationBasedDiscovery
    return ValidationBasedDiscovery

__version__ = "3.5.1"
__all__ = [
    # Availability flag
    "FULL_STACK_AVAILABLE",
    # Universal API (recommended)
    "UniversalCircuit",
    "UniversalResult",
    "measure_consciousness",
    "CachedUniversalCircuit",
    "get_adaptive_layer_fraction",
    "get_ensemble_layers",
    # v3.2 scoring improvements
    "length_normalization_factor",
    "compute_dimension_diversity",
    "detect_anomalies",
    "compute_confidence",
    "entropy_weight_tokens",
    "normalize_dimensions_adaptive",
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
    "CompressibilityPlugin",
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
