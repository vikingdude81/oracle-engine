"""
Standalone Metrics Package
===========================

Each metric module can be copied independently to any project.
Zero coupling - only requires numpy.

Available Metrics:
- lyapunov: Lyapunov exponent (chaos/stability)
- hurst: Hurst exponent (memory/persistence)
- msd: Mean squared displacement (diffusion)
- entropy: Spectral entropy, runs test, autocorrelation
- agency: Goal-directedness and agentic behavior

Usage:
    # Import what you need
    from consciousness_circuit.metrics import compute_lyapunov, compute_hurst
    
    result_lyap = compute_lyapunov(trajectory)
    result_hurst = compute_hurst(time_series)
    
    # Or import modules
    from consciousness_circuit.metrics import lyapunov, hurst
    
    # Or copy individual files to another project
    # Each file works standalone with only numpy!
"""

# Lyapunov exponent
from .lyapunov import (
    LyapunovResult,
    compute_lyapunov,
    compute_lyapunov_exponent,  # Alias
    LyapunovAnalyzer,
)

# Hurst exponent
from .hurst import (
    HurstResult,
    compute_hurst,
    compute_hurst_exponent,  # Alias
)

# Mean squared displacement
from .msd import (
    MSDResult,
    compute_msd,
    compute_diffusion_exponent,
    compute_msd_from_trajectory,  # Alias for compatibility
    classify_motion,
)

# Entropy and randomness
from .entropy import (
    SpectralEntropyResult,
    RunsTestResult,
    AutocorrelationResult,
    compute_spectral_entropy,
    compute_runs_test,
    compute_autocorrelation,
)

# Agency and goal-directedness
from .agency import (
    AgencyResult,
    TAMEMetrics,
    compute_agency_score,
    compute_path_efficiency,
    detect_attractor_convergence,
    compute_trajectory_coherence,
    compute_tame_metrics,
)


__all__ = [
    # Lyapunov
    "LyapunovResult",
    "compute_lyapunov",
    "compute_lyapunov_exponent",
    "LyapunovAnalyzer",
    # Hurst
    "HurstResult",
    "compute_hurst",
    "compute_hurst_exponent",
    # MSD
    "MSDResult",
    "compute_msd",
    "compute_diffusion_exponent",
    "compute_msd_from_trajectory",
    "classify_motion",
    # Entropy
    "SpectralEntropyResult",
    "RunsTestResult",
    "AutocorrelationResult",
    "compute_spectral_entropy",
    "compute_runs_test",
    "compute_autocorrelation",
    # Agency
    "AgencyResult",
    "TAMEMetrics",
    "compute_agency_score",
    "compute_path_efficiency",
    "detect_attractor_convergence",
    "compute_trajectory_coherence",
    "compute_tame_metrics",
]
