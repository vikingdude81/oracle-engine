"""
Consciousness Circuit Metrics

Individual metric modules - each fully standalone.

Usage:
    # Import specific metric
    from consciousness_circuit.metrics.lyapunov import compute_lyapunov
    
    # Or import all
    from consciousness_circuit.metrics import (
        compute_lyapunov,
        compute_hurst,
        compute_msd,
        compute_agency_score
    )
"""

from .lyapunov import compute_lyapunov, compute_lyapunov_1d, LyapunovAnalyzer, LyapunovResult
from .hurst import compute_hurst, HurstAnalyzer, HurstResult
from .msd import compute_msd, compute_diffusion_exponent, MSDResult
from .entropy import compute_spectral_entropy, compute_runs_test, compute_autocorrelation
from .agency import compute_agency_score, compute_path_efficiency, TAMEMetrics, AgencyResult

__all__ = [
    # Lyapunov
    'compute_lyapunov',
    'compute_lyapunov_1d', 
    'LyapunovAnalyzer',
    'LyapunovResult',
    # Hurst
    'compute_hurst',
    'HurstAnalyzer',
    'HurstResult',
    # MSD
    'compute_msd',
    'compute_diffusion_exponent',
    'MSDResult',
    # Entropy
    'compute_spectral_entropy',
    'compute_runs_test',
    'compute_autocorrelation',
    # Agency
    'compute_agency_score',
    'compute_path_efficiency',
    'TAMEMetrics',
    'AgencyResult',
]
