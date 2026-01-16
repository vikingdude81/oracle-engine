"""
Plugin Architecture for Consciousness Analysis
==============================================

Extensible plugin system for analysis and intervention.

Analysis Plugins (existing):
- TrajectoryPlugin: MSD and motion analysis
- ChaosPlugin: Lyapunov and Hurst exponents
- AgencyPlugin: Goal-directedness metrics

Intervention Plugins (new):
- AttractorLockPlugin: Stabilize chaos by nudging toward attractors
- CoherenceBoostPlugin: Maintain memory by injecting early context
- GoalDirectorPlugin: Enhance agency by amplifying direction

Base Classes:
- AnalysisPlugin: For computing metrics
- InterventionPlugin: For modifying hidden states
- PluginRegistry: For managing multiple plugins
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np

# Import base classes
from .base import (
    PluginResult,
    AnalysisPlugin,
    InterventionPlugin,
    PluginRegistry,
)

# Import intervention plugins
from .attractor_lock import AttractorLockPlugin, AttractorMemory
from .coherence_boost import CoherenceBoostPlugin
from .goal_director import GoalDirectorPlugin


# Original analysis plugins (for backward compatibility)
class TrajectoryPlugin(AnalysisPlugin):
    """Plugin for trajectory analysis (MSD, ballistic/diffusive motion)."""
    
    def __init__(self):
        super().__init__("trajectory")
    
    def analyze(self, hidden_states: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Analyze trajectory patterns in hidden states.
        
        Returns:
            - msd: Mean squared displacement values
            - trajectory_class: Classification (ballistic, diffusive, confined)
            - diffusion_coefficient: Measure of spreading
        """
        from ..helios_metrics import compute_msd_from_trajectory, classify_trajectory
        
        msd_values = compute_msd_from_trajectory(hidden_states)
        trajectory_class = classify_trajectory(msd_values)
        
        # Compute diffusion coefficient from early MSD
        if len(msd_values) > 2:
            diffusion_coef = msd_values[1] / 2  # D = MSD / 2t for small t
        else:
            diffusion_coef = 0.0
        
        return {
            "msd": msd_values,
            "trajectory_class": trajectory_class,
            "diffusion_coefficient": float(diffusion_coef),
        }


class ChaosPlugin(AnalysisPlugin):
    """Plugin for chaos and predictability analysis (Lyapunov, Hurst exponents)."""
    
    def __init__(self):
        super().__init__("chaos")
    
    def analyze(self, hidden_states: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Analyze chaos and predictability in hidden states.
        
        Returns:
            - lyapunov: Lyapunov exponent (chaos measure)
            - hurst: Hurst exponent (persistence/anti-persistence)
            - signal_class: Signal classification
        """
        from ..helios_metrics import (
            compute_lyapunov_exponent,
            compute_hurst_exponent,
            verify_signal,
        )
        
        lyapunov = compute_lyapunov_exponent(hidden_states)
        hurst = compute_hurst_exponent(hidden_states)
        signal_class = verify_signal(hidden_states, lyapunov, hurst)
        
        return {
            "lyapunov": float(lyapunov),
            "hurst": float(hurst),
            "signal_class": signal_class,
        }


class AgencyPlugin(AnalysisPlugin):
    """Plugin for goal-directedness and agency analysis."""
    
    def __init__(self):
        super().__init__("agency")
    
    def analyze(self, hidden_states: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Analyze goal-directedness and agency in hidden states.
        
        Returns:
            - agency_score: Measure of goal-directed behavior
            - attractor_strength: How strongly converging to attractor
            - goal_directedness: Overall goal pursuit metric
        """
        from ..tame_metrics import compute_agency_score, detect_attractor_convergence
        
        agency_score = compute_agency_score(hidden_states)
        attractor_strength, is_converging = detect_attractor_convergence(hidden_states)
        
        return {
            "agency_score": float(agency_score),
            "attractor_strength": float(attractor_strength),
            "is_converging": bool(is_converging),
            "goal_directedness": float(agency_score * attractor_strength),
        }


__all__ = [
    # Base classes
    "PluginResult",
    "AnalysisPlugin",
    "InterventionPlugin",
    "PluginRegistry",
    # Analysis plugins
    "TrajectoryPlugin",
    "ChaosPlugin",
    "AgencyPlugin",
    # Intervention plugins
    "AttractorLockPlugin",
    "AttractorMemory",
    "CoherenceBoostPlugin",
    "GoalDirectorPlugin",
]
