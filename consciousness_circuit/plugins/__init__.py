"""
Plugin Architecture for Consciousness Analysis
==============================================

Extensible plugin system for adding new analysis methods to consciousness measurement.
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
import numpy as np


class AnalysisPlugin(ABC):
    """Base class for analysis plugins."""
    
    def __init__(self, name: str):
        self.name = name
    
    @abstractmethod
    def analyze(self, hidden_states: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Analyze hidden states and return metrics.
        
        Args:
            hidden_states: Hidden state trajectories [sequence_length, hidden_dim]
            **kwargs: Additional parameters specific to the plugin
            
        Returns:
            Dictionary of analysis results
        """
        raise NotImplementedError
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}')"


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
    "AnalysisPlugin",
    "TrajectoryPlugin",
    "ChaosPlugin",
    "AgencyPlugin",
]
