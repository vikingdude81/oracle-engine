"""
Trajectory Analysis for Consciousness Circuit
==============================================

Combines consciousness measurement with trajectory dynamics:
- Trajectory classification (attractor, chaotic, ballistic, etc.)
- Chaos metrics (Lyapunov, Hurst)
- Agency and goal-directedness

Designed to work with ANY transformer model.

Usage:
    >>> from consciousness_circuit.analyzers import ConsciousnessTrajectoryAnalyzer
    >>> analyzer = ConsciousnessTrajectoryAnalyzer()
    >>> analyzer.bind_model(model, tokenizer)
    >>> result = analyzer.deep_analyze("Let me think about this...")
    >>> print(result.interpretation())

Dependencies: numpy, consciousness_circuit.metrics, consciousness_circuit.classifiers
"""

import numpy as np
from typing import Dict, List, Optional, Any
from dataclasses import dataclass

from ..universal import UniversalCircuit
from ..visualization import ConsciousnessVisualizer, TokenTrajectory
from ..metrics import (
    compute_lyapunov_exponent,
    compute_hurst_exponent,
    compute_msd_from_trajectory,
    TAMEMetrics,
)
from ..classifiers import SignalClass, classify_signal


@dataclass
class TrajectoryAnalysisResult:
    """Complete result from trajectory analysis."""
    # Base consciousness measurement
    consciousness_score: float
    dimension_scores: Dict[str, float]
    
    # Trajectory dynamics
    trajectory_class: str  # ATTRACTOR, DRIFT, BALLISTIC, etc.
    lyapunov: float       # Chaos measure
    hurst: float          # Memory/persistence
    msd_values: np.ndarray  # Mean squared displacement
    
    # Agency and goal-directedness
    agency_score: float
    attractor_strength: float
    is_converging: bool
    goal_directedness: float
    trajectory_coherence: float
    
    # Per-token trajectory
    token_trajectory: Optional[TokenTrajectory] = None
    
    # Raw data
    hidden_states: Optional[np.ndarray] = None
    
    def interpretation(self) -> str:
        """Generate human-readable interpretation."""
        lines = []
        
        # Consciousness level
        if self.consciousness_score > 0.7:
            lines.append("🧠 HIGH consciousness - reflective, meta-cognitive processing")
        elif self.consciousness_score > 0.4:
            lines.append("🧠 MODERATE consciousness - some reflective processing")
        else:
            lines.append("🧠 LOW consciousness - automatic processing")
        
        # Trajectory dynamics
        traj_class = self.trajectory_class.lower()
        if traj_class == "attractor":
            lines.append("🎯 ATTRACTOR-LOCKED - converging to coherent reasoning pattern")
        elif traj_class == "ballistic":
            lines.append("🚀 BALLISTIC motion - directed, purposeful thought")
        elif traj_class == "chaotic":
            lines.append("🌀 CHAOTIC dynamics - exploring solution space")
        elif traj_class == "diffusive":
            lines.append("🔄 DIFFUSIVE motion - random walk, uncertain")
        elif traj_class == "drift":
            lines.append("💨 DRIFT - slow wandering, no clear direction")
        
        # Chaos/stability
        if self.lyapunov > 0.5:
            lines.append(f"⚡ HIGH chaos (λ={self.lyapunov:.3f}) - unstable, sensitive to perturbations")
        elif self.lyapunov < -0.3:
            lines.append(f"🔒 STABLE (λ={self.lyapunov:.3f}) - robust, converging")
        else:
            lines.append(f"⚖️  NEUTRAL (λ={self.lyapunov:.3f}) - balanced dynamics")
        
        # Memory/persistence
        if self.hurst > 0.6:
            lines.append(f"🧵 PERSISTENT (H={self.hurst:.3f}) - strong memory, trending")
        elif self.hurst < 0.4:
            lines.append(f"🎲 ANTI-PERSISTENT (H={self.hurst:.3f}) - mean-reverting, noisy")
        else:
            lines.append(f"🎯 RANDOM WALK (H={self.hurst:.3f}) - no long-term memory")
        
        # Agency
        if self.agency_score > 0.6:
            lines.append(f"🎮 HIGH agency ({self.agency_score:.3f}) - goal-directed behavior")
        elif self.agency_score > 0.3:
            lines.append(f"🎮 MODERATE agency ({self.agency_score:.3f}) - some goal-directed behavior")
        else:
            lines.append(f"🎮 LOW agency ({self.agency_score:.3f}) - reactive, not goal-directed")
        
        # Attractor convergence
        if self.is_converging:
            lines.append(f"🌊 CONVERGING to attractor (strength={self.attractor_strength:.3f})")
        
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        """Export to dictionary."""
        return {
            "consciousness_score": float(self.consciousness_score),
            "dimension_scores": self.dimension_scores,
            "trajectory_class": self.trajectory_class,
            "lyapunov": float(self.lyapunov),
            "hurst": float(self.hurst),
            "agency_score": float(self.agency_score),
            "attractor_strength": float(self.attractor_strength),
            "is_converging": bool(self.is_converging),
            "goal_directedness": float(self.goal_directedness),
            "trajectory_coherence": float(self.trajectory_coherence),
            "interpretation": self.interpretation(),
        }


class ConsciousnessTrajectoryAnalyzer:
    """
    Enhances consciousness measurement with trajectory dynamics.
    
    Detects:
    - Attractor lock (coherent reasoning)
    - Chaotic transitions (uncertainty/exploration)
    - Ballistic motion (directed thought)
    - Goal-directedness (agency)
    
    Example:
        >>> analyzer = ConsciousnessTrajectoryAnalyzer()
        >>> analyzer.bind_model(model, tokenizer)
        >>> result = analyzer.deep_analyze("Let me think about this...")
        >>> print(result.interpretation())
    """
    
    def __init__(self, model=None, tokenizer=None):
        """
        Initialize analyzer.
        
        Args:
            model: Optional HuggingFace model
            tokenizer: Optional HuggingFace tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
        self.circuit = UniversalCircuit()
        self.visualizer = ConsciousnessVisualizer(self.circuit)
    
    def bind_model(self, model, tokenizer):
        """
        Bind to any HuggingFace transformer model.
        
        Args:
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer
        """
        self.model = model
        self.tokenizer = tokenizer
    
    def deep_analyze(self, 
                    prompt: str,
                    include_per_token: bool = True) -> TrajectoryAnalysisResult:
        """
        Full consciousness + trajectory analysis.
        
        Args:
            prompt: Input text to analyze
            include_per_token: Include per-token trajectory
            
        Returns:
            TrajectoryAnalysisResult with all metrics
        """
        if self.model is None or self.tokenizer is None:
            raise ValueError("Model not bound. Call bind_model() first.")
        
        # Get per-token trajectory with hidden states
        token_trajectory = self.visualizer.measure_per_token(
            self.model, 
            self.tokenizer, 
            prompt
        )
        
        # Extract hidden states for trajectory analysis
        if token_trajectory.raw_activations is not None:
            hidden_states = token_trajectory.raw_activations
        else:
            # Fallback: get hidden states from measurement
            result = self.circuit.measure(
                self.model, 
                self.tokenizer, 
                prompt,
                return_hidden_states=True
            )
            hidden_states = np.array(result.hidden_states) if result.hidden_states else None
        
        # If we couldn't get hidden states, create dummy trajectory from scores
        if hidden_states is None:
            hidden_states = np.array(token_trajectory.scores).reshape(-1, 1)
        
        # Run trajectory analysis
        trajectory_metrics = self._analyze_trajectory(hidden_states)
        
        # Run TAME metrics
        tame = TAMEMetrics()
        tame_metrics = tame.compute_all(hidden_states)
        
        # Combine results
        result = TrajectoryAnalysisResult(
            consciousness_score=token_trajectory.mean_score,
            dimension_scores={},  # Could extract from token_trajectory
            trajectory_class=trajectory_metrics["trajectory_class"],
            lyapunov=trajectory_metrics["lyapunov"],
            hurst=trajectory_metrics["hurst"],
            msd_values=trajectory_metrics["msd"],
            agency_score=tame_metrics["agency_score"],
            attractor_strength=tame_metrics["attractor_strength"],
            is_converging=tame_metrics["is_converging"],
            goal_directedness=tame_metrics["goal_directedness"],
            trajectory_coherence=tame_metrics["trajectory_coherence"],
            token_trajectory=token_trajectory if include_per_token else None,
            hidden_states=hidden_states,
        )
        
        return result
    
    def _analyze_trajectory(self, hidden_states: np.ndarray) -> Dict[str, Any]:
        """
        Internal method to compute trajectory metrics.
        
        Args:
            hidden_states: Hidden state trajectory
            
        Returns:
            Dictionary of trajectory metrics
        """
        # Compute MSD
        msd_result = compute_msd_from_trajectory(hidden_states)
        msd = msd_result.msd if hasattr(msd_result, 'msd') else msd_result
        
        # Compute chaos metrics
        lyap_result = compute_lyapunov_exponent(hidden_states)
        lyapunov = lyap_result.exponent if hasattr(lyap_result, 'exponent') else lyap_result
        
        hurst_result = compute_hurst_exponent(hidden_states)
        hurst = hurst_result.exponent if hasattr(hurst_result, 'exponent') else hurst_result
        
        # Classify signal
        metrics = {
            'lyapunov': lyapunov,
            'hurst': hurst,
            'msd': msd if isinstance(msd, np.ndarray) else np.array([msd]),
        }
        classification = classify_signal(metrics)
        signal_class = classification.signal_class if hasattr(classification, 'signal_class') else str(classification)
        
        # Handle enum
        if hasattr(signal_class, 'value'):
            signal_class = signal_class.value
        
        return {
            "msd": msd,
            "trajectory_class": signal_class,
            "lyapunov": lyapunov,
            "hurst": hurst,
        }
    
    def analyze_batch(self, 
                     prompts: List[str],
                     include_per_token: bool = False) -> List[TrajectoryAnalysisResult]:
        """
        Batch analysis for efficiency.
        
        Args:
            prompts: List of prompts to analyze
            include_per_token: Include per-token trajectories
            
        Returns:
            List of TrajectoryAnalysisResult objects
        """
        results = []
        
        for prompt in prompts:
            try:
                result = self.deep_analyze(prompt, include_per_token=include_per_token)
                results.append(result)
            except Exception as e:
                # Log error but continue with batch
                print(f"Error analyzing prompt '{prompt[:50]}...': {e}")
                # Create empty result
                results.append(None)
        
        return results
    
    def compare_prompts(self, prompts: List[str]) -> Dict[str, Any]:
        """
        Compare trajectory dynamics across multiple prompts.
        
        Args:
            prompts: List of prompts to compare
            
        Returns:
            Dictionary with comparative analysis
        """
        results = self.analyze_batch(prompts, include_per_token=False)
        results = [r for r in results if r is not None]
        
        if not results:
            return {"error": "No successful analyses"}
        
        return {
            "num_prompts": len(results),
            "avg_consciousness": np.mean([r.consciousness_score for r in results]),
            "avg_lyapunov": np.mean([r.lyapunov for r in results]),
            "avg_hurst": np.mean([r.hurst for r in results]),
            "avg_agency": np.mean([r.agency_score for r in results]),
            "trajectory_classes": [r.trajectory_class for r in results],
            "converging_count": sum(r.is_converging for r in results),
        }


__all__ = [
    "ConsciousnessTrajectoryAnalyzer",
    "TrajectoryAnalysisResult",
]
