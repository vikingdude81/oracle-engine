"""
Attractor Lock Plugin

Standalone intervention plugin that detects chaotic states
and nudges toward known stable attractors.

Usage:
    from consciousness_circuit.plugins.attractor_lock import AttractorLockPlugin
    
    plugin = AttractorLockPlugin(lyapunov_threshold=0.3)
    
    if plugin.should_intervene(metrics):
        modified = plugin.intervene(hidden_states, metrics)
"""

import numpy as np
from typing import Dict, Any, List, Optional
from dataclasses import dataclass, field


# Note: No imports from other consciousness_circuit modules
# This file is fully standalone


@dataclass
class AttractorMemory:
    """Storage for learned attractors."""
    states: List[np.ndarray] = field(default_factory=list)
    qualities: List[float] = field(default_factory=list)
    max_size: int = 100
    
    def add(self, state: np.ndarray, quality: float):
        """
        Add a new attractor state.
        
        Args:
            state: Hidden state array
            quality: Quality score for this attractor
        """
        if len(self.states) >= self.max_size:
            # Remove lowest quality
            min_idx = np.argmin(self.qualities)
            self.states.pop(min_idx)
            self.qualities.pop(min_idx)
        
        self.states.append(state.copy())
        self.qualities.append(quality)
    
    def find_nearest(self, state: np.ndarray) -> Optional[np.ndarray]:
        """
        Find nearest attractor to given state.
        
        Args:
            state: Query state
            
        Returns:
            Nearest attractor state or None if memory is empty
        """
        if not self.states:
            return None
        
        distances = [np.linalg.norm(state - s) for s in self.states]
        return self.states[np.argmin(distances)]
    
    def find_best_quality(self) -> Optional[np.ndarray]:
        """
        Get highest quality attractor.
        
        Returns:
            Best attractor state or None if memory is empty
        """
        if not self.states:
            return None
        
        best_idx = np.argmax(self.qualities)
        return self.states[best_idx]
    
    def clear(self):
        """Clear all stored attractors."""
        self.states.clear()
        self.qualities.clear()


class AttractorLockPlugin:
    """
    Intervention plugin that stabilizes chaotic hidden states.
    
    When Lyapunov exponent exceeds threshold (indicating chaos),
    gently nudges hidden states toward known stable attractors.
    
    This is a standalone module with no dependencies on other
    consciousness_circuit components.
    """
    
    name = "attractor_lock"
    
    def __init__(
        self,
        lyapunov_threshold: float = 0.3,
        nudge_strength: float = 0.1,
        memory_size: int = 100,
        use_quality_based: bool = True
    ):
        """
        Initialize attractor lock plugin.
        
        Args:
            lyapunov_threshold: Lyapunov value above which to intervene
            nudge_strength: Strength of nudge toward attractor (0-1)
            memory_size: Maximum number of attractors to remember
            use_quality_based: If True, prefer highest quality attractor
        """
        self.lyapunov_threshold = lyapunov_threshold
        self.nudge_strength = nudge_strength
        self.memory = AttractorMemory(max_size=memory_size)
        self.use_quality_based = use_quality_based
        
        self.intervention_count = 0
        self.total_nudge_magnitude = 0.0
    
    def should_intervene(self, metrics: Dict[str, Any]) -> bool:
        """
        Decide whether to intervene based on Lyapunov exponent.
        
        Args:
            metrics: Dictionary containing 'lyapunov' key
            
        Returns:
            True if Lyapunov exceeds threshold
        """
        lyap = metrics.get('lyapunov', 0)
        return lyap > self.lyapunov_threshold
    
    def intervene(
        self,
        hidden_states: np.ndarray,
        metrics: Dict[str, Any]
    ) -> np.ndarray:
        """
        Nudge hidden states toward nearest known attractor.
        
        Args:
            hidden_states: Current hidden state array
            metrics: Current analysis metrics
            
        Returns:
            Modified hidden states
        """
        if self.use_quality_based:
            target = self.memory.find_best_quality()
        else:
            target = self.memory.find_nearest(hidden_states)
        
        if target is None:
            # No attractors learned yet, return unchanged
            return hidden_states
        
        # Gentle interpolation toward attractor
        alpha = self.nudge_strength
        modified = hidden_states * (1 - alpha) + target * alpha
        
        # Track statistics
        self.intervention_count += 1
        nudge_magnitude = np.linalg.norm(modified - hidden_states)
        self.total_nudge_magnitude += nudge_magnitude
        
        return modified
    
    def learn_attractor(
        self,
        hidden_states: np.ndarray,
        quality: float
    ):
        """
        Remember a high-quality state as an attractor.
        
        Args:
            hidden_states: Hidden state to remember
            quality: Quality score (0-1, higher is better)
        """
        if quality > 0.7:
            self.memory.add(hidden_states, quality)
    
    def reset_memory(self):
        """Clear all learned attractors."""
        self.memory.clear()
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get intervention statistics.
        
        Returns:
            Dictionary with statistics
        """
        avg_nudge = (
            self.total_nudge_magnitude / self.intervention_count
            if self.intervention_count > 0
            else 0.0
        )
        
        return {
            'intervention_count': self.intervention_count,
            'average_nudge_magnitude': avg_nudge,
            'attractors_stored': len(self.memory.states),
            'memory_capacity': self.memory.max_size,
        }
    
    def reset_statistics(self):
        """Reset intervention statistics."""
        self.intervention_count = 0
        self.total_nudge_magnitude = 0.0
    
    def save_attractors(self, filepath: str):
        """
        Save learned attractors to file.
        
        Args:
            filepath: Path to save file
        """
        import pickle
        
        data = {
            'states': self.memory.states,
            'qualities': self.memory.qualities,
            'config': {
                'lyapunov_threshold': self.lyapunov_threshold,
                'nudge_strength': self.nudge_strength,
                'memory_size': self.memory.max_size,
            }
        }
        
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load_attractors(self, filepath: str):
        """
        Load attractors from file.
        
        Args:
            filepath: Path to load file
        """
        import pickle
        
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.memory.states = data['states']
        self.memory.qualities = data['qualities']
        
        # Optionally update config
        config = data.get('config', {})
        if config:
            self.lyapunov_threshold = config.get('lyapunov_threshold', self.lyapunov_threshold)
            self.nudge_strength = config.get('nudge_strength', self.nudge_strength)
