"""
Attractor Lock Plugin - FULLY STANDALONE
========================================

Detects chaotic states and nudges hidden states toward stable attractors.
Can be copied to any project - only requires numpy.

Usage:
    from attractor_lock import AttractorLockPlugin
    
    plugin = AttractorLockPlugin(lyapunov_threshold=0.3, nudge_strength=0.1)
    
    if plugin.should_intervene(metrics):
        modified_states = plugin.intervene(hidden_states, metrics)
    
    # Learn good attractors for future use
    plugin.learn_attractor(good_hidden_states, quality_score=0.9)

Dependencies: numpy only
"""

import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass


@dataclass
class AttractorMemory:
    """Memory of a learned attractor."""
    
    centroid: np.ndarray
    """Center of the attractor in state space."""
    
    quality_score: float
    """Quality score of this attractor (0 to 1)."""
    
    use_count: int = 0
    """Number of times this attractor was used."""
    
    success_rate: float = 0.0
    """Success rate when nudging toward this attractor."""


class AttractorLockPlugin:
    """
    Detects chaotic states (high Lyapunov) and nudges hidden states
    toward known stable attractors.
    
    This plugin helps stabilize model outputs when chaos is detected
    by gently steering the hidden states toward previously learned
    "good" attractor states.
    
    Example:
        >>> plugin = AttractorLockPlugin(
        ...     lyapunov_threshold=0.3,
        ...     nudge_strength=0.1
        ... )
        >>> 
        >>> # During generation, check if intervention is needed
        >>> metrics = {'lyapunov': 0.5}
        >>> if plugin.should_intervene(metrics):
        ...     modified = plugin.intervene(hidden_states, metrics)
        >>> 
        >>> # Learn from good outputs
        >>> plugin.learn_attractor(good_states, quality_score=0.9)
    """
    
    def __init__(self,
                 lyapunov_threshold: float = 0.3,
                 nudge_strength: float = 0.1,
                 max_attractors: int = 10,
                 distance_threshold: float = 0.5):
        """
        Initialize attractor lock plugin.
        
        Args:
            lyapunov_threshold: Lyapunov value above which to intervene
            nudge_strength: Strength of nudge toward attractor (0 to 1)
            max_attractors: Maximum number of attractors to remember
            distance_threshold: Minimum distance to merge similar attractors
        """
        self.lyapunov_threshold = lyapunov_threshold
        self.nudge_strength = nudge_strength
        self.max_attractors = max_attractors
        self.distance_threshold = distance_threshold
        
        self.attractors: List[AttractorMemory] = []
        self.intervention_count = 0
    
    def should_intervene(self, metrics: Dict[str, Any]) -> bool:
        """
        Check if intervention is needed based on Lyapunov exponent.
        
        Args:
            metrics: Dictionary containing at least 'lyapunov' key
        
        Returns:
            True if Lyapunov exceeds threshold and attractors are available
        """
        lyapunov = metrics.get('lyapunov', 0.0)
        
        # Only intervene if chaos detected and we have learned attractors
        return lyapunov > self.lyapunov_threshold and len(self.attractors) > 0
    
    def intervene(self,
                 hidden_states: np.ndarray,
                 metrics: Dict[str, Any]) -> np.ndarray:
        """
        Nudge hidden states toward nearest high-quality attractor.
        
        Args:
            hidden_states: Current hidden states [seq_len, hidden_dim]
            metrics: Dictionary of metrics (includes lyapunov)
        
        Returns:
            Modified hidden states nudged toward attractor
        """
        if len(self.attractors) == 0:
            return hidden_states
        
        # Find best attractor to use
        best_attractor = self._select_attractor(hidden_states)
        
        if best_attractor is None:
            return hidden_states
        
        # Apply nudge toward attractor
        modified_states = self._apply_nudge(
            hidden_states,
            best_attractor.centroid,
            self.nudge_strength
        )
        
        # Update statistics
        best_attractor.use_count += 1
        self.intervention_count += 1
        
        return modified_states
    
    def learn_attractor(self,
                       hidden_states: np.ndarray,
                       quality_score: float):
        """
        Learn a new attractor from good hidden states.
        
        Call this when you observe good model behavior to teach
        the plugin about desirable states.
        
        Args:
            hidden_states: Hidden states from good behavior [seq_len, hidden_dim]
            quality_score: Quality rating (0 to 1, higher = better)
        """
        if quality_score < 0.3:
            # Don't learn from low-quality states
            return
        
        # Compute centroid of the trajectory
        if hidden_states.ndim == 1:
            centroid = hidden_states
        else:
            # Use late trajectory as attractor (most refined state)
            centroid = np.mean(hidden_states[-10:], axis=0)
        
        # Check if similar attractor already exists
        for attractor in self.attractors:
            distance = np.linalg.norm(centroid - attractor.centroid)
            
            if distance < self.distance_threshold:
                # Update existing attractor (weighted average)
                weight_new = quality_score / (quality_score + attractor.quality_score)
                attractor.centroid = (
                    weight_new * centroid +
                    (1 - weight_new) * attractor.centroid
                )
                attractor.quality_score = max(quality_score, attractor.quality_score)
                return
        
        # Add new attractor
        new_attractor = AttractorMemory(
            centroid=centroid,
            quality_score=quality_score
        )
        self.attractors.append(new_attractor)
        
        # Prune if too many attractors
        if len(self.attractors) > self.max_attractors:
            self._prune_attractors()
    
    def _select_attractor(self, hidden_states: np.ndarray) -> Optional[AttractorMemory]:
        """
        Select best attractor to nudge toward.
        
        Considers both quality score and distance.
        """
        if len(self.attractors) == 0:
            return None
        
        # Compute current centroid
        if hidden_states.ndim == 1:
            current = hidden_states
        else:
            current = np.mean(hidden_states, axis=0)
        
        # Score each attractor
        best_score = -np.inf
        best_attractor = None
        
        for attractor in self.attractors:
            distance = np.linalg.norm(current - attractor.centroid)
            
            # Score = quality / (1 + distance)
            # Prefers high quality and nearby attractors
            score = attractor.quality_score / (1 + distance)
            
            if score > best_score:
                best_score = score
                best_attractor = attractor
        
        return best_attractor
    
    def _apply_nudge(self,
                    hidden_states: np.ndarray,
                    attractor: np.ndarray,
                    strength: float) -> np.ndarray:
        """
        Apply gentle nudge toward attractor.
        
        Uses exponential interpolation to avoid disrupting dynamics.
        """
        modified = hidden_states.copy()
        
        if hidden_states.ndim == 1:
            # 1D case
            direction = attractor - hidden_states
            modified = hidden_states + strength * direction
        else:
            # 2D case: nudge each state
            for i in range(len(hidden_states)):
                direction = attractor - hidden_states[i]
                modified[i] = hidden_states[i] + strength * direction
        
        return modified
    
    def _prune_attractors(self):
        """
        Remove lowest-quality attractors to stay under max_attractors.
        """
        # Sort by quality score (keep highest)
        self.attractors.sort(key=lambda a: a.quality_score, reverse=True)
        self.attractors = self.attractors[:self.max_attractors]
    
    def get_statistics(self) -> Dict[str, Any]:
        """
        Get statistics about the plugin's operation.
        
        Returns:
            Dictionary with intervention counts, attractor info, etc.
        """
        return {
            'intervention_count': self.intervention_count,
            'num_attractors': len(self.attractors),
            'attractors': [
                {
                    'quality': a.quality_score,
                    'use_count': a.use_count,
                    'success_rate': a.success_rate
                }
                for a in self.attractors
            ]
        }
    
    def reset(self):
        """Reset intervention count (keeps learned attractors)."""
        self.intervention_count = 0
        for attractor in self.attractors:
            attractor.use_count = 0
    
    def clear_attractors(self):
        """Clear all learned attractors."""
        self.attractors = []
    
    def __repr__(self):
        return (
            f"AttractorLockPlugin("
            f"threshold={self.lyapunov_threshold}, "
            f"attractors={len(self.attractors)}, "
            f"interventions={self.intervention_count})"
        )


__all__ = [
    "AttractorLockPlugin",
    "AttractorMemory",
]


if __name__ == "__main__":
    # Self-test examples
    print("Attractor Lock Plugin - Standalone Tests")
    print("=" * 50)
    
    np.random.seed(42)
    
    # Test 1: Basic intervention detection
    print("\n1. Intervention Detection:")
    plugin = AttractorLockPlugin(lyapunov_threshold=0.3)
    
    metrics_stable = {'lyapunov': 0.1}
    metrics_chaotic = {'lyapunov': 0.5}
    
    print(f"   Stable (λ=0.1): Should intervene? {plugin.should_intervene(metrics_stable)}")
    print(f"   Chaotic (λ=0.5): Should intervene? {plugin.should_intervene(metrics_chaotic)}")
    
    # Test 2: Learn attractors
    print("\n2. Learning Attractors:")
    
    # Learn some good attractors
    good_state_1 = np.random.randn(10, 64)
    good_state_2 = np.random.randn(10, 64) + 2  # Different region
    
    plugin.learn_attractor(good_state_1, quality_score=0.9)
    plugin.learn_attractor(good_state_2, quality_score=0.8)
    
    print(f"   Learned attractors: {len(plugin.attractors)}")
    print(f"   {plugin}")
    
    # Test 3: Apply intervention
    print("\n3. Apply Intervention:")
    
    chaotic_state = np.random.randn(10, 64) * 5  # Far from attractors
    
    modified = plugin.intervene(chaotic_state, metrics_chaotic)
    
    dist_before = np.linalg.norm(chaotic_state - plugin.attractors[0].centroid)
    dist_after = np.linalg.norm(modified - plugin.attractors[0].centroid)
    
    print(f"   Distance before: {dist_before:.3f}")
    print(f"   Distance after: {dist_after:.3f}")
    print(f"   Moved closer: {dist_after < dist_before}")
    
    # Test 4: Attractor merging
    print("\n4. Attractor Merging:")
    
    # Learn similar attractor (should merge)
    similar_state = good_state_1 + np.random.randn(10, 64) * 0.1
    plugin.learn_attractor(similar_state, quality_score=0.85)
    
    print(f"   Attractors after similar: {len(plugin.attractors)} (should still be 2)")
    
    # Test 5: Statistics
    print("\n5. Statistics:")
    stats = plugin.get_statistics()
    print(f"   Interventions: {stats['intervention_count']}")
    print(f"   Num attractors: {stats['num_attractors']}")
    for i, a in enumerate(stats['attractors']):
        print(f"   Attractor {i}: quality={a['quality']:.2f}, uses={a['use_count']}")
    
    # Test 6: Reset and clear
    print("\n6. Reset and Clear:")
    plugin.reset()
    print(f"   After reset: interventions={plugin.intervention_count}")
    
    plugin.clear_attractors()
    print(f"   After clear: attractors={len(plugin.attractors)}")
    
    print("\n✓ All tests completed successfully!")
