"""
Goal Director Plugin - FULLY STANDALONE
=======================================

Enhances agency and goal-directedness in trajectories.
Can be copied to any project - only requires numpy.

Usage:
    from goal_director import GoalDirectorPlugin
    
    plugin = GoalDirectorPlugin(agency_threshold=0.3, direction_strength=0.1)
    
    if plugin.should_intervene(metrics):
        modified_states = plugin.intervene(hidden_states, metrics)

Dependencies: numpy only
"""

import numpy as np
from typing import Dict, Any, Optional


class GoalDirectorPlugin:
    """
    Enhances agency when goal-directedness is low.
    Amplifies consistent movement patterns to increase purposefulness.
    
    When agency score is low, this plugin identifies and amplifies
    any existing directional trends in the hidden states to make
    behavior more goal-directed.
    
    Example:
        >>> plugin = GoalDirectorPlugin(agency_threshold=0.3)
        >>> 
        >>> metrics = {'agency': 0.2}  # Low agency
        >>> if plugin.should_intervene(metrics):
        ...     # Amplify directional movement
        ...     modified = plugin.intervene(hidden_states, metrics)
    """
    
    def __init__(self,
                 agency_threshold: float = 0.3,
                 direction_strength: float = 0.1,
                 momentum: float = 0.9):
        """
        Initialize goal director plugin.
        
        Args:
            agency_threshold: Agency below which to intervene
            direction_strength: Strength of directional amplification
            momentum: Momentum for directional smoothing (0 to 1)
        """
        self.agency_threshold = agency_threshold
        self.direction_strength = direction_strength
        self.momentum = momentum
        self.intervention_count = 0
    
    def should_intervene(self, metrics: Dict[str, Any]) -> bool:
        """
        Check if intervention is needed based on agency score.
        
        Args:
            metrics: Dictionary containing 'agency' key
        
        Returns:
            True if agency is below threshold
        """
        agency = metrics.get('agency', metrics.get('agency_score', 0.5))
        return agency < self.agency_threshold
    
    def intervene(self,
                 hidden_states: np.ndarray,
                 metrics: Dict[str, Any]) -> np.ndarray:
        """
        Amplify directional movement to increase goal-directedness.
        
        Args:
            hidden_states: Current hidden states [seq_len, hidden_dim]
            metrics: Dictionary of metrics (includes agency)
        
        Returns:
            Modified hidden states with enhanced directionality
        """
        if len(hidden_states) < 3:
            return hidden_states
        
        modified = hidden_states.copy()
        
        # Compute velocity (movement direction)
        velocities = np.diff(hidden_states, axis=0)
        
        # Compute smoothed velocity (dominant direction)
        smoothed_velocity = velocities[0].copy()
        
        for i in range(1, len(velocities)):
            # Apply momentum to smooth velocity
            smoothed_velocity = (
                self.momentum * smoothed_velocity +
                (1 - self.momentum) * velocities[i]
            )
            
            # Amplify this direction
            modified[i + 1] = (
                hidden_states[i + 1] +
                self.direction_strength * smoothed_velocity
            )
        
        self.intervention_count += 1
        return modified
    
    def reset(self):
        """Reset intervention count."""
        self.intervention_count = 0
    
    def __repr__(self):
        return (
            f"GoalDirectorPlugin("
            f"threshold={self.agency_threshold}, "
            f"strength={self.direction_strength}, "
            f"interventions={self.intervention_count})"
        )


__all__ = ["GoalDirectorPlugin"]


if __name__ == "__main__":
    # Self-test
    print("Goal Director Plugin - Standalone Tests")
    print("=" * 50)
    
    np.random.seed(42)
    
    # Test 1: Intervention detection
    print("\n1. Intervention Detection:")
    plugin = GoalDirectorPlugin(agency_threshold=0.3)
    
    metrics_high = {'agency': 0.8}
    metrics_low = {'agency': 0.2}
    
    print(f"   High agency (0.8): {plugin.should_intervene(metrics_high)}")
    print(f"   Low agency (0.2): {plugin.should_intervene(metrics_low)}")
    
    # Test 2: Apply intervention
    print("\n2. Apply Intervention:")
    
    # Create random walk (low agency)
    states = np.cumsum(np.random.randn(50, 64), axis=0)
    
    modified = plugin.intervene(states, metrics_low)
    
    # Measure directional consistency before/after
    def measure_consistency(traj):
        velocities = np.diff(traj, axis=0)
        norms = np.linalg.norm(velocities, axis=1, keepdims=True)
        norms[norms == 0] = 1
        normalized = velocities / norms
        mean_direction = np.mean(normalized, axis=0)
        return np.linalg.norm(mean_direction)
    
    consistency_orig = measure_consistency(states)
    consistency_modified = measure_consistency(modified)
    
    print(f"   Original consistency: {consistency_orig:.3f}")
    print(f"   Modified consistency: {consistency_modified:.3f}")
    print(f"   Agency improved: {consistency_modified > consistency_orig}")
    
    # Test 3: Multiple interventions
    print("\n3. Multiple Interventions:")
    plugin.reset()
    
    for i in range(5):
        plugin.intervene(states, metrics_low)
    
    print(f"   Intervention count: {plugin.intervention_count}")
    
    print("\n✓ All tests completed successfully!")
