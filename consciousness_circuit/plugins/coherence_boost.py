"""
Coherence Boost Plugin - FULLY STANDALONE
=========================================

Boosts coherence when Hurst exponent drops (model losing memory).
Injects context from earlier in reasoning to maintain consistency.
Can be copied to any project - only requires numpy.

Usage:
    from coherence_boost import CoherenceBoostPlugin
    
    plugin = CoherenceBoostPlugin(hurst_threshold=0.4, boost_strength=0.15)
    
    if plugin.should_intervene(metrics):
        modified_states = plugin.intervene(hidden_states, metrics)

Dependencies: numpy only
"""

import numpy as np
from typing import Dict, Any, Optional


class CoherenceBoostPlugin:
    """
    Boosts coherence when Hurst exponent drops (model losing memory).
    Injects context from earlier in reasoning chain.
    
    When the model's hidden states show low Hurst exponent (< 0.4),
    it indicates loss of long-term memory. This plugin reinforces
    coherence by blending in earlier hidden states.
    
    Example:
        >>> plugin = CoherenceBoostPlugin(hurst_threshold=0.4)
        >>> 
        >>> metrics = {'hurst': 0.3}  # Low memory
        >>> if plugin.should_intervene(metrics):
        ...     # Boost coherence by injecting early context
        ...     modified = plugin.intervene(hidden_states, metrics)
    """
    
    def __init__(self,
                 hurst_threshold: float = 0.4,
                 boost_strength: float = 0.15,
                 context_window: int = 10):
        """
        Initialize coherence boost plugin.
        
        Args:
            hurst_threshold: Hurst value below which to intervene
            boost_strength: Strength of context injection (0 to 1)
            context_window: Size of early context to use
        """
        self.hurst_threshold = hurst_threshold
        self.boost_strength = boost_strength
        self.context_window = context_window
        self.intervention_count = 0
    
    def should_intervene(self, metrics: Dict[str, Any]) -> bool:
        """
        Check if intervention is needed based on Hurst exponent.
        
        Args:
            metrics: Dictionary containing at least 'hurst' key
        
        Returns:
            True if Hurst is below threshold (memory loss detected)
        """
        hurst = metrics.get('hurst', 0.5)
        return hurst < self.hurst_threshold
    
    def intervene(self,
                 hidden_states: np.ndarray,
                 metrics: Dict[str, Any]) -> np.ndarray:
        """
        Boost coherence by injecting early context.
        
        Args:
            hidden_states: Current hidden states [seq_len, hidden_dim]
            metrics: Dictionary of metrics (includes hurst)
        
        Returns:
            Modified hidden states with boosted coherence
        """
        if len(hidden_states) < self.context_window:
            # Not enough context, can't intervene
            return hidden_states
        
        modified = hidden_states.copy()
        
        # Extract early context (first context_window states)
        early_context = hidden_states[:self.context_window]
        context_summary = np.mean(early_context, axis=0)
        
        # Inject context into later states
        # Use exponentially decaying strength
        for i in range(self.context_window, len(hidden_states)):
            # Decay strength over distance
            distance = i - self.context_window
            decay = np.exp(-distance / len(hidden_states))
            strength = self.boost_strength * decay
            
            # Blend with context
            modified[i] = (
                (1 - strength) * hidden_states[i] +
                strength * context_summary
            )
        
        self.intervention_count += 1
        return modified
    
    def reset(self):
        """Reset intervention count."""
        self.intervention_count = 0
    
    def __repr__(self):
        return (
            f"CoherenceBoostPlugin("
            f"threshold={self.hurst_threshold}, "
            f"strength={self.boost_strength}, "
            f"interventions={self.intervention_count})"
        )


__all__ = ["CoherenceBoostPlugin"]


if __name__ == "__main__":
    # Self-test
    print("Coherence Boost Plugin - Standalone Tests")
    print("=" * 50)
    
    np.random.seed(42)
    
    # Test 1: Intervention detection
    print("\n1. Intervention Detection:")
    plugin = CoherenceBoostPlugin(hurst_threshold=0.4)
    
    metrics_high = {'hurst': 0.6}
    metrics_low = {'hurst': 0.3}
    
    print(f"   High memory (H=0.6): {plugin.should_intervene(metrics_high)}")
    print(f"   Low memory (H=0.3): {plugin.should_intervene(metrics_low)}")
    
    # Test 2: Apply intervention
    print("\n2. Apply Intervention:")
    
    # Create states with drifting pattern
    states = np.random.randn(50, 64).cumsum(axis=0)
    
    # Early states have pattern
    states[:10] *= 2
    
    modified = plugin.intervene(states, metrics_low)
    
    # Check if later states now have more influence from early states
    early_mean = np.mean(states[:10], axis=0)
    late_orig = states[-1]
    late_modified = modified[-1]
    
    dist_orig = np.linalg.norm(late_orig - early_mean)
    dist_modified = np.linalg.norm(late_modified - early_mean)
    
    print(f"   Distance to early context (original): {dist_orig:.3f}")
    print(f"   Distance to early context (modified): {dist_modified:.3f}")
    print(f"   Coherence improved: {dist_modified < dist_orig}")
    
    # Test 3: Multiple interventions
    print("\n3. Multiple Interventions:")
    plugin.reset()
    
    for i in range(5):
        plugin.intervene(states, metrics_low)
    
    print(f"   Intervention count: {plugin.intervention_count}")
    
    print("\n✓ All tests completed successfully!")
