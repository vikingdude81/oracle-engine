"""
Consciousness Reward Model - FULLY STANDALONE
==============================================

Reward model for evaluating consciousness-aligned responses.
Computes weighted rewards from consciousness metrics.
Can be copied to any project - only requires numpy.

Usage:
    from reward_model import ConsciousnessRewardModel, RewardConfig
    
    config = RewardConfig(
        consciousness_weight=0.4,
        stability_weight=0.3,
        memory_weight=0.2,
        agency_weight=0.1
    )
    
    metrics = {
        'lyapunov': -0.5,
        'hurst': 0.6,
        'spectral_entropy': 0.7,
        'agency_score': 0.8
    }
    
    result = ConsciousnessRewardModel.compute_from_metrics(metrics, config)
    print(f"Reward: {result.reward:.3f}")
    print(result.explanation)

Dependencies: numpy only
"""

import numpy as np
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class RewardConfig:
    """Configuration for consciousness reward computation."""
    
    consciousness_weight: float = 0.4
    """Weight for consciousness indicators (entropy, complexity)."""
    
    stability_weight: float = 0.3
    """Weight for system stability (Lyapunov, controlled chaos)."""
    
    memory_weight: float = 0.2
    """Weight for memory and temporal structure (Hurst, autocorrelation)."""
    
    agency_weight: float = 0.1
    """Weight for agency and goal-directedness."""
    
    # Normalization ranges for metrics
    lyapunov_range: tuple = (-1.0, 1.0)
    """Expected range for Lyapunov exponent (negative to positive)."""
    
    hurst_range: tuple = (0.3, 0.7)
    """Expected range for Hurst exponent (anti-persistent to persistent)."""
    
    entropy_range: tuple = (0.0, 1.0)
    """Expected range for spectral entropy (deterministic to random)."""
    
    agency_range: tuple = (0.0, 1.0)
    """Expected range for agency score (passive to active)."""
    
    # Target values for consciousness
    target_lyapunov: float = 0.0
    """Target Lyapunov (edge of chaos, ~0)."""
    
    target_hurst: float = 0.5
    """Target Hurst (balanced, ~0.5)."""
    
    target_entropy: float = 0.5
    """Target entropy (mixed structure/randomness, ~0.5)."""
    
    target_agency: float = 0.7
    """Target agency (somewhat goal-directed, ~0.7)."""
    
    def __post_init__(self):
        """Validate configuration."""
        total_weight = (
            self.consciousness_weight +
            self.stability_weight +
            self.memory_weight +
            self.agency_weight
        )
        if not np.isclose(total_weight, 1.0, atol=0.01):
            raise ValueError(
                f"Weights must sum to 1.0, got {total_weight:.3f}"
            )
    
    def normalize_weights(self):
        """Normalize weights to sum to 1.0."""
        total = (
            self.consciousness_weight +
            self.stability_weight +
            self.memory_weight +
            self.agency_weight
        )
        if total > 0:
            self.consciousness_weight /= total
            self.stability_weight /= total
            self.memory_weight /= total
            self.agency_weight /= total


@dataclass
class RewardResult:
    """Result from consciousness reward computation."""
    
    reward: float
    """Overall reward score [0, 1]."""
    
    components: Dict[str, float]
    """Individual component scores."""
    
    raw_metrics: Dict[str, float]
    """Raw input metrics."""
    
    normalized_metrics: Dict[str, float]
    """Normalized metric values."""
    
    explanation: str = ""
    """Human-readable explanation of the reward."""
    
    def __post_init__(self):
        """Generate explanation if not provided."""
        if not self.explanation:
            self.explanation = self._generate_explanation()
    
    def _generate_explanation(self) -> str:
        """Generate human-readable explanation."""
        lines = [
            f"Reward Score: {self.reward:.3f}",
            "",
            "Component Breakdown:"
        ]
        
        for name, value in self.components.items():
            lines.append(f"  • {name}: {value:.3f}")
        
        lines.append("")
        lines.append("Metric Analysis:")
        
        # Analyze each metric
        if 'lyapunov' in self.raw_metrics:
            lyap = self.raw_metrics['lyapunov']
            if lyap < -0.5:
                lines.append(f"  • Lyapunov={lyap:.3f}: Stable, converging")
            elif lyap > 0.5:
                lines.append(f"  • Lyapunov={lyap:.3f}: Chaotic, diverging")
            else:
                lines.append(f"  • Lyapunov={lyap:.3f}: Edge of chaos (optimal)")
        
        if 'hurst' in self.raw_metrics:
            hurst = self.raw_metrics['hurst']
            if hurst < 0.4:
                lines.append(f"  • Hurst={hurst:.3f}: Anti-persistent, mean-reverting")
            elif hurst > 0.6:
                lines.append(f"  • Hurst={hurst:.3f}: Persistent, trending")
            else:
                lines.append(f"  • Hurst={hurst:.3f}: Random walk (optimal)")
        
        if 'spectral_entropy' in self.raw_metrics:
            entropy = self.raw_metrics['spectral_entropy']
            if entropy < 0.3:
                lines.append(f"  • Entropy={entropy:.3f}: Deterministic, low complexity")
            elif entropy > 0.7:
                lines.append(f"  • Entropy={entropy:.3f}: Random, high complexity")
            else:
                lines.append(f"  • Entropy={entropy:.3f}: Mixed (optimal)")
        
        if 'agency_score' in self.raw_metrics:
            agency = self.raw_metrics['agency_score']
            if agency < 0.3:
                lines.append(f"  • Agency={agency:.3f}: Passive, reactive")
            elif agency > 0.7:
                lines.append(f"  • Agency={agency:.3f}: Active, goal-directed")
            else:
                lines.append(f"  • Agency={agency:.3f}: Moderate agency")
        
        return "\n".join(lines)
    
    def __repr__(self):
        return f"RewardResult(reward={self.reward:.4f}, components={len(self.components)})"


class ConsciousnessRewardModel:
    """
    Reward model for evaluating consciousness-aligned LLM responses.
    
    Computes rewards based on consciousness metrics like stability,
    complexity, memory, and agency. Designed for RLHF-style training.
    
    This is a standalone implementation requiring only numpy.
    """
    
    @staticmethod
    def compute_from_metrics(
        metrics: Dict[str, float],
        config: Optional[RewardConfig] = None
    ) -> RewardResult:
        """
        Compute reward from consciousness metrics.
        
        Args:
            metrics: Dictionary of metric names to values. Expected keys:
                - 'lyapunov': Lyapunov exponent (stability)
                - 'hurst': Hurst exponent (memory)
                - 'spectral_entropy': Spectral entropy (complexity)
                - 'agency_score': Agency score (goal-directedness)
            config: Reward configuration (uses defaults if None)
        
        Returns:
            RewardResult with overall reward, components, and explanation
        
        Examples:
            >>> metrics = {
            ...     'lyapunov': -0.2,
            ...     'hurst': 0.55,
            ...     'spectral_entropy': 0.6,
            ...     'agency_score': 0.75
            ... }
            >>> result = ConsciousnessRewardModel.compute_from_metrics(metrics)
            >>> result.reward > 0.7  # Should be high for consciousness-aligned metrics
            True
            
            >>> # Poor metrics
            >>> poor_metrics = {
            ...     'lyapunov': -2.0,  # Too stable
            ...     'hurst': 0.9,      # Too persistent
            ...     'spectral_entropy': 0.1,  # Too deterministic
            ...     'agency_score': 0.2  # Too passive
            ... }
            >>> result = ConsciousnessRewardModel.compute_from_metrics(poor_metrics)
            >>> result.reward < 0.5  # Should be low
            True
        """
        if config is None:
            config = RewardConfig()
        
        # Extract and normalize metrics
        normalized = {}
        components = {}
        
        # Stability component (Lyapunov)
        if 'lyapunov' in metrics:
            lyap = metrics['lyapunov']
            # Normalize to [0, 1]
            lyap_norm = ConsciousnessRewardModel._normalize(
                lyap, config.lyapunov_range
            )
            # Compute distance from target (0 = edge of chaos)
            target_norm = ConsciousnessRewardModel._normalize(
                config.target_lyapunov, config.lyapunov_range
            )
            stability_score = 1.0 - abs(lyap_norm - target_norm)
            
            normalized['lyapunov'] = lyap_norm
            components['stability'] = stability_score * config.stability_weight
        else:
            components['stability'] = 0.0
        
        # Memory component (Hurst)
        if 'hurst' in metrics:
            hurst = metrics['hurst']
            # Normalize to [0, 1]
            hurst_norm = ConsciousnessRewardModel._normalize(
                hurst, config.hurst_range
            )
            # Compute distance from target (0.5 = random walk)
            target_norm = ConsciousnessRewardModel._normalize(
                config.target_hurst, config.hurst_range
            )
            memory_score = 1.0 - abs(hurst_norm - target_norm)
            
            normalized['hurst'] = hurst_norm
            components['memory'] = memory_score * config.memory_weight
        else:
            components['memory'] = 0.0
        
        # Consciousness component (Entropy)
        if 'spectral_entropy' in metrics:
            entropy = metrics['spectral_entropy']
            # Already in [0, 1]
            entropy_norm = np.clip(entropy, 0.0, 1.0)
            # Compute distance from target (0.5 = mixed)
            consciousness_score = 1.0 - abs(entropy_norm - config.target_entropy)
            
            normalized['spectral_entropy'] = entropy_norm
            components['consciousness'] = consciousness_score * config.consciousness_weight
        else:
            components['consciousness'] = 0.0
        
        # Agency component
        if 'agency_score' in metrics:
            agency = metrics['agency_score']
            # Already in [0, 1]
            agency_norm = np.clip(agency, 0.0, 1.0)
            # Compute distance from target
            agency_score = 1.0 - abs(agency_norm - config.target_agency)
            
            normalized['agency_score'] = agency_norm
            components['agency'] = agency_score * config.agency_weight
        else:
            components['agency'] = 0.0
        
        # Compute overall reward
        reward = sum(components.values())
        reward = np.clip(reward, 0.0, 1.0)
        
        return RewardResult(
            reward=float(reward),
            components=components,
            raw_metrics=metrics.copy(),
            normalized_metrics=normalized
        )
    
    @staticmethod
    def _normalize(value: float, value_range: tuple) -> float:
        """
        Normalize value to [0, 1] range.
        
        Args:
            value: Value to normalize
            value_range: (min, max) tuple
        
        Returns:
            Normalized value in [0, 1]
        """
        min_val, max_val = value_range
        if max_val == min_val:
            return 0.5
        normalized = (value - min_val) / (max_val - min_val)
        return np.clip(normalized, 0.0, 1.0)
    
    @staticmethod
    def compare_responses(
        metrics_a: Dict[str, float],
        metrics_b: Dict[str, float],
        config: Optional[RewardConfig] = None
    ) -> tuple:
        """
        Compare two responses based on consciousness metrics.
        
        Args:
            metrics_a: Metrics for response A
            metrics_b: Metrics for response B
            config: Reward configuration
        
        Returns:
            Tuple of (reward_a, reward_b, preferred_index)
            where preferred_index is 0 for A, 1 for B
        
        Examples:
            >>> metrics_a = {'lyapunov': 0.0, 'hurst': 0.5, 'spectral_entropy': 0.5, 'agency_score': 0.7}
            >>> metrics_b = {'lyapunov': -2.0, 'hurst': 0.9, 'spectral_entropy': 0.1, 'agency_score': 0.2}
            >>> reward_a, reward_b, preferred = ConsciousnessRewardModel.compare_responses(metrics_a, metrics_b)
            >>> preferred  # Should prefer A
            0
        """
        result_a = ConsciousnessRewardModel.compute_from_metrics(metrics_a, config)
        result_b = ConsciousnessRewardModel.compute_from_metrics(metrics_b, config)
        
        preferred_index = 0 if result_a.reward >= result_b.reward else 1
        
        return result_a, result_b, preferred_index


__all__ = [
    "RewardConfig",
    "RewardResult",
    "ConsciousnessRewardModel",
]


if __name__ == "__main__":
    # Self-test examples
    print("Consciousness Reward Model - Standalone Tests")
    print("=" * 50)
    
    # Test 1: Optimal consciousness metrics
    print("\n1. Optimal Consciousness Metrics:")
    metrics = {
        'lyapunov': 0.0,      # Edge of chaos
        'hurst': 0.5,         # Random walk
        'spectral_entropy': 0.5,  # Mixed structure
        'agency_score': 0.7   # Goal-directed
    }
    config = RewardConfig()
    result = ConsciousnessRewardModel.compute_from_metrics(metrics, config)
    print(f"   Reward: {result.reward:.3f}")
    print(f"   Components: {result.components}")
    assert result.reward > 0.9, f"Expected high reward, got {result.reward:.3f}"
    
    # Test 2: Poor metrics (too stable/deterministic)
    print("\n2. Poor Metrics (Too Stable/Deterministic):")
    poor_metrics = {
        'lyapunov': -2.0,     # Too stable
        'hurst': 0.9,         # Too persistent
        'spectral_entropy': 0.1,  # Too deterministic
        'agency_score': 0.2   # Too passive
    }
    result = ConsciousnessRewardModel.compute_from_metrics(poor_metrics, config)
    print(f"   Reward: {result.reward:.3f}")
    print(f"   Components: {result.components}")
    assert result.reward < 0.6, f"Expected low reward, got {result.reward:.3f}"
    
    # Test 3: Poor metrics (too chaotic/random)
    print("\n3. Poor Metrics (Too Chaotic/Random):")
    chaotic_metrics = {
        'lyapunov': 2.0,      # Too chaotic
        'hurst': 0.2,         # Anti-persistent
        'spectral_entropy': 0.95,  # Too random
        'agency_score': 0.1   # Very passive
    }
    result = ConsciousnessRewardModel.compute_from_metrics(chaotic_metrics, config)
    print(f"   Reward: {result.reward:.3f}")
    print(f"   Components: {result.components}")
    assert result.reward < 0.6, f"Expected low reward, got {result.reward:.3f}"
    
    # Test 4: Compare responses
    print("\n4. Compare Two Responses:")
    good_metrics = {
        'lyapunov': -0.1,
        'hurst': 0.52,
        'spectral_entropy': 0.55,
        'agency_score': 0.75
    }
    bad_metrics = {
        'lyapunov': -1.5,
        'hurst': 0.85,
        'spectral_entropy': 0.2,
        'agency_score': 0.3
    }
    result_a, result_b, preferred = ConsciousnessRewardModel.compare_responses(
        good_metrics, bad_metrics, config
    )
    print(f"   Response A reward: {result_a.reward:.3f}")
    print(f"   Response B reward: {result_b.reward:.3f}")
    print(f"   Preferred: {'A' if preferred == 0 else 'B'}")
    assert preferred == 0, "Should prefer response A"
    
    # Test 5: Custom weights
    print("\n5. Custom Weight Configuration:")
    custom_config = RewardConfig(
        consciousness_weight=0.5,
        stability_weight=0.3,
        memory_weight=0.1,
        agency_weight=0.1
    )
    result = ConsciousnessRewardModel.compute_from_metrics(metrics, custom_config)
    print(f"   Reward: {result.reward:.3f}")
    print(f"   Components: {result.components}")
    
    # Test 6: Partial metrics
    print("\n6. Partial Metrics (Only Some Available):")
    partial_metrics = {
        'spectral_entropy': 0.6,
        'agency_score': 0.7
    }
    result = ConsciousnessRewardModel.compute_from_metrics(partial_metrics, config)
    print(f"   Reward: {result.reward:.3f}")
    print(f"   Components: {result.components}")
    
    # Test 7: Explanation generation
    print("\n7. Reward Explanation:")
    print(result.explanation)
    
    # Test 8: Weight normalization
    print("\n8. Weight Normalization:")
    # Create config with unnormalized weights (bypass validation)
    unnormalized_config = RewardConfig.__new__(RewardConfig)
    unnormalized_config.consciousness_weight = 2.0
    unnormalized_config.stability_weight = 1.5
    unnormalized_config.memory_weight = 1.0
    unnormalized_config.agency_weight = 0.5
    unnormalized_config.lyapunov_range = (-1.0, 1.0)
    unnormalized_config.hurst_range = (0.3, 0.7)
    unnormalized_config.entropy_range = (0.0, 1.0)
    unnormalized_config.agency_range = (0.0, 1.0)
    unnormalized_config.target_lyapunov = 0.0
    unnormalized_config.target_hurst = 0.5
    unnormalized_config.target_entropy = 0.5
    unnormalized_config.target_agency = 0.7
    
    unnormalized_config.normalize_weights()
    print(f"   Normalized weights: c={unnormalized_config.consciousness_weight:.2f}, "
          f"s={unnormalized_config.stability_weight:.2f}, "
          f"m={unnormalized_config.memory_weight:.2f}, "
          f"a={unnormalized_config.agency_weight:.2f}")
    total = (unnormalized_config.consciousness_weight + 
             unnormalized_config.stability_weight +
             unnormalized_config.memory_weight + 
             unnormalized_config.agency_weight)
    assert np.isclose(total, 1.0), f"Weights should sum to 1.0, got {total}"
    
    print("\n✓ All tests completed successfully!")
