"""
Consciousness Reward Model

Standalone reward computation for RLHF/DPO training.
Can be used independently or integrated with training frameworks.

Usage:
    from consciousness_circuit.training.reward_model import ConsciousnessRewardModel
    
    # Standalone usage (provide your own metrics)
    reward = ConsciousnessRewardModel.compute_from_metrics(metrics_dict)
    
    # With analyzer (if available)
    model = ConsciousnessRewardModel(analyzer=my_analyzer)
    reward = model.compute_reward(prompt, response)
"""

import numpy as np
from typing import Dict, Any, Optional, Callable, Tuple
from dataclasses import dataclass


@dataclass
class RewardConfig:
    """Configuration for reward computation."""
    consciousness_weight: float = 0.4
    stability_weight: float = 0.2  # Penalize high Lyapunov
    memory_weight: float = 0.2      # Reward high Hurst
    agency_weight: float = 0.2      # Reward goal-directedness
    attractor_bonus: float = 1.2    # Multiplier for attractor lock
    chaos_penalty: float = 0.8      # Multiplier for chaotic states
    
    def validate(self):
        """Validate that weights sum to 1.0."""
        total = (
            self.consciousness_weight + 
            self.stability_weight + 
            self.memory_weight + 
            self.agency_weight
        )
        
        if abs(total - 1.0) > 0.01:
            raise ValueError(f"Weights must sum to 1.0, got {total}")
    
    @property
    def weight_dict(self) -> Dict[str, float]:
        """Get weights as dictionary."""
        return {
            'consciousness': self.consciousness_weight,
            'stability': self.stability_weight,
            'memory': self.memory_weight,
            'agency': self.agency_weight,
        }


@dataclass
class RewardResult:
    """Detailed reward breakdown."""
    total: float
    components: Dict[str, float]
    bonuses: Dict[str, float]
    penalties: Dict[str, float]
    metrics_used: Dict[str, float]
    
    def __str__(self) -> str:
        return f"Reward: {self.total:.3f} (components: {self.components})"


class ConsciousnessRewardModel:
    """
    Compute training rewards from consciousness metrics.
    
    Can be used standalone (provide metrics dict) or with an analyzer.
    
    Standalone mode allows integration with any training framework
    without coupling to the full consciousness circuit.
    """
    
    def __init__(
        self,
        config: Optional[RewardConfig] = None,
        analyzer: Optional[Any] = None  # Optional: ConsciousnessTrajectoryAnalyzer
    ):
        """
        Initialize reward model.
        
        Args:
            config: Reward configuration (uses defaults if None)
            analyzer: Optional analyzer for computing metrics from text
        """
        self.config = config or RewardConfig()
        self.config.validate()
        self.analyzer = analyzer
    
    @staticmethod
    def compute_from_metrics(
        metrics: Dict[str, float],
        config: Optional[RewardConfig] = None
    ) -> float:
        """
        Compute reward from pre-computed metrics.
        
        This is the standalone interface - works without any dependencies.
        
        Expected metrics keys:
            - consciousness_score: 0-1 (optional, default 0.5)
            - lyapunov: float (optional, default 0)
            - hurst: 0-1 (optional, default 0.5)
            - agency_score: 0-1 (optional, default 0.5)
            - trajectory_class: str (optional, for bonuses)
        
        Args:
            metrics: Dictionary of computed metrics
            config: Optional reward configuration
            
        Returns:
            Reward value (0-1)
        """
        cfg = config or RewardConfig()
        
        # Extract metrics with defaults
        consciousness = metrics.get('consciousness_score', 0.5)
        lyapunov = metrics.get('lyapunov', 0.0)
        hurst = metrics.get('hurst', 0.5)
        agency = metrics.get('agency_score', 0.5)
        trajectory_class = metrics.get('trajectory_class', '')
        
        # Compute components
        
        # Stability: penalize chaos (high positive Lyapunov)
        # Use sigmoid-like function: stability = 1 / (1 + |λ|)
        stability = 1.0 / (1.0 + abs(lyapunov)) if lyapunov > 0 else 1.0
        
        # Base reward (weighted sum)
        reward = (
            cfg.consciousness_weight * consciousness +
            cfg.stability_weight * stability +
            cfg.memory_weight * hurst +
            cfg.agency_weight * agency
        )
        
        # Apply bonuses/penalties
        
        # Attractor bonus (convergent behavior is good)
        if trajectory_class == 'ATTRACTOR':
            reward *= cfg.attractor_bonus
        
        # Chaos penalty (very chaotic behavior is bad)
        if lyapunov > 0.5:
            reward *= cfg.chaos_penalty
        
        # Clamp to valid range
        return float(np.clip(reward, 0.0, 1.0))
    
    def compute_reward(self, prompt: str, response: str) -> RewardResult:
        """
        Compute reward for a prompt-response pair.
        
        Requires analyzer to be set.
        
        Args:
            prompt: Input prompt
            response: Model response
            
        Returns:
            RewardResult with detailed breakdown
        """
        if self.analyzer is None:
            raise ValueError(
                "Analyzer required for text-based rewards. "
                "Use compute_from_metrics() for standalone mode."
            )
        
        # Analyze full text
        full_text = f"{prompt}\n{response}"
        metrics = self.analyzer.deep_analyze(full_text)
        
        # Extract values
        consciousness = metrics.get('consciousness_score', 0.5)
        lyapunov = metrics.get('lyapunov', 0.0)
        hurst = metrics.get('hurst', 0.5)
        agency = metrics.get('agency_score', 0.5)
        trajectory_class = metrics.get('trajectory_class', '')
        
        # Compute stability component
        stability = 1.0 / (1.0 + abs(lyapunov)) if lyapunov > 0 else 1.0
        
        # Component contributions
        components = {
            'consciousness': consciousness * self.config.consciousness_weight,
            'stability': stability * self.config.stability_weight,
            'memory': hurst * self.config.memory_weight,
            'agency': agency * self.config.agency_weight,
        }
        
        # Base reward
        base_reward = sum(components.values())
        
        # Bonuses and penalties
        bonuses = {}
        penalties = {}
        
        if trajectory_class == 'ATTRACTOR':
            bonuses['attractor'] = self.config.attractor_bonus - 1.0
        
        if lyapunov > 0.5:
            penalties['chaos'] = 1.0 - self.config.chaos_penalty
        
        # Apply multipliers
        total = base_reward
        
        for bonus in bonuses.values():
            total *= (1.0 + bonus)
        
        for penalty in penalties.values():
            total *= (1.0 - penalty)
        
        total = float(np.clip(total, 0.0, 1.0))
        
        return RewardResult(
            total=total,
            components=components,
            bonuses=bonuses,
            penalties=penalties,
            metrics_used=metrics
        )
    
    def compute_preference(
        self,
        prompt: str,
        response_a: str,
        response_b: str
    ) -> Tuple[str, str, float]:
        """
        Compare two responses for DPO training.
        
        Returns (chosen, rejected, margin) tuple.
        
        Args:
            prompt: Input prompt
            response_a: First response
            response_b: Second response
            
        Returns:
            Tuple of (chosen_response, rejected_response, reward_margin)
        """
        reward_a = self.compute_reward(prompt, response_a)
        reward_b = self.compute_reward(prompt, response_b)
        
        if reward_a.total >= reward_b.total:
            return (response_a, response_b, reward_a.total - reward_b.total)
        else:
            return (response_b, response_a, reward_b.total - reward_a.total)
    
    def compute_batch_rewards(
        self,
        prompts: list,
        responses: list
    ) -> list:
        """
        Compute rewards for a batch of prompt-response pairs.
        
        Args:
            prompts: List of prompts
            responses: List of responses
            
        Returns:
            List of reward values
        """
        if len(prompts) != len(responses):
            raise ValueError("prompts and responses must have same length")
        
        rewards = []
        
        for prompt, response in zip(prompts, responses):
            if self.analyzer:
                result = self.compute_reward(prompt, response)
                rewards.append(result.total)
            else:
                # Need metrics for standalone mode
                raise ValueError("Batch rewards require analyzer")
        
        return rewards
    
    def set_config(self, config: RewardConfig):
        """Update reward configuration."""
        config.validate()
        self.config = config


def create_reward_function(
    config: Optional[RewardConfig] = None
) -> Callable[[Dict[str, float]], float]:
    """
    Create a standalone reward function for integration with training frameworks.
    
    Returns a function that takes metrics dict and returns reward.
    
    Args:
        config: Optional reward configuration
        
    Returns:
        Reward function
    """
    def reward_fn(metrics: Dict[str, float]) -> float:
        return ConsciousnessRewardModel.compute_from_metrics(metrics, config)
    
    return reward_fn
