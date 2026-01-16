"""
Consciousness Circuit Training Utilities

Standalone training modules for RLHF/DPO.

Usage:
    from consciousness_circuit.training import ConsciousnessRewardModel
    
    # Standalone usage
    reward = ConsciousnessRewardModel.compute_from_metrics(metrics)
"""

from .reward_model import ConsciousnessRewardModel, RewardConfig, RewardResult

__all__ = [
    'ConsciousnessRewardModel',
    'RewardConfig',
    'RewardResult',
]
