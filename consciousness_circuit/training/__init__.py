"""
Training Utilities
==================

Standalone modules for consciousness-aligned LLM training.

Modules:
    - reward_model: Compute rewards from consciousness metrics
    - preference_generator: Generate preference pairs for RLHF

All modules are fully standalone and require only numpy.
"""

from .reward_model import (
    RewardConfig,
    RewardResult,
    ConsciousnessRewardModel,
)

from .preference_generator import (
    PreferencePair,
    compute_consciousness_reward,
    generate_preference_pairs,
    rank_responses,
    filter_pairs_by_quality,
    balance_preference_dataset,
)

__all__ = [
    # Reward model
    "RewardConfig",
    "RewardResult",
    "ConsciousnessRewardModel",
    # Preference generator
    "PreferencePair",
    "compute_consciousness_reward",
    "generate_preference_pairs",
    "rank_responses",
    "filter_pairs_by_quality",
    "balance_preference_dataset",
]
