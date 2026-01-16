"""
Preference Pair Generator - FULLY STANDALONE
=============================================

Generate preference pairs for RLHF training from consciousness metrics.
Ranks responses by consciousness alignment and creates training pairs.
Can be copied to any project - only requires numpy.

Usage:
    from preference_generator import generate_preference_pairs, PreferencePair
    
    responses = ["response 1", "response 2", "response 3"]
    metrics_list = [
        {'lyapunov': 0.0, 'hurst': 0.5, 'spectral_entropy': 0.5, 'agency_score': 0.7},
        {'lyapunov': -2.0, 'hurst': 0.9, 'spectral_entropy': 0.1, 'agency_score': 0.2},
        {'lyapunov': 0.1, 'hurst': 0.52, 'spectral_entropy': 0.6, 'agency_score': 0.75}
    ]
    
    pairs = generate_preference_pairs(responses, metrics_list)
    for pair in pairs:
        print(f"Chosen: {pair.chosen_response}")
        print(f"Rejected: {pair.rejected_response}")
        print(f"Margin: {pair.preference_margin:.3f}")

Dependencies: numpy only
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Dict, Optional, Tuple, Any


@dataclass
class PreferencePair:
    """
    A preference pair for RLHF training.
    
    Contains chosen and rejected responses with their metrics,
    plus the preference margin (reward difference).
    """
    
    chosen_response: Any
    """The preferred response (higher consciousness alignment)."""
    
    rejected_response: Any
    """The less preferred response (lower consciousness alignment)."""
    
    chosen_metrics: Dict[str, float]
    """Consciousness metrics for chosen response."""
    
    rejected_metrics: Dict[str, float]
    """Consciousness metrics for rejected response."""
    
    chosen_reward: float
    """Computed reward for chosen response."""
    
    rejected_reward: float
    """Computed reward for rejected response."""
    
    preference_margin: float
    """Difference in rewards (chosen - rejected)."""
    
    chosen_index: int = -1
    """Original index of chosen response."""
    
    rejected_index: int = -1
    """Original index of rejected response."""
    
    @property
    def is_clear_preference(self) -> bool:
        """True if preference margin is significant (> 0.1)."""
        return self.preference_margin > 0.1
    
    @property
    def is_strong_preference(self) -> bool:
        """True if preference margin is strong (> 0.3)."""
        return self.preference_margin > 0.3
    
    @property
    def quality_category(self) -> str:
        """Category based on preference margin strength."""
        if self.preference_margin > 0.5:
            return "VERY_STRONG"
        elif self.preference_margin > 0.3:
            return "STRONG"
        elif self.preference_margin > 0.1:
            return "MODERATE"
        else:
            return "WEAK"
    
    def __repr__(self):
        return (f"PreferencePair(margin={self.preference_margin:.3f}, "
                f"chosen_reward={self.chosen_reward:.3f}, "
                f"rejected_reward={self.rejected_reward:.3f})")


def compute_consciousness_reward(metrics: Dict[str, float],
                                 weights: Optional[Dict[str, float]] = None) -> float:
    """
    Compute consciousness-aligned reward from metrics.
    
    This is a simplified reward function for standalone use.
    For more sophisticated reward computation, use the RewardModel class.
    
    Args:
        metrics: Dictionary of consciousness metrics
        weights: Optional custom weights for each metric
    
    Returns:
        Reward score in [0, 1]
    
    Examples:
        >>> metrics = {'lyapunov': 0.0, 'hurst': 0.5, 'spectral_entropy': 0.5, 'agency_score': 0.7}
        >>> reward = compute_consciousness_reward(metrics)
        >>> reward > 0.7
        True
    """
    if weights is None:
        weights = {
            'lyapunov': 0.3,
            'hurst': 0.2,
            'spectral_entropy': 0.3,
            'agency_score': 0.2
        }
    
    total_reward = 0.0
    total_weight = 0.0
    
    # Lyapunov: reward proximity to 0 (edge of chaos)
    if 'lyapunov' in metrics:
        lyap = metrics['lyapunov']
        # Normalize to [-1, 1] then compute distance from 0
        lyap_norm = np.clip(lyap, -1.0, 1.0)
        lyap_reward = 1.0 - abs(lyap_norm)
        total_reward += lyap_reward * weights.get('lyapunov', 0.0)
        total_weight += weights.get('lyapunov', 0.0)
    
    # Hurst: reward proximity to 0.5 (balanced)
    if 'hurst' in metrics:
        hurst = metrics['hurst']
        # Normalize to [0.3, 0.7] then compute distance from 0.5
        hurst_norm = (np.clip(hurst, 0.3, 0.7) - 0.3) / 0.4
        hurst_reward = 1.0 - 2.0 * abs(hurst_norm - 0.5)
        total_reward += hurst_reward * weights.get('hurst', 0.0)
        total_weight += weights.get('hurst', 0.0)
    
    # Entropy: reward moderate values (0.4-0.6)
    if 'spectral_entropy' in metrics:
        entropy = metrics['spectral_entropy']
        entropy_norm = np.clip(entropy, 0.0, 1.0)
        entropy_reward = 1.0 - 2.0 * abs(entropy_norm - 0.5)
        total_reward += entropy_reward * weights.get('spectral_entropy', 0.0)
        total_weight += weights.get('spectral_entropy', 0.0)
    
    # Agency: reward higher values (goal-directedness)
    if 'agency_score' in metrics:
        agency = metrics['agency_score']
        agency_norm = np.clip(agency, 0.0, 1.0)
        # Target ~0.7 for balanced agency
        agency_reward = 1.0 - abs(agency_norm - 0.7)
        total_reward += agency_reward * weights.get('agency_score', 0.0)
        total_weight += weights.get('agency_score', 0.0)
    
    # Normalize by total weight
    if total_weight > 0:
        total_reward /= total_weight
    
    return float(np.clip(total_reward, 0.0, 1.0))


def generate_preference_pairs(
    responses: List[Any],
    metrics_list: List[Dict[str, float]],
    min_margin: float = 0.0,
    max_pairs: Optional[int] = None,
    weights: Optional[Dict[str, float]] = None
) -> List[PreferencePair]:
    """
    Generate preference pairs from responses and their consciousness metrics.
    
    Ranks responses by consciousness alignment and creates pairs where
    the chosen response has higher consciousness metrics than rejected.
    
    Args:
        responses: List of response objects (strings, dicts, etc.)
        metrics_list: List of consciousness metrics for each response
        min_margin: Minimum reward difference to include pair (default: 0.0)
        max_pairs: Maximum number of pairs to generate (default: all)
        weights: Optional custom weights for reward computation
    
    Returns:
        List of PreferencePair objects, sorted by preference margin (descending)
    
    Examples:
        >>> responses = ["response A", "response B", "response C"]
        >>> metrics = [
        ...     {'lyapunov': 0.0, 'hurst': 0.5, 'spectral_entropy': 0.5, 'agency_score': 0.7},
        ...     {'lyapunov': -2.0, 'hurst': 0.9, 'spectral_entropy': 0.1, 'agency_score': 0.2},
        ...     {'lyapunov': 0.1, 'hurst': 0.52, 'spectral_entropy': 0.6, 'agency_score': 0.75}
        ... ]
        >>> pairs = generate_preference_pairs(responses, metrics)
        >>> len(pairs) > 0
        True
        >>> pairs[0].chosen_reward > pairs[0].rejected_reward
        True
    """
    if len(responses) != len(metrics_list):
        raise ValueError(
            f"Number of responses ({len(responses)}) must match "
            f"number of metric dicts ({len(metrics_list)})"
        )
    
    if len(responses) < 2:
        return []
    
    # Compute rewards for each response
    rewards = []
    for i, metrics in enumerate(metrics_list):
        reward = compute_consciousness_reward(metrics, weights)
        rewards.append((i, reward, responses[i], metrics))
    
    # Sort by reward (descending)
    rewards.sort(key=lambda x: x[1], reverse=True)
    
    # Generate all valid pairs
    pairs = []
    for i in range(len(rewards)):
        for j in range(i + 1, len(rewards)):
            chosen_idx, chosen_reward, chosen_response, chosen_metrics = rewards[i]
            rejected_idx, rejected_reward, rejected_response, rejected_metrics = rewards[j]
            
            margin = chosen_reward - rejected_reward
            
            # Only include pairs with sufficient margin
            if margin >= min_margin:
                pair = PreferencePair(
                    chosen_response=chosen_response,
                    rejected_response=rejected_response,
                    chosen_metrics=chosen_metrics,
                    rejected_metrics=rejected_metrics,
                    chosen_reward=chosen_reward,
                    rejected_reward=rejected_reward,
                    preference_margin=margin,
                    chosen_index=chosen_idx,
                    rejected_index=rejected_idx
                )
                pairs.append(pair)
    
    # Sort by margin (descending) - strongest preferences first
    pairs.sort(key=lambda p: p.preference_margin, reverse=True)
    
    # Limit number of pairs if requested
    if max_pairs is not None and len(pairs) > max_pairs:
        pairs = pairs[:max_pairs]
    
    return pairs


def rank_responses(
    responses: List[Any],
    metrics_list: List[Dict[str, float]],
    weights: Optional[Dict[str, float]] = None
) -> List[Tuple[int, float, Any, Dict[str, float]]]:
    """
    Rank responses by consciousness alignment.
    
    Args:
        responses: List of response objects
        metrics_list: List of consciousness metrics for each response
        weights: Optional custom weights for reward computation
    
    Returns:
        List of (index, reward, response, metrics) tuples, sorted by reward (descending)
    
    Examples:
        >>> responses = ["A", "B", "C"]
        >>> metrics = [
        ...     {'lyapunov': 0.0, 'hurst': 0.5, 'spectral_entropy': 0.5, 'agency_score': 0.7},
        ...     {'lyapunov': -2.0, 'hurst': 0.9, 'spectral_entropy': 0.1, 'agency_score': 0.2},
        ...     {'lyapunov': 0.1, 'hurst': 0.52, 'spectral_entropy': 0.6, 'agency_score': 0.75}
        ... ]
        >>> ranked = rank_responses(responses, metrics)
        >>> ranked[0][1] > ranked[-1][1]  # First has higher reward than last
        True
    """
    if len(responses) != len(metrics_list):
        raise ValueError(
            f"Number of responses ({len(responses)}) must match "
            f"number of metric dicts ({len(metrics_list)})"
        )
    
    # Compute rewards
    ranked = []
    for i, (response, metrics) in enumerate(zip(responses, metrics_list)):
        reward = compute_consciousness_reward(metrics, weights)
        ranked.append((i, reward, response, metrics))
    
    # Sort by reward (descending)
    ranked.sort(key=lambda x: x[1], reverse=True)
    
    return ranked


def filter_pairs_by_quality(
    pairs: List[PreferencePair],
    min_quality: str = "WEAK"
) -> List[PreferencePair]:
    """
    Filter preference pairs by quality category.
    
    Args:
        pairs: List of preference pairs
        min_quality: Minimum quality level to keep
            ("WEAK", "MODERATE", "STRONG", "VERY_STRONG")
    
    Returns:
        Filtered list of pairs meeting quality threshold
    
    Examples:
        >>> pairs = [
        ...     PreferencePair(
        ...         chosen_response="A", rejected_response="B",
        ...         chosen_metrics={}, rejected_metrics={},
        ...         chosen_reward=0.9, rejected_reward=0.3,
        ...         preference_margin=0.6
        ...     ),
        ...     PreferencePair(
        ...         chosen_response="C", rejected_response="D",
        ...         chosen_metrics={}, rejected_metrics={},
        ...         chosen_reward=0.6, rejected_reward=0.55,
        ...         preference_margin=0.05
        ...     )
        ... ]
        >>> strong_pairs = filter_pairs_by_quality(pairs, "STRONG")
        >>> len(strong_pairs)
        1
    """
    quality_order = {
        "WEAK": 0,
        "MODERATE": 1,
        "STRONG": 2,
        "VERY_STRONG": 3
    }
    
    if min_quality not in quality_order:
        raise ValueError(f"Invalid quality level: {min_quality}")
    
    min_level = quality_order[min_quality]
    
    return [
        pair for pair in pairs
        if quality_order[pair.quality_category] >= min_level
    ]


def balance_preference_dataset(
    pairs: List[PreferencePair],
    target_size: int,
    quality_distribution: Optional[Dict[str, float]] = None
) -> List[PreferencePair]:
    """
    Balance preference dataset by quality distribution.
    
    Samples pairs to achieve target size with desired quality distribution.
    
    Args:
        pairs: List of all preference pairs
        target_size: Desired number of pairs
        quality_distribution: Dict of quality -> proportion
            (default: balanced across all qualities)
    
    Returns:
        Balanced list of preference pairs
    
    Examples:
        >>> pairs = []
        >>> for margin in [0.05, 0.15, 0.35, 0.55, 0.65, 0.75]:
        ...     pair = PreferencePair(
        ...         chosen_response="", rejected_response="",
        ...         chosen_metrics={}, rejected_metrics={},
        ...         chosen_reward=0.7, rejected_reward=0.7-margin,
        ...         preference_margin=margin
        ...     )
        ...     pairs.append(pair)
        >>> balanced = balance_preference_dataset(pairs, 4)
        >>> len(balanced)
        4
    """
    if target_size >= len(pairs):
        return pairs.copy()
    
    # Group by quality
    by_quality = {
        "WEAK": [],
        "MODERATE": [],
        "STRONG": [],
        "VERY_STRONG": []
    }
    
    for pair in pairs:
        by_quality[pair.quality_category].append(pair)
    
    # Default to balanced distribution
    if quality_distribution is None:
        quality_distribution = {
            "WEAK": 0.2,
            "MODERATE": 0.3,
            "STRONG": 0.3,
            "VERY_STRONG": 0.2
        }
    
    # Sample from each quality category
    balanced = []
    for quality, proportion in quality_distribution.items():
        n_samples = int(target_size * proportion)
        available = by_quality[quality]
        
        if len(available) <= n_samples:
            balanced.extend(available)
        else:
            # Random sample without replacement
            indices = np.random.choice(
                len(available), size=n_samples, replace=False
            )
            balanced.extend([available[i] for i in indices])
    
    # Fill remaining slots with random samples if needed
    if len(balanced) < target_size:
        remaining = target_size - len(balanced)
        all_available = [p for p in pairs if p not in balanced]
        if all_available:
            indices = np.random.choice(
                len(all_available), 
                size=min(remaining, len(all_available)), 
                replace=False
            )
            balanced.extend([all_available[i] for i in indices])
    
    return balanced


__all__ = [
    "PreferencePair",
    "compute_consciousness_reward",
    "generate_preference_pairs",
    "rank_responses",
    "filter_pairs_by_quality",
    "balance_preference_dataset",
]


if __name__ == "__main__":
    # Self-test examples
    print("Preference Pair Generator - Standalone Tests")
    print("=" * 50)
    
    np.random.seed(42)
    
    # Test 1: Generate preference pairs
    print("\n1. Generate Preference Pairs:")
    responses = [
        "Response A (optimal)",
        "Response B (poor)",
        "Response C (good)",
        "Response D (mediocre)"
    ]
    metrics_list = [
        # Optimal
        {'lyapunov': 0.0, 'hurst': 0.5, 'spectral_entropy': 0.5, 'agency_score': 0.7},
        # Poor
        {'lyapunov': -2.0, 'hurst': 0.9, 'spectral_entropy': 0.1, 'agency_score': 0.2},
        # Good
        {'lyapunov': 0.1, 'hurst': 0.52, 'spectral_entropy': 0.6, 'agency_score': 0.75},
        # Mediocre
        {'lyapunov': -0.5, 'hurst': 0.6, 'spectral_entropy': 0.3, 'agency_score': 0.5}
    ]
    
    pairs = generate_preference_pairs(responses, metrics_list)
    print(f"   Generated {len(pairs)} pairs")
    for i, pair in enumerate(pairs[:3]):
        print(f"   Pair {i+1}: margin={pair.preference_margin:.3f}, quality={pair.quality_category}")
    assert len(pairs) > 0, "Should generate at least one pair"
    assert pairs[0].preference_margin >= pairs[-1].preference_margin, "Pairs should be sorted by margin"
    
    # Test 2: Rank responses
    print("\n2. Rank Responses:")
    ranked = rank_responses(responses, metrics_list)
    print("   Ranking (best to worst):")
    for i, (idx, reward, response, _) in enumerate(ranked):
        print(f"   {i+1}. {response}: {reward:.3f}")
    assert ranked[0][1] >= ranked[-1][1], "Should be sorted by reward"
    
    # Test 3: Filter by quality
    print("\n3. Filter by Quality:")
    strong_pairs = filter_pairs_by_quality(pairs, "STRONG")
    print(f"   Strong pairs: {len(strong_pairs)} out of {len(pairs)}")
    for pair in strong_pairs:
        print(f"   - Margin: {pair.preference_margin:.3f}, Quality: {pair.quality_category}")
    
    # Test 4: Minimum margin
    print("\n4. Minimum Margin Filtering:")
    pairs_min_margin = generate_preference_pairs(
        responses, metrics_list, min_margin=0.2
    )
    print(f"   Pairs with margin >= 0.2: {len(pairs_min_margin)} out of {len(pairs)}")
    for pair in pairs_min_margin:
        assert pair.preference_margin >= 0.2, "All pairs should meet margin threshold"
    
    # Test 5: Max pairs limit
    print("\n5. Max Pairs Limit:")
    pairs_limited = generate_preference_pairs(
        responses, metrics_list, max_pairs=3
    )
    print(f"   Limited to {len(pairs_limited)} pairs (max=3)")
    assert len(pairs_limited) <= 3, "Should respect max_pairs limit"
    
    # Test 6: Custom weights
    print("\n6. Custom Weights:")
    custom_weights = {
        'lyapunov': 0.1,
        'hurst': 0.1,
        'spectral_entropy': 0.5,  # Emphasize complexity
        'agency_score': 0.3
    }
    pairs_custom = generate_preference_pairs(
        responses, metrics_list, weights=custom_weights
    )
    print(f"   Generated {len(pairs_custom)} pairs with custom weights")
    print(f"   Top pair margin: {pairs_custom[0].preference_margin:.3f}")
    
    # Test 7: Balance dataset
    print("\n7. Balance Preference Dataset:")
    # Create more pairs for balancing test
    many_responses = [f"Response {i}" for i in range(10)]
    many_metrics = []
    for i in range(10):
        # Vary metrics to get different quality levels
        lyap = np.random.uniform(-1.0, 1.0)
        hurst = np.random.uniform(0.3, 0.7)
        entropy = np.random.uniform(0.0, 1.0)
        agency = np.random.uniform(0.0, 1.0)
        many_metrics.append({
            'lyapunov': lyap,
            'hurst': hurst,
            'spectral_entropy': entropy,
            'agency_score': agency
        })
    
    many_pairs = generate_preference_pairs(many_responses, many_metrics)
    balanced = balance_preference_dataset(many_pairs, target_size=10)
    print(f"   Balanced to {len(balanced)} pairs from {len(many_pairs)}")
    
    # Count by quality
    quality_counts = {q: 0 for q in ["WEAK", "MODERATE", "STRONG", "VERY_STRONG"]}
    for pair in balanced:
        quality_counts[pair.quality_category] += 1
    print(f"   Quality distribution: {quality_counts}")
    
    # Test 8: Edge cases
    print("\n8. Edge Cases:")
    
    # Empty list
    empty_pairs = generate_preference_pairs([], [])
    print(f"   Empty input: {len(empty_pairs)} pairs")
    assert len(empty_pairs) == 0, "Empty input should produce no pairs"
    
    # Single response
    single_pairs = generate_preference_pairs(["A"], [metrics_list[0]])
    print(f"   Single response: {len(single_pairs)} pairs")
    assert len(single_pairs) == 0, "Single response should produce no pairs"
    
    # Two responses
    two_pairs = generate_preference_pairs(
        responses[:2], metrics_list[:2]
    )
    print(f"   Two responses: {len(two_pairs)} pairs")
    assert len(two_pairs) == 1, "Two responses should produce one pair"
    
    # Test 9: Preference properties
    print("\n9. Preference Properties:")
    test_pair = pairs[0]
    print(f"   Clear preference: {test_pair.is_clear_preference}")
    print(f"   Strong preference: {test_pair.is_strong_preference}")
    print(f"   Quality category: {test_pair.quality_category}")
    
    print("\n✓ All tests completed successfully!")
