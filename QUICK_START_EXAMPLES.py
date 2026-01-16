#!/usr/bin/env python3
"""
Quick Reference Examples - Modular Architecture
================================================

Examples of using the new modular architecture in different ways.

Note: These examples show direct imports from metric modules.
Package-level imports work when torch is installed.
"""

import sys
import os
import numpy as np

# Add metrics to path for direct import
sys.path.insert(0, 'consciousness_circuit/metrics')
sys.path.insert(0, 'consciousness_circuit/classifiers')
sys.path.insert(0, 'consciousness_circuit/plugins')
sys.path.insert(0, 'consciousness_circuit/training')
sys.path.insert(0, 'consciousness_circuit/benchmarks')

# ============================================================================
# Example 1: Copy Single File (Zero Setup)
# ============================================================================
print("Example 1: Standalone File Usage")
print("=" * 50)
print("""
# Literally copy one file to another project:
$ cp consciousness_circuit/metrics/lyapunov.py /path/to/other/project/
$ cd /path/to/other/project

# Use it immediately (only numpy required):
from lyapunov import compute_lyapunov

trajectory = np.random.randn(100, 64)
result = compute_lyapunov(trajectory)
print(f"Lyapunov: {result.exponent:.3f}")
print(f"Is chaotic: {result.is_chaotic}")
""")


# ============================================================================
# Example 2: Import Individual Metrics (Direct Import)
# ============================================================================
print("\nExample 2: Import Individual Metrics")
print("=" * 50)

# Import directly from modules (works without torch)
from lyapunov import compute_lyapunov
from hurst import compute_hurst
from msd import compute_msd
from agency import compute_agency_score

# Generate test trajectory
np.random.seed(42)
trajectory = np.cumsum(np.random.randn(100, 64), axis=0)

# Compute metrics
lyap = compute_lyapunov(trajectory)
print(f"Lyapunov: {lyap.exponent:.3f} - {lyap.interpretation}")

hurst = compute_hurst(trajectory)
print(f"Hurst: {hurst.exponent:.3f} - {hurst.interpretation}")

msd = compute_msd(trajectory)
print(f"Motion: {msd.motion_type} (α={msd.diffusion_exponent:.2f})")

agency = compute_agency_score(trajectory)
print(f"Agency: {agency.agency:.3f} - {agency.interpretation}")


# ============================================================================
# Example 3: Classify Signal
# ============================================================================
print("\nExample 3: Classify Signal")
print("=" * 50)

from signal_class import classify_signal

# Combine metrics into classification
metrics_dict = {
    'lyapunov': lyap.exponent,
    'hurst': hurst.exponent,
    'msd_exponent': msd.diffusion_exponent,
    'agency': agency.agency,
}

classification = classify_signal(metrics_dict)
print(f"Signal class: {classification.signal_class.value}")
print(f"Confidence: {classification.confidence:.2%}")
print(f"Interpretation: {classification.interpretation}")


# ============================================================================
# Example 4: Custom Analysis Pipeline
# ============================================================================
print("\nExample 4: Custom Analysis Pipeline")
print("=" * 50)

def my_custom_analyzer(trajectory):
    """Build your own analysis pipeline."""
    # Direct imports (works without torch)
    from lyapunov import compute_lyapunov
    from hurst import compute_hurst
    from msd import compute_msd
    from agency import compute_agency_score
    from signal_class import classify_signal
    
    # Compute all metrics
    lyap = compute_lyapunov(trajectory)
    hurst = compute_hurst(trajectory)
    msd = compute_msd(trajectory)
    agency = compute_agency_score(trajectory)
    
    # Classify
    metrics = {
        'lyapunov': lyap.exponent,
        'hurst': hurst.exponent,
        'msd_exponent': msd.diffusion_exponent,
        'agency': agency.agency,
    }
    classification = classify_signal(metrics)
    
    # Return custom summary
    return {
        'is_chaotic': lyap.is_chaotic,
        'has_memory': hurst.has_memory,
        'motion_type': msd.motion_type,
        'is_agentic': agency.is_agentic,
        'signal_class': classification.signal_class.value,
        'confidence': classification.confidence,
    }

result = my_custom_analyzer(trajectory)
print("Custom Analysis Result:")
for key, value in result.items():
    print(f"  {key}: {value}")


# ============================================================================
# Example 5: Intervention Plugins
# ============================================================================
print("\nExample 5: Intervention Plugins")
print("=" * 50)

from attractor_lock import AttractorLockPlugin
from coherence_boost import CoherenceBoostPlugin

# Create plugins (they don't inherit from base classes in standalone mode)
attractor_plugin = AttractorLockPlugin(lyapunov_threshold=0.3)
coherence_plugin = CoherenceBoostPlugin(hurst_threshold=0.4)

print("Created plugins:")
print(f"  - {attractor_plugin}")
print(f"  - {coherence_plugin}")

# Test attractor plugin
chaotic_trajectory = np.random.randn(100, 64) * 5
metrics_chaotic = {'lyapunov': 0.5, 'hurst': 0.3}

if attractor_plugin.should_intervene(metrics_chaotic):
    print("\n✓ Attractor plugin would intervene (high chaos detected)")
else:
    print("\n✗ Attractor plugin would not intervene")

if coherence_plugin.should_intervene(metrics_chaotic):
    print("✓ Coherence plugin would intervene (low memory detected)")
else:
    print("✗ Coherence plugin would not intervene")


# ============================================================================
# Example 6: Training Utilities (Consciousness Rewards)
# ============================================================================
print("\nExample 6: Training with Consciousness Rewards")
print("=" * 50)

from reward_model import ConsciousnessRewardModel, RewardConfig
from preference_generator import generate_preference_pairs

# Configure reward weights
config = RewardConfig(
    consciousness_weight=0.4,
    stability_weight=0.2,
    memory_weight=0.2,
    agency_weight=0.2,
)

# Compute reward from metrics
reward_result = ConsciousnessRewardModel.compute_from_metrics(
    metrics={
        'lyapunov': 0.2,  # Stable
        'hurst': 0.6,     # Good memory
        'agency': 0.7,    # Agentic
    },
    config=config
)

print(f"Total reward: {reward_result.reward:.3f}")
print(f"Explanation: {reward_result.explanation}")

# Generate preference pairs for DPO
responses = ["Response A", "Response B", "Response C"]
metrics_list = [
    {'lyapunov': 0.2, 'hurst': 0.6, 'agency': 0.7},  # Good
    {'lyapunov': 0.8, 'hurst': 0.3, 'agency': 0.2},  # Poor
    {'lyapunov': 0.4, 'hurst': 0.5, 'agency': 0.5},  # Medium
]

pairs = generate_preference_pairs(responses, metrics_list)
print(f"\nGenerated {len(pairs)} preference pairs for DPO training")
for i, pair in enumerate(pairs[:2]):  # Show first 2
    print(f"  Pair {i+1}: '{pair.chosen_response}' > '{pair.rejected_response}' (margin={pair.preference_margin:.2f})")


# ============================================================================
# Example 7: Benchmarking and Profiling
# ============================================================================
print("\nExample 7: Benchmarking and Profiling")
print("=" * 50)

from test_suites import get_test_suite, get_full_benchmark

# Get categorized prompts
philosophical = get_test_suite('philosophical')
reasoning = get_test_suite('reasoning')

print(f"Philosophical prompts: {len(philosophical)}")
print(f"  Example: {philosophical[0][:50]}...")

print(f"\nReasoning prompts: {len(reasoning)}")
print(f"  Example: {reasoning[0][:50]}...")

# Get all benchmarks
full_benchmark = get_full_benchmark()
print(f"\nTotal benchmark prompts: {sum(len(v) for v in full_benchmark.values())}")
print(f"Categories: {list(full_benchmark.keys())}")


# ============================================================================
# Example 8: Full Pipeline (Backward Compatible)
# ============================================================================
print("\nExample 8: Full Integrated Pipeline")
print("=" * 50)
print("""
# The complete analyzer still works exactly as before:

from consciousness_circuit import ConsciousnessTrajectoryAnalyzer

analyzer = ConsciousnessTrajectoryAnalyzer()
analyzer.bind_model(model, tokenizer)

result = analyzer.deep_analyze("Let me think about this...")

print(f"Consciousness: {result.consciousness_score:.3f}")
print(f"Trajectory class: {result.trajectory_class}")
print(f"Lyapunov: {result.lyapunov:.3f}")
print(f"Agency: {result.agency_score:.3f}")
print(result.interpretation())

# Full backward compatibility maintained!
""")


# ============================================================================
# Summary
# ============================================================================
print("\n" + "=" * 50)
print("✅ All Examples Complete!")
print("=" * 50)
print("""
The modular architecture provides:

✅ Standalone files (copy anywhere)
✅ Package imports (use what you need)
✅ Custom pipelines (compose your own)
✅ Intervention plugins (steer behavior)
✅ Training utilities (RLHF/DPO ready)
✅ Benchmarking tools (evaluate models)
✅ Full pipeline (backward compatible)

See MODULAR_ARCHITECTURE.md for complete documentation.
""")
