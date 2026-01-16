"""
Quick Reference Guide - New Modular Components
===============================================

This guide provides quick code snippets for using the new analyzers and benchmarks.

TABLE OF CONTENTS
-----------------
1. Basic Setup
2. Test Suites
3. Single Prompt Analysis
4. Batch Analysis
5. Profiling
6. Comparing Profiles
7. Advanced Usage

"""

# ============================================================================
# 1. BASIC SETUP
# ============================================================================

from consciousness_circuit.analyzers import ConsciousnessTrajectoryAnalyzer
from consciousness_circuit.benchmarks import (
    get_test_suite,
    get_full_benchmark,
    ModelProfiler,
    ProfileResult,
)
from transformers import AutoModel, AutoTokenizer

# Load model
model = AutoModel.from_pretrained("gpt2")
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Create analyzer
analyzer = ConsciousnessTrajectoryAnalyzer()
analyzer.bind_model(model, tokenizer)


# ============================================================================
# 2. TEST SUITES
# ============================================================================

# Get a specific category
philosophical = get_test_suite('philosophical')  # 15 prompts
factual = get_test_suite('factual')             # 15 prompts
reasoning = get_test_suite('reasoning')         # 15 prompts
creative = get_test_suite('creative')           # 15 prompts

# Get all categories
all_tests = get_full_benchmark()
# Returns: {'philosophical': [...], 'factual': [...], ...}

# Get category metadata
from consciousness_circuit.benchmarks import get_category_info
info = get_category_info()
print(info['philosophical']['description'])
# Output: "Self-reflection, consciousness, and ethical reasoning"


# ============================================================================
# 3. SINGLE PROMPT ANALYSIS
# ============================================================================

# Analyze a single prompt
result = analyzer.deep_analyze(
    "What does it mean to be conscious?",
    include_per_token=True  # Include per-token trajectory
)

# Access results
print(f"Consciousness: {result.consciousness_score:.3f}")
print(f"Trajectory class: {result.trajectory_class}")
print(f"Lyapunov: {result.lyapunov:.3f}")
print(f"Hurst: {result.hurst:.3f}")
print(f"Agency: {result.agency_score:.3f}")
print(f"Converging: {result.is_converging}")

# Get interpretation
print(result.interpretation())
# Output: Human-readable analysis with emojis

# Export to dict
data = result.to_dict()


# ============================================================================
# 4. BATCH ANALYSIS
# ============================================================================

# Analyze multiple prompts
prompts = [
    "What is consciousness?",
    "How do you think?",
    "Explain your reasoning process.",
]

results = analyzer.analyze_batch(
    prompts,
    include_per_token=False  # Skip per-token for speed
)

# Process results
for prompt, result in zip(prompts, results):
    if result is not None:  # Check for errors
        print(f"{prompt[:30]:30s} → {result.consciousness_score:.3f}")


# ============================================================================
# 5. PROFILING
# ============================================================================

# Create profiler
profiler = ModelProfiler(analyzer)

# Profile on a test suite
philosophical = get_test_suite('philosophical')
profile = profiler.profile(
    philosophical,
    name='gpt2-philosophical',
    store_results=False,  # Don't keep full results (saves memory)
    metadata={
        'model': 'gpt2',
        'category': 'philosophical',
        'date': '2024-01-16',
    }
)

# View summary
print(profile.summary())
# Output: Formatted summary with all metrics

# Access specific metrics
print(f"Average consciousness: {profile.avg_consciousness:.3f}")
print(f"Average Lyapunov: {profile.avg_lyapunov:.3f}")
print(f"Convergence rate: {profile.convergence_rate:.1%}")
print(f"Trajectory classes: {profile.trajectory_classes}")

# Export profile
profile_data = profile.to_dict()


# ============================================================================
# 6. COMPARING PROFILES
# ============================================================================

# Create two profiles
factual = get_test_suite('factual')
profile1 = profiler.profile(philosophical, name='gpt2-philosophical')
profile2 = profiler.profile(factual, name='gpt2-factual')

# Compare directly
comparison = profile1.compare(profile2)
print(f"Consciousness diff: {comparison['consciousness_diff']:+.3f}")
print(f"Lyapunov diff: {comparison['lyapunov_diff']:+.3f}")
print(f"Hurst diff: {comparison['hurst_diff']:+.3f}")
print(f"Agency diff: {comparison['agency_diff']:+.3f}")

# Or use profiler
comparison = profiler.compare(profile1, profile2)

# Compare all stored profiles
all_comparisons = profiler.compare_all()
for key, comp in all_comparisons.items():
    print(f"\n{key}:")
    print(f"  Δ Consciousness: {comp['consciousness_diff']:+.3f}")
    print(f"  Δ Chaos: {comp['lyapunov_diff']:+.3f}")


# ============================================================================
# 7. ADVANCED USAGE
# ============================================================================

# Profile across all categories
results = {}
for category in ['philosophical', 'factual', 'reasoning', 'creative']:
    prompts = get_test_suite(category)
    profile = profiler.profile(
        prompts,
        name=f"gpt2-{category}",
        metadata={'category': category}
    )
    results[category] = profile
    
    print(f"\n{category.upper()}")
    print(f"  Consciousness: {profile.avg_consciousness:.3f}")
    print(f"  Lyapunov: {profile.avg_lyapunov:.3f}")
    print(f"  Agency: {profile.avg_agency:.3f}")

# Find which category has highest consciousness
best_category = max(results.items(), key=lambda x: x[1].avg_consciousness)
print(f"\nHighest consciousness: {best_category[0]}")

# Compare prompt categories
comparison = analyzer.compare_prompts(philosophical[:5])
print(comparison)

# Retrieve stored profile
stored = profiler.get_profile('gpt2-philosophical')
print(stored.summary())

# List all profiles
profile_names = profiler.list_profiles()
print(f"Stored profiles: {profile_names}")


# ============================================================================
# EXAMPLE: Complete Benchmark Workflow
# ============================================================================

def benchmark_model(model, tokenizer, model_name="model"):
    """Complete benchmarking workflow."""
    
    # Setup
    analyzer = ConsciousnessTrajectoryAnalyzer()
    analyzer.bind_model(model, tokenizer)
    profiler = ModelProfiler(analyzer)
    
    # Profile all categories
    results = {}
    for category in ['philosophical', 'factual', 'reasoning', 'creative']:
        prompts = get_test_suite(category)
        profile = profiler.profile(
            prompts,
            name=f"{model_name}-{category}",
            metadata={'model': model_name, 'category': category}
        )
        results[category] = profile
        
        print(f"\n{'='*60}")
        print(f"{model_name.upper()} - {category.upper()}")
        print('='*60)
        print(profile.summary())
    
    # Compare all profiles
    print(f"\n{'='*60}")
    print("CROSS-CATEGORY COMPARISONS")
    print('='*60)
    
    comparisons = profiler.compare_all()
    for key, comp in comparisons.items():
        print(f"\n{key}:")
        print(f"  Consciousness: {comp['consciousness_diff']:+.3f}")
        print(f"  Lyapunov: {comp['lyapunov_diff']:+.3f}")
        print(f"  Hurst: {comp['hurst_diff']:+.3f}")
        print(f"  Agency: {comp['agency_diff']:+.3f}")
    
    return results, profiler

# Run benchmark
# results, profiler = benchmark_model(model, tokenizer, "gpt2")


# ============================================================================
# TIPS & TRICKS
# ============================================================================

"""
Performance Tips:
- Set include_per_token=False for faster batch analysis
- Set store_results=False in profiler to save memory
- Use analyze_batch() instead of loop for better performance

Error Handling:
- analyze_batch() returns None for failed analyses
- Always check if result is not None before using
- Profiler continues on errors and reports count

Memory Management:
- Per-token trajectories can be large, disable if not needed
- ProfileResult without results is lightweight (~1KB)
- Full results can be 100KB+ per prompt

Best Practices:
- Use descriptive profile names (e.g., "gpt2-philosophical")
- Add metadata for later reference
- Store profiles for reproducibility
- Export to dict for saving/loading
"""
