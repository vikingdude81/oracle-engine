"""
Benchmarks Package
==================

Standardized test suites and profiling tools for consciousness analysis.

Available Components:
- Test Suites: Categorized prompt collections
- ModelProfiler: Profile and compare model behavior

Usage:
    from consciousness_circuit.benchmarks import (
        get_test_suite,
        get_full_benchmark,
        ModelProfiler,
        ProfileResult,
    )
    
    # Get test prompts
    philosophical = get_test_suite('philosophical')
    factual = get_test_suite('factual')
    
    # Profile a model
    from consciousness_circuit.analyzers import ConsciousnessTrajectoryAnalyzer
    analyzer = ConsciousnessTrajectoryAnalyzer()
    analyzer.bind_model(model, tokenizer)
    
    profiler = ModelProfiler(analyzer)
    profile = profiler.profile(philosophical, name='model-philosophical')
    print(profile.summary())
    
    # Compare profiles
    profile2 = profiler.profile(factual, name='model-factual')
    comparison = profile.compare(profile2)
"""

from .test_suites import (
    PHILOSOPHICAL_PROMPTS,
    FACTUAL_PROMPTS,
    REASONING_PROMPTS,
    CREATIVE_PROMPTS,
    get_test_suite,
    get_full_benchmark,
    get_category_info,
)

from .profiler import (
    ProfileResult,
    ModelProfiler,
)


__all__ = [
    # Test suites
    'PHILOSOPHICAL_PROMPTS',
    'FACTUAL_PROMPTS',
    'REASONING_PROMPTS',
    'CREATIVE_PROMPTS',
    'get_test_suite',
    'get_full_benchmark',
    'get_category_info',
    # Profiler
    'ProfileResult',
    'ModelProfiler',
]
