#!/usr/bin/env python3
"""
Test script demonstrating standalone usage of consciousness circuit components.

This script shows how individual modules can be used independently
without the full consciousness circuit framework.
"""

import numpy as np
import sys
import importlib.util


def load_module_directly(filepath, module_name):
    """Load a module directly without triggering package imports."""
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = module
    spec.loader.exec_module(module)
    return module


def test_standalone_metrics():
    """Test standalone metric modules."""
    print("=" * 70)
    print("Testing Standalone Metrics")
    print("=" * 70)
    
    # Load modules
    base = 'consciousness_circuit/metrics'
    lyapunov = load_module_directly(f'{base}/lyapunov.py', 'lyapunov_standalone')
    hurst = load_module_directly(f'{base}/hurst.py', 'hurst_standalone')
    msd = load_module_directly(f'{base}/msd.py', 'msd_standalone')
    entropy = load_module_directly(f'{base}/entropy.py', 'entropy_standalone')
    agency = load_module_directly(f'{base}/agency.py', 'agency_standalone')
    
    # Create test data
    np.random.seed(42)
    x = np.cumsum(np.random.randn(200)) * 0.1  # Random walk
    y = np.cumsum(np.random.randn(200)) * 0.1
    sequence = np.random.randn(300)
    
    print("\n1. Lyapunov Exponent (chaos detection)")
    lyap = lyapunov.compute_lyapunov(x, y)
    print(f"   λ = {lyap:.4f} ({lyapunov.LyapunovResult(lyap, 0.8, 'rosenstein', {}).interpretation})")
    
    print("\n2. Hurst Exponent (memory/persistence)")
    h = hurst.compute_hurst(sequence)
    print(f"   H = {h:.4f} ", end="")
    if h > 0.55:
        print("(trending/persistent)")
    elif h < 0.45:
        print("(mean-reverting)")
    else:
        print("(random walk)")
    
    print("\n3. Mean Squared Displacement (diffusion analysis)")
    lags, msd_vals = msd.compute_msd(x, y, max_lag=50)
    alpha = msd.compute_diffusion_exponent(x, y)
    print(f"   α = {alpha:.4f} ", end="")
    if alpha > 1.2:
        print("(superdiffusive)")
    elif alpha < 0.8:
        print("(subdiffusive)")
    else:
        print("(normal diffusion)")
    
    print("\n4. Spectral Entropy (randomness)")
    ent = entropy.compute_spectral_entropy(sequence)
    print(f"   S = {ent:.4f} ", end="")
    if ent > 0.8:
        print("(random)")
    elif ent < 0.3:
        print("(structured)")
    else:
        print("(mixed)")
    
    print("\n5. Agency Score (goal-directedness)")
    trajectory = np.column_stack([x[:100], y[:100]])
    agency_score = agency.compute_agency_score(trajectory)
    print(f"   Score = {agency_score:.4f} ", end="")
    if agency_score > 0.6:
        print("(agentic)")
    else:
        print("(random/non-agentic)")
    
    print("\n✅ All metrics computed successfully!\n")
    
    return {
        'lyapunov': lyap,
        'hurst': h,
        'diffusion_exponent': alpha,
        'spectral_entropy': ent,
        'agency_score': agency_score
    }


def test_standalone_classifier(metrics):
    """Test standalone classifier."""
    print("=" * 70)
    print("Testing Standalone Classifier")
    print("=" * 70)
    
    signal_class = load_module_directly(
        'consciousness_circuit/classifiers/signal_class.py',
        'signal_class_standalone'
    )
    
    print("\nClassifying signal based on metrics...")
    result = signal_class.classify_signal(metrics)
    
    print(f"\n   Class: {result.signal_class.name}")
    print(f"   Confidence: {result.confidence:.2f}")
    print(f"   Structured: {result.is_structured}")
    print(f"   Evidence: {list(result.evidence.keys())}")
    
    print("\n✅ Classification completed!\n")
    
    return result


def test_standalone_plugin():
    """Test standalone plugin."""
    print("=" * 70)
    print("Testing Standalone Plugin")
    print("=" * 70)
    
    attractor_lock = load_module_directly(
        'consciousness_circuit/plugins/attractor_lock.py',
        'attractor_lock_standalone'
    )
    
    print("\nInitializing AttractorLockPlugin...")
    plugin = attractor_lock.AttractorLockPlugin(
        lyapunov_threshold=0.3,
        nudge_strength=0.1
    )
    
    # Test intervention decision
    metrics_stable = {'lyapunov': 0.1}
    metrics_chaotic = {'lyapunov': 0.5}
    
    print(f"   Should intervene (λ=0.1): {plugin.should_intervene(metrics_stable)}")
    print(f"   Should intervene (λ=0.5): {plugin.should_intervene(metrics_chaotic)}")
    
    # Learn an attractor
    print("\n   Learning high-quality attractor...")
    hidden_states = np.random.randn(256)
    plugin.learn_attractor(hidden_states, quality=0.85)
    
    # Test intervention
    print("   Applying intervention to chaotic state...")
    chaotic_states = np.random.randn(256)
    modified = plugin.intervene(chaotic_states, metrics_chaotic)
    
    change = np.linalg.norm(modified - chaotic_states)
    print(f"   Change magnitude: {change:.4f}")
    
    stats = plugin.get_statistics()
    print(f"   Interventions: {stats['intervention_count']}")
    print(f"   Attractors stored: {stats['attractors_stored']}")
    
    print("\n✅ Plugin tested successfully!\n")


def test_standalone_training():
    """Test standalone training module."""
    print("=" * 70)
    print("Testing Standalone Training Module")
    print("=" * 70)
    
    reward_model = load_module_directly(
        'consciousness_circuit/training/reward_model.py',
        'reward_model_standalone'
    )
    
    print("\nComputing rewards from metrics...")
    
    # Test case 1: High consciousness, stable
    metrics1 = {
        'consciousness_score': 0.8,
        'lyapunov': -0.2,
        'hurst': 0.65,
        'agency_score': 0.7,
        'trajectory_class': 'ATTRACTOR'
    }
    reward1 = reward_model.ConsciousnessRewardModel.compute_from_metrics(metrics1)
    print(f"\n   Case 1 (high quality): {reward1:.4f}")
    
    # Test case 2: Low consciousness, chaotic
    metrics2 = {
        'consciousness_score': 0.3,
        'lyapunov': 0.6,
        'hurst': 0.45,
        'agency_score': 0.3,
        'trajectory_class': 'CHAOTIC'
    }
    reward2 = reward_model.ConsciousnessRewardModel.compute_from_metrics(metrics2)
    print(f"   Case 2 (low quality): {reward2:.4f}")
    
    # Test case 3: Medium quality
    metrics3 = {
        'consciousness_score': 0.6,
        'lyapunov': 0.1,
        'hurst': 0.55,
        'agency_score': 0.5,
    }
    reward3 = reward_model.ConsciousnessRewardModel.compute_from_metrics(metrics3)
    print(f"   Case 3 (medium quality): {reward3:.4f}")
    
    print("\n✅ Training module tested successfully!\n")


def main():
    """Run all standalone tests."""
    print("\n" + "=" * 70)
    print("Consciousness Circuit - Standalone Components Test")
    print("=" * 70)
    print("\nThis demonstrates that all modules work independently")
    print("without requiring the full consciousness circuit framework.\n")
    
    # Test metrics
    metrics = test_standalone_metrics()
    
    # Test classifier
    classification = test_standalone_classifier(metrics)
    
    # Test plugin
    test_standalone_plugin()
    
    # Test training
    test_standalone_training()
    
    print("=" * 70)
    print("✅ All Standalone Tests Passed!")
    print("=" * 70)
    print("\nKey Features Demonstrated:")
    print("  • Zero coupling to torch/transformers")
    print("  • Each module works independently")
    print("  • Easy to copy single files to other projects")
    print("  • Consistent API across all modules")
    print("  • Can be composed into pipelines OR used individually")
    print()


if __name__ == '__main__':
    main()
