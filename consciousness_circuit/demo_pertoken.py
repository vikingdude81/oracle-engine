#!/usr/bin/env python3
"""
Consciousness Circuit Demo - Per-Token Analysis & Visualization
================================================================

This demo shows:
1. Per-token consciousness trajectory analysis
2. Comparison across multiple prompts  
3. Dimension contribution visualization
"""

import sys
sys.path.insert(0, '/home/akbon/unsloth_train/consciousness_circuit')

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from consciousness_circuit import (
    UniversalCircuit,
    ConsciousnessVisualizer,
    measure_consciousness,
)

def main():
    print("="*70)
    print("  CONSCIOUSNESS CIRCUIT v3.0 - Per-Token Analysis Demo")
    print("="*70)
    
    # Load model
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"\nLoading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    
    # Initialize tools
    circuit = UniversalCircuit()
    viz = ConsciousnessVisualizer(circuit)
    
    # Test prompts with expected consciousness levels
    test_cases = [
        ("HIGH", "I need to carefully examine the underlying assumptions and consider multiple perspectives before reaching a conclusion about this complex matter..."),
        ("HIGH", "Let me reflect on what this question really means and whether my initial interpretation captures the full scope of the inquiry..."),
        ("MEDIUM", "I think this is probably the right approach, though I'm not entirely certain about all the details."),
        ("MEDIUM", "Based on my understanding, this solution should work in most cases."),
        ("LOW", "The answer is 42."),
        ("LOW", "Yes."),
    ]
    
    print("\n" + "="*70)
    print("  ANALYSIS RESULTS")
    print("="*70)
    
    results_by_level = {"HIGH": [], "MEDIUM": [], "LOW": []}
    
    for expected_level, prompt in test_cases:
        # Get overall score
        result = circuit.measure(model, tokenizer, prompt)
        
        # Get per-token trajectory
        trajectory = viz.measure_per_token(model, tokenizer, prompt)
        
        results_by_level[expected_level].append({
            "prompt": prompt,
            "score": result.score,
            "trajectory": trajectory,
        })
        
        print(f"\n[{expected_level}] Score: {result.score:.3f}")
        print(f"  Prompt: {prompt[:65]}...")
        print(f"  Peak: {trajectory.peak_score:.3f} at '{trajectory.peak_token[1]}'")
        print(f"  Mean: {trajectory.mean_score:.3f}, Trend: {trajectory.trajectory_slope:+.4f}/token")
        
        # Show top 5 dimension contributions
        if result.dimension_contributions:
            sorted_dims = sorted(
                result.dimension_contributions.items(),
                key=lambda x: abs(x[1]),
                reverse=True
            )[:3]
            contribs = ", ".join([f"{k}:{v:+.2f}" for k, v in sorted_dims])
            print(f"  Top dims: {contribs}")
    
    # Summary
    print("\n" + "="*70)
    print("  SUMMARY BY LEVEL")
    print("="*70)
    
    import numpy as np
    
    for level in ["HIGH", "MEDIUM", "LOW"]:
        scores = [r["score"] for r in results_by_level[level]]
        mean = np.mean(scores)
        std = np.std(scores)
        print(f"  {level:6s}: {mean:.3f} ± {std:.3f}  (n={len(scores)})")
    
    # Validation
    h_mean = np.mean([r["score"] for r in results_by_level["HIGH"]])
    m_mean = np.mean([r["score"] for r in results_by_level["MEDIUM"]])
    l_mean = np.mean([r["score"] for r in results_by_level["LOW"]])
    
    proper_order = h_mean >= m_mean >= l_mean
    discrimination = h_mean - l_mean
    
    print(f"\n  Discrimination (H-L): {discrimination:.3f}")
    print(f"  Proper ordering (H≥M≥L): {'✓ YES' if proper_order else '✗ NO'}")
    
    # Token-level insight
    print("\n" + "="*70)
    print("  TOKEN-LEVEL INSIGHTS")
    print("="*70)
    
    # Compare HIGH vs LOW prompts token by token
    high_traj = results_by_level["HIGH"][0]["trajectory"]
    low_traj = results_by_level["LOW"][0]["trajectory"]
    
    print("\n  HIGH consciousness prompt - token trajectory:")
    for i, (tok, score) in enumerate(zip(high_traj.tokens[:12], high_traj.scores[:12])):
        bar = "█" * int(score * 30)
        print(f"    {i:2d}. {tok:15s} {score:.3f} {bar}")
    
    print("\n  LOW consciousness prompt - token trajectory:")
    for i, (tok, score) in enumerate(zip(low_traj.tokens[:7], low_traj.scores[:7])):
        bar = "█" * int(score * 30)
        print(f"    {i:2d}. {tok:15s} {score:.3f} {bar}")
    
    print("\n" + "="*70)
    print("  ✓ Demo complete! Package working correctly.")
    print("="*70)

if __name__ == "__main__":
    main()
