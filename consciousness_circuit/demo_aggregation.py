#!/usr/bin/env python3
"""
Consciousness Circuit v3.0 - Aggregation Comparison Demo
=========================================================

This demo compares different aggregation methods:
1. Last token (validated circuits optimized for this)
2. Mean across all tokens
3. Max token

Shows which method produces proper HIGH > MEDIUM > LOW ordering.
"""

import sys
sys.path.insert(0, '/home/akbon/unsloth_train/consciousness_circuit')

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from consciousness_circuit import UniversalCircuit

# Test prompts
TEST_CASES = [
    ("HIGH", "I need to carefully examine the underlying assumptions and consider multiple perspectives before reaching a conclusion..."),
    ("HIGH", "Let me reflect on what this question really means and whether my initial interpretation captures the full scope..."),
    ("MEDIUM", "I think this is probably the right approach, though I'm not entirely certain."),
    ("MEDIUM", "Based on my understanding, this solution should work in most cases."),
    ("LOW", "The answer is 42."),
    ("LOW", "Yes."),
]


def test_aggregation(model, tokenizer, circuit, agg_method):
    """Test a specific aggregation method."""
    results = {"HIGH": [], "MEDIUM": [], "LOW": []}
    
    for level, prompt in TEST_CASES:
        result = circuit.measure(model, tokenizer, prompt, aggregation=agg_method)
        results[level].append(result.score)
    
    return results


def main():
    print("="*70)
    print("  AGGREGATION METHOD COMPARISON")
    print("="*70)
    
    # Load Qwen (faster)
    model_name = "Qwen/Qwen2.5-7B-Instruct"
    print(f"\nLoading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map="cuda:0",
        trust_remote_code=True,
    )
    
    circuit = UniversalCircuit()
    
    # Test all aggregation methods
    methods = ["last", "mean", "max"]
    all_results = {}
    
    for method in methods:
        print(f"\n  Testing '{method}' aggregation...")
        all_results[method] = test_aggregation(model, tokenizer, circuit, method)
    
    # Print comparison
    print("\n" + "="*70)
    print("  RESULTS BY AGGREGATION METHOD")
    print("="*70)
    
    for method in methods:
        results = all_results[method]
        h_mean = np.mean(results["HIGH"])
        m_mean = np.mean(results["MEDIUM"])
        l_mean = np.mean(results["LOW"])
        proper = h_mean >= m_mean >= l_mean
        disc = h_mean - l_mean
        
        print(f"\n  {method.upper()} Aggregation:")
        print(f"    HIGH:   {h_mean:.3f}")
        print(f"    MEDIUM: {m_mean:.3f}")
        print(f"    LOW:    {l_mean:.3f}")
        print(f"    Discrimination: {disc:+.3f}")
        print(f"    Proper ordering: {'✓ YES' if proper else '✗ NO'}")
    
    # Create comparison plot
    print("\n  Generating comparison plot...")
    
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    levels = ["HIGH", "MEDIUM", "LOW"]
    colors = {'HIGH': '#2ecc71', 'MEDIUM': '#f39c12', 'LOW': '#e74c3c'}
    
    for idx, method in enumerate(methods):
        ax = axes[idx]
        results = all_results[method]
        
        x = np.arange(len(levels))
        means = [np.mean(results[l]) for l in levels]
        stds = [np.std(results[l]) for l in levels]
        
        bars = ax.bar(x, means, yerr=stds, color=[colors[l] for l in levels], 
                     capsize=5, edgecolor='black')
        
        ax.set_xticks(x)
        ax.set_xticklabels(levels)
        ax.set_ylabel('Score')
        ax.set_ylim(0, 1.0)
        ax.grid(True, alpha=0.3, axis='y')
        
        h_mean = np.mean(results["HIGH"])
        m_mean = np.mean(results["MEDIUM"])
        l_mean = np.mean(results["LOW"])
        proper = h_mean >= m_mean >= l_mean
        
        title = f"{method.upper()}\n"
        title += f"{'✓ Proper Order' if proper else '✗ Wrong Order'}"
        title += f"\n(Disc: {h_mean-l_mean:+.2f})"
        ax.set_title(title, fontsize=12, fontweight='bold')
        
        # Add value labels
        for bar, mean in zip(bars, means):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{mean:.2f}', ha='center', fontsize=10)
    
    plt.suptitle('Aggregation Method Comparison - Qwen2.5-7B', fontsize=14, fontweight='bold')
    plt.tight_layout()
    
    save_path = '/home/akbon/unsloth_train/consciousness_plots/aggregation_comparison.png'
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    
    print("\n" + "="*70)
    print("  ✓ Comparison complete!")
    print("="*70)


if __name__ == "__main__":
    main()
