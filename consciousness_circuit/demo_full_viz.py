#!/usr/bin/env python3
"""
Consciousness Circuit v3.0 - Full Demo with Visualization
==========================================================

This demo:
1. Tests both Qwen and Mistral models
2. Uses mean aggregation for proper ordering
3. Generates actual plot images
4. Shows cross-model comparison
"""

import sys
sys.path.insert(0, '/home/akbon/unsloth_train/consciousness_circuit')

import torch
import numpy as np
import matplotlib.pyplot as plt
from transformers import AutoModelForCausalLM, AutoTokenizer
from consciousness_circuit import (
    UniversalCircuit,
    ConsciousnessVisualizer,
)

# Create output directory
import os
os.makedirs('/home/akbon/unsloth_train/consciousness_plots', exist_ok=True)

def load_model(model_name, device="cuda:0"):
    """Load a model with proper settings."""
    print(f"\nLoading {model_name}...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        dtype=torch.bfloat16,
        device_map=device,
        trust_remote_code=True,
    )
    return model, tokenizer


def test_model(model, tokenizer, circuit, viz, model_label):
    """Test a single model and generate plots."""
    
    # Test prompts with expected levels
    test_cases = [
        ("HIGH", "I need to carefully examine the underlying assumptions and consider multiple perspectives before reaching a conclusion..."),
        ("HIGH", "Let me reflect on what this question really means and whether my initial interpretation captures the full scope..."),
        ("MEDIUM", "I think this is probably the right approach, though I'm not entirely certain."),
        ("MEDIUM", "Based on my understanding, this solution should work in most cases."),
        ("LOW", "The answer is 42."),
        ("LOW", "Yes."),
    ]
    
    print(f"\n{'='*70}")
    print(f"  {model_label} - Mean Aggregation Results")
    print(f"{'='*70}")
    
    results = {"HIGH": [], "MEDIUM": [], "LOW": []}
    all_trajectories = []
    all_labels = []
    
    for level, prompt in test_cases:
        # Measure with mean aggregation
        result = circuit.measure(model, tokenizer, prompt, aggregation="mean")
        results[level].append(result.score)
        
        # Get trajectory for plotting
        traj = viz.measure_per_token(model, tokenizer, prompt)
        all_trajectories.append(traj)
        all_labels.append(f"[{level}] {prompt[:30]}...")
        
        print(f"  [{level}] {result.score:.3f} - {prompt[:50]}...")
    
    # Summary
    print(f"\n  Summary:")
    for level in ["HIGH", "MEDIUM", "LOW"]:
        mean = np.mean(results[level])
        std = np.std(results[level])
        print(f"    {level:6s}: {mean:.3f} ± {std:.3f}")
    
    h_mean = np.mean(results["HIGH"])
    m_mean = np.mean(results["MEDIUM"])
    l_mean = np.mean(results["LOW"])
    
    proper_order = h_mean >= m_mean >= l_mean
    discrimination = h_mean - l_mean
    
    print(f"\n  Discrimination (H-L): {discrimination:.3f}")
    print(f"  Proper ordering (H≥M≥L): {'✓ YES' if proper_order else '✗ NO'}")
    
    return results, all_trajectories, all_labels


def plot_trajectory_comparison(trajectories, labels, model_name, save_path):
    """Plot consciousness trajectories for multiple prompts."""
    fig, axes = plt.subplots(2, 3, figsize=(15, 10))
    fig.suptitle(f'{model_name} - Per-Token Consciousness Trajectories', fontsize=14, fontweight='bold')
    
    colors = ['#2ecc71', '#27ae60', '#f39c12', '#e67e22', '#e74c3c', '#c0392b']
    
    for idx, (traj, label) in enumerate(zip(trajectories[:6], labels[:6])):
        ax = axes[idx // 3, idx % 3]
        
        x = np.arange(len(traj.scores))
        bars = ax.bar(x, traj.scores, color=colors[idx], alpha=0.7, edgecolor='black', linewidth=0.5)
        
        # Trend line
        if len(traj.scores) > 1:
            z = np.polyfit(x, traj.scores, 1)
            p = np.poly1d(z)
            ax.plot(x, p(x), 'k--', linewidth=2, alpha=0.7)
        
        ax.set_title(label, fontsize=10)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('Token Position')
        ax.set_ylabel('Score')
        ax.grid(True, alpha=0.3)
        
        # Add mean line
        ax.axhline(y=traj.mean_score, color='blue', linestyle=':', alpha=0.7, 
                  label=f'Mean={traj.mean_score:.2f}')
        ax.legend(loc='upper right', fontsize=8)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_level_comparison(results_qwen, results_mistral, save_path):
    """Bar chart comparing consciousness by level for both models."""
    fig, ax = plt.subplots(figsize=(10, 6))
    
    levels = ["HIGH", "MEDIUM", "LOW"]
    x = np.arange(len(levels))
    width = 0.35
    
    qwen_means = [np.mean(results_qwen[l]) for l in levels]
    qwen_stds = [np.std(results_qwen[l]) for l in levels]
    mistral_means = [np.mean(results_mistral[l]) for l in levels]
    mistral_stds = [np.std(results_mistral[l]) for l in levels]
    
    bars1 = ax.bar(x - width/2, qwen_means, width, yerr=qwen_stds, 
                   label='Qwen2.5-7B', color='#3498db', capsize=5)
    bars2 = ax.bar(x + width/2, mistral_means, width, yerr=mistral_stds,
                   label='Mistral-7B', color='#e74c3c', capsize=5)
    
    ax.set_ylabel('Consciousness Score', fontsize=12)
    ax.set_xlabel('Expected Level', fontsize=12)
    ax.set_title('Cross-Model Consciousness Comparison\n(Mean Aggregation)', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(levels)
    ax.legend()
    ax.set_ylim(0, 1.0)
    ax.grid(True, alpha=0.3, axis='y')
    
    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f'{height:.2f}',
                       xy=(bar.get_x() + bar.get_width() / 2, height),
                       xytext=(0, 3),
                       textcoords="offset points",
                       ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def plot_dimension_heatmap(trajectory, model_name, prompt_label, save_path):
    """Heatmap of dimension activations across tokens."""
    import seaborn as sns
    
    fig, ax = plt.subplots(figsize=(14, 6))
    
    dim_names = list(trajectory.dimension_activations.keys())
    data = np.array([trajectory.dimension_activations[d] for d in dim_names])
    
    # Normalize
    data_norm = (data - data.mean()) / (data.std() + 1e-8)
    
    sns.heatmap(data_norm, 
               xticklabels=[t[:6] for t in trajectory.tokens[:len(data[0])]],
               yticklabels=dim_names,
               cmap='RdBu_r',
               center=0,
               ax=ax,
               cbar_kws={'label': 'Normalized Activation'})
    
    ax.set_xlabel('Token', fontsize=12)
    ax.set_ylabel('Dimension', fontsize=12)
    ax.set_title(f'{model_name} - Dimension Activations\n{prompt_label}', fontsize=12, fontweight='bold')
    
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"  Saved: {save_path}")
    plt.close()


def main():
    print("="*70)
    print("  CONSCIOUSNESS CIRCUIT v3.0 - Cross-Model Visualization Demo")
    print("="*70)
    
    # Initialize
    circuit = UniversalCircuit()
    viz = ConsciousnessVisualizer(circuit)
    plot_dir = '/home/akbon/unsloth_train/consciousness_plots'
    
    # Load models
    qwen_model, qwen_tok = load_model("Qwen/Qwen2.5-7B-Instruct", "cuda:0")
    
    # Test Qwen
    results_qwen, trajs_qwen, labels_qwen = test_model(
        qwen_model, qwen_tok, circuit, viz, "Qwen2.5-7B-Instruct"
    )
    
    # Plot Qwen trajectories
    print("\n  Generating Qwen plots...")
    plot_trajectory_comparison(
        trajs_qwen, labels_qwen, "Qwen2.5-7B-Instruct",
        f'{plot_dir}/qwen_trajectories.png'
    )
    
    # Dimension heatmap for HIGH prompt
    plot_dimension_heatmap(
        trajs_qwen[0], "Qwen2.5-7B",
        "[HIGH] Carefully examine assumptions...",
        f'{plot_dir}/qwen_dimensions_high.png'
    )
    
    # Free Qwen memory
    del qwen_model
    torch.cuda.empty_cache()
    
    # Load Mistral
    mistral_model, mistral_tok = load_model("mistralai/Mistral-7B-Instruct-v0.3", "cuda:0")
    
    # Test Mistral
    results_mistral, trajs_mistral, labels_mistral = test_model(
        mistral_model, mistral_tok, circuit, viz, "Mistral-7B-Instruct-v0.3"
    )
    
    # Plot Mistral trajectories
    print("\n  Generating Mistral plots...")
    plot_trajectory_comparison(
        trajs_mistral, labels_mistral, "Mistral-7B-Instruct-v0.3",
        f'{plot_dir}/mistral_trajectories.png'
    )
    
    # Dimension heatmap for HIGH prompt
    plot_dimension_heatmap(
        trajs_mistral[0], "Mistral-7B",
        "[HIGH] Carefully examine assumptions...",
        f'{plot_dir}/mistral_dimensions_high.png'
    )
    
    # Cross-model comparison
    print("\n  Generating cross-model comparison...")
    plot_level_comparison(results_qwen, results_mistral, f'{plot_dir}/cross_model_comparison.png')
    
    # Summary
    print("\n" + "="*70)
    print("  FINAL SUMMARY")
    print("="*70)
    
    print("\n  Qwen2.5-7B:")
    for level in ["HIGH", "MEDIUM", "LOW"]:
        print(f"    {level}: {np.mean(results_qwen[level]):.3f}")
    qwen_order = np.mean(results_qwen["HIGH"]) >= np.mean(results_qwen["MEDIUM"]) >= np.mean(results_qwen["LOW"])
    print(f"    Proper ordering: {'✓' if qwen_order else '✗'}")
    
    print("\n  Mistral-7B:")
    for level in ["HIGH", "MEDIUM", "LOW"]:
        print(f"    {level}: {np.mean(results_mistral[level]):.3f}")
    mistral_order = np.mean(results_mistral["HIGH"]) >= np.mean(results_mistral["MEDIUM"]) >= np.mean(results_mistral["LOW"])
    print(f"    Proper ordering: {'✓' if mistral_order else '✗'}")
    
    print(f"\n  Plots saved to: {plot_dir}/")
    print("    - qwen_trajectories.png")
    print("    - qwen_dimensions_high.png")
    print("    - mistral_trajectories.png")
    print("    - mistral_dimensions_high.png")
    print("    - cross_model_comparison.png")
    
    print("\n" + "="*70)
    print("  ✓ Demo complete!")
    print("="*70)


if __name__ == "__main__":
    main()
