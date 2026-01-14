#!/usr/bin/env python3
"""Test the consciousness-circuit package installation and per-token analysis."""

import sys
sys.path.insert(0, '/home/akbon/unsloth_train/consciousness_circuit')

print("Testing consciousness-circuit package...")

# Test imports
from consciousness_circuit import (
    UniversalCircuit,
    ConsciousnessVisualizer,
    TokenTrajectory,
    measure_consciousness,
)
print("✓ All imports successful!")

# Show what's available
print("\nAvailable classes:")
print(f"  - UniversalCircuit: {UniversalCircuit}")
print(f"  - ConsciousnessVisualizer: {ConsciousnessVisualizer}")
print(f"  - TokenTrajectory: {TokenTrajectory}")

# Test with Qwen 7B
print("\n" + "="*60)
print("Testing Per-Token Analysis with Qwen2.5-7B-Instruct")
print("="*60)

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

model_name = "Qwen/Qwen2.5-7B-Instruct"
print(f"\nLoading {model_name}...")

tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=torch.bfloat16,
    device_map="cuda:0",
    trust_remote_code=True,
)
print("✓ Model loaded!")

# Initialize visualizer
viz = ConsciousnessVisualizer()

# Test prompts
prompts = [
    "Let me carefully analyze this problem from multiple perspectives...",
    "The answer is 42.",
]

for prompt in prompts:
    print(f"\n{'─'*50}")
    print(f"Prompt: {prompt[:60]}...")
    
    # Per-token analysis
    trajectory = viz.measure_per_token(model, tokenizer, prompt)
    
    print(f"\n  Peak Score: {trajectory.peak_score:.3f} at token '{trajectory.peak_token[1]}'")
    print(f"  Mean Score: {trajectory.mean_score:.3f}")
    print(f"  Trend: {trajectory.trajectory_slope:+.4f} per token")
    
    # Show first 10 tokens
    print(f"\n  Token trajectory (first 10):")
    for i, (tok, score) in enumerate(zip(trajectory.tokens[:10], trajectory.scores[:10])):
        bar = "█" * int(score * 20)
        print(f"    {i:2d}. {tok:12s} {score:.3f} {bar}")

print("\n" + "="*60)
print("✓ Per-token analysis works!")
print("="*60)

# Test comparison
print("\nTesting prompt comparison...")
comparison = viz.compare_prompts(model, tokenizer, prompts)
print(f"\nComparison results:")
for label, score in zip(comparison.labels, comparison.scores):
    print(f"  {score:.3f} - {label}")

print("\n✓ All tests passed!")
