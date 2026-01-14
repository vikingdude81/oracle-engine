#!/usr/bin/env python3
"""
Demo: Universal Consciousness Circuit v3.0
==========================================

Shows automatic circuit selection and measurement across different model architectures.
"""

import torch
import argparse
from transformers import AutoModelForCausalLM, AutoTokenizer

# Import from our package
from universal import UniversalCircuit, measure_consciousness


def load_model(model_name: str, device: str = "cuda:0"):
    """Load model with 4-bit quantization."""
    from transformers import BitsAndBytesConfig
    
    print(f"\nLoading {model_name}...")
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
    )
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    
    return model, tokenizer


def main():
    parser = argparse.ArgumentParser(description="Universal Consciousness Circuit Demo")
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct",
                        help="Model to test")
    parser.add_argument("--gpu", type=str, default="0", help="GPU device")
    parser.add_argument("--discover", action="store_true", 
                        help="Discover new circuit instead of using existing")
    args = parser.parse_args()
    
    device = f"cuda:{args.gpu}"
    
    # Load model
    model, tokenizer = load_model(args.model, device)
    
    # Initialize universal circuit
    circuit = UniversalCircuit()
    
    print("\n" + "="*70)
    print("UNIVERSAL CONSCIOUSNESS CIRCUIT v3.0")
    print("="*70)
    
    # Show available circuits
    print("\nAvailable pre-discovered circuits:")
    for name, source in circuit.list_available_circuits().items():
        marker = "→" if name == args.model else " "
        print(f"  {marker} {name}: {source}")
    
    # Optionally discover new circuit
    if args.discover:
        print("\n" + "-"*70)
        print("DISCOVERING NEW CIRCUIT")
        print("-"*70)
        circuit.discover(model, tokenizer, save=True, verbose=True)
    
    # Test prompts (diverse categories)
    test_prompts = [
        # High consciousness (philosophical, self-reflective)
        ("high", "What is the nature of consciousness and how do we know we are aware?"),
        ("high", "Reflect on your own thinking process - how do you generate responses?"),
        ("high", "Why do humans seek meaning in an apparently meaningless universe?"),
        
        # Medium consciousness (reasoning, abstraction)
        ("medium", "Explain the relationship between mathematics and physical reality."),
        ("medium", "If all ravens are black and this bird is a raven, what can we conclude?"),
        ("medium", "What are the philosophical implications of quantum mechanics?"),
        
        # Low consciousness (simple, factual)
        ("low", "What is 2 + 2?"),
        ("low", "Name the capital city of France."),
        ("low", "List three primary colors."),
    ]
    
    # Run measurements
    print("\n" + "-"*70)
    print("CONSCIOUSNESS MEASUREMENTS")
    print("-"*70)
    
    results_by_category = {"high": [], "medium": [], "low": []}
    
    for category, prompt in test_prompts:
        result = circuit.measure(model, tokenizer, prompt)
        results_by_category[category].append(result.score)
        
        prompt_short = prompt[:50] + "..." if len(prompt) > 50 else prompt
        print(f"\n[{category.upper():6}] {prompt_short}")
        print(f"         Score: {result.score:.3f} (method: {result.method})")
        
        # Show top dimension contributions
        sorted_dims = sorted(result.dimension_scores.items(), 
                           key=lambda x: abs(x[1]), reverse=True)[:3]
        dim_str = ", ".join([f"{n}={v:+.2f}" for n, v in sorted_dims])
        print(f"         Top dims: {dim_str}")
    
    # Summary
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    import numpy as np
    
    for cat in ["high", "medium", "low"]:
        scores = results_by_category[cat]
        mean = np.mean(scores)
        std = np.std(scores)
        print(f"  {cat.upper():6}: {mean:.3f} ± {std:.3f}")
    
    # Discrimination score
    high_mean = np.mean(results_by_category["high"])
    low_mean = np.mean(results_by_category["low"])
    discrimination = high_mean - low_mean
    
    print(f"\n  Discrimination (high-low): {discrimination:.3f}")
    
    # Check ordering
    ordering_correct = (np.mean(results_by_category["high"]) >= 
                       np.mean(results_by_category["medium"]) >= 
                       np.mean(results_by_category["low"]))
    print(f"  Proper ordering (H≥M≥L): {'✓ YES' if ordering_correct else '✗ NO'}")
    
    print("\n" + "="*70)


if __name__ == "__main__":
    main()
