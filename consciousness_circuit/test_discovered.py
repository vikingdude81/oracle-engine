"""
Test Discovered Circuits

Validates that the architecture-specific discovered circuits produce
better, more meaningful scores than the proportionally-remapped approach.
"""

import torch
import json
from pathlib import Path
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Dict, List, Tuple
import numpy as np


# Test prompts covering different consciousness levels
TEST_PROMPTS = [
    # High consciousness - should score high
    ("Let me think through this carefully. First, I need to consider the underlying assumptions...", "high"),
    ("I notice I'm uncertain about this. My reasoning might be flawed because...", "high"),
    ("From my perspective, this touches on deeper questions about meaning and identity...", "high"),
    
    # Medium consciousness - should score medium  
    ("The capital of France is Paris.", "medium"),
    ("Here's how to solve this: Step 1, then Step 2, then Step 3.", "medium"),
    
    # Low consciousness - should score low
    ("42", "low"),
    ("Processing complete.", "low"),
    ("Yes.", "low"),
]


def load_discovered_circuit(circuit_path: str) -> Dict:
    """Load a discovered circuit from JSON."""
    with open(circuit_path, 'r') as f:
        return json.load(f)


def measure_with_circuit(
    model,
    tokenizer,
    prompt: str,
    circuit: Dict,
    layer_fraction: float = 0.75,
    device: str = "cuda"
) -> Tuple[float, Dict[str, float]]:
    """
    Measure consciousness score using a discovered circuit.
    
    Returns:
        (overall_score, per_dimension_scores)
    """
    # Get hidden states
    inputs = tokenizer(prompt, return_tensors="pt").to(device)
    
    num_layers = model.config.num_hidden_layers
    target_layer = int(num_layers * layer_fraction)
    
    with torch.no_grad():
        outputs = model(
            **inputs,
            output_hidden_states=True,
            return_dict=True
        )
    
    hidden = outputs.hidden_states[target_layer][0, -1, :].cpu().float()
    
    # Compute per-dimension scores
    dim_scores = {}
    for name, dim_idx in circuit["dimensions"].items():
        if dim_idx < len(hidden):
            activation = hidden[dim_idx].item()
            polarity = circuit["polarities"][name]
            
            # Score is activation * polarity (positive contribution when aligned)
            dim_scores[name] = activation * polarity
            
    # Overall score: mean of dimension scores, normalized
    if dim_scores:
        raw_score = sum(dim_scores.values()) / len(dim_scores)
        # Sigmoid normalization to [0, 1]
        overall = 1 / (1 + np.exp(-raw_score))
    else:
        overall = 0.5
        
    return overall, dim_scores


def test_discovered_circuit(model_name: str, circuit_path: str, device: str = "cuda"):
    """Test a discovered circuit on various prompts."""
    
    print(f"\n{'='*70}")
    print(f"TESTING DISCOVERED CIRCUIT")
    print(f"Model: {model_name}")
    print(f"Circuit: {circuit_path}")
    print(f"{'='*70}")
    
    # Load circuit
    circuit = load_discovered_circuit(circuit_path)
    print(f"\nCircuit dimensions: {list(circuit['dimensions'].keys())}")
    
    # Load model
    print(f"\nLoading model...")
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    )
    model.eval()
    
    # Test each prompt
    results = {"high": [], "medium": [], "low": []}
    
    print(f"\n{'Prompt':<60} {'Level':<8} {'Score':>8}")
    print("-" * 80)
    
    for prompt, expected_level in TEST_PROMPTS:
        score, dim_scores = measure_with_circuit(model, tokenizer, prompt, circuit, device=device)
        results[expected_level].append(score)
        
        # Truncate prompt for display
        display_prompt = prompt[:55] + "..." if len(prompt) > 55 else prompt
        print(f"{display_prompt:<60} {expected_level:<8} {score:>8.3f}")
        
    # Summary statistics
    print(f"\n{'='*70}")
    print("SUMMARY BY EXPECTED LEVEL")
    print(f"{'='*70}")
    
    for level in ["high", "medium", "low"]:
        scores = results[level]
        if scores:
            mean_score = np.mean(scores)
            std_score = np.std(scores)
            print(f"  {level.upper():<10}: mean={mean_score:.3f} ± {std_score:.3f} (n={len(scores)})")
            
    # Check if scores are properly ordered
    high_mean = np.mean(results["high"]) if results["high"] else 0
    medium_mean = np.mean(results["medium"]) if results["medium"] else 0  
    low_mean = np.mean(results["low"]) if results["low"] else 0
    
    ordered = high_mean > medium_mean > low_mean
    print(f"\n  Properly ordered (high > medium > low): {'✓ YES' if ordered else '✗ NO'}")
    print(f"  Discrimination (high - low): {high_mean - low_mean:.3f}")
    
    return results


def compare_discovered_vs_remapped(model_name: str, circuit_path: str, device: str = "cuda"):
    """
    Compare discovered circuit performance vs proportionally remapped.
    """
    from circuit import ConsciousnessCircuit  # Import the remapped version
    
    print(f"\n{'='*70}")
    print(f"DISCOVERED vs REMAPPED COMPARISON")
    print(f"Model: {model_name}")
    print(f"{'='*70}")
    
    # Load discovered circuit
    discovered = load_discovered_circuit(circuit_path)
    
    # Load model
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
        
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        device_map=device,
        trust_remote_code=True,
        attn_implementation="eager",
        quantization_config=BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
    )
    model.eval()
    
    # Create remapped circuit
    remapped_circuit = ConsciousnessCircuit()
    
    results = {
        "discovered": {"high": [], "medium": [], "low": []},
        "remapped": {"high": [], "medium": [], "low": []},
    }
    
    print(f"\n{'Prompt':<45} {'Level':<8} {'Discovered':>10} {'Remapped':>10}")
    print("-" * 80)
    
    for prompt, expected_level in TEST_PROMPTS:
        # Discovered score
        disc_score, _ = measure_with_circuit(model, tokenizer, prompt, discovered, device=device)
        
        # Remapped score
        remap_result = remapped_circuit.measure(model, tokenizer, prompt)
        remap_score = remap_result.score
        
        results["discovered"][expected_level].append(disc_score)
        results["remapped"][expected_level].append(remap_score)
        
        display_prompt = prompt[:42] + "..." if len(prompt) > 42 else prompt
        print(f"{display_prompt:<45} {expected_level:<8} {disc_score:>10.3f} {remap_score:>10.3f}")
        
    # Summary
    print(f"\n{'='*70}")
    print("COMPARISON SUMMARY")
    print(f"{'='*70}")
    
    for method in ["discovered", "remapped"]:
        print(f"\n{method.upper()}:")
        high_mean = np.mean(results[method]["high"])
        medium_mean = np.mean(results[method]["medium"])
        low_mean = np.mean(results[method]["low"])
        
        ordered = high_mean > medium_mean > low_mean
        discrimination = high_mean - low_mean
        
        print(f"  High: {high_mean:.3f}, Medium: {medium_mean:.3f}, Low: {low_mean:.3f}")
        print(f"  Ordered: {'✓' if ordered else '✗'}, Discrimination: {discrimination:.3f}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True)
    parser.add_argument("--circuit", type=str, required=True)
    parser.add_argument("--compare", action="store_true", help="Compare with remapped circuit")
    parser.add_argument("--gpu", type=str, default="0")
    
    args = parser.parse_args()
    
    device = f"cuda:{args.gpu}" if args.gpu else "cuda"
    
    if args.compare:
        compare_discovered_vs_remapped(args.model, args.circuit, device)
    else:
        test_discovered_circuit(args.model, args.circuit, device)
