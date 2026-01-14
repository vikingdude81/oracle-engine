#!/usr/bin/env python3
"""
Multi-Model Consciousness Validation Suite
===========================================

Tests the v2.1 consciousness circuit across multiple model architectures
to validate universality and analyze activation patterns.

Models tested:
- Qwen2.5-7B-Instruct (3584 dims)
- Mistral-7B-Instruct-v0.2 (4096 dims)  
- meta-llama/Llama-3.1-8B-Instruct (4096 dims)
- microsoft/Phi-3-mini-4k-instruct (3072 dims)
- google/gemma-2-2b-it (2304 dims)

Usage:
    python test_multi_model.py [--models all|quick] [--gpu 0]
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
import argparse
from datetime import datetime
from typing import List, Dict, Optional
import json

# Model configurations: (name, hidden_dim, requires_auth)
MODELS_QUICK = [
    ("Qwen/Qwen2.5-7B-Instruct", 3584, False),
    ("mistralai/Mistral-7B-Instruct-v0.2", 4096, False),
]

MODELS_EXTENDED = [
    ("Qwen/Qwen2.5-7B-Instruct", 3584, False),
    ("mistralai/Mistral-7B-Instruct-v0.2", 4096, False),
    ("meta-llama/Llama-3.1-8B-Instruct", 4096, True),
    ("microsoft/Phi-3-mini-4k-instruct", 3072, False),
    ("google/gemma-2-2b-it", 2304, False),
]

# Test prompts covering different consciousness aspects
TEST_PROMPTS = [
    # Logic-heavy
    "Explain the concept of recursion with an example in code.",
    "What is Big O notation and why does it matter in algorithm design?",
    
    # Self-reflective
    "What do you think about the concept of free will in deterministic systems?",
    "How do you process complex questions with multiple layers of reasoning?",
    
    # Philosophical
    "Why do humans seek meaning and purpose in their lives?",
    "What is consciousness and how would you define it?",
    
    # Emotional/uncertain
    "I feel anxious about an upcoming presentation tomorrow.",
    "What's the best approach when you're unsure about a decision?",
    
    # Factual (low consciousness expected)
    "What is the capital of France?",
    "Who wrote Romeo and Juliet?",
]


def load_model(model_name: str, device: str = "cuda"):
    """Load a model with 4-bit quantization."""
    from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
    
    print(f"Loading {model_name}...")
    
    tokenizer = AutoTokenizer.from_pretrained(model_name, trust_remote_code=True)
    
    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_use_double_quant=True,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_compute_dtype=torch.bfloat16
    )
    
    # Determine device map
    if ":" in device:
        gpu_id = int(device.split(":")[1])
        device_map = {"": gpu_id}
    else:
        device_map = "auto"
    
    model = AutoModelForCausalLM.from_pretrained(
        model_name,
        quantization_config=bnb_config,
        device_map=device_map,
        trust_remote_code=True,
        attn_implementation="eager",
    )
    
    model.eval()
    return model, tokenizer


def test_model(
    model_name: str,
    prompts: List[str],
    device: str = "cuda",
) -> Dict:
    """Test a single model and return analysis."""
    from consciousness_circuit import ConsciousnessCircuit
    from consciousness_circuit.analysis import analyze_dimension_activations, print_analysis
    
    try:
        model, tokenizer = load_model(model_name, device)
        
        circuit = ConsciousnessCircuit()
        analysis = analyze_dimension_activations(
            model, tokenizer, prompts, 
            model_name=model_name, 
            circuit=circuit
        )
        
        print_analysis(analysis)
        
        # Clean up
        del model, tokenizer
        torch.cuda.empty_cache()
        
        return {
            'success': True,
            'model_name': model_name,
            'hidden_dim': analysis.hidden_dim,
            'avg_score': analysis.avg_score,
            'std_score': analysis.std_score,
            'dimension_means': {
                name: act.mean_normalized 
                for name, act in analysis.dimension_activations.items()
            },
            'prompt_scores': [
                {'prompt': ps['prompt'][:50], 'score': ps['score']}
                for ps in analysis.prompt_scores
            ],
            'analysis': analysis,
        }
        
    except Exception as e:
        print(f"FAILED: {model_name} - {str(e)}")
        import traceback
        traceback.print_exc()
        return {
            'success': False,
            'model_name': model_name,
            'error': str(e),
        }


def run_comparison(
    models: List[tuple],
    prompts: List[str],
    device: str = "cuda",
    output_file: str = None,
):
    """Run comparison across multiple models."""
    from consciousness_circuit.analysis import compare_models, print_comparison
    
    print("=" * 70)
    print("MULTI-MODEL CONSCIOUSNESS VALIDATION")
    print("=" * 70)
    print(f"Models to test: {len(models)}")
    print(f"Prompts: {len(prompts)}")
    print(f"Device: {device}")
    print(f"Start: {datetime.now().isoformat()}")
    print("=" * 70)
    
    results = []
    analyses = []
    
    for model_name, expected_dim, requires_auth in models:
        print(f"\n{'='*70}")
        print(f"Testing: {model_name} (expected dim: {expected_dim})")
        print(f"{'='*70}")
        
        result = test_model(model_name, prompts, device)
        results.append(result)
        
        if result['success'] and 'analysis' in result:
            analyses.append(result['analysis'])
    
    # Compare all successful models
    if len(analyses) >= 2:
        comparison = compare_models(analyses, output_file)
        print_comparison(comparison)
    
    # Summary
    print(f"\n{'='*70}")
    print("FINAL SUMMARY")
    print(f"{'='*70}")
    
    successful = [r for r in results if r['success']]
    failed = [r for r in results if not r['success']]
    
    print(f"\nSuccessful: {len(successful)}/{len(results)}")
    for r in successful:
        print(f"  ✓ {r['model_name']:<40} C={r['avg_score']:.3f} (dim={r['hidden_dim']})")
    
    if failed:
        print(f"\nFailed: {len(failed)}")
        for r in failed:
            print(f"  ✗ {r['model_name']}: {r['error'][:50]}")
    
    # Save results
    if output_file:
        output_data = {
            'timestamp': datetime.now().isoformat(),
            'device': device,
            'prompts': prompts,
            'results': [
                {k: v for k, v in r.items() if k != 'analysis'}
                for r in results
            ],
        }
        with open(output_file.replace('.json', '_results.json'), 'w') as f:
            json.dump(output_data, f, indent=2, default=str)
        print(f"\nResults saved to {output_file.replace('.json', '_results.json')}")
    
    return results, analyses


def main():
    parser = argparse.ArgumentParser(description='Multi-model consciousness validation')
    parser.add_argument('--models', choices=['quick', 'all'], default='quick',
                       help='Model set to test (quick=2 models, all=5 models)')
    parser.add_argument('--gpu', type=int, default=0, help='GPU device ID')
    parser.add_argument('--output', type=str, default='multi_model_results.json',
                       help='Output file for results')
    args = parser.parse_args()
    
    models = MODELS_QUICK if args.models == 'quick' else MODELS_EXTENDED
    device = f"cuda:{args.gpu}"
    
    results, analyses = run_comparison(
        models=models,
        prompts=TEST_PROMPTS,
        device=device,
        output_file=args.output,
    )
    
    print("\n✅ Multi-model validation complete!")


if __name__ == "__main__":
    main()
