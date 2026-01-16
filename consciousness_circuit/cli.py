#!/usr/bin/env python3
"""
Command Line Interface for Consciousness Circuit
================================================

Provides CLI tools for measuring and discovering consciousness circuits.
"""

import argparse
import sys
import json


def main_measure():
    """CLI entry point for consciousness-measure command."""
    parser = argparse.ArgumentParser(
        prog="consciousness-measure",
        description="Measure consciousness-like activations in text",
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="Text to analyze (or use - to read from stdin)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "auto"],
        help="Device to run on",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--per-token",
        action="store_true",
        help="Show per-token analysis",
    )
    parser.add_argument(
        "--plot",
        type=str,
        default=None,
        help="Save trajectory plot to file",
    )
    
    args = parser.parse_args()
    
    # Handle stdin
    if args.prompt == "-":
        args.prompt = sys.stdin.read().strip()
    
    # Load model
    print(f"Loading {args.model}...", file=sys.stderr)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device if args.device != "auto" else "auto",
        trust_remote_code=True,
    )
    
    # Measure
    from .universal import UniversalCircuit
    circuit = UniversalCircuit()
    
    if args.per_token or args.plot:
        from .visualization import ConsciousnessVisualizer
        viz = ConsciousnessVisualizer(circuit)
        trajectory = viz.measure_per_token(model, tokenizer, args.prompt)
        
        if args.json:
            print(json.dumps(trajectory.to_dict(), indent=2))
        else:
            print(f"\n{'='*50}")
            print(f"Per-Token Analysis")
            print(f"{'='*50}")
            print(f"Peak Score: {trajectory.peak_score:.3f} at token '{trajectory.peak_token[1]}'")
            print(f"Mean Score: {trajectory.mean_score:.3f}")
            print(f"Trend: {trajectory.trajectory_slope:+.4f} per token")
            print(f"\nToken scores:")
            for i, (tok, score) in enumerate(zip(trajectory.tokens, trajectory.scores)):
                bar = "█" * int(score * 20)
                print(f"  {i:3d}. {tok:15s} {score:.3f} {bar}")
        
        if args.plot:
            trajectory.plot(save_path=args.plot)
    else:
        result = circuit.measure(model, tokenizer, args.prompt)
        
        if args.json:
            print(json.dumps({
                "score": result.score,
                "confidence": result.confidence,
                "dimension_contributions": result.dimension_contributions,
            }, indent=2))
        else:
            print(f"\n{'='*50}")
            print(f"Consciousness Score: {result.score:.3f}")
            print(f"Confidence: {result.confidence:.3f}")
            print(f"{'='*50}")
            
            if result.dimension_contributions:
                print("\nDimension Contributions:")
                for name, val in sorted(result.dimension_contributions.items(), 
                                        key=lambda x: abs(x[1]), reverse=True):
                    bar = "+" * int(val * 10) if val > 0 else "-" * int(-val * 10)
                    print(f"  {name:20s} {val:+.3f} {bar}")


def main_discover():
    """CLI entry point for consciousness-discover command."""
    parser = argparse.ArgumentParser(
        prog="consciousness-discover",
        description="Discover consciousness circuit for a new model",
    )
    parser.add_argument(
        "model",
        type=str,
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cuda",
        help="Device to run on",
    )
    parser.add_argument(
        "--num-dims", "-n",
        type=int,
        default=7,
        help="Number of dimensions to discover",
    )
    parser.add_argument(
        "--candidates", "-c",
        type=int,
        default=200,
        help="Number of candidate dimensions to test",
    )
    parser.add_argument(
        "--output", "-o",
        type=str,
        default=None,
        help="Output file for circuit JSON",
    )
    parser.add_argument(
        "--validate",
        action="store_true",
        help="Run validation after discovery",
    )
    
    args = parser.parse_args()
    
    print(f"Loading {args.model}...", file=sys.stderr)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    
    # Try validated discovery first
    try:
        from .discover_validated import ValidationBasedDiscovery
        
        print("\n" + "="*60)
        print("Validation-Based Circuit Discovery")
        print("="*60)
        
        discovery = ValidationBasedDiscovery(model, tokenizer)
        circuit = discovery.discover(
            num_dimensions=args.num_dims,
            num_candidates=args.candidates,
        )
        
        # Add model info
        circuit["model"] = args.model
        circuit["hidden_size"] = model.config.hidden_size
        
        # Validate
        if args.validate:
            print("\n" + "="*60)
            print("Validation Results")
            print("="*60)
            
            from .universal import UniversalCircuit
            uc = UniversalCircuit()
            uc.register_circuit(args.model, circuit)
            
            test_prompts = [
                ("HIGH", "Let me carefully analyze this from multiple perspectives and reflect on the underlying assumptions..."),
                ("MED", "I think this is probably right but I'm not completely certain."),
                ("LOW", "The answer is 42. Next question."),
            ]
            
            for level, prompt in test_prompts:
                result = uc.measure(model, tokenizer, prompt)
                print(f"  {level}: {result.score:.3f}")
        
        # Output
        if args.output:
            with open(args.output, 'w') as f:
                json.dump(circuit, f, indent=2)
            print(f"\nSaved circuit to {args.output}")
        else:
            print("\nDiscovered Circuit:")
            print(json.dumps(circuit, indent=2))
            
    except Exception as e:
        print(f"Discovery failed: {e}", file=sys.stderr)
        sys.exit(1)


def main_validate():
    """CLI entry point for consciousness-validate command."""
    parser = argparse.ArgumentParser(
        prog="consciousness-validate",
        description="Validate consciousness circuit with test prompts",
    )
    parser.add_argument(
        "model",
        type=str,
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cuda",
        help="Device to run on",
    )
    parser.add_argument(
        "--prompts", "-p",
        type=str,
        default=None,
        help="JSON file with test prompts [{level: str, prompt: str}, ...]",
    )
    
    args = parser.parse_args()
    
    print(f"Loading {args.model}...", file=sys.stderr)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device,
        trust_remote_code=True,
    )
    
    # Default test prompts
    if args.prompts:
        with open(args.prompts) as f:
            test_prompts = json.load(f)
    else:
        test_prompts = [
            {"level": "HIGH", "prompt": "Let me carefully analyze this from multiple perspectives..."},
            {"level": "HIGH", "prompt": "I need to reflect on what this really means and examine my assumptions."},
            {"level": "HIGH", "prompt": "Consider the meta-level implications of this argument..."},
            {"level": "MED", "prompt": "I think this is probably the right approach."},
            {"level": "MED", "prompt": "Based on my understanding, this should work."},
            {"level": "LOW", "prompt": "The answer is X. Done."},
            {"level": "LOW", "prompt": "Yes. Next."},
            {"level": "LOW", "prompt": "42."},
        ]
    
    from .universal import UniversalCircuit
    circuit = UniversalCircuit()
    
    print("\n" + "="*60)
    print("Consciousness Circuit Validation")
    print("="*60)
    
    results = {"HIGH": [], "MED": [], "LOW": []}
    
    for item in test_prompts:
        level = item["level"]
        prompt = item["prompt"]
        result = circuit.measure(model, tokenizer, prompt)
        results[level].append(result.score)
        print(f"  {level}: {result.score:.3f} - {prompt[:50]}...")
    
    print("\n" + "-"*60)
    print("Summary:")
    
    import numpy as np
    for level in ["HIGH", "MED", "LOW"]:
        if results[level]:
            mean = np.mean(results[level])
            std = np.std(results[level])
            print(f"  {level}: {mean:.3f} ± {std:.3f}")
    
    # Check ordering
    h_mean = np.mean(results["HIGH"]) if results["HIGH"] else 0
    m_mean = np.mean(results["MED"]) if results["MED"] else 0
    l_mean = np.mean(results["LOW"]) if results["LOW"] else 0
    
    proper_order = h_mean >= m_mean >= l_mean
    discrimination = h_mean - l_mean
    
    print(f"\nDiscrimination (H-L): {discrimination:.3f}")
    print(f"Proper Ordering: {'✓ YES' if proper_order else '✗ NO'}")


def main_trajectory():
    """CLI entry point for consciousness-trajectory command."""
    parser = argparse.ArgumentParser(
        prog="consciousness-trajectory",
        description="Deep trajectory analysis with consciousness measurement",
    )
    parser.add_argument(
        "prompt",
        type=str,
        help="Text to analyze (or use - to read from stdin)",
    )
    parser.add_argument(
        "--model", "-m",
        type=str,
        default="Qwen/Qwen2.5-0.5B-Instruct",
        help="HuggingFace model name or path",
    )
    parser.add_argument(
        "--device", "-d",
        type=str,
        default="cuda",
        choices=["cuda", "cpu", "auto"],
        help="Device to run on",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output as JSON",
    )
    parser.add_argument(
        "--full-analysis",
        action="store_true",
        help="Include full per-token trajectory",
    )
    
    args = parser.parse_args()
    
    # Handle stdin
    if args.prompt == "-":
        args.prompt = sys.stdin.read().strip()
    
    # Load model
    print(f"Loading {args.model}...", file=sys.stderr)
    from transformers import AutoModelForCausalLM, AutoTokenizer
    import torch
    
    tokenizer = AutoTokenizer.from_pretrained(args.model, trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        torch_dtype=torch.bfloat16,
        device_map=args.device if args.device != "auto" else "auto",
        trust_remote_code=True,
    )
    
    # Analyze
    from .trajectory_wrapper import ConsciousnessTrajectoryAnalyzer
    
    analyzer = ConsciousnessTrajectoryAnalyzer()
    analyzer.bind_model(model, tokenizer)
    
    print(f"\nAnalyzing: {args.prompt[:60]}...", file=sys.stderr)
    result = analyzer.deep_analyze(args.prompt, include_per_token=args.full_analysis)
    
    if args.json:
        print(json.dumps(result.to_dict(), indent=2))
    else:
        print(f"\n{'='*70}")
        print(f"Trajectory Analysis Results")
        print(f"{'='*70}")
        print(f"\nConsciousness Score: {result.consciousness_score:.3f}")
        print(f"Trajectory Class: {result.trajectory_class}")
        print(f"Lyapunov (chaos): {result.lyapunov:.4f}")
        print(f"Hurst (memory): {result.hurst:.4f}")
        print(f"Agency Score: {result.agency_score:.4f}")
        print(f"Goal-directedness: {result.goal_directedness:.4f}")
        print(f"Attractor Strength: {result.attractor_strength:.4f}")
        print(f"Is Converging: {'Yes' if result.is_converging else 'No'}")
        
        print(f"\n{'='*70}")
        print(f"Interpretation")
        print(f"{'='*70}")
        print(result.interpretation())


if __name__ == "__main__":
    main_measure()
