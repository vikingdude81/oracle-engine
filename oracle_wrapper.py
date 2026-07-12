#!/usr/bin/env python3
"""
Oracle Wrapper - Modular Pipeline for Model Analysis & Improvement
===================================================================

A composable CLI wrapper around the consciousness circuit suite.
Designed for iterative model analysis and improvement loops.

Pipeline Architecture:
----------------------
    [Model] --> [Analyzer] --> [Profiler] --> [Reporter] --> [JSON/CSV/Viz]
        ^                                          |
        |_________ [Improvement Suggestions] ______|

Core Components (pipe-able):
---------------------------
1. load    - Load model (custom or base)
2. analyze - Single prompt analysis  
3. profile - Full test suite profiling
4. compare - Compare two profiles/models
5. report  - Generate improvement suggestions
6. export  - Output to JSON/CSV for downstream tools

Usage Examples:
--------------
    # Quick single analysis
    python oracle_wrapper.py analyze "What is consciousness?"
    
    # Full profile with export
    python oracle_wrapper.py profile --category philosophical --export json
    
    # Compare custom vs base model
    python oracle_wrapper.py compare --custom --base --export report
    
    # Pipe to improvement suggestions
    python oracle_wrapper.py profile --all | python oracle_wrapper.py suggest
    
    # Generate training feedback
    python oracle_wrapper.py feedback --profile philosophical --target 0.8

WSL2 Usage:
-----------
    cd /home/akbon/unsloth_train
    source .venv/bin/activate
    python /mnt/c/Users/.../oracle-engine/oracle_wrapper.py <command>
"""

# ==============================================================================
# Environment setup (MUST be first for RTX 5090)
# ==============================================================================
import os
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

import builtins
try:
    import psutil
    builtins.psutil = psutil
except ImportError:
    pass

import sys
import json
import argparse
from pathlib import Path
from datetime import datetime
from typing import Optional, Dict, Any, List
from dataclasses import dataclass, asdict

# ==============================================================================
# Path Configuration (override via ORACLE_CUSTOM_MODEL_PATH / ORACLE_BASE_MODEL)
# ==============================================================================
from oracle_config import CUSTOM_MODEL_PATH, BASE_MODEL_NAME

# NanoGPT presets (smaller models for quick testing)
NANOGPT_PRESETS = {
    'shakespeare': 'shakespeare_final',
    'shakespeare_final': 'shakespeare_final',
    'shakespeare_epoch5': 'shakespeare_epoch5',
    'shakespeare_epoch1': 'shakespeare_epoch1',
    'tinystories': 'tinystories',
    'trm': 'trm',
    'trm_best': 'trm_best',
}

# ==============================================================================
# Lazy Imports (to avoid loading everything upfront)
# ==============================================================================

_model = None
_tokenizer = None
_nanogpt_adapter = None
_analyzer = None
_profiler = None


def get_analyzer():
    """Lazy load the trajectory analyzer."""
    global _analyzer
    if _analyzer is None:
        from consciousness_circuit.analyzers import ConsciousnessTrajectoryAnalyzer
        _analyzer = ConsciousnessTrajectoryAnalyzer()
    return _analyzer


def get_profiler():
    """Lazy load the profiler."""
    global _profiler
    if _profiler is None:
        from consciousness_circuit.benchmarks import ModelProfiler
        _profiler = ModelProfiler(get_analyzer())
    return _profiler


def load_nanogpt_model(preset_or_path: str, device: str = 'cuda'):
    """
    Load a NanoGPT model for analysis.
    
    Args:
        preset_or_path: Preset name ('shakespeare', 'tinystories', 'trm')
                        or full path to .pt checkpoint
        device: 'cuda' or 'cpu'
    
    Returns:
        NanoGPTAdapter instance
    """
    global _nanogpt_adapter
    
    from consciousness_circuit.model_adapters import load_nanogpt, list_nanogpt_models
    
    # Resolve preset name
    if preset_or_path in NANOGPT_PRESETS:
        preset_name = NANOGPT_PRESETS[preset_or_path]
    else:
        preset_name = preset_or_path
    
    print(f"🔧 Loading NanoGPT model: {preset_name}")
    
    # Show available models if invalid
    available = list_nanogpt_models()
    if preset_name not in available and not Path(preset_name).exists():
        print(f"   ❌ Model not found: {preset_name}")
        print(f"   Available models:")
        for name, info in available.items():
            status = "✅" if info['exists'] else "❌"
            size = f"{info['size_mb']:.1f}MB" if info['size_mb'] else "N/A"
            print(f"      {status} {name}: {size}")
        return None
    
    _nanogpt_adapter = load_nanogpt(preset_name, device=device)
    
    print(f"   ✅ Loaded NanoGPT model")
    print(f"   Hidden size: 768 | Layers: 12\n")
    
    return _nanogpt_adapter


def load_model(use_base: bool = False, model_path: Optional[str] = None, nanogpt: Optional[str] = None):
    """
    Load model and bind to analyzer.
    
    Args:
        use_base: Use base model instead of custom
        model_path: Custom path to model (overrides defaults)
        nanogpt: Load a NanoGPT model instead (preset name or path)
    """
    global _model, _tokenizer, _nanogpt_adapter
    
    # Handle NanoGPT models
    if nanogpt:
        _nanogpt_adapter = load_nanogpt_model(nanogpt, device='cpu')
        return _nanogpt_adapter, None
    
    if model_path:
        target = model_path
    elif use_base:
        target = BASE_MODEL_NAME
    else:
        target = CUSTOM_MODEL_PATH
    
    print(f"🔧 Loading model: {target}")
    print("   (This may take a few minutes...)\n")
    
    try:
        from unsloth import FastLanguageModel
        _model, _tokenizer = FastLanguageModel.from_pretrained(
            model_name=target,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(_model)
        print(f"   ✅ Loaded via Unsloth (4-bit)")
    except ImportError:
        from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
        import torch
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.float16,
        )
        _model = AutoModelForCausalLM.from_pretrained(
            target,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
        )
        _tokenizer = AutoTokenizer.from_pretrained(target, trust_remote_code=True)
        print(f"   ✅ Loaded via Transformers (4-bit)")
    
    print(f"   Hidden size: {_model.config.hidden_size}")
    print(f"   Layers: {_model.config.num_hidden_layers}\n")
    
    # Bind to analyzer
    get_analyzer().bind_model(_model, _tokenizer)
    
    return _model, _tokenizer


# ==============================================================================
# Analysis Functions (Pipe-able)
# ==============================================================================

@dataclass
class AnalysisResult:
    """Single prompt analysis result."""
    prompt: str
    consciousness_score: float
    trajectory_class: str
    lyapunov: float
    hurst: float
    agency_score: float
    is_converging: bool
    interpretation: str
    dimension_scores: Dict[str, float]
    timestamp: str = ""
    
    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
    
    def to_dict(self) -> dict:
        return asdict(self)
    
    def to_json(self) -> str:
        return json.dumps(self.to_dict(), indent=2)


def analyze_prompt(prompt: str, verbose: bool = True) -> AnalysisResult:
    """
    Analyze a single prompt.
    
    Args:
        prompt: The prompt to analyze
        verbose: Print interpretation
    
    Returns:
        AnalysisResult with all metrics
    """
    analyzer = get_analyzer()
    result = analyzer.deep_analyze(prompt)
    
    analysis = AnalysisResult(
        prompt=prompt,
        consciousness_score=result.consciousness_score,
        trajectory_class=result.trajectory_class,
        lyapunov=result.lyapunov,
        hurst=result.hurst,
        agency_score=result.agency_score,
        is_converging=result.is_converging,
        interpretation=result.interpretation(),
        dimension_scores=result.dimension_scores,
    )
    
    if verbose:
        print("=" * 80)
        print(f"PROMPT: {prompt[:60]}...")
        print("=" * 80)
        print(result.interpretation())
        print(f"\nScore: {result.consciousness_score:.3f}")
        print("=" * 80)
    
    return analysis


def profile_category(
    category: str,
    name: Optional[str] = None,
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Profile model on a specific category.
    
    Args:
        category: One of 'philosophical', 'factual', 'reasoning', 'creative'
        name: Profile name (auto-generated if None)
        verbose: Print summary
    
    Returns:
        ProfileResult as dictionary
    """
    from consciousness_circuit.benchmarks import get_test_suite
    
    prompts = get_test_suite(category)
    profile_name = name or f"{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    profiler = get_profiler()
    profile = profiler.profile(prompts, name=profile_name, store_results=True)
    
    if verbose:
        print(profile.summary())
    
    return profile.to_dict()


def profile_all(verbose: bool = True) -> Dict[str, Dict[str, Any]]:
    """
    Profile model on all categories.
    
    Returns:
        Dictionary of category -> ProfileResult
    """
    from consciousness_circuit.benchmarks import get_full_benchmark
    
    all_prompts = get_full_benchmark()
    results = {}
    
    for category, prompts in all_prompts.items():
        print(f"\n{'='*80}")
        print(f"PROFILING: {category.upper()}")
        print(f"{'='*80}")
        
        profiler = get_profiler()
        profile_name = f"{category}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        profile = profiler.profile(prompts, name=profile_name, store_results=True)
        results[category] = profile.to_dict()
        
        if verbose:
            print(profile.summary())
    
    return results


def compare_models(
    custom_profile: Dict[str, Any],
    base_profile: Dict[str, Any],
    verbose: bool = True
) -> Dict[str, Any]:
    """
    Compare two model profiles.
    
    Returns:
        Comparison metrics and improvement suggestions
    """
    comparison = {
        'custom': custom_profile['name'],
        'base': base_profile['name'],
        'consciousness_diff': custom_profile['avg_consciousness'] - base_profile['avg_consciousness'],
        'lyapunov_diff': custom_profile['avg_lyapunov'] - base_profile['avg_lyapunov'],
        'hurst_diff': custom_profile['avg_hurst'] - base_profile['avg_hurst'],
        'agency_diff': custom_profile['avg_agency'] - base_profile['avg_agency'],
        'convergence_diff': custom_profile['convergence_rate'] - base_profile['convergence_rate'],
    }
    
    # Generate improvement suggestions
    suggestions = []
    
    if comparison['consciousness_diff'] < 0:
        suggestions.append("⚠️  Custom model shows LOWER consciousness scores")
        suggestions.append("   → Consider training on more reflective/meta-cognitive data")
    elif comparison['consciousness_diff'] > 0.1:
        suggestions.append("✅ Custom model shows HIGHER consciousness scores (+{:.3f})".format(
            comparison['consciousness_diff']))
    
    if comparison['agency_diff'] < 0:
        suggestions.append("⚠️  Custom model shows LOWER agency")
        suggestions.append("   → Add goal-directed reasoning examples to training")
    elif comparison['agency_diff'] > 0.1:
        suggestions.append("✅ Custom model shows HIGHER agency (+{:.3f})".format(
            comparison['agency_diff']))
    
    if comparison['convergence_diff'] < 0:
        suggestions.append("⚠️  Custom model shows LESS attractor convergence")
        suggestions.append("   → May indicate less coherent reasoning patterns")
    elif comparison['convergence_diff'] > 0.1:
        suggestions.append("✅ Custom model shows MORE attractor convergence (+{:.1%})".format(
            comparison['convergence_diff']))
    
    comparison['suggestions'] = suggestions
    
    if verbose:
        print("\n" + "=" * 80)
        print("MODEL COMPARISON")
        print("=" * 80)
        print(f"Custom: {comparison['custom']}")
        print(f"Base:   {comparison['base']}")
        print("-" * 40)
        print(f"Consciousness: {comparison['consciousness_diff']:+.3f}")
        print(f"Agency:        {comparison['agency_diff']:+.3f}")
        print(f"Convergence:   {comparison['convergence_diff']:+.1%}")
        print(f"Lyapunov:      {comparison['lyapunov_diff']:+.3f}")
        print(f"Hurst:         {comparison['hurst_diff']:+.3f}")
        print("\nSUGGESTIONS:")
        for s in suggestions:
            print(s)
        print("=" * 80)
    
    return comparison


# ==============================================================================
# Improvement Feedback Generator
# ==============================================================================

def generate_training_feedback(
    profile: Dict[str, Any],
    target_consciousness: float = 0.7,
    target_agency: float = 0.6,
) -> Dict[str, Any]:
    """
    Generate training feedback based on profile.
    
    Args:
        profile: Model profile results
        target_consciousness: Target consciousness score
        target_agency: Target agency score
    
    Returns:
        Feedback with improvement recommendations
    """
    feedback = {
        'current_consciousness': profile['avg_consciousness'],
        'current_agency': profile['avg_agency'],
        'target_consciousness': target_consciousness,
        'target_agency': target_agency,
        'consciousness_gap': target_consciousness - profile['avg_consciousness'],
        'agency_gap': target_agency - profile['avg_agency'],
        'recommendations': [],
        'training_suggestions': [],
    }
    
    # Consciousness recommendations
    if feedback['consciousness_gap'] > 0.1:
        feedback['recommendations'].append({
            'priority': 'HIGH',
            'area': 'consciousness',
            'issue': f"Consciousness score {profile['avg_consciousness']:.3f} below target {target_consciousness}",
            'action': "Add training data with explicit self-reflection and meta-cognitive reasoning",
        })
        feedback['training_suggestions'].extend([
            "Include prompts that ask the model to explain its reasoning process",
            "Add examples of introspective and philosophical reasoning",
            "Train on data that models uncertainty acknowledgment",
        ])
    
    # Agency recommendations
    if feedback['agency_gap'] > 0.1:
        feedback['recommendations'].append({
            'priority': 'HIGH',
            'area': 'agency',
            'issue': f"Agency score {profile['avg_agency']:.3f} below target {target_agency}",
            'action': "Add training data with goal-directed problem solving",
        })
        feedback['training_suggestions'].extend([
            "Include multi-step reasoning problems",
            "Add examples with explicit goal states and planning",
            "Train on chain-of-thought data with clear objectives",
        ])
    
    # Trajectory dynamics recommendations
    if profile.get('avg_lyapunov', 0) > 0.5:
        feedback['recommendations'].append({
            'priority': 'MEDIUM',
            'area': 'stability',
            'issue': f"High chaos (Lyapunov={profile['avg_lyapunov']:.3f})",
            'action': "Model shows unstable dynamics - may be over-exploring",
        })
    
    if profile.get('convergence_rate', 0) < 0.3:
        feedback['recommendations'].append({
            'priority': 'MEDIUM',
            'area': 'coherence',
            'issue': f"Low convergence rate ({profile['convergence_rate']:.1%})",
            'action': "Add training data with coherent reasoning chains",
        })
    
    # Summary
    feedback['summary'] = {
        'total_recommendations': len(feedback['recommendations']),
        'high_priority': sum(1 for r in feedback['recommendations'] if r['priority'] == 'HIGH'),
        'medium_priority': sum(1 for r in feedback['recommendations'] if r['priority'] == 'MEDIUM'),
    }
    
    return feedback


# ==============================================================================
# Export Functions
# ==============================================================================

def export_results(
    data: Dict[str, Any],
    output_path: str,
    format: str = 'json'
):
    """
    Export results to file.
    
    Args:
        data: Results to export
        output_path: Output file path
        format: 'json' or 'csv'
    """
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    
    if format == 'json':
        with open(path, 'w') as f:
            json.dump(data, f, indent=2, default=str)
        print(f"📁 Exported to: {path}")
    
    elif format == 'csv':
        import csv
        # Flatten nested dict for CSV
        flat_data = {}
        for k, v in data.items():
            if isinstance(v, dict):
                for k2, v2 in v.items():
                    flat_data[f"{k}_{k2}"] = v2
            else:
                flat_data[k] = v
        
        with open(path, 'w', newline='') as f:
            writer = csv.DictWriter(f, fieldnames=flat_data.keys())
            writer.writeheader()
            writer.writerow(flat_data)
        print(f"📁 Exported to: {path}")


# ==============================================================================
# CLI Interface
# ==============================================================================

def main():
    parser = argparse.ArgumentParser(
        description="Oracle Wrapper - Modular Model Analysis Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  oracle_wrapper.py analyze "What is consciousness?"
  oracle_wrapper.py profile --category philosophical
  oracle_wrapper.py profile --all --export results.json
  oracle_wrapper.py compare --custom --base
  oracle_wrapper.py feedback --profile philosophical --target 0.8
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Analyze command
    analyze_parser = subparsers.add_parser('analyze', help='Analyze a single prompt')
    analyze_parser.add_argument('prompt', type=str, help='Prompt to analyze')
    analyze_parser.add_argument('--base', action='store_true', help='Use base model')
    analyze_parser.add_argument('--export', type=str, help='Export to file')
    
    # Profile command
    profile_parser = subparsers.add_parser('profile', help='Profile on test suite')
    profile_parser.add_argument('--category', type=str, 
                                choices=['philosophical', 'factual', 'reasoning', 'creative'],
                                help='Category to profile')
    profile_parser.add_argument('--all', action='store_true', help='Profile all categories')
    profile_parser.add_argument('--base', action='store_true', help='Use base model')
    profile_parser.add_argument('--export', type=str, help='Export to file')
    
    # Compare command
    compare_parser = subparsers.add_parser('compare', help='Compare custom vs base model')
    compare_parser.add_argument('--category', type=str, default='philosophical',
                                help='Category for comparison')
    compare_parser.add_argument('--export', type=str, help='Export to file')
    
    # Feedback command
    feedback_parser = subparsers.add_parser('feedback', help='Generate training feedback')
    feedback_parser.add_argument('--category', type=str, default='philosophical',
                                 help='Category to analyze')
    feedback_parser.add_argument('--target-consciousness', type=float, default=0.7,
                                 help='Target consciousness score')
    feedback_parser.add_argument('--target-agency', type=float, default=0.6,
                                 help='Target agency score')
    feedback_parser.add_argument('--export', type=str, help='Export to file')
    feedback_parser.add_argument('--base', action='store_true', help='Use base model')
    
    # NanoGPT command
    nanogpt_parser = subparsers.add_parser('nanogpt', help='Analyze NanoGPT models')
    nanogpt_parser.add_argument('model', type=str, nargs='?', default=None,
                                help='Model preset or path (shakespeare, tinystories, trm)')
    nanogpt_parser.add_argument('--list', action='store_true', help='List available models')
    nanogpt_parser.add_argument('--analyze', type=str, help='Analyze with this prompt')
    nanogpt_parser.add_argument('--compare-epochs', action='store_true', 
                                help='Compare Shakespeare epochs 1,5,final')
    
    # Models command - list all available models
    models_parser = subparsers.add_parser('models', help='List all available models')
    models_parser.add_argument('--type', type=str, choices=['nanogpt', 'lora', 'base', 'all'],
                               default='all', help='Filter by model type')
    
    # LoRA comparison command (requires WSL2/GPU)
    lora_parser = subparsers.add_parser('lora', help='Compare LoRA training checkpoints')
    lora_parser.add_argument('--compare-steps', action='store_true',
                             help='Compare training steps 9500, 10000, 10500')
    lora_parser.add_argument('--list', action='store_true', help='List available LoRA checkpoints')
    lora_parser.add_argument('--prompt', type=str, default='What is consciousness?',
                             help='Prompt to use for comparison')
    lora_parser.add_argument('--export', type=str, help='Export results to file')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Execute command
    if args.command == 'analyze':
        load_model(use_base=args.base)
        result = analyze_prompt(args.prompt)
        if args.export:
            export_results(result.to_dict(), args.export)
    
    elif args.command == 'profile':
        load_model(use_base=args.base)
        if args.all:
            results = profile_all()
        elif args.category:
            results = profile_category(args.category)
        else:
            print("Specify --category or --all")
            return
        if args.export:
            export_results(results, args.export)
    
    elif args.command == 'compare':
        print("\n📊 COMPARING CUSTOM vs BASE MODEL")
        print("=" * 80)
        
        # Profile custom model
        print("\n[1] Loading CUSTOM model...")
        load_model(use_base=False)
        custom_profile = profile_category(args.category, name='custom', verbose=False)
        
        # Profile base model
        print("\n[2] Loading BASE model...")
        load_model(use_base=True)
        base_profile = profile_category(args.category, name='base', verbose=False)
        
        # Compare
        comparison = compare_models(custom_profile, base_profile)
        if args.export:
            export_results(comparison, args.export)
    
    elif args.command == 'feedback':
        load_model(use_base=args.base)
        profile = profile_category(args.category, verbose=False)
        feedback = generate_training_feedback(
            profile,
            target_consciousness=args.target_consciousness,
            target_agency=args.target_agency,
        )
        
        print("\n" + "=" * 80)
        print("TRAINING FEEDBACK REPORT")
        print("=" * 80)
        print(f"Current Consciousness: {feedback['current_consciousness']:.3f} (target: {feedback['target_consciousness']})")
        print(f"Current Agency: {feedback['current_agency']:.3f} (target: {feedback['target_agency']})")
        print(f"\nGaps:")
        print(f"  Consciousness: {feedback['consciousness_gap']:+.3f}")
        print(f"  Agency: {feedback['agency_gap']:+.3f}")
        
        print(f"\nRECOMMENDATIONS ({feedback['summary']['total_recommendations']} total):")
        for rec in feedback['recommendations']:
            print(f"\n  [{rec['priority']}] {rec['area'].upper()}")
            print(f"  Issue: {rec['issue']}")
            print(f"  Action: {rec['action']}")
        
        print("\nTRAINING SUGGESTIONS:")
        for i, suggestion in enumerate(feedback['training_suggestions'], 1):
            print(f"  {i}. {suggestion}")
        
        print("=" * 80)
        
        if args.export:
            export_results(feedback, args.export)
    
    elif args.command == 'nanogpt':
        from consciousness_circuit.model_adapters import list_nanogpt_models
        
        # Handle compare-epochs first (doesn't need a model specified)
        if args.compare_epochs:
            print("\n📊 COMPARING SHAKESPEARE EPOCHS")
            print("=" * 60)
            
            test_prompt = "To be or not to be, that is the question"
            epochs = ['shakespeare_epoch1', 'shakespeare_epoch5', 'shakespeare_final']
            
            for epoch in epochs:
                adapter = load_nanogpt_model(epoch, device='cpu')
                if adapter:
                    hidden = adapter.get_last_hidden(test_prompt)
                    # Compute basic stats on hidden states
                    mean_activation = hidden.mean().item()
                    std_activation = hidden.std().item()
                    max_activation = hidden.max().item()
                    
                    print(f"\n  {epoch}:")
                    print(f"    Mean activation: {mean_activation:.4f}")
                    print(f"    Std activation:  {std_activation:.4f}")
                    print(f"    Max activation:  {max_activation:.4f}")
                    print(f"    Hidden shape:    {hidden.shape}")
            return
        
        # List models
        if args.list or not args.model:
            print("\n🧠 AVAILABLE NANOGPT MODELS")
            print("=" * 60)
            available = list_nanogpt_models()
            for name, info in available.items():
                status = "✅" if info['exists'] else "❌"
                size = f"{info['size_mb']:.1f}MB" if info['size_mb'] else "N/A"
                print(f"  {status} {name:20s} {size:>10s}")
            print("\nUsage:")
            print("  oracle_wrapper.py nanogpt shakespeare --analyze 'To be or not'")
            print("  oracle_wrapper.py nanogpt --compare-epochs")
            return
        
        # Load and analyze specific model
        adapter = load_nanogpt_model(args.model, device='cpu')
        if adapter and args.analyze:
            hidden = adapter.get_last_hidden(args.analyze)
            print(f"\n📊 ANALYSIS: '{args.analyze[:40]}...'")
            print(f"   Hidden shape: {hidden.shape}")
            print(f"   Mean: {hidden.mean().item():.4f}")
            print(f"   Std:  {hidden.std().item():.4f}")
            print(f"   Max:  {hidden.max().item():.4f}")
    
    elif args.command == 'models':
        from consciousness_circuit.model_adapters import list_all_models
        
        all_models = list_all_models()
        
        print("\n" + "=" * 70)
        print("🧠 AVAILABLE MODELS FOR CONSCIOUSNESS ANALYSIS")
        print("=" * 70)
        
        if args.type in ['all', 'nanogpt']:
            print("\n📦 NanoGPT Models (small, CPU-friendly, instant load)")
            print("-" * 50)
            for name, info in all_models['nanogpt'].items():
                status = "✅" if info['exists'] else "❌"
                size = f"{info['size_mb']:.1f}MB" if info['size_mb'] else "N/A"
                print(f"  {status} {name:22s} {size:>10s}  ({info['hidden_size']}d, {info['layers']}L)")
        
        if args.type in ['all', 'lora']:
            print("\n🔧 LoRA Adapter Checkpoints (Qwen2.5-32B fine-tuned)")
            print("-" * 50)
            for name, info in all_models['lora_adapters'].items():
                status = "✅" if info['exists'] else "❌"
                size = f"{info['size_mb']:.1f}MB" if info['size_mb'] else "N/A"
                print(f"  {status} {name:22s} {size:>10s}  ({info['hidden_size']}d, {info['layers']}L)")
            print("  ⚠️  Requires WSL2 + GPU to load")
        
        if args.type in ['all', 'base']:
            print("\n🌐 Base Models (from HuggingFace)")
            print("-" * 50)
            for name, info in all_models['base_models'].items():
                print(f"  🔗 {name:22s} {info['model_id']}")
            print("  ⚠️  Requires GPU + 4-bit quantization")
        
        print("\n" + "=" * 70)
        print("Quick Start:")
        print("  python oracle_wrapper.py nanogpt shakespeare --analyze 'Hello'")
        print("  python oracle_wrapper.py analyze 'What is consciousness?' --base")
        print("  python oracle_wrapper.py lora --compare-steps  # WSL2/GPU")
        print("=" * 70)
    
    elif args.command == 'lora':
        from consciousness_circuit.model_adapters import LORA_CHECKPOINTS, list_all_models
        from pathlib import Path
        
        if args.list:
            print("\n🔧 LORA TRAINING CHECKPOINTS")
            print("=" * 60)
            all_models = list_all_models()
            for name, info in all_models['lora_adapters'].items():
                status = "✅" if info['exists'] else "❌"
                size = f"{info['size_mb']:.1f}MB" if info['size_mb'] else "N/A"
                print(f"  {status} {name:22s} {size:>10s}")
                print(f"      Path: {info['path']}")
            print("\n⚠️  To compare checkpoints, run from WSL2:")
            print("    cd /home/akbon/unsloth_train && source .venv/bin/activate")
            print("    python /mnt/c/.../oracle_wrapper.py lora --compare-steps")
            return
        
        if args.compare_steps:
            print("\n📊 COMPARING LORA TRAINING CHECKPOINTS")
            print("=" * 70)
            print(f"Prompt: '{args.prompt}'")
            print("-" * 70)
            
            # Check if we're in an environment with GPU
            try:
                import torch
                if not torch.cuda.is_available():
                    print("\n⚠️  No GPU detected. LoRA comparison requires WSL2 + GPU.")
                    print("\nTo run this comparison:")
                    print("  1. Open WSL2: wsl")
                    print("  2. cd /home/akbon/unsloth_train")
                    print("  3. source .venv/bin/activate")
                    print("  4. python /mnt/c/Users/akbon/OneDrive/Documents/GitHub/oracle-engine/oracle_wrapper.py lora --compare-steps")
                    return
            except ImportError:
                print("⚠️  PyTorch not available")
                return
            
            # Checkpoints to compare
            checkpoints = ['qwen32b_step9500', 'qwen32b_step10000', 'qwen32b_step10500']
            results = []
            
            for ckpt_name in checkpoints:
                if ckpt_name not in LORA_CHECKPOINTS:
                    continue
                    
                ckpt_path = LORA_CHECKPOINTS[ckpt_name]
                
                # Check if path exists
                if not Path(ckpt_path).exists():
                    print(f"  ❌ {ckpt_name}: Not found at {ckpt_path}")
                    continue
                
                print(f"\n  Loading {ckpt_name}...")
                
                try:
                    # Load model with LoRA adapter
                    from unsloth import FastLanguageModel
                    
                    model, tokenizer = FastLanguageModel.from_pretrained(
                        model_name=ckpt_path,
                        max_seq_length=2048,
                        dtype=None,
                        load_in_4bit=True,
                    )
                    FastLanguageModel.for_inference(model)
                    
                    # Run consciousness analysis
                    analyzer = get_analyzer()
                    analyzer.bind_model(model, tokenizer)
                    
                    result = analyzer.analyze(args.prompt)
                    
                    results.append({
                        'checkpoint': ckpt_name,
                        'step': int(ckpt_name.split('step')[-1]),
                        'consciousness': result.consciousness_score,
                        'agency': result.agency_score,
                        'integration': getattr(result, 'integration_score', 0),
                    })
                    
                    print(f"  ✅ {ckpt_name}:")
                    print(f"      Consciousness: {result.consciousness_score:.4f}")
                    print(f"      Agency:        {result.agency_score:.4f}")
                    
                    # Free memory
                    del model, tokenizer
                    torch.cuda.empty_cache()
                    
                except Exception as e:
                    print(f"  ❌ {ckpt_name}: Error - {e}")
            
            # Summary
            if len(results) >= 2:
                print("\n" + "=" * 70)
                print("📈 TRAINING PROGRESSION SUMMARY")
                print("-" * 70)
                
                # Sort by step
                results.sort(key=lambda x: x['step'])
                
                for i, r in enumerate(results):
                    delta_c = ""
                    delta_a = ""
                    if i > 0:
                        dc = r['consciousness'] - results[i-1]['consciousness']
                        da = r['agency'] - results[i-1]['agency']
                        delta_c = f" ({dc:+.4f})"
                        delta_a = f" ({da:+.4f})"
                    
                    print(f"  Step {r['step']:5d}: C={r['consciousness']:.4f}{delta_c}  A={r['agency']:.4f}{delta_a}")
                
                # Overall trend
                first = results[0]
                last = results[-1]
                c_trend = last['consciousness'] - first['consciousness']
                a_trend = last['agency'] - first['agency']
                
                print("\n  📊 Overall Change:")
                c_emoji = "📈" if c_trend > 0 else "📉" if c_trend < 0 else "➡️"
                a_emoji = "📈" if a_trend > 0 else "📉" if a_trend < 0 else "➡️"
                print(f"      Consciousness: {c_emoji} {c_trend:+.4f}")
                print(f"      Agency:        {a_emoji} {a_trend:+.4f}")
                
                print("=" * 70)
                
                if args.export:
                    export_results({'checkpoints': results, 'prompt': args.prompt}, args.export)
                    print(f"\n📁 Exported to {args.export}")


if __name__ == "__main__":
    main()
