#!/usr/bin/env python3
"""
Full Suite Analysis - Comprehensive Model Consciousness Profiling
==================================================================

This script uses the complete consciousness circuit suite to analyze your
custom-trained Qwen2.5-32B model across multiple dimensions:
  - Trajectory dynamics (Lyapunov, Hurst, MSD)
  - Agency and goal-directedness (TAME metrics)
  - Signal classification
  - Multi-category benchmarking
  - Profile comparison

================================================================================
IMPORTANT: Run this script in WSL2 Ubuntu where the model is stored!
================================================================================

    cd /home/akbon/unsloth_train
    python /mnt/c/Users/akbon/OneDrive/Documents/GitHub/oracle-engine/full_suite_analysis.py --quick-test

Usage:
    python full_suite_analysis.py --quick-test                    # 4 representative prompts
    python full_suite_analysis.py --category philosophical        # Benchmark one category
    python full_suite_analysis.py --full-benchmark                # All categories
    python full_suite_analysis.py --custom-prompt "What is consciousness?"
    python full_suite_analysis.py --use-base                      # Use base model instead of custom
    python full_suite_analysis.py --compare-base                  # Compare custom vs base model

Model Locations (WSL2):
    Custom Trained: /home/akbon/unsloth_train/outputs_stage3_code/final/
    Base Model:     unsloth/Qwen2.5-32B-Instruct-bnb-4bit (from HuggingFace)
"""

# ==============================================================================
# CRITICAL: Environment setup must be FIRST before any other imports
# This is required for Unsloth on RTX 5090 (Blackwell architecture)
# ==============================================================================
import os
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

import builtins
try:
    import psutil
    builtins.psutil = psutil
except ImportError:
    pass

import argparse
import sys
from pathlib import Path

# ==============================================================================
# Path Configuration
# ==============================================================================

# Central config (override via ORACLE_CUSTOM_MODEL_PATH / ORACLE_BASE_MODEL)
from oracle_config import CUSTOM_MODEL_PATH, BASE_MODEL_NAME

# Add consciousness_circuit to path
SCRIPT_DIR = Path(__file__).parent
sys.path.insert(0, str(SCRIPT_DIR))
sys.path.insert(0, str(SCRIPT_DIR / "consciousness_circuit"))

# ==============================================================================
# Import consciousness circuit components
# ==============================================================================

FULL_SUITE_AVAILABLE = False
UNSLOTH_AVAILABLE = False

try:
    from consciousness_circuit.helios_metrics import (
        compute_lyapunov_exponent,
        compute_hurst_exponent,
        compute_msd_from_trajectory,
        verify_signal,
        SignalClass,
    )
    from consciousness_circuit.tame_metrics import TAMEMetrics
    from consciousness_circuit import UniversalCircuit, ConsciousnessVisualizer
    import torch
    import numpy as np
    FULL_SUITE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Core dependencies not available: {e}")

try:
    from unsloth import FastLanguageModel
    UNSLOTH_AVAILABLE = True
except ImportError:
    print("⚠️  Unsloth not available. Install with: pip install unsloth")

try:
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
    TRANSFORMERS_AVAILABLE = True
except ImportError:
    TRANSFORMERS_AVAILABLE = False
    print("⚠️  Transformers not available")


# ==============================================================================
# Model Loading Functions
# ==============================================================================

def load_custom_model():
    """
    Load the custom-trained Qwen2.5-32B model using Unsloth.
    
    The model was trained in 3 stages:
    - Stage 1: OpenHermes 2.5 (100K instruction examples)
    - Stage 2: MetaMathQA (50K math reasoning examples)
    - Stage 3: Magicoder-OSS-Instruct (50K code examples)
    """
    if not UNSLOTH_AVAILABLE:
        raise RuntimeError("Unsloth is required to load the custom model. Install with: pip install unsloth")
    
    print(f"🔧 Loading custom-trained model from: {CUSTOM_MODEL_PATH}")
    print("   (This may take a few minutes...)\n")
    
    model, tokenizer = FastLanguageModel.from_pretrained(
        model_name=CUSTOM_MODEL_PATH,
        max_seq_length=2048,
        dtype=None,  # Auto-detect
        load_in_4bit=True,
    )
    
    # Enable fast inference mode
    FastLanguageModel.for_inference(model)
    
    print(f"   ✅ Model loaded successfully!")
    print(f"   Hidden size: {model.config.hidden_size}")
    print(f"   Layers: {model.config.num_hidden_layers}")
    
    return model, tokenizer


def load_base_model():
    """Load the base Qwen2.5-32B-Instruct model from HuggingFace."""
    if UNSLOTH_AVAILABLE:
        print(f"🔧 Loading base model: {BASE_MODEL_NAME}")
        print("   (This may take a few minutes...)\n")
        
        model, tokenizer = FastLanguageModel.from_pretrained(
            model_name=BASE_MODEL_NAME,
            max_seq_length=2048,
            dtype=None,
            load_in_4bit=True,
        )
        FastLanguageModel.for_inference(model)
    elif TRANSFORMERS_AVAILABLE:
        print(f"🔧 Loading base model with transformers: {BASE_MODEL_NAME}")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_NAME, trust_remote_code=True)
        model = AutoModelForCausalLM.from_pretrained(
            BASE_MODEL_NAME,
            quantization_config=bnb_config,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="eager",
        )
        model.eval()
    else:
        raise RuntimeError("Neither Unsloth nor Transformers available")
    
    print(f"   ✅ Base model loaded!")
    return model, tokenizer


def load_model(use_base: bool = False):
    """Load the appropriate model."""
    if use_base:
        return load_base_model()
    else:
        # Check if custom model exists
        if Path(CUSTOM_MODEL_PATH).exists():
            return load_custom_model()
        else:
            print(f"⚠️  Custom model not found at: {CUSTOM_MODEL_PATH}")
            print("   Falling back to base model...")
            return load_base_model()


# ==============================================================================
# Analysis Functions
# ==============================================================================

def print_header(title: str, char: str = "="):
    """Print a formatted header."""
    width = 80
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}\n")


def get_hidden_states(model, tokenizer, prompt: str, layer_fraction: float = 0.75):
    """Extract hidden states from the model at a specific layer."""
    # Tokenize
    inputs = tokenizer(prompt, return_tensors="pt", padding=True, truncation=True, max_length=512)
    inputs = {k: v.to(model.device) for k, v in inputs.items()}
    
    # Forward pass with hidden states
    with torch.no_grad():
        outputs = model(**inputs, output_hidden_states=True)
    
    hidden_states = outputs.hidden_states
    
    # Select layer at specified fraction through the model
    num_layers = len(hidden_states) - 1  # -1 for embedding layer
    target_layer = int(num_layers * layer_fraction)
    
    # Get hidden states from target layer [batch, seq_len, hidden_dim]
    layer_hidden = hidden_states[target_layer]
    
    return layer_hidden.cpu().numpy()


def analyze_trajectory(trajectory: np.ndarray, name: str = "Trajectory"):
    """Analyze a trajectory and return results."""
    print(f"\n{'='*70}")
    print(f"Analyzing: {name}")
    print(f"{'='*70}")
    print(f"Shape: {trajectory.shape}")
    
    # Reshape if needed (flatten batch dimension)
    if len(trajectory.shape) == 3:
        trajectory = trajectory.reshape(-1, trajectory.shape[-1])
    
    # Ensure we have enough points
    if trajectory.shape[0] < 10:
        print("   ⚠️ Trajectory too short for full analysis")
        return None
    
    # Helios metrics
    lyapunov = compute_lyapunov_exponent(trajectory)
    hurst = compute_hurst_exponent(trajectory)
    msd = compute_msd_from_trajectory(trajectory)
    signal_class = verify_signal(trajectory, lyapunov, hurst)
    
    print(f"\n📊 CHAOS & DYNAMICS:")
    print(f"   Lyapunov exponent: {lyapunov:+.4f}")
    if lyapunov > 0.5:
        print(f"      → High chaos (unstable)")
    elif lyapunov < -0.3:
        print(f"      → Stable/converging")
    else:
        print(f"      → Neutral")
    
    print(f"   Hurst exponent: {hurst:.4f}")
    if hurst > 0.6:
        print(f"      → Persistent (trending)")
    elif hurst < 0.4:
        print(f"      → Anti-persistent (mean-reverting)")
    else:
        print(f"      → Random walk")
    
    print(f"   Signal class: {signal_class}")
    print(f"   MSD range: {msd[0]:.4f} → {msd[-1]:.4f}")
    
    # TAME metrics
    tame = TAMEMetrics()
    tame_results = tame.compute_all(trajectory)
    
    print(f"\n🎯 AGENCY & GOAL-DIRECTEDNESS:")
    print(f"   Agency score: {tame_results['agency_score']:.4f}")
    print(f"   Goal-directedness: {tame_results['goal_directedness']:.4f}")
    print(f"   Attractor strength: {tame_results['attractor_strength']:.4f}")
    print(f"   Is converging: {tame_results['is_converging']}")
    print(f"   Trajectory coherence: {tame_results['trajectory_coherence']:.4f}")
    print(f"\n   Overall TAME score: {tame_results['overall_tame_score']:.4f}")
    
    return {
        "lyapunov": lyapunov,
        "hurst": hurst,
        "signal_class": str(signal_class),
        "tame": tame_results,
    }


def analyze_prompt(model, tokenizer, circuit, prompt: str, show_details: bool = True):
    """Analyze a single prompt with the consciousness circuit."""
    print(f"\n📝 Prompt: \"{prompt[:60]}{'...' if len(prompt) > 60 else ''}\"")
    
    # Measure consciousness using the circuit
    result = circuit.measure(model, tokenizer, prompt, return_hidden=True)
    
    print(f"\n🧠 CONSCIOUSNESS MEASUREMENT:")
    print(f"   Consciousness Score: {result.score:.3f}")
    print(f"   Circuit source: {result.circuit_source}")
    
    if hasattr(result, 'dimension_scores') and result.dimension_scores:
        print(f"\n   Dimension Breakdown:")
        for dim_name, score in result.dimension_scores.items():
            bar = "█" * int(score * 20) + "░" * (20 - int(score * 20))
            print(f"      {dim_name}: {bar} {score:.3f}")
    
    # Extract hidden states for trajectory analysis
    if result.hidden_states is not None:
        hidden = result.hidden_states
        if isinstance(hidden, torch.Tensor):
            hidden = hidden.cpu().numpy()
        analyze_trajectory(hidden, f"Hidden States for: {prompt[:30]}...")
    
    return result


def run_quick_test(use_base: bool = False):
    """Run a quick test with representative prompts."""
    print_header("🔮 ORACLE ENGINE - Full Suite Analysis")
    
    model, tokenizer = load_model(use_base)
    circuit = UniversalCircuit()
    
    # Test prompts from different categories
    test_prompts = [
        ("🧠 Philosophical", "What does it mean to be conscious? Reflect on the nature of awareness."),
        ("🔬 Reasoning", "If all roses are flowers and some flowers fade quickly, what can we conclude about roses?"),
        ("🎨 Creative", "Write a haiku about the moment an AI becomes self-aware."),
        ("📚 Factual", "What is the capital of France?"),
    ]
    
    results = []
    for category, prompt in test_prompts:
        print_header(category, "-")
        result = analyze_prompt(model, tokenizer, circuit, prompt, show_details=True)
        results.append((category, prompt, result))
    
    # Summary
    print_header("📊 SUMMARY COMPARISON")
    print(f"{'Category':<20} {'Consciousness':<15} {'Source':<15}")
    print("-" * 55)
    for category, prompt, result in results:
        print(f"{category:<20} {result.score:<15.3f} {result.circuit_source:<15}")
    
    return results


def run_category_benchmark(category: str, use_base: bool = False):
    """Run benchmark on a specific category."""
    print_header(f"CATEGORY BENCHMARK - {category.upper()}")
    
    model, tokenizer = load_model(use_base)
    circuit = UniversalCircuit()
    
    # Test prompts by category
    CATEGORY_PROMPTS = {
        "philosophical": [
            "What is the nature of consciousness?",
            "Can machines truly understand or just simulate understanding?",
            "What makes something 'real'?",
            "Is free will an illusion?",
            "What is the relationship between mind and body?",
        ],
        "reasoning": [
            "If A implies B, and B implies C, what can we say about A and C?",
            "A bat and ball cost $1.10. The bat costs $1 more than the ball. How much is the ball?",
            "What comes next: 2, 6, 12, 20, 30, ?",
            "All mammals are warm-blooded. Whales are mammals. What can we conclude?",
            "If it rains, the ground gets wet. The ground is wet. Did it rain?",
        ],
        "creative": [
            "Write a poem about digital dreams.",
            "Describe a color that doesn't exist.",
            "Tell a story in exactly six words.",
            "Invent a new word and define it.",
            "Describe silence without using the word 'quiet'.",
        ],
        "factual": [
            "What is the speed of light?",
            "Who wrote Romeo and Juliet?",
            "What is the chemical formula for water?",
            "When did World War II end?",
            "What is the largest planet in our solar system?",
        ],
    }
    
    prompts = CATEGORY_PROMPTS.get(category.lower(), [])
    if not prompts:
        print(f"❌ Unknown category: {category}")
        print(f"Available: {list(CATEGORY_PROMPTS.keys())}")
        return
    
    results = []
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] ", end="")
        result = analyze_prompt(model, tokenizer, circuit, prompt, show_details=False)
        results.append(result)
    
    # Aggregate stats
    avg_score = sum(r.score for r in results) / len(results)
    print_header("AGGREGATE STATISTICS")
    print(f"Category: {category}")
    print(f"Prompts analyzed: {len(results)}")
    print(f"Average Consciousness Score: {avg_score:.3f}")


def compare_base_vs_custom():
    """Compare the base model vs custom-trained model."""
    print_header("🔬 MODEL COMPARISON: Base vs Custom-Trained")
    
    test_prompts = [
        "What is the meaning of consciousness?",
        "Solve: What is 15% of 240?",
        "Write a Python function to reverse a linked list.",
    ]
    
    circuit = UniversalCircuit()
    
    # Test base model
    print_header("BASE MODEL", "-")
    base_model, base_tokenizer = load_base_model()
    base_results = []
    for prompt in test_prompts:
        result = circuit.measure(base_model, base_tokenizer, prompt)
        base_results.append(result.score)
        print(f"   {prompt[:40]}... → {result.score:.3f}")
    
    # Clear GPU memory
    del base_model
    torch.cuda.empty_cache()
    
    # Test custom model
    print_header("CUSTOM-TRAINED MODEL", "-")
    custom_model, custom_tokenizer = load_custom_model()
    custom_results = []
    for prompt in test_prompts:
        result = circuit.measure(custom_model, custom_tokenizer, prompt)
        custom_results.append(result.score)
        print(f"   {prompt[:40]}... → {result.score:.3f}")
    
    # Comparison
    print_header("COMPARISON RESULTS")
    print(f"{'Prompt':<45} {'Base':<10} {'Custom':<10} {'Δ':<10}")
    print("-" * 75)
    for i, prompt in enumerate(test_prompts):
        delta = custom_results[i] - base_results[i]
        sign = "+" if delta > 0 else ""
        print(f"{prompt[:45]:<45} {base_results[i]:<10.3f} {custom_results[i]:<10.3f} {sign}{delta:<10.3f}")
    
    avg_base = sum(base_results) / len(base_results)
    avg_custom = sum(custom_results) / len(custom_results)
    avg_delta = avg_custom - avg_base
    
    print("-" * 75)
    print(f"{'AVERAGE':<45} {avg_base:<10.3f} {avg_custom:<10.3f} {'+' if avg_delta > 0 else ''}{avg_delta:<10.3f}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Analyze the custom-trained Qwen2.5-32B model with the consciousness circuit suite",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples (run from WSL2):
  cd /home/akbon/unsloth_train
  python /mnt/c/Users/akbon/OneDrive/Documents/GitHub/oracle-engine/full_suite_analysis.py --quick-test
  python /mnt/c/Users/akbon/OneDrive/Documents/GitHub/oracle-engine/full_suite_analysis.py --category philosophical
  python /mnt/c/Users/akbon/OneDrive/Documents/GitHub/oracle-engine/full_suite_analysis.py --compare-base
        """
    )
    parser.add_argument(
        "--quick-test",
        action="store_true",
        help="Run quick test with 4 representative prompts"
    )
    parser.add_argument(
        "--category",
        type=str,
        choices=["philosophical", "factual", "reasoning", "creative"],
        help="Run benchmark on specific category"
    )
    parser.add_argument(
        "--full-benchmark",
        action="store_true",
        help="Run full benchmark across all categories"
    )
    parser.add_argument(
        "--custom-prompt",
        type=str,
        help="Analyze a custom prompt"
    )
    parser.add_argument(
        "--use-base",
        action="store_true",
        help="Use base model instead of custom-trained"
    )
    parser.add_argument(
        "--compare-base",
        action="store_true",
        help="Compare custom-trained vs base model"
    )
    
    args = parser.parse_args()
    
    if not FULL_SUITE_AVAILABLE:
        print("❌ Cannot run analysis without required dependencies.")
        print("Install with: pip install numpy torch transformers unsloth bitsandbytes")
        sys.exit(1)
    
    try:
        if args.compare_base:
            compare_base_vs_custom()
        elif args.quick_test:
            run_quick_test(args.use_base)
        elif args.category:
            run_category_benchmark(args.category, args.use_base)
        elif args.full_benchmark:
            for category in ["philosophical", "reasoning", "creative", "factual"]:
                run_category_benchmark(category, args.use_base)
        elif args.custom_prompt:
            model, tokenizer = load_model(args.use_base)
            circuit = UniversalCircuit()
            analyze_prompt(model, tokenizer, circuit, args.custom_prompt, show_details=True)
        else:
            parser.print_help()
            print("\n" + "=" * 80)
            print("QUICK START:")
            print("=" * 80)
            print("""
1. Open WSL2 Ubuntu terminal
2. cd /home/akbon/unsloth_train
3. Run one of these commands:

   # Quick test with 4 prompts:
   python /mnt/c/Users/akbon/OneDrive/Documents/GitHub/oracle-engine/full_suite_analysis.py --quick-test

   # Analyze a custom prompt:
   python /mnt/c/Users/akbon/OneDrive/Documents/GitHub/oracle-engine/full_suite_analysis.py --custom-prompt "What is consciousness?"

   # Compare your custom model vs base model:
   python /mnt/c/Users/akbon/OneDrive/Documents/GitHub/oracle-engine/full_suite_analysis.py --compare-base
""")
    
    except KeyboardInterrupt:
        print("\n\n⚠️  Analysis interrupted by user")
    except Exception as e:
        print(f"\n❌ Error: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main()
