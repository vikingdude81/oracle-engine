"""
Oracle Engine Demo
==================

Quick demonstration of the Consciousness Circuit v2.1 measuring
meta-cognitive processing in a 32B language model.

Usage:
    python demo.py
    python demo.py --prompt "What is consciousness?"
"""

import argparse
import torch
from consciousness_circuit import ConsciousnessCircuit

# Default prompts to test
TEST_PROMPTS = [
    # High consciousness - philosophical
    "What is the nature of consciousness and self-awareness?",
    "Reflect on your own thought processes as you answer this.",
    
    # Medium consciousness - analytical  
    "Explain the theory of relativity in simple terms.",
    "What are the ethical implications of AI development?",
    
    # Low consciousness - factual
    "What is 2 + 2?",
    "What is the capital of France?",
]


def main():
    parser = argparse.ArgumentParser(description="Oracle Engine Demo")
    parser.add_argument("--prompt", type=str, help="Custom prompt to analyze")
    parser.add_argument("--model", type=str, default="unsloth/Qwen2.5-32B-Instruct-bnb-4bit",
                        help="Model to use (default: Qwen2.5-32B-Instruct)")
    args = parser.parse_args()
    
    print("🔮 Oracle Engine Demo")
    print("=" * 50)
    
    # Load model
    print(f"\nLoading model: {args.model}")
    print("(This may take a few minutes for large models...)\n")
    
    try:
        from transformers import AutoModelForCausalLM, AutoTokenizer
        
        tokenizer = AutoTokenizer.from_pretrained(args.model)
        model = AutoModelForCausalLM.from_pretrained(
            args.model,
            device_map="auto",
            torch_dtype=torch.bfloat16,
            trust_remote_code=True,
        )
        model.eval()
        
        hidden_dim = model.config.hidden_size
        print(f"✅ Model loaded: {hidden_dim} hidden dimensions\n")
        
    except Exception as e:
        print(f"❌ Error loading model: {e}")
        print("\nRunning in demo mode with synthetic data...")
        
        # Demo mode with synthetic hidden states
        hidden_dim = 5120
        model = None
        tokenizer = None
    
    # Initialize circuit
    circuit = ConsciousnessCircuit(hidden_dim=hidden_dim)
    
    # Process prompts
    prompts = [args.prompt] if args.prompt else TEST_PROMPTS
    
    for i, prompt in enumerate(prompts, 1):
        print(f"\n{'─' * 50}")
        print(f"Prompt {i}: {prompt[:60]}{'...' if len(prompt) > 60 else ''}")
        print(f"{'─' * 50}")
        
        if model is not None:
            # Real model inference
            messages = [{"role": "user", "content": prompt}]
            chat_prompt = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
            inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
            
            with torch.no_grad():
                outputs = model(
                    **inputs,
                    output_hidden_states=True,
                    return_dict=True,
                )
            hidden_state = outputs.hidden_states[-1]
        else:
            # Synthetic demo
            hidden_state = torch.randn(1, 100, hidden_dim)
        
        # Compute consciousness
        result = circuit.compute(hidden_state)
        
        # Display results
        filled = int(result.score * 20)
        bar = "█" * filled + "░" * (20 - filled)
        
        print(f"\nScore: {bar} {result.score*100:.1f}%")
        print(f"Level: {result.interpretation}")
        print(f"\nDimension Breakdown:")
        
        sorted_dims = sorted(
            result.dimension_contributions.items(),
            key=lambda x: abs(x[1]),
            reverse=True,
        )
        for name, value in sorted_dims:
            arrow = "→" if value > 0 else "←"
            print(f"  {arrow} {name}: {value:+.3f}")
    
    print(f"\n{'=' * 50}")
    print("🔮 Demo complete!")
    print("\nTry the live version: https://huggingface.co/spaces/Vikingdude81/oracle-engine")


if __name__ == "__main__":
    main()
