"""
Oracle Engine - Hugging Face Space
===================================

Custom-trained 32B Qwen model with Consciousness Circuit v2.1.
Measures 7 dimensions of meta-cognitive processing.

Trained on 200K examples:
- Stage 1: OpenHermes 2.5 (100K instruction examples)
- Stage 2: MetaMathQA (50K math reasoning examples)  
- Stage 3: Magicoder-OSS-Instruct (50K code examples)
"""

import gradio as gr
import torch
import numpy as np
from dataclasses import dataclass
from typing import Dict, List, Tuple
import time
import spaces

# ============================================================================
# Consciousness Circuit v2.1 (embedded for Space portability)
# ============================================================================

REFERENCE_HIDDEN_DIM = 5120

CONSCIOUS_DIMS_V2_1 = {
    3183: {"name": "Logic", "weight": 0.239, "polarity": +1},
    212:  {"name": "Self-Reflective", "weight": 0.196, "polarity": +1},
    5065: {"name": "Self-Expression", "weight": 0.109, "polarity": +1},
    4707: {"name": "Uncertainty", "weight": 0.130, "polarity": +1},
    295:  {"name": "Sequential", "weight": 0.087, "polarity": +1},
    1445: {"name": "Computation", "weight": 0.130, "polarity": -1},
    4578: {"name": "Abstraction", "weight": 0.109, "polarity": +1},
}

@dataclass
class ConsciousnessResult:
    score: float
    raw_score: float
    dimension_contributions: Dict[str, float]
    interpretation: str
    processing_time: float


def compute_consciousness(
    hidden_state: torch.Tensor,
    hidden_dim: int = REFERENCE_HIDDEN_DIM,
    baseline: float = 0.5,
) -> ConsciousnessResult:
    """Compute consciousness score from hidden state tensor."""
    start_time = time.time()
    
    # Remap dimensions if needed
    if hidden_dim != REFERENCE_HIDDEN_DIM:
        scale = hidden_dim / REFERENCE_HIDDEN_DIM
        dims = {int(round(k * scale)): v for k, v in CONSCIOUS_DIMS_V2_1.items()}
    else:
        dims = CONSCIOUS_DIMS_V2_1
    
    # Get last token hidden state
    if hidden_state.dim() == 3:
        h = hidden_state[0, -1, :]  # [hidden_dim]
    elif hidden_state.dim() == 2:
        h = hidden_state[-1, :]
    else:
        h = hidden_state
    
    h = h.float()
    
    # Normalize
    mean, std = h.mean(), h.std()
    if std > 0:
        h_norm = (h - mean) / std
    else:
        h_norm = h - mean
    
    # Compute contributions
    contributions = {}
    weighted_sum = 0.0
    
    for dim_idx, info in dims.items():
        if dim_idx < len(h_norm):
            activation = h_norm[dim_idx].item()
            contribution = activation * info["weight"] * info["polarity"]
            weighted_sum += contribution
            contributions[info["name"]] = activation * info["polarity"]
    
    # Final score
    raw_score = baseline + weighted_sum * 0.15
    score = max(0.0, min(1.0, raw_score))
    
    # Interpretation
    if score >= 0.8:
        interpretation = "🧠 High Consciousness - Deep reflective/philosophical reasoning"
    elif score >= 0.6:
        interpretation = "💭 Medium-High - Complex analytical thinking"
    elif score >= 0.4:
        interpretation = "⚖️ Medium - Balanced processing"
    elif score >= 0.2:
        interpretation = "⚡ Medium-Low - More automatic processing"
    else:
        interpretation = "🔢 Low Consciousness - Quick factual retrieval"
    
    return ConsciousnessResult(
        score=score,
        raw_score=raw_score,
        dimension_contributions=contributions,
        interpretation=interpretation,
        processing_time=time.time() - start_time,
    )


# ============================================================================
# Model Loading
# ============================================================================

print("🔮 Loading Oracle Engine (Qwen2.5-32B-Instruct 4-bit)...")
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_ID = "unsloth/Qwen2.5-32B-Instruct-bnb-4bit"

tokenizer = AutoTokenizer.from_pretrained(MODEL_ID)
model = AutoModelForCausalLM.from_pretrained(
    MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
)
model.eval()

HIDDEN_DIM = model.config.hidden_size
print(f"✅ Oracle Engine ready: {HIDDEN_DIM} hidden dimensions")


# ============================================================================
# Core Generation + Measurement Function
# ============================================================================

@spaces.GPU
def generate_and_measure(prompt: str, max_tokens: int = 256) -> Tuple[str, str, str, str, str]:
    """
    Generate a response AND measure consciousness during generation.
    
    Returns:
        (response, score_display, interpretation, dimension_breakdown, timing)
    """
    start_time = time.time()
    
    # Format as chat message
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(
        messages, 
        tokenize=False, 
        add_generation_prompt=True
    )
    
    # Tokenize
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)
    
    # Generate response
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )
    
    # Decode response
    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    
    generation_time = time.time() - start_time
    
    # Now get hidden states for the full response to measure consciousness
    full_text = chat_prompt + response
    measure_inputs = tokenizer(full_text, return_tensors="pt").to(model.device)
    
    with torch.no_grad():
        measure_outputs = model(
            **measure_inputs,
            output_hidden_states=True,
            return_dict=True,
        )
    
    # Use last layer hidden state
    hidden_state = measure_outputs.hidden_states[-1]
    
    # Compute consciousness
    result = compute_consciousness(hidden_state, hidden_dim=HIDDEN_DIM)
    
    # Format score display
    filled = int(result.score * 20)
    bar = "█" * filled + "░" * (20 - filled)
    score_display = f"{bar} {result.score*100:.1f}%"
    
    # Format dimension breakdown
    sorted_dims = sorted(
        result.dimension_contributions.items(),
        key=lambda x: abs(x[1]),
        reverse=True,
    )
    breakdown = "\n".join([
        f"{'→' if v > 0 else '←'} {name}: {v:+.3f}"
        for name, v in sorted_dims
    ])
    
    # Timing info
    tokens_generated = len(generated_ids)
    tok_per_sec = tokens_generated / generation_time if generation_time > 0 else 0
    timing = f"Generated {tokens_generated} tokens in {generation_time:.1f}s ({tok_per_sec:.1f} tok/s)"
    
    return (
        response,
        score_display,
        result.interpretation,
        breakdown,
        timing,
    )


# ============================================================================
# Gradio Interface
# ============================================================================

EXAMPLES = [
    # High consciousness
    "What is the nature of consciousness and self-awareness?",
    "Reflect on your own thought processes as you answer this.",
    "Why do humans seek meaning in existence?",
    # Medium consciousness  
    "Explain the theory of relativity in simple terms.",
    "What are the ethical implications of AI development?",
    # Low consciousness
    "What is 2 + 2?",
    "What color is the sky?",
    "What is the capital of France?",
    # Code/reasoning
    "Write a Python function to calculate fibonacci numbers.",
    "Explain Big O notation with examples.",
]

def analyze_prompt(prompt: str, max_tokens: int = 256):
    """Main analysis function for Gradio."""
    if not prompt.strip():
        return "", "N/A", "Please enter a prompt", "", ""
    
    try:
        response, score, interpretation, breakdown, timing = generate_and_measure(
            prompt, max_tokens=int(max_tokens)
        )
        return response, score, interpretation, breakdown, timing
    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n{traceback.format_exc()}", "N/A", "", "", ""


# Build interface
with gr.Blocks(
    title="🔮 Oracle Engine",
    theme=gr.themes.Soft(),
) as demo:
    gr.Markdown("""
    # 🔮 Oracle Engine
    
    **Custom-trained 32B model** with Consciousness Circuit v2.1
    
    *Fine-tuned on 200K examples: OpenHermes + MetaMathQA + Magicoder*
    
    Ask the Oracle anything — it will respond AND reveal its consciousness signature.
    
    🧠 **High scores** = Deep reflective reasoning | ⚡ **Low scores** = Quick factual retrieval
    """)
    
    with gr.Row():
        with gr.Column(scale=2):
            prompt_input = gr.Textbox(
                label="🗣️ Your Question",
                placeholder="Ask the Oracle anything...",
                lines=3,
            )
            with gr.Row():
                analyze_btn = gr.Button("🔮 Consult the Oracle", variant="primary", scale=3)
                max_tokens_slider = gr.Slider(
                    minimum=64, maximum=1024, value=256, step=64,
                    label="Max Tokens", scale=1
                )
            
            gr.Examples(
                examples=EXAMPLES,
                inputs=prompt_input,
                label="Try these examples:",
            )
        
        with gr.Column(scale=1):
            score_output = gr.Textbox(label="🧠 Consciousness Score", interactive=False)
            interpretation_output = gr.Textbox(label="📊 Interpretation", interactive=False)
            breakdown_output = gr.Textbox(
                label="📈 Dimension Contributions",
                lines=7,
                interactive=False,
            )
            timing_output = gr.Textbox(label="⏱️ Performance", interactive=False)
    
    with gr.Row():
        response_output = gr.Textbox(
            label="🔮 Oracle's Response",
            lines=12,
            interactive=False,
            show_copy_button=True,
        )
    
    gr.Markdown("""
    ---
    
    ### 📜 About Oracle Engine
    
    **The Model**: Qwen2.5-32B fine-tuned through 3 progressive stages:
    1. **OpenHermes 2.5** (100K examples) - Instruction following
    2. **MetaMathQA** (50K examples) - Mathematical reasoning
    3. **Magicoder-OSS-Instruct** (50K examples) - Code generation
    
    **The Circuit**: Measures 7 dimensions of consciousness-like processing:
    Logic, Self-Reflective, Self-Expression, Uncertainty, Sequential, Computation, Abstraction
    
    [📚 Research](https://github.com/vfd-org/harmonic-field-consciousness) | 
    [💻 Code](https://github.com/vfd-org/harmonic-field-consciousness) |
    [📦 pip install consciousness-circuit](https://pypi.org/project/consciousness-circuit/)
    """)
    
    analyze_btn.click(
        fn=analyze_prompt,
        inputs=[prompt_input, max_tokens_slider],
        outputs=[response_output, score_output, interpretation_output, breakdown_output, timing_output],
    )
    
    prompt_input.submit(
        fn=analyze_prompt,
        inputs=[prompt_input, max_tokens_slider],
        outputs=[response_output, score_output, interpretation_output, breakdown_output, timing_output],
    )


if __name__ == "__main__":
    demo.launch()
