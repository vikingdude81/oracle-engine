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

import os
os.environ['GRADIO_ALLOW_FLAGGING'] = 'never'

import gradio as gr
import torch
import numpy as np
from typing import Tuple
import time
import spaces

# ============================================================================
# Consciousness Circuit v2.1 (embedded for Space portability)
# ============================================================================

REFERENCE_HIDDEN_DIM = 5120

CONSCIOUS_DIMS_V2_1 = {
    3183: {"name": "Logic", "weight": 0.239, "polarity": +1},
    212:  {"name": "Self-Reflective", "weight": 0.196, "polarity": +1},
    5064: {"name": "Self-Expression", "weight": 0.109, "polarity": +1},  # Fixed: was 5065, out of bounds for hidden=5120
    4707: {"name": "Uncertainty", "weight": 0.130, "polarity": +1},
    295:  {"name": "Sequential", "weight": 0.087, "polarity": +1},
    1445: {"name": "Computation", "weight": 0.130, "polarity": -1},
    4578: {"name": "Abstraction", "weight": 0.109, "polarity": +1},
}

class ConsciousnessResult:
    """Simple result container without dataclass to avoid Gradio schema issues."""
    def __init__(self, score, raw_score, dimension_contributions, interpretation, processing_time):
        self.score = score
        self.raw_score = raw_score
        self.dimension_contributions = dimension_contributions
        self.interpretation = interpretation
        self.processing_time = processing_time


def compute_consciousness(
    hidden_state: torch.Tensor,
    hidden_dim: int = REFERENCE_HIDDEN_DIM,
    baseline: float = 0.5,
) -> ConsciousnessResult:
    """Compute consciousness score from hidden state tensor.

    v3.4.1 hybrid: z-score + tanh(z * 0.15) for smooth bounding.
    Preserves absolute z-score levels (strong dimension-level signals)
    while eliminating hard ceiling effects from v3.3 clamp.
    """
    import math
    start_time = time.time()
    TANH_SCALE = 0.15  # tanh(z*0.15): z=3→0.42, z=5→0.64, z=8→0.83, z=10→0.91

    # Remap dimensions if needed
    if hidden_dim != REFERENCE_HIDDEN_DIM:
        # WARNING: linear index scaling is NOT a validated remapping. Dimension
        # semantics do not survive index arithmetic across models/hidden sizes
        # (see consciousness_circuit/correlation_remapper.py for the validated
        # correlation-based approach). Scores produced through this branch are
        # UNCALIBRATED and should not be reported as circuit v2.1 measurements.
        import warnings
        warnings.warn(
            f"Hidden dim {hidden_dim} != calibrated {REFERENCE_HIDDEN_DIM}: "
            "falling back to naive index scaling - scores are UNCALIBRATED. "
            "Use consciousness_circuit.correlation_remapper for a validated remap.",
            stacklevel=2,
        )
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

    # Z-score normalize against full hidden state
    h_mean = h.mean().item()
    h_std = h.std().item()

    # Compute contributions with tanh smooth bounding
    contributions = {}
    weighted_sum = 0.0

    for dim_idx, info in dims.items():
        if dim_idx < len(h):
            z = (h[dim_idx].item() - h_mean) / (h_std + 1e-8)
            # Smooth bounding: preserves absolute level, no hard ceiling
            activation = math.tanh(z * TANH_SCALE)
            contribution = activation * info["weight"] * info["polarity"]
            weighted_sum += contribution
            contributions[info["name"]] = activation * info["polarity"]

    # Final score
    raw_score = baseline + weighted_sum * 0.15
    score = max(0.0, min(1.0, raw_score))

    # Interpretation
    if score >= 0.8:
        interpretation = "\U0001f9e0 High Consciousness - Deep reflective/philosophical reasoning"
    elif score >= 0.6:
        interpretation = "\U0001f4ad Medium-High - Complex analytical thinking"
    elif score >= 0.4:
        interpretation = "\u2696\ufe0f Medium - Balanced processing"
    elif score >= 0.2:
        interpretation = "\u26a1 Medium-Low - More automatic processing"
    else:
        interpretation = "\U0001f522 Low Consciousness - Quick factual retrieval"

    return ConsciousnessResult(
        score=score,
        raw_score=raw_score,
        dimension_contributions=contributions,
        interpretation=interpretation,
        processing_time=time.time() - start_time,
    )


# ============================================================================
# Compressibility Analysis (Weaver et al. PNAS 2026)
# ============================================================================

def analyze_compressibility(hidden_states_np, max_dims=200, seed=42):
    """
    Analyze representational compressibility of hidden states.
    Embedded version of CompressibilityPlugin for Space portability.

    Args:
        hidden_states_np: numpy array [seq_len, hidden_dim]
        max_dims: max dimensions to subsample for correlation analysis
        seed: random seed for reproducibility

    Returns:
        dict of compressibility metrics
    """
    seq_len, hidden_dim = hidden_states_np.shape

    if seq_len < 3 or hidden_dim < 2:
        return {"compressibility_corr": 0.0, "error": "too few tokens"}

    # Subsample dimensions for tractability
    if hidden_dim > max_dims:
        rng = np.random.RandomState(seed)
        dim_indices = np.sort(rng.choice(hidden_dim, max_dims, replace=False))
        states = hidden_states_np[:, dim_indices]
    else:
        states = hidden_states_np

    n_dims = states.shape[1]

    # Center the data
    states_centered = states - states.mean(axis=0, keepdims=True)

    # --- Eigenvalue-based metrics ---
    # Use Gram matrix approach since seq_len < hidden_dim typically
    if seq_len >= n_dims:
        cov = np.cov(states_centered, rowvar=False)
        eigenvalues = np.linalg.eigvalsh(cov)
    else:
        gram = states_centered @ states_centered.T / max(seq_len - 1, 1)
        eigenvalues = np.linalg.eigvalsh(gram)

    eigenvalues = np.sort(np.maximum(eigenvalues, 0))[::-1]
    eigenvalues = eigenvalues[eigenvalues > 1e-12]

    if len(eigenvalues) == 0:
        return {"compressibility_corr": 0.0, "error": "no eigenvalues"}

    total_var = eigenvalues.sum()
    cumvar = np.cumsum(eigenvalues) / total_var
    n_eig = len(eigenvalues)

    # Spectral entropy
    p = eigenvalues / total_var
    p = p[p > 0]
    spectral_entropy = float(-np.sum(p * np.log(p)))
    max_entropy = np.log(len(p))
    norm_spectral_entropy = float(spectral_entropy / max_entropy if max_entropy > 0 else 0)

    # Participation ratio
    participation_ratio = float(total_var ** 2 / np.sum(eigenvalues ** 2))

    # Effective dimensionality (90% variance)
    effective_dim = int(np.searchsorted(cumvar, 0.9) + 1)
    effective_dim = min(effective_dim, n_eig)

    # Top variance fractions
    top1_frac = float(eigenvalues[0] / total_var)
    top5_frac = float(eigenvalues[:min(5, n_eig)].sum() / total_var)
    top10_frac = float(eigenvalues[:min(10, n_eig)].sum() / total_var)

    # --- Correlation-based compression (paper's approach) ---
    corr_metrics = {}
    if n_dims <= 500 and seq_len >= max(10, n_dims // 5):
        stds = np.std(states_centered, axis=0)
        stds[stds < 1e-12] = 1.0
        states_norm = states_centered / stds
        corr = states_norm.T @ states_norm / max(seq_len - 1, 1)
        np.fill_diagonal(corr, 1.0)

        i_upper, j_upper = np.triu_indices(n_dims, k=1)
        correlations = corr[i_upper, j_upper]
        n_corr = len(correlations)

        if n_corr > 0:
            abs_corr = np.abs(correlations)
            sort_idx = np.argsort(abs_corr)[::-1]
            sorted_abs = abs_corr[sort_idx]

            rho_sq = np.clip(sorted_abs ** 2, 0, 0.9999)
            delta_s = -0.5 * np.log(1.0 - rho_sq)
            total_delta = delta_s.sum()

            if total_delta > 1e-12:
                cum_reduction = np.cumsum(delta_s) / total_delta
                fractions = np.arange(1, n_corr + 1) / n_corr
                c_corr = float(np.trapz(cum_reduction, fractions))
                idx_50 = int(np.searchsorted(cum_reduction, 0.5) + 1)
                idx_90 = int(np.searchsorted(cum_reduction, 0.9) + 1)

                corr_metrics = {
                    "compressibility_corr": c_corr,
                    "n_correlations": int(n_corr),
                    "fraction_for_50pct": float(min(idx_50 / n_corr, 1.0)),
                    "fraction_for_90pct": float(min(idx_90 / n_corr, 1.0)),
                    "mean_abs_correlation": float(abs_corr.mean()),
                    "max_abs_correlation": float(abs_corr.max()),
                    "median_abs_correlation": float(np.median(abs_corr)),
                    "strong_correlations_pct": float((abs_corr > 0.3).mean() * 100),
                }

    result = {
        "spectral_entropy": norm_spectral_entropy,
        "participation_ratio": participation_ratio,
        "effective_dimensionality": effective_dim,
        "effective_dim_fraction": float(effective_dim / n_eig),
        "top1_variance_fraction": top1_frac,
        "top5_variance_fraction": top5_frac,
        "top10_variance_fraction": top10_frac,
        "n_dims_analyzed": n_dims,
        "seq_len": seq_len,
    }
    result.update(corr_metrics)

    return result


# ============================================================================
# Model Loading
# ============================================================================

print("🔮 Loading Oracle Engine (Qwen2.5-32B-Instruct 4-bit + LoRA)...")
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel

BASE_MODEL_ID = "unsloth/Qwen2.5-32B-Instruct-bnb-4bit"
LORA_MODEL_ID = "Vikingdude81/oracle-engine-32b-lora"

# Get HF token from environment (set in Space secrets)
# Try multiple possible env var names
HF_TOKEN = os.environ.get("HF_TOKEN") or os.environ.get("HUGGING_FACE_HUB_TOKEN") or os.environ.get("HUGGINGFACE_TOKEN")
print(f"🔍 Environment vars: {[k for k in os.environ.keys() if 'HF' in k or 'HUGGING' in k or 'TOKEN' in k]}")

if HF_TOKEN:
    print(f"🔑 Found token: {HF_TOKEN[:10]}...{HF_TOKEN[-4:]} ({len(HF_TOKEN)} chars)")
else:
    print("⚠️ No HF token found in environment, attempting public access...")

# Load tokenizer from base model (LoRA only has weights, not tokenizer)
tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL_ID, token=HF_TOKEN)

# Load base model
base_model = AutoModelForCausalLM.from_pretrained(
    BASE_MODEL_ID,
    device_map="auto",
    torch_dtype=torch.bfloat16,
    trust_remote_code=True,
    token=HF_TOKEN,
)

# Apply LoRA adapter
print("🔗 Applying LoRA adapter...")
model = PeftModel.from_pretrained(base_model, LORA_MODEL_ID, token=HF_TOKEN)
model.eval()

HIDDEN_DIM = model.config.hidden_size
print(f"✅ Oracle Engine ready: {HIDDEN_DIM} hidden dimensions (with LoRA)")


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
# Experiment API - Returns JSON with all metrics
# ============================================================================

@spaces.GPU
def experiment_measure(prompt: str, max_tokens: int = 512) -> str:
    """
    API endpoint for experiments. Returns JSON with consciousness score,
    dimension scores, AND compressibility metrics.

    Args:
        prompt: Input text
        max_tokens: Max generation tokens

    Returns:
        JSON string with all metrics
    """
    import json
    start_time = time.time()

    # Format as chat message
    messages = [{"role": "user", "content": prompt}]
    chat_prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    # Tokenize
    inputs = tokenizer(chat_prompt, return_tensors="pt").to(model.device)

    # Generate
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=int(max_tokens),
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
            pad_token_id=tokenizer.eos_token_id,
        )

    generated_ids = outputs[0][inputs.input_ids.shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    gen_time = time.time() - start_time

    # Forward pass on full sequence for hidden states
    full_text = chat_prompt + response
    measure_inputs = tokenizer(full_text, return_tensors="pt").to(model.device)

    with torch.no_grad():
        measure_outputs = model(
            **measure_inputs,
            output_hidden_states=True,
            return_dict=True,
        )

    # --- Consciousness Score (last layer, last token) ---
    hidden_state_last = measure_outputs.hidden_states[-1]
    result = compute_consciousness(hidden_state_last, hidden_dim=HIDDEN_DIM)

    # --- Compressibility Analysis (75% layer, all tokens) ---
    n_layers = len(measure_outputs.hidden_states) - 1  # exclude embedding
    target_layer = int(n_layers * 0.75)
    hidden_seq = measure_outputs.hidden_states[target_layer][0].cpu().float().numpy()
    seq_len = hidden_seq.shape[0]

    compress_metrics = analyze_compressibility(hidden_seq, max_dims=200)

    # Build JSON result
    output = {
        "response": response,
        "consciousness_score": round(result.score, 4),
        "dimension_scores": {k: round(v, 4) for k, v in result.dimension_contributions.items()},
        "compressibility": compress_metrics,
        "meta": {
            "target_layer": target_layer,
            "seq_len": seq_len,
            "hidden_dim": HIDDEN_DIM,
            "tokens_generated": len(generated_ids),
            "generation_time": round(gen_time, 2),
        },
    }

    return json.dumps(output)


# ============================================================================
# Gradio Interface
# ============================================================================

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import io
import base64
from PIL import Image

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

# Global history for tracking
consciousness_history = []

def create_history_plot(history):
    """Create a consciousness history graph."""
    if len(history) < 1:
        return None
    
    fig, ax = plt.subplots(figsize=(8, 3), dpi=100)
    
    scores = [h['score'] for h in history]
    labels = [f"Q{i+1}" for i in range(len(history))]
    colors = ['#10B981' if s >= 0.6 else '#F59E0B' if s >= 0.4 else '#EF4444' for s in scores]
    
    bars = ax.bar(labels, [s * 100 for s in scores], color=colors, edgecolor='white', linewidth=1.5)
    
    ax.set_ylim(0, 100)
    ax.set_ylabel('Consciousness %', fontsize=10)
    ax.set_xlabel('Conversation Turn', fontsize=10)
    ax.axhline(y=60, color='#10B981', linestyle='--', alpha=0.5, label='High')
    ax.axhline(y=40, color='#F59E0B', linestyle='--', alpha=0.5, label='Medium')
    
    # Add value labels on bars
    for bar, score in zip(bars, scores):
        ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{score*100:.0f}%', ha='center', va='bottom', fontsize=9)
    
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)
    ax.set_facecolor('#1a1a2e')
    fig.patch.set_facecolor('#1a1a2e')
    ax.tick_params(colors='white')
    ax.xaxis.label.set_color('white')
    ax.yaxis.label.set_color('white')
    for spine in ax.spines.values():
        spine.set_color('white')
    
    plt.tight_layout()
    
    # Convert to PIL Image
    buf = io.BytesIO()
    plt.savefig(buf, format='png', facecolor='#1a1a2e', edgecolor='none')
    buf.seek(0)
    plt.close(fig)
    
    return Image.open(buf)

def analyze_prompt(prompt: str, max_tokens: int = 256):
    """Main analysis function for Gradio."""
    global consciousness_history
    
    if not prompt.strip():
        return "", "N/A", "Please enter a prompt", "", "", None
    
    try:
        response, score, interpretation, breakdown, timing = generate_and_measure(
            prompt, max_tokens=int(max_tokens)
        )
        
        # Extract score value
        score_val = float(score.split()[-1].replace('%', '')) / 100
        
        # Add to history
        consciousness_history.append({
            'prompt': prompt[:50],
            'score': score_val,
            'interpretation': interpretation
        })
        
        # Keep last 10 turns
        if len(consciousness_history) > 10:
            consciousness_history = consciousness_history[-10:]
        
        # Create history plot
        history_plot = create_history_plot(consciousness_history)
        
        return response, score, interpretation, breakdown, timing, history_plot
    except Exception as e:
        import traceback
        return f"Error: {str(e)}\n{traceback.format_exc()}", "N/A", "", "", "", None

def clear_history():
    """Clear conversation history."""
    global consciousness_history
    consciousness_history = []
    return None

def chat_respond(message, chat_history, max_tokens):
    """Chat mode - multi-turn conversation with consciousness tracking."""
    global consciousness_history
    
    if not message.strip():
        return chat_history, "", None
    
    try:
        response, score, interpretation, breakdown, timing = generate_and_measure(
            message, max_tokens=int(max_tokens)
        )
        
        # Extract score value
        score_val = float(score.split()[-1].replace('%', '')) / 100
        
        # Add to history
        consciousness_history.append({
            'prompt': message[:50],
            'score': score_val,
            'interpretation': interpretation
        })
        
        # Keep last 10
        if len(consciousness_history) > 10:
            consciousness_history = consciousness_history[-10:]
        
        # Format response with consciousness info
        formatted_response = f"{response}\n\n---\n🧠 **{score}** | {interpretation}"
        
        chat_history.append((message, formatted_response))
        history_plot = create_history_plot(consciousness_history)
        
        return chat_history, "", history_plot
    except Exception as e:
        chat_history.append((message, f"Error: {str(e)}"))
        return chat_history, "", None


# Build interface
with gr.Blocks(title="🔮 Oracle Engine") as demo:
    gr.Markdown("""
    # 🔮 Oracle Engine
    
    **Custom-trained 32B model** with Consciousness Circuit v2.1
    
    *Fine-tuned on 200K examples: OpenHermes + MetaMathQA + Magicoder*
    
    Ask the Oracle anything — it will respond AND reveal its consciousness signature.
    
    🧠 **High scores (60%+)** = Deep reflective reasoning | ⚡ **Low scores (<40%)** = Quick factual retrieval
    """)
    
    with gr.Tabs():
        # TAB 1: Single Query Mode
        with gr.TabItem("🔮 Single Query"):
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
                    lines=10,
                    interactive=False,
                )
            
            with gr.Row():
                history_plot = gr.Image(label="📊 Consciousness History", height=200)
                clear_btn = gr.Button("🗑️ Clear History", size="sm")
        
        # TAB 2: Chat Mode
        with gr.TabItem("💬 Chat Mode"):
            gr.Markdown("**Multi-turn conversation** with real-time consciousness tracking")
            
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(
                        label="Oracle Conversation",
                        height=400,
                    )
                    with gr.Row():
                        chat_input = gr.Textbox(
                            placeholder="Type your message...",
                            label="Message",
                            scale=4,
                        )
                        chat_max_tokens = gr.Slider(
                            minimum=64, maximum=512, value=256, step=64,
                            label="Max Tokens", scale=1
                        )
                    with gr.Row():
                        chat_send = gr.Button("Send 📤", variant="primary")
                        chat_clear = gr.Button("Clear Chat 🗑️")
                
                with gr.Column(scale=1):
                    chat_history_plot = gr.Image(label="📊 Consciousness Over Time", height=300)
    
    gr.Markdown("""
    ---
    
    ### 📜 About Oracle Engine
    
    **The Model**: Qwen2.5-32B fine-tuned through 3 progressive stages:
    1. **OpenHermes 2.5** (100K examples) - Instruction following
    2. **MetaMathQA** (50K examples) - Mathematical reasoning
    3. **Magicoder-OSS-Instruct** (50K examples) - Code generation
    
    **The Circuit**: Measures 7 dimensions of consciousness-like processing:
    Logic, Self-Reflective, Self-Expression, Uncertainty, Sequential, Computation, Abstraction
    
    [📚 GitHub](https://github.com/vikingdude81/oracle-engine) | 
    [🤗 Model](https://huggingface.co/Vikingdude81/oracle-engine-32b-lora) |
    [📖 Research](https://github.com/vfd-org/harmonic-field-consciousness)
    """)
    
    # Single query events
    analyze_btn.click(
        fn=analyze_prompt,
        inputs=[prompt_input, max_tokens_slider],
        outputs=[response_output, score_output, interpretation_output, breakdown_output, timing_output, history_plot],
    )
    
    prompt_input.submit(
        fn=analyze_prompt,
        inputs=[prompt_input, max_tokens_slider],
        outputs=[response_output, score_output, interpretation_output, breakdown_output, timing_output, history_plot],
    )
    
    clear_btn.click(fn=clear_history, outputs=[history_plot])
    
    # Chat mode events
    chat_send.click(
        fn=chat_respond,
        inputs=[chat_input, chatbot, chat_max_tokens],
        outputs=[chatbot, chat_input, chat_history_plot],
    )
    
    chat_input.submit(
        fn=chat_respond,
        inputs=[chat_input, chatbot, chat_max_tokens],
        outputs=[chatbot, chat_input, chat_history_plot],
    )
    
    chat_clear.click(
        fn=lambda: ([], None),
        outputs=[chatbot, chat_history_plot],
    ).then(fn=clear_history, outputs=[chat_history_plot])

    # Hidden API endpoint for experiments (callable via gradio_client)
    with gr.Row(visible=False):
        api_prompt = gr.Textbox()
        api_max_tokens = gr.Number(value=512)
        api_result = gr.Textbox()
        api_btn = gr.Button("api_trigger")
        api_btn.click(
            fn=experiment_measure,
            inputs=[api_prompt, api_max_tokens],
            outputs=api_result,
            api_name="experiment_measure",
        )


if __name__ == "__main__":
    demo.launch()
