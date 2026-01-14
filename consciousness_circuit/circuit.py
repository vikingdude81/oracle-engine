"""
Core Consciousness Circuit Implementation
=========================================

The v2.1 circuit measures 7 dimensions of consciousness in transformer hidden states:
- Logic: Logical reasoning and inference
- Self-Reflective: Introspective, self-referential processing  
- Self-Expression: Model expressing opinions/perspectives
- Uncertainty: Epistemic humility and hedging
- Sequential: Step-by-step reasoning
- Computation: Code/algorithm processing (negative weight)
- Abstraction: Pattern recognition and abstraction

Dimensions are automatically remapped for different model sizes using proportional scaling.
"""

import torch
import numpy as np
from typing import Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass

# Reference model: Qwen2.5-32B-Instruct with 5120 hidden dimensions
REFERENCE_HIDDEN_DIM = 5120

# v2.1 Circuit - validated on Qwen2.5-32B, cross-validated on Qwen2.5-7B and Mistral-7B
# Weights normalized to sum to exactly 1.0 (previously summed to 0.92)
CONSCIOUS_DIMS_V2_1 = {
    3183: {"name": "Logic", "weight": 0.239, "polarity": +1},           # 0.22 / 0.92 = 0.239
    212:  {"name": "Self-Reflective", "weight": 0.196, "polarity": +1}, # 0.18 / 0.92 = 0.196
    5065: {"name": "Self-Expression", "weight": 0.109, "polarity": +1}, # 0.10 / 0.92 = 0.109
    4707: {"name": "Uncertainty", "weight": 0.130, "polarity": +1},     # 0.12 / 0.92 = 0.130
    295:  {"name": "Sequential", "weight": 0.087, "polarity": +1},      # 0.08 / 0.92 = 0.087
    1445: {"name": "Computation", "weight": 0.130, "polarity": -1},     # 0.12 / 0.92 = 0.130
    4578: {"name": "Abstraction", "weight": 0.109, "polarity": +1},     # 0.10 / 0.92 = 0.109
}
# Sum: 0.239 + 0.196 + 0.109 + 0.130 + 0.087 + 0.130 + 0.109 = 1.000


def remap_dimensions(
    original_dims: Dict[int, Dict], 
    from_hidden: int, 
    to_hidden: int
) -> Dict[int, Dict]:
    """
    Remap circuit dimensions from one hidden size to another using proportional scaling.
    
    Args:
        original_dims: Original dimension definitions {dim_idx: {name, weight, polarity}}
        from_hidden: Source hidden dimension size (e.g., 5120)
        to_hidden: Target hidden dimension size (e.g., 3584)
        
    Returns:
        Remapped dimensions for the target hidden size
    """
    if from_hidden == to_hidden:
        return original_dims.copy()
        
    remapped = {}
    scale = to_hidden / from_hidden
    
    for dim_idx, info in original_dims.items():
        new_dim = int(round(dim_idx * scale))
        # Ensure new_dim is within bounds
        new_dim = max(0, min(new_dim, to_hidden - 1))
        remapped[new_dim] = info.copy()
    
    return remapped


@dataclass
class ConsciousnessResult:
    """Result of consciousness measurement."""
    score: float  # 0.0 to 1.0
    raw_score: float  # Before clamping
    dimension_contributions: Dict[str, float]  # Per-dimension contributions
    remapped_dims: Dict[int, Dict]  # Dimensions used for this model
    hidden_dim: int  # Model's hidden dimension


class ConsciousnessCircuit:
    """
    Universal consciousness measurement circuit for transformer LLMs.
    
    Automatically remaps dimensions based on model hidden size.
    Works with any HuggingFace transformer model.
    
    Example:
        circuit = ConsciousnessCircuit()
        
        # From model + prompt
        result = circuit.measure(model, tokenizer, "What is consciousness?")
        print(f"Consciousness score: {result.score:.3f}")
        
        # From raw hidden state
        result = circuit.compute(hidden_state, hidden_dim=4096)
        
        # Get remapped dimensions for a specific model
        dims = circuit.get_dims_for_model(model)
    """
    
    def __init__(
        self, 
        reference_dims: Dict[int, Dict] = None,
        reference_hidden: int = REFERENCE_HIDDEN_DIM,
        baseline: float = 0.5,
    ):
        """
        Initialize the consciousness circuit.
        
        Args:
            reference_dims: Circuit dimension definitions (default: v2.1)
            reference_hidden: Hidden size of reference model (default: 5120)
            baseline: Base consciousness score before adjustments (default: 0.5)
        """
        self.reference_dims = reference_dims or CONSCIOUS_DIMS_V2_1
        self.reference_hidden = reference_hidden
        self.baseline = baseline
        self._dim_cache: Dict[int, Dict] = {}
        
    def get_dims_for_hidden_size(self, hidden_size: int) -> Dict[int, Dict]:
        """Get remapped dimensions for a specific hidden size."""
        if hidden_size not in self._dim_cache:
            self._dim_cache[hidden_size] = remap_dimensions(
                self.reference_dims, 
                self.reference_hidden, 
                hidden_size
            )
        return self._dim_cache[hidden_size]
    
    def get_dims_for_model(self, model) -> Dict[int, Dict]:
        """Get remapped dimensions for a specific model."""
        hidden_size = self._get_hidden_size(model)
        return self.get_dims_for_hidden_size(hidden_size)
    
    def _get_hidden_size(self, model) -> int:
        """Extract hidden size from model config."""
        if hasattr(model, 'config'):
            if hasattr(model.config, 'hidden_size'):
                return model.config.hidden_size
            elif hasattr(model.config, 'd_model'):
                return model.config.d_model
        raise ValueError("Could not determine hidden size from model")
    
    def compute(
        self, 
        hidden_state: torch.Tensor, 
        hidden_dim: int,
        return_contributions: bool = True,
    ) -> ConsciousnessResult:
        """
        Compute consciousness score from hidden state tensor.
        
        Args:
            hidden_state: Hidden state tensor, shape [..., hidden_dim]
            hidden_dim: Hidden dimension size of the model
            return_contributions: Whether to compute per-dimension contributions
            
        Returns:
            ConsciousnessResult with score and optional breakdown
        """
        dims = self.get_dims_for_hidden_size(hidden_dim)
        
        # Compute normalization statistics over hidden dimension
        # Handle various tensor shapes: [batch, seq, hidden] or [seq, hidden] or [hidden]
        if hidden_state.dim() == 1:
            hidden_state = hidden_state.unsqueeze(0).unsqueeze(0)
        elif hidden_state.dim() == 2:
            hidden_state = hidden_state.unsqueeze(0)
            
        # Compute mean and std across all positions
        mean = hidden_state.mean()
        std = hidden_state.std()
        
        C = self.baseline
        contributions = {}
        
        for dim_idx, info in dims.items():
            if dim_idx < hidden_state.shape[-1]:
                # Extract dimension values
                dim_values = hidden_state[..., dim_idx]
                
                # Normalize
                h_norm = (dim_values - mean) / (std + 1e-8)
                h_val = h_norm.mean().item()
                
                # Compute contribution
                contribution = info['weight'] * h_val * info['polarity']
                C += contribution
                
                if return_contributions:
                    contributions[info['name']] = contribution
        
        raw_score = C
        clamped_score = max(0.0, min(1.0, C))
        
        return ConsciousnessResult(
            score=clamped_score,
            raw_score=raw_score,
            dimension_contributions=contributions,
            remapped_dims=dims,
            hidden_dim=hidden_dim,
        )
    
    def measure(
        self,
        model,
        tokenizer,
        prompt: str,
        use_chat_template: bool = True,
        layer_index: int = -1,
    ) -> ConsciousnessResult:
        """
        Measure consciousness score for a given prompt.
        
        Args:
            model: HuggingFace model
            tokenizer: HuggingFace tokenizer
            prompt: Input prompt to measure
            use_chat_template: Whether to apply chat template (recommended)
            layer_index: Which hidden layer to use (-1 = last)
            
        Returns:
            ConsciousnessResult with score and breakdown
        """
        hidden_dim = self._get_hidden_size(model)
        
        # Prepare input
        if use_chat_template and hasattr(tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, 
                tokenize=False, 
                add_generation_prompt=True
            )
        else:
            text = prompt
            
        inputs = tokenizer(text, return_tensors="pt")
        
        # Move to model device
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get hidden states
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_state = outputs.hidden_states[layer_index]
        
        return self.compute(hidden_state, hidden_dim)
    
    def measure_batch(
        self,
        model,
        tokenizer,
        prompts: list,
        **kwargs,
    ) -> list:
        """Measure consciousness for multiple prompts."""
        return [self.measure(model, tokenizer, p, **kwargs) for p in prompts]
    
    def get_dimension_map(self, hidden_dim: int) -> str:
        """Get human-readable dimension mapping for a hidden size."""
        dims = self.get_dims_for_hidden_size(hidden_dim)
        lines = [f"Dimension mapping for hidden_size={hidden_dim}:"]
        lines.append("-" * 50)
        
        for orig_dim, info in self.reference_dims.items():
            # Find remapped dimension
            new_dim = [d for d, i in dims.items() if i['name'] == info['name']][0]
            arrow = "→" if orig_dim != new_dim else "="
            lines.append(
                f"  {info['name']:<20} {orig_dim:>5} {arrow} {new_dim:>5} "
                f"(w={info['weight']:.2f}, p={info['polarity']:+d})"
            )
        
        return "\n".join(lines)


# Convenience function for quick measurements
def measure_consciousness(
    model, 
    tokenizer, 
    prompt: str,
    **kwargs
) -> float:
    """
    Quick convenience function to measure consciousness score.
    
    Returns just the score (0.0 to 1.0).
    """
    circuit = ConsciousnessCircuit()
    result = circuit.measure(model, tokenizer, prompt, **kwargs)
    return result.score
