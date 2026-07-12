"""
Activation patching utilities: clean→corrupt residual swaps to localize causal layers.
"""
from typing import Any, Dict, List, Optional

import torch

from .hooks import _find_layer_stack


def _run_forward(model, tokenizer, prompt: str, device: Optional[str] = None):
    device = device or str(next(model.parameters()).device)
    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(
            **encoded,
            output_hidden_states=True,
            use_cache=False,
        )
    return encoded, out.hidden_states, out.logits


def residual_patch_sweep(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt_clean: str,
    prompt_corrupt: str,
    target_fn,
    layer_indices: Optional[List[int]] = None,
    token_pos: int = -1,
) -> Dict[int, float]:
    """
    For each layer, patch corrupt run with clean residuals and measure target metric delta.

    target_fn: callable(logits, encoded_inputs) -> float
    returns: {layer_idx: metric_with_patch}
    """
    device = str(next(model.parameters()).device)
    layers = _find_layer_stack(model)
    if layer_indices is None:
        layer_indices = list(range(len(layers)))

    encoded_clean, hidden_clean, _ = _run_forward(model, tokenizer, prompt_clean, device)
    encoded_corrupt, _, _ = _run_forward(model, tokenizer, prompt_corrupt, device)

    results: Dict[int, float] = {}

    for layer_idx in layer_indices:
        # Patch hook
        def make_hook(clean_tensor):
            def hook(_mod, _inp, out):
                if not torch.is_tensor(out):
                    return out
                patched = out.clone()
                patched[:, token_pos, :] = clean_tensor[:, token_pos, :]
                return patched
            return hook

        handle = layers[layer_idx].register_forward_hook(make_hook(hidden_clean[layer_idx]))
        with torch.no_grad():
            out = model(
                **encoded_corrupt,
                output_hidden_states=False,
                use_cache=False,
            )
        metric = float(target_fn(out.logits, encoded_corrupt))
        results[layer_idx] = metric
        handle.remove()

    return results
