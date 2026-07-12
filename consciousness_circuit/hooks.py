"""
Lightweight hooks and caching utilities for transformer models.

Goals:
- Grab hidden states / attentions via forward pass (no model surgery).
- Provide best-effort layer stack discovery for common HF models.
- Optional logit-lens helper on cached hidden states.
"""
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import torch


@dataclass
class CaptureResult:
    tokens: List[str]
    token_ids: torch.Tensor
    logits: torch.Tensor
    hidden_states: List[torch.Tensor]
    attentions: Optional[List[torch.Tensor]]

    def to_cpu(self) -> "CaptureResult":  # type: ignore
        return CaptureResult(
            tokens=self.tokens,
            token_ids=self.token_ids.cpu(),
            logits=self.logits.cpu(),
            hidden_states=[h.cpu() for h in self.hidden_states],
            attentions=[a.cpu() for a in self.attentions] if self.attentions else None,
        )


def _find_layer_stack(model: torch.nn.Module) -> List[torch.nn.Module]:
    """Best-effort discovery of transformer block stack."""
    candidates = [
        getattr(model, "model", None),
        getattr(model, "transformer", None),
        getattr(model, "decoder", None),
        model,
    ]
    for root in candidates:
        if root is None:
            continue
        for attr in ("layers", "h", "blocks", "encoder_layer", "layer"):
            stack = getattr(root, attr, None)
            if isinstance(stack, (list, tuple)) and len(stack) > 0:
                return list(stack)
            if hasattr(stack, "__len__") and len(stack) > 0:
                return list(stack)
    raise ValueError("Could not find transformer layer stack (layers/h/blocks)")


def capture_forward(
    model: torch.nn.Module,
    tokenizer: Any,
    prompt: str,
    device: Optional[str] = None,
    max_new_tokens: int = 0,
    output_attentions: bool = True,
) -> CaptureResult:
    """Run a forward pass and cache hidden states/attentions.

    Note: this does not generate text; it runs a single forward with the prompt tokens.
    """
    model_device = next(model.parameters()).device
    device = device or str(model_device)

    encoded = tokenizer(prompt, return_tensors="pt").to(device)
    with torch.no_grad():
        out = model(
            **encoded,
            output_hidden_states=True,
            output_attentions=output_attentions,
            use_cache=False,
        )
    hidden_states = list(out.hidden_states)  # type: ignore
    attentions = list(out.attentions) if output_attentions and hasattr(out, "attentions") else None
    logits = out.logits  # type: ignore
    tokens = tokenizer.convert_ids_to_tokens(encoded.input_ids[0])
    return CaptureResult(tokens=tokens, token_ids=encoded.input_ids[0], logits=logits, hidden_states=hidden_states, attentions=attentions)


def logit_lens(
    hidden_states: List[torch.Tensor],
    model: torch.nn.Module,
    tokenizer: Any,
    top_k: int = 5,
) -> List[List[Tuple[str, float]]]:
    """Apply a simple logit lens: project intermediate hidden states to token probs."""
    if not hasattr(model, "lm_head"):
        raise ValueError("Model has no lm_head for logit lens")
    results: List[List[Tuple[str, float]]] = []
    device = hidden_states[0].device
    lm_head = model.lm_head
    norm = getattr(model, "model", None)
    norm = getattr(norm, "norm", None) if norm else None

    for hs in hidden_states:
        x = hs
        if x.dim() == 3:
            x_last = x[:, -1, :]
        else:
            x_last = x
        if norm:
            x_last = norm(x_last)
        logits = lm_head(x_last)  # type: ignore
        probs = torch.softmax(logits, dim=-1)[0]
        topv, topi = torch.topk(probs, k=top_k)
        results.append([(tokenizer.decode([i.item()]), v.item()) for v, i in zip(topv, topi)])
    return results


def apply_residual_steering(
    model: torch.nn.Module,
    layer_idx: int,
    direction: torch.Tensor,
    alpha: float = 1.0,
    token_pos: int = -1,
) -> torch.utils.hooks.RemovableHandle:
    """Add a steering vector to the residual stream at a given layer output."""
    layers = _find_layer_stack(model)
    target = layers[layer_idx]
    direction = direction.to(next(model.parameters()).device)

    def hook(_module, _inp, out):
        if not torch.is_tensor(out):
            return out
        if out.dim() == 3:
            out = out.clone()
            out[:, token_pos, :] += alpha * direction
            return out
        return out

    return target.register_forward_hook(hook)
