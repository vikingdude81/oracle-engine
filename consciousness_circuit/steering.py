"""
Residual steering utilities: apply probe/feature directions with norm caps.
"""
from dataclasses import dataclass
from typing import Optional

import torch

from .hooks import _find_layer_stack


@dataclass
class SteeringConfig:
    layer_idx: int
    alpha: float = 1.0
    token_pos: int = -1  # last token by default
    max_norm: Optional[float] = None


def add_residual_steering(
    model: torch.nn.Module,
    direction: torch.Tensor,
    config: SteeringConfig,
) -> torch.utils.hooks.RemovableHandle:
    """
    Adds direction to residual stream at the specified layer/token position.
    direction is expected to be 1D (hidden_dim,) or broadcastable.
    """
    layers = _find_layer_stack(model)
    target = layers[config.layer_idx]
    device = next(model.parameters()).device
    direction = direction.to(device)

    def hook(_mod, _inp, out):
        if not torch.is_tensor(out):
            return out
        delta = direction
        if config.max_norm:
            norm = torch.norm(delta)
            if norm > config.max_norm:
                delta = delta * (config.max_norm / (norm + 1e-8))
        if out.dim() == 3:
            out = out.clone()
            out[:, config.token_pos, :] += config.alpha * delta
            return out
        return out

    return target.register_forward_hook(hook)
