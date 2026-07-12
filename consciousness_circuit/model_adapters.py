"""
Model Adapters for Consciousness Circuit
=========================================

Provides a unified interface for different model types:
- HuggingFace Transformers (AutoModel, AutoModelForCausalLM)
- NanoGPT (HarmonicGPT V5/V6)
- Unsloth (FastLanguageModel)

This decouples the consciousness measurement code from specific model implementations.

Usage:
    from consciousness_circuit.model_adapters import create_adapter

    # Auto-detect model type
    adapter = create_adapter(model, tokenizer)

    # Get hidden states
    hidden_states = adapter.get_hidden_states(prompt)

    # Get model info
    print(f"Hidden size: {adapter.hidden_size}")
    print(f"Num layers: {adapter.num_layers}")
"""

import torch
import torch.nn as nn
from abc import ABC, abstractmethod
from typing import List, Tuple, Optional, Dict, Any, Union
from dataclasses import dataclass

from .logging_config import get_logger

logger = get_logger(__name__)


@dataclass
class ModelInfo:
    """Information about a model."""
    hidden_size: int
    num_layers: int
    model_type: str
    device: str
    dtype: torch.dtype


class ModelAdapter(ABC):
    """
    Abstract base class for model adapters.

    All adapters must implement:
    - get_hidden_states(): Extract hidden states from a prompt
    - forward(): Run forward pass with optional hidden state output
    - generate(): Generate text (optional)
    """

    def __init__(self, model: nn.Module, tokenizer: Any):
        self.model = model
        self.tokenizer = tokenizer
        self._info: Optional[ModelInfo] = None

    @property
    @abstractmethod
    def hidden_size(self) -> int:
        """Return the model's hidden dimension size."""
        pass

    @property
    @abstractmethod
    def num_layers(self) -> int:
        """Return the number of transformer layers."""
        pass

    @property
    def device(self) -> torch.device:
        """Return the model's device."""
        return next(self.model.parameters()).device

    @property
    def dtype(self) -> torch.dtype:
        """Return the model's dtype."""
        return next(self.model.parameters()).dtype

    @abstractmethod
    def get_hidden_states(
        self,
        prompt: str,
        layer_indices: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        """
        Get hidden states for a prompt.

        Args:
            prompt: Input text
            layer_indices: Specific layers to return (None = all layers)

        Returns:
            List of hidden state tensors, one per requested layer
            Each tensor has shape (batch=1, seq_len, hidden_size)
        """
        pass

    @abstractmethod
    def forward(
        self,
        input_ids: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        """
        Run forward pass.

        Args:
            input_ids: Token IDs (batch, seq_len)
            output_hidden_states: Whether to return hidden states

        Returns:
            (logits, hidden_states) where hidden_states is None if not requested
        """
        pass

    def tokenize(self, prompt: str) -> torch.Tensor:
        """Tokenize a prompt and return input_ids on model device."""
        if hasattr(self.tokenizer, 'encode'):
            # tiktoken style
            input_ids = torch.tensor([self.tokenizer.encode(prompt)], device=self.device)
        else:
            # HuggingFace style
            inputs = self.tokenizer(prompt, return_tensors="pt")
            input_ids = inputs["input_ids"].to(self.device)
        return input_ids

    def get_info(self) -> ModelInfo:
        """Get model information."""
        if self._info is None:
            self._info = ModelInfo(
                hidden_size=self.hidden_size,
                num_layers=self.num_layers,
                model_type=self.__class__.__name__,
                device=str(self.device),
                dtype=self.dtype,
            )
        return self._info


class HuggingFaceAdapter(ModelAdapter):
    """
    Adapter for HuggingFace Transformers models.

    Works with:
    - AutoModelForCausalLM
    - AutoModel
    - Qwen2ForCausalLM
    - LlamaForCausalLM
    - etc.
    """

    @property
    def hidden_size(self) -> int:
        if hasattr(self.model.config, 'hidden_size'):
            return self.model.config.hidden_size
        elif hasattr(self.model.config, 'd_model'):
            return self.model.config.d_model
        raise ValueError("Could not determine hidden_size from model config")

    @property
    def num_layers(self) -> int:
        if hasattr(self.model.config, 'num_hidden_layers'):
            return self.model.config.num_hidden_layers
        elif hasattr(self.model.config, 'n_layer'):
            return self.model.config.n_layer
        raise ValueError("Could not determine num_layers from model config")

    def get_hidden_states(
        self,
        prompt: str,
        layer_indices: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        input_ids = self.tokenize(prompt)

        self.model.eval()
        with torch.no_grad():
            outputs = self.model(input_ids, output_hidden_states=True, return_dict=True)

        all_hidden = outputs.hidden_states  # Tuple of (batch, seq, hidden)

        if layer_indices is None:
            return list(all_hidden)
        else:
            return [all_hidden[i] for i in layer_indices if i < len(all_hidden)]

    def forward(
        self,
        input_ids: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        self.model.eval()
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                output_hidden_states=output_hidden_states,
                return_dict=True
            )

        logits = outputs.logits
        hidden_states = list(outputs.hidden_states) if output_hidden_states else None

        return logits, hidden_states


class NanoGPTAdapter(ModelAdapter):
    """
    Adapter for NanoGPT models (HarmonicGPT V3/V4/V5/V6).

    Works with:
    - HarmonicGPTV5
    - HarmonicGPTV6
    - GPT (base NanoGPT)
    - ConsciousnessWrapper
    """

    def __init__(self, model: nn.Module, tokenizer: Any):
        super().__init__(model, tokenizer)

        # Unwrap if it's a ConsciousnessWrapper
        if hasattr(model, 'base_model'):
            self._base_model = model.base_model
            self._is_wrapped = True
        else:
            self._base_model = model
            self._is_wrapped = False

    @property
    def hidden_size(self) -> int:
        # NanoGPT uses config dict or n_embd attribute
        if hasattr(self._base_model, 'config'):
            config = self._base_model.config
            if isinstance(config, dict):
                return config.get('n_embd', 768)
            else:
                return getattr(config, 'n_embd', 768)
        elif hasattr(self._base_model, 'n_embd'):
            return self._base_model.n_embd
        return 768  # Default

    @property
    def num_layers(self) -> int:
        if hasattr(self._base_model, 'config'):
            config = self._base_model.config
            if isinstance(config, dict):
                return config.get('n_layer', 12)
            else:
                return getattr(config, 'n_layer', 12)
        elif hasattr(self._base_model, 'n_layer'):
            return self._base_model.n_layer

        # Try to count transformer blocks
        if hasattr(self._base_model, 'transformer') and hasattr(self._base_model.transformer, 'h'):
            return len(self._base_model.transformer.h)
        elif hasattr(self._base_model, 'blocks'):
            return len(self._base_model.blocks)

        return 12  # Default

    def get_hidden_states(
        self,
        prompt: str,
        layer_indices: Optional[List[int]] = None,
    ) -> List[torch.Tensor]:
        input_ids = self.tokenize(prompt)
        _, hidden_states = self.forward(input_ids, output_hidden_states=True)

        if hidden_states is None:
            return []

        if layer_indices is None:
            return hidden_states
        else:
            return [hidden_states[i] for i in layer_indices if i < len(hidden_states)]

    def forward(
        self,
        input_ids: torch.Tensor,
        output_hidden_states: bool = False,
    ) -> Tuple[torch.Tensor, Optional[List[torch.Tensor]]]:
        self._base_model.eval()

        if output_hidden_states:
            # Need to use hooks to capture hidden states
            hidden_states = []

            def hook_fn(module, input, output):
                if isinstance(output, tuple):
                    hidden_states.append(output[0].detach())
                else:
                    hidden_states.append(output.detach())

            # Register hooks on transformer blocks
            hooks = []
            if hasattr(self._base_model, 'transformer') and hasattr(self._base_model.transformer, 'h'):
                for block in self._base_model.transformer.h:
                    hooks.append(block.register_forward_hook(hook_fn))
            elif hasattr(self._base_model, 'blocks'):
                for block in self._base_model.blocks:
                    hooks.append(block.register_forward_hook(hook_fn))

            try:
                with torch.no_grad():
                    logits = self._base_model(input_ids)
                    if isinstance(logits, tuple):
                        logits = logits[0]
            finally:
                # Remove hooks
                for hook in hooks:
                    hook.remove()

            return logits, hidden_states
        else:
            with torch.no_grad():
                logits = self._base_model(input_ids)
                if isinstance(logits, tuple):
                    logits = logits[0]
            return logits, None


class UnslothAdapter(HuggingFaceAdapter):
    """
    Adapter for Unsloth FastLanguageModel.

    Unsloth models are HuggingFace-compatible but may have
    slightly different behaviors for quantized models.
    """

    def __init__(self, model: nn.Module, tokenizer: Any):
        super().__init__(model, tokenizer)
        logger.info("Using Unsloth adapter for quantized model")

    # Inherits all methods from HuggingFaceAdapter
    # Can override specific methods if needed for quantization handling


def detect_model_type(model: nn.Module) -> str:
    """
    Detect the type of model.

    Returns:
        "huggingface", "nanogpt", "unsloth", or "unknown"
    """
    # Check for Unsloth
    model_name = getattr(model.config, '_name_or_path', '') if hasattr(model, 'config') else ''
    if 'unsloth' in model_name.lower() or 'bnb' in model_name.lower():
        return "unsloth"

    # Check for HuggingFace
    if hasattr(model, 'config') and hasattr(model.config, 'model_type'):
        return "huggingface"

    # Check for NanoGPT
    if hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
        return "nanogpt"
    if hasattr(model, 'blocks') and hasattr(model, 'tok_emb'):
        return "nanogpt"

    # Check for ConsciousnessWrapper
    if hasattr(model, 'base_model'):
        return "nanogpt"

    return "unknown"


def create_adapter(
    model: nn.Module,
    tokenizer: Any,
    model_type: Optional[str] = None,
) -> ModelAdapter:
    """
    Create the appropriate adapter for a model.

    Args:
        model: The model instance
        tokenizer: The tokenizer
        model_type: Override auto-detection ("huggingface", "nanogpt", "unsloth")

    Returns:
        ModelAdapter instance

    Example:
        adapter = create_adapter(model, tokenizer)
        hidden_states = adapter.get_hidden_states("What is consciousness?")
    """
    if model_type is None:
        model_type = detect_model_type(model)

    logger.info(f"Creating adapter for model type: {model_type}")

    if model_type == "huggingface":
        return HuggingFaceAdapter(model, tokenizer)
    elif model_type == "nanogpt":
        return NanoGPTAdapter(model, tokenizer)
    elif model_type == "unsloth":
        return UnslothAdapter(model, tokenizer)
    else:
        # Default to HuggingFace
        logger.warning(f"Unknown model type '{model_type}', defaulting to HuggingFace adapter")
        return HuggingFaceAdapter(model, tokenizer)


# Convenience function
def get_hidden_states(
    model: nn.Module,
    tokenizer: Any,
    prompt: str,
    layer_indices: Optional[List[int]] = None,
) -> List[torch.Tensor]:
    """
    Quick function to get hidden states from any model.

    Args:
        model: Any supported model
        tokenizer: The tokenizer
        prompt: Input text
        layer_indices: Specific layers (None = all)

    Returns:
        List of hidden state tensors
    """
    adapter = create_adapter(model, tokenizer)
    return adapter.get_hidden_states(prompt, layer_indices)
