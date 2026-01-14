"""
Correlation-Based Dimension Remapping
======================================

Improved dimension remapping that uses activation correlations
instead of proportional scaling to find best dimension mappings
between source and target models.

This addresses the issue in NANOGPT_IMPROVEMENTS.md where simple
proportional scaling may not preserve semantic meaning.

Usage:
    from consciousness_circuit.correlation_remapper import CorrelationRemapper

    remapper = CorrelationRemapper()

    # Learn mapping from test prompts
    mapping = remapper.learn_mapping(
        source_model=qwen_model,
        target_model=nanogpt_model,
        source_dims=[3183, 212, 5065, 4707, 295, 1445, 4578],
        test_prompts=[
            "What is consciousness?",
            "Explain photosynthesis.",
            "What is 2 + 2?"
        ]
    )

    # Use mapping to measure consciousness
    score = remapper.measure(nanogpt_model, prompt, mapping)
"""

import torch
import numpy as np
from typing import List, Dict, Tuple, Optional, Union, Any
from dataclasses import dataclass
import warnings


@dataclass
class DimensionMapping:
    """Mapping from source dimensions to target dimensions."""
    source_to_target: Dict[int, int]  # {source_dim: target_dim}
    correlations: Dict[int, float]     # {source_dim: correlation_strength}
    source_hidden_size: int
    target_hidden_size: int
    confidence: float  # Overall mapping confidence (0-1)

    def __repr__(self):
        return (f"DimensionMapping(source={self.source_hidden_size}, "
                f"target={self.target_hidden_size}, "
                f"confidence={self.confidence:.3f}, "
                f"mappings={len(self.source_to_target)})")


class CorrelationRemapper:
    """
    Learn and apply correlation-based dimension mappings.

    Instead of simple proportional scaling (source_dim * scale = target_dim),
    this finds target dimensions that have highest activation correlation
    with source dimensions across a set of test prompts.
    """

    def __init__(self, layer_fraction: float = 0.75, device: str = 'cuda'):
        """
        Args:
            layer_fraction: Which layer to extract activations from (0-1)
            device: Device for computation
        """
        self.layer_fraction = layer_fraction
        self.device = device if torch.cuda.is_available() else 'cpu'

    def _get_activations(
        self,
        model: torch.nn.Module,
        tokenizer: Any,
        prompts: List[str],
        layer_idx: Optional[int] = None
    ) -> torch.Tensor:
        """
        Get hidden state activations for a list of prompts.

        Returns:
            activations: (n_prompts, hidden_size) tensor
        """
        model.eval()
        activations = []

        # Determine layer index
        if layer_idx is None:
            if hasattr(model, 'n_layer'):
                n_layers = model.n_layer
            elif hasattr(model, 'config') and hasattr(model.config, 'n_layer'):
                n_layers = model.config.n_layer
            elif hasattr(model, 'transformer') and hasattr(model.transformer, 'h'):
                n_layers = len(model.transformer.h)
            else:
                # Try to infer from model
                try:
                    # HuggingFace style
                    n_layers = model.config.num_hidden_layers
                except:
                    n_layers = 12  # Default fallback

            layer_idx = int(n_layers * self.layer_fraction)

        with torch.no_grad():
            for prompt in prompts:
                # Tokenize
                if hasattr(tokenizer, 'encode'):
                    input_ids = torch.tensor([tokenizer.encode(prompt)], device=self.device)
                else:
                    input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(self.device)

                # Forward pass with hidden states
                try:
                    # Check if it's a NanoGPT wrapper (has base_model and different signature)
                    if hasattr(model, 'base_model'):
                        # NanoGPT ConsciousnessWrapper style
                        result = model(input_ids, None, output_hidden_states=True)
                        if len(result) >= 3:
                            # Returns (logits, loss, hidden_states)
                            hidden_states_list = result[2]
                            hidden_states = hidden_states_list[layer_idx]
                        else:
                            raise ValueError("NanoGPT model did not return hidden states")
                    elif hasattr(model, '__call__'):
                        outputs = model(input_ids, output_hidden_states=True)
                        if hasattr(outputs, 'hidden_states'):
                            hidden_states = outputs.hidden_states[layer_idx]
                        else:
                            # NanoGPT style - might return tuple
                            hidden_states = outputs[1][layer_idx] if len(outputs) > 1 else outputs[0]
                    else:
                        raise ValueError("Model does not support standard forward pass")

                    # Take mean over sequence (aggregate to single vector)
                    activation = hidden_states.mean(dim=1).squeeze(0)  # (hidden_size,)
                    activations.append(activation.cpu())

                except Exception as e:
                    warnings.warn(f"Failed to get activations for prompt '{prompt[:30]}...': {e}")
                    continue

        if len(activations) == 0:
            raise ValueError("Could not extract activations for any prompts")

        return torch.stack(activations)  # (n_prompts, hidden_size)

    def learn_mapping(
        self,
        source_model: torch.nn.Module,
        source_tokenizer: Any,
        target_model: torch.nn.Module,
        target_tokenizer: Any,
        source_dims: List[int],
        test_prompts: List[str],
        min_correlation: float = 0.3
    ) -> DimensionMapping:
        """
        Learn dimension mapping from source to target model.

        Args:
            source_model: Source model (e.g., Qwen with known circuit)
            source_tokenizer: Tokenizer for source model
            target_model: Target model (e.g., NanoGPT)
            target_tokenizer: Tokenizer for target model
            source_dims: List of source dimension indices to map
            test_prompts: Prompts to use for correlation analysis
            min_correlation: Minimum correlation to accept mapping

        Returns:
            DimensionMapping object
        """
        print(f"\n[CorrelationRemapper] Learning dimension mapping...")
        print(f"  Source dims: {len(source_dims)}")
        print(f"  Test prompts: {len(test_prompts)}")

        # Get activations from both models
        print(f"  Extracting source activations...")
        source_acts = self._get_activations(source_model, source_tokenizer, test_prompts)

        print(f"  Extracting target activations...")
        target_acts = self._get_activations(target_model, target_tokenizer, test_prompts)

        source_hidden_size = source_acts.shape[1]
        target_hidden_size = target_acts.shape[1]

        print(f"  Source hidden size: {source_hidden_size}")
        print(f"  Target hidden size: {target_hidden_size}")

        # For each source dimension, find best target dimension by correlation
        mapping = {}
        correlations = {}

        print(f"\n  Finding best correlations...")
        for source_dim in source_dims:
            if source_dim >= source_hidden_size:
                warnings.warn(f"Source dimension {source_dim} >= {source_hidden_size}, skipping")
                continue

            # Get activation pattern for this dimension across prompts
            source_pattern = source_acts[:, source_dim].numpy()  # (n_prompts,)

            # Correlate with all target dimensions
            best_target_dim = None
            best_corr = -1.0

            for target_dim in range(target_hidden_size):
                target_pattern = target_acts[:, target_dim].numpy()

                # Compute Pearson correlation
                if np.std(source_pattern) > 1e-6 and np.std(target_pattern) > 1e-6:
                    corr = np.corrcoef(source_pattern, target_pattern)[0, 1]

                    if abs(corr) > abs(best_corr):
                        best_corr = corr
                        best_target_dim = target_dim

            # Accept mapping if correlation is strong enough
            if best_target_dim is not None and abs(best_corr) >= min_correlation:
                mapping[source_dim] = best_target_dim
                correlations[source_dim] = best_corr
                print(f"    Dim {source_dim} -> {best_target_dim} (r={best_corr:.3f})")
            else:
                # Fallback to proportional mapping
                scale = target_hidden_size / source_hidden_size
                fallback_dim = int(source_dim * scale)
                mapping[source_dim] = fallback_dim
                correlations[source_dim] = 0.0
                print(f"    Dim {source_dim} -> {fallback_dim} (fallback, r={best_corr:.3f})")

        # Compute overall confidence
        confidence = np.mean([abs(c) for c in correlations.values()]) if correlations else 0.0

        print(f"\n  Mapping complete!")
        print(f"    Mapped: {len(mapping)}/{len(source_dims)} dimensions")
        print(f"    Mean correlation: {confidence:.3f}")

        return DimensionMapping(
            source_to_target=mapping,
            correlations=correlations,
            source_hidden_size=source_hidden_size,
            target_hidden_size=target_hidden_size,
            confidence=confidence
        )

    def apply_mapping(
        self,
        target_model: torch.nn.Module,
        tokenizer: Any,
        prompt: str,
        mapping: DimensionMapping,
        dimension_weights: Dict[int, float],
        dimension_polarities: Dict[int, int],
        aggregation: str = 'last'
    ) -> float:
        """
        Measure consciousness using learned mapping.

        Args:
            target_model: Model to measure
            tokenizer: Tokenizer for model
            prompt: Input prompt
            mapping: Learned dimension mapping
            dimension_weights: Weights for each source dimension
            dimension_polarities: Polarities (+1 or -1) for each dimension
            aggregation: 'last', 'mean', or 'all'

        Returns:
            Consciousness score (0-1)
        """
        target_model.eval()

        # Tokenize
        if hasattr(tokenizer, 'encode'):
            input_ids = torch.tensor([tokenizer.encode(prompt)], device=self.device)
        else:
            input_ids = tokenizer(prompt, return_tensors='pt').input_ids.to(self.device)

        # Get hidden states
        with torch.no_grad():
            # Check if it's a NanoGPT wrapper
            if hasattr(target_model, 'base_model'):
                # NanoGPT ConsciousnessWrapper style
                result = target_model(input_ids, None, output_hidden_states=True)
                if len(result) >= 3:
                    hidden_states = result[2]  # (logits, loss, hidden_states)
                else:
                    raise ValueError("NanoGPT model did not return hidden states")
            else:
                outputs = target_model(input_ids, output_hidden_states=True)
                if hasattr(outputs, 'hidden_states'):
                    hidden_states = outputs.hidden_states
                else:
                    hidden_states = outputs[1] if len(outputs) > 1 else outputs[0]

        # Select layer
        n_layers = None
        # Check wrapper's base_model first
        base = getattr(target_model, 'base_model', target_model)
        if hasattr(base, 'n_layer'):
            n_layers = base.n_layer
        elif hasattr(base, 'config'):
            config = base.config
            if isinstance(config, dict):
                n_layers = config.get('n_layer', len(hidden_states))
            else:
                n_layers = getattr(config, 'n_layer', len(hidden_states))
        if n_layers is None:
            n_layers = len(hidden_states)

        layer_idx = int(n_layers * self.layer_fraction)
        layer_hidden = hidden_states[layer_idx]  # (batch, seq_len, hidden_size)

        # Aggregate across sequence
        if aggregation == 'last':
            activations = layer_hidden[:, -1, :]  # (batch, hidden_size)
        elif aggregation == 'mean':
            activations = layer_hidden.mean(dim=1)  # (batch, hidden_size)
        elif aggregation == 'all':
            activations = layer_hidden.reshape(-1, layer_hidden.shape[-1])  # (batch*seq, hidden_size)
        else:
            raise ValueError(f"Unknown aggregation: {aggregation}")

        # Apply mapped circuit
        scores = []
        for batch_idx in range(activations.shape[0]):
            act = activations[batch_idx]  # (hidden_size,)

            weighted_sum = 0.0
            total_weight = 0.0

            for source_dim, weight in dimension_weights.items():
                target_dim = mapping.source_to_target.get(source_dim)
                if target_dim is not None and target_dim < act.shape[0]:
                    polarity = dimension_polarities.get(source_dim, 1)
                    correlation_strength = abs(mapping.correlations.get(source_dim, 0.5))

                    # Weight by both circuit weight and mapping correlation
                    effective_weight = weight * correlation_strength

                    weighted_sum += effective_weight * polarity * act[target_dim].item()
                    total_weight += effective_weight

            if total_weight > 0:
                score = weighted_sum / total_weight
            else:
                score = 0.0

            # Normalize to [0, 1] range (sigmoid)
            score = 1.0 / (1.0 + np.exp(-score))
            scores.append(score)

        # Return mean score
        return float(np.mean(scores))

    def save_mapping(self, mapping: DimensionMapping, filepath: str):
        """Save mapping to JSON file."""
        import json

        data = {
            'source_to_target': {str(k): v for k, v in mapping.source_to_target.items()},
            'correlations': {str(k): v for k, v in mapping.correlations.items()},
            'source_hidden_size': mapping.source_hidden_size,
            'target_hidden_size': mapping.target_hidden_size,
            'confidence': mapping.confidence,
        }

        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)

        print(f"[OK] Mapping saved to {filepath}")

    @staticmethod
    def load_mapping(filepath: str) -> DimensionMapping:
        """Load mapping from JSON file."""
        import json

        with open(filepath, 'r') as f:
            data = json.load(f)

        return DimensionMapping(
            source_to_target={int(k): v for k, v in data['source_to_target'].items()},
            correlations={int(k): v for k, v in data['correlations'].items()},
            source_hidden_size=data['source_hidden_size'],
            target_hidden_size=data['target_hidden_size'],
            confidence=data['confidence'],
        )


# Example usage
if __name__ == "__main__":
    print("CorrelationRemapper - Improved Dimension Mapping")
    print("=" * 80)
    print("\nUsage example:")
    print("""
from consciousness_circuit.correlation_remapper import CorrelationRemapper

# Initialize remapper
remapper = CorrelationRemapper()

# Learn mapping from Qwen to NanoGPT
mapping = remapper.learn_mapping(
    source_model=qwen_model,
    source_tokenizer=qwen_tokenizer,
    target_model=nanogpt_model,
    target_tokenizer=nanogpt_tokenizer,
    source_dims=[3183, 212, 5065, 4707, 295, 1445, 4578],
    test_prompts=[
        "What is consciousness?",
        "Explain photosynthesis.",
        "What is 2 + 2?"
    ]
)

# Use mapping to measure consciousness
score = remapper.apply_mapping(
    target_model=nanogpt_model,
    tokenizer=nanogpt_tokenizer,
    prompt="What is the nature of self-awareness?",
    mapping=mapping,
    dimension_weights={3183: 0.239, 212: 0.196, ...},
    dimension_polarities={3183: 1, 212: 1, ...}
)

# Save for future use
remapper.save_mapping(mapping, "nanogpt_v6_mapping.json")
""")
