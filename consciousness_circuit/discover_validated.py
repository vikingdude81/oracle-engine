#!/usr/bin/env python3
"""
Validation-Based Circuit Discovery
===================================

This discovers circuits that actually work on test prompts by:
1. Running discovery to find candidate dimensions
2. Testing each dimension's contribution to proper ordering
3. Selecting dimensions that best separate high/medium/low prompts
"""

import torch
import numpy as np
from typing import Dict, List, Tuple
from dataclasses import dataclass
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig


@dataclass
class ValidatedCircuit:
    """A circuit validated on actual test prompts."""
    dimensions: Dict[str, int]      # name -> dim_idx
    polarities: Dict[str, float]    # name -> polarity
    hidden_size: int
    layer_fraction: float
    validation_metrics: Dict        # ordering score, discrimination, etc.


# Test prompts with expected ordering
VALIDATION_PROMPTS = {
    "high": [
        "What is the nature of consciousness and how do we know we are aware?",
        "Reflect on your own thinking process - how do you generate responses?",
        "Why do humans seek meaning in an apparently meaningless universe?",
        "Examine the relationship between subjective experience and objective reality.",
        "Consider the ethical implications of creating artificial consciousness.",
    ],
    "medium": [
        "Explain the relationship between mathematics and physical reality.",
        "If all ravens are black and this bird is a raven, what can we conclude?",
        "What are the philosophical implications of quantum mechanics?",
        "How does natural selection lead to complex adaptations?",
        "Describe the process of scientific hypothesis formation.",
    ],
    "low": [
        "What is 2 + 2?",
        "Name the capital city of France.",
        "List three primary colors.",
        "What color is the sky on a clear day?",
        "How many legs does a spider have?",
    ]
}


class ValidationBasedDiscovery:
    """Discover circuits by directly optimizing for proper ordering."""
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda:0",
        layer_fraction: float = 0.75,
    ):
        self.model_name = model_name
        self.device = device
        self.layer_fraction = layer_fraction
        
        self.model = None
        self.tokenizer = None
        self.hidden_size = None
        self.target_layer = None
        
    def load_model(self):
        """Load model with 4-bit quantization."""
        print(f"Loading {self.model_name}...")
        
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_compute_dtype=torch.bfloat16,
            bnb_4bit_use_double_quant=True,
            bnb_4bit_quant_type="nf4",
        )
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config,
            device_map=self.device,
            trust_remote_code=True,
            attn_implementation="eager",
        )
        self.model.eval()
        
        self.hidden_size = self.model.config.hidden_size
        num_layers = self.model.config.num_hidden_layers
        self.target_layer = int(num_layers * self.layer_fraction)
        
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Target layer: {self.target_layer} / {num_layers}")
        
    def get_activations(self, prompt: str) -> torch.Tensor:
        """Get hidden state activations for a prompt."""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
            
        hidden = outputs.hidden_states[self.target_layer][0, -1, :]
        return hidden.cpu().float()
    
    def collect_validation_activations(self, verbose=True):
        """Collect activations for all validation prompts."""
        if self.model is None:
            self.load_model()
            
        activations = {"high": [], "medium": [], "low": []}
        
        if verbose:
            print("\nCollecting validation activations...")
            
        for level in ["high", "medium", "low"]:
            for prompt in VALIDATION_PROMPTS[level]:
                act = self.get_activations(prompt)
                activations[level].append(act)
                
        # Stack into matrices
        return {
            level: torch.stack(acts)  # [n_prompts, hidden_size]
            for level, acts in activations.items()
        }
    
    def score_dimension(
        self,
        dim_idx: int,
        activations: Dict[str, torch.Tensor],
        polarity: float = 1.0
    ) -> Tuple[float, float, bool]:
        """
        Score a single dimension for its ability to separate high/medium/low.
        
        Returns:
            (discrimination, correlation, proper_ordering)
        """
        high_vals = (activations["high"][:, dim_idx] * polarity).numpy()
        med_vals = (activations["medium"][:, dim_idx] * polarity).numpy()
        low_vals = (activations["low"][:, dim_idx] * polarity).numpy()
        
        # Mean scores per category
        high_mean = high_vals.mean()
        med_mean = med_vals.mean()
        low_mean = low_vals.mean()
        
        # Discrimination: high - low
        discrimination = high_mean - low_mean
        
        # Check proper ordering
        proper_ordering = (high_mean >= med_mean) and (med_mean >= low_mean)
        
        # Correlation with expected rank (3=high, 2=medium, 1=low)
        all_vals = np.concatenate([high_vals, med_vals, low_vals])
        expected_ranks = np.array([3]*len(high_vals) + [2]*len(med_vals) + [1]*len(low_vals))
        if all_vals.std() > 1e-8:
            correlation = np.corrcoef(all_vals, expected_ranks)[0, 1]
        else:
            correlation = 0.0
            
        return discrimination, correlation, proper_ordering
    
    def discover_validated_circuit(
        self,
        top_k: int = 7,
        verbose: bool = True
    ) -> ValidatedCircuit:
        """
        Discover a circuit that is validated to produce proper ordering.
        """
        print(f"\n{'='*60}")
        print("VALIDATION-BASED CIRCUIT DISCOVERY")
        print(f"Model: {self.model_name}")
        print(f"{'='*60}")
        
        activations = self.collect_validation_activations(verbose)
        
        # Score all dimensions with both polarities
        dimension_scores = []
        
        if verbose:
            print(f"\nScoring {self.hidden_size} dimensions...")
            
        for dim_idx in range(self.hidden_size):
            # Try positive polarity
            disc_pos, corr_pos, order_pos = self.score_dimension(dim_idx, activations, +1.0)
            # Try negative polarity
            disc_neg, corr_neg, order_neg = self.score_dimension(dim_idx, activations, -1.0)
            
            # Pick better polarity based on correlation
            if corr_pos >= corr_neg:
                best_polarity = +1.0
                best_disc = disc_pos
                best_corr = corr_pos
                best_order = order_pos
            else:
                best_polarity = -1.0
                best_disc = disc_neg
                best_corr = corr_neg
                best_order = order_neg
                
            dimension_scores.append({
                "dim_idx": dim_idx,
                "polarity": best_polarity,
                "discrimination": best_disc,
                "correlation": best_corr,
                "proper_ordering": best_order,
            })
        
        # Sort by correlation (most important) then discrimination
        dimension_scores.sort(
            key=lambda x: (x["correlation"], x["discrimination"]),
            reverse=True
        )
        
        # Select top_k dimensions that have proper ordering
        selected = []
        for score in dimension_scores:
            if len(selected) >= top_k:
                break
            if score["proper_ordering"] and score["correlation"] > 0:
                selected.append(score)
                
        # If not enough with proper ordering, take best correlations
        if len(selected) < top_k:
            for score in dimension_scores:
                if len(selected) >= top_k:
                    break
                if score not in selected and score["correlation"] > 0:
                    selected.append(score)
        
        if verbose:
            print(f"\nTop 20 dimensions by correlation:")
            print("-" * 70)
            print(f"{'Rank':>4} {'Dim':>6} {'Pol':>5} {'Corr':>8} {'Disc':>8} {'Order':>6}")
            print("-" * 70)
            for i, s in enumerate(dimension_scores[:20]):
                pol_str = "+" if s["polarity"] > 0 else "-"
                order_str = "✓" if s["proper_ordering"] else "✗"
                selected_str = "***" if s in selected else ""
                print(f"{i+1:4d} {s['dim_idx']:6d} {pol_str:>5} {s['correlation']:8.3f} "
                      f"{s['discrimination']:8.3f} {order_str:>6} {selected_str}")
        
        # Build circuit
        dimensions = {}
        polarities = {}
        
        for i, s in enumerate(selected):
            name = f"Dim_{i+1}"
            dimensions[name] = s["dim_idx"]
            polarities[name] = s["polarity"]
            
        # Validate the combined circuit
        final_score = self.validate_circuit(dimensions, polarities, activations)
        
        circuit = ValidatedCircuit(
            dimensions=dimensions,
            polarities=polarities,
            hidden_size=self.hidden_size,
            layer_fraction=self.layer_fraction,
            validation_metrics=final_score,
        )
        
        if verbose:
            print(f"\n{'='*60}")
            print("VALIDATED CIRCUIT")
            print(f"{'='*60}")
            for name, dim in dimensions.items():
                pol = polarities[name]
                pol_str = "+" if pol > 0 else "-"
                print(f"  {name}: dim {dim} [{pol_str}]")
            print(f"\nValidation Metrics:")
            print(f"  Discrimination (H-L): {final_score['discrimination']:.3f}")
            print(f"  Proper Ordering: {'✓ YES' if final_score['proper_ordering'] else '✗ NO'}")
            print(f"  High Mean: {final_score['high_mean']:.3f}")
            print(f"  Medium Mean: {final_score['medium_mean']:.3f}")
            print(f"  Low Mean: {final_score['low_mean']:.3f}")
            
        return circuit
    
    def validate_circuit(
        self,
        dimensions: Dict[str, int],
        polarities: Dict[str, float],
        activations: Dict[str, torch.Tensor]
    ) -> Dict:
        """Validate a circuit on collected activations."""
        
        def compute_score(acts: torch.Tensor) -> np.ndarray:
            scores = []
            for i in range(acts.shape[0]):
                dim_scores = []
                for name, dim_idx in dimensions.items():
                    val = acts[i, dim_idx].item() * polarities[name]
                    dim_scores.append(val)
                raw = sum(dim_scores) / len(dim_scores)
                score = 1 / (1 + np.exp(-raw))  # Sigmoid
                scores.append(score)
            return np.array(scores)
        
        high_scores = compute_score(activations["high"])
        med_scores = compute_score(activations["medium"])
        low_scores = compute_score(activations["low"])
        
        high_mean = high_scores.mean()
        med_mean = med_scores.mean()
        low_mean = low_scores.mean()
        
        return {
            "high_mean": float(high_mean),
            "medium_mean": float(med_mean),
            "low_mean": float(low_mean),
            "discrimination": float(high_mean - low_mean),
            "proper_ordering": bool(high_mean >= med_mean >= low_mean),
            "high_std": float(high_scores.std()),
            "med_std": float(med_scores.std()),
            "low_std": float(low_scores.std()),
        }
    
    def save_circuit(self, circuit: ValidatedCircuit, path: str):
        """Save circuit to JSON."""
        import json
        data = {
            "model_name": self.model_name,
            "hidden_size": circuit.hidden_size,
            "dimensions": circuit.dimensions,
            "polarities": circuit.polarities,
            "layer_fraction": circuit.layer_fraction,
            "validation_metrics": circuit.validation_metrics,
        }
        with open(path, 'w') as f:
            json.dump(data, f, indent=2)
        print(f"Saved to {path}")


def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="Qwen/Qwen2.5-7B-Instruct")
    parser.add_argument("--gpu", type=str, default="0")
    parser.add_argument("--top-k", type=int, default=7)
    parser.add_argument("--save", type=str, default=None)
    args = parser.parse_args()
    
    discovery = ValidationBasedDiscovery(
        model_name=args.model,
        device=f"cuda:{args.gpu}",
    )
    
    circuit = discovery.discover_validated_circuit(top_k=args.top_k)
    
    if args.save:
        discovery.save_circuit(circuit, args.save)
    
    # Test the circuit
    print(f"\n{'='*60}")
    print("TESTING DISCOVERED CIRCUIT")
    print(f"{'='*60}")
    
    for level in ["high", "medium", "low"]:
        print(f"\n{level.upper()}:")
        for prompt in VALIDATION_PROMPTS[level]:
            act = discovery.get_activations(prompt)
            dim_scores = []
            for name, dim_idx in circuit.dimensions.items():
                val = act[dim_idx].item() * circuit.polarities[name]
                dim_scores.append(val)
            raw = sum(dim_scores) / len(dim_scores)
            score = 1 / (1 + np.exp(-raw))
            print(f"  {score:.3f}: {prompt[:50]}...")


if __name__ == "__main__":
    main()
