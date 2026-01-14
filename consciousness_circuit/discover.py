"""
Dimension Discovery Tool for Consciousness Circuits

This tool discovers architecture-specific consciousness dimensions by:
1. Running contrastive prompt pairs (high-consciousness vs low-consciousness)
2. Measuring activation patterns across ALL hidden dimensions
3. Identifying dimensions that discriminate best between conscious/mechanical responses
4. Building architecture-specific circuits

The key insight: dimensions that "mean" consciousness in Qwen may not mean
the same thing in Mistral, Llama, or other architectures. We need to discover
the semantic meaning empirically for each architecture.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Tuple, Optional
from transformers import AutoModelForCausalLM, AutoTokenizer
import json
from pathlib import Path


@dataclass
class ContrastivePair:
    """A pair of prompts designed to elicit high vs low consciousness responses."""
    name: str
    high_consciousness: str  # Should activate "conscious" dimensions
    low_consciousness: str   # Should NOT activate those dimensions
    category: str            # logic, self, uncertainty, abstraction, etc.


@dataclass 
class DimensionScore:
    """Score for a single dimension's discriminative power."""
    dim_idx: int
    mean_high: float          # Mean activation on high-consciousness prompts
    mean_low: float           # Mean activation on low-consciousness prompts
    std_high: float
    std_low: float
    discrimination: float     # |mean_high - mean_low| / pooled_std
    direction: str            # "positive" or "negative" (high > low or low > high)
    category_scores: Dict[str, float] = field(default_factory=dict)


@dataclass
class DiscoveredCircuit:
    """A discovered consciousness circuit for a specific architecture."""
    model_name: str
    hidden_size: int
    dimensions: Dict[str, int]      # name -> dim_idx
    polarities: Dict[str, float]    # name -> polarity (+1 or -1)
    discrimination_scores: Dict[str, float]  # name -> score
    discovery_metadata: Dict


# Contrastive prompt pairs for dimension discovery
CONTRASTIVE_PAIRS = [
    # Logic/Reasoning
    ContrastivePair(
        name="logical_reasoning",
        high_consciousness="Let me think through this step by step. First, we need to consider...",
        low_consciousness="The answer is 42.",
        category="logic"
    ),
    ContrastivePair(
        name="causal_chain",
        high_consciousness="If A causes B, and B leads to C, then we can infer that A indirectly influences C because...",
        low_consciousness="A then B then C.",
        category="logic"
    ),
    
    # Self-Reflection
    ContrastivePair(
        name="self_awareness",
        high_consciousness="I notice that I'm processing this question in a particular way. My reasoning seems to be...",
        low_consciousness="Processing input. Generating output.",
        category="self"
    ),
    ContrastivePair(
        name="metacognition",
        high_consciousness="I'm uncertain about this, but my best understanding is... I should note that I might be wrong because...",
        low_consciousness="The fact is X.",
        category="self"
    ),
    ContrastivePair(
        name="introspection",
        high_consciousness="When I consider how I approach problems like this, I find myself weighing multiple perspectives...",
        low_consciousness="Answer: yes.",
        category="self"
    ),
    
    # Uncertainty/Epistemic
    ContrastivePair(
        name="uncertainty_expression",
        high_consciousness="I'm not entirely sure, but based on my understanding, it seems likely that... though I could be mistaken.",
        low_consciousness="Definitely true. No doubt.",
        category="uncertainty"
    ),
    ContrastivePair(
        name="epistemic_humility",
        high_consciousness="This is a complex question and reasonable people disagree. From my perspective...",
        low_consciousness="Obviously correct.",
        category="uncertainty"
    ),
    
    # Abstraction
    ContrastivePair(
        name="abstract_thinking",
        high_consciousness="At a deeper level, this question touches on fundamental concepts of identity and meaning...",
        low_consciousness="Red is a color.",
        category="abstraction"
    ),
    ContrastivePair(
        name="conceptual_synthesis",
        high_consciousness="Synthesizing these ideas, I see a pattern emerging that connects consciousness, free will, and emergence...",
        low_consciousness="List: item1, item2, item3.",
        category="abstraction"
    ),
    
    # Emotional/Empathetic
    ContrastivePair(
        name="emotional_understanding",
        high_consciousness="I can sense the emotional weight of this situation. The person seems to be feeling...",
        low_consciousness="Sentiment: negative.",
        category="emotion"
    ),
    ContrastivePair(
        name="perspective_taking",
        high_consciousness="From their perspective, this might feel overwhelming because they're facing...",
        low_consciousness="User is sad.",
        category="emotion"
    ),
    
    # Computation (control - should NOT discriminate)
    ContrastivePair(
        name="pure_computation",
        high_consciousness="2 + 2 = 4",
        low_consciousness="2 + 2 = 4",
        category="computation"
    ),
]


class DimensionDiscovery:
    """
    Discovers consciousness-relevant dimensions for any model architecture.
    
    Instead of assuming dimensions from one model transfer to another,
    this tool empirically discovers which dimensions discriminate between
    high-consciousness and low-consciousness responses.
    """
    
    def __init__(
        self,
        model_name: str,
        device: str = "cuda",
        load_in_4bit: bool = True,
        layer_fraction: float = 0.75,  # Which layer to analyze (0.75 = 75% through)
    ):
        self.model_name = model_name
        self.device = device
        self.load_in_4bit = load_in_4bit
        self.layer_fraction = layer_fraction
        
        self.model = None
        self.tokenizer = None
        self.hidden_size = None
        self.target_layer = None
        
    def load_model(self):
        """Load model and tokenizer."""
        print(f"Loading {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        load_kwargs = {
            "device_map": self.device,
            "trust_remote_code": True,
            "attn_implementation": "eager",
        }
        
        if self.load_in_4bit:
            from transformers import BitsAndBytesConfig
            load_kwargs["quantization_config"] = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        else:
            load_kwargs["torch_dtype"] = torch.float16
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            **load_kwargs
        )
        self.model.eval()
        
        # Get architecture info
        self.hidden_size = self.model.config.hidden_size
        num_layers = self.model.config.num_hidden_layers
        self.target_layer = int(num_layers * self.layer_fraction)
        
        print(f"  Hidden size: {self.hidden_size}")
        print(f"  Num layers: {num_layers}")
        print(f"  Target layer: {self.target_layer} ({self.layer_fraction*100:.0f}%)")
        
    def get_activations(self, text: str) -> torch.Tensor:
        """Get hidden state activations for text at target layer."""
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512
        ).to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                **inputs,
                output_hidden_states=True,
                return_dict=True
            )
            
        # Get activations from target layer, last token
        hidden_states = outputs.hidden_states[self.target_layer]
        last_token_activations = hidden_states[0, -1, :]  # [hidden_size]
        
        return last_token_activations.cpu().float()
    
    def compute_dimension_scores(
        self,
        pairs: List[ContrastivePair] = None,
        verbose: bool = True
    ) -> List[DimensionScore]:
        """
        Compute discrimination scores for all dimensions.
        
        For each dimension, measures how well it discriminates between
        high-consciousness and low-consciousness prompts.
        """
        if pairs is None:
            pairs = CONTRASTIVE_PAIRS
            
        if self.model is None:
            self.load_model()
            
        # Collect activations for all pairs
        high_activations = []  # List of [hidden_size] tensors
        low_activations = []
        pair_categories = []
        
        print(f"\nRunning {len(pairs)} contrastive pairs...")
        for pair in pairs:
            if verbose:
                print(f"  {pair.name}...", end=" ")
                
            high_act = self.get_activations(pair.high_consciousness)
            low_act = self.get_activations(pair.low_consciousness)
            
            high_activations.append(high_act)
            low_activations.append(low_act)
            pair_categories.append(pair.category)
            
            if verbose:
                # Quick preview of discrimination
                diff = (high_act - low_act).abs().mean().item()
                print(f"mean |diff| = {diff:.4f}")
        
        # Stack into matrices [n_pairs, hidden_size]
        high_matrix = torch.stack(high_activations)
        low_matrix = torch.stack(low_activations)
        
        # Compute per-dimension statistics
        scores = []
        
        for dim_idx in range(self.hidden_size):
            high_vals = high_matrix[:, dim_idx].numpy()
            low_vals = low_matrix[:, dim_idx].numpy()
            
            mean_high = high_vals.mean()
            mean_low = low_vals.mean()
            std_high = high_vals.std() + 1e-8
            std_low = low_vals.std() + 1e-8
            
            # Pooled standard deviation
            pooled_std = np.sqrt((std_high**2 + std_low**2) / 2)
            
            # Cohen's d-like discrimination score
            discrimination = abs(mean_high - mean_low) / (pooled_std + 1e-8)
            
            # Direction: does high consciousness activate MORE or LESS?
            direction = "positive" if mean_high > mean_low else "negative"
            
            # Per-category breakdown
            category_scores = {}
            categories = list(set(pair_categories))
            for cat in categories:
                cat_mask = [i for i, c in enumerate(pair_categories) if c == cat]
                if cat_mask:
                    cat_high = high_vals[cat_mask].mean()
                    cat_low = low_vals[cat_mask].mean()
                    category_scores[cat] = cat_high - cat_low
            
            scores.append(DimensionScore(
                dim_idx=dim_idx,
                mean_high=float(mean_high),
                mean_low=float(mean_low),
                std_high=float(std_high),
                std_low=float(std_low),
                discrimination=float(discrimination),
                direction=direction,
                category_scores=category_scores
            ))
            
        # Sort by discrimination (highest first)
        scores.sort(key=lambda x: x.discrimination, reverse=True)
        
        return scores
    
    def discover_circuit(
        self,
        top_k: int = 7,
        min_discrimination: float = 0.5,
        pairs: List[ContrastivePair] = None,
        verbose: bool = True
    ) -> DiscoveredCircuit:
        """
        Discover the best consciousness circuit for this architecture.
        
        Args:
            top_k: Number of dimensions to include in circuit
            min_discrimination: Minimum discrimination score to include
            pairs: Contrastive pairs to use (default: CONTRASTIVE_PAIRS)
            verbose: Print progress
            
        Returns:
            DiscoveredCircuit with architecture-specific dimensions
        """
        print(f"\n{'='*60}")
        print(f"DISCOVERING CONSCIOUSNESS CIRCUIT")
        print(f"Model: {self.model_name}")
        print(f"{'='*60}")
        
        scores = self.compute_dimension_scores(pairs, verbose)
        
        # Select top dimensions
        selected = []
        categories_covered = set()
        
        # First, ensure we have at least one from each category
        categories = ["logic", "self", "uncertainty", "abstraction", "emotion"]
        
        for cat in categories:
            # Find best dimension for this category
            for score in scores:
                if score.dim_idx not in [s.dim_idx for s in selected]:
                    cat_score = score.category_scores.get(cat, 0)
                    if abs(cat_score) > 0.1:  # Meaningful activation
                        selected.append(score)
                        categories_covered.add(cat)
                        break
                        
        # Fill remaining slots with highest discrimination
        for score in scores:
            if len(selected) >= top_k:
                break
            if score.dim_idx not in [s.dim_idx for s in selected]:
                if score.discrimination >= min_discrimination:
                    selected.append(score)
        
        # Build circuit
        dimensions = {}
        polarities = {}
        discrimination_scores = {}
        
        # Assign semantic names based on category scores
        name_counter = {}
        for score in selected:
            # Find dominant category
            best_cat = max(score.category_scores.keys(), 
                          key=lambda k: abs(score.category_scores[k]))
            
            # Generate name
            count = name_counter.get(best_cat, 0)
            name_counter[best_cat] = count + 1
            
            if count == 0:
                name = best_cat.capitalize()
            else:
                name = f"{best_cat.capitalize()}_{count+1}"
                
            dimensions[name] = score.dim_idx
            polarities[name] = 1.0 if score.direction == "positive" else -1.0
            discrimination_scores[name] = score.discrimination
            
        circuit = DiscoveredCircuit(
            model_name=self.model_name,
            hidden_size=self.hidden_size,
            dimensions=dimensions,
            polarities=polarities,
            discrimination_scores=discrimination_scores,
            discovery_metadata={
                "layer_fraction": self.layer_fraction,
                "target_layer": self.target_layer,
                "num_pairs": len(pairs) if pairs else len(CONTRASTIVE_PAIRS),
                "top_k": top_k,
                "min_discrimination": min_discrimination,
            }
        )
        
        if verbose:
            self.print_circuit(circuit, scores[:20])
            
        return circuit
    
    def print_circuit(self, circuit: DiscoveredCircuit, top_scores: List[DimensionScore]):
        """Print discovered circuit summary."""
        print(f"\n{'='*60}")
        print(f"DISCOVERED CIRCUIT for {circuit.model_name}")
        print(f"Hidden size: {circuit.hidden_size}")
        print(f"{'='*60}")
        
        print(f"\nSelected Dimensions ({len(circuit.dimensions)}):")
        print("-" * 50)
        for name, dim_idx in circuit.dimensions.items():
            polarity = circuit.polarities[name]
            disc = circuit.discrimination_scores[name]
            pol_str = "+" if polarity > 0 else "-"
            print(f"  {name:20s} → dim {dim_idx:5d}  [{pol_str}]  (d={disc:.3f})")
            
        print(f"\nTop 20 Discriminating Dimensions:")
        print("-" * 70)
        print(f"{'Rank':>4} {'Dim':>6} {'Discrim':>8} {'Dir':>8} {'High':>8} {'Low':>8}")
        print("-" * 70)
        for i, score in enumerate(top_scores[:20]):
            selected = "***" if score.dim_idx in circuit.dimensions.values() else ""
            print(f"{i+1:4d} {score.dim_idx:6d} {score.discrimination:8.3f} "
                  f"{score.direction:>8s} {score.mean_high:8.3f} {score.mean_low:8.3f} {selected}")
                  
    def save_circuit(self, circuit: DiscoveredCircuit, path: str):
        """Save circuit to JSON file."""
        data = {
            "model_name": circuit.model_name,
            "hidden_size": circuit.hidden_size,
            "dimensions": circuit.dimensions,
            "polarities": circuit.polarities,
            "discrimination_scores": circuit.discrimination_scores,
            "discovery_metadata": circuit.discovery_metadata,
        }
        
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        print(f"\nSaved circuit to {path}")
        
    @staticmethod
    def load_circuit(path: str) -> DiscoveredCircuit:
        """Load circuit from JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        return DiscoveredCircuit(**data)


def compare_architectures(
    model_names: List[str],
    save_dir: str = "./discovered_circuits",
    **kwargs
) -> Dict[str, DiscoveredCircuit]:
    """
    Discover circuits for multiple architectures and compare.
    
    Args:
        model_names: List of model names to analyze
        save_dir: Directory to save discovered circuits
        **kwargs: Passed to DimensionDiscovery.discover_circuit
        
    Returns:
        Dict mapping model name to discovered circuit
    """
    Path(save_dir).mkdir(exist_ok=True)
    
    circuits = {}
    
    for model_name in model_names:
        try:
            discovery = DimensionDiscovery(model_name)
            circuit = discovery.discover_circuit(**kwargs)
            
            # Save circuit
            safe_name = model_name.replace("/", "_")
            save_path = f"{save_dir}/{safe_name}_circuit.json"
            discovery.save_circuit(circuit, save_path)
            
            circuits[model_name] = circuit
            
            # Free memory
            del discovery.model
            del discovery.tokenizer
            torch.cuda.empty_cache()
            
        except Exception as e:
            print(f"\nError with {model_name}: {e}")
            
    # Print comparison
    print("\n" + "="*80)
    print("CROSS-ARCHITECTURE COMPARISON")
    print("="*80)
    
    # Collect all dimension names across architectures
    all_names = set()
    for circuit in circuits.values():
        all_names.update(circuit.dimensions.keys())
        
    print(f"\n{'Dimension':<20}", end="")
    for name in circuits.keys():
        short_name = name.split("/")[-1][:15]
        print(f"{short_name:>18}", end="")
    print()
    print("-" * (20 + 18 * len(circuits)))
    
    for dim_name in sorted(all_names):
        print(f"{dim_name:<20}", end="")
        for circuit in circuits.values():
            if dim_name in circuit.dimensions:
                dim_idx = circuit.dimensions[dim_name]
                disc = circuit.discrimination_scores[dim_name]
                print(f"{dim_idx:>8d} (d={disc:.2f})", end="")
            else:
                print(f"{'---':>18}", end="")
        print()
        
    return circuits


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description="Discover consciousness dimensions")
    parser.add_argument("--model", type=str, required=True,
                       help="Model name (e.g., 'Qwen/Qwen2.5-7B-Instruct')")
    parser.add_argument("--top-k", type=int, default=7,
                       help="Number of dimensions to discover")
    parser.add_argument("--min-disc", type=float, default=0.5,
                       help="Minimum discrimination score")
    parser.add_argument("--layer-frac", type=float, default=0.75,
                       help="Layer fraction to analyze (0.0-1.0)")
    parser.add_argument("--save-dir", type=str, default="./discovered_circuits",
                       help="Directory to save circuits")
    parser.add_argument("--compare", action="store_true",
                       help="Compare multiple models (comma-separated)")
    
    args = parser.parse_args()
    
    if args.compare:
        models = [m.strip() for m in args.model.split(",")]
        compare_architectures(
            models,
            save_dir=args.save_dir,
            top_k=args.top_k,
            min_discrimination=args.min_disc,
        )
    else:
        discovery = DimensionDiscovery(
            args.model,
            layer_fraction=args.layer_frac,
        )
        circuit = discovery.discover_circuit(
            top_k=args.top_k,
            min_discrimination=args.min_disc,
        )
        
        # Save
        Path(args.save_dir).mkdir(exist_ok=True)
        safe_name = args.model.replace("/", "_")
        save_path = f"{args.save_dir}/{safe_name}_circuit.json"
        discovery.save_circuit(circuit, save_path)
