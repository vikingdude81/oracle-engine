"""
Universal Consciousness Circuit v3.0
=====================================

A truly universal consciousness measurement system that:
1. Auto-detects model architecture
2. Loads pre-discovered circuits if available  
3. Falls back to proportional remapping if not
4. Can discover new circuits on-the-fly

Usage:
    from consciousness_circuit.universal import UniversalCircuit
    
    circuit = UniversalCircuit()
    
    # Measure consciousness (auto-selects best circuit)
    result = circuit.measure(model, tokenizer, prompt)
    
    # Or discover a new circuit for this model
    circuit.discover(model, tokenizer, save=True)
"""

import torch
import json
import numpy as np
from pathlib import Path
from typing import Dict, Optional, Tuple, Any, Union
from dataclasses import dataclass
import hashlib


@dataclass
class UniversalResult:
    """Result from universal consciousness measurement."""
    score: float
    method: str  # "discovered" or "remapped"
    dimension_scores: Dict[str, float]
    model_name: str
    circuit_path: Optional[str]
    confidence: float = 1.0
    
    @property
    def dimension_contributions(self) -> Dict[str, float]:
        """Alias for dimension_scores for API compatibility."""
        return self.dimension_scores


# Default circuits directory
CIRCUITS_DIR = Path(__file__).parent / "discovered_circuits"


def get_model_id(model_name: str) -> str:
    """Generate safe filename from model name."""
    return model_name.replace("/", "_").replace("\\", "_")


class UniversalCircuit:
    """
    Universal Consciousness Circuit with auto-detection.
    
    Automatically selects the best measurement approach for each model:
    1. Use pre-discovered circuit if available
    2. Fall back to proportional remapping
    3. Option to discover new circuit on-the-fly
    """
    
    # Pre-packaged VALIDATED circuits (discovered via validation-based discovery)
    # These circuits achieve proper H > M > L ordering on test prompts
    BUNDLED_CIRCUITS = {
        # Qwen2.5-7B: Discrimination=0.621, High=0.734, Med=0.438, Low=0.113
        "Qwen/Qwen2.5-7B-Instruct": {
            "dimensions": {
                "Dim_1": 2023, "Dim_2": 411, "Dim_3": 1116,
                "Dim_4": 2628, "Dim_5": 419, "Dim_6": 2728, "Dim_7": 3209
            },
            "polarities": {
                "Dim_1": 1.0, "Dim_2": 1.0, "Dim_3": 1.0,
                "Dim_4": -1.0, "Dim_5": 1.0, "Dim_6": -1.0, "Dim_7": 1.0
            },
            "hidden_size": 3584,
            "layer_fraction": 0.75,
            "validation_metrics": {
                "discrimination": 0.621,
                "high_mean": 0.734,
                "medium_mean": 0.438,
                "low_mean": 0.113,
                "proper_ordering": True
            }
        },
        # Mistral-7B: Discrimination=0.101, High=0.554, Med=0.497, Low=0.454
        "mistralai/Mistral-7B-Instruct-v0.2": {
            "dimensions": {
                "Dim_1": 3362, "Dim_2": 777, "Dim_3": 284,
                "Dim_4": 1222, "Dim_5": 378, "Dim_6": 1669, "Dim_7": 3463
            },
            "polarities": {
                "Dim_1": 1.0, "Dim_2": -1.0, "Dim_3": 1.0,
                "Dim_4": -1.0, "Dim_5": 1.0, "Dim_6": 1.0, "Dim_7": -1.0
            },
            "hidden_size": 4096,
            "layer_fraction": 0.75,
            "validation_metrics": {
                "discrimination": 0.101,
                "high_mean": 0.554,
                "medium_mean": 0.497,
                "low_mean": 0.454,
                "proper_ordering": True
            }
        },
        # Qwen2.5-32B reference circuit (original v2.1, needs re-validation)
        "Qwen/Qwen2.5-32B-Instruct": {
            "dimensions": {
                "Logic": 3183, "Self-Reflective": 212, "Self-Expression": 5065,
                "Uncertainty": 4707, "Sequential": 295,
                "Computation": 1445, "Abstraction": 4578
            },
            "polarities": {
                "Logic": 1.0, "Self-Reflective": 1.0, "Self-Expression": 1.0,
                "Uncertainty": 1.0, "Sequential": 1.0,
                "Computation": -1.0, "Abstraction": 1.0
            },
            "hidden_size": 5120,
            "layer_fraction": 0.75,
            "validation_metrics": None  # Needs re-validation with discover_validated.py
        }
    }
    
    def __init__(
        self,
        circuits_dir: Optional[str] = None,
        enable_discovery: bool = True,
        cache_hidden_states: bool = True,
    ):
        """
        Initialize universal circuit.
        
        Args:
            circuits_dir: Directory to look for/save discovered circuits
            enable_discovery: Whether to allow on-the-fly discovery
            cache_hidden_states: Cache hidden states for efficiency
        """
        self.circuits_dir = Path(circuits_dir) if circuits_dir else CIRCUITS_DIR
        self.circuits_dir.mkdir(exist_ok=True)
        self.enable_discovery = enable_discovery
        self.cache_hidden_states = cache_hidden_states
        
        # Cache of loaded circuits
        self._circuit_cache: Dict[str, Dict] = {}
        
    def get_circuit(self, model_name: str, hidden_size: int) -> Tuple[Dict, str]:
        """
        Get the best available circuit for a model.
        
        Returns:
            (circuit_dict, method) where method is "discovered" or "remapped"
        """
        # Check cache
        if model_name in self._circuit_cache:
            return self._circuit_cache[model_name], "discovered"
            
        # Check bundled circuits (exact match)
        if model_name in self.BUNDLED_CIRCUITS:
            circuit = self.BUNDLED_CIRCUITS[model_name]
            self._circuit_cache[model_name] = circuit
            return circuit, "discovered"
            
        # Check for saved discovered circuit
        model_id = get_model_id(model_name)
        circuit_path = self.circuits_dir / f"{model_id}_circuit.json"
        
        if circuit_path.exists():
            with open(circuit_path, 'r') as f:
                circuit = json.load(f)
            self._circuit_cache[model_name] = circuit
            return circuit, "discovered"
            
        # Fall back to remapping from Qwen2.5-32B reference
        circuit = self._create_remapped_circuit(hidden_size)
        return circuit, "remapped"
        
    def _create_remapped_circuit(self, target_hidden: int) -> Dict:
        """Create a proportionally remapped circuit from Qwen2.5-32B reference."""
        reference = self.BUNDLED_CIRCUITS["Qwen/Qwen2.5-32B-Instruct"]
        ref_hidden = reference["hidden_size"]
        
        dimensions = {}
        for name, orig_dim in reference["dimensions"].items():
            new_dim = int((orig_dim / ref_hidden) * target_hidden)
            new_dim = min(new_dim, target_hidden - 1)  # Clamp
            dimensions[name] = new_dim
            
        return {
            "dimensions": dimensions,
            "polarities": reference["polarities"].copy(),
            "hidden_size": target_hidden,
            "layer_fraction": reference["layer_fraction"],
        }
        
    def measure(
        self,
        model,
        tokenizer,
        prompt: str,
        return_hidden: bool = False,
        aggregation: str = "mean",  # "mean", "last", or "max"
    ) -> UniversalResult:
        """
        Measure consciousness score for a prompt.
        
        Args:
            model: Loaded transformer model
            tokenizer: Model tokenizer
            prompt: Text to measure
            return_hidden: Also return raw hidden states
            aggregation: How to aggregate token scores - "mean", "last", or "max"
            
        Returns:
            UniversalResult with score and metadata
        """
        model_name = getattr(model.config, '_name_or_path', 'unknown')
        hidden_size = model.config.hidden_size
        num_layers = model.config.num_hidden_layers
        
        # Get circuit
        circuit, method = self.get_circuit(model_name, hidden_size)
        layer_frac = circuit.get("layer_fraction", 0.75)
        target_layer = int(num_layers * layer_frac)
        
        # Get activations
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        
        # Get full sequence hidden states [seq_len, hidden_size]
        hidden_seq = outputs.hidden_states[target_layer][0].cpu().float()
        seq_len = hidden_seq.shape[0]
        
        # Compute per-token scores
        token_scores = []
        for pos in range(seq_len):
            hidden = hidden_seq[pos]
            dim_scores_pos = {}
            for name, dim_idx in circuit["dimensions"].items():
                if dim_idx < len(hidden):
                    activation = hidden[dim_idx].item()
                    polarity = circuit["polarities"][name]
                    dim_scores_pos[name] = activation * polarity
            if dim_scores_pos:
                raw = sum(dim_scores_pos.values()) / len(dim_scores_pos)
                token_scores.append(1 / (1 + np.exp(-raw)))
            else:
                token_scores.append(0.5)
        
        # Aggregate based on method
        if aggregation == "mean":
            score = float(np.mean(token_scores))
        elif aggregation == "max":
            score = float(np.max(token_scores))
        else:  # "last"
            score = token_scores[-1] if token_scores else 0.5
        
        # Compute dimension contributions using last token (for interpretability)
        hidden = hidden_seq[-1]
        dim_scores = {}
        for name, dim_idx in circuit["dimensions"].items():
            if dim_idx < len(hidden):
                activation = hidden[dim_idx].item()
                polarity = circuit["polarities"][name]
                dim_scores[name] = activation * polarity
                
        circuit_path = None
        if method == "discovered":
            model_id = get_model_id(model_name)
            path = self.circuits_dir / f"{model_id}_circuit.json"
            if path.exists():
                circuit_path = str(path)
                
        return UniversalResult(
            score=score,
            method=method,
            dimension_scores=dim_scores,
            model_name=model_name,
            circuit_path=circuit_path,
        )
        
    def discover(
        self,
        model,
        tokenizer,
        top_k: int = 7,
        save: bool = True,
        verbose: bool = True,
    ) -> Dict:
        """
        Discover consciousness dimensions for this model.
        
        Uses contrastive prompt pairs to identify discriminating dimensions.
        """
        try:
            from .discover import DimensionDiscovery
        except ImportError:
            from discover import DimensionDiscovery
        
        model_name = getattr(model.config, '_name_or_path', 'unknown')
        
        if verbose:
            print(f"Discovering circuit for {model_name}...")
            
        # We need to use the discovery tool with already-loaded model
        # For now, create a simple inline discovery
        discovery = DimensionDiscovery(
            model_name,
            layer_fraction=0.75,
        )
        
        # Inject already-loaded model
        discovery.model = model
        discovery.tokenizer = tokenizer
        discovery.hidden_size = model.config.hidden_size
        discovery.target_layer = int(model.config.num_hidden_layers * 0.75)
        
        circuit = discovery.discover_circuit(top_k=top_k, verbose=verbose)
        
        if save:
            model_id = get_model_id(model_name)
            save_path = self.circuits_dir / f"{model_id}_circuit.json"
            discovery.save_circuit(circuit, str(save_path))
            
        # Cache the result
        circuit_dict = {
            "dimensions": circuit.dimensions,
            "polarities": circuit.polarities,
            "hidden_size": circuit.hidden_size,
            "layer_fraction": circuit.discovery_metadata["layer_fraction"],
        }
        self._circuit_cache[model_name] = circuit_dict
        
        return circuit_dict
        
    def list_available_circuits(self) -> Dict[str, str]:
        """List all available pre-discovered circuits."""
        circuits = {}
        
        # Bundled
        for name in self.BUNDLED_CIRCUITS:
            circuits[name] = "bundled"
            
        # Discovered
        for path in self.circuits_dir.glob("*_circuit.json"):
            with open(path, 'r') as f:
                data = json.load(f)
            name = data.get("model_name", path.stem)
            circuits[name] = str(path)
            
        return circuits


# Convenience function
def measure_consciousness(
    model,
    tokenizer, 
    prompt: str,
    circuits_dir: Optional[str] = None,
) -> UniversalResult:
    """
    Quick consciousness measurement with auto-detection.
    
    Example:
        from consciousness_circuit.universal import measure_consciousness
        result = measure_consciousness(model, tokenizer, "What is consciousness?")
        print(f"Score: {result.score:.3f} ({result.method})")
    """
    circuit = UniversalCircuit(circuits_dir=circuits_dir)
    return circuit.measure(model, tokenizer, prompt)


if __name__ == "__main__":
    # Demo
    circuit = UniversalCircuit()
    
    print("Available circuits:")
    for name, source in circuit.list_available_circuits().items():
        print(f"  {name}: {source}")
