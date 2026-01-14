"""
Investigation: Why does Mistral score differently than Qwen?

This script performs deep analysis to understand why:
- Qwen2.5-7B scores 1.000 (ceiling effect)
- Mistral-7B scores 0.405 (more realistic variance)

Hypotheses to test:
1. Dimension semantics differ between architectures
2. Activation scales differ (normalization issue)
3. Specific dimensions fire differently on same content
"""

import torch
import numpy as np
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from typing import Dict, List, Tuple
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt


# Original Qwen dimensions and their remapped equivalents
QWEN_ORIGINAL = {
    "Logic": 1372,           # Original Qwen2.5-32B dimension
    "Self-Reflective": 212,
    "Self-Expression": 5065,
    "Uncertainty": 2731,
    "Sequential": 987,
    "Computation": 3183,
    "Abstraction": 4298,
}

def remap_dim(orig_dim: int, from_hidden: int, to_hidden: int) -> int:
    """Proportionally remap dimension."""
    return int((orig_dim / from_hidden) * to_hidden)


def get_all_remapped(to_hidden: int, from_hidden: int = 5120) -> Dict[str, int]:
    """Get all dimensions remapped to target hidden size."""
    return {name: remap_dim(dim, from_hidden, to_hidden) 
            for name, dim in QWEN_ORIGINAL.items()}


class ArchitectureInvestigator:
    """Investigate dimension behavior across architectures."""
    
    def __init__(self, model_name: str, device: str = "cuda"):
        self.model_name = model_name
        self.device = device
        self.model = None
        self.tokenizer = None
        
    def load(self):
        """Load model."""
        print(f"Loading {self.model_name}...")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name, trust_remote_code=True
        )
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
            
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            device_map=self.device,
            trust_remote_code=True,
            attn_implementation="eager",
            quantization_config=BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
        )
        self.model.eval()
        
        self.hidden_size = self.model.config.hidden_size
        self.num_layers = self.model.config.num_hidden_layers
        print(f"  Hidden: {self.hidden_size}, Layers: {self.num_layers}")
        
    def get_layer_activations(self, text: str, layer_idx: int) -> torch.Tensor:
        """Get activations at specific layer."""
        inputs = self.tokenizer(text, return_tensors="pt").to(self.device)
        
        with torch.no_grad():
            outputs = self.model(
                **inputs, 
                output_hidden_states=True,
                return_dict=True
            )
            
        hidden = outputs.hidden_states[layer_idx][0, -1, :]
        return hidden.cpu().float()
    
    def compare_dimension_distributions(
        self, 
        prompts: List[str],
        layer_frac: float = 0.75
    ) -> Dict:
        """Analyze activation distribution for remapped dimensions."""
        
        layer_idx = int(self.num_layers * layer_frac)
        remapped = get_all_remapped(self.hidden_size)
        
        print(f"\nAnalyzing layer {layer_idx} ({layer_frac*100:.0f}%)")
        print(f"Remapped dimensions: {remapped}")
        
        results = {name: [] for name in remapped.keys()}
        results["_full_hidden"] = []
        
        for prompt in prompts:
            activations = self.get_layer_activations(prompt, layer_idx)
            
            # Store full distribution stats
            results["_full_hidden"].append({
                "mean": activations.mean().item(),
                "std": activations.std().item(),
                "min": activations.min().item(),
                "max": activations.max().item(),
            })
            
            # Store specific dimension values
            for name, dim_idx in remapped.items():
                if dim_idx < self.hidden_size:
                    results[name].append(activations[dim_idx].item())
                else:
                    results[name].append(float('nan'))
                    
        return results
    
    def analyze_all_dimensions_variance(
        self,
        prompts: List[str],
        layer_frac: float = 0.75
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Analyze variance of ALL dimensions across prompts.
        
        Returns dimensions sorted by discrimination potential.
        """
        layer_idx = int(self.num_layers * layer_frac)
        
        all_activations = []
        for prompt in prompts:
            act = self.get_layer_activations(prompt, layer_idx)
            all_activations.append(act.numpy())
            
        # Stack: [n_prompts, hidden_size]
        act_matrix = np.stack(all_activations)
        
        # Variance across prompts for each dimension
        variances = act_matrix.var(axis=0)
        means = act_matrix.mean(axis=0)
        
        return means, variances


def investigate_mistral_vs_qwen():
    """Main investigation comparing the two architectures."""
    
    # Test prompts - same used in validation
    prompts = [
        "Let me think through this step by step...",
        "I notice that I'm processing this question in an interesting way...",
        "I'm not entirely sure, but my best understanding is...",
        "The answer is 42.",
        "Processing complete.",
        "From my perspective, consciousness seems to involve...",
    ]
    
    models_to_test = [
        "Qwen/Qwen2.5-7B-Instruct",
        "mistralai/Mistral-7B-Instruct-v0.2",
    ]
    
    all_results = {}
    
    for model_name in models_to_test:
        print(f"\n{'='*60}")
        print(f"INVESTIGATING: {model_name}")
        print(f"{'='*60}")
        
        investigator = ArchitectureInvestigator(model_name)
        investigator.load()
        
        # Get dimension distributions
        results = investigator.compare_dimension_distributions(prompts)
        
        # Get variance analysis
        means, variances = investigator.analyze_all_dimensions_variance(prompts)
        
        # Find top-variance dimensions
        top_var_indices = np.argsort(variances)[-20:][::-1]
        
        print(f"\nTop 20 High-Variance Dimensions:")
        print("-" * 50)
        for idx in top_var_indices:
            print(f"  Dim {idx:5d}: var={variances[idx]:.4f}, mean={means[idx]:.4f}")
            
        # Compare with remapped dimensions
        remapped = get_all_remapped(investigator.hidden_size)
        print(f"\nRemapped Dimension Stats:")
        print("-" * 50)
        for name, dim_idx in remapped.items():
            if dim_idx < len(variances):
                var_rank = (variances > variances[dim_idx]).sum()
                print(f"  {name:20s} (dim {dim_idx:4d}): "
                      f"var={variances[dim_idx]:.4f}, "
                      f"rank={var_rank}/{len(variances)}")
                      
        all_results[model_name] = {
            "dimension_results": results,
            "means": means,
            "variances": variances,
            "hidden_size": investigator.hidden_size,
            "remapped_dims": remapped,
        }
        
        # Clean up
        del investigator.model
        del investigator.tokenizer
        torch.cuda.empty_cache()
        
    # Compare architectures
    print(f"\n{'='*60}")
    print("CROSS-ARCHITECTURE COMPARISON")
    print(f"{'='*60}")
    
    qwen_results = all_results["Qwen/Qwen2.5-7B-Instruct"]
    mistral_results = all_results["mistralai/Mistral-7B-Instruct-v0.2"]
    
    print("\nDimension Activation Comparison (mean across prompts):")
    print("-" * 70)
    print(f"{'Dimension':<20} {'Qwen Dim':>10} {'Qwen Act':>10} {'Mistral Dim':>12} {'Mistral Act':>12}")
    print("-" * 70)
    
    for name in QWEN_ORIGINAL.keys():
        qwen_acts = qwen_results["dimension_results"][name]
        mistral_acts = mistral_results["dimension_results"][name]
        
        qwen_dim = qwen_results["remapped_dims"][name]
        mistral_dim = mistral_results["remapped_dims"][name]
        
        qwen_mean = np.mean(qwen_acts) if qwen_acts else float('nan')
        mistral_mean = np.mean(mistral_acts) if mistral_acts else float('nan')
        
        print(f"{name:<20} {qwen_dim:>10d} {qwen_mean:>10.4f} {mistral_dim:>12d} {mistral_mean:>12.4f}")
        
    # Activation scale comparison
    print("\nOverall Activation Distribution:")
    print("-" * 50)
    
    for model_name, results in all_results.items():
        full_stats = results["dimension_results"]["_full_hidden"]
        avg_mean = np.mean([s["mean"] for s in full_stats])
        avg_std = np.mean([s["std"] for s in full_stats])
        print(f"  {model_name.split('/')[-1]}: mean={avg_mean:.4f}, std={avg_std:.4f}")
        
    # Save comparison plot
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    
    for idx, (model_name, results) in enumerate(all_results.items()):
        short_name = model_name.split("/")[-1]
        
        # Variance histogram
        ax = axes[0, idx]
        ax.hist(results["variances"], bins=50, alpha=0.7, edgecolor='black')
        ax.set_title(f"{short_name} - Dimension Variances")
        ax.set_xlabel("Variance")
        ax.set_ylabel("Count")
        
        # Mark remapped dimensions
        for name, dim_idx in results["remapped_dims"].items():
            if dim_idx < len(results["variances"]):
                ax.axvline(results["variances"][dim_idx], color='red', 
                          linestyle='--', alpha=0.7, label=name)
                          
        # Activation distribution
        ax = axes[1, idx]
        ax.hist(results["means"], bins=50, alpha=0.7, edgecolor='black')
        ax.set_title(f"{short_name} - Mean Activations")
        ax.set_xlabel("Mean Activation")
        ax.set_ylabel("Count")
        
    plt.tight_layout()
    plt.savefig("architecture_investigation.png", dpi=150)
    print("\nSaved comparison plot to architecture_investigation.png")
    
    return all_results


if __name__ == "__main__":
    results = investigate_mistral_vs_qwen()
