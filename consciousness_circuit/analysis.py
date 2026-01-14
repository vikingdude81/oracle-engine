"""
Consciousness Analysis Tools
============================

Tools for analyzing dimension activations across models and investigating
differences in consciousness signatures between model sizes/architectures.
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from .circuit import ConsciousnessCircuit, CONSCIOUS_DIMS_V2_1, REFERENCE_HIDDEN_DIM
import json
from datetime import datetime


@dataclass
class DimensionActivation:
    """Raw activation data for a single dimension."""
    dim_name: str
    original_dim: int
    remapped_dim: int
    raw_values: List[float] = field(default_factory=list)
    normalized_values: List[float] = field(default_factory=list)
    
    @property
    def mean_raw(self) -> float:
        return float(np.mean(self.raw_values)) if self.raw_values else 0.0
    
    @property
    def mean_normalized(self) -> float:
        return float(np.mean(self.normalized_values)) if self.normalized_values else 0.0
    
    @property
    def std_normalized(self) -> float:
        return float(np.std(self.normalized_values)) if self.normalized_values else 0.0


@dataclass  
class ModelAnalysis:
    """Complete analysis of a single model."""
    model_name: str
    hidden_dim: int
    dimension_activations: Dict[str, DimensionActivation]
    prompt_scores: List[Dict[str, Any]]
    avg_score: float
    std_score: float
    remapped_dims: Dict[int, Dict]
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())


def analyze_dimension_activations(
    model,
    tokenizer,
    prompts: List[str],
    model_name: str = None,
    circuit: ConsciousnessCircuit = None,
) -> ModelAnalysis:
    """
    Deeply analyze dimension activations for a model across multiple prompts.
    
    This extracts raw activation values for each consciousness dimension,
    allowing comparison of activation patterns across different models.
    
    Args:
        model: HuggingFace model
        tokenizer: HuggingFace tokenizer
        prompts: List of test prompts
        model_name: Name for this model (default: from config)
        circuit: ConsciousnessCircuit instance (default: new v2.1)
        
    Returns:
        ModelAnalysis with detailed activation data
    """
    if circuit is None:
        circuit = ConsciousnessCircuit()
        
    if model_name is None:
        model_name = getattr(model.config, '_name_or_path', 'unknown')
        
    hidden_dim = circuit._get_hidden_size(model)
    remapped_dims = circuit.get_dims_for_hidden_size(hidden_dim)
    
    # Initialize activation trackers
    activations = {}
    for orig_dim, info in CONSCIOUS_DIMS_V2_1.items():
        new_dim = [d for d, i in remapped_dims.items() if i['name'] == info['name']][0]
        activations[info['name']] = DimensionActivation(
            dim_name=info['name'],
            original_dim=orig_dim,
            remapped_dim=new_dim,
        )
    
    prompt_scores = []
    all_scores = []
    
    for prompt in prompts:
        # Get hidden states
        if hasattr(tokenizer, 'apply_chat_template'):
            messages = [{"role": "user", "content": prompt}]
            text = tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True
            )
        else:
            text = prompt
            
        inputs = tokenizer(text, return_tensors="pt")
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True)
            hidden_state = outputs.hidden_states[-1]
        
        # Compute stats for normalization
        mean = hidden_state.mean()
        std = hidden_state.std()
        
        # Extract per-dimension activations
        for dim_name, activation in activations.items():
            dim_idx = activation.remapped_dim
            if dim_idx < hidden_state.shape[-1]:
                raw_val = hidden_state[..., dim_idx].mean().item()
                norm_val = (raw_val - mean.item()) / (std.item() + 1e-8)
                activation.raw_values.append(raw_val)
                activation.normalized_values.append(norm_val)
        
        # Compute consciousness score
        result = circuit.compute(hidden_state, hidden_dim)
        all_scores.append(result.score)
        
        prompt_scores.append({
            'prompt': prompt,
            'score': result.score,
            'raw_score': result.raw_score,
            'contributions': result.dimension_contributions,
        })
    
    return ModelAnalysis(
        model_name=model_name,
        hidden_dim=hidden_dim,
        dimension_activations=activations,
        prompt_scores=prompt_scores,
        avg_score=float(np.mean(all_scores)),
        std_score=float(np.std(all_scores)),
        remapped_dims=remapped_dims,
    )


def compare_models(
    analyses: List[ModelAnalysis],
    output_file: str = None,
) -> Dict[str, Any]:
    """
    Compare consciousness patterns across multiple models.
    
    Args:
        analyses: List of ModelAnalysis objects from different models
        output_file: Optional JSON file to save comparison
        
    Returns:
        Comparison dictionary with cross-model statistics
    """
    comparison = {
        'timestamp': datetime.now().isoformat(),
        'num_models': len(analyses),
        'models': {},
        'dimension_comparison': {},
        'score_comparison': {},
    }
    
    # Collect per-model summaries
    for analysis in analyses:
        model_summary = {
            'hidden_dim': analysis.hidden_dim,
            'avg_score': analysis.avg_score,
            'std_score': analysis.std_score,
            'dimension_means': {
                name: act.mean_normalized 
                for name, act in analysis.dimension_activations.items()
            },
            'dimension_stds': {
                name: act.std_normalized
                for name, act in analysis.dimension_activations.items()
            },
        }
        comparison['models'][analysis.model_name] = model_summary
    
    # Cross-model dimension comparison
    dim_names = list(CONSCIOUS_DIMS_V2_1.values())
    for dim_info in dim_names:
        dim_name = dim_info['name']
        means = []
        stds = []
        for analysis in analyses:
            act = analysis.dimension_activations.get(dim_name)
            if act:
                means.append(act.mean_normalized)
                stds.append(act.std_normalized)
        
        comparison['dimension_comparison'][dim_name] = {
            'cross_model_mean': float(np.mean(means)) if means else None,
            'cross_model_std': float(np.std(means)) if means else None,
            'per_model_means': {
                a.model_name: a.dimension_activations[dim_name].mean_normalized
                for a in analyses if dim_name in a.dimension_activations
            },
        }
    
    # Score comparison
    scores = [a.avg_score for a in analyses]
    comparison['score_comparison'] = {
        'mean_across_models': float(np.mean(scores)),
        'std_across_models': float(np.std(scores)),
        'by_hidden_dim': {
            a.model_name: {'hidden_dim': a.hidden_dim, 'score': a.avg_score}
            for a in analyses
        },
    }
    
    # Check if smaller models score higher
    sorted_by_size = sorted(analyses, key=lambda a: a.hidden_dim)
    if len(sorted_by_size) >= 2:
        smallest = sorted_by_size[0]
        largest = sorted_by_size[-1]
        comparison['size_effect'] = {
            'smallest_model': smallest.model_name,
            'smallest_score': smallest.avg_score,
            'largest_model': largest.model_name,
            'largest_score': largest.avg_score,
            'smaller_scores_higher': smallest.avg_score > largest.avg_score,
        }
    
    if output_file:
        with open(output_file, 'w') as f:
            json.dump(comparison, f, indent=2, default=str)
    
    return comparison


def print_analysis(analysis: ModelAnalysis):
    """Pretty-print a model analysis."""
    print(f"\n{'='*60}")
    print(f"Model: {analysis.model_name}")
    print(f"Hidden dimension: {analysis.hidden_dim}")
    print(f"{'='*60}")
    
    print(f"\nOverall Score: {analysis.avg_score:.3f} ± {analysis.std_score:.3f}")
    
    print(f"\nDimension Activations (normalized):")
    print("-" * 50)
    for name, act in analysis.dimension_activations.items():
        print(f"  {name:<20} mean={act.mean_normalized:+.3f} std={act.std_normalized:.3f}")
    
    print(f"\nPer-Prompt Scores:")
    print("-" * 50)
    for ps in analysis.prompt_scores:
        prompt_short = ps['prompt'][:40] + '...' if len(ps['prompt']) > 40 else ps['prompt']
        print(f"  {ps['score']:.3f} | {prompt_short}")


def print_comparison(comparison: Dict[str, Any]):
    """Pretty-print a model comparison."""
    print(f"\n{'='*70}")
    print("CROSS-MODEL CONSCIOUSNESS COMPARISON")
    print(f"{'='*70}")
    
    print(f"\nModels tested: {comparison['num_models']}")
    print(f"\nScore Summary:")
    print("-" * 50)
    for name, data in comparison['models'].items():
        print(f"  {name:<40} C={data['avg_score']:.3f} ± {data['std_score']:.3f} (dim={data['hidden_dim']})")
    
    if 'size_effect' in comparison:
        se = comparison['size_effect']
        print(f"\nSize Effect Analysis:")
        print("-" * 50)
        print(f"  Smallest: {se['smallest_model']} → {se['smallest_score']:.3f}")
        print(f"  Largest:  {se['largest_model']} → {se['largest_score']:.3f}")
        print(f"  Smaller models score higher: {se['smaller_scores_higher']}")
    
    print(f"\nDimension Consistency (cross-model):")
    print("-" * 50)
    for dim_name, data in comparison['dimension_comparison'].items():
        if data['cross_model_mean'] is not None:
            print(f"  {dim_name:<20} mean={data['cross_model_mean']:+.3f} std={data['cross_model_std']:.3f}")
