#!/usr/bin/env python3
"""
Per-Token Analysis and Visualization
=====================================

Analyze consciousness-like activations at each token position,
visualize trajectories, and create interpretable plots.
"""

import torch
import numpy as np
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple
import json


@dataclass
class TokenTrajectory:
    """Consciousness trajectory across tokens in a sequence."""
    tokens: List[str]                    # Token strings
    scores: List[float]                  # Consciousness score per token
    dimension_activations: Dict[str, List[float]]  # Per-dimension activations
    raw_activations: Optional[np.ndarray] = None   # Full hidden states
    
    @property
    def peak_score(self) -> float:
        """Maximum consciousness score in trajectory."""
        return max(self.scores) if self.scores else 0.0
    
    @property
    def peak_token(self) -> Tuple[int, str]:
        """Index and string of peak token."""
        if not self.scores:
            return (0, "")
        idx = np.argmax(self.scores)
        return (idx, self.tokens[idx])
    
    @property
    def mean_score(self) -> float:
        """Mean consciousness across all tokens."""
        return np.mean(self.scores) if self.scores else 0.0
    
    @property
    def trajectory_slope(self) -> float:
        """Linear trend of consciousness (positive = increasing)."""
        if len(self.scores) < 2:
            return 0.0
        x = np.arange(len(self.scores))
        slope, _ = np.polyfit(x, self.scores, 1)
        return float(slope)
    
    def to_dict(self) -> dict:
        """Export to dictionary."""
        return {
            "tokens": self.tokens,
            "scores": self.scores,
            "dimension_activations": self.dimension_activations,
            "peak_score": self.peak_score,
            "peak_token": self.peak_token,
            "mean_score": self.mean_score,
            "trajectory_slope": self.trajectory_slope,
        }
    
    def plot(self, figsize=(12, 6), save_path: Optional[str] = None):
        """
        Plot consciousness trajectory.
        
        Requires matplotlib (install with: pip install consciousness-circuit[viz])
        """
        try:
            import matplotlib.pyplot as plt
            import matplotlib.patches as mpatches
        except ImportError:
            raise ImportError(
                "Visualization requires matplotlib. "
                "Install with: pip install consciousness-circuit[viz]"
            )
        
        fig, axes = plt.subplots(2, 1, figsize=figsize, height_ratios=[2, 1])
        
        # Top plot: Overall consciousness trajectory
        ax1 = axes[0]
        x = np.arange(len(self.scores))
        
        # Color by score intensity
        colors = plt.cm.RdYlGn(np.array(self.scores))
        ax1.bar(x, self.scores, color=colors, edgecolor='black', linewidth=0.5)
        
        # Add trend line
        if len(self.scores) > 1:
            z = np.polyfit(x, self.scores, 1)
            p = np.poly1d(z)
            ax1.plot(x, p(x), 'b--', linewidth=2, alpha=0.7, label=f'Trend (slope={self.trajectory_slope:.3f})')
        
        # Mark peak
        peak_idx, peak_tok = self.peak_token
        ax1.axvline(x=peak_idx, color='red', linestyle=':', alpha=0.7)
        ax1.annotate(f'Peak: {self.peak_score:.3f}', 
                    xy=(peak_idx, self.peak_score),
                    xytext=(peak_idx + 1, self.peak_score + 0.05),
                    fontsize=10, color='red')
        
        ax1.set_ylabel('Consciousness Score', fontsize=12)
        ax1.set_title('Consciousness Trajectory Across Tokens', fontsize=14, fontweight='bold')
        ax1.set_ylim(0, 1.05)
        ax1.legend(loc='upper right')
        ax1.grid(True, alpha=0.3)
        
        # Bottom plot: Token labels
        ax2 = axes[1]
        ax2.set_xlim(-0.5, len(self.tokens) - 0.5)
        ax2.set_ylim(0, 1)
        
        for i, tok in enumerate(self.tokens):
            # Truncate long tokens
            display_tok = tok[:10] + "..." if len(tok) > 10 else tok
            ax2.text(i, 0.5, display_tok, rotation=45, ha='right', va='center',
                    fontsize=8, fontfamily='monospace')
        
        ax2.set_xticks([])
        ax2.set_yticks([])
        ax2.set_xlabel('Tokens', fontsize=12)
        ax2.axis('off')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
            print(f"Saved plot to {save_path}")
        
        plt.show()
        return fig


# ---------------------------------------------------------------------------
# Logit lens visualization
# ---------------------------------------------------------------------------

def plot_logit_lens_top1(logit_lens_results: List[List[Tuple[str, float]]], title: str = "Logit lens (top-1 per layer)"):
    """Plot the top-1 token probability per layer from logit lens output.

    logit_lens_results: list over layers, each entry is list of (token, prob).
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Visualization requires matplotlib. Install with: pip install consciousness-circuit[viz]")

    top_tokens = [layer_res[0][0] if layer_res else "" for layer_res in logit_lens_results]
    top_probs = [layer_res[0][1] if layer_res else 0.0 for layer_res in logit_lens_results]
    layers = np.arange(len(logit_lens_results))

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(layers, top_probs, marker="o")
    for i, (tok, prob) in enumerate(zip(top_tokens, top_probs)):
        ax.text(i, prob + 0.01, tok, rotation=45, ha="right", va="bottom", fontsize=8)

    ax.set_xlabel("Layer")
    ax.set_ylabel("Top-1 prob")
    ax.set_title(title)
    ax.set_ylim(0, 1.05)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Patch impact heatmap
# ---------------------------------------------------------------------------

def plot_patch_heatmap(layer_metrics: Dict[int, float], title: str = "Patch impact by layer"):
    """Visualize patching results (layer -> metric) as a heatmap/bar hybrid."""
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        raise ImportError("Visualization requires matplotlib and seaborn. Install with: pip install consciousness-circuit[viz]")

    layers = sorted(layer_metrics.keys())
    values = [layer_metrics[l] for l in layers]

    fig, ax = plt.subplots(figsize=(10, 3))
    sns.heatmap(np.array(values)[None, :], cmap="YlGnBu", annot=True, fmt=".3f",
                xticklabels=layers, yticklabels=["metric"], ax=ax, cbar_kws={"label": "target metric"})
    ax.set_xlabel("Layer")
    ax.set_title(title)
    plt.tight_layout()
    plt.show()
    return fig


# ---------------------------------------------------------------------------
# Residual scatter (PCA/UMAP)
# ---------------------------------------------------------------------------

def plot_residual_scatter(residuals: np.ndarray, labels: List[str], method: str = "pca", title: str = "Residual scatter"):
    """Project residuals to 2D for quick inspection.

    residuals: shape (n, d)
    labels: list of strings for coloring
    method: "pca" or "umap" (umap requires umap-learn)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        raise ImportError("Visualization requires matplotlib. Install with: pip install consciousness-circuit[viz]")

    if residuals.ndim != 2:
        raise ValueError("residuals must be 2D (n, d)")

    if method == "pca":
        try:
            from sklearn.decomposition import PCA
        except ImportError:
            raise ImportError("PCA requires scikit-learn. Install with: pip install scikit-learn")
        reducer = PCA(n_components=2)
        emb = reducer.fit_transform(residuals)
    elif method == "umap":
        try:
            import umap  # type: ignore
        except ImportError:
            raise ImportError("UMAP requires umap-learn. Install with: pip install umap-learn")
        reducer = umap.UMAP(n_components=2)
        emb = reducer.fit_transform(residuals)
    else:
        raise ValueError("method must be 'pca' or 'umap'")

    fig, ax = plt.subplots(figsize=(6, 5))
    unique_labels = sorted(set(labels))
    colors = plt.cm.tab10(np.linspace(0, 1, len(unique_labels)))
    for c, lbl in zip(colors, unique_labels):
        mask = [l == lbl for l in labels]
        ax.scatter(emb[mask, 0], emb[mask, 1], s=20, color=c, alpha=0.8, label=lbl)

    ax.set_title(title)
    ax.set_xlabel("dim 1")
    ax.set_ylabel("dim 2")
    ax.legend(loc="best", fontsize=8)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()
    return fig
    
    def plot_dimensions(self, figsize=(14, 8), save_path: Optional[str] = None):
        """
        Plot per-dimension activations across tokens.
        
        Shows how each consciousness dimension contributes at each token.
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            raise ImportError(
                "Visualization requires matplotlib and seaborn. "
                "Install with: pip install consciousness-circuit[viz]"
            )
        
        fig, ax = plt.subplots(figsize=figsize)
        
        # Create heatmap data
        dim_names = list(self.dimension_activations.keys())
        data = np.array([self.dimension_activations[d] for d in dim_names])
        
        # Normalize for visualization
        data_norm = (data - data.min()) / (data.max() - data.min() + 1e-8)
        
        # Create heatmap
        sns.heatmap(data_norm, 
                   xticklabels=[t[:8] for t in self.tokens],
                   yticklabels=dim_names,
                   cmap='RdBu_r',
                   center=0.5,
                   ax=ax,
                   cbar_kws={'label': 'Normalized Activation'})
        
        ax.set_xlabel('Token Position', fontsize=12)
        ax.set_ylabel('Dimension', fontsize=12)
        ax.set_title('Per-Dimension Consciousness Activations', fontsize=14, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        return fig


@dataclass
class ComparisonResult:
    """Compare consciousness across multiple prompts or models."""
    labels: List[str]
    scores: List[float]
    trajectories: List[TokenTrajectory]
    
    def plot_comparison(self, figsize=(10, 6), save_path: Optional[str] = None):
        """Bar chart comparing consciousness scores."""
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            raise ImportError("Visualization requires matplotlib.")
        
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(len(self.labels))
        colors = plt.cm.RdYlGn(np.array(self.scores))
        
        bars = ax.bar(x, self.scores, color=colors, edgecolor='black')
        
        ax.set_xticks(x)
        ax.set_xticklabels(self.labels, rotation=45, ha='right')
        ax.set_ylabel('Consciousness Score', fontsize=12)
        ax.set_title('Consciousness Comparison', fontsize=14, fontweight='bold')
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis='y')
        
        # Add value labels
        for bar, score in zip(bars, self.scores):
            ax.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                   f'{score:.3f}', ha='center', va='bottom', fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        return fig


class ConsciousnessVisualizer:
    """
    High-level visualization tools for consciousness analysis.
    
    Usage:
        viz = ConsciousnessVisualizer(circuit)
        viz.plot_trajectory(model, tokenizer, "Let me think about this...")
        viz.compare_prompts(model, tokenizer, ["prompt1", "prompt2"])
    """
    
    def __init__(self, circuit=None):
        """
        Initialize visualizer.
        
        Args:
            circuit: UniversalCircuit instance (optional, creates one if not provided)
        """
        if circuit is None:
            from .universal import UniversalCircuit
            circuit = UniversalCircuit()
        self.circuit = circuit
    
    def measure_per_token(
        self,
        model,
        tokenizer,
        text: str,
        return_raw: bool = False,
    ) -> TokenTrajectory:
        """
        Measure consciousness at each token position.
        
        Args:
            model: Loaded transformer model
            tokenizer: Model tokenizer
            text: Text to analyze
            return_raw: Also return raw hidden states
            
        Returns:
            TokenTrajectory with per-token scores
        """
        model_name = getattr(model.config, '_name_or_path', 'unknown')
        hidden_size = model.config.hidden_size
        num_layers = model.config.num_hidden_layers
        
        # Get circuit
        circuit_data, method = self.circuit.get_circuit(model_name, hidden_size)
        layer_frac = circuit_data.get("layer_fraction", 0.75)
        target_layer = int(num_layers * layer_frac)
        
        # Tokenize
        inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=512)
        device = next(model.parameters()).device
        inputs = {k: v.to(device) for k, v in inputs.items()}
        
        # Get activations
        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)
        
        hidden = outputs.hidden_states[target_layer][0].cpu().float()  # [seq_len, hidden_size]
        
        # Decode tokens
        tokens = tokenizer.convert_ids_to_tokens(inputs['input_ids'][0].cpu())
        
        # Compute per-token scores
        scores = []
        dimension_activations = {name: [] for name in circuit_data["dimensions"].keys()}
        
        for pos in range(hidden.shape[0]):
            dim_scores = {}
            for name, dim_idx in circuit_data["dimensions"].items():
                if dim_idx < hidden.shape[1]:
                    activation = hidden[pos, dim_idx].item()
                    polarity = circuit_data["polarities"][name]
                    dim_scores[name] = activation * polarity
                    dimension_activations[name].append(dim_scores[name])
            
            if dim_scores:
                raw_score = sum(dim_scores.values()) / len(dim_scores)
                score = 1 / (1 + np.exp(-raw_score))
            else:
                score = 0.5
            scores.append(score)
        
        return TokenTrajectory(
            tokens=tokens,
            scores=scores,
            dimension_activations=dimension_activations,
            raw_activations=hidden.numpy() if return_raw else None,
        )
    
    def compare_prompts(
        self,
        model,
        tokenizer,
        prompts: List[str],
        labels: Optional[List[str]] = None,
    ) -> ComparisonResult:
        """
        Compare consciousness scores across multiple prompts.
        
        Args:
            model: Loaded transformer model
            tokenizer: Model tokenizer
            prompts: List of prompts to compare
            labels: Optional labels (defaults to truncated prompts)
            
        Returns:
            ComparisonResult with scores and optional trajectories
        """
        if labels is None:
            labels = [p[:30] + "..." if len(p) > 30 else p for p in prompts]
        
        trajectories = []
        scores = []
        
        for prompt in prompts:
            traj = self.measure_per_token(model, tokenizer, prompt)
            trajectories.append(traj)
            # Use last token score (validated circuits optimized for this)
            result = self.circuit.measure(model, tokenizer, prompt)
            scores.append(result.score)
        
        return ComparisonResult(
            labels=labels,
            scores=scores,
            trajectories=trajectories,
        )
    
    def plot_model_comparison(
        self,
        models: List[Tuple],  # [(model, tokenizer, name), ...]
        prompts: List[str],
        figsize=(12, 8),
        save_path: Optional[str] = None,
    ):
        """
        Compare consciousness scores across multiple models and prompts.
        
        Creates a grouped bar chart showing how different models score
        on the same prompts.
        """
        try:
            import matplotlib.pyplot as plt
            import numpy as np
        except ImportError:
            raise ImportError("Visualization requires matplotlib.")
        
        n_models = len(models)
        n_prompts = len(prompts)
        
        # Collect scores
        all_scores = []
        model_names = []
        
        for model, tokenizer, name in models:
            model_names.append(name)
            scores = []
            for prompt in prompts:
                result = self.circuit.measure(model, tokenizer, prompt)
                scores.append(result.score)
            all_scores.append(scores)
        
        # Plot
        fig, ax = plt.subplots(figsize=figsize)
        
        x = np.arange(n_prompts)
        width = 0.8 / n_models
        
        for i, (scores, name) in enumerate(zip(all_scores, model_names)):
            offset = (i - n_models/2 + 0.5) * width
            bars = ax.bar(x + offset, scores, width, label=name)
        
        ax.set_ylabel('Consciousness Score', fontsize=12)
        ax.set_title('Cross-Model Consciousness Comparison', fontsize=14, fontweight='bold')
        ax.set_xticks(x)
        ax.set_xticklabels([p[:25]+"..." if len(p)>25 else p for p in prompts], 
                          rotation=45, ha='right')
        ax.legend()
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.3, axis='y')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=150, bbox_inches='tight')
        
        plt.show()
        return fig


def create_interactive_dashboard(trajectories: List[TokenTrajectory], labels: List[str]):
    """
    Create an interactive Plotly dashboard for trajectory analysis.
    
    Requires: pip install consciousness-circuit[viz] (includes plotly)
    """
    try:
        import plotly.graph_objects as go
        from plotly.subplots import make_subplots
    except ImportError:
        raise ImportError(
            "Interactive dashboards require plotly. "
            "Install with: pip install consciousness-circuit[viz]"
        )
    
    fig = make_subplots(
        rows=2, cols=1,
        row_heights=[0.7, 0.3],
        subplot_titles=("Consciousness Trajectories", "Summary Statistics"),
    )
    
    # Add trajectory lines
    for traj, label in zip(trajectories, labels):
        fig.add_trace(
            go.Scatter(
                x=list(range(len(traj.scores))),
                y=traj.scores,
                mode='lines+markers',
                name=label,
                hovertemplate='Token %{x}: %{text}<br>Score: %{y:.3f}',
                text=traj.tokens,
            ),
            row=1, col=1,
        )
    
    # Add summary bars
    fig.add_trace(
        go.Bar(
            x=labels,
            y=[t.mean_score for t in trajectories],
            name='Mean Score',
            marker_color='steelblue',
        ),
        row=2, col=1,
    )
    
    fig.update_layout(
        height=700,
        title_text="Consciousness Circuit Analysis Dashboard",
        showlegend=True,
    )
    
    fig.update_xaxes(title_text="Token Position", row=1, col=1)
    fig.update_yaxes(title_text="Consciousness Score", row=1, col=1, range=[0, 1])
    fig.update_xaxes(title_text="Prompt", row=2, col=1)
    fig.update_yaxes(title_text="Mean Score", row=2, col=1, range=[0, 1])
    
    return fig
