"""
Compressibility Plugin - FULLY STANDALONE
=========================================

Measures how compressible a model's hidden state representations are,
inspired by "Quantifying the Compressibility of the Human Brain"
(Weaver, Faskowitz, Betzel & Lynn, PNAS 2026).

Key insight: When a model uses distributed, multi-dimensional reasoning
(reflective processing), its hidden states should be LESS compressible —
it's utilizing more of its representational capacity. Quick, automatic
responses should be MORE compressible — operating on a smaller subspace.

The paper shows that the human brain is highly compressible (C ~ 0.96):
only ~10% of inter-regional correlations are needed to predict 90% of
neural activity. We apply the same framework to LLM hidden states.

Metrics computed:
  - compressibility: 0-1, how concentrated variance is (higher = more compressible)
  - effective_dimensionality: Number of components for 90% variance
  - participation_ratio: How many dimensions carry meaningful variance
  - spectral_entropy: Uniformity of variance distribution
  - correlation_compression: How few correlations explain the structure

Can be copied to any project - only requires numpy.

Usage:
    from compressibility import CompressibilityPlugin

    plugin = CompressibilityPlugin(max_dims=200)
    results = plugin.analyze(hidden_states)  # [seq_len, hidden_dim]
    print(f"C = {results['compressibility']:.3f}")
    print(f"Effective dims: {results['effective_dimensionality']}")

Dependencies: numpy only
"""

import numpy as np
from typing import Dict, Any, List, Optional

try:
    from .base import AnalysisPlugin
    _HAS_BASE = True
except ImportError:
    _HAS_BASE = False

# Inherit from AnalysisPlugin when used within the package,
# standalone otherwise (numpy-only dependency)
_BaseClass = AnalysisPlugin if _HAS_BASE else object


class CompressibilityPlugin(_BaseClass):
    """
    Measures representational compressibility of hidden states.

    Based on the minimax entropy framework from Weaver et al. (PNAS 2026):
    the brain is highly compressible — only ~10% of correlations explain 90%
    of neural activity. We apply this to LLM hidden states to measure how
    efficiently the model is using its representational space.

    Two complementary analyses:
    1. Eigenvalue-based: How concentrated is the variance spectrum?
       (Fast, always works regardless of seq_len vs hidden_dim)
    2. Correlation-based: How many inter-dimension correlations are needed
       to explain the full correlation structure? (Paper's approach)

    Example:
        >>> plugin = CompressibilityPlugin()
        >>>
        >>> # Analyze hidden states from a model forward pass
        >>> results = plugin.analyze(hidden_states)  # [seq_len, hidden_dim]
        >>> print(f"C = {results['compressibility']:.3f}")
        >>> print(f"Effective dims: {results['effective_dimensionality']}")
        >>> print(f"Participation ratio: {results['participation_ratio']:.1f}")
    """

    def __init__(self,
                 max_dims: int = 200,
                 variance_threshold: float = 0.9,
                 seed: int = 42):
        """
        Initialize compressibility plugin.

        Args:
            max_dims: Maximum dimensions to analyze (subsamples if hidden_dim exceeds this).
                      Paper used 100-200 brain regions; we do the same for tractability.
            variance_threshold: Fraction of variance for effective dimensionality (default 0.9).
            seed: Random seed for dimension subsampling reproducibility.
        """
        if _HAS_BASE:
            super().__init__("compressibility")
        else:
            self.name = "compressibility"
            self.enabled = True
        self.max_dims = max_dims
        self.variance_threshold = variance_threshold
        self.seed = seed

    def analyze(self, hidden_states: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Analyze compressibility of hidden states.

        Args:
            hidden_states: Hidden state trajectories [seq_len, hidden_dim]
            **kwargs: Optional parameters:
                - consciousness_dims: List of dimension indices to check backbone overlap
                - full_correlation: If True, force correlation analysis even if expensive

        Returns:
            Dictionary with compressibility metrics:
              - compressibility: float (0-1), area under compression curve
              - effective_dimensionality: int, components for variance_threshold
              - effective_dim_fraction: float, effective_dim / total dims
              - participation_ratio: float, number of "active" dimensions
              - pr_fraction: float, participation_ratio / total dims
              - spectral_entropy: float (0-1), uniformity of eigenvalue distribution
              - top1_variance_fraction: float, variance in first component
              - top5_variance_fraction: float, variance in top 5 components
              - top10_variance_fraction: float, variance in top 10 components
              - n_dims_analyzed: int, dimensions used in analysis
              - seq_len: int, sequence length
              - eigenvalue_spectrum: list, top eigenvalues
              - compression_curve: dict with fractions and entropy arrays
              - correlation_compression: dict (if computed), correlation-based metrics
              - backbone_overlap: dict (if consciousness_dims provided)
        """
        if hidden_states.ndim != 2:
            return self._empty_result(0)

        seq_len, hidden_dim = hidden_states.shape

        if seq_len < 3 or hidden_dim < 2:
            return self._empty_result(hidden_dim)

        # Subsample dimensions if needed for tractability
        if hidden_dim > self.max_dims:
            rng = np.random.RandomState(self.seed)
            dim_indices = np.sort(rng.choice(hidden_dim, self.max_dims, replace=False))
            states = hidden_states[:, dim_indices]
        else:
            dim_indices = np.arange(hidden_dim)
            states = hidden_states

        n_dims = states.shape[1]

        # Center the data
        states_centered = states - states.mean(axis=0, keepdims=True)

        # Compute eigenvalues of the covariance matrix
        eigenvalues = self._compute_eigenvalues(states_centered, seq_len, n_dims)

        # Sort descending, clip numerical noise
        eigenvalues = np.sort(np.maximum(eigenvalues, 0))[::-1]
        eigenvalues = eigenvalues[eigenvalues > 1e-12]

        if len(eigenvalues) == 0:
            return self._empty_result(hidden_dim)

        # === 1. Eigenvalue-based compressibility ===
        total_var = eigenvalues.sum()
        cumvar = np.cumsum(eigenvalues) / total_var

        # Compression curve: normalized entropy S̃(f) = 1 - cumulative_variance(f)
        n_eig = len(eigenvalues)
        fractions = np.arange(1, n_eig + 1) / n_eig
        normalized_entropy = 1.0 - cumvar

        # Compressibility C = area above compression curve
        # C → 1 for perfectly compressible (one eigenvalue dominates)
        # C → 0 for incompressible (uniform eigenvalues)
        compressibility = float(1.0 - np.trapezoid(normalized_entropy, fractions))

        # === 2. Effective dimensionality ===
        effective_dim = int(np.searchsorted(cumvar, self.variance_threshold) + 1)
        effective_dim = min(effective_dim, n_eig)
        effective_dim_fraction = effective_dim / n_eig

        # === 3. Participation ratio: PR = (Σλ)² / Σλ² ===
        # Measures how many dimensions carry meaningful variance
        participation_ratio = float(total_var ** 2 / np.sum(eigenvalues ** 2))
        pr_fraction = participation_ratio / n_eig

        # === 4. Spectral entropy (Shannon entropy of normalized eigenvalues) ===
        p = eigenvalues / total_var
        p = p[p > 0]
        spectral_entropy = float(-np.sum(p * np.log(p)))
        max_entropy = np.log(len(p))
        normalized_spectral_entropy = float(
            spectral_entropy / max_entropy if max_entropy > 0 else 0
        )

        # === 5. Key variance fractions ===
        top1_frac = float(eigenvalues[0] / total_var)
        top5_frac = float(eigenvalues[:min(5, n_eig)].sum() / total_var)
        top10_frac = float(eigenvalues[:min(10, n_eig)].sum() / total_var)

        # === 6. Correlation-based compression (paper's approach) ===
        correlation_analysis = {}
        do_correlation = kwargs.get("full_correlation", False)
        if n_dims <= 500 and seq_len >= max(10, n_dims // 5):
            do_correlation = True
        if do_correlation:
            correlation_analysis = self._correlation_compression(
                states_centered, seq_len, n_dims
            )

        # === 7. Backbone overlap with consciousness dimensions ===
        backbone_overlap = {}
        consciousness_dims = kwargs.get("consciousness_dims", None)
        if consciousness_dims is not None and len(eigenvalues) > 0:
            backbone_overlap = self._check_backbone_overlap(
                states_centered, dim_indices, consciousness_dims, eigenvalues
            )

        # Subsample arrays for compact output
        step = max(1, n_eig // 50)

        result = {
            "compressibility": compressibility,
            "effective_dimensionality": int(effective_dim),
            "effective_dim_fraction": float(effective_dim_fraction),
            "participation_ratio": float(participation_ratio),
            "pr_fraction": float(pr_fraction),
            "spectral_entropy": float(normalized_spectral_entropy),
            "top1_variance_fraction": top1_frac,
            "top5_variance_fraction": top5_frac,
            "top10_variance_fraction": top10_frac,
            "n_eigenvalues": int(n_eig),
            "n_dims_analyzed": int(n_dims),
            "original_hidden_dim": int(hidden_dim),
            "seq_len": int(seq_len),
            "eigenvalue_spectrum": eigenvalues[:min(50, n_eig)].tolist(),
            "compression_curve": {
                "fractions": fractions[::step].tolist(),
                "normalized_entropy": normalized_entropy[::step].tolist(),
            },
        }

        if correlation_analysis:
            result["correlation_compression"] = correlation_analysis

        if backbone_overlap:
            result["backbone_overlap"] = backbone_overlap

        return result

    def _compute_eigenvalues(
        self, states_centered: np.ndarray, seq_len: int, n_dims: int
    ) -> np.ndarray:
        """
        Compute eigenvalues of the covariance matrix efficiently.
        Uses Gram matrix trick when seq_len < n_dims.
        """
        if seq_len >= n_dims:
            # Standard covariance: [n_dims, n_dims]
            cov = np.cov(states_centered, rowvar=False)
            return np.linalg.eigvalsh(cov)
        else:
            # Gram matrix approach: [seq_len, seq_len] — much smaller
            # The non-zero eigenvalues of X.T @ X / (n-1) equal those of X @ X.T / (n-1)
            gram = states_centered @ states_centered.T / max(seq_len - 1, 1)
            return np.linalg.eigvalsh(gram)

    def _correlation_compression(
        self,
        states_centered: np.ndarray,
        seq_len: int,
        n_dims: int,
    ) -> Dict[str, Any]:
        """
        Correlation-based compression analysis (Weaver et al. approach).

        Estimates how many inter-dimension correlations are needed to explain
        the full correlation structure, using the weak-correlation approximation:
            ΔS ≈ -½ ln(1 - ρ²)

        The correlations are added greedily by absolute strength, and we
        track cumulative entropy reduction.
        """
        # Compute correlation matrix
        stds = np.std(states_centered, axis=0)
        stds[stds < 1e-12] = 1.0
        states_norm = states_centered / stds
        corr = states_norm.T @ states_norm / max(seq_len - 1, 1)
        np.fill_diagonal(corr, 1.0)

        # Extract upper triangle (unique correlations)
        i_upper, j_upper = np.triu_indices(n_dims, k=1)
        correlations = corr[i_upper, j_upper]
        n_corr = len(correlations)

        if n_corr == 0:
            return {}

        abs_corr = np.abs(correlations)

        # Sort by absolute correlation (strongest first)
        sort_idx = np.argsort(abs_corr)[::-1]
        sorted_abs = abs_corr[sort_idx]

        # Entropy reduction per correlation (weak-correlation approximation)
        rho_sq = np.clip(sorted_abs ** 2, 0, 0.9999)
        delta_s = -0.5 * np.log(1.0 - rho_sq)

        total_delta = delta_s.sum()

        if total_delta < 1e-12:
            return {
                "compressibility_corr": 0.0,
                "n_correlations": int(n_corr),
            }

        # Cumulative entropy reduction curve
        cum_reduction = np.cumsum(delta_s) / total_delta
        fractions = np.arange(1, n_corr + 1) / n_corr

        # Compressibility from correlation curve (area under cumulative reduction)
        c_corr = float(np.trapezoid(cum_reduction, fractions))

        # How many correlations for 50% and 90% reduction
        idx_50 = int(np.searchsorted(cum_reduction, 0.5) + 1)
        idx_90 = int(np.searchsorted(cum_reduction, 0.9) + 1)

        return {
            "compressibility_corr": c_corr,
            "n_correlations": int(n_corr),
            "fraction_for_50pct": float(min(idx_50 / n_corr, 1.0)),
            "fraction_for_90pct": float(min(idx_90 / n_corr, 1.0)),
            "mean_abs_correlation": float(abs_corr.mean()),
            "max_abs_correlation": float(abs_corr.max()),
            "median_abs_correlation": float(np.median(abs_corr)),
            "strong_correlations_pct": float((abs_corr > 0.3).mean() * 100),
        }

    def _check_backbone_overlap(
        self,
        states_centered: np.ndarray,
        dim_indices: np.ndarray,
        consciousness_dims: List[int],
        eigenvalues: np.ndarray,
    ) -> Dict[str, Any]:
        """
        Check whether consciousness dimensions are part of the sparse backbone.

        Computes how much variance the consciousness dimensions contribute
        to the top principal components vs random dimensions.
        """
        n_eig = len(eigenvalues)
        total_var = eigenvalues.sum()

        # Which of the subsampled dims are consciousness dims?
        c_dim_mask = np.isin(dim_indices, consciousness_dims)
        n_c_dims_present = int(c_dim_mask.sum())

        if n_c_dims_present == 0:
            return {"consciousness_dims_in_sample": 0}

        # Compute variance contributed by consciousness dims
        c_dim_states = states_centered[:, c_dim_mask]
        c_var = float(np.var(c_dim_states, axis=0).sum())

        # Compare to same number of random dims
        non_c_mask = ~c_dim_mask
        if non_c_mask.sum() >= n_c_dims_present:
            rng = np.random.RandomState(self.seed + 1)
            random_idx = rng.choice(
                np.where(non_c_mask)[0], n_c_dims_present, replace=False
            )
            random_var = float(np.var(states_centered[:, random_idx], axis=0).sum())
        else:
            random_var = c_var

        # Variance ratio: how much more/less variance consciousness dims carry
        variance_ratio = float(c_var / random_var) if random_var > 0 else 1.0

        return {
            "consciousness_dims_in_sample": n_c_dims_present,
            "consciousness_dims_variance": c_var,
            "random_dims_variance": random_var,
            "variance_ratio": variance_ratio,
            "in_backbone": variance_ratio > 1.0,
        }

    def _empty_result(self, hidden_dim: int) -> Dict[str, Any]:
        """Return empty result for degenerate inputs."""
        return {
            "compressibility": 0.0,
            "effective_dimensionality": hidden_dim,
            "effective_dim_fraction": 1.0,
            "participation_ratio": float(hidden_dim),
            "pr_fraction": 1.0,
            "spectral_entropy": 1.0,
            "top1_variance_fraction": 0.0,
            "top5_variance_fraction": 0.0,
            "top10_variance_fraction": 0.0,
            "n_eigenvalues": 0,
            "n_dims_analyzed": 0,
            "original_hidden_dim": hidden_dim,
            "seq_len": 0,
            "eigenvalue_spectrum": [],
            "compression_curve": {"fractions": [], "normalized_entropy": []},
        }

    def analyze_bottleneck(self, all_hidden_states: list, **kwargs) -> Dict[str, Any]:
        """
        Analyze the compression bottleneck across all layers.

        Layer sweep analysis (Feb 2026) revealed that both Qwen and Mistral
        compress representations to ~1 effective dimension in their middle
        layers, but with very different profiles:
          - Qwen: abrupt collapse at L4, flat until L27, sudden decompression
          - Mistral: early compression at L2, gradual expansion through L32

        This method characterizes the bottleneck shape for any model.

        Args:
            all_hidden_states: List of hidden state arrays, one per layer.
                               Each is [seq_len, hidden_dim]. Index 0 = embedding layer.
            **kwargs: Optional parameters

        Returns:
            Dictionary with bottleneck metrics:
              - bottleneck_layer: int, layer with minimum participation ratio
              - bottleneck_depth_pct: float, depth of bottleneck as % of total
              - bottleneck_pr: float, participation ratio at bottleneck
              - bottleneck_top1: float, top-1 eigenvalue fraction at bottleneck
              - compression_ratio: float, max_pr / min_pr
              - decompression_layer: int, where PR starts expanding (>2x bottleneck)
              - bottleneck_width: int, number of layers within 2x of min PR
              - phase: str, 'abrupt' if compression happens in <=2 layers, 'gradual' otherwise
              - layer_pr: list of (layer_idx, pr) tuples
              - optimal_analysis_layer: int, recommended layer for compressibility analysis
        """
        if len(all_hidden_states) < 3:
            return {"error": "Need at least 3 layers"}

        # Skip embedding layer (index 0), analyze transformer layers
        num_layers = len(all_hidden_states) - 1
        layer_prs = []

        for layer_idx in range(1, len(all_hidden_states)):
            h = all_hidden_states[layer_idx]
            if hasattr(h, 'cpu'):
                h = h.cpu().float().numpy()
            if h.ndim == 3:
                h = h[0]  # remove batch dim

            seq_len, hidden_dim = h.shape
            if seq_len < 3:
                layer_prs.append((layer_idx, float(hidden_dim)))
                continue

            centered = h - h.mean(axis=0, keepdims=True)

            # Use efficient SVD for eigenvalues
            try:
                if seq_len >= hidden_dim:
                    cov = np.cov(centered, rowvar=False)
                    eigenvalues = np.linalg.eigvalsh(cov)
                else:
                    gram = centered @ centered.T / max(seq_len - 1, 1)
                    eigenvalues = np.linalg.eigvalsh(gram)

                eigenvalues = np.sort(np.maximum(eigenvalues, 0))[::-1]
                eigenvalues = eigenvalues[eigenvalues > 1e-12]

                if len(eigenvalues) > 0:
                    total_var = eigenvalues.sum()
                    pr = float(total_var ** 2 / np.sum(eigenvalues ** 2))
                    top1 = float(eigenvalues[0] / total_var)
                else:
                    pr = 0.0
                    top1 = 0.0
            except Exception:
                pr = float(hidden_dim)
                top1 = 0.0

            layer_prs.append((layer_idx, pr, top1))

        if not layer_prs:
            return {"error": "No valid layers"}

        # Extract arrays
        layers = [x[0] for x in layer_prs]
        prs = np.array([x[1] for x in layer_prs])
        top1s = np.array([x[2] if len(x) > 2 else 0 for x in layer_prs])

        # Bottleneck detection
        min_idx = np.argmin(prs)
        max_idx = np.argmax(prs)
        min_pr = prs[min_idx]
        max_pr = prs[max_idx]

        # Compression ratio
        compression_ratio = float(max_pr / min_pr) if min_pr > 0 else float('inf')

        # Bottleneck width: consecutive layers within 2x of minimum
        threshold = min_pr * 2
        in_bottleneck = prs < threshold
        width = int(in_bottleneck.sum())

        # Phase detection: how quickly does compression happen?
        # Look at PR change rate in the first quarter
        quarter = max(1, len(prs) // 4)
        pr_first_quarter = prs[:quarter]
        if len(pr_first_quarter) >= 2:
            pr_drop = (pr_first_quarter[0] - pr_first_quarter[-1]) / (pr_first_quarter[0] + 1e-10)
            phase = "abrupt" if pr_drop > 0.8 else "gradual"
        else:
            phase = "unknown"

        # Decompression layer: first layer after bottleneck where PR > 2x min
        decompression_layer = None
        for i in range(min_idx + 1, len(prs)):
            if prs[i] > threshold:
                decompression_layer = layers[i]
                break

        # Optimal analysis layer: where CS-SE correlation is likely strongest
        # Based on our findings: the transition zone between compressed and expanded
        # For abrupt bottlenecks: just before decompression
        # For gradual expansion: the midpoint of the expansion phase
        if decompression_layer is not None:
            optimal_layer = layers[min(min_idx + (decompression_layer - layers[min_idx]) // 2,
                                       len(layers) - 1)]
        else:
            optimal_layer = layers[int(len(layers) * 0.56)]  # default to 56% (Mistral peak)

        return {
            "bottleneck_layer": layers[min_idx],
            "bottleneck_depth_pct": float(layers[min_idx] / num_layers * 100),
            "bottleneck_pr": float(min_pr),
            "bottleneck_top1": float(top1s[min_idx]),
            "max_pr": float(max_pr),
            "max_pr_layer": layers[max_idx],
            "compression_ratio": compression_ratio,
            "bottleneck_width": width,
            "decompression_layer": decompression_layer,
            "phase": phase,
            "optimal_analysis_layer": optimal_layer,
            "num_layers": num_layers,
            "layer_pr": [(int(l), float(p)) for l, p in zip(layers, prs)],
        }

    def reset(self):
        """Reset plugin state (no-op for analysis plugin)."""
        pass

    def __repr__(self):
        return (
            f"CompressibilityPlugin("
            f"max_dims={self.max_dims}, "
            f"variance_threshold={self.variance_threshold})"
        )


__all__ = ["CompressibilityPlugin"]


if __name__ == "__main__":
    # Self-test
    print("Compressibility Plugin - Standalone Tests")
    print("=" * 55)

    np.random.seed(42)

    # Test 1: Highly compressible signal (low-rank)
    print("\n1. Highly Compressible (rank-3 in 100 dims):")
    n_tokens, n_dims = 50, 100
    # Create rank-3 data: most variance in 3 directions
    basis = np.random.randn(3, n_dims)
    coeffs = np.random.randn(n_tokens, 3) * np.array([10, 5, 2])
    states_compressed = coeffs @ basis + np.random.randn(n_tokens, n_dims) * 0.1

    plugin = CompressibilityPlugin(max_dims=100)
    result = plugin.analyze(states_compressed)
    print(f"   Compressibility:  C = {result['compressibility']:.3f}")
    print(f"   Effective dims:   {result['effective_dimensionality']} / {result['n_dims_analyzed']}")
    print(f"   Participation:    {result['participation_ratio']:.1f}")
    print(f"   Spectral entropy: {result['spectral_entropy']:.3f}")
    print(f"   Top 5 variance:   {result['top5_variance_fraction']:.1%}")
    assert result["compressibility"] > 0.8, "Low-rank data should be highly compressible"

    # Test 2: Incompressible signal (full-rank, uniform variance)
    print("\n2. Incompressible (isotropic noise):")
    states_random = np.random.randn(n_tokens, n_dims)

    result2 = plugin.analyze(states_random)
    print(f"   Compressibility:  C = {result2['compressibility']:.3f}")
    print(f"   Effective dims:   {result2['effective_dimensionality']} / {result2['n_dims_analyzed']}")
    print(f"   Participation:    {result2['participation_ratio']:.1f}")
    print(f"   Spectral entropy: {result2['spectral_entropy']:.3f}")
    print(f"   Top 5 variance:   {result2['top5_variance_fraction']:.1%}")
    assert result2["compressibility"] < result["compressibility"], \
        "Random data should be less compressible than low-rank"

    # Test 3: Correlation-based compression
    print("\n3. Correlation Compression:")
    if "correlation_compression" in result:
        cc = result["correlation_compression"]
        print(f"   Correlation C:    {cc['compressibility_corr']:.3f}")
        print(f"   50% reduction at: {cc['fraction_for_50pct']:.1%} of correlations")
        print(f"   90% reduction at: {cc['fraction_for_90pct']:.1%} of correlations")
        print(f"   Mean |corr|:      {cc['mean_abs_correlation']:.3f}")
        print(f"   Strong (>0.3):    {cc['strong_correlations_pct']:.1f}%")
    else:
        print("   Skipped (not enough tokens for dims)")

    # Test 4: High-dimensional subsampling
    print("\n4. High-Dimensional (3584 dims, subsampled):")
    states_big = np.random.randn(30, 3584)
    # Inject structure in first 10 dims
    states_big[:, :10] = np.random.randn(30, 1) * np.random.randn(1, 10) * 5

    result3 = plugin.analyze(states_big)
    print(f"   Analyzed dims:    {result3['n_dims_analyzed']} / {result3['original_hidden_dim']}")
    print(f"   Compressibility:  C = {result3['compressibility']:.3f}")
    print(f"   Effective dims:   {result3['effective_dimensionality']}")

    # Test 5: Backbone overlap with consciousness dims
    print("\n5. Backbone Overlap Check:")
    c_dims = [0, 1, 2, 3, 4]  # First 5 dims (where we injected structure)
    result4 = plugin.analyze(states_big, consciousness_dims=c_dims)
    if "backbone_overlap" in result4:
        bo = result4["backbone_overlap"]
        print(f"   C-dims in sample: {bo['consciousness_dims_in_sample']}")
        if bo["consciousness_dims_in_sample"] > 0:
            print(f"   C-dims variance:  {bo['consciousness_dims_variance']:.3f}")
            print(f"   Random variance:  {bo['random_dims_variance']:.3f}")
            print(f"   Variance ratio:   {bo['variance_ratio']:.2f}x")
            print(f"   In backbone:      {bo['in_backbone']}")
    else:
        print("   Not computed")

    # Test 6: Edge cases
    print("\n6. Edge Cases:")
    tiny = np.random.randn(2, 5)
    result_tiny = plugin.analyze(tiny)
    print(f"   Tiny (2x5):       C = {result_tiny['compressibility']:.3f}")

    single = np.random.randn(1, 10)
    result_single = plugin.analyze(single)
    print(f"   Single token:     C = {result_single['compressibility']:.3f} (degenerate)")

    print("\n" + "=" * 55)
    print("All tests completed successfully!")
