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

import os
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
    method: str  # "discovered", "remapped", or "ensemble"
    dimension_scores: Dict[str, float]
    model_name: str
    circuit_path: Optional[str]
    confidence: float = 1.0
    # Extended analysis (optional)
    hidden_states: Optional[list] = None
    logit_lens: Optional[list] = None
    target_layer: Optional[int] = None
    # Ensemble info (optional)
    ensemble_layers: Optional[list] = None
    ensemble_scores: Optional[Dict[int, float]] = None
    # Quality metrics (v3.2)
    raw_score: Optional[float] = None  # Score before length normalization
    token_count: Optional[int] = None
    dimension_diversity: Optional[float] = None  # 0-1, how evenly distributed dimensions are
    dominant_dimension: Optional[str] = None  # Which dimension dominates the signal
    anomaly_flags: Optional[list] = None  # List of detected anomalies

    @property
    def dimension_contributions(self) -> Dict[str, float]:
        """Alias for dimension_scores for API compatibility."""
        return self.dimension_scores


def get_adaptive_layer_fraction(num_layers: int, model_name: str = "") -> float:
    """
    Get optimal layer fraction based on model depth and architecture.

    Layer sweep analysis (Feb 2026) across Mistral-7B and Qwen-7B revealed that
    the CS-SE correlation peaks at architecture-specific depths:
      - Mistral: peak at ~56% depth (L18/32), significant window 50-59%
      - Qwen: representations are maximally compressed at 14-93% depth (SE≈0.005),
              with the CS-SE link spread weakly across the compressed zone
      - General: the old 75% heuristic misses the peak in most architectures

    Args:
        num_layers: Number of transformer layers in the model
        model_name: Model name/path for architecture-specific selection

    Returns:
        Optimal layer fraction (0.0 to 1.0)
    """
    model_lower = model_name.lower()

    # Architecture-specific fractions based on layer sweep data
    if "mistral" in model_lower:
        # Peak CS-SE at 56% depth (L18/32), significant zone L16-L19
        if num_layers <= 32:
            return 0.56
        elif num_layers <= 48:
            return 0.54
        else:
            return 0.52
    elif "qwen" in model_lower:
        # Qwen compresses representations to SE≈0.005 at L4-L26 (28 layers)
        # The CS-SE link is weak but steadily grows through the compressed zone
        # Use late-compressed zone where SE starts to recover (~85% depth)
        if num_layers <= 28:
            return 0.85
        elif num_layers <= 48:
            return 0.80
        else:
            return 0.75
    elif "yi" in model_lower:
        # Yi architecture is similar to Llama; use mid-depth
        if num_layers <= 32:
            return 0.65
        else:
            return 0.60
    elif "llama" in model_lower:
        if num_layers <= 32:
            return 0.65
        else:
            return 0.60

    # General fallback heuristic
    if num_layers <= 12:
        return 0.75  # Small models (GPT-2, etc.)
    elif num_layers <= 24:
        return 0.65  # Medium models
    elif num_layers <= 32:
        return 0.60  # Large models (7B-14B range)
    elif num_layers <= 48:
        return 0.58  # Very large models
    elif num_layers <= 64:
        return 0.55  # 32B+ models
    else:
        return 0.52  # 70B+ models


def get_ensemble_layers(num_layers: int, n_layers: int = 3) -> list:
    """
    Get layer indices for ensemble measurement.

    Returns layers at different depths to capture consciousness at multiple stages.

    Args:
        num_layers: Total number of transformer layers
        n_layers: Number of layers to include in ensemble (default: 3)

    Returns:
        List of layer indices
    """
    if n_layers == 3:
        # Early-middle, middle, late-middle
        fractions = [0.55, 0.70, 0.85]
    elif n_layers == 5:
        fractions = [0.50, 0.60, 0.70, 0.80, 0.90]
    else:
        # Distribute evenly in the 50-90% range
        fractions = [0.50 + (0.40 * i / (n_layers - 1)) for i in range(n_layers)]

    return [int(num_layers * f) for f in fractions]


# ============================================================================
# Scoring Improvements v3.2
# ============================================================================

# Minimum tokens for reliable measurement
MIN_RELIABLE_TOKENS = 4

# Length normalization curve: maps token count to a scaling factor
# Short prompts get boosted less (they genuinely have less context),
# but we remove the penalty floor that makes 1-token prompts score ~0.1
def length_normalization_factor(n_tokens: int) -> float:
    """
    Compute length-based score adjustment.

    The problem: longer prompts score higher simply because they have
    more tokens activating the network. This function adjusts scores
    to compensate.

    Approach: We model the expected "baseline inflation" as a function
    of token count and subtract it. The adjustment is ADDITIVE, not
    multiplicative, because the bias is additive (more tokens = higher
    floor, not higher multiplier).

    Based on experimental data:
    - 1-token prompts: ~0.12 average (mostly embedding noise)
    - 5-token prompts: ~0.35 average
    - 10-token prompts: ~0.45 average
    - 20+ tokens: ~0.50 (stabilizes)

    We return the expected baseline score for n tokens. The caller
    subtracts this and re-centers.

    Args:
        n_tokens: Number of tokens in the input

    Returns:
        Expected baseline score for this token count
    """
    # Logarithmic baseline: rises quickly then plateaus
    # baseline = floor + scale * (1 - exp(-n/tau))
    floor = 0.08   # Minimum baseline (even 1 token has some activation)
    ceiling = 0.48  # Where baseline stabilizes for long inputs
    tau = 5.0       # Ramp speed
    baseline = floor + (ceiling - floor) * (1 - np.exp(-n_tokens / tau))
    return float(baseline)


def compute_dimension_diversity(dim_scores: Dict[str, float]) -> Tuple[float, Optional[str]]:
    """
    Measure how evenly the consciousness signal is distributed across dimensions.

    Returns a diversity score (0 = one dimension dominates entirely,
    1 = all dimensions contribute equally) and the name of the dominant dimension.

    Uses normalized entropy: H / H_max where H = -sum(p_i * log(p_i))

    Args:
        dim_scores: Dictionary of dimension name -> activation value

    Returns:
        (diversity_score, dominant_dimension_name)
    """
    if not dim_scores:
        return 1.0, None

    abs_scores = {k: abs(v) for k, v in dim_scores.items()}
    total = sum(abs_scores.values())

    if total < 1e-10:
        return 1.0, None

    # Normalize to probability distribution
    probs = [v / total for v in abs_scores.values()]

    # Shannon entropy
    entropy = -sum(p * np.log(p + 1e-10) for p in probs)
    max_entropy = np.log(len(probs))

    if max_entropy < 1e-10:
        return 1.0, None

    diversity = float(entropy / max_entropy)

    # Find dominant dimension
    dominant = max(abs_scores, key=abs_scores.get)

    return diversity, dominant


def detect_anomalies(
    token_scores: list,
    dim_scores: Dict[str, float],
    n_tokens: int,
) -> list:
    """
    Detect anomalous activation patterns that suggest unreliable measurement.

    Checks for:
    1. Single-token inputs with extreme scores (embedding artifacts)
    2. Token scores with very high variance (unstable activations)
    3. All dimensions near-zero (dead activation)

    Args:
        token_scores: Per-token consciousness scores
        dim_scores: Dimension contribution scores
        n_tokens: Number of input tokens

    Returns:
        List of anomaly flag strings (empty = no anomalies)
    """
    flags = []

    # 1. Single-token extreme score
    # A single token shouldn't score above 0.55 or below 0.15 reliably
    if n_tokens <= 2 and token_scores:
        max_score = max(token_scores)
        min_score = min(token_scores)
        if max_score > 0.55 or min_score < 0.15:
            flags.append("single_token_extreme")

    # 2. High variance across tokens suggests instability
    if len(token_scores) > 2:
        std = float(np.std(token_scores))
        if std > 0.2:
            flags.append("high_token_variance")

    # 3. Dead activation - all dimensions near zero
    if dim_scores:
        max_abs = max(abs(v) for v in dim_scores.values())
        if max_abs < 0.05:
            flags.append("dead_activation")

    return flags


def compute_confidence(
    n_tokens: int,
    diversity: float,
    anomalies: list,
) -> float:
    """
    Compute measurement confidence score.

    Combines token count reliability, dimension diversity, and anomaly detection
    into a single confidence value.

    Args:
        n_tokens: Number of input tokens
        diversity: Dimension diversity score (0-1)
        anomalies: List of detected anomaly flags

    Returns:
        Confidence score in [0.0, 1.0]
    """
    # Token count confidence: ramps from 0.3 (1 token) to 1.0 (10+ tokens)
    token_conf = min(1.0, 0.3 + 0.7 * (n_tokens / 10.0))

    # Diversity confidence: low diversity = less reliable
    div_conf = 0.5 + 0.5 * diversity

    # Anomaly penalty: each anomaly reduces confidence
    anomaly_penalty = 0.15 * len(anomalies)

    confidence = (token_conf * 0.5 + div_conf * 0.5) - anomaly_penalty
    return max(0.0, min(1.0, confidence))


def entropy_weight_tokens(
    hidden_seq: 'torch.Tensor',
) -> np.ndarray:
    """
    Compute per-token weights based on activation entropy.

    Tokens with higher entropy (more information content) in their hidden
    state get weighted more heavily. This reduces the influence of padding-like
    or repetitive tokens.

    Args:
        hidden_seq: Hidden states tensor [seq_len, hidden_size]

    Returns:
        Normalized weight array of shape [seq_len], sums to 1.0
    """
    seq_len = hidden_seq.shape[0]
    if seq_len <= 1:
        return np.ones(1)

    entropies = []
    for pos in range(seq_len):
        h = hidden_seq[pos].float()
        # Compute entropy of absolute activation distribution
        abs_h = torch.abs(h)
        total = abs_h.sum()
        if total < 1e-10:
            entropies.append(0.0)
            continue
        probs = abs_h / total
        # Clamp for numerical stability
        probs = torch.clamp(probs, min=1e-10)
        ent = -torch.sum(probs * torch.log(probs)).item()
        entropies.append(ent)

    entropies = np.array(entropies)

    # Normalize to weights that sum to 1
    if entropies.sum() < 1e-10:
        return np.ones(seq_len) / seq_len

    weights = entropies / entropies.sum()
    return weights


def normalize_dimensions_adaptive(
    hidden_seq: 'torch.Tensor',
    circuit: Dict[str, Any],
    scale: float = 0.5,
) -> Dict[str, np.ndarray]:
    """
    v3.4: Two-pass per-dimension adaptive normalization.

    Replaces the v3.3 global z-score + hard clamp with a per-dimension
    approach that eliminates ceiling effects.

    Pass 1: Compute z-score for each consciousness dimension at every
            token position (relative to the full hidden-state distribution
            at that position — this provides model-size independence).
    Pass 2: Re-normalize each dimension against its own mean/std across
            the sequence, then apply tanh for smooth bounding.

    The result is that every dimension has meaningful variance regardless
    of its absolute z-score level. Dimensions like Logic (which previously
    pegged at +3.0 for every math token) now show within-sequence variation.

    Args:
        hidden_seq: Hidden states [seq_len, hidden_size]
        circuit: Circuit dict with "dimensions" and "polarities"
        scale: tanh scale factor (default 0.5). Controls sensitivity:
               tanh(1*0.5)=0.46, tanh(2*0.5)=0.76, tanh(3*0.5)=0.91

    Returns:
        Dict mapping dimension name -> np.array of shape [seq_len]
        with normalized, polarity-applied values in roughly (-1, +1).
    """
    seq_len = hidden_seq.shape[0]

    # Pass 1: raw z-scores per dimension per position
    dim_z_raw: Dict[str, list] = {name: [] for name in circuit["dimensions"]}
    for pos in range(seq_len):
        hidden = hidden_seq[pos]
        h_mean = hidden.mean().item()
        h_std = hidden.std().item()
        for name, dim_idx in circuit["dimensions"].items():
            if dim_idx < len(hidden):
                z = (hidden[dim_idx].item() - h_mean) / (h_std + 1e-8)
            else:
                z = 0.0
            dim_z_raw[name].append(z)

    # Pass 2: per-dimension re-normalization + tanh
    dim_normalized: Dict[str, np.ndarray] = {}
    for name in circuit["dimensions"]:
        z_arr = np.array(dim_z_raw[name])
        z_mean = z_arr.mean()
        z_std = z_arr.std()
        # Floor std to avoid amplifying noise when a dimension is truly flat
        z_std = max(z_std, 0.1)
        # Re-center against this dimension's own baseline in this sequence
        renormed = (z_arr - z_mean) / z_std
        # Smooth bounding (no hard ceiling)
        bounded = np.tanh(renormed * scale)
        # Apply polarity
        polarity = circuit["polarities"][name]
        dim_normalized[name] = bounded * polarity

    return dim_normalized


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
        # Qwen2.5-32B reference circuit (v2.1 with dimension fix)
        "Qwen/Qwen2.5-32B-Instruct": {
            "dimensions": {
                "Logic": 3183, "Self-Reflective": 212, "Self-Expression": 5064,  # Fixed: was 5065, out of bounds
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
        num_threads: Optional[int] = None,
    ):
        """
        Initialize universal circuit.
        
        Args:
            circuits_dir: Directory to look for/save discovered circuits
            enable_discovery: Whether to allow on-the-fly discovery
            cache_hidden_states: Cache hidden states for efficiency
            num_threads: Optional override for torch CPU threads (helps high-core CPUs)
        """
        self.circuits_dir = Path(circuits_dir) if circuits_dir else CIRCUITS_DIR
        self.circuits_dir.mkdir(exist_ok=True)
        self.enable_discovery = enable_discovery
        self.cache_hidden_states = cache_hidden_states

        # Optional CPU threading control for high-core machines (e.g., 5995WX)
        threads_env = os.getenv("CONSCIOUSNESS_NUM_THREADS")
        threads_cfg = num_threads or (int(threads_env) if threads_env else None)
        if threads_cfg and threads_cfg > 0:
            # Bound torch intra/interop threads; helps CPU-bound runs
            torch.set_num_threads(threads_cfg)
            torch.set_num_interop_threads(max(1, threads_cfg // 2))
        
        # Cache of loaded circuits
        self._circuit_cache: Dict[str, Dict] = {}
        
    def _validate_circuit(self, circuit: Dict, hidden_size: int) -> bool:
        """
        Validate that circuit dimensions are within bounds.

        Args:
            circuit: Circuit dictionary with 'dimensions' key
            hidden_size: Target model's hidden dimension

        Returns:
            True if all dimensions are valid, False otherwise
        """
        if "dimensions" not in circuit:
            return False
        for name, dim_idx in circuit["dimensions"].items():
            if not isinstance(dim_idx, int) or dim_idx < 0 or dim_idx >= hidden_size:
                return False
        return True

    def get_circuit(self, model_name: str, hidden_size: int) -> Tuple[Dict, str]:
        """
        Get the best available circuit for a model.

        Returns:
            (circuit_dict, method) where method is "discovered" or "remapped"

        Raises:
            ValueError: If circuit validation fails
        """
        # Check cache
        if model_name in self._circuit_cache:
            cached = self._circuit_cache[model_name]
            if self._validate_circuit(cached, hidden_size):
                return cached, "discovered"
            # Cache is invalid for this hidden_size, remove it
            del self._circuit_cache[model_name]

        # Check bundled circuits (exact match)
        if model_name in self.BUNDLED_CIRCUITS:
            circuit = self.BUNDLED_CIRCUITS[model_name]
            if self._validate_circuit(circuit, hidden_size):
                self._circuit_cache[model_name] = circuit
                return circuit, "discovered"
            # Bundled circuit has wrong hidden_size, fall through to remapping

        # Check for saved discovered circuit
        model_id = get_model_id(model_name)
        circuit_path = self.circuits_dir / f"{model_id}_circuit.json"

        if circuit_path.exists():
            with open(circuit_path, 'r') as f:
                circuit = json.load(f)
            if self._validate_circuit(circuit, hidden_size):
                self._circuit_cache[model_name] = circuit
                return circuit, "discovered"
            # Saved circuit is invalid, fall through to remapping

        # Fall back to remapping from Qwen2.5-32B reference
        circuit = self._create_remapped_circuit(hidden_size)

        # Final validation (should always pass after remapping, but be safe)
        if not self._validate_circuit(circuit, hidden_size):
            raise ValueError(
                f"Failed to create valid circuit for {model_name} with hidden_size={hidden_size}. "
                f"Remapped dimensions are out of bounds."
            )

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
        aggregation: str = "mean",  # "mean", "last", "max", or "entropy_weighted"
        run_logit_lens: bool = False,
        logit_lens_top_k: int = 5,
        use_adaptive_layer: bool = True,
        layer_override: Optional[int] = None,
        length_normalize: bool = True,
    ) -> UniversalResult:
        """
        Measure consciousness score for a prompt.

        Args:
            model: Loaded transformer model
            tokenizer: Model tokenizer
            prompt: Text to measure
            return_hidden: Also return raw hidden states (all layers)
            aggregation: How to aggregate token scores - "mean", "last", "max",
                         or "entropy_weighted" (weights tokens by information content)
            run_logit_lens: If True, also compute logit lens per layer
            logit_lens_top_k: Number of top tokens to return per layer
            use_adaptive_layer: Use depth-aware layer selection (recommended for 32B+)
            layer_override: Manually specify target layer (ignores adaptive)
            length_normalize: Apply length normalization to reduce token-count bias (v3.2)

        Returns:
            UniversalResult with score, metadata, and optional hidden_states/logit_lens
        """
        model_name = getattr(model.config, '_name_or_path', 'unknown')
        hidden_size = model.config.hidden_size
        num_layers = model.config.num_hidden_layers

        # Get circuit
        circuit, method = self.get_circuit(model_name, hidden_size)

        # Determine target layer
        if layer_override is not None:
            target_layer = layer_override
        elif use_adaptive_layer:
            # Use adaptive layer selection for deep models
            layer_frac = get_adaptive_layer_fraction(num_layers, model_name)
            target_layer = int(num_layers * layer_frac)
        else:
            # Use circuit's default layer fraction
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
        # v3.4.1 hybrid: z-score + tanh(z * scale) for smooth bounding
        # Preserves absolute z-score level (strong Computation signal from v3.3)
        # while eliminating hard ceiling effects (from v3.4 insight)
        TANH_SCALE = 0.15  # tanh(z*0.15): z=3→0.42, z=5→0.64, z=8→0.83, z=10→0.91

        token_scores = []
        for pos in range(seq_len):
            hidden = hidden_seq[pos]
            h_mean = hidden.mean().item()
            h_std = hidden.std().item()

            dim_scores_pos = {}
            for name, dim_idx in circuit["dimensions"].items():
                if dim_idx < len(hidden):
                    raw_act = hidden[dim_idx].item()
                    z = (raw_act - h_mean) / (h_std + 1e-8)
                    # Smooth bounding: preserves absolute level, no hard ceiling
                    normalized = float(np.tanh(z * TANH_SCALE))
                    polarity = circuit["polarities"][name]
                    dim_scores_pos[name] = normalized * polarity
            if dim_scores_pos:
                raw = sum(dim_scores_pos.values()) / len(dim_scores_pos)
                token_scores.append(1 / (1 + np.exp(-raw)))
            else:
                token_scores.append(0.5)

        # Aggregate based on method (v3.2: added entropy_weighted)
        if aggregation == "entropy_weighted" and seq_len > 1:
            # Weight tokens by their information content
            ent_weights = entropy_weight_tokens(hidden_seq)
            score = float(np.sum(np.array(token_scores) * ent_weights))
        elif aggregation == "mean":
            score = float(np.mean(token_scores))
        elif aggregation == "max":
            score = float(np.max(token_scores))
        else:  # "last"
            score = token_scores[-1] if token_scores else 0.5

        raw_score = score  # Save pre-normalization score

        # v3.2: Length normalization (additive baseline correction)
        # Removes the score inflation caused by longer token sequences
        if length_normalize:
            expected_baseline = length_normalization_factor(seq_len)
            # Re-center: how much does this prompt deviate from what's
            # expected for its length? Then place on a 0-1 scale.
            # deviation > 0 means more conscious than expected for length
            deviation = score - expected_baseline
            # Scale deviation to fill [0, 1] range, centered at 0.5
            # A deviation of +/-0.3 maps to roughly 0.8/0.2
            score = max(0.0, min(1.0, 0.5 + deviation * 1.5))

        # Compute dimension contributions using last token (for interpretability)
        # v3.4.1 hybrid: tanh on raw z-scores
        hidden = hidden_seq[-1]
        h_mean = hidden.mean().item()
        h_std = hidden.std().item()
        dim_scores = {}
        for name, dim_idx in circuit["dimensions"].items():
            if dim_idx < len(hidden):
                raw_act = hidden[dim_idx].item()
                z = (raw_act - h_mean) / (h_std + 1e-8)
                normalized = float(np.tanh(z * TANH_SCALE))
                polarity = circuit["polarities"][name]
                dim_scores[name] = normalized * polarity

        # v3.2: Quality metrics
        diversity, dominant_dim = compute_dimension_diversity(dim_scores)
        anomalies = detect_anomalies(token_scores, dim_scores, seq_len)
        confidence = compute_confidence(seq_len, diversity, anomalies)

        # v3.2: Dampen score toward 0.5 when anomalies detected
        # This prevents embedding artifacts from producing extreme scores
        if anomalies:
            score = 0.5 + (score - 0.5) * confidence

        circuit_path = None
        if method == "discovered":
            model_id = get_model_id(model_name)
            path = self.circuits_dir / f"{model_id}_circuit.json"
            if path.exists():
                circuit_path = str(path)

        # Optional: return hidden states
        hidden_states_out = None
        if return_hidden:
            hidden_states_out = [h.cpu() for h in outputs.hidden_states]

        # Optional: run logit lens
        logit_lens_out = None
        if run_logit_lens:
            try:
                from .hooks import logit_lens
                logit_lens_out = logit_lens(
                    list(outputs.hidden_states),
                    model,
                    tokenizer,
                    top_k=logit_lens_top_k
                )
            except Exception as e:
                # Fail gracefully if logit lens doesn't work for this model
                logit_lens_out = None

        return UniversalResult(
            score=score,
            method=method,
            dimension_scores=dim_scores,
            model_name=model_name,
            circuit_path=circuit_path,
            confidence=confidence,
            hidden_states=hidden_states_out,
            logit_lens=logit_lens_out,
            target_layer=target_layer,
            raw_score=raw_score,
            token_count=seq_len,
            dimension_diversity=diversity,
            dominant_dimension=dominant_dim,
            anomaly_flags=anomalies,
        )

    def measure_ensemble(
        self,
        model,
        tokenizer,
        prompt: str,
        n_layers: int = 3,
        weights: Optional[list] = None,
        aggregation: str = "mean",
    ) -> UniversalResult:
        """
        Measure consciousness using an ensemble of multiple layers.

        More robust than single-layer measurement, especially for deep models.
        Averages consciousness scores from multiple depths to reduce layer selection sensitivity.

        Args:
            model: Loaded transformer model
            tokenizer: Model tokenizer
            prompt: Text to measure
            n_layers: Number of layers to include in ensemble (default: 3)
            weights: Optional weights for each layer (default: equal weights)
            aggregation: How to aggregate token scores within each layer

        Returns:
            UniversalResult with ensemble score and per-layer breakdown
        """
        model_name = getattr(model.config, '_name_or_path', 'unknown')
        hidden_size = model.config.hidden_size
        num_layers = model.config.num_hidden_layers

        # Get circuit
        circuit, base_method = self.get_circuit(model_name, hidden_size)

        # Get ensemble layers
        ensemble_layers = get_ensemble_layers(num_layers, n_layers)

        # Default: equal weights
        if weights is None:
            weights = [1.0 / n_layers] * n_layers

        # Get activations once
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)

        # Compute score at each layer
        layer_scores = {}
        all_dim_scores = {}

        for layer_idx in ensemble_layers:
            if layer_idx >= len(outputs.hidden_states):
                continue

            hidden_seq = outputs.hidden_states[layer_idx][0].cpu().float()
            seq_len = hidden_seq.shape[0]

            # v3.4.1 hybrid: z-score + tanh(z * scale)
            TANH_SCALE = 0.15

            token_scores = []
            for pos in range(seq_len):
                hidden = hidden_seq[pos]
                h_mean = hidden.mean().item()
                h_std = hidden.std().item()

                dim_scores_pos = {}
                for name, dim_idx in circuit["dimensions"].items():
                    if dim_idx < len(hidden):
                        raw_act = hidden[dim_idx].item()
                        z = (raw_act - h_mean) / (h_std + 1e-8)
                        normalized = float(np.tanh(z * TANH_SCALE))
                        polarity = circuit["polarities"][name]
                        dim_scores_pos[name] = normalized * polarity
                if dim_scores_pos:
                    raw = sum(dim_scores_pos.values()) / len(dim_scores_pos)
                    token_scores.append(1 / (1 + np.exp(-raw)))
                else:
                    token_scores.append(0.5)

            # Aggregate
            if aggregation == "mean":
                layer_score = float(np.mean(token_scores))
            elif aggregation == "max":
                layer_score = float(np.max(token_scores))
            else:
                layer_score = token_scores[-1] if token_scores else 0.5

            layer_scores[layer_idx] = layer_score

            # Store dimension scores from last token (v3.4.1 hybrid)
            hidden = hidden_seq[-1]
            h_mean = hidden.mean().item()
            h_std = hidden.std().item()
            for name, dim_idx in circuit["dimensions"].items():
                if dim_idx < len(hidden):
                    raw_act = hidden[dim_idx].item()
                    z = (raw_act - h_mean) / (h_std + 1e-8)
                    normalized = float(np.tanh(z * TANH_SCALE))
                    polarity = circuit["polarities"][name]
                    if name not in all_dim_scores:
                        all_dim_scores[name] = []
                    all_dim_scores[name].append(normalized * polarity)

        # Weighted ensemble score
        ensemble_score = 0.0
        for (layer_idx, layer_score), weight in zip(layer_scores.items(), weights):
            ensemble_score += layer_score * weight

        # Average dimension scores across layers
        avg_dim_scores = {name: float(np.mean(scores)) for name, scores in all_dim_scores.items()}

        return UniversalResult(
            score=ensemble_score,
            method="ensemble",
            dimension_scores=avg_dim_scores,
            model_name=model_name,
            circuit_path=None,
            target_layer=None,
            ensemble_layers=ensemble_layers,
            ensemble_scores=layer_scores,
        )

    def measure_batch(
        self,
        model,
        tokenizer,
        prompts: list,
        batch_size: int = 4,
        aggregation: str = "mean",
        use_adaptive_layer: bool = True,
        show_progress: bool = True,
    ) -> list:
        """
        Efficiently measure consciousness for multiple prompts.

        Uses batched inference for better GPU utilization. Includes memory management
        for large models (32B+).

        Args:
            model: Loaded transformer model
            tokenizer: Model tokenizer
            prompts: List of prompts to measure
            batch_size: Number of prompts per batch (lower for large models)
            aggregation: How to aggregate token scores
            use_adaptive_layer: Use depth-aware layer selection
            show_progress: Show progress bar

        Returns:
            List of UniversalResult objects
        """
        import gc
        from tqdm import tqdm

        model_name = getattr(model.config, '_name_or_path', 'unknown')
        hidden_size = model.config.hidden_size
        num_layers = model.config.num_hidden_layers

        # Get circuit once
        circuit, method = self.get_circuit(model_name, hidden_size)

        # Determine target layer
        if use_adaptive_layer:
            layer_frac = get_adaptive_layer_fraction(num_layers, model_name)
            target_layer = int(num_layers * layer_frac)
        else:
            layer_frac = circuit.get("layer_fraction", 0.75)
            target_layer = int(num_layers * layer_frac)

        device = next(model.parameters()).device
        results = []

        # Auto-adjust batch size for large models
        if hidden_size > 4096 and batch_size > 2:
            batch_size = max(2, batch_size // 2)

        n_batches = (len(prompts) + batch_size - 1) // batch_size
        iterator = range(n_batches)
        if show_progress:
            iterator = tqdm(iterator, desc="Measuring consciousness", total=n_batches)

        for batch_idx in iterator:
            batch_start = batch_idx * batch_size
            batch_end = min(batch_start + batch_size, len(prompts))
            batch_prompts = prompts[batch_start:batch_end]

            # Tokenize batch with padding
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=512
            )
            inputs = {k: v.to(device) for k, v in inputs.items()}

            # Single forward pass for batch
            model.eval()
            with torch.no_grad():
                outputs = model(**inputs, output_hidden_states=True, return_dict=True)

            # Process each item in batch
            for i, prompt in enumerate(batch_prompts):
                hidden_seq = outputs.hidden_states[target_layer][i].cpu().float()

                # Find actual sequence length (before padding)
                attention_mask = inputs['attention_mask'][i]
                seq_len = attention_mask.sum().item()
                hidden_seq = hidden_seq[:seq_len]

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

                # Aggregate
                if aggregation == "mean":
                    score = float(np.mean(token_scores))
                elif aggregation == "max":
                    score = float(np.max(token_scores))
                else:
                    score = token_scores[-1] if token_scores else 0.5

                # Dimension scores from last token
                hidden = hidden_seq[-1]
                dim_scores = {}
                for name, dim_idx in circuit["dimensions"].items():
                    if dim_idx < len(hidden):
                        activation = hidden[dim_idx].item()
                        polarity = circuit["polarities"][name]
                        dim_scores[name] = activation * polarity

                results.append(UniversalResult(
                    score=score,
                    method=method,
                    dimension_scores=dim_scores,
                    model_name=model_name,
                    circuit_path=None,
                    target_layer=target_layer,
                ))

            # Clear GPU memory after each batch
            del outputs, inputs
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            gc.collect()

        return results

    def patch_sweep(
        self,
        model,
        tokenizer,
        prompt_clean: str,
        prompt_corrupt: str,
        layer_indices: Optional[list] = None,
        token_pos: int = -1,
    ) -> Dict[int, float]:
        """
        Run a clean→corrupt residual patch sweep using consciousness score as the metric.
        
        Args:
            model: Loaded transformer model
            tokenizer: Model tokenizer
            prompt_clean: Prompt expected to yield high consciousness
            prompt_corrupt: Prompt expected to yield low consciousness
            layer_indices: Layers to sweep (default: all)
            token_pos: Token position to patch (default: last)
            
        Returns:
            Dict mapping layer_idx → consciousness score after patching
        """
        from .patching import residual_patch_sweep
        
        circuit_instance = self
        
        def consciousness_metric(logits, encoded_inputs):
            # Re-run measure to get consciousness score
            # This is a bit inefficient but keeps the API clean
            prompt = tokenizer.decode(encoded_inputs.input_ids[0], skip_special_tokens=True)
            result = circuit_instance.measure(model, tokenizer, prompt)
            return result.score
        
        return residual_patch_sweep(
            model=model,
            tokenizer=tokenizer,
            prompt_clean=prompt_clean,
            prompt_corrupt=prompt_corrupt,
            target_fn=consciousness_metric,
            layer_indices=layer_indices,
            token_pos=token_pos,
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


class CachedUniversalCircuit(UniversalCircuit):
    """
    UniversalCircuit with activation caching for repeated measurements.

    Useful when measuring the same prompts multiple times (e.g., during experiments).
    Uses an LRU-style cache to store hidden states and avoid redundant forward passes.

    Example:
        circuit = CachedUniversalCircuit(cache_size=100)

        # First call: computes hidden states
        result1 = circuit.measure(model, tokenizer, "What is consciousness?")

        # Second call: uses cached hidden states (much faster)
        result2 = circuit.measure(model, tokenizer, "What is consciousness?")
    """

    def __init__(
        self,
        cache_size: int = 100,
        **kwargs
    ):
        """
        Initialize cached circuit.

        Args:
            cache_size: Maximum number of prompt/hidden_state pairs to cache
            **kwargs: Additional arguments passed to UniversalCircuit
        """
        super().__init__(**kwargs)
        self.cache_size = cache_size
        self._hidden_cache: Dict[str, torch.Tensor] = {}
        self._cache_order: list = []  # For LRU eviction

    def _get_cache_key(self, model_name: str, prompt: str, layer: int) -> str:
        """Generate cache key from model, prompt, and layer."""
        return hashlib.md5(f"{model_name}:{prompt}:{layer}".encode()).hexdigest()

    def _evict_if_needed(self):
        """Evict oldest cache entries if cache is full."""
        while len(self._cache_order) >= self.cache_size:
            oldest_key = self._cache_order.pop(0)
            if oldest_key in self._hidden_cache:
                del self._hidden_cache[oldest_key]

    def get_cached_hidden_state(
        self,
        model,
        tokenizer,
        prompt: str,
        target_layer: int,
    ) -> torch.Tensor:
        """
        Get hidden state from cache or compute if not cached.

        Args:
            model: Loaded transformer model
            tokenizer: Model tokenizer
            prompt: Text prompt
            target_layer: Which layer's hidden state to return

        Returns:
            Hidden state tensor [seq_len, hidden_size]
        """
        model_name = getattr(model.config, '_name_or_path', 'unknown')
        cache_key = self._get_cache_key(model_name, prompt, target_layer)

        if cache_key in self._hidden_cache:
            # Move to end of LRU order (most recently used)
            if cache_key in self._cache_order:
                self._cache_order.remove(cache_key)
            self._cache_order.append(cache_key)
            return self._hidden_cache[cache_key]

        # Not in cache, compute
        device = next(model.parameters()).device
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=512)
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = model(**inputs, output_hidden_states=True, return_dict=True)

        hidden_state = outputs.hidden_states[target_layer][0].cpu().float()

        # Cache the result
        self._evict_if_needed()
        self._hidden_cache[cache_key] = hidden_state
        self._cache_order.append(cache_key)

        return hidden_state

    def measure(
        self,
        model,
        tokenizer,
        prompt: str,
        use_cache: bool = True,
        **kwargs
    ) -> UniversalResult:
        """
        Measure consciousness with optional caching.

        Args:
            model: Loaded transformer model
            tokenizer: Model tokenizer
            prompt: Text to measure
            use_cache: Whether to use activation cache (default: True)
            **kwargs: Additional arguments passed to parent measure()

        Returns:
            UniversalResult with score and metadata
        """
        if not use_cache:
            return super().measure(model, tokenizer, prompt, **kwargs)

        model_name = getattr(model.config, '_name_or_path', 'unknown')
        hidden_size = model.config.hidden_size
        num_layers = model.config.num_hidden_layers

        # Get circuit
        circuit, method = self.get_circuit(model_name, hidden_size)

        # Determine target layer
        layer_override = kwargs.get('layer_override')
        use_adaptive = kwargs.get('use_adaptive_layer', True)

        if layer_override is not None:
            target_layer = layer_override
        elif use_adaptive:
            layer_frac = get_adaptive_layer_fraction(num_layers, model_name)
            target_layer = int(num_layers * layer_frac)
        else:
            layer_frac = circuit.get("layer_fraction", 0.75)
            target_layer = int(num_layers * layer_frac)

        # Get cached hidden state
        hidden_seq = self.get_cached_hidden_state(model, tokenizer, prompt, target_layer)
        seq_len = hidden_seq.shape[0]

        aggregation = kwargs.get('aggregation', 'mean')

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

        # Aggregate
        if aggregation == "mean":
            score = float(np.mean(token_scores))
        elif aggregation == "max":
            score = float(np.max(token_scores))
        else:
            score = token_scores[-1] if token_scores else 0.5

        # Dimension scores from last token
        hidden = hidden_seq[-1]
        dim_scores = {}
        for name, dim_idx in circuit["dimensions"].items():
            if dim_idx < len(hidden):
                activation = hidden[dim_idx].item()
                polarity = circuit["polarities"][name]
                dim_scores[name] = activation * polarity

        return UniversalResult(
            score=score,
            method=method,
            dimension_scores=dim_scores,
            model_name=model_name,
            circuit_path=None,
            target_layer=target_layer,
        )

    def clear_cache(self):
        """Clear the hidden state cache."""
        self._hidden_cache.clear()
        self._cache_order.clear()

    def cache_stats(self) -> Dict:
        """Get cache statistics."""
        return {
            "size": len(self._hidden_cache),
            "max_size": self.cache_size,
            "utilization": len(self._hidden_cache) / self.cache_size if self.cache_size > 0 else 0,
        }


# Convenience function
def measure_consciousness(
    model,
    tokenizer,
    prompt: str,
    circuits_dir: Optional[str] = None,
    use_ensemble: bool = False,
) -> UniversalResult:
    """
    Quick consciousness measurement with auto-detection.

    Example:
        from consciousness_circuit.universal import measure_consciousness
        result = measure_consciousness(model, tokenizer, "What is consciousness?")
        print(f"Score: {result.score:.3f} ({result.method})")

    Args:
        model: Loaded transformer model
        tokenizer: Model tokenizer
        prompt: Text to measure
        circuits_dir: Optional custom circuits directory
        use_ensemble: Use multi-layer ensemble (more robust, slightly slower)

    Returns:
        UniversalResult with score and metadata
    """
    circuit = UniversalCircuit(circuits_dir=circuits_dir)
    if use_ensemble:
        return circuit.measure_ensemble(model, tokenizer, prompt)
    return circuit.measure(model, tokenizer, prompt)


if __name__ == "__main__":
    # Demo
    circuit = UniversalCircuit()

    print("Available circuits:")
    for name, source in circuit.list_available_circuits().items():
        print(f"  {name}: {source}")

    print("\nNew 32B optimizations:")
    print("  - Adaptive layer selection: get_adaptive_layer_fraction()")
    print("  - Ensemble measurement: circuit.measure_ensemble()")
    print("  - Batch processing: circuit.measure_batch()")
    print("  - Activation caching: CachedUniversalCircuit()")
