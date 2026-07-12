# Consciousness Circuit v3.0 🧠⚡

**Measure and analyze consciousness-like activation patterns in transformer LLMs**

[![PyPI version](https://badge.fury.io/py/consciousness-circuit.svg)](https://badge.fury.io/py/consciousness-circuit)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)

---

## 🆕 What's New in v3.0

| Feature | Description |
|---------|-------------|
| **Adaptive Layer Selection** | `get_adaptive_layer_fraction()` - Depth-aware layer selection (0.65 for 64-layer 32B models) |
| **Ensemble Measurement** | `measure_ensemble()` - Multi-layer scoring for robustness |
| **Batch Processing** | `measure_batch()` - Memory-efficient batched inference for 32B+ models |
| **Activation Caching** | `CachedUniversalCircuit` - LRU cache for repeated measurements |
| **Model Adapters** | Unified interface for HuggingFace, NanoGPT, Unsloth models |
| **Centralized Logging** | `ExperimentLogger` for structured experiment tracking |
| **Patching & Steering** | `residual_patch_sweep()`, `add_residual_steering()` for interpretability |

---

## 🎯 What Is This?

**Consciousness Circuit** is a Python library that measures "consciousness-like" activation patterns in Large Language Models. It identifies specific hidden dimensions that correlate with:

- **Self-reflection** ("I think...", "In my view...")
- **Uncertainty expression** ("I'm not sure, but...")
- **Abstract reasoning** ("At a deeper level...")
- **Metacognition** ("Let me reconsider...")

### The Key Insight

Different model architectures (Qwen, Mistral, Llama, etc.) encode these patterns in **completely different dimensions**. You can't simply copy dimension indices between models—you need to **discover** the circuit empirically for each architecture.

---

## 🚀 Quick Start

### Installation

```bash
pip install consciousness-circuit

# With visualization support
pip install consciousness-circuit[viz]

# Optional: set CPU threads (useful on high-core CPUs like 5995WX)
# export CONSCIOUSNESS_NUM_THREADS=32
```

### Basic Usage

```python
from consciousness_circuit import measure_consciousness
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load your model
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-7B-Instruct")
tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-7B-Instruct")

# Measure consciousness score
result = measure_consciousness(model, tokenizer, "What is the meaning of life?")
print(f"Consciousness Score: {result.score:.3f}")
# Output: Consciousness Score: 0.742
```

### Per-Token Analysis

```python
from consciousness_circuit import UniversalCircuit

circuit = UniversalCircuit()

# Get per-token consciousness trajectory
trajectory = circuit.measure_per_token(model, tokenizer, "Let me think about this...")

# Visualize
trajectory.plot()  # Shows consciousness evolving across tokens
```

### 🆕 32B Model Optimizations

```python
from consciousness_circuit import (
    UniversalCircuit, 
    CachedUniversalCircuit,
    get_adaptive_layer_fraction,
)

# Adaptive layer selection for deep models
num_layers = 64  # Qwen2.5-32B
layer_frac = get_adaptive_layer_fraction(num_layers)  # Returns 0.65
target_layer = int(num_layers * layer_frac)  # Layer 41

# Ensemble measurement (more robust)
circuit = UniversalCircuit()
result = circuit.measure_ensemble(model, tokenizer, "What is consciousness?", n_layers=3)
print(f"Ensemble score: {result.score:.3f}")
print(f"Per-layer scores: {result.ensemble_scores}")

# Batch processing with memory management
prompts = ["prompt1", "prompt2", "prompt3", ...]
results = circuit.measure_batch(model, tokenizer, prompts, batch_size=2)

# Activation caching for experiments
cached_circuit = CachedUniversalCircuit(cache_size=100)
result1 = cached_circuit.measure(model, tokenizer, "Test prompt")  # Computes
result2 = cached_circuit.measure(model, tokenizer, "Test prompt")  # Uses cache (faster)
print(cached_circuit.cache_stats())  # {'size': 1, 'max_size': 100, 'utilization': 0.01}
```

### Discover Circuit for New Model

```python
from consciousness_circuit import ValidationBasedDiscovery

discovery = ValidationBasedDiscovery("meta-llama/Llama-3-8B-Instruct")
circuit = discovery.discover_validated_circuit()

# Circuit is now cached and ready to use
result = measure_consciousness(model, tokenizer, "Some prompt")
```

---

## 📊 How It Works

### 1. Validation-Based Discovery

Instead of manually selecting dimensions or copying from other models, we **discover** consciousness-correlated dimensions by:

1. Running a diverse set of prompts (philosophical, factual, self-reflective)
2. Measuring activation patterns across ALL hidden dimensions
3. Finding dimensions that best correlate with expected "consciousness levels"
4. Validating that the circuit achieves proper ordering: High > Medium > Low

### 2. Architecture-Specific Circuits

Each model family has its own circuit:

| Model | Discrimination | High→Low Range |
|-------|----------------|----------------|
| Qwen2.5-7B | 0.667 | 0.755 → 0.088 |
| Mistral-7B | 0.106 | 0.559 → 0.453 |
| *Your Model* | *Discoverable* | *Varies* |

### 3. Dimension Examples

**Qwen2.5-7B Circuit:**
- Dim 2023 [+]: Activates on philosophical content
- Dim 1116 [+]: Activates on self-reflection
- Dim 2628 [-]: Deactivates on simple factual queries

**Mistral-7B Circuit:**
- Dim 3362 [+]: Different dimension, similar function
- Dim 777 [-]: Unique to Mistral architecture

---

## 💡 Use Cases

### 1. **Model Evaluation & Selection**
Compare how different models handle consciousness-adjacent tasks:
```python
models = ["Qwen/Qwen2.5-7B", "mistralai/Mistral-7B"]
for model_name in models:
    score = measure_consciousness(model, tok, "Explain your reasoning process")
    print(f"{model_name}: {score:.3f}")
```

### 2. **Prompt Engineering**
Measure which prompts elicit more reflective responses:
```python
prompts = [
    "What is 2+2?",                          # Low: ~0.09
    "Explain quantum mechanics",              # Medium: ~0.45
    "Reflect on the nature of consciousness", # High: ~0.75
]
```

### 3. **Fine-Tuning Guidance**
Track consciousness metrics during training:
```python
# Before training
baseline = measure_consciousness(model, tok, test_prompt)

# After training
post_training = measure_consciousness(model, tok, test_prompt)

print(f"Consciousness shift: {post_training.score - baseline.score:+.3f}")
```

### 4. **AI Safety Research**
Identify when models exhibit self-aware or metacognitive patterns:
```python
trajectory = circuit.measure_per_token(model, tok, response)
if trajectory.peak_score > 0.8:
    print("⚠️ High metacognitive activation detected")
```

### 5. **Interpretability Research**
Study which dimensions encode which concepts:
```python
analysis = circuit.analyze_dimensions(model, tok, [
    "I think therefore I am",
    "The capital of France is Paris",
])
analysis.plot_dimension_heatmap()
```

---

## 🔬 The Science

### What We're Measuring

We measure activation patterns in the model's hidden states at approximately 75% depth (layer 21/28 for Qwen-7B). At this layer, the model has:
- Processed the input semantically
- Not yet committed to specific output tokens
- Rich representations of reasoning state

### Validation Methodology

Our circuits are validated to achieve **proper ordering**:
- **High-consciousness prompts** (philosophical, self-reflective) → High scores
- **Medium-consciousness prompts** (reasoning, analysis) → Medium scores  
- **Low-consciousness prompts** (simple factual) → Low scores

This is measured using 15 test prompts across 3 categories.

### Limitations

⚠️ **This is NOT a consciousness detector.** We measure activation patterns that *correlate* with human judgments of "consciousness-like" language. The model is not actually conscious.

⚠️ **Results are architecture-specific.** A circuit discovered for Qwen won't work for Mistral without re-discovery.

⚠️ **Scores are relative, not absolute.** A score of 0.7 means "high for this model," not "70% conscious."

---

## 📦 API Reference

### `measure_consciousness(model, tokenizer, prompt) → UniversalResult`
Quick one-shot measurement with auto-detection.

### `UniversalCircuit`
Main class for consciousness measurement:
- `.measure(model, tokenizer, prompt)` → Single measurement
- `.measure_per_token(model, tokenizer, prompt)` → Token-by-token trajectory
- `.measure_batch(model, tokenizer, prompts)` → Batch measurement
- `.discover(model, tokenizer)` → Discover new circuit
- `.list_available_circuits()` → Show bundled circuits

### `ValidationBasedDiscovery`
Discover circuits for new models:
- `.discover_validated_circuit(top_k=7)` → Find best dimensions
- `.save_circuit(circuit, path)` → Save to JSON

### `UniversalResult`
Measurement result:
- `.score` → Float 0-1
- `.method` → "discovered" or "remapped"
- `.dimension_scores` → Dict of per-dimension activations
- `.model_name` → Which model was used

---

## 🛠️ CLI Tools

```bash
# Discover circuit for a new model
consciousness-discover --model "your/model-name" --save circuit.json

# Measure a prompt
consciousness-measure --model "Qwen/Qwen2.5-7B-Instruct" --prompt "What is consciousness?"
```

---

## 🤝 Contributing

We welcome contributions! Key areas:
- **New model circuits**: Run discovery on models we haven't tested
- **Visualization**: Better plots and interactive dashboards
- **Validation prompts**: More diverse test cases
- **Documentation**: Examples and tutorials

---

## 📄 Citation

If you use this in research:

```bibtex
@software{consciousness_circuit,
  author = {VFD-Org},
  title = {Consciousness Circuit: Measuring Consciousness-Like Patterns in LLMs},
  year = {2025},
  url = {https://github.com/vfd-org/consciousness-circuit}
}
```

---

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.

---

## 🙋 FAQ

**Q: Is this measuring actual consciousness?**
A: No. We measure activation patterns that correlate with self-reflective language. The model is not conscious.

**Q: Why do different models need different circuits?**
A: Each architecture learns different internal representations. Dimension 2023 in Qwen means something completely different than dimension 2023 in Mistral.

**Q: How accurate is this?**
A: On validation sets, we achieve proper High > Medium > Low ordering with 0.1-0.7 discrimination depending on model architecture.

**Q: Can I use this for production?**
A: Yes, with caveats. The measurement is fast (~10ms per prompt after model loading), but interpretations should be validated for your specific use case.
