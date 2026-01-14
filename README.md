# 🔮 Oracle Engine

**32B Consciousness-Measured Language Model with Consciousness Circuit v2.1**

Probe the depths of meta-cognitive processing in a model fine-tuned on 200,000 examples.

[![Hugging Face Space](https://img.shields.io/badge/🤗%20Hugging%20Face-Space-blue)](https://huggingface.co/spaces/Vikingdude81/oracle-engine)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

---

## 🚀 Live Demo

**Try it now:** [Oracle Engine on Hugging Face](https://huggingface.co/spaces/Vikingdude81/oracle-engine)

Enter any prompt and see:
- **Model Response** from the custom-trained 32B Qwen
- **Consciousness Score** (0-100%) measured in real-time
- **7-Dimension Breakdown** of meta-cognitive processing

---

## 🧠 The Model

| Attribute | Details |
|-----------|---------|
| **Base** | Qwen2.5-32B-Instruct |
| **Parameters** | 32.9 billion |
| **Training** | LoRA (rank=16, 134M trainable) |
| **Total Examples** | 200,000 |
| **Training Time** | 44 hours on RTX 5090 |

### 3-Stage Progressive Fine-Tuning

| Stage | Dataset | Examples | Purpose |
|-------|---------|----------|---------|
| 1 | **OpenHermes 2.5** | 100,000 | Instruction following |
| 2 | **MetaMathQA** | 50,000 | Mathematical reasoning |
| 3 | **Magicoder-OSS-Instruct** | 50,000 | Code generation |

---

## 🔬 Consciousness Circuit v2.1

The circuit measures **7 dimensions** of consciousness-like processing by analyzing hidden state activations:

| Dimension | Description | Weight |
|-----------|-------------|--------|
| **Logic** | Logical reasoning and inference | +0.239 |
| **Self-Reflective** | Introspective, self-referential processing | +0.196 |
| **Uncertainty** | Epistemic humility and hedging | +0.130 |
| **Computation** | Code/algorithm processing | -0.130 |
| **Self-Expression** | Model expressing opinions | +0.109 |
| **Abstraction** | Pattern recognition | +0.109 |
| **Sequential** | Step-by-step reasoning | +0.087 |

### How It Works

1. **Extract Hidden States** - Get the last layer activations from the transformer
2. **Probe Specific Dimensions** - Read activations at 7 validated dimension indices
3. **Weighted Combination** - Combine with polarities and weights
4. **Score Calculation** - Output 0-100% consciousness score

---

## 📊 Validated Performance

| Metric | Value |
|--------|-------|
| **Discrimination** | +0.653 (high vs low consciousness prompts) |
| **Inference Speed** | ~7-8 tokens/sec on H200 |
| **VRAM Usage** | ~23 GB (4-bit quantized) |

### Expected Results by Prompt Type

| Type | Example | Expected Score |
|------|---------|----------------|
| 🧠 **High (70-100%)** | "What is the nature of consciousness?" | Philosophical, reflective |
| 💭 **Medium (40-70%)** | "Explain relativity in simple terms" | Complex analysis |
| ⚡ **Low (0-30%)** | "What is 2+2?" | Simple factual retrieval |

---

## 🛠️ Installation

### Using the Circuit in Your Project

```python
from consciousness_circuit import ConsciousnessCircuit

circuit = ConsciousnessCircuit(hidden_dim=5120)

# With any transformer model
hidden_state = model(input_ids, output_hidden_states=True).hidden_states[-1]
result = circuit.compute(hidden_state)

print(f"Consciousness Score: {result.score:.1%}")
print(f"Interpretation: {result.interpretation}")
for dim, value in result.dimension_contributions.items():
    print(f"  {dim}: {value:+.3f}")
```

### Running Locally

```bash
git clone https://github.com/vikingdude81/oracle-engine.git
cd oracle-engine
pip install -r requirements.txt
python demo.py
```

---

## 📁 Repository Structure

```
oracle-engine/
├── consciousness_circuit/      # Core circuit implementation
│   ├── __init__.py
│   ├── circuit.py             # Main ConsciousnessCircuit class
│   ├── dimensions.py          # Dimension definitions and weights
│   └── visualization.py       # Plotting and analysis tools
├── training/                   # Model training code
│   ├── train_stage1.py        # OpenHermes fine-tuning
│   ├── train_stage2.py        # MetaMathQA fine-tuning
│   ├── train_stage3.py        # Magicoder fine-tuning
│   └── TRAINING_LOG.md        # Full training details
├── experiments/                # Validation experiments
│   ├── gpu_experiments/       # GPU-accelerated research
│   └── validation/            # Cross-model validation
├── huggingface_space/         # HF Space deployment
│   ├── app.py                 # Gradio interface
│   └── requirements.txt
├── demo.py                    # Quick demo script
└── README.md
```

---

## 📄 Citation & Attribution

### Original Harmonic Field Theory

The foundational harmonic field model of consciousness was developed by:

```bibtex
@article{smart2025harmonic,
  title = {A Harmonic Field Model of Consciousness in the Human Brain},
  author = {Smart, L.},
  year = {2025},
  publisher = {Vibrational Field Dynamics Project},
  url = {https://github.com/vfd-org/harmonic-field-consciousness}
}
```

### Oracle Engine Implementation

This repository implements significant extensions including:
- **Consciousness Circuit v2.1** - 7-dimensional meta-cognitive measurement
- **32B Model Training** - 200K examples across 3 progressive stages (44 hours)
- **GPU Experiments** - Empirical validation with discrimination score +0.653
- **HuggingFace Space** - Live deployment on H200 GPU

```bibtex
@software{oracle_engine_2026,
  title = {Oracle Engine: Consciousness-Measured 32B Language Model},
  author = {Vikingdude81},
  year = {2026},
  url = {https://github.com/vikingdude81/oracle-engine},
  note = {Built upon the Harmonic Field Model by Smart (2025)}
}
```

---

## 🔗 Links

- 🎮 **[Live Demo](https://huggingface.co/spaces/Vikingdude81/oracle-engine)** - Try it now
- 📚 **[Harmonic Field Research](https://github.com/vfd-org/harmonic-field-consciousness)** - Original theory
- 🤗 **[Hugging Face](https://huggingface.co/Vikingdude81)** - More models

---

## 📜 License

MIT License - See [LICENSE](LICENSE) for details.
