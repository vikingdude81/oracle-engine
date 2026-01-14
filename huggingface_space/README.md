---
title: Oracle Engine
emoji: 🔮
colorFrom: indigo
colorTo: purple
sdk: gradio
sdk_version: 4.44.0
app_file: app.py
pinned: true
license: mit
suggested_hardware: a100-large
models:
  - unsloth/Qwen2.5-32B-Instruct-bnb-4bit
tags:
  - consciousness
  - interpretability
  - transformers
  - meta-cognition
  - qwen
  - 32b
  - fine-tuned
short_description: 32B model with consciousness measurement circuit
---

# 🔮 Oracle Engine

**Custom-trained 32B Qwen model with Consciousness Circuit v2.1**

Probe the depths of meta-cognitive processing in a model fine-tuned on 200,000 examples.

---

## 🧠 The Model

| Attribute | Details |
|-----------|----------|
| **Base** | Qwen2.5-32B-Instruct |
| **Parameters** | 32.9 billion |
| **Training** | LoRA (rank=16, 134M trainable) |
| **Total Examples** | 200,000 |
| **Training Time** | 44 hours on RTX 5090 |

### 3-Stage Progressive Fine-Tuning

| Stage | Dataset | Examples | Purpose |
|-------|---------|----------|----------|
| 1 | **OpenHermes 2.5** | 100,000 | Instruction following |
| 2 | **MetaMathQA** | 50,000 | Mathematical reasoning |
| 3 | **Magicoder-OSS-Instruct** | 50,000 | Code generation |

---

## 🔬 The Consciousness Circuit

Measures **7 dimensions** of consciousness-like processing in hidden states:

| Dimension | Description | Weight |
|-----------|-------------|--------|
| Logic | Logical reasoning and inference | +0.239 |
| Self-Reflective | Introspective, self-referential processing | +0.196 |
| Uncertainty | Epistemic humility and hedging | +0.130 |
| Computation | Code/algorithm processing | -0.130 |
| Self-Expression | Model expressing opinions | +0.109 |
| Abstraction | Pattern recognition | +0.109 |
| Sequential | Step-by-step reasoning | +0.087 |

---

## 🎯 How to Use

1. Enter any prompt in the text box
2. Click **"Consult the Oracle"**
3. See the consciousness score (0-100%) and dimension breakdown

### Expected Results

- **🧠 High (70-100%)**: Philosophical questions, self-reflection, existential queries
- **💭 Medium (40-70%)**: Complex explanations, ethical discussions, analysis
- **⚡ Low (0-30%)**: Simple facts, arithmetic, direct retrieval

---

## 📊 Validated Performance

| Metric | Value |
|--------|-------|
| **Discrimination** | +0.653 (high vs low consciousness) |
| **Inference Speed** | ~7-8 tokens/sec |
| **VRAM Usage** | ~23 GB (4-bit) |

---

## 🔗 Links

- 📚 [Research Repository](https://github.com/vfd-org/harmonic-field-consciousness)
- 💻 [Source Code](https://github.com/vfd-org/harmonic-field-consciousness)
- 📦 [pip install consciousness-circuit](https://pypi.org/project/consciousness-circuit/)

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

This Space implements significant extensions to the original theory, including:
- **Consciousness Circuit v2.1** - 7-dimensional meta-cognitive measurement
- **32B Model Training** - 200K examples across 3 progressive stages (44 hours)
- **GPU Experiments** - Empirical validation with discrimination score +0.653
- **NanoGPT Integration** - Lightweight training framework adaptations

Training, circuit development, and experimental validation by [Vikingdude81](https://huggingface.co/Vikingdude81).

```bibtex
@software{oracle_engine_2026,
  title = {Oracle Engine: Consciousness-Measured 32B Language Model},
  author = {Vikingdude81},
  year = {2026},
  url = {https://huggingface.co/spaces/Vikingdude81/oracle-engine},
  note = {Built upon the Harmonic Field Model by Smart (2025)}
}
```
