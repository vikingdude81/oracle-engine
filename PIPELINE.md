# Oracle Engine - Modular Consciousness Analysis Pipeline

A composable toolkit for analyzing and improving AI models using consciousness circuit measurements.

## Architecture

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         ORACLE ENGINE PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│   [Model]  ──▶  [Analyzer]  ──▶  [Profiler]  ──▶  [Feedback]  ──▶  [Export] │
│      │              │               │                │               │       │
│   Unsloth     Trajectory      Test Suites      Improvement        JSON      │
│   or HF       + Dynamics       Benchmark       Suggestions         CSV      │
│                                                                              │
│                          ▲                            │                      │
│                          │                            │                      │
│                          └────── Training Loop ───────┘                      │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

## Components

| Module | Purpose | Import |
|--------|---------|--------|
| **oracle_api.py** | High-level Python API | `from oracle_api import Oracle` |
| **oracle_wrapper.py** | CLI wrapper for piping | `python oracle_wrapper.py <cmd>` |
| **consciousness_circuit/** | Core metrics & analysis | `from consciousness_circuit import *` |

## Quick Start

### 1. Python API (Recommended)

```python
from oracle_api import Oracle

# Load custom-trained model
oracle = Oracle.from_custom_model()

# Single analysis
result = oracle.analyze("What is consciousness?")
print(f"Score: {result.score:.3f}")
print(result.interpretation)

# Profile a category
profile = oracle.profile("philosophical")
print(profile.summary())

# Get training feedback
feedback = oracle.get_feedback(target_consciousness=0.8)
for rec in feedback['recommendations']:
    print(f"[{rec['priority']}] {rec['action']}")
```

### 2. Command Line (Pipe-able)

```bash
# Single analysis
python oracle_wrapper.py analyze "What is consciousness?"

# Profile category
python oracle_wrapper.py profile --category philosophical

# Compare custom vs base
python oracle_wrapper.py compare --category philosophical

# Export results
python oracle_wrapper.py profile --all --export results.json

# Get training feedback
python oracle_wrapper.py feedback --target-consciousness 0.8
```

### 3. Full Suite Analysis

```bash
# From WSL2:
cd /home/akbon/unsloth_train
source .venv/bin/activate

# Quick test
python /mnt/c/.../oracle-engine/full_suite_analysis.py --quick-test

# Compare models
python /mnt/c/.../oracle-engine/full_suite_analysis.py --compare-base
```

## Core Components

### consciousness_circuit/

```
consciousness_circuit/
├── analyzers/
│   └── trajectory.py      # ConsciousnessTrajectoryAnalyzer
├── benchmarks/
│   ├── profiler.py        # ModelProfiler
│   └── test_suites.py     # Categorized test prompts
├── metrics/
│   ├── lyapunov.py        # Chaos measurement
│   ├── hurst.py           # Memory/persistence
│   ├── msd.py             # Mean squared displacement
│   └── agency.py          # Goal-directedness
├── classifiers/
│   └── signal.py          # Trajectory classification
├── universal.py           # UniversalCircuit
├── circuit.py             # ConsciousnessCircuit
└── visualization.py       # ConsciousnessVisualizer
```

### Key Classes

| Class | Description |
|-------|-------------|
| `ConsciousnessTrajectoryAnalyzer` | Combines consciousness + trajectory dynamics |
| `ModelProfiler` | Systematic benchmarking across test suites |
| `UniversalCircuit` | Universal consciousness measurement |
| `TAMEMetrics` | Agency and goal-directedness |

## Model Improvement Loop

```python
from oracle_api import Oracle

# 1. Profile current model
oracle = Oracle.from_custom_model()
profile = oracle.profile("philosophical")

# 2. Get improvement feedback
feedback = oracle.get_feedback(
    target_consciousness=0.8,
    target_agency=0.7
)

# 3. Review recommendations
for rec in feedback['recommendations']:
    print(f"[{rec['priority']}] {rec['area']}: {rec['action']}")

# 4. Get training data suggestions
for suggestion in feedback['training_data_suggestions']:
    print(f"  → {suggestion}")

# 5. After retraining, compare
custom = Oracle.from_custom_model()  # new version
base = Oracle.from_base_model()
comparison = Oracle.compare(custom, base, "philosophical")
print(comparison['suggestions'])
```

## Test Suite Categories

| Category | Focus | Prompts |
|----------|-------|---------|
| **philosophical** | Self-reflection, consciousness, ethics | 15 |
| **factual** | Knowledge recall, definitions | 15 |
| **reasoning** | Logic, math, problem-solving | 15 |
| **creative** | Storytelling, poetry, imagination | 15 |

## Metrics Explained

### Consciousness Score
- **0.0-0.3**: Low - automatic processing
- **0.3-0.6**: Moderate - some reflection
- **0.6-1.0**: High - meta-cognitive, reflective

### Trajectory Classes
- **ATTRACTOR**: Converging to coherent reasoning
- **BALLISTIC**: Directed, purposeful thought
- **CHAOTIC**: Exploring solution space
- **DIFFUSIVE**: Random walk, uncertain
- **DRIFT**: Slow wandering

### Agency Score
- Measures goal-directedness and intentional behavior
- Higher = more purposeful reasoning

## Requirements

```bash
pip install -e ./consciousness_circuit
pip install unsloth torch transformers bitsandbytes
```

## WSL2 Setup (RTX 5090)

```bash
cd /home/akbon/unsloth_train
source .venv/bin/activate

# Required for Blackwell architecture
export UNSLOTH_COMPILE_DISABLE=1

python /mnt/c/.../oracle-engine/oracle_wrapper.py <command>
```

## Model Paths

| Model | Path |
|-------|------|
| Custom Trained | `/home/akbon/unsloth_train/outputs_stage3_code/final` |
| Base Model | `unsloth/Qwen2.5-32B-Instruct-bnb-4bit` |

## License

MIT
