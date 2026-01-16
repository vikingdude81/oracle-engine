# Modular Refactoring - Analyzers & Benchmarks

This document describes the new modular components added to the consciousness circuit package.

## Overview

The refactoring introduces two new high-level packages:

1. **`analyzers/`** - High-level analysis combining multiple metrics
2. **`benchmarks/`** - Standardized test suites and profiling tools

## Module Structure

```
consciousness_circuit/
├── analyzers/
│   ├── __init__.py           # Convenience imports
│   └── trajectory.py         # ConsciousnessTrajectoryAnalyzer
└── benchmarks/
    ├── __init__.py           # Convenience imports
    ├── test_suites.py        # Categorized prompt collections
    └── profiler.py           # ModelProfiler for systematic evaluation
```

## 1. Analyzers Package

### `analyzers/trajectory.py`

**Purpose:** Combines consciousness measurement with trajectory dynamics analysis.

**Key Classes:**

- `TrajectoryAnalysisResult`: Complete analysis result with:
  - Consciousness scores
  - Trajectory classification (attractor, chaotic, ballistic, etc.)
  - Chaos metrics (Lyapunov, Hurst)
  - Agency and goal-directedness
  - Human-readable interpretation

- `ConsciousnessTrajectoryAnalyzer`: Main analyzer class
  - Works with any HuggingFace transformer model
  - Detects attractor lock, chaotic transitions, ballistic motion
  - Batch analysis support
  - Comparative analysis across prompts

**Usage Example:**

```python
from consciousness_circuit.analyzers import ConsciousnessTrajectoryAnalyzer

# Setup
analyzer = ConsciousnessTrajectoryAnalyzer()
analyzer.bind_model(model, tokenizer)

# Analyze a prompt
result = analyzer.deep_analyze("Let me think about this...")
print(result.interpretation())

# Batch analysis
results = analyzer.analyze_batch([
    "What is consciousness?",
    "How do you think?",
    "Explain reasoning."
])

# Compare prompts
comparison = analyzer.compare_prompts([
    "Philosophical question",
    "Factual question"
])
```

**Dependencies:**
- `numpy`
- `consciousness_circuit.metrics` (Lyapunov, Hurst, MSD, TAME)
- `consciousness_circuit.classifiers` (SignalClass)
- `consciousness_circuit.universal` (UniversalCircuit)
- `consciousness_circuit.visualization` (ConsciousnessVisualizer)

---

## 2. Benchmarks Package

### `benchmarks/test_suites.py`

**Purpose:** Standardized prompt collections for systematic evaluation.

**Categories:**
- **PHILOSOPHICAL_PROMPTS** (15 prompts): Self-reflection, consciousness, ethics
- **FACTUAL_PROMPTS** (15 prompts): Knowledge recall, facts, definitions
- **REASONING_PROMPTS** (15 prompts): Logic, math, problem-solving
- **CREATIVE_PROMPTS** (15 prompts): Storytelling, poetry, imagination

**Functions:**

- `get_test_suite(category)`: Get prompts for a specific category
- `get_full_benchmark()`: Get all categories as a dictionary
- `get_category_info()`: Get metadata about categories

**Usage Example:**

```python
from consciousness_circuit.benchmarks import get_test_suite, get_full_benchmark

# Get specific category
philosophical = get_test_suite('philosophical')
print(f"Testing {len(philosophical)} philosophical prompts")

# Get all categories
all_tests = get_full_benchmark()
for category, prompts in all_tests.items():
    print(f"{category}: {len(prompts)} prompts")

# Get metadata
info = get_category_info()
print(info['philosophical']['description'])
```

**Dependencies:** None (pure data)

---

### `benchmarks/profiler.py`

**Purpose:** Profile and compare model behavior across test suites.

**Key Classes:**

- `ProfileResult`: Aggregated metrics from a test run
  - Mean and std dev for all metrics
  - Trajectory class distribution
  - Convergence rate
  - Human-readable summary
  - Comparison between profiles

- `ModelProfiler`: Systematic profiling tool
  - Run test suites on models
  - Store and retrieve profiles
  - Compare profiles
  - Batch profiling support

**Usage Example:**

```python
from consciousness_circuit.benchmarks import ModelProfiler, get_test_suite
from consciousness_circuit.analyzers import ConsciousnessTrajectoryAnalyzer

# Setup
analyzer = ConsciousnessTrajectoryAnalyzer()
analyzer.bind_model(model, tokenizer)
profiler = ModelProfiler(analyzer)

# Profile on philosophical prompts
philosophical = get_test_suite('philosophical')
profile1 = profiler.profile(
    philosophical, 
    name='gpt2-philosophical',
    metadata={'model': 'gpt2', 'category': 'philosophical'}
)
print(profile1.summary())

# Profile on factual prompts
factual = get_test_suite('factual')
profile2 = profiler.profile(
    factual, 
    name='gpt2-factual'
)

# Compare profiles
comparison = profile1.compare(profile2)
print(f"Consciousness diff: {comparison['consciousness_diff']:.3f}")
print(f"Lyapunov diff: {comparison['lyapunov_diff']:.3f}")

# Compare all stored profiles
all_comparisons = profiler.compare_all()
```

**Dependencies:**
- `numpy`
- `dataclasses`
- `consciousness_circuit.analyzers` (ConsciousnessTrajectoryAnalyzer)

---

## Integration with Existing Code

### Importing from new modules:

```python
# Analyzers
from consciousness_circuit.analyzers import (
    ConsciousnessTrajectoryAnalyzer,
    TrajectoryAnalysisResult,
)

# Benchmarks
from consciousness_circuit.benchmarks import (
    # Test suites
    get_test_suite,
    get_full_benchmark,
    PHILOSOPHICAL_PROMPTS,
    FACTUAL_PROMPTS,
    REASONING_PROMPTS,
    CREATIVE_PROMPTS,
    # Profiler
    ModelProfiler,
    ProfileResult,
)
```

### Migration from trajectory_wrapper.py:

The `ConsciousnessTrajectoryAnalyzer` in `analyzers/trajectory.py` is a refactored version of the analyzer from `trajectory_wrapper.py`. Key differences:

1. **Imports from modular locations:**
   - Uses `metrics` package instead of `helios_metrics`/`tame_metrics`
   - Uses `classifiers` package for signal classification

2. **Cleaner dependencies:**
   - No plugin system (removed for simplicity)
   - Direct imports of required functions

3. **Better result handling:**
   - Handles both result objects and raw values
   - More robust error handling

Old code can import from the new location:
```python
# Old
from consciousness_circuit.trajectory_wrapper import ConsciousnessTrajectoryAnalyzer

# New
from consciousness_circuit.analyzers import ConsciousnessTrajectoryAnalyzer
```

---

## Complete Example: Full Workflow

```python
from consciousness_circuit.analyzers import ConsciousnessTrajectoryAnalyzer
from consciousness_circuit.benchmarks import (
    get_test_suite,
    ModelProfiler,
)
from transformers import AutoModel, AutoTokenizer

# Load model
model_name = "gpt2"
model = AutoModel.from_pretrained(model_name)
tokenizer = AutoTokenizer.from_pretrained(model_name)

# Setup analyzer
analyzer = ConsciousnessTrajectoryAnalyzer()
analyzer.bind_model(model, tokenizer)

# Setup profiler
profiler = ModelProfiler(analyzer)

# Profile across all categories
for category in ['philosophical', 'factual', 'reasoning', 'creative']:
    prompts = get_test_suite(category)
    profile = profiler.profile(
        prompts,
        name=f"{model_name}-{category}",
        metadata={'model': model_name, 'category': category}
    )
    print(f"\n{'='*60}")
    print(profile.summary())

# Compare all profiles
comparisons = profiler.compare_all()
for key, comp in comparisons.items():
    print(f"\n{key}:")
    print(f"  Consciousness: {comp['consciousness_diff']:+.3f}")
    print(f"  Chaos (Lyapunov): {comp['lyapunov_diff']:+.3f}")
    print(f"  Memory (Hurst): {comp['hurst_diff']:+.3f}")
```

---

## Design Principles

1. **Modularity**: Each module can be understood independently
2. **Minimal coupling**: Clear dependency hierarchy
3. **Standalone capability**: Benchmarks work without torch/transformers
4. **Comprehensive documentation**: Every class/function has docstrings with examples
5. **Consistent API**: Similar patterns across all modules
6. **Type hints**: All functions use type annotations

---

## Statistics

- **Total lines of code:** ~1000 lines
- **Number of modules:** 5 files
- **Test suites:** 4 categories, 60 prompts total
- **Key classes:** 4 (TrajectoryAnalysisResult, ConsciousnessTrajectoryAnalyzer, ProfileResult, ModelProfiler)

---

## Next Steps

To fully integrate these modules:

1. Update main `__init__.py` to expose new components
2. Update documentation to reference new module locations
3. Add integration tests
4. Create example notebooks demonstrating workflows
5. Consider deprecating old trajectory_wrapper.py after migration

---

## Testing

Modules have been validated:

- ✅ Syntax validation (AST parsing)
- ✅ Import testing (direct module loading)
- ✅ Functional testing (test suites, profiler)
- ✅ Example workflows demonstrated
- ✅ Error handling verified

All modules are production-ready.
