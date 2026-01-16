# Modular Architecture Documentation

This document describes the fully modular, standalone architecture of the Consciousness Circuit package.

## 🎯 Design Philosophy: Zero Coupling

Each module in the modular structure is **fully standalone** and can be copied to any project independently. They only depend on `numpy` and standard library modules.

### Key Principles

✅ **Each standalone file:**
- Only imports `numpy` (and stdlib: `typing`, `dataclasses`, `enum`)
- Does NOT import from sibling modules
- Has complete docstrings with usage examples
- Returns structured dataclass results
- Works if you literally copy just that one file to another project

❌ **Each standalone file does NOT:**
- Import from other metrics/classifier modules
- Depend on package structure or configuration
- Require setup beyond `import numpy`

## 📁 Directory Structure

```
consciousness_circuit/
├── metrics/                       # STANDALONE metric modules
│   ├── __init__.py                # Convenience imports
│   ├── README.md                  # Documentation for metrics
│   ├── lyapunov.py                # Lyapunov exponent - FULLY STANDALONE
│   ├── hurst.py                   # Hurst exponent - FULLY STANDALONE  
│   ├── msd.py                     # Mean squared displacement - FULLY STANDALONE
│   ├── entropy.py                 # Spectral entropy, runs test - FULLY STANDALONE
│   └── agency.py                  # Goal-directedness, TAME metrics - FULLY STANDALONE
│
├── classifiers/                   # STANDALONE classification
│   ├── __init__.py
│   └── signal_class.py            # SignalClass enum + classify_signal()
│
├── plugins/                       # STANDALONE plugins
│   ├── __init__.py                # Plugin registry
│   ├── base.py                    # Abstract base classes
│   ├── attractor_lock.py          # Attractor intervention - STANDALONE
│   ├── coherence_boost.py         # Memory persistence - STANDALONE
│   └── goal_director.py           # Agency enhancement - STANDALONE
│
├── training/                      # STANDALONE training utilities
│   ├── __init__.py
│   ├── reward_model.py            # ConsciousnessRewardModel - STANDALONE
│   └── preference_generator.py    # Generate DPO pairs - STANDALONE
│
├── analyzers/                     # Higher-level (composes standalone modules)
│   ├── __init__.py
│   └── trajectory.py              # Full ConsciousnessTrajectoryAnalyzer
│
├── benchmarks/                    # STANDALONE evaluation
│   ├── __init__.py
│   ├── test_suites.py             # Categorized prompt suites
│   └── profiler.py                # Model profiler
│
└── __init__.py                    # Main package with all imports
```

## 🚀 Usage Patterns

### Pattern 1: Copy Single File

Literally copy one file to another project:

```bash
# Copy just the Lyapunov metric
cp consciousness_circuit/metrics/lyapunov.py /path/to/other/project/

# Use it immediately
cd /path/to/other/project
python3 -c "from lyapunov import compute_lyapunov; print('Works!')"
```

**Example:**
```python
# In any project, just copy lyapunov.py and use it
from lyapunov import compute_lyapunov, LyapunovResult

trajectory = np.random.randn(100, 64)
result = compute_lyapunov(trajectory)

print(f"Lyapunov exponent: {result.exponent:.3f}")
print(f"Is chaotic: {result.is_chaotic}")
print(result.interpretation)
```

### Pattern 2: Import from Package

Use the convenience imports when working within the package:

```python
from consciousness_circuit.metrics import (
    compute_lyapunov,
    compute_hurst,
    compute_msd,
    compute_agency_score,
)

# Use the metrics
lyap_result = compute_lyapunov(trajectory)
hurst_result = compute_hurst(time_series)
msd_result = compute_msd(trajectory)
agency_result = compute_agency_score(trajectory)
```

### Pattern 3: Compose Custom Pipeline

Build your own analyzer combining standalone modules:

```python
from consciousness_circuit.metrics import (
    compute_lyapunov,
    compute_hurst,
    compute_msd,
    compute_agency_score,
)
from consciousness_circuit.classifiers import classify_signal

def my_custom_analyzer(trajectory):
    """Custom analysis pipeline."""
    # Compute metrics
    lyap = compute_lyapunov(trajectory)
    hurst = compute_hurst(trajectory)
    msd = compute_msd(trajectory)
    agency = compute_agency_score(trajectory)
    
    # Classify
    metrics_dict = {
        'lyapunov': lyap.exponent,
        'hurst': hurst.exponent,
        'msd_exponent': msd.diffusion_exponent,
        'agency': agency.agency,
    }
    classification = classify_signal(metrics_dict)
    
    return {
        'chaos': lyap.is_chaotic,
        'memory': hurst.has_memory,
        'motion': msd.motion_type,
        'agentic': agency.is_agentic,
        'class': classification.signal_class.value,
    }
```

### Pattern 4: Use Intervention Plugins

Apply interventions during generation:

```python
from consciousness_circuit.plugins import (
    AttractorLockPlugin,
    CoherenceBoostPlugin,
    PluginRegistry,
)
from consciousness_circuit.metrics import compute_lyapunov, compute_hurst

# Setup plugins
registry = PluginRegistry()
registry.register(AttractorLockPlugin(lyapunov_threshold=0.3))
registry.register(CoherenceBoostPlugin(hurst_threshold=0.4))

# During generation
hidden_states = model.get_hidden_states(prompt)

# Compute metrics
lyap = compute_lyapunov(hidden_states)
hurst = compute_hurst(hidden_states)
metrics = {'lyapunov': lyap.exponent, 'hurst': hurst.exponent}

# Apply interventions
modified_states, results = registry.apply_interventions(hidden_states, metrics)

# Continue generation with modified states
output = model.generate_with_states(modified_states)
```

### Pattern 5: Train with Consciousness Rewards

Use for RLHF, DPO, or ORPO training:

```python
from consciousness_circuit.training import (
    ConsciousnessRewardModel,
    RewardConfig,
    generate_preference_pairs,
)
from consciousness_circuit.metrics import (
    compute_lyapunov,
    compute_hurst,
    compute_agency_score,
)

# Configure reward weights
config = RewardConfig(
    consciousness_weight=0.4,
    stability_weight=0.2,
    memory_weight=0.2,
    agency_weight=0.2,
)

# Compute metrics for responses
responses = ["response_a", "response_b", "response_c"]
metrics_list = []

for response in responses:
    hidden_states = get_hidden_states(response)
    metrics = {
        'lyapunov': compute_lyapunov(hidden_states).exponent,
        'hurst': compute_hurst(hidden_states).exponent,
        'agency': compute_agency_score(hidden_states).agency,
    }
    metrics_list.append(metrics)

# Generate preference pairs for DPO
pairs = generate_preference_pairs(responses, metrics_list)

# Or compute rewards for RLHF
for metrics in metrics_list:
    reward = ConsciousnessRewardModel.compute_from_metrics(metrics, config)
    print(f"Reward: {reward.total_reward:.3f}")
    print(f"Explanation: {reward.explanation}")
```

### Pattern 6: Benchmark and Profile Models

Compare consciousness characteristics across models:

```python
from consciousness_circuit.benchmarks import (
    ModelProfiler,
    get_test_suite,
    get_full_benchmark,
)
from consciousness_circuit.analyzers import ConsciousnessTrajectoryAnalyzer

# Setup analyzer
analyzer = ConsciousnessTrajectoryAnalyzer()
analyzer.bind_model(model, tokenizer)

# Create profiler
profiler = ModelProfiler(analyzer)

# Get test prompts
philosophical = get_test_suite('philosophical')
reasoning = get_test_suite('reasoning')

# Profile model
profile = profiler.profile(philosophical + reasoning)

print(f"Average consciousness: {profile.avg_consciousness:.3f}")
print(f"Average agency: {profile.avg_agency:.3f}")
print(f"Most common class: {profile.most_common_class}")

# Compare with another model
other_profile = profiler.profile_other_model(other_model, other_tokenizer)
comparison = profile.compare(other_profile)

print(f"Consciousness difference: {comparison.consciousness_diff:.3f}")
print(f"Agency difference: {comparison.agency_diff:.3f}")
```

### Pattern 7: Full Pipeline (Unchanged from PR #1)

Use the complete integrated analyzer:

```python
from consciousness_circuit import ConsciousnessTrajectoryAnalyzer

analyzer = ConsciousnessTrajectoryAnalyzer()
analyzer.bind_model(model, tokenizer)

result = analyzer.deep_analyze("Let me think deeply about this question...")

print(f"Consciousness: {result.consciousness_score:.3f}")
print(f"Trajectory class: {result.trajectory_class}")
print(f"Lyapunov: {result.lyapunov:.3f}")
print(f"Agency: {result.agency_score:.3f}")
print("\nInterpretation:")
print(result.interpretation())
```

## 📚 Module Documentation

### Metrics (`metrics/`)

Each metric module provides:
- A `Result` dataclass with computed values
- A main `compute_*()` function
- Properties for interpretation (e.g., `is_chaotic`, `has_memory`)
- Full docstrings with mathematical references
- Self-test code in `__main__`

**Available metrics:**
1. **lyapunov.py** - Chaos and sensitivity (Rosenstein algorithm)
2. **hurst.py** - Long-term memory (R/S and DFA methods)
3. **msd.py** - Diffusion and motion type
4. **entropy.py** - Spectral entropy, runs test, autocorrelation
5. **agency.py** - Goal-directedness and TAME metrics

See `metrics/README.md` for detailed documentation.

### Classifiers (`classifiers/`)

Signal classification based on metrics:
- `SignalClass` enum with 9 classes
- `classify_signal()` function with fuzzy/hard modes
- `SignalClassifier` for custom thresholds

### Plugins (`plugins/`)

**Analysis plugins** (compute metrics without modification):
- Base: `AnalysisPlugin` class
- Existing: `TrajectoryPlugin`, `ChaosPlugin`, `AgencyPlugin`

**Intervention plugins** (modify hidden states):
- Base: `InterventionPlugin` class
- `AttractorLockPlugin` - Stabilize chaos by nudging toward learned attractors
- `CoherenceBoostPlugin` - Maintain memory by injecting early context
- `GoalDirectorPlugin` - Enhance agency by amplifying directional movement

**Plugin registry:**
- `PluginRegistry` for managing multiple plugins
- Run analysis plugins with `run_analysis_plugins()`
- Apply interventions with `apply_interventions()`

### Training (`training/`)

**Reward model for RLHF/DPO:**
- `RewardConfig` - Configurable weights
- `ConsciousnessRewardModel` - Compute rewards from metrics
- `compute_from_metrics()` - Standalone reward computation

**Preference generation for DPO:**
- `PreferencePair` - Chosen/rejected pair with metrics
- `generate_preference_pairs()` - Rank and create training pairs
- `rank_responses()` - Rank by consciousness metrics
- `filter_pairs_by_quality()` - Quality filtering
- `balance_preference_dataset()` - Balance dataset

### Analyzers (`analyzers/`)

Higher-level analyzer that composes standalone modules:
- `ConsciousnessTrajectoryAnalyzer` - Full trajectory analysis
- `TrajectoryAnalysisResult` - Complete result with all metrics
- Methods: `deep_analyze()`, `analyze_batch()`, `compare_prompts()`

### Benchmarks (`benchmarks/`)

**Test suites:**
- 60 categorized prompts (philosophical, factual, reasoning, creative)
- `get_test_suite(category)` - Get prompts for a category
- `get_full_benchmark()` - Get all prompts

**Model profiler:**
- `ModelProfiler` - Profile model consciousness characteristics
- `ProfileResult` - Aggregated profile with statistics
- `profile()` - Run full profile
- `compare()` - Compare two profiles

## 🧪 Testing Standalone Modules

Each standalone module includes self-tests. Run them directly:

```bash
# Test individual modules
python3 consciousness_circuit/metrics/lyapunov.py
python3 consciousness_circuit/metrics/hurst.py
python3 consciousness_circuit/metrics/msd.py
python3 consciousness_circuit/metrics/entropy.py
python3 consciousness_circuit/metrics/agency.py

python3 consciousness_circuit/classifiers/signal_class.py

python3 consciousness_circuit/plugins/base.py
python3 consciousness_circuit/plugins/attractor_lock.py
python3 consciousness_circuit/plugins/coherence_boost.py
python3 consciousness_circuit/plugins/goal_director.py

python3 consciousness_circuit/training/reward_model.py
python3 consciousness_circuit/training/preference_generator.py
```

All tests should pass with `✓ All tests completed successfully!`

## 🔄 Backward Compatibility

The modular architecture maintains full backward compatibility:

1. **Legacy imports still work:**
   ```python
   from consciousness_circuit import (
       compute_lyapunov_exponent,  # Still available
       compute_hurst_exponent,      # Still available
       SignalClass,                 # Still available
   )
   ```

2. **Old code continues to function:**
   ```python
   from consciousness_circuit import ConsciousnessTrajectoryAnalyzer
   
   analyzer = ConsciousnessTrajectoryAnalyzer()
   analyzer.bind_model(model, tokenizer)
   result = analyzer.deep_analyze("...")
   # Works exactly as before!
   ```

3. **Original files unchanged:**
   - `helios_metrics.py` - Still exists and works
   - `tame_metrics.py` - Still exists and works
   - `trajectory_wrapper.py` - Still exists and works

The new modular structure is **additive**, providing more flexibility without breaking existing code.

## 💡 Migration Guide

### For Existing Users

Your code will continue to work without changes. To adopt the modular approach:

**Before (still works):**
```python
from consciousness_circuit import compute_lyapunov_exponent

lyap = compute_lyapunov_exponent(trajectory)
```

**After (new modular way):**
```python
from consciousness_circuit.metrics import compute_lyapunov

result = compute_lyapunov(trajectory)
print(result.exponent)
print(result.is_chaotic)
print(result.interpretation)
```

### For New Projects

To use standalone modules in a new project:

1. **Copy the file:** `cp consciousness_circuit/metrics/lyapunov.py your_project/`
2. **Install numpy:** `pip install numpy`
3. **Use it:** `from lyapunov import compute_lyapunov`

That's it! No other dependencies or setup required.

## 📖 Examples

See detailed examples in:
- `metrics/README.md` - Standalone metrics usage
- Each module's docstrings - Function-level examples
- Each module's `__main__` block - Self-test examples

## 🤝 Contributing

When adding new standalone modules:

1. **Follow the zero-coupling principle** - Only numpy + stdlib
2. **Include comprehensive docstrings** - With usage examples
3. **Add self-tests** - In `__main__` block
4. **Use dataclasses** - For structured results
5. **Provide properties** - For easy interpretation
6. **Add to `__init__.py`** - For convenience imports

## 📝 License

MIT License - Same as the main Consciousness Circuit package.

## ✨ Summary

The modular architecture provides:

✅ **Flexibility** - Use just what you need
✅ **Portability** - Copy single files to any project
✅ **Simplicity** - Zero coupling, minimal dependencies
✅ **Quality** - Comprehensive docs, tests, type hints
✅ **Backward compatible** - Existing code still works
✅ **Training ready** - RLHF/DPO utilities included
✅ **Intervention capable** - Plugins for steering behavior

Perfect for research, production, and everything in between!
