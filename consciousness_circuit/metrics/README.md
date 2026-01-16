# Consciousness Circuit Metrics

Standalone metric modules for analyzing trajectories, time series, and dynamical systems.

## Key Features

✅ **Zero coupling** - No dependencies on torch, transformers, or other consciousness_circuit modules  
✅ **Fully standalone** - Each module can be copied to any project  
✅ **Consistent API** - Same patterns across all metrics  
✅ **Type hints** - Full typing for IDE support  
✅ **Dataclass results** - Structured output with properties  

## Metrics Overview

### 1. Lyapunov Exponent (`lyapunov.py`)

Measures sensitivity to initial conditions and chaos in dynamical systems.

**What it measures:**
- λ > 0: Chaotic (exponential divergence)
- λ < 0: Stable (convergence to attractor)
- λ ≈ 0: Neutral (random walk)

**Usage:**
```python
from consciousness_circuit.metrics.lyapunov import compute_lyapunov

# From 2D trajectory
x = [1.0, 1.1, 1.3, 1.8, 2.5, ...]
y = [0.5, 0.6, 0.9, 1.2, 1.4, ...]
lyap = compute_lyapunov(x, y)

# From 1D sequence (uses time-delay embedding)
sequence = [1, 2, 3, 5, 8, 13, ...]
lyap = compute_lyapunov_1d(sequence)

# Full analysis with diagnostics
from consciousness_circuit.metrics.lyapunov import LyapunovAnalyzer
analyzer = LyapunovAnalyzer()
result = analyzer.analyze((x, y))
print(f"Value: {result.value:.3f}")
print(f"Interpretation: {result.interpretation}")
print(f"Is chaotic: {result.is_chaotic}")
```

### 2. Hurst Exponent (`hurst.py`)

Measures long-term memory and persistence in time series.

**What it measures:**
- H = 0.5: Random walk (no memory)
- H > 0.5: Trending/persistent (positive autocorrelation)
- H < 0.5: Mean-reverting (negative autocorrelation)

**Usage:**
```python
from consciousness_circuit.metrics.hurst import compute_hurst

sequence = [1.2, 1.5, 1.3, 1.8, 2.1, ...]
h = compute_hurst(sequence)

# Full analysis
from consciousness_circuit.metrics.hurst import HurstAnalyzer
analyzer = HurstAnalyzer(method='dfa')  # or 'rs'
result = analyzer.analyze(sequence)
print(f"Hurst: {result.value:.3f}")
print(f"Has memory: {result.has_memory}")
print(f"Is trending: {result.is_trending}")
```

### 3. Mean Squared Displacement (`msd.py`)

Analyzes diffusion and motion patterns.

**What it measures:**
- α = 1: Normal diffusion (Brownian motion)
- α > 1: Superdiffusion (faster than random walk)
- α < 1: Subdiffusion (constrained motion)
- α ≈ 2: Ballistic motion

**Usage:**
```python
from consciousness_circuit.metrics.msd import compute_msd, compute_diffusion_exponent

x = [0, 1, 2, 4, 7, ...]
y = [0, 0, 1, 2, 4, ...]

# Compute MSD at various lags
lags, msd_values = compute_msd(x, y)

# Get diffusion exponent
alpha = compute_diffusion_exponent(x, y)

# Full analysis
from consciousness_circuit.metrics.msd import analyze_msd
result = analyze_msd(x, y)
print(f"Exponent: {result.diffusion_exponent:.3f}")
print(f"Motion type: {result.motion_type}")
print(f"Is ballistic: {result.is_ballistic}")
```

### 4. Entropy (`entropy.py`)

Measures randomness and structure in sequences.

**What it measures:**
- High entropy: Random/unstructured
- Low entropy: Periodic/structured
- Also includes runs test for randomness

**Usage:**
```python
from consciousness_circuit.metrics.entropy import (
    compute_spectral_entropy,
    compute_runs_test,
    compute_autocorrelation
)

sequence = [1, 2, 1, 3, 2, 4, ...]

# Spectral entropy
entropy = compute_spectral_entropy(sequence)

# Runs test for randomness
runs_result = compute_runs_test(sequence)
print(f"Is random: {runs_result.is_random}")
print(f"p-value: {runs_result.p_value:.3f}")

# Autocorrelation
acf = compute_autocorrelation(sequence, max_lag=20)
```

### 5. Agency (`agency.py`)

Measures goal-directedness and purposeful behavior.

**What it measures:**
- Goal directedness: Progress toward goal
- Path efficiency: Direct path vs actual path
- Adaptability: Course correction ability
- Persistence: Consistency of direction

**Usage:**
```python
from consciousness_circuit.metrics.agency import compute_agency_score, TAMEMetrics

trajectory = [[0, 0], [1, 1], [2, 2], [3, 3]]  # Moving toward goal
goal = [5, 5]

# Simple score
score = compute_agency_score(trajectory, goal)

# Full TAME analysis
tame = TAMEMetrics()
result = tame.analyze(trajectory, goal)
print(f"Overall score: {result.score:.3f}")
print(f"Goal directedness: {result.goal_directedness:.3f}")
print(f"Path efficiency: {result.path_efficiency:.3f}")
print(f"Is agentic: {result.is_agentic}")
```

## Copy-Paste Ready

Each metric file is **completely self-contained**. You can literally copy any `.py` file to another project:

```bash
# Copy just the Lyapunov calculator
cp consciousness_circuit/metrics/lyapunov.py ~/my_other_project/

# Use it immediately (only needs numpy)
from lyapunov import compute_lyapunov
```

## Dependencies

**Required:**
- `numpy` - All metrics use NumPy arrays

**Not required:**
- ❌ torch
- ❌ transformers  
- ❌ Other consciousness_circuit modules

## Testing

Run standalone tests:

```bash
python test_standalone.py
```

Or test individual modules:

```python
# Direct import without package overhead
import importlib.util
spec = importlib.util.spec_from_file_location("lyapunov", "consciousness_circuit/metrics/lyapunov.py")
lyapunov = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lyapunov)

# Now use it
lyap = lyapunov.compute_lyapunov(x, y)
```

## Design Philosophy

1. **Standalone** - Works independently, can be imported alone
2. **Zero coupling** - No hard dependencies on sibling modules
3. **Consistent API** - Same patterns across all modules
4. **Composable** - Can be combined into pipelines OR used individually
5. **Repo-portable** - Easy to copy single files to other projects

## Use Cases

### Research & Analysis
```python
# Analyze experimental data
from consciousness_circuit.metrics import compute_lyapunov, compute_hurst

lyap = compute_lyapunov(x_data, y_data)
hurst = compute_hurst(time_series)

if lyap > 0.3 and hurst > 0.6:
    print("System shows chaotic behavior with long-term memory")
```

### Machine Learning
```python
# Feature extraction from trajectories
from consciousness_circuit.metrics import compute_msd, compute_agency_score

features = {
    'diffusion_exponent': compute_diffusion_exponent(x, y),
    'agency_score': compute_agency_score(trajectory),
}
```

### Model Training
```python
# Reward signals for RL
from consciousness_circuit.training import ConsciousnessRewardModel

metrics = {
    'lyapunov': -0.1,  # Stable
    'hurst': 0.6,      # Persistent
    'agency_score': 0.7,  # Goal-directed
}

reward = ConsciousnessRewardModel.compute_from_metrics(metrics)
```

## API Reference

All metrics follow this pattern:

```python
# Simple function for quick use
value = compute_metric(data, **kwargs)

# Analyzer class for advanced use
analyzer = MetricAnalyzer(**config)
result = analyzer.analyze(data)  # Returns dataclass with properties
```

Result dataclasses always include:
- `.value` - The computed metric
- Properties like `.is_chaotic`, `.is_trending`, etc.
- `.interpretation` - Human-readable description

## Contributing

When adding new metrics, follow these principles:

1. ✅ No imports from other consciousness_circuit modules
2. ✅ Only numpy as core dependency
3. ✅ Return dataclass with properties
4. ✅ Include docstring with usage example
5. ✅ Add to `metrics/__init__.py`

## License

MIT - Same as parent package
