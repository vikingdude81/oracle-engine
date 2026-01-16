# Standalone Metrics - Zero Coupling Design

Each metric module in this directory is **fully standalone** and can be copied to any project independently. They only depend on `numpy`.

## Design Principle: Zero Coupling

✅ **Each file:**
- Only imports `numpy` (and standard library: `typing`, `dataclasses`)
- Does NOT import from sibling modules
- Has complete docstrings with usage examples
- Returns structured dataclass results
- Works if you literally copy just that one file

❌ **Each file does NOT:**
- Import from other metrics modules
- Depend on package structure
- Require any setup or configuration

## Available Metrics

### 1. `lyapunov.py` - Chaos and Sensitivity

Measures sensitivity to initial conditions using Rosenstein algorithm.

```python
from lyapunov import compute_lyapunov, LyapunovResult

result = compute_lyapunov(trajectory)
print(f"Lyapunov exponent: {result.exponent:.3f}")
print(f"Is chaotic: {result.is_chaotic}")
print(result.interpretation)
```

**Properties:**
- `λ > 0`: Chaotic (exponential divergence)
- `λ ≈ 0`: Neutral/marginally stable
- `λ < 0`: Stable (converging)

### 2. `hurst.py` - Long-Term Memory

Measures persistence and long-range dependence in time series.

```python
from hurst import compute_hurst, HurstResult

result = compute_hurst(time_series, method='rs')  # or method='dfa'
print(f"Hurst exponent: {result.exponent:.3f}")
print(f"Has memory: {result.has_memory}")
print(result.interpretation)
```

**Properties:**
- `H > 0.5`: Persistent/trending (past trends continue)
- `H = 0.5`: Random walk (no memory)
- `H < 0.5`: Anti-persistent/mean-reverting

**Methods:**
- `'rs'`: Rescaled Range analysis (classical)
- `'dfa'`: Detrended Fluctuation Analysis (more robust)

### 3. `msd.py` - Diffusion and Motion Type

Computes Mean Squared Displacement to characterize trajectory spreading.

```python
from msd import compute_msd, MSDResult

result = compute_msd(trajectory)
print(f"Motion type: {result.motion_type}")
print(f"Diffusion exponent: {result.diffusion_exponent:.2f}")
print(result.interpretation)
```

**Motion Types:**
- `α ≈ 2`: Ballistic (directed motion)
- `α ≈ 1`: Diffusive (random walk)
- `α < 1`: Subdiffusive (constrained)
- `α < 0.5`: Confined (bounded)

### 4. `entropy.py` - Randomness and Structure

Spectral entropy, runs test, and autocorrelation for time series analysis.

```python
from entropy import (
    compute_spectral_entropy,
    compute_runs_test,
    compute_autocorrelation
)

# Spectral entropy (signal regularity)
entropy_result = compute_spectral_entropy(signal)
print(f"Entropy: {entropy_result.entropy:.3f}")
print(f"Is random: {entropy_result.is_random}")

# Runs test (randomness test)
runs_result = compute_runs_test(sequence)
print(f"Is random: {runs_result.is_random}")

# Autocorrelation (memory detection)
acf_result = compute_autocorrelation(signal)
print(f"Has memory: {acf_result.has_memory}")
print(f"Memory length: {acf_result.memory_length}")
```

### 5. `agency.py` - Goal-Directedness

Measures agentic, goal-directed behavior in trajectories.

```python
from agency import compute_agency_score, compute_tame_metrics, AgencyResult

# Basic agency
result = compute_agency_score(trajectory)
print(f"Agency: {result.agency:.3f}")
print(f"Is agentic: {result.is_agentic}")
print(result.interpretation)

# Full TAME metrics
tame = compute_tame_metrics(trajectory)
print(f"Agency: {tame.agency_score:.3f}")
print(f"Attractor strength: {tame.attractor_strength:.3f}")
print(f"Goal-directedness: {tame.goal_directedness:.3f}")
print(f"Overall TAME: {tame.overall_tame_score:.3f}")
```

**TAME = Trajectory Analysis for Meta-Emergence**

## Usage Patterns

### Pattern 1: Copy Single File

Literally copy one file to another project:

```bash
# Copy just the Lyapunov metric
cp lyapunov.py /path/to/other/project/

# Use it
cd /path/to/other/project
python3 -c "from lyapunov import compute_lyapunov; print('Works!')"
```

### Pattern 2: Import from Package

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
```

### Pattern 3: Compose Custom Pipeline

```python
from consciousness_circuit.metrics import (
    compute_lyapunov,
    compute_hurst,
    compute_msd,
    compute_agency_score,
)

def my_custom_analyzer(trajectory):
    """Custom analysis combining multiple metrics."""
    lyap = compute_lyapunov(trajectory)
    hurst = compute_hurst(trajectory)
    msd = compute_msd(trajectory)
    agency = compute_agency_score(trajectory)
    
    return {
        'chaos': lyap.is_chaotic,
        'memory': hurst.has_memory,
        'motion': msd.motion_type,
        'agentic': agency.is_agentic,
    }
```

## Self-Tests

Each module includes self-tests. Run them directly:

```bash
python3 lyapunov.py
python3 hurst.py
python3 msd.py
python3 entropy.py
python3 agency.py
```

## Dependencies

**Only numpy required!**

```bash
pip install numpy
```

No other dependencies. No configuration. No setup.

## Examples

See docstrings in each file for detailed examples and mathematical references.

## Design Philosophy

These modules follow the **"copy-paste reusable"** design philosophy:

1. **Self-contained**: Everything needed is in one file
2. **Well-documented**: Comprehensive docstrings with examples
3. **Tested**: Self-test code included in each module
4. **Backward compatible**: Includes aliases for old function names
5. **Type-annotated**: Clear type hints throughout
6. **Scientific**: Includes references to original papers

This makes the code maximally reusable while maintaining high quality.
