# Trajectory Analysis Integration

This implementation adds trajectory analysis capabilities to the consciousness circuit, combining consciousness measurement with dynamics analysis.

## New Components

### 1. `helios_metrics.py` - Chaos and Dynamics Analysis

Mathematical tools from dynamical systems theory:

- **Lyapunov Exponent**: Measures chaos/sensitivity (λ > 0 = chaotic, λ < 0 = stable)
- **Hurst Exponent**: Measures long-term memory (H > 0.5 = persistent, H < 0.5 = anti-persistent)
- **Mean Squared Displacement (MSD)**: Tracks diffusion patterns
- **Signal Classification**: Categorizes as NOISE, DRIFT, ATTRACTOR, PERIODIC, CHAOTIC, etc.

### 2. `tame_metrics.py` - Agency and Goal-Directedness

Metrics for meta-cognitive behavior:

- **Agency Score**: Measures goal-directed vs reactive behavior
- **Attractor Convergence**: Detects if states converge to coherent patterns
- **Goal-Directedness**: Quantifies purposeful movement toward goals
- **Trajectory Coherence**: Measures structured vs random motion

### 3. `trajectory_wrapper.py` - Main Integration

`ConsciousnessTrajectoryAnalyzer` combines all metrics:

```python
from consciousness_circuit import ConsciousnessTrajectoryAnalyzer

analyzer = ConsciousnessTrajectoryAnalyzer()
analyzer.bind_model(model, tokenizer)

result = analyzer.deep_analyze("Let me think about this...")
print(result.interpretation())
```

### 4. `plugins/` - Extensible Architecture

Plugin system for adding new analysis methods:
- `AnalysisPlugin` base class
- `TrajectoryPlugin` for MSD analysis
- `ChaosPlugin` for Lyapunov/Hurst
- `AgencyPlugin` for goal-directedness

## Integration with Existing Code

### `universal.py` Updates

```python
# Optional trajectory analysis in measurement
result = circuit.measure(
    model, tokenizer, prompt,
    include_trajectory=True  # New parameter
)

# Access trajectory metrics
if result.trajectory_metrics:
    print(f"Lyapunov: {result.trajectory_metrics['lyapunov']}")
    print(f"Agency: {result.trajectory_metrics['agency_score']}")
```

### New CLI Command

```bash
# Deep trajectory analysis
consciousness-trajectory "prompt here" \
    --model Qwen/Qwen2.5-0.5B-Instruct \
    --full-analysis \
    --json
```

## Example Usage

### Basic Analysis

```python
from consciousness_circuit import ConsciousnessTrajectoryAnalyzer

analyzer = ConsciousnessTrajectoryAnalyzer()
analyzer.bind_model(model, tokenizer)

result = analyzer.deep_analyze("Reflect on the nature of consciousness...")

print(f"Consciousness: {result.consciousness_score:.3f}")
print(f"Trajectory: {result.trajectory_class}")
print(f"Lyapunov (chaos): {result.lyapunov:.4f}")
print(f"Hurst (memory): {result.hurst:.4f}")
print(f"Agency: {result.agency_score:.4f}")
```

### Batch Analysis

```python
prompts = [
    "What is consciousness?",
    "2 + 2 = ?",
    "Let me carefully consider this..."
]

results = analyzer.analyze_batch(prompts)

for prompt, result in zip(prompts, results):
    if result:
        print(f"{prompt[:30]}: {result.trajectory_class}")
```

### Comparative Analysis

```python
comparison = analyzer.compare_prompts([
    "Quick answer: 42",
    "Let me think deeply about this question..."
])

print(f"Average consciousness: {comparison['avg_consciousness']:.3f}")
print(f"Average agency: {comparison['avg_agency']:.3f}")
```

## Interpretation Guide

### Trajectory Classes

- **BALLISTIC**: Directed, purposeful reasoning
- **DIFFUSIVE**: Exploratory, uncertain
- **ATTRACTOR**: Converging to coherent pattern
- **CHAOTIC**: Highly sensitive, unstable
- **DRIFT**: Slow wandering, no clear direction
- **CONFINED**: Bounded, repetitive

### Key Metrics

1. **Lyapunov Exponent (λ)**
   - λ > 0.5: High chaos, exploring solution space
   - λ ≈ 0: Neutral, balanced
   - λ < -0.3: Stable, converging

2. **Hurst Exponent (H)**
   - H > 0.6: Persistent, has memory
   - H ≈ 0.5: Random walk
   - H < 0.4: Anti-persistent, mean-reverting

3. **Agency Score**
   - High (> 0.6): Goal-directed behavior
   - Medium (0.3-0.6): Mixed behavior  
   - Low (< 0.3): Reactive, not goal-directed

## Technical Details

### Algorithms Implemented

1. **Rosenstein Algorithm** (Lyapunov)
   - Tracks divergence of nearest neighbors
   - Robust to noise and short trajectories

2. **R/S Analysis** (Hurst)
   - Rescaled range method
   - Detects long-term correlations

3. **Mean Squared Displacement**
   - ⟨|r(t+τ) - r(t)|²⟩ over time lags
   - Classifies motion type by exponent

4. **Attractor Detection**
   - Variance reduction analysis
   - Convergence trend fitting

## Dependencies

- numpy >= 1.20.0
- scipy >= 1.7.0 (for signal processing)
- torch >= 2.0.0
- transformers >= 4.30.0

## Testing

Run the demo to see trajectory analysis in action:

```bash
cd consciousness_circuit
python demo_trajectory.py
```

This demonstrates the analysis of:
- Ballistic (directed) motion
- Diffusive (random walk) motion
- Confined (oscillatory) motion
- Converging (attractor) motion

## Future Extensions

The plugin architecture allows easy addition of:
- Fractal dimension analysis
- Entropy measures
- Phase space reconstruction
- Recurrence quantification
- Transfer entropy between layers

## References

- Rosenstein et al. (1993) - Lyapunov exponent estimation
- Hurst (1951) - R/S analysis for long-term dependence
- Mean Squared Displacement in stochastic processes
- TAME framework for meta-cognitive emergence
