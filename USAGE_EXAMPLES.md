# Modular Consciousness Analysis Toolkit - Usage Examples

This document demonstrates the various ways to use the modular consciousness circuit components.

## Table of Contents

1. [Standalone Usage (Copy-Paste Ready)](#standalone-usage)
2. [Composed Pipeline](#composed-pipeline)
3. [Integration Examples](#integration-examples)
4. [Full API Examples](#full-api-examples)

---

## Standalone Usage (Copy-Paste Ready)

Each module works independently with just numpy.

### Example 1: Just Lyapunov Exponent

```python
# Copy lyapunov.py to your project
from lyapunov import compute_lyapunov
import numpy as np

# Your trajectory data
x = np.cumsum(np.random.randn(200))
y = np.cumsum(np.random.randn(200))

# Compute chaos indicator
lyap = compute_lyapunov(x, y)

if lyap > 0.3:
    print("System is chaotic!")
elif lyap < -0.1:
    print("System is stable/convergent")
else:
    print("System is neutral/random")
```

### Example 2: Just Classification

```python
# Copy signal_class.py to your project
from signal_class import classify_signal

metrics = {
    'lyapunov': 0.05,
    'hurst': 0.52,
    'diffusion_exponent': 1.1,
}

result = classify_signal(metrics)
print(f"Signal type: {result.signal_class.name}")
print(f"Confidence: {result.confidence:.2f}")
```

### Example 3: Just Reward Model

```python
# Copy reward_model.py to your project
from reward_model import ConsciousnessRewardModel

metrics = {
    'consciousness_score': 0.75,
    'lyapunov': -0.1,
    'hurst': 0.6,
    'agency_score': 0.7,
}

reward = ConsciousnessRewardModel.compute_from_metrics(metrics)
print(f"Training reward: {reward:.3f}")
```

---

## Composed Pipeline

Combine multiple metrics for comprehensive analysis.

### Custom Analysis Pipeline

```python
from consciousness_circuit.metrics import (
    compute_lyapunov,
    compute_hurst,
    compute_msd,
    compute_agency_score,
)
from consciousness_circuit.classifiers import classify_signal
from consciousness_circuit.training import ConsciousnessRewardModel

import numpy as np


def analyze_trajectory(x, y):
    """Complete trajectory analysis pipeline."""
    
    # Step 1: Compute all metrics
    metrics = {}
    
    # Chaos detection
    metrics['lyapunov'] = compute_lyapunov(x, y)
    
    # Memory/persistence (from 1D projection)
    sequence = np.sqrt(x**2 + y**2)  # Distance from origin
    metrics['hurst'] = compute_hurst(sequence)
    
    # Diffusion analysis
    metrics['diffusion_exponent'] = compute_diffusion_exponent(x, y)
    
    # Goal-directedness
    trajectory = np.column_stack([x, y])
    metrics['agency_score'] = compute_agency_score(trajectory)
    
    # Step 2: Classify signal type
    classification = classify_signal(metrics)
    metrics['trajectory_class'] = classification.signal_class.name
    
    # Step 3: Compute reward (for training)
    metrics['consciousness_score'] = 0.7  # Could come from another analysis
    reward = ConsciousnessRewardModel.compute_from_metrics(metrics)
    
    # Step 4: Generate report
    report = {
        'metrics': metrics,
        'classification': classification.signal_class.name,
        'confidence': classification.confidence,
        'reward': reward,
        'interpretation': {
            'chaos': 'high' if metrics['lyapunov'] > 0.3 else 'low',
            'memory': 'high' if metrics['hurst'] > 0.6 else 'low',
            'agency': 'high' if metrics['agency_score'] > 0.6 else 'low',
        }
    }
    
    return report


# Usage
x = np.cumsum(np.random.randn(200)) * 0.1
y = np.cumsum(np.random.randn(200)) * 0.1

report = analyze_trajectory(x, y)

print("Trajectory Analysis Report")
print("=" * 50)
print(f"Classification: {report['classification']}")
print(f"Confidence: {report['confidence']:.2f}")
print(f"Training Reward: {report['reward']:.3f}")
print(f"\nMetrics:")
for key, value in report['metrics'].items():
    if isinstance(value, float):
        print(f"  {key}: {value:.3f}")
    else:
        print(f"  {key}: {value}")
```

---

## Integration Examples

### Example 1: Feature Extraction for ML

```python
from consciousness_circuit.metrics import (
    compute_lyapunov,
    compute_hurst,
    compute_msd,
    compute_agency_score,
)
import numpy as np


def extract_trajectory_features(trajectories):
    """Extract features from multiple trajectories for ML."""
    
    features = []
    
    for x, y in trajectories:
        feature_dict = {
            'lyapunov': compute_lyapunov(x, y),
            'hurst': compute_hurst(np.sqrt(x**2 + y**2)),
            'diffusion_exp': compute_diffusion_exponent(x, y),
            'agency': compute_agency_score(np.column_stack([x, y])),
            'path_length': np.sum(np.sqrt(np.diff(x)**2 + np.diff(y)**2)),
            'displacement': np.sqrt((x[-1] - x[0])**2 + (y[-1] - y[0])**2),
        }
        
        features.append(feature_dict)
    
    return features


# Generate sample trajectories
trajectories = [
    (np.cumsum(np.random.randn(100)), np.cumsum(np.random.randn(100)))
    for _ in range(10)
]

features = extract_trajectory_features(trajectories)

# Convert to array for sklearn
import pandas as pd
df = pd.DataFrame(features)
print(df.describe())
```

### Example 2: Real-Time Monitoring with Plugins

```python
from consciousness_circuit.plugins import AttractorLockPlugin
from consciousness_circuit.metrics import compute_lyapunov
import numpy as np


class TrajectoryMonitor:
    """Monitor trajectory in real-time and intervene if needed."""
    
    def __init__(self):
        self.plugin = AttractorLockPlugin(
            lyapunov_threshold=0.3,
            nudge_strength=0.1
        )
        self.history = []
    
    def step(self, hidden_states, trajectory_window):
        """Process one step of trajectory."""
        
        # Compute current metrics
        x, y = trajectory_window[-50:, 0], trajectory_window[-50:, 1]
        lyap = compute_lyapunov(x, y)
        
        metrics = {'lyapunov': lyap}
        self.history.append(metrics)
        
        # Check if intervention needed
        if self.plugin.should_intervene(metrics):
            print(f"⚠️  High chaos detected (λ={lyap:.3f}), intervening...")
            modified_states = self.plugin.intervene(hidden_states, metrics)
            return modified_states, True
        
        return hidden_states, False
    
    def learn_from_success(self, hidden_states, quality):
        """Learn good attractors from successful states."""
        self.plugin.learn_attractor(hidden_states, quality)


# Usage in simulation
monitor = TrajectoryMonitor()

for step in range(100):
    # Simulate hidden states
    hidden_states = np.random.randn(256)
    
    # Simulate trajectory
    trajectory = np.random.randn(100, 2)
    
    # Monitor and potentially intervene
    modified_states, intervened = monitor.step(hidden_states, trajectory)
    
    if intervened:
        # Use modified states
        hidden_states = modified_states
    
    # Learn from good states
    if step % 10 == 0:
        monitor.learn_from_success(hidden_states, quality=0.8)

stats = monitor.plugin.get_statistics()
print(f"Total interventions: {stats['intervention_count']}")
print(f"Attractors learned: {stats['attractors_stored']}")
```

### Example 3: Training Loop Integration

```python
from consciousness_circuit.training import ConsciousnessRewardModel, RewardConfig
from consciousness_circuit.metrics import (
    compute_lyapunov,
    compute_hurst,
    compute_agency_score,
)
import numpy as np


class ConsciousnessTrainer:
    """Trainer that uses consciousness metrics as reward signal."""
    
    def __init__(self):
        # Custom reward weights
        config = RewardConfig(
            consciousness_weight=0.4,
            stability_weight=0.3,
            memory_weight=0.2,
            agency_weight=0.1,
        )
        
        self.reward_model = ConsciousnessRewardModel(config=config)
    
    def compute_trajectory_reward(self, trajectory):
        """Compute reward from trajectory."""
        
        x, y = trajectory[:, 0], trajectory[:, 1]
        
        # Compute all metrics
        metrics = {
            'consciousness_score': 0.7,  # From your consciousness measurement
            'lyapunov': compute_lyapunov(x, y),
            'hurst': compute_hurst(np.linalg.norm(trajectory, axis=1)),
            'agency_score': compute_agency_score(trajectory),
        }
        
        # Classify for bonus
        if metrics['lyapunov'] < -0.1:
            metrics['trajectory_class'] = 'ATTRACTOR'
        
        return self.reward_model.compute_from_metrics(metrics, self.reward_model.config)
    
    def train_step(self, trajectories):
        """One training step with multiple trajectories."""
        
        rewards = []
        
        for traj in trajectories:
            reward = self.compute_trajectory_reward(traj)
            rewards.append(reward)
        
        # Use rewards for policy gradient, DPO, etc.
        avg_reward = np.mean(rewards)
        
        return avg_reward, rewards


# Usage
trainer = ConsciousnessTrainer()

# Generate sample trajectories (in practice, from model)
trajectories = [
    np.cumsum(np.random.randn(100, 2) * 0.1, axis=0)
    for _ in range(5)
]

avg_reward, rewards = trainer.train_step(trajectories)

print(f"Average reward: {avg_reward:.3f}")
print(f"Individual rewards: {[f'{r:.3f}' for r in rewards]}")
```

---

## Full API Examples

### Complete Analysis with All Metrics

```python
from consciousness_circuit.metrics import (
    LyapunovAnalyzer,
    HurstAnalyzer,
    MSDAnalyzer,
    TAMEMetrics,
    EntropyAnalyzer,
)
import numpy as np


def comprehensive_analysis(trajectory):
    """Analyze trajectory with all available metrics."""
    
    x, y = trajectory[:, 0], trajectory[:, 1]
    sequence = np.linalg.norm(trajectory, axis=1)
    
    results = {}
    
    # 1. Lyapunov (chaos)
    lyap = LyapunovAnalyzer()
    results['lyapunov'] = lyap.analyze((x, y))
    
    # 2. Hurst (memory)
    hurst = HurstAnalyzer(method='dfa')
    results['hurst'] = hurst.analyze(sequence)
    
    # 3. MSD (diffusion)
    msd = MSDAnalyzer()
    results['msd'] = msd.analyze((x, y))
    
    # 4. Agency (goal-directedness)
    tame = TAMEMetrics()
    results['agency'] = tame.analyze(trajectory)
    
    # 5. Entropy (randomness)
    entropy = EntropyAnalyzer()
    results['entropy'] = entropy.analyze(sequence)
    
    # Generate report
    print("=" * 70)
    print("COMPREHENSIVE TRAJECTORY ANALYSIS")
    print("=" * 70)
    
    print("\n1. CHAOS ANALYSIS (Lyapunov)")
    print(f"   Value: {results['lyapunov'].value:.4f}")
    print(f"   Interpretation: {results['lyapunov'].interpretation}")
    print(f"   Confidence: {results['lyapunov'].confidence:.2f}")
    
    print("\n2. MEMORY ANALYSIS (Hurst)")
    print(f"   Value: {results['hurst'].value:.4f}")
    print(f"   Interpretation: {results['hurst'].interpretation}")
    print(f"   Has long-term memory: {results['hurst'].has_memory}")
    
    print("\n3. DIFFUSION ANALYSIS (MSD)")
    print(f"   Exponent: {results['msd'].diffusion_exponent:.4f}")
    print(f"   Motion type: {results['msd'].motion_type}")
    print(f"   Coefficient: {results['msd'].diffusion_coefficient:.4f}")
    
    print("\n4. AGENCY ANALYSIS (TAME)")
    print(f"   Overall score: {results['agency'].score:.4f}")
    print(f"   Goal directedness: {results['agency'].goal_directedness:.4f}")
    print(f"   Path efficiency: {results['agency'].path_efficiency:.4f}")
    print(f"   Adaptability: {results['agency'].adaptability:.4f}")
    print(f"   Interpretation: {results['agency'].interpretation}")
    
    print("\n5. ENTROPY ANALYSIS")
    entropy_data = results['entropy']
    print(f"   Spectral entropy: {entropy_data['spectral_entropy']:.4f}")
    print(f"   Normalized entropy: {entropy_data['normalized_entropy']:.4f}")
    print(f"   Is random: {entropy_data['is_random_entropy']}")
    print(f"   Is structured: {entropy_data['is_structured']}")
    
    return results


# Usage
np.random.seed(42)
trajectory = np.cumsum(np.random.randn(200, 2) * 0.1, axis=0)

results = comprehensive_analysis(trajectory)
```

---

## Tips for Using Standalone Components

### 1. Copy Single Files

Each metric file is completely self-contained:

```bash
# Copy just what you need
cp consciousness_circuit/metrics/lyapunov.py ~/my_project/
cp consciousness_circuit/classifiers/signal_class.py ~/my_project/
```

### 2. Direct Import (No Package Overhead)

```python
import importlib.util

spec = importlib.util.spec_from_file_location("lyapunov", "lyapunov.py")
lyapunov = importlib.util.module_from_spec(spec)
spec.loader.exec_module(lyapunov)

# Use it
result = lyapunov.compute_lyapunov(x, y)
```

### 3. Create Custom Combinations

Mix and match only what you need:

```python
from consciousness_circuit.metrics import compute_lyapunov, compute_hurst
from consciousness_circuit.classifiers import classify_signal

def my_custom_analysis(data):
    metrics = {
        'lyapunov': compute_lyapunov(data['x'], data['y']),
        'hurst': compute_hurst(data['sequence']),
    }
    return classify_signal(metrics)
```

### 4. Zero Coupling Guarantee

No modules import torch, transformers, or other heavy dependencies. Only numpy is required.

---

## License

MIT - Same as parent package
