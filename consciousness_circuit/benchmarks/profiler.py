"""
Model Profiler for Benchmarking
================================

Profile model behavior across test suites.
Compares trajectory dynamics, consciousness scores, and agency metrics.

Usage:
    from consciousness_circuit.benchmarks import ModelProfiler, get_test_suite
    from consciousness_circuit.analyzers import ConsciousnessTrajectoryAnalyzer
    
    # Setup
    analyzer = ConsciousnessTrajectoryAnalyzer()
    analyzer.bind_model(model, tokenizer)
    profiler = ModelProfiler(analyzer)
    
    # Profile on test suite
    prompts = get_test_suite('philosophical')
    profile = profiler.profile(prompts, name='my-model-philosophical')
    
    # Compare profiles
    profile2 = profiler.profile(get_test_suite('factual'), name='my-model-factual')
    comparison = profile.compare(profile2)
    print(comparison)

Dependencies: numpy, dataclasses, consciousness_circuit.analyzers
"""

import numpy as np
from typing import List, Dict, Optional, Any
from dataclasses import dataclass, field
from datetime import datetime


@dataclass
class ProfileResult:
    """
    Profile of model behavior on a test suite.
    
    Attributes:
        name: Profile identifier
        timestamp: When the profile was created
        num_prompts: Number of prompts analyzed
        avg_consciousness: Mean consciousness score
        avg_lyapunov: Mean Lyapunov exponent
        avg_hurst: Mean Hurst exponent
        avg_agency: Mean agency score
        trajectory_classes: Distribution of trajectory classes
        convergence_rate: Fraction of prompts showing attractor convergence
        results: Full analysis results (optional)
        metadata: Additional information
    """
    name: str
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    num_prompts: int = 0
    
    # Aggregated metrics
    avg_consciousness: float = 0.0
    std_consciousness: float = 0.0
    avg_lyapunov: float = 0.0
    std_lyapunov: float = 0.0
    avg_hurst: float = 0.0
    std_hurst: float = 0.0
    avg_agency: float = 0.0
    std_agency: float = 0.0
    avg_goal_directedness: float = 0.0
    avg_trajectory_coherence: float = 0.0
    
    # Distributions
    trajectory_classes: Dict[str, int] = field(default_factory=dict)
    convergence_rate: float = 0.0
    
    # Raw results
    results: Optional[List[Any]] = None
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def summary(self) -> str:
        """Generate human-readable summary."""
        lines = [
            f"Profile: {self.name}",
            f"Timestamp: {self.timestamp}",
            f"Prompts: {self.num_prompts}",
            "",
            "Consciousness:",
            f"  Mean: {self.avg_consciousness:.3f} ± {self.std_consciousness:.3f}",
            "",
            "Trajectory Dynamics:",
            f"  Lyapunov: {self.avg_lyapunov:.3f} ± {self.std_lyapunov:.3f}",
            f"  Hurst: {self.avg_hurst:.3f} ± {self.std_hurst:.3f}",
            "",
            "Agency:",
            f"  Score: {self.avg_agency:.3f} ± {self.std_agency:.3f}",
            f"  Goal-directedness: {self.avg_goal_directedness:.3f}",
            f"  Coherence: {self.avg_trajectory_coherence:.3f}",
            "",
            "Trajectory Classes:",
        ]
        
        for traj_class, count in sorted(self.trajectory_classes.items(), key=lambda x: x[1], reverse=True):
            pct = 100 * count / self.num_prompts if self.num_prompts > 0 else 0
            lines.append(f"  {traj_class}: {count} ({pct:.1f}%)")
        
        lines.append(f"\nConvergence Rate: {100*self.convergence_rate:.1f}%")
        
        return "\n".join(lines)
    
    def compare(self, other: 'ProfileResult') -> Dict[str, Any]:
        """
        Compare this profile to another.
        
        Args:
            other: Another ProfileResult to compare against
        
        Returns:
            Dictionary with comparison metrics
        
        Example:
            >>> comparison = profile1.compare(profile2)
            >>> print(f"Consciousness diff: {comparison['consciousness_diff']:.3f}")
        """
        return {
            'name1': self.name,
            'name2': other.name,
            'consciousness_diff': self.avg_consciousness - other.avg_consciousness,
            'lyapunov_diff': self.avg_lyapunov - other.avg_lyapunov,
            'hurst_diff': self.avg_hurst - other.avg_hurst,
            'agency_diff': self.avg_agency - other.avg_agency,
            'convergence_diff': self.convergence_rate - other.convergence_rate,
            'trajectory_class_overlap': self._compute_overlap(
                self.trajectory_classes, 
                other.trajectory_classes
            ),
        }
    
    def _compute_overlap(self, dist1: Dict[str, int], dist2: Dict[str, int]) -> float:
        """Compute overlap between two distributions."""
        all_keys = set(dist1.keys()) | set(dist2.keys())
        if not all_keys:
            return 0.0
        
        total1 = sum(dist1.values()) or 1
        total2 = sum(dist2.values()) or 1
        
        overlap = 0.0
        for key in all_keys:
            p1 = dist1.get(key, 0) / total1
            p2 = dist2.get(key, 0) / total2
            overlap += min(p1, p2)
        
        return overlap
    
    def to_dict(self) -> dict:
        """Export to dictionary (without full results)."""
        return {
            'name': self.name,
            'timestamp': self.timestamp,
            'num_prompts': self.num_prompts,
            'avg_consciousness': self.avg_consciousness,
            'std_consciousness': self.std_consciousness,
            'avg_lyapunov': self.avg_lyapunov,
            'std_lyapunov': self.std_lyapunov,
            'avg_hurst': self.avg_hurst,
            'std_hurst': self.std_hurst,
            'avg_agency': self.avg_agency,
            'std_agency': self.std_agency,
            'avg_goal_directedness': self.avg_goal_directedness,
            'avg_trajectory_coherence': self.avg_trajectory_coherence,
            'trajectory_classes': self.trajectory_classes,
            'convergence_rate': self.convergence_rate,
            'metadata': self.metadata,
        }


class ModelProfiler:
    """
    Profile model behavior across test suites.
    
    Runs systematic benchmarks and aggregates results.
    
    Example:
        >>> analyzer = ConsciousnessTrajectoryAnalyzer()
        >>> analyzer.bind_model(model, tokenizer)
        >>> profiler = ModelProfiler(analyzer)
        >>> 
        >>> # Profile on philosophical prompts
        >>> from consciousness_circuit.benchmarks import get_test_suite
        >>> prompts = get_test_suite('philosophical')
        >>> profile = profiler.profile(prompts, name='gpt2-philosophical')
        >>> print(profile.summary())
        >>> 
        >>> # Compare two profiles
        >>> prompts2 = get_test_suite('factual')
        >>> profile2 = profiler.profile(prompts2, name='gpt2-factual')
        >>> comparison = profiler.compare(profile, profile2)
    """
    
    def __init__(self, analyzer=None):
        """
        Initialize profiler.
        
        Args:
            analyzer: Optional ConsciousnessTrajectoryAnalyzer instance
        """
        self.analyzer = analyzer
        self.profiles: Dict[str, ProfileResult] = {}
    
    def set_analyzer(self, analyzer):
        """Set or update the analyzer."""
        self.analyzer = analyzer
    
    def profile(self, 
                test_suite: List[str],
                name: str,
                store_results: bool = False,
                metadata: Optional[Dict[str, Any]] = None) -> ProfileResult:
        """
        Profile model on a test suite.
        
        Args:
            test_suite: List of prompts to analyze
            name: Identifier for this profile
            store_results: Whether to keep full analysis results
            metadata: Optional additional information
        
        Returns:
            ProfileResult with aggregated metrics
        
        Example:
            >>> prompts = ["What is consciousness?", "How do you think?"]
            >>> profile = profiler.profile(prompts, name='test-profile')
            >>> print(profile.summary())
        """
        if self.analyzer is None:
            raise ValueError("No analyzer set. Call set_analyzer() first.")
        
        # Run analysis
        results = self.analyzer.analyze_batch(test_suite, include_per_token=False)
        results = [r for r in results if r is not None]
        
        if not results:
            raise ValueError("No successful analyses in test suite")
        
        # Aggregate metrics
        consciousness_scores = [r.consciousness_score for r in results]
        lyapunov_scores = [r.lyapunov for r in results]
        hurst_scores = [r.hurst for r in results]
        agency_scores = [r.agency_score for r in results]
        goal_scores = [r.goal_directedness for r in results]
        coherence_scores = [r.trajectory_coherence for r in results]
        
        # Count trajectory classes
        trajectory_classes = {}
        for r in results:
            traj_class = r.trajectory_class
            trajectory_classes[traj_class] = trajectory_classes.get(traj_class, 0) + 1
        
        # Convergence rate
        convergence_rate = sum(r.is_converging for r in results) / len(results)
        
        # Create profile
        profile = ProfileResult(
            name=name,
            num_prompts=len(results),
            avg_consciousness=float(np.mean(consciousness_scores)),
            std_consciousness=float(np.std(consciousness_scores)),
            avg_lyapunov=float(np.mean(lyapunov_scores)),
            std_lyapunov=float(np.std(lyapunov_scores)),
            avg_hurst=float(np.mean(hurst_scores)),
            std_hurst=float(np.std(hurst_scores)),
            avg_agency=float(np.mean(agency_scores)),
            std_agency=float(np.std(agency_scores)),
            avg_goal_directedness=float(np.mean(goal_scores)),
            avg_trajectory_coherence=float(np.mean(coherence_scores)),
            trajectory_classes=trajectory_classes,
            convergence_rate=convergence_rate,
            results=results if store_results else None,
            metadata=metadata or {},
        )
        
        # Store profile
        self.profiles[name] = profile
        
        return profile
    
    def compare(self, 
                profile1: ProfileResult, 
                profile2: ProfileResult) -> Dict[str, Any]:
        """
        Compare two profiles.
        
        Args:
            profile1: First profile
            profile2: Second profile
        
        Returns:
            Dictionary with comparison metrics
        
        Example:
            >>> comparison = profiler.compare(profile1, profile2)
            >>> print(f"Consciousness difference: {comparison['consciousness_diff']:.3f}")
        """
        return profile1.compare(profile2)
    
    def get_profile(self, name: str) -> Optional[ProfileResult]:
        """Retrieve a stored profile by name."""
        return self.profiles.get(name)
    
    def list_profiles(self) -> List[str]:
        """List all stored profile names."""
        return list(self.profiles.keys())
    
    def compare_all(self) -> Dict[str, Any]:
        """
        Compare all stored profiles.
        
        Returns:
            Dictionary with pairwise comparisons
        
        Example:
            >>> comparisons = profiler.compare_all()
            >>> for key, value in comparisons.items():
            >>>     print(f"{key}: {value}")
        """
        names = self.list_profiles()
        comparisons = {}
        
        for i, name1 in enumerate(names):
            for name2 in names[i+1:]:
                key = f"{name1}_vs_{name2}"
                comparisons[key] = self.compare(
                    self.profiles[name1],
                    self.profiles[name2]
                )
        
        return comparisons


__all__ = [
    'ProfileResult',
    'ModelProfiler',
]
