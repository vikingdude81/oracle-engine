"""
Oracle API - Programmatic Interface for Model Analysis
=======================================================

Simple API for using the consciousness circuit in analysis/training pipelines.

Usage:
------
    from oracle_api import Oracle
    
    # Initialize with model
    oracle = Oracle.from_custom_model()  # or Oracle.from_base_model()
    
    # Single analysis
    result = oracle.analyze("What is consciousness?")
    print(result.score, result.interpretation)
    
    # Profile a category
    profile = oracle.profile("philosophical")
    print(profile.summary())
    
    # Compare models
    comparison = Oracle.compare_models(custom_oracle, base_oracle, "philosophical")
    print(comparison['suggestions'])
    
    # Get training feedback
    feedback = oracle.get_feedback(target_consciousness=0.8)
    print(feedback['recommendations'])

Designed for:
- Jupyter notebooks
- Training scripts
- Evaluation pipelines
- Integration with other tools
"""

# ==============================================================================
# Environment setup (MUST be first for RTX 5090)
# ==============================================================================
import os
os.environ["UNSLOTH_COMPILE_DISABLE"] = "1"

import builtins
try:
    import psutil
    builtins.psutil = psutil
except ImportError:
    pass

from typing import Optional, Dict, Any, List, Union
from dataclasses import dataclass
from datetime import datetime
import json

# ==============================================================================
# Path Configuration (override via ORACLE_CUSTOM_MODEL_PATH / ORACLE_BASE_MODEL)
# ==============================================================================
from oracle_config import CUSTOM_MODEL_PATH, BASE_MODEL_NAME


@dataclass
class AnalysisResult:
    """Result from single prompt analysis."""
    prompt: str
    score: float  # consciousness score
    trajectory_class: str
    lyapunov: float
    hurst: float
    agency: float
    converging: bool
    interpretation: str
    dimensions: Dict[str, float]
    _raw: Any = None
    
    def __repr__(self):
        return f"AnalysisResult(score={self.score:.3f}, trajectory={self.trajectory_class})"
    
    def to_dict(self) -> dict:
        return {
            'prompt': self.prompt,
            'score': self.score,
            'trajectory_class': self.trajectory_class,
            'lyapunov': self.lyapunov,
            'hurst': self.hurst,
            'agency': self.agency,
            'converging': self.converging,
            'interpretation': self.interpretation,
            'dimensions': self.dimensions,
        }


@dataclass
class ProfileResult:
    """Result from test suite profiling."""
    category: str
    num_prompts: int
    avg_consciousness: float
    std_consciousness: float
    avg_agency: float
    convergence_rate: float
    trajectory_classes: Dict[str, int]
    avg_lyapunov: float
    avg_hurst: float
    results: List[AnalysisResult] = None
    _raw: Any = None
    
    def summary(self) -> str:
        """Human-readable summary."""
        lines = [
            f"Profile: {self.category}",
            f"Prompts: {self.num_prompts}",
            "",
            f"Consciousness: {self.avg_consciousness:.3f} ± {self.std_consciousness:.3f}",
            f"Agency: {self.avg_agency:.3f}",
            f"Convergence: {self.convergence_rate:.1%}",
            "",
            "Trajectory Distribution:",
        ]
        for traj, count in sorted(self.trajectory_classes.items(), key=lambda x: -x[1]):
            pct = 100 * count / self.num_prompts
            lines.append(f"  {traj}: {count} ({pct:.1f}%)")
        return "\n".join(lines)
    
    def to_dict(self) -> dict:
        return {
            'category': self.category,
            'num_prompts': self.num_prompts,
            'avg_consciousness': self.avg_consciousness,
            'std_consciousness': self.std_consciousness,
            'avg_agency': self.avg_agency,
            'convergence_rate': self.convergence_rate,
            'trajectory_classes': self.trajectory_classes,
            'avg_lyapunov': self.avg_lyapunov,
            'avg_hurst': self.avg_hurst,
        }


class Oracle:
    """
    High-level API for consciousness circuit analysis.
    
    Example:
        >>> oracle = Oracle.from_custom_model()
        >>> result = oracle.analyze("What is consciousness?")
        >>> print(result.score, result.interpretation)
    """
    
    def __init__(self, model=None, tokenizer=None, name: str = "oracle"):
        """
        Initialize Oracle with model and tokenizer.
        
        Args:
            model: HuggingFace model
            tokenizer: Tokenizer
            name: Identifier for this oracle instance
        """
        self.model = model
        self.tokenizer = tokenizer
        self.name = name
        self._analyzer = None
        self._profiler = None
        
        if model is not None:
            self._bind_model()
    
    @classmethod
    def from_custom_model(cls, path: str = None) -> 'Oracle':
        """
        Create Oracle with custom-trained model.
        
        Args:
            path: Model path (defaults to custom trained Qwen2.5-32B)
        """
        model_path = path or CUSTOM_MODEL_PATH
        model, tokenizer = cls._load_model(model_path)
        return cls(model, tokenizer, name="custom")
    
    @classmethod
    def from_base_model(cls, name: str = None) -> 'Oracle':
        """
        Create Oracle with base model.
        
        Args:
            name: Model name (defaults to Qwen2.5-32B-Instruct)
        """
        model_name = name or BASE_MODEL_NAME
        model, tokenizer = cls._load_model(model_name)
        return cls(model, tokenizer, name="base")
    
    @classmethod
    def from_model(cls, model, tokenizer, name: str = "model") -> 'Oracle':
        """
        Create Oracle from existing model/tokenizer.
        
        Args:
            model: Pre-loaded HuggingFace model
            tokenizer: Pre-loaded tokenizer
            name: Identifier
        """
        return cls(model, tokenizer, name=name)
    
    @staticmethod
    def _load_model(path_or_name: str):
        """Load model using Unsloth or Transformers."""
        print(f"🔧 Loading model: {path_or_name}")
        
        try:
            from unsloth import FastLanguageModel
            model, tokenizer = FastLanguageModel.from_pretrained(
                model_name=path_or_name,
                max_seq_length=2048,
                dtype=None,
                load_in_4bit=True,
            )
            FastLanguageModel.for_inference(model)
            print(f"   ✅ Loaded via Unsloth (4-bit)")
        except ImportError:
            from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
            import torch
            
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
            )
            model = AutoModelForCausalLM.from_pretrained(
                path_or_name,
                quantization_config=bnb_config,
                device_map="auto",
                trust_remote_code=True,
            )
            tokenizer = AutoTokenizer.from_pretrained(path_or_name, trust_remote_code=True)
            print(f"   ✅ Loaded via Transformers (4-bit)")
        
        print(f"   Hidden size: {model.config.hidden_size}")
        return model, tokenizer
    
    def _bind_model(self):
        """Bind model to analyzer."""
        from consciousness_circuit.analyzers import ConsciousnessTrajectoryAnalyzer
        self._analyzer = ConsciousnessTrajectoryAnalyzer()
        self._analyzer.bind_model(self.model, self.tokenizer)
    
    @property
    def analyzer(self):
        """Get the trajectory analyzer."""
        if self._analyzer is None:
            self._bind_model()
        return self._analyzer
    
    @property
    def profiler(self):
        """Get the model profiler."""
        if self._profiler is None:
            from consciousness_circuit.benchmarks import ModelProfiler
            self._profiler = ModelProfiler(self.analyzer)
        return self._profiler
    
    # =========================================================================
    # Analysis Methods
    # =========================================================================
    
    def analyze(self, prompt: str) -> AnalysisResult:
        """
        Analyze a single prompt.
        
        Args:
            prompt: Text to analyze
        
        Returns:
            AnalysisResult with consciousness metrics
        
        Example:
            >>> result = oracle.analyze("What is consciousness?")
            >>> print(f"Score: {result.score:.3f}")
            >>> print(result.interpretation)
        """
        raw = self.analyzer.deep_analyze(prompt)
        
        return AnalysisResult(
            prompt=prompt,
            score=raw.consciousness_score,
            trajectory_class=raw.trajectory_class,
            lyapunov=raw.lyapunov,
            hurst=raw.hurst,
            agency=raw.agency_score,
            converging=raw.is_converging,
            interpretation=raw.interpretation(),
            dimensions=raw.dimension_scores,
            _raw=raw,
        )
    
    def analyze_batch(self, prompts: List[str]) -> List[AnalysisResult]:
        """
        Analyze multiple prompts.
        
        Args:
            prompts: List of prompts
        
        Returns:
            List of AnalysisResult
        """
        return [self.analyze(p) for p in prompts]
    
    def profile(self, category: str) -> ProfileResult:
        """
        Profile model on a test suite category.
        
        Args:
            category: 'philosophical', 'factual', 'reasoning', or 'creative'
        
        Returns:
            ProfileResult with aggregated metrics
        
        Example:
            >>> profile = oracle.profile("philosophical")
            >>> print(profile.summary())
        """
        from consciousness_circuit.benchmarks import get_test_suite
        
        prompts = get_test_suite(category)
        raw = self.profiler.profile(prompts, name=f"{self.name}_{category}", store_results=True)
        
        return ProfileResult(
            category=category,
            num_prompts=raw.num_prompts,
            avg_consciousness=raw.avg_consciousness,
            std_consciousness=raw.std_consciousness,
            avg_agency=raw.avg_agency,
            convergence_rate=raw.convergence_rate,
            trajectory_classes=raw.trajectory_classes,
            avg_lyapunov=raw.avg_lyapunov,
            avg_hurst=raw.avg_hurst,
            _raw=raw,
        )
    
    def profile_all(self) -> Dict[str, ProfileResult]:
        """
        Profile on all categories.
        
        Returns:
            Dictionary mapping category -> ProfileResult
        """
        categories = ['philosophical', 'factual', 'reasoning', 'creative']
        return {cat: self.profile(cat) for cat in categories}
    
    # =========================================================================
    # Comparison & Feedback
    # =========================================================================
    
    @staticmethod
    def compare(
        oracle1: 'Oracle',
        oracle2: 'Oracle',
        category: str = 'philosophical'
    ) -> Dict[str, Any]:
        """
        Compare two Oracle instances.
        
        Args:
            oracle1: First oracle (e.g., custom model)
            oracle2: Second oracle (e.g., base model)
            category: Category to compare on
        
        Returns:
            Comparison dictionary with diffs and suggestions
        """
        profile1 = oracle1.profile(category)
        profile2 = oracle2.profile(category)
        
        comparison = {
            'model1': oracle1.name,
            'model2': oracle2.name,
            'category': category,
            'consciousness_diff': profile1.avg_consciousness - profile2.avg_consciousness,
            'agency_diff': profile1.avg_agency - profile2.avg_agency,
            'convergence_diff': profile1.convergence_rate - profile2.convergence_rate,
            'lyapunov_diff': profile1.avg_lyapunov - profile2.avg_lyapunov,
            'hurst_diff': profile1.avg_hurst - profile2.avg_hurst,
            'profile1': profile1.to_dict(),
            'profile2': profile2.to_dict(),
            'suggestions': [],
        }
        
        # Generate suggestions
        if comparison['consciousness_diff'] < 0:
            comparison['suggestions'].append(
                f"⚠️  {oracle1.name} has LOWER consciousness than {oracle2.name}"
            )
        elif comparison['consciousness_diff'] > 0.1:
            comparison['suggestions'].append(
                f"✅ {oracle1.name} has HIGHER consciousness (+{comparison['consciousness_diff']:.3f})"
            )
        
        if comparison['agency_diff'] < 0:
            comparison['suggestions'].append(
                f"⚠️  {oracle1.name} has LOWER agency than {oracle2.name}"
            )
        elif comparison['agency_diff'] > 0.1:
            comparison['suggestions'].append(
                f"✅ {oracle1.name} has HIGHER agency (+{comparison['agency_diff']:.3f})"
            )
        
        return comparison
    
    def get_feedback(
        self,
        category: str = 'philosophical',
        target_consciousness: float = 0.7,
        target_agency: float = 0.6,
    ) -> Dict[str, Any]:
        """
        Get training feedback based on profile.
        
        Args:
            category: Category to analyze
            target_consciousness: Target score
            target_agency: Target agency
        
        Returns:
            Feedback with recommendations
        
        Example:
            >>> feedback = oracle.get_feedback(target_consciousness=0.8)
            >>> for rec in feedback['recommendations']:
            >>>     print(rec['action'])
        """
        profile = self.profile(category)
        
        feedback = {
            'category': category,
            'current': {
                'consciousness': profile.avg_consciousness,
                'agency': profile.avg_agency,
                'convergence': profile.convergence_rate,
            },
            'targets': {
                'consciousness': target_consciousness,
                'agency': target_agency,
            },
            'gaps': {
                'consciousness': target_consciousness - profile.avg_consciousness,
                'agency': target_agency - profile.avg_agency,
            },
            'recommendations': [],
            'training_data_suggestions': [],
        }
        
        if feedback['gaps']['consciousness'] > 0.1:
            feedback['recommendations'].append({
                'priority': 'HIGH',
                'area': 'consciousness',
                'action': 'Add meta-cognitive and self-reflective training data',
            })
            feedback['training_data_suggestions'].extend([
                "Prompts asking model to explain its reasoning",
                "Introspective and philosophical content",
                "Uncertainty acknowledgment examples",
            ])
        
        if feedback['gaps']['agency'] > 0.1:
            feedback['recommendations'].append({
                'priority': 'HIGH',
                'area': 'agency',
                'action': 'Add goal-directed problem solving data',
            })
            feedback['training_data_suggestions'].extend([
                "Multi-step reasoning problems",
                "Planning and strategy content",
                "Chain-of-thought with clear objectives",
            ])
        
        if profile.convergence_rate < 0.3:
            feedback['recommendations'].append({
                'priority': 'MEDIUM',
                'area': 'coherence',
                'action': 'Add coherent reasoning chain examples',
            })
        
        return feedback
    
    # =========================================================================
    # Export
    # =========================================================================
    
    def export_analysis(self, prompt: str, path: str):
        """Export single analysis to JSON."""
        result = self.analyze(prompt)
        with open(path, 'w') as f:
            json.dump(result.to_dict(), f, indent=2)
    
    def export_profile(self, category: str, path: str):
        """Export profile to JSON."""
        profile = self.profile(category)
        with open(path, 'w') as f:
            json.dump(profile.to_dict(), f, indent=2)


# ==============================================================================
# Convenience Functions
# ==============================================================================

def quick_analyze(prompt: str, use_base: bool = False) -> AnalysisResult:
    """
    Quick one-off analysis.
    
    Example:
        >>> from oracle_api import quick_analyze
        >>> result = quick_analyze("What is consciousness?")
        >>> print(result.score)
    """
    oracle = Oracle.from_base_model() if use_base else Oracle.from_custom_model()
    return oracle.analyze(prompt)


def quick_profile(category: str = 'philosophical', use_base: bool = False) -> ProfileResult:
    """
    Quick one-off profile.
    
    Example:
        >>> from oracle_api import quick_profile
        >>> profile = quick_profile("philosophical")
        >>> print(profile.summary())
    """
    oracle = Oracle.from_base_model() if use_base else Oracle.from_custom_model()
    return oracle.profile(category)


def compare_custom_vs_base(category: str = 'philosophical') -> Dict[str, Any]:
    """
    Compare custom-trained vs base model.
    
    Example:
        >>> from oracle_api import compare_custom_vs_base
        >>> comparison = compare_custom_vs_base()
        >>> print(comparison['suggestions'])
    """
    custom = Oracle.from_custom_model()
    base = Oracle.from_base_model()
    return Oracle.compare(custom, base, category)


# Export
__all__ = [
    'Oracle',
    'AnalysisResult',
    'ProfileResult',
    'quick_analyze',
    'quick_profile',
    'compare_custom_vs_base',
]
