"""
Plugin Base Classes - FULLY STANDALONE
======================================

Abstract base classes for analysis and intervention plugins.
Can be copied to any project - only requires numpy.

Usage:
    from base import AnalysisPlugin, InterventionPlugin
    
    class MyPlugin(InterventionPlugin):
        def should_intervene(self, metrics):
            return metrics.get('chaos', 0) > 0.5
        
        def intervene(self, hidden_states, metrics):
            # Apply intervention
            return modified_states

Dependencies: numpy only
"""

import numpy as np
from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class PluginResult:
    """Result from plugin execution."""
    
    success: bool
    """Whether the plugin executed successfully."""
    
    modified: bool
    """Whether the plugin modified the data."""
    
    metrics: Dict[str, Any]
    """Metrics or information from the plugin."""
    
    message: Optional[str] = None
    """Optional message describing what happened."""
    
    def __repr__(self):
        return f"PluginResult(success={self.success}, modified={self.modified})"


class AnalysisPlugin(ABC):
    """
    Base class for analysis plugins.
    
    Analysis plugins compute metrics from data without modifying it.
    """
    
    def __init__(self, name: str):
        """
        Initialize plugin.
        
        Args:
            name: Unique name for this plugin
        """
        self.name = name
        self.enabled = True
    
    @abstractmethod
    def analyze(self, hidden_states: np.ndarray, **kwargs) -> Dict[str, Any]:
        """
        Analyze hidden states and return metrics.
        
        Args:
            hidden_states: Hidden state trajectories [sequence_length, hidden_dim]
            **kwargs: Additional parameters specific to the plugin
        
        Returns:
            Dictionary of analysis results
        """
        raise NotImplementedError
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', enabled={self.enabled})"


class InterventionPlugin(ABC):
    """
    Base class for intervention plugins.
    
    Intervention plugins can modify hidden states based on metrics.
    Useful for steering model behavior during generation.
    """
    
    def __init__(self, name: str):
        """
        Initialize plugin.
        
        Args:
            name: Unique name for this plugin
        """
        self.name = name
        self.enabled = True
        self.intervention_count = 0
    
    @abstractmethod
    def should_intervene(self, metrics: Dict[str, Any]) -> bool:
        """
        Determine if intervention is needed based on metrics.
        
        Args:
            metrics: Dictionary of computed metrics
        
        Returns:
            True if intervention should be applied
        """
        raise NotImplementedError
    
    @abstractmethod
    def intervene(self, 
                 hidden_states: np.ndarray, 
                 metrics: Dict[str, Any]) -> np.ndarray:
        """
        Apply intervention to hidden states.
        
        Args:
            hidden_states: Current hidden states [sequence_length, hidden_dim]
            metrics: Dictionary of computed metrics
        
        Returns:
            Modified hidden states with same shape
        """
        raise NotImplementedError
    
    def apply(self, 
             hidden_states: np.ndarray, 
             metrics: Dict[str, Any],
             force: bool = False) -> PluginResult:
        """
        Apply intervention if conditions are met.
        
        Args:
            hidden_states: Current hidden states
            metrics: Computed metrics
            force: If True, bypasses should_intervene check
        
        Returns:
            PluginResult with intervention outcome
        """
        if not self.enabled and not force:
            return PluginResult(
                success=True,
                modified=False,
                metrics={},
                message="Plugin is disabled"
            )
        
        if not force and not self.should_intervene(metrics):
            return PluginResult(
                success=True,
                modified=False,
                metrics=metrics,
                message="Intervention not needed"
            )
        
        try:
            modified_states = self.intervene(hidden_states, metrics)
            self.intervention_count += 1
            
            return PluginResult(
                success=True,
                modified=True,
                metrics=metrics,
                message=f"Intervention applied (count: {self.intervention_count})"
            )
        except Exception as e:
            return PluginResult(
                success=False,
                modified=False,
                metrics=metrics,
                message=f"Intervention failed: {str(e)}"
            )
    
    def reset(self):
        """Reset plugin state (e.g., intervention count)."""
        self.intervention_count = 0
    
    def __repr__(self):
        return f"{self.__class__.__name__}(name='{self.name}', interventions={self.intervention_count})"


class PluginRegistry:
    """
    Registry for managing multiple plugins.
    
    Example:
        >>> registry = PluginRegistry()
        >>> registry.register(my_plugin)
        >>> results = registry.run_all(hidden_states, metrics)
    """
    
    def __init__(self):
        self.plugins: Dict[str, AnalysisPlugin | InterventionPlugin] = {}
    
    def register(self, plugin: AnalysisPlugin | InterventionPlugin):
        """
        Register a plugin.
        
        Args:
            plugin: Plugin instance to register
        """
        self.plugins[plugin.name] = plugin
    
    def unregister(self, name: str):
        """
        Unregister a plugin by name.
        
        Args:
            name: Name of plugin to remove
        """
        if name in self.plugins:
            del self.plugins[name]
    
    def get(self, name: str) -> Optional[AnalysisPlugin | InterventionPlugin]:
        """
        Get a plugin by name.
        
        Args:
            name: Plugin name
        
        Returns:
            Plugin instance or None if not found
        """
        return self.plugins.get(name)
    
    def list_plugins(self) -> Dict[str, bool]:
        """
        List all registered plugins and their enabled status.
        
        Returns:
            Dictionary mapping plugin names to enabled status
        """
        return {name: plugin.enabled for name, plugin in self.plugins.items()}
    
    def run_analysis_plugins(self, 
                            hidden_states: np.ndarray,
                            **kwargs) -> Dict[str, Any]:
        """
        Run all analysis plugins.
        
        Args:
            hidden_states: Hidden states to analyze
            **kwargs: Additional arguments for plugins
        
        Returns:
            Combined results from all analysis plugins
        """
        results = {}
        
        for name, plugin in self.plugins.items():
            if isinstance(plugin, AnalysisPlugin) and plugin.enabled:
                try:
                    plugin_results = plugin.analyze(hidden_states, **kwargs)
                    results[name] = plugin_results
                except Exception as e:
                    results[name] = {'error': str(e)}
        
        return results
    
    def apply_interventions(self,
                          hidden_states: np.ndarray,
                          metrics: Dict[str, Any]) -> tuple[np.ndarray, list[PluginResult]]:
        """
        Apply all intervention plugins sequentially.
        
        Args:
            hidden_states: Current hidden states
            metrics: Computed metrics
        
        Returns:
            (modified_states, results) tuple
        """
        current_states = hidden_states.copy()
        results = []
        
        for name, plugin in self.plugins.items():
            if isinstance(plugin, InterventionPlugin) and plugin.enabled:
                result = plugin.apply(current_states, metrics)
                results.append(result)
                
                if result.modified:
                    # Update states for next plugin
                    current_states = plugin.intervene(current_states, metrics)
        
        return current_states, results
    
    def __repr__(self):
        return f"PluginRegistry(plugins={len(self.plugins)})"


__all__ = [
    "PluginResult",
    "AnalysisPlugin",
    "InterventionPlugin",
    "PluginRegistry",
]


if __name__ == "__main__":
    # Self-test examples
    print("Plugin Base Classes - Standalone Tests")
    print("=" * 50)
    
    # Test 1: Simple analysis plugin
    print("\n1. Analysis Plugin:")
    
    class MeanPlugin(AnalysisPlugin):
        def __init__(self):
            super().__init__("mean")
        
        def analyze(self, hidden_states, **kwargs):
            return {
                'mean': float(np.mean(hidden_states)),
                'std': float(np.std(hidden_states))
            }
    
    plugin = MeanPlugin()
    data = np.random.randn(100, 64)
    result = plugin.analyze(data)
    print(f"   {plugin}")
    print(f"   Mean: {result['mean']:.3f}, Std: {result['std']:.3f}")
    
    # Test 2: Simple intervention plugin
    print("\n2. Intervention Plugin:")
    
    class NoiseReductionPlugin(InterventionPlugin):
        def __init__(self, threshold=1.0):
            super().__init__("noise_reduction")
            self.threshold = threshold
        
        def should_intervene(self, metrics):
            return metrics.get('noise_level', 0) > self.threshold
        
        def intervene(self, hidden_states, metrics):
            # Smooth by averaging with neighbors
            smoothed = hidden_states.copy()
            for i in range(1, len(smoothed) - 1):
                smoothed[i] = (hidden_states[i-1] + hidden_states[i] + hidden_states[i+1]) / 3
            return smoothed
    
    plugin = NoiseReductionPlugin(threshold=0.5)
    
    # Test without intervention
    metrics_low = {'noise_level': 0.3}
    result = plugin.apply(data, metrics_low)
    print(f"   Low noise: {result}")
    
    # Test with intervention
    metrics_high = {'noise_level': 0.8}
    result = plugin.apply(data, metrics_high)
    print(f"   High noise: {result}")
    
    # Test 3: Plugin registry
    print("\n3. Plugin Registry:")
    
    registry = PluginRegistry()
    registry.register(MeanPlugin())
    registry.register(NoiseReductionPlugin(threshold=0.5))
    
    print(f"   {registry}")
    print(f"   Plugins: {list(registry.plugins.keys())}")
    
    # Run analysis plugins
    analysis_results = registry.run_analysis_plugins(data)
    print(f"   Analysis results: {analysis_results}")
    
    # Apply interventions
    metrics = {'noise_level': 0.8}
    modified, intervention_results = registry.apply_interventions(data, metrics)
    print(f"   Interventions applied: {len([r for r in intervention_results if r.modified])}")
    print(f"   Data modified: {not np.array_equal(data, modified)}")
    
    print("\n✓ All tests completed successfully!")
