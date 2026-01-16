"""
Plugin Base Classes

Standalone abstract interfaces for analysis and intervention plugins.

Usage:
    from consciousness_circuit.plugins.base import AnalysisPlugin, InterventionPlugin
    
    class MyPlugin(AnalysisPlugin):
        def analyze(self, data, **kwargs):
            return {'my_metric': 42}
"""

from abc import ABC, abstractmethod
from typing import Dict, Any, Optional
from dataclasses import dataclass, field


@dataclass
class PluginResult:
    """Standard result format for all plugins."""
    name: str
    success: bool
    data: Dict[str, Any]
    errors: Optional[list] = None
    
    def __str__(self) -> str:
        status = "✓" if self.success else "✗"
        return f"{status} {self.name}: {len(self.data)} results"


class AnalysisPlugin(ABC):
    """
    Base class for passive analysis plugins.
    
    Analysis plugins observe and measure without modifying the system.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name identifier."""
        pass
    
    @abstractmethod
    def analyze(self, data, **kwargs) -> PluginResult:
        """
        Analyze data and return results.
        
        Args:
            data: Input data to analyze
            **kwargs: Additional parameters
            
        Returns:
            PluginResult with analysis results
        """
        pass
    
    def validate_input(self, data) -> bool:
        """
        Validate input data before analysis.
        
        Override this to add custom validation.
        
        Args:
            data: Input data
            
        Returns:
            True if data is valid, False otherwise
        """
        return True


class InterventionPlugin(ABC):
    """
    Base class for active intervention plugins.
    
    Intervention plugins can modify the system state.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name identifier."""
        pass
    
    @abstractmethod
    def should_intervene(self, metrics: Dict[str, Any]) -> bool:
        """
        Decide whether intervention is needed based on metrics.
        
        Args:
            metrics: Current system metrics
            
        Returns:
            True if intervention should occur
        """
        pass
    
    @abstractmethod
    def intervene(self, data, metrics: Dict[str, Any]):
        """
        Perform intervention on data.
        
        Args:
            data: System state to modify
            metrics: Current metrics
            
        Returns:
            Modified data
        """
        pass
    
    def get_intervention_strength(self, metrics: Dict[str, Any]) -> float:
        """
        Compute intervention strength from metrics.
        
        Override to customize intervention intensity.
        
        Args:
            metrics: Current metrics
            
        Returns:
            Strength factor (0-1)
        """
        return 0.5


class TrainingPlugin(ABC):
    """
    Base class for training signal generation plugins.
    
    Training plugins compute reward/loss signals for model training.
    """
    
    @property
    @abstractmethod
    def name(self) -> str:
        """Plugin name identifier."""
        pass
    
    @abstractmethod
    def compute_signal(self, data, **kwargs) -> float:
        """
        Compute training signal (reward/loss) from data.
        
        Args:
            data: Training data
            **kwargs: Additional parameters
            
        Returns:
            Training signal value
        """
        pass
    
    def compute_batch_signals(self, batch_data: list, **kwargs) -> list:
        """
        Compute signals for a batch of data.
        
        Args:
            batch_data: List of data items
            **kwargs: Additional parameters
            
        Returns:
            List of signal values
        """
        return [self.compute_signal(data, **kwargs) for data in batch_data]


class PluginRegistry:
    """
    Registry for managing plugins.
    
    Allows dynamic registration and discovery of plugins.
    """
    
    def __init__(self):
        """Initialize empty registry."""
        self._analysis_plugins: Dict[str, AnalysisPlugin] = {}
        self._intervention_plugins: Dict[str, InterventionPlugin] = {}
        self._training_plugins: Dict[str, TrainingPlugin] = {}
    
    def register_analysis(self, plugin: AnalysisPlugin):
        """Register an analysis plugin."""
        self._analysis_plugins[plugin.name] = plugin
    
    def register_intervention(self, plugin: InterventionPlugin):
        """Register an intervention plugin."""
        self._intervention_plugins[plugin.name] = plugin
    
    def register_training(self, plugin: TrainingPlugin):
        """Register a training plugin."""
        self._training_plugins[plugin.name] = plugin
    
    def get_analysis(self, name: str) -> Optional[AnalysisPlugin]:
        """Get analysis plugin by name."""
        return self._analysis_plugins.get(name)
    
    def get_intervention(self, name: str) -> Optional[InterventionPlugin]:
        """Get intervention plugin by name."""
        return self._intervention_plugins.get(name)
    
    def get_training(self, name: str) -> Optional[TrainingPlugin]:
        """Get training plugin by name."""
        return self._training_plugins.get(name)
    
    def list_analysis_plugins(self) -> list:
        """List all registered analysis plugins."""
        return list(self._analysis_plugins.keys())
    
    def list_intervention_plugins(self) -> list:
        """List all registered intervention plugins."""
        return list(self._intervention_plugins.keys())
    
    def list_training_plugins(self) -> list:
        """List all registered training plugins."""
        return list(self._training_plugins.keys())
