"""
Agency and Goal-Directedness Metrics

Standalone module for measuring purposeful behavior.
Based on TAME (Technological Approach to Mind Everywhere) framework.

Usage:
    from consciousness_circuit.metrics.agency import compute_agency_score, TAMEMetrics
    
    score = compute_agency_score(trajectory, goal_state)
    
    tame = TAMEMetrics()
    result = tame.analyze(trajectory)
"""

import numpy as np
from typing import List, Optional, Union, Tuple
from dataclasses import dataclass


@dataclass
class AgencyResult:
    """Result of agency/goal-directedness analysis."""
    score: float  # 0-1 overall agency score
    goal_directedness: float  # Progress toward goal
    path_efficiency: float  # Direct path vs actual path
    adaptability: float  # Ability to correct course
    persistence: float  # Consistency of direction
    
    @property
    def is_agentic(self) -> bool:
        """Returns True if behavior appears goal-directed (score > 0.6)."""
        return self.score > 0.6
    
    @property
    def interpretation(self) -> str:
        """Human-readable interpretation."""
        if self.score > 0.8:
            return "highly agentic"
        elif self.score > 0.6:
            return "moderately agentic"
        elif self.score > 0.4:
            return "weakly agentic"
        else:
            return "non-agentic/random"


def compute_agency_score(
    trajectory: np.ndarray,
    goal_state: Optional[np.ndarray] = None,
    weights: Optional[dict] = None
) -> float:
    """
    Compute overall agency score from trajectory.
    
    Combines multiple indicators of goal-directed behavior.
    
    Args:
        trajectory: Nx2 or NxD array of positions over time
        goal_state: Optional target state (if None, uses trajectory endpoint)
        weights: Optional custom weights for components
        
    Returns:
        Agency score (0-1)
    """
    trajectory = np.asarray(trajectory)
    
    if trajectory.ndim == 1:
        # 1D trajectory, reshape to 2D
        trajectory = trajectory.reshape(-1, 1)
    
    if len(trajectory) < 3:
        return 0.0
    
    # Default weights
    if weights is None:
        weights = {
            'goal_directedness': 0.35,
            'path_efficiency': 0.25,
            'adaptability': 0.2,
            'persistence': 0.2
        }
    
    # Infer goal if not provided
    if goal_state is None:
        goal_state = trajectory[-1]
    
    start_state = trajectory[0]
    
    # Component metrics
    goal_dir = _compute_goal_directedness(trajectory, start_state, goal_state)
    path_eff = compute_path_efficiency(trajectory, start_state, goal_state)
    adapt = _compute_adaptability(trajectory, goal_state)
    persist = _compute_persistence(trajectory)
    
    # Weighted combination
    score = (
        weights['goal_directedness'] * goal_dir +
        weights['path_efficiency'] * path_eff +
        weights['adaptability'] * adapt +
        weights['persistence'] * persist
    )
    
    return float(np.clip(score, 0.0, 1.0))


def _compute_goal_directedness(
    trajectory: np.ndarray,
    start: np.ndarray,
    goal: np.ndarray
) -> float:
    """
    Measure progress toward goal over time.
    
    Computes fraction of distance to goal that is covered.
    """
    initial_distance = np.linalg.norm(start - goal)
    
    if initial_distance == 0:
        return 1.0
    
    final_distance = np.linalg.norm(trajectory[-1] - goal)
    
    # Progress = distance covered / initial distance
    progress = (initial_distance - final_distance) / initial_distance
    
    return float(np.clip(progress, 0.0, 1.0))


def compute_path_efficiency(
    trajectory: np.ndarray,
    start: np.ndarray,
    end: np.ndarray
) -> float:
    """
    Compute path efficiency: ratio of direct distance to actual path length.
    
    Efficiency = 1 means straight line, < 1 means wandering.
    
    Args:
        trajectory: Nx2 or NxD array
        start: Starting position
        end: Ending position
        
    Returns:
        Path efficiency (0-1)
    """
    # Direct distance
    direct_distance = np.linalg.norm(end - start)
    
    if direct_distance == 0:
        return 1.0
    
    # Actual path length
    displacements = np.diff(trajectory, axis=0)
    path_length = np.sum(np.linalg.norm(displacements, axis=1))
    
    if path_length == 0:
        return 0.0
    
    efficiency = direct_distance / path_length
    
    return float(np.clip(efficiency, 0.0, 1.0))


def _compute_adaptability(
    trajectory: np.ndarray,
    goal: np.ndarray
) -> float:
    """
    Measure ability to correct course toward goal.
    
    Looks at whether direction vectors tend to point toward goal.
    """
    if len(trajectory) < 3:
        return 0.0
    
    # Compute velocity vectors
    velocities = np.diff(trajectory, axis=0)
    
    # For each segment, compute alignment with goal direction
    alignments = []
    
    for i in range(len(velocities)):
        position = trajectory[i]
        velocity = velocities[i]
        
        # Direction to goal
        to_goal = goal - position
        
        # Normalize
        vel_norm = np.linalg.norm(velocity)
        goal_norm = np.linalg.norm(to_goal)
        
        if vel_norm > 0 and goal_norm > 0:
            # Cosine similarity
            alignment = np.dot(velocity, to_goal) / (vel_norm * goal_norm)
            alignments.append(alignment)
    
    if not alignments:
        return 0.0
    
    # Average alignment (mapped to 0-1)
    mean_alignment = np.mean(alignments)
    
    # Map from [-1, 1] to [0, 1]
    adaptability = (mean_alignment + 1.0) / 2.0
    
    return float(adaptability)


def _compute_persistence(trajectory: np.ndarray) -> float:
    """
    Measure consistency of direction over time.
    
    High persistence = maintains direction, low = erratic.
    """
    if len(trajectory) < 3:
        return 0.0
    
    # Compute velocity vectors
    velocities = np.diff(trajectory, axis=0)
    
    # Normalize velocities
    norms = np.linalg.norm(velocities, axis=1, keepdims=True)
    norms[norms == 0] = 1  # Avoid division by zero
    
    normalized_velocities = velocities / norms
    
    # Compute angular changes between consecutive velocities
    angular_consistency = []
    
    for i in range(len(normalized_velocities) - 1):
        dot_product = np.dot(normalized_velocities[i], normalized_velocities[i+1])
        # Clamp to [-1, 1] for numerical stability
        dot_product = np.clip(dot_product, -1.0, 1.0)
        angular_consistency.append(dot_product)
    
    if not angular_consistency:
        return 0.0
    
    # Mean cosine similarity (already in [-1, 1])
    mean_consistency = np.mean(angular_consistency)
    
    # Map to [0, 1]
    persistence = (mean_consistency + 1.0) / 2.0
    
    return float(persistence)


class TAMEMetrics:
    """
    Full TAME framework analysis for agency and goal-directedness.
    
    TAME (Technological Approach to Mind Everywhere) provides a framework
    for measuring goal-directed behavior in various systems.
    """
    
    def __init__(self, weights: Optional[dict] = None):
        """
        Initialize TAME analyzer.
        
        Args:
            weights: Custom weights for agency components
        """
        self.weights = weights or {
            'goal_directedness': 0.35,
            'path_efficiency': 0.25,
            'adaptability': 0.2,
            'persistence': 0.2
        }
    
    def analyze(
        self,
        trajectory: Union[np.ndarray, Tuple[np.ndarray, np.ndarray]],
        goal_state: Optional[np.ndarray] = None
    ) -> AgencyResult:
        """
        Complete TAME analysis of trajectory.
        
        Args:
            trajectory: Either (x, y) tuple or Nx2 array
            goal_state: Optional target state
            
        Returns:
            AgencyResult with all metrics
        """
        # Handle different input formats
        if isinstance(trajectory, tuple):
            x, y = trajectory
            traj_array = np.column_stack([x, y])
        else:
            traj_array = np.asarray(trajectory)
        
        if traj_array.ndim == 1:
            traj_array = traj_array.reshape(-1, 1)
        
        # Infer goal if not provided
        if goal_state is None:
            goal_state = traj_array[-1]
        
        start_state = traj_array[0]
        
        # Compute all components
        goal_dir = _compute_goal_directedness(traj_array, start_state, goal_state)
        path_eff = compute_path_efficiency(traj_array, start_state, goal_state)
        adapt = _compute_adaptability(traj_array, goal_state)
        persist = _compute_persistence(traj_array)
        
        # Overall score
        score = (
            self.weights['goal_directedness'] * goal_dir +
            self.weights['path_efficiency'] * path_eff +
            self.weights['adaptability'] * adapt +
            self.weights['persistence'] * persist
        )
        
        return AgencyResult(
            score=float(np.clip(score, 0.0, 1.0)),
            goal_directedness=goal_dir,
            path_efficiency=path_eff,
            adaptability=adapt,
            persistence=persist
        )
    
    def compare_trajectories(
        self,
        trajectories: List[np.ndarray],
        goal_state: Optional[np.ndarray] = None
    ) -> List[AgencyResult]:
        """
        Analyze and compare multiple trajectories.
        
        Args:
            trajectories: List of trajectory arrays
            goal_state: Optional shared goal state
            
        Returns:
            List of AgencyResult, one per trajectory
        """
        results = []
        
        for traj in trajectories:
            result = self.analyze(traj, goal_state)
            results.append(result)
        
        return results
    
    def set_weights(self, weights: dict):
        """Update component weights."""
        required_keys = {'goal_directedness', 'path_efficiency', 'adaptability', 'persistence'}
        
        if not all(k in weights for k in required_keys):
            raise ValueError(f"Weights must contain keys: {required_keys}")
        
        # Normalize weights to sum to 1
        total = sum(weights.values())
        
        if total > 0:
            self.weights = {k: v/total for k, v in weights.items()}
        else:
            raise ValueError("Weight values must be positive")
