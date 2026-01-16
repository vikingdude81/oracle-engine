"""
TAME Metrics - Trajectory Analysis for Meta-Emergence
======================================================

Metrics for goal-directedness, agency, and attractor dynamics.
Ported from cell-research-emergence project.

TAME = Trajectory Analysis for Meta-Emergence
Measures how hidden states exhibit goal-directed, agentic behavior.
"""

import numpy as np
from typing import Tuple, Optional
from scipy.spatial.distance import cdist


def compute_agency_score(trajectory: np.ndarray, 
                        window_size: int = 5) -> float:
    """
    Compute agency score - measure of goal-directed behavior.
    
    Agency is characterized by:
    1. Consistent direction of movement
    2. Acceleration toward specific regions
    3. Non-random trajectory structure
    
    Args:
        trajectory: Hidden state trajectory [time_steps, features]
        window_size: Window for computing local direction
        
    Returns:
        Agency score in [0, 1], where 1 is highly agentic
    """
    if len(trajectory) < window_size + 1:
        return 0.0
    
    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)
    
    n_steps, n_features = trajectory.shape
    
    # Compute velocity (direction of movement)
    velocities = np.diff(trajectory, axis=0)
    
    if len(velocities) < window_size:
        return 0.0
    
    # Compute directional consistency
    directions = []
    for i in range(len(velocities) - window_size + 1):
        window_vel = velocities[i:i + window_size]
        # Normalize velocities
        norms = np.linalg.norm(window_vel, axis=1, keepdims=True)
        norms[norms == 0] = 1  # Avoid division by zero
        normalized_vel = window_vel / norms
        
        # Compute mean direction
        mean_direction = np.mean(normalized_vel, axis=0)
        direction_strength = np.linalg.norm(mean_direction)
        directions.append(direction_strength)
    
    if not directions:
        return 0.0
    
    # Agency is high when direction is consistent
    directional_consistency = np.mean(directions)
    
    # Compute acceleration (changes in velocity)
    accelerations = np.diff(velocities, axis=0)
    acceleration_norm = np.linalg.norm(accelerations, axis=1)
    
    # Low variance in acceleration suggests controlled movement
    if len(acceleration_norm) > 0 and np.std(acceleration_norm) > 0:
        acceleration_control = 1.0 / (1.0 + np.std(acceleration_norm))
    else:
        acceleration_control = 0.5
    
    # Combine metrics
    agency = 0.7 * directional_consistency + 0.3 * acceleration_control
    
    return float(np.clip(agency, 0.0, 1.0))


def detect_attractor_convergence(trajectory: np.ndarray,
                                 convergence_threshold: float = 0.1) -> Tuple[float, bool]:
    """
    Detect if trajectory is converging to an attractor.
    
    An attractor is a region in state space that trajectory tends toward.
    Convergence indicates the system is settling into a stable pattern.
    
    Args:
        trajectory: Hidden state trajectory [time_steps, features]
        convergence_threshold: Threshold for determining convergence
        
    Returns:
        (attractor_strength, is_converging) tuple
        - attractor_strength: How strongly converging [0, 1]
        - is_converging: Boolean flag for convergence
    """
    if len(trajectory) < 10:
        return 0.0, False
    
    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)
    
    n_steps, n_features = trajectory.shape
    
    # Compute centroid of late trajectory (potential attractor)
    split_point = max(n_steps // 2, n_steps - 10)
    late_trajectory = trajectory[split_point:]
    attractor_candidate = np.mean(late_trajectory, axis=0)
    
    # Compute distances to attractor over time
    distances = np.linalg.norm(trajectory - attractor_candidate, axis=1)
    
    # Check if distances are decreasing
    if len(distances) > 2:
        # Fit linear trend to distances
        time_points = np.arange(len(distances))
        trend_coef = np.polyfit(time_points, distances, 1)[0]
        
        # Negative slope = converging
        is_converging = trend_coef < -convergence_threshold / n_steps
        
        # Strength based on rate of convergence and final variance
        if is_converging:
            convergence_rate = abs(trend_coef)
            final_variance = np.var(late_trajectory, axis=0).mean()
            
            # High strength if converging quickly and final variance is low
            strength = np.clip(convergence_rate * 10, 0, 0.8)
            if final_variance < 0.1:
                strength += 0.2
            
            return float(np.clip(strength, 0.0, 1.0)), True
        else:
            return 0.0, False
    
    return 0.0, False


def compute_goal_directedness(trajectory: np.ndarray,
                              goal_estimate: Optional[np.ndarray] = None) -> float:
    """
    Compute goal-directedness metric.
    
    Measures how consistently trajectory moves toward a goal state.
    If no goal is provided, assumes goal is the final state.
    
    Args:
        trajectory: Hidden state trajectory [time_steps, features]
        goal_estimate: Optional goal state to measure toward
        
    Returns:
        Goal-directedness score in [0, 1]
    """
    if len(trajectory) < 3:
        return 0.0
    
    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)
    
    # Use final state as goal if not provided
    if goal_estimate is None:
        goal_estimate = trajectory[-1]
    
    # Compute distances to goal over time
    distances = np.linalg.norm(trajectory - goal_estimate, axis=1)
    
    # Measure monotonic decrease toward goal
    decreases = 0
    total_changes = 0
    
    for i in range(len(distances) - 1):
        if distances[i] > distances[i + 1]:
            decreases += 1
        total_changes += 1
    
    if total_changes == 0:
        return 0.0
    
    # Goal-directedness is proportion of steps moving toward goal
    goal_directedness = decreases / total_changes
    
    # Bonus for smooth, continuous approach
    if len(distances) > 2:
        smoothness = 1.0 / (1.0 + np.std(np.diff(distances)))
        goal_directedness = 0.8 * goal_directedness + 0.2 * smoothness
    
    return float(np.clip(goal_directedness, 0.0, 1.0))


def compute_trajectory_coherence(trajectory: np.ndarray,
                                window_size: int = 5) -> float:
    """
    Compute trajectory coherence - how structured and non-random.
    
    High coherence indicates organized, purposeful movement.
    Low coherence suggests random drift or noise.
    
    Args:
        trajectory: Hidden state trajectory [time_steps, features]
        window_size: Window for local analysis
        
    Returns:
        Coherence score in [0, 1]
    """
    if len(trajectory) < window_size:
        return 0.0
    
    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)
    
    # Compute local principal directions
    coherence_scores = []
    
    for i in range(len(trajectory) - window_size + 1):
        window = trajectory[i:i + window_size]
        
        # Center the window
        centered = window - np.mean(window, axis=0)
        
        # Compute covariance and eigenvalues
        if window.shape[1] > 1:
            cov = np.cov(centered.T)
            eigenvalues = np.linalg.eigvalsh(cov)
            eigenvalues = np.sort(eigenvalues)[::-1]
            
            # Coherence = ratio of largest to sum (like explained variance)
            if np.sum(eigenvalues) > 0:
                local_coherence = eigenvalues[0] / np.sum(eigenvalues)
            else:
                local_coherence = 0.0
        else:
            # For 1D, coherence is based on variance
            var = np.var(centered)
            local_coherence = min(1.0, var / (var + 0.1))
        
        coherence_scores.append(local_coherence)
    
    if not coherence_scores:
        return 0.0
    
    # Overall coherence is mean of local coherences
    return float(np.clip(np.mean(coherence_scores), 0.0, 1.0))


class TAMEMetrics:
    """
    Container for TAME (Trajectory Analysis for Meta-Emergence) metrics.
    
    Provides a unified interface for computing goal-directedness, agency,
    and attractor dynamics in hidden state trajectories.
    """
    
    def __init__(self):
        self.agency_score = 0.0
        self.attractor_strength = 0.0
        self.is_converging = False
        self.goal_directedness = 0.0
        self.trajectory_coherence = 0.0
    
    def compute_all(self, trajectory: np.ndarray) -> dict:
        """
        Compute all TAME metrics for a trajectory.
        
        Args:
            trajectory: Hidden state trajectory [time_steps, features]
            
        Returns:
            Dictionary of all metrics
        """
        self.agency_score = compute_agency_score(trajectory)
        self.attractor_strength, self.is_converging = detect_attractor_convergence(trajectory)
        self.goal_directedness = compute_goal_directedness(trajectory)
        self.trajectory_coherence = compute_trajectory_coherence(trajectory)
        
        return {
            "agency_score": self.agency_score,
            "attractor_strength": self.attractor_strength,
            "is_converging": self.is_converging,
            "goal_directedness": self.goal_directedness,
            "trajectory_coherence": self.trajectory_coherence,
            "overall_tame_score": self._compute_overall_score(),
        }
    
    def _compute_overall_score(self) -> float:
        """Compute weighted overall TAME score."""
        weights = {
            "agency": 0.3,
            "attractor": 0.25,
            "goal": 0.25,
            "coherence": 0.2,
        }
        
        score = (
            weights["agency"] * self.agency_score +
            weights["attractor"] * self.attractor_strength +
            weights["goal"] * self.goal_directedness +
            weights["coherence"] * self.trajectory_coherence
        )
        
        return float(np.clip(score, 0.0, 1.0))
    
    def __repr__(self):
        return (
            f"TAMEMetrics(agency={self.agency_score:.3f}, "
            f"attractor={self.attractor_strength:.3f}, "
            f"goal={self.goal_directedness:.3f}, "
            f"coherence={self.trajectory_coherence:.3f})"
        )


__all__ = [
    "compute_agency_score",
    "detect_attractor_convergence",
    "compute_goal_directedness",
    "compute_trajectory_coherence",
    "TAMEMetrics",
]
