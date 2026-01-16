"""
Agency and Goal-Directedness Metrics - FULLY STANDALONE
=======================================================

Measures goal-directed behavior and agentic patterns in trajectories.
Can be copied to any project - only requires numpy.

Usage:
    from agency import compute_agency_score, compute_path_efficiency, AgencyResult
    
    result = compute_agency_score(trajectory)
    print(f"Agency score: {result.agency:.3f}")
    print(f"Is agentic: {result.is_agentic}")
    print(result.interpretation)

Dependencies: numpy only
"""

import numpy as np
from dataclasses import dataclass
from typing import Optional, Tuple


@dataclass
class AgencyResult:
    """Result from agency computation."""
    
    agency: float
    """Overall agency score (0 to 1)."""
    
    directional_consistency: float
    """Consistency of movement direction (0 to 1)."""
    
    acceleration_control: float
    """How controlled the acceleration is (0 to 1)."""
    
    path_efficiency: float
    """Efficiency of path from start to end (0 to 1)."""
    
    goal_directedness: float
    """How well trajectory moves toward goal (0 to 1)."""
    
    @property
    def is_agentic(self) -> bool:
        """True if agency score > 0.6 (strong goal-directed behavior)."""
        return self.agency > 0.6
    
    @property
    def is_reactive(self) -> bool:
        """True if agency score < 0.3 (reactive, not goal-directed)."""
        return self.agency < 0.3
    
    @property
    def interpretation(self) -> str:
        """Human-readable interpretation."""
        if self.is_agentic:
            return f"AGENTIC (agency={self.agency:.3f}) - Strong goal-directed behavior, purposeful movement"
        elif self.is_reactive:
            return f"REACTIVE (agency={self.agency:.3f}) - Reactive behavior, no clear goals"
        else:
            return f"MODERATE AGENCY (agency={self.agency:.3f}) - Some goal-directed behavior"
    
    def __repr__(self):
        return f"AgencyResult(agency={self.agency:.3f}, agentic={self.is_agentic})"


@dataclass
class TAMEMetrics:
    """
    TAME = Trajectory Analysis for Meta-Emergence
    
    Container for all agency-related metrics.
    """
    
    agency_score: float
    """Overall agency score (0 to 1)."""
    
    attractor_strength: float
    """How strongly converging to attractor (0 to 1)."""
    
    is_converging: bool
    """True if trajectory is converging to attractor."""
    
    goal_directedness: float
    """Goal-directedness score (0 to 1)."""
    
    trajectory_coherence: float
    """How structured and non-random (0 to 1)."""
    
    overall_tame_score: float
    """Weighted combination of all metrics (0 to 1)."""
    
    def __repr__(self):
        return (
            f"TAMEMetrics(agency={self.agency_score:.3f}, "
            f"attractor={self.attractor_strength:.3f}, "
            f"goal={self.goal_directedness:.3f}, "
            f"coherence={self.trajectory_coherence:.3f})"
        )


def compute_agency_score(trajectory: np.ndarray,
                        window_size: int = 5,
                        include_details: bool = False) -> AgencyResult:
    """
    Compute agency score - measure of goal-directed behavior.
    
    Agency is characterized by:
    1. Consistent direction of movement
    2. Controlled acceleration
    3. Efficient path toward a goal
    
    This is a standalone implementation requiring only numpy.
    
    Args:
        trajectory: Hidden state trajectory [time_steps] or [time_steps, features]
        window_size: Window for computing local direction consistency
        include_details: If True, computes additional detailed metrics
    
    Returns:
        AgencyResult with agency score and component metrics
    
    Examples:
        >>> # Goal-directed motion
        >>> t = np.arange(100)
        >>> directed = np.stack([t, t * 0.5], axis=1)
        >>> result = compute_agency_score(directed)
        >>> result.is_agentic
        True
        
        >>> # Random walk (not agentic)
        >>> np.random.seed(42)
        >>> random = np.cumsum(np.random.randn(100, 2), axis=0)
        >>> result = compute_agency_score(random)
        >>> result.is_reactive
        True
    """
    if len(trajectory) < window_size + 1:
        return AgencyResult(
            agency=0.0,
            directional_consistency=0.0,
            acceleration_control=0.0,
            path_efficiency=0.0,
            goal_directedness=0.0
        )
    
    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)
    
    n_steps, n_features = trajectory.shape
    
    # Compute velocity (direction of movement)
    velocities = np.diff(trajectory, axis=0)
    
    if len(velocities) < window_size:
        return AgencyResult(
            agency=0.0,
            directional_consistency=0.0,
            acceleration_control=0.0,
            path_efficiency=0.0,
            goal_directedness=0.0
        )
    
    # 1. Directional consistency
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
    
    directional_consistency = np.mean(directions) if directions else 0.0
    
    # 2. Acceleration control (low variance = controlled)
    accelerations = np.diff(velocities, axis=0)
    acceleration_norm = np.linalg.norm(accelerations, axis=1)
    
    if len(acceleration_norm) > 0 and np.std(acceleration_norm) > 0:
        # Normalize by mean to get coefficient of variation
        cv = np.std(acceleration_norm) / (np.mean(acceleration_norm) + 1e-10)
        acceleration_control = 1.0 / (1.0 + cv)
    else:
        acceleration_control = 0.5
    
    # 3. Path efficiency
    path_efficiency = compute_path_efficiency(trajectory)
    
    # 4. Goal-directedness (toward final point)
    goal_directedness = compute_goal_directedness_internal(trajectory)
    
    # Combine metrics
    # Weight: directional consistency is most important
    agency = (
        0.4 * directional_consistency +
        0.2 * acceleration_control +
        0.2 * path_efficiency +
        0.2 * goal_directedness
    )
    
    return AgencyResult(
        agency=float(np.clip(agency, 0.0, 1.0)),
        directional_consistency=float(directional_consistency),
        acceleration_control=float(acceleration_control),
        path_efficiency=float(path_efficiency),
        goal_directedness=float(goal_directedness)
    )


def compute_path_efficiency(trajectory: np.ndarray) -> float:
    """
    Compute path efficiency = straight-line distance / path length.
    
    Measures how direct the path is from start to end:
    - 1.0: Perfectly straight line
    - < 1.0: Meandering path
    - → 0: Very inefficient path
    
    Args:
        trajectory: Position data [time_steps] or [time_steps, features]
    
    Returns:
        Path efficiency (0 to 1)
    
    Examples:
        >>> # Straight line (efficient)
        >>> t = np.arange(100)
        >>> straight = np.stack([t, t], axis=1).astype(float)
        >>> efficiency = compute_path_efficiency(straight)
        >>> efficiency > 0.99
        True
        
        >>> # Meandering path (inefficient)
        >>> t = np.linspace(0, 10, 100)
        >>> meander = np.stack([t, np.sin(t * 5)], axis=1)
        >>> efficiency = compute_path_efficiency(meander)
        >>> efficiency < 0.5
        True
    """
    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)
    
    if len(trajectory) < 2:
        return 0.0
    
    # Straight-line distance from start to end
    displacement = np.linalg.norm(trajectory[-1] - trajectory[0])
    
    # Total path length
    displacements = np.diff(trajectory, axis=0)
    path_length = np.sum(np.linalg.norm(displacements, axis=1))
    
    if path_length == 0:
        return 0.0
    
    efficiency = displacement / path_length
    
    return float(np.clip(efficiency, 0.0, 1.0))


def compute_goal_directedness_internal(trajectory: np.ndarray,
                                      goal: Optional[np.ndarray] = None) -> float:
    """
    Compute goal-directedness metric.
    
    Measures how consistently trajectory moves toward a goal state.
    If no goal provided, uses final state as goal.
    
    Args:
        trajectory: Position data [time_steps] or [time_steps, features]
        goal: Optional goal state (default: final position)
    
    Returns:
        Goal-directedness score (0 to 1)
    """
    if len(trajectory) < 3:
        return 0.0
    
    if trajectory.ndim == 1:
        trajectory = trajectory.reshape(-1, 1)
    
    # Use final state as goal if not provided
    if goal is None:
        goal = trajectory[-1]
    
    # Compute distances to goal over time
    distances = np.linalg.norm(trajectory - goal, axis=1)
    
    # Count steps that move toward goal
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
        # Low standard deviation in distance changes = smooth
        distance_changes = np.abs(np.diff(distances))
        if np.std(distance_changes) > 0:
            smoothness = 1.0 / (1.0 + np.std(distance_changes))
        else:
            smoothness = 1.0
        goal_directedness = 0.8 * goal_directedness + 0.2 * smoothness
    
    return float(np.clip(goal_directedness, 0.0, 1.0))


def detect_attractor_convergence(trajectory: np.ndarray,
                                 convergence_threshold: float = 0.1) -> Tuple[float, bool]:
    """
    Detect if trajectory is converging to an attractor.
    
    An attractor is a region in state space that trajectory tends toward.
    Convergence indicates the system is settling into a stable pattern.
    
    Args:
        trajectory: Position data [time_steps] or [time_steps, features]
        convergence_threshold: Threshold for determining convergence
    
    Returns:
        (attractor_strength, is_converging) tuple
        - attractor_strength: How strongly converging [0, 1]
        - is_converging: Boolean flag for convergence
    
    Examples:
        >>> # Converging trajectory
        >>> t = np.linspace(0, 10, 100)
        >>> converging = np.exp(-t).reshape(-1, 1)
        >>> strength, is_conv = detect_attractor_convergence(converging)
        >>> is_conv
        True
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


def compute_trajectory_coherence(trajectory: np.ndarray,
                                window_size: int = 5) -> float:
    """
    Compute trajectory coherence - how structured and non-random.
    
    High coherence indicates organized, purposeful movement.
    Low coherence suggests random drift or noise.
    
    Args:
        trajectory: Position data [time_steps] or [time_steps, features]
        window_size: Window for local analysis
    
    Returns:
        Coherence score (0 to 1)
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
            
            # Coherence = ratio of largest to sum (explained variance)
            if np.sum(eigenvalues) > 0:
                local_coherence = eigenvalues[0] / np.sum(eigenvalues)
            else:
                local_coherence = 0.0
        else:
            # For 1D, coherence based on variance
            var = np.var(centered)
            local_coherence = min(1.0, var / (var + 0.1))
        
        coherence_scores.append(local_coherence)
    
    if not coherence_scores:
        return 0.0
    
    return float(np.clip(np.mean(coherence_scores), 0.0, 1.0))


def compute_tame_metrics(trajectory: np.ndarray) -> TAMEMetrics:
    """
    Compute all TAME (Trajectory Analysis for Meta-Emergence) metrics.
    
    Args:
        trajectory: Position data [time_steps] or [time_steps, features]
    
    Returns:
        TAMEMetrics with all computed metrics
    """
    agency_result = compute_agency_score(trajectory)
    attractor_strength, is_converging = detect_attractor_convergence(trajectory)
    coherence = compute_trajectory_coherence(trajectory)
    
    # Overall TAME score (weighted combination)
    overall = (
        0.3 * agency_result.agency +
        0.25 * attractor_strength +
        0.25 * agency_result.goal_directedness +
        0.2 * coherence
    )
    
    return TAMEMetrics(
        agency_score=agency_result.agency,
        attractor_strength=attractor_strength,
        is_converging=is_converging,
        goal_directedness=agency_result.goal_directedness,
        trajectory_coherence=coherence,
        overall_tame_score=float(np.clip(overall, 0.0, 1.0))
    )


__all__ = [
    "AgencyResult",
    "TAMEMetrics",
    "compute_agency_score",
    "compute_path_efficiency",
    "detect_attractor_convergence",
    "compute_trajectory_coherence",
    "compute_tame_metrics",
]


if __name__ == "__main__":
    # Self-test examples
    print("Agency and Goal-Directedness Metrics - Standalone Tests")
    print("=" * 50)
    
    np.random.seed(42)
    
    # Test 1: Goal-directed motion
    print("\n1. Goal-Directed Motion (straight line):")
    t = np.arange(100)
    directed = np.stack([t, t * 0.5], axis=1).astype(float)
    result = compute_agency_score(directed)
    print(f"   {result.interpretation}")
    print(f"   Path efficiency: {result.path_efficiency:.3f}")
    
    # Test 2: Random walk
    print("\n2. Random Walk (no goal):")
    random = np.cumsum(np.random.randn(100, 2), axis=0)
    result = compute_agency_score(random)
    print(f"   {result.interpretation}")
    print(f"   Path efficiency: {result.path_efficiency:.3f}")
    
    # Test 3: Converging trajectory
    print("\n3. Converging Trajectory (attractor):")
    t = np.linspace(0, 10, 100)
    converging = np.exp(-t).reshape(-1, 1) * np.random.randn(100, 3)
    strength, is_conv = detect_attractor_convergence(converging)
    print(f"   Converging: {is_conv}")
    print(f"   Attractor strength: {strength:.3f}")
    
    # Test 4: Path efficiency
    print("\n4. Path Efficiency:")
    straight = np.stack([t, t], axis=1).astype(float)
    efficiency = compute_path_efficiency(straight)
    print(f"   Straight line efficiency: {efficiency:.3f}")
    
    meander = np.stack([t, np.sin(t * 0.5) * 5], axis=1)
    efficiency = compute_path_efficiency(meander)
    print(f"   Meandering path efficiency: {efficiency:.3f}")
    
    # Test 5: Trajectory coherence
    print("\n5. Trajectory Coherence:")
    coherent = directed  # Straight line
    coherence = compute_trajectory_coherence(coherent)
    print(f"   Coherent trajectory: {coherence:.3f}")
    
    noisy = random  # Random walk
    coherence = compute_trajectory_coherence(noisy)
    print(f"   Noisy trajectory: {coherence:.3f}")
    
    # Test 6: Full TAME metrics
    print("\n6. Full TAME Metrics:")
    tame = compute_tame_metrics(directed)
    print(f"   {tame}")
    print(f"   Overall TAME score: {tame.overall_tame_score:.3f}")
    
    print("\n✓ All tests completed successfully!")
