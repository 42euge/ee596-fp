"""
Reward Metrics Tracker

Tracks and aggregates reward signals from multiple reward functions during training.
Provides detailed breakdowns and quality metrics for each reward component.
"""

import jax.numpy as jnp
import numpy as np
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass, field
from collections import defaultdict
import time


@dataclass
class RewardMetrics:
    """Container for reward metrics at a given step."""
    step: int
    timestamp: float

    # Individual reward function scores
    reward_scores: Dict[str, List[float]] = field(default_factory=dict)

    # Aggregated statistics
    total_reward: float = 0.0
    mean_reward: float = 0.0
    std_reward: float = 0.0
    min_reward: float = 0.0
    max_reward: float = 0.0

    # Per-function statistics
    function_means: Dict[str, float] = field(default_factory=dict)
    function_stds: Dict[str, float] = field(default_factory=dict)
    function_contributions: Dict[str, float] = field(default_factory=dict)

    # Quality indicators
    reward_variance: float = 0.0
    reward_entropy: float = 0.0
    signal_to_noise: float = 0.0

    # Training dynamics
    kl_divergence: Optional[float] = None
    policy_entropy: Optional[float] = None
    value_estimate: Optional[float] = None
    advantage_mean: Optional[float] = None
    advantage_std: Optional[float] = None

    # Batch information
    batch_size: int = 0
    num_generations: int = 0


class RewardMetricsTracker:
    """
    Tracks reward signals across training steps and provides analytics.

    Features:
    - Per-reward-function tracking
    - Statistical aggregation (mean, std, percentiles)
    - Quality metrics (SNR, entropy, variance)
    - Historical trend analysis
    - Anomaly detection
    """

    def __init__(
        self,
        reward_function_names: List[str],
        window_size: int = 100,
        track_distributions: bool = True,
    ):
        """
        Initialize the metrics tracker.

        Args:
            reward_function_names: Names of reward functions to track
            window_size: Number of recent steps to keep for moving averages
            track_distributions: Whether to track full distributions (memory intensive)
        """
        self.reward_function_names = reward_function_names
        self.window_size = window_size
        self.track_distributions = track_distributions

        # Historical metrics storage
        self.metrics_history: List[RewardMetrics] = []
        self.step_to_metrics: Dict[int, RewardMetrics] = {}

        # Running statistics for efficiency
        self.running_totals = defaultdict(float)
        self.running_counts = defaultdict(int)
        self.running_squared = defaultdict(float)

        # Anomaly tracking
        self.anomaly_threshold = 3.0  # Standard deviations
        self.detected_anomalies: List[Dict[str, Any]] = []

        # Performance tracking
        self.start_time = time.time()

    def track_step(
        self,
        step: int,
        rewards_by_function: Dict[str, np.ndarray],
        kl_divergence: Optional[float] = None,
        policy_entropy: Optional[float] = None,
        value_estimate: Optional[float] = None,
        advantages: Optional[np.ndarray] = None,
        batch_size: int = 1,
        num_generations: int = 1,
    ) -> RewardMetrics:
        """
        Track rewards and metrics for a single training step.

        Args:
            step: Training step number
            rewards_by_function: Dict mapping function names to reward arrays
            kl_divergence: KL divergence between policy and reference
            policy_entropy: Entropy of the policy distribution
            value_estimate: Estimated value from critic (if applicable)
            advantages: Advantage estimates for the batch
            batch_size: Number of examples in batch
            num_generations: Number of generations per example

        Returns:
            RewardMetrics object for this step
        """
        timestamp = time.time() - self.start_time

        # Initialize metrics object
        metrics = RewardMetrics(
            step=step,
            timestamp=timestamp,
            batch_size=batch_size,
            num_generations=num_generations,
            kl_divergence=kl_divergence,
            policy_entropy=policy_entropy,
            value_estimate=value_estimate,
        )

        # Compute per-function statistics
        total_rewards = np.zeros(batch_size * num_generations)

        for func_name, rewards in rewards_by_function.items():
            # Convert JAX arrays to numpy if needed
            if hasattr(rewards, '__array__'):
                rewards = np.array(rewards)
            else:
                rewards = np.asarray(rewards)

            # Flatten if needed
            rewards_flat = rewards.flatten()

            # Store raw scores if tracking distributions
            if self.track_distributions:
                metrics.reward_scores[func_name] = rewards_flat.tolist()

            # Compute statistics
            func_mean = float(np.mean(rewards_flat))
            func_std = float(np.std(rewards_flat))

            metrics.function_means[func_name] = func_mean
            metrics.function_stds[func_name] = func_std

            # Update running statistics
            self.running_totals[func_name] += np.sum(rewards_flat)
            self.running_counts[func_name] += len(rewards_flat)
            self.running_squared[func_name] += np.sum(rewards_flat ** 2)

            # Accumulate total rewards
            total_rewards += rewards_flat

        # Compute aggregate statistics
        metrics.total_reward = float(np.sum(total_rewards))
        metrics.mean_reward = float(np.mean(total_rewards))
        metrics.std_reward = float(np.std(total_rewards))
        metrics.min_reward = float(np.min(total_rewards))
        metrics.max_reward = float(np.max(total_rewards))

        # Compute per-function contributions (percentage of total)
        if metrics.total_reward > 0:
            for func_name in rewards_by_function.keys():
                func_total = np.sum(rewards_by_function[func_name])
                metrics.function_contributions[func_name] = (
                    float(func_total) / metrics.total_reward * 100
                )

        # Compute quality indicators
        metrics.reward_variance = float(np.var(total_rewards))

        # Signal-to-noise ratio (mean / std)
        if metrics.std_reward > 0:
            metrics.signal_to_noise = abs(metrics.mean_reward) / metrics.std_reward

        # Reward entropy (measure of distribution uniformity)
        # Normalize rewards to probabilities
        if len(total_rewards) > 1:
            rewards_normalized = total_rewards - np.min(total_rewards)
            if np.sum(rewards_normalized) > 0:
                reward_probs = rewards_normalized / np.sum(rewards_normalized)
                # Add small epsilon to avoid log(0)
                reward_probs = reward_probs + 1e-10
                metrics.reward_entropy = float(-np.sum(reward_probs * np.log(reward_probs)))

        # Track advantages if provided
        if advantages is not None:
            advantages_array = np.array(advantages) if hasattr(advantages, '__array__') else np.asarray(advantages)
            metrics.advantage_mean = float(np.mean(advantages_array))
            metrics.advantage_std = float(np.std(advantages_array))

        # Anomaly detection
        self._detect_anomalies(step, metrics)

        # Store metrics
        self.metrics_history.append(metrics)
        self.step_to_metrics[step] = metrics

        # Trim history if needed (keep window_size + 10% buffer)
        if len(self.metrics_history) > self.window_size * 1.1:
            self.metrics_history = self.metrics_history[-self.window_size:]

        return metrics

    def _detect_anomalies(self, step: int, metrics: RewardMetrics):
        """Detect anomalous reward signals."""
        if len(self.metrics_history) < 10:  # Need baseline
            return

        # Get recent history for comparison
        recent_means = [m.mean_reward for m in self.metrics_history[-50:]]
        historical_mean = np.mean(recent_means)
        historical_std = np.std(recent_means)

        if historical_std > 0:
            z_score = abs(metrics.mean_reward - historical_mean) / historical_std

            if z_score > self.anomaly_threshold:
                self.detected_anomalies.append({
                    'step': step,
                    'metric': 'mean_reward',
                    'value': metrics.mean_reward,
                    'z_score': z_score,
                    'historical_mean': historical_mean,
                    'historical_std': historical_std,
                })

    def get_moving_average(
        self,
        metric_name: str,
        window: Optional[int] = None,
    ) -> List[float]:
        """
        Compute moving average of a metric.

        Args:
            metric_name: Name of metric (e.g., 'mean_reward', 'signal_to_noise')
            window: Window size (defaults to self.window_size)

        Returns:
            List of moving average values
        """
        if window is None:
            window = self.window_size

        values = [getattr(m, metric_name, 0.0) for m in self.metrics_history]

        if len(values) < window:
            return values

        moving_avg = []
        for i in range(len(values)):
            start_idx = max(0, i - window + 1)
            window_values = values[start_idx:i+1]
            moving_avg.append(np.mean(window_values))

        return moving_avg

    def get_function_statistics(self, func_name: str) -> Dict[str, float]:
        """
        Get cumulative statistics for a specific reward function.

        Args:
            func_name: Name of the reward function

        Returns:
            Dict with mean, std, total, count
        """
        if func_name not in self.running_counts or self.running_counts[func_name] == 0:
            return {
                'mean': 0.0,
                'std': 0.0,
                'total': 0.0,
                'count': 0,
            }

        count = self.running_counts[func_name]
        total = self.running_totals[func_name]
        squared = self.running_squared[func_name]

        mean = total / count
        variance = (squared / count) - (mean ** 2)
        std = np.sqrt(max(0, variance))

        return {
            'mean': float(mean),
            'std': float(std),
            'total': float(total),
            'count': int(count),
        }

    def get_recent_trends(self, num_steps: int = 50) -> Dict[str, Any]:
        """
        Analyze recent trends in reward signals.

        Args:
            num_steps: Number of recent steps to analyze

        Returns:
            Dict with trend analysis
        """
        if len(self.metrics_history) < 2:
            return {}

        recent = self.metrics_history[-num_steps:]

        # Compute trends (simple linear regression)
        steps = np.array([m.step for m in recent])
        means = np.array([m.mean_reward for m in recent])

        # Slope of reward over time
        if len(steps) > 1:
            slope, intercept = np.polyfit(steps, means, 1)
        else:
            slope, intercept = 0.0, means[0]

        trends = {
            'num_steps_analyzed': len(recent),
            'mean_reward_trend': {
                'slope': float(slope),
                'intercept': float(intercept),
                'direction': 'increasing' if slope > 0 else 'decreasing',
                'current': float(recent[-1].mean_reward),
                'start': float(recent[0].mean_reward),
                'change': float(recent[-1].mean_reward - recent[0].mean_reward),
            },
            'signal_to_noise': {
                'current': float(recent[-1].signal_to_noise),
                'mean': float(np.mean([m.signal_to_noise for m in recent])),
                'std': float(np.std([m.signal_to_noise for m in recent])),
            },
            'reward_variance': {
                'current': float(recent[-1].reward_variance),
                'mean': float(np.mean([m.reward_variance for m in recent])),
                'trend': 'increasing' if recent[-1].reward_variance > recent[0].reward_variance else 'decreasing',
            },
        }

        # Per-function trends
        function_trends = {}
        for func_name in self.reward_function_names:
            func_means = [m.function_means.get(func_name, 0.0) for m in recent]
            if len(func_means) > 1:
                func_slope, func_intercept = np.polyfit(steps, func_means, 1)
            else:
                func_slope = 0.0

            function_trends[func_name] = {
                'slope': float(func_slope),
                'current': float(func_means[-1]),
                'mean': float(np.mean(func_means)),
                'direction': 'increasing' if func_slope > 0 else 'decreasing',
            }

        trends['function_trends'] = function_trends

        return trends

    def get_summary(self) -> Dict[str, Any]:
        """
        Get comprehensive summary of all tracked metrics.

        Returns:
            Dict with summary statistics
        """
        if not self.metrics_history:
            return {'message': 'No metrics tracked yet'}

        latest = self.metrics_history[-1]

        summary = {
            'latest_step': latest.step,
            'total_steps_tracked': len(self.metrics_history),
            'elapsed_time': latest.timestamp,

            'current_metrics': {
                'mean_reward': latest.mean_reward,
                'std_reward': latest.std_reward,
                'min_reward': latest.min_reward,
                'max_reward': latest.max_reward,
                'signal_to_noise': latest.signal_to_noise,
                'reward_entropy': latest.reward_entropy,
            },

            'function_statistics': {
                func_name: self.get_function_statistics(func_name)
                for func_name in self.reward_function_names
            },

            'recent_trends': self.get_recent_trends(num_steps=50),

            'anomalies_detected': len(self.detected_anomalies),
            'recent_anomalies': self.detected_anomalies[-5:] if self.detected_anomalies else [],
        }

        # Add training dynamics if available
        if latest.kl_divergence is not None:
            summary['training_dynamics'] = {
                'kl_divergence': latest.kl_divergence,
                'policy_entropy': latest.policy_entropy,
                'value_estimate': latest.value_estimate,
                'advantage_mean': latest.advantage_mean,
                'advantage_std': latest.advantage_std,
            }

        return summary

    def export_to_dict(self) -> Dict[str, Any]:
        """Export all metrics to a dictionary for serialization."""
        return {
            'config': {
                'reward_function_names': self.reward_function_names,
                'window_size': self.window_size,
                'track_distributions': self.track_distributions,
            },
            'metrics_history': [
                {
                    'step': m.step,
                    'timestamp': m.timestamp,
                    'mean_reward': m.mean_reward,
                    'std_reward': m.std_reward,
                    'function_means': m.function_means,
                    'function_contributions': m.function_contributions,
                    'signal_to_noise': m.signal_to_noise,
                    'kl_divergence': m.kl_divergence,
                }
                for m in self.metrics_history
            ],
            'summary': self.get_summary(),
        }
