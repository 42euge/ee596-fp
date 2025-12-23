"""
Reward Monitoring System

Tracks reward signal quality, detects anomalies, and provides statistical analysis
during GRPO training runs.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable, Any
import numpy as np
from collections import defaultdict, deque
import warnings


@dataclass
class RewardStats:
    """Statistics for a single reward function over a time window."""

    name: str
    mean: float = 0.0
    std: float = 0.0
    min: float = float('inf')
    max: float = float('-inf')
    median: float = 0.0
    count: int = 0
    sum: float = 0.0

    # Distribution percentiles
    p25: float = 0.0
    p75: float = 0.0
    p90: float = 0.0
    p99: float = 0.0

    # Moving averages
    ema_short: float = 0.0  # Exponential moving average (fast)
    ema_long: float = 0.0   # Exponential moving average (slow)

    # Quality indicators
    zeros_fraction: float = 0.0  # Fraction of zero rewards
    negative_fraction: float = 0.0  # Fraction of negative rewards
    positive_fraction: float = 0.0  # Fraction of positive rewards

    def to_dict(self) -> Dict[str, float]:
        """Convert stats to dictionary for logging."""
        return {
            f"{self.name}/mean": self.mean,
            f"{self.name}/std": self.std,
            f"{self.name}/min": self.min,
            f"{self.name}/max": self.max,
            f"{self.name}/median": self.median,
            f"{self.name}/count": float(self.count),
            f"{self.name}/p25": self.p25,
            f"{self.name}/p75": self.p75,
            f"{self.name}/p90": self.p90,
            f"{self.name}/p99": self.p99,
            f"{self.name}/ema_short": self.ema_short,
            f"{self.name}/ema_long": self.ema_long,
            f"{self.name}/zeros_fraction": self.zeros_fraction,
            f"{self.name}/negative_fraction": self.negative_fraction,
            f"{self.name}/positive_fraction": self.positive_fraction,
        }


@dataclass
class RewardAnomalyDetector:
    """Detects anomalies in reward signals using statistical methods."""

    # Configuration
    window_size: int = 100
    std_threshold: float = 3.0  # Number of standard deviations for outliers
    drop_threshold: float = 0.5  # Fraction drop in moving average to trigger alert
    spike_threshold: float = 2.0  # Multiplier for sudden spikes

    # State
    history: deque = field(default_factory=lambda: deque(maxlen=100))
    alerts: List[str] = field(default_factory=list)

    def __post_init__(self):
        """Initialize history with correct maxlen."""
        self.history = deque(maxlen=self.window_size)

    def detect(self, value: float, name: str, stats: RewardStats) -> List[str]:
        """
        Detect anomalies in reward signal.

        Args:
            value: Current reward value
            name: Name of reward function
            stats: Current statistics

        Returns:
            List of alert messages
        """
        alerts = []
        self.history.append(value)

        if len(self.history) < 10:  # Need minimum samples
            return alerts

        # Check for outliers (z-score method)
        if stats.std > 0:
            z_score = abs((value - stats.mean) / stats.std)
            if z_score > self.std_threshold:
                alerts.append(
                    f"OUTLIER: {name} value {value:.3f} is {z_score:.1f} std "
                    f"deviations from mean {stats.mean:.3f}"
                )

        # Check for sudden drops in reward
        if stats.ema_long > 0 and stats.ema_short < stats.ema_long * (1 - self.drop_threshold):
            drop_pct = (1 - stats.ema_short / stats.ema_long) * 100
            alerts.append(
                f"DROP: {name} short EMA {stats.ema_short:.3f} dropped {drop_pct:.1f}% "
                f"below long EMA {stats.ema_long:.3f}"
            )

        # Check for sudden spikes
        if stats.ema_long > 0 and stats.ema_short > stats.ema_long * self.spike_threshold:
            spike_mult = stats.ema_short / stats.ema_long
            alerts.append(
                f"SPIKE: {name} short EMA {stats.ema_short:.3f} is {spike_mult:.1f}x "
                f"higher than long EMA {stats.ema_long:.3f}"
            )

        # Check for all zeros (reward function not working)
        if len(self.history) >= self.window_size and stats.zeros_fraction > 0.95:
            alerts.append(
                f"FLATLINE: {name} is returning zero {stats.zeros_fraction*100:.1f}% of the time"
            )

        # Check for all negative rewards (potential issue)
        if stats.negative_fraction > 0.9:
            alerts.append(
                f"NEGATIVE: {name} is negative {stats.negative_fraction*100:.1f}% of the time"
            )

        return alerts


class RewardMonitor:
    """
    Central monitoring system for tracking reward signals during training.

    Tracks individual reward functions, computes statistics, detects anomalies,
    and logs metrics for observability.
    """

    def __init__(
        self,
        reward_names: List[str],
        window_size: int = 1000,
        ema_alpha_short: float = 0.1,  # ~10 sample window
        ema_alpha_long: float = 0.01,  # ~100 sample window
        enable_anomaly_detection: bool = True,
        verbose: bool = True,
    ):
        """
        Initialize reward monitor.

        Args:
            reward_names: Names of reward functions to track
            window_size: Number of samples to keep in rolling window
            ema_alpha_short: Smoothing factor for short-term EMA
            ema_alpha_long: Smoothing factor for long-term EMA
            enable_anomaly_detection: Whether to detect and alert on anomalies
            verbose: Whether to print alerts to console
        """
        self.reward_names = reward_names
        self.window_size = window_size
        self.ema_alpha_short = ema_alpha_short
        self.ema_alpha_long = ema_alpha_long
        self.enable_anomaly_detection = enable_anomaly_detection
        self.verbose = verbose

        # Initialize tracking structures
        self.reward_history: Dict[str, deque] = {
            name: deque(maxlen=window_size) for name in reward_names
        }

        self.stats: Dict[str, RewardStats] = {
            name: RewardStats(name=name) for name in reward_names
        }

        self.anomaly_detectors: Dict[str, RewardAnomalyDetector] = {
            name: RewardAnomalyDetector(window_size=min(window_size, 100))
            for name in reward_names
        }

        self.global_step = 0
        self.all_alerts: List[Dict[str, Any]] = []

    def update(
        self,
        reward_values: Dict[str, float],
        step: Optional[int] = None,
    ) -> Dict[str, RewardStats]:
        """
        Update monitoring with new reward values.

        Args:
            reward_values: Dictionary mapping reward name to value
            step: Optional training step number

        Returns:
            Dictionary of updated statistics for each reward
        """
        if step is not None:
            self.global_step = step
        else:
            self.global_step += 1

        updated_stats = {}

        for name, value in reward_values.items():
            if name not in self.reward_names:
                warnings.warn(f"Unknown reward function '{name}', adding to monitoring")
                self.reward_names.append(name)
                self.reward_history[name] = deque(maxlen=self.window_size)
                self.stats[name] = RewardStats(name=name)
                self.anomaly_detectors[name] = RewardAnomalyDetector(
                    window_size=min(self.window_size, 100)
                )

            # Update history
            self.reward_history[name].append(value)

            # Compute statistics
            stats = self._compute_stats(name)
            self.stats[name] = stats
            updated_stats[name] = stats

            # Detect anomalies
            if self.enable_anomaly_detection:
                alerts = self.anomaly_detectors[name].detect(value, name, stats)
                for alert in alerts:
                    alert_record = {
                        "step": self.global_step,
                        "reward": name,
                        "message": alert,
                        "value": value,
                        "stats": stats,
                    }
                    self.all_alerts.append(alert_record)

                    if self.verbose:
                        print(f"⚠️  Step {self.global_step}: {alert}")

        return updated_stats

    def _compute_stats(self, name: str) -> RewardStats:
        """Compute statistics for a reward function."""
        history = list(self.reward_history[name])

        if not history:
            return RewardStats(name=name)

        arr = np.array(history)

        # Basic statistics
        mean = float(np.mean(arr))
        std = float(np.std(arr))
        min_val = float(np.min(arr))
        max_val = float(np.max(arr))
        median = float(np.median(arr))
        count = len(arr)
        sum_val = float(np.sum(arr))

        # Percentiles
        p25 = float(np.percentile(arr, 25))
        p75 = float(np.percentile(arr, 75))
        p90 = float(np.percentile(arr, 90))
        p99 = float(np.percentile(arr, 99))

        # Update EMAs
        current_ema_short = self.stats[name].ema_short
        current_ema_long = self.stats[name].ema_long

        latest_value = history[-1]

        if current_ema_short == 0:  # First update
            ema_short = latest_value
            ema_long = latest_value
        else:
            ema_short = self.ema_alpha_short * latest_value + (1 - self.ema_alpha_short) * current_ema_short
            ema_long = self.ema_alpha_long * latest_value + (1 - self.ema_alpha_long) * current_ema_long

        # Quality indicators
        zeros_count = np.sum(arr == 0)
        negative_count = np.sum(arr < 0)
        positive_count = np.sum(arr > 0)

        zeros_fraction = zeros_count / count
        negative_fraction = negative_count / count
        positive_fraction = positive_count / count

        return RewardStats(
            name=name,
            mean=mean,
            std=std,
            min=min_val,
            max=max_val,
            median=median,
            count=count,
            sum=sum_val,
            p25=p25,
            p75=p75,
            p90=p90,
            p99=p99,
            ema_short=ema_short,
            ema_long=ema_long,
            zeros_fraction=zeros_fraction,
            negative_fraction=negative_fraction,
            positive_fraction=positive_fraction,
        )

    def get_all_stats(self) -> Dict[str, RewardStats]:
        """Get current statistics for all reward functions."""
        return self.stats.copy()

    def get_metrics_dict(self) -> Dict[str, float]:
        """
        Get all metrics as a flat dictionary for logging to W&B/TensorBoard.

        Returns:
            Dictionary with all reward statistics
        """
        metrics = {}

        # Individual reward statistics
        for name, stats in self.stats.items():
            metrics.update(stats.to_dict())

        # Aggregate statistics
        if self.stats:
            all_means = [s.mean for s in self.stats.values()]
            all_stds = [s.std for s in self.stats.values()]

            metrics["rewards/total_mean"] = float(np.mean(all_means))
            metrics["rewards/total_std"] = float(np.mean(all_stds))
            metrics["rewards/sum"] = sum(all_means)
            metrics["rewards/min_mean"] = min(all_means)
            metrics["rewards/max_mean"] = max(all_means)

        # Alert counts
        recent_alerts = [a for a in self.all_alerts if a["step"] >= self.global_step - 100]
        metrics["monitoring/alert_count"] = len(recent_alerts)
        metrics["monitoring/total_alerts"] = len(self.all_alerts)

        return metrics

    def get_summary(self) -> str:
        """Get a human-readable summary of reward statistics."""
        lines = ["=" * 80]
        lines.append(f"Reward Monitoring Summary (Step {self.global_step})")
        lines.append("=" * 80)

        for name, stats in self.stats.items():
            lines.append(f"\n{name}:")
            lines.append(f"  Mean: {stats.mean:.4f} ± {stats.std:.4f}")
            lines.append(f"  Range: [{stats.min:.4f}, {stats.max:.4f}]")
            lines.append(f"  Median: {stats.median:.4f}")
            lines.append(f"  Percentiles: P25={stats.p25:.4f}, P75={stats.p75:.4f}, P90={stats.p90:.4f}")
            lines.append(f"  EMAs: Short={stats.ema_short:.4f}, Long={stats.ema_long:.4f}")
            lines.append(f"  Distribution: {stats.positive_fraction*100:.1f}% positive, "
                        f"{stats.negative_fraction*100:.1f}% negative, "
                        f"{stats.zeros_fraction*100:.1f}% zero")
            lines.append(f"  Samples: {stats.count}")

        if self.all_alerts:
            lines.append(f"\n⚠️  Total Alerts: {len(self.all_alerts)}")
            recent = [a for a in self.all_alerts if a["step"] >= self.global_step - 100]
            if recent:
                lines.append(f"Recent Alerts (last 100 steps): {len(recent)}")
                for alert in recent[-5:]:  # Show last 5
                    lines.append(f"  Step {alert['step']}: {alert['message']}")

        lines.append("=" * 80)
        return "\n".join(lines)

    def reset(self):
        """Reset all monitoring state."""
        self.reward_history = {
            name: deque(maxlen=self.window_size) for name in self.reward_names
        }
        self.stats = {name: RewardStats(name=name) for name in self.reward_names}
        self.anomaly_detectors = {
            name: RewardAnomalyDetector(window_size=min(self.window_size, 100))
            for name in self.reward_names
        }
        self.global_step = 0
        self.all_alerts = []
