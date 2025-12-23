"""
Metrics Collection and Logging for W&B and TensorBoard

Provides utilities to log reward distributions, histograms, and time series data
to various monitoring backends.
"""

from typing import Dict, List, Optional, Any, Union
import numpy as np
from collections import defaultdict
import warnings

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False
    warnings.warn("wandb not available, W&B logging will be disabled")


class MetricsCollector:
    """
    Collects metrics over multiple batches for aggregation and logging.

    Useful for collecting reward values across multiple training steps
    before computing distributions and histograms.
    """

    def __init__(self):
        """Initialize metrics collector."""
        self.metrics: Dict[str, List[float]] = defaultdict(list)
        self.step_metrics: List[Dict[str, float]] = []
        self.current_step = 0

    def add(self, name: str, value: Union[float, int, np.ndarray]):
        """
        Add a metric value.

        Args:
            name: Metric name
            value: Metric value (scalar or array)
        """
        if isinstance(value, np.ndarray):
            self.metrics[name].extend(value.flatten().tolist())
        else:
            self.metrics[name].append(float(value))

    def add_dict(self, metrics: Dict[str, Union[float, int, np.ndarray]]):
        """
        Add multiple metrics from a dictionary.

        Args:
            metrics: Dictionary of metric names to values
        """
        for name, value in metrics.items():
            self.add(name, value)

    def get_stats(self, name: str) -> Dict[str, float]:
        """
        Get statistics for a collected metric.

        Args:
            name: Metric name

        Returns:
            Dictionary with mean, std, min, max, median, count
        """
        if name not in self.metrics or not self.metrics[name]:
            return {}

        values = np.array(self.metrics[name])

        return {
            f"{name}/mean": float(np.mean(values)),
            f"{name}/std": float(np.std(values)),
            f"{name}/min": float(np.min(values)),
            f"{name}/max": float(np.max(values)),
            f"{name}/median": float(np.median(values)),
            f"{name}/count": len(values),
        }

    def get_all_stats(self) -> Dict[str, float]:
        """Get statistics for all collected metrics."""
        all_stats = {}
        for name in self.metrics.keys():
            all_stats.update(self.get_stats(name))
        return all_stats

    def get_histogram_data(self, name: str, bins: int = 50) -> Optional[Dict[str, Any]]:
        """
        Get histogram data for a metric.

        Args:
            name: Metric name
            bins: Number of histogram bins

        Returns:
            Dictionary with histogram counts and bin edges
        """
        if name not in self.metrics or not self.metrics[name]:
            return None

        values = np.array(self.metrics[name])
        counts, bin_edges = np.histogram(values, bins=bins)

        return {
            "counts": counts.tolist(),
            "bin_edges": bin_edges.tolist(),
            "values": values.tolist(),
        }

    def reset(self):
        """Clear all collected metrics."""
        self.metrics.clear()
        self.step_metrics.clear()

    def __len__(self):
        """Return number of unique metrics collected."""
        return len(self.metrics)


class RewardMetricsLogger:
    """
    Logs reward metrics to W&B and TensorBoard with support for histograms,
    distributions, and time series data.
    """

    def __init__(
        self,
        use_wandb: bool = True,
        log_histograms: bool = True,
        histogram_bins: int = 50,
        log_frequency: int = 1,
    ):
        """
        Initialize metrics logger.

        Args:
            use_wandb: Whether to log to W&B
            log_histograms: Whether to log histogram data
            histogram_bins: Number of bins for histograms
            log_frequency: Log every N steps
        """
        self.use_wandb = use_wandb and WANDB_AVAILABLE
        self.log_histograms = log_histograms
        self.histogram_bins = histogram_bins
        self.log_frequency = log_frequency

        self.step = 0
        self.collector = MetricsCollector()

        if self.use_wandb and not wandb.run:
            warnings.warn("W&B run not initialized, W&B logging will be disabled")
            self.use_wandb = False

    def log_reward_batch(
        self,
        reward_values: Dict[str, Union[float, List[float], np.ndarray]],
        step: Optional[int] = None,
        prefix: str = "rewards",
    ):
        """
        Log a batch of reward values.

        Args:
            reward_values: Dictionary mapping reward names to values
            step: Optional training step
            prefix: Prefix for metric names
        """
        if step is not None:
            self.step = step

        # Collect values for histogram generation
        for name, values in reward_values.items():
            full_name = f"{prefix}/{name}"
            self.collector.add(full_name, values)

    def log_step(
        self,
        metrics: Dict[str, float],
        step: Optional[int] = None,
        commit: bool = True,
    ):
        """
        Log metrics for a single step.

        Args:
            metrics: Dictionary of metrics to log
            step: Training step
            commit: Whether to commit the step (W&B)
        """
        if step is not None:
            self.step = step

        # Only log if at frequency interval
        if self.step % self.log_frequency != 0:
            return

        if self.use_wandb:
            wandb.log(metrics, step=self.step, commit=commit)

    def log_distributions(
        self,
        step: Optional[int] = None,
        reset_after_log: bool = True,
    ):
        """
        Log collected distributions and histograms.

        Args:
            step: Training step
            reset_after_log: Whether to reset collector after logging
        """
        if step is not None:
            self.step = step

        if not self.collector.metrics:
            return

        # Log statistics for all collected metrics
        stats = self.collector.get_all_stats()
        self.log_step(stats, step=self.step, commit=False)

        # Log histograms if enabled
        if self.log_histograms and self.use_wandb:
            for name in self.collector.metrics.keys():
                hist_data = self.collector.get_histogram_data(name, bins=self.histogram_bins)
                if hist_data and hist_data["values"]:
                    try:
                        wandb.log(
                            {f"{name}/histogram": wandb.Histogram(np_histogram=(
                                np.array(hist_data["counts"]),
                                np.array(hist_data["bin_edges"])
                            ))},
                            step=self.step,
                            commit=False,
                        )
                    except Exception as e:
                        warnings.warn(f"Failed to log histogram for {name}: {e}")

        # Commit all logs for this step
        if self.use_wandb:
            wandb.log({}, step=self.step, commit=True)

        # Reset collector if requested
        if reset_after_log:
            self.collector.reset()

    def log_summary_table(
        self,
        reward_stats: Dict[str, Any],
        step: Optional[int] = None,
    ):
        """
        Log a summary table of reward statistics.

        Args:
            reward_stats: Dictionary of reward statistics
            step: Training step
        """
        if step is not None:
            self.step = step

        if not self.use_wandb:
            return

        try:
            # Create a table with reward statistics
            table_data = []

            for name, stats in reward_stats.items():
                if hasattr(stats, "to_dict"):
                    stats_dict = stats.to_dict()
                else:
                    stats_dict = stats

                table_data.append([
                    name,
                    stats_dict.get(f"{name}/mean", 0.0),
                    stats_dict.get(f"{name}/std", 0.0),
                    stats_dict.get(f"{name}/min", 0.0),
                    stats_dict.get(f"{name}/max", 0.0),
                    stats_dict.get(f"{name}/median", 0.0),
                ])

            table = wandb.Table(
                columns=["Reward", "Mean", "Std", "Min", "Max", "Median"],
                data=table_data,
            )

            wandb.log({f"reward_summary_step_{self.step}": table}, step=self.step)

        except Exception as e:
            warnings.warn(f"Failed to log summary table: {e}")

    def log_alerts(
        self,
        alerts: List[Dict[str, Any]],
        step: Optional[int] = None,
    ):
        """
        Log anomaly alerts.

        Args:
            alerts: List of alert dictionaries
            step: Training step
        """
        if step is not None:
            self.step = step

        if not alerts:
            return

        # Count alerts by type
        alert_types = defaultdict(int)
        for alert in alerts:
            msg = alert.get("message", "")
            alert_type = msg.split(":")[0] if ":" in msg else "UNKNOWN"
            alert_types[alert_type] += 1

        # Log alert counts
        alert_metrics = {
            f"alerts/{alert_type}": count
            for alert_type, count in alert_types.items()
        }
        alert_metrics["alerts/total"] = len(alerts)

        self.log_step(alert_metrics, step=self.step, commit=False)

        # Log alert messages to W&B if available
        if self.use_wandb:
            try:
                alert_table = wandb.Table(
                    columns=["Step", "Reward", "Type", "Message", "Value"],
                    data=[
                        [
                            alert.get("step", self.step),
                            alert.get("reward", "unknown"),
                            alert.get("message", "").split(":")[0],
                            alert.get("message", ""),
                            alert.get("value", 0.0),
                        ]
                        for alert in alerts
                    ],
                )
                wandb.log({f"alerts_step_{self.step}": alert_table}, step=self.step)
            except Exception as e:
                warnings.warn(f"Failed to log alert table: {e}")

    def close(self):
        """Cleanup and finalize logging."""
        if self.use_wandb and wandb.run:
            # Log any remaining collected metrics
            if self.collector.metrics:
                self.log_distributions(reset_after_log=False)
