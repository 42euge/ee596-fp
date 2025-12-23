"""
Integration utilities for adding monitoring to GRPO training.

Provides easy-to-use wrappers and helpers for integrating reward monitoring
into GRPO training scripts.
"""

from typing import List, Callable, Optional, Dict, Any
import warnings

from .reward_monitor import RewardMonitor
from .metrics_logger import RewardMetricsLogger
from .reward_wrapper import RewardFunctionMonitor
from .visualization import RewardVisualizer, create_monitoring_summary_html

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False


class GRPOMonitoringIntegration:
    """
    All-in-one monitoring integration for GRPO training.

    Wraps reward functions, tracks statistics, logs to W&B, and creates visualizations.
    """

    def __init__(
        self,
        reward_functions: List[Callable],
        reward_names: Optional[List[str]] = None,
        use_wandb: bool = True,
        enable_anomaly_detection: bool = True,
        log_frequency: int = 10,
        summary_frequency: int = 100,
        visualization_frequency: int = 500,
        output_dir: str = "/tmp/reward_monitoring",
        verbose: bool = True,
    ):
        """
        Initialize GRPO monitoring integration.

        Args:
            reward_functions: List of reward functions to monitor
            reward_names: Optional list of reward names (defaults to function names)
            use_wandb: Whether to log to W&B
            enable_anomaly_detection: Whether to detect anomalies
            log_frequency: Log metrics every N steps
            summary_frequency: Print summary every N steps
            visualization_frequency: Create visualizations every N steps
            output_dir: Directory for saving outputs
            verbose: Whether to print monitoring information
        """
        self.reward_functions = reward_functions
        self.output_dir = output_dir
        self.verbose = verbose
        self.log_frequency = log_frequency
        self.summary_frequency = summary_frequency
        self.visualization_frequency = visualization_frequency

        # Get reward names
        if reward_names is None:
            reward_names = [
                getattr(fn, "__name__", f"reward_{i}")
                for i, fn in enumerate(reward_functions)
            ]
        self.reward_names = reward_names

        # Initialize monitoring components
        self.reward_monitor = RewardMonitor(
            reward_names=reward_names,
            window_size=1000,
            enable_anomaly_detection=enable_anomaly_detection,
            verbose=verbose,
        )

        self.metrics_logger = RewardMetricsLogger(
            use_wandb=use_wandb and WANDB_AVAILABLE,
            log_histograms=True,
            histogram_bins=50,
            log_frequency=log_frequency,
        )

        self.function_monitor = RewardFunctionMonitor(verbose=verbose)

        self.visualizer = RewardVisualizer()

        # Wrap reward functions
        self.wrapped_functions = []
        for fn, name in zip(reward_functions, reward_names):
            wrapped = self.function_monitor.wrap(fn, name=name)
            self.wrapped_functions.append(wrapped)

        # Create output directory
        import os
        os.makedirs(output_dir, exist_ok=True)

        if self.verbose:
            print("\n" + "=" * 80)
            print("Reward Monitoring System Initialized")
            print("=" * 80)
            print(f"Monitoring {len(reward_functions)} reward functions:")
            for name in reward_names:
                print(f"  - {name}")
            print(f"Output directory: {output_dir}")
            print(f"W&B logging: {'enabled' if use_wandb and WANDB_AVAILABLE else 'disabled'}")
            print(f"Anomaly detection: {'enabled' if enable_anomaly_detection else 'disabled'}")
            print("=" * 80 + "\n")

    def get_wrapped_reward_functions(self) -> List[Callable]:
        """Get the list of wrapped reward functions to pass to GRPO trainer."""
        return self.wrapped_functions

    def update_step(self, step: int, reward_values: Optional[Dict[str, float]] = None):
        """
        Update monitoring for a training step.

        Args:
            step: Training step number
            reward_values: Optional dictionary of reward values
                          (can be auto-extracted from wrapped functions if None)
        """
        # If reward values not provided, get from recent calls to wrapped functions
        if reward_values is None:
            reward_values = {}
            for name, wrapper in zip(self.reward_names, self.wrapped_functions):
                if wrapper.recent_values:
                    # Use most recent value
                    reward_values[name] = wrapper.recent_values[-1]

        if not reward_values:
            return

        # Update reward monitor
        stats = self.reward_monitor.update(reward_values, step=step)

        # Log to metrics logger
        self.metrics_logger.log_reward_batch(reward_values, step=step)

        # Log function execution metrics
        fn_metrics = self.function_monitor.get_metrics_dict()
        if fn_metrics:
            self.metrics_logger.log_step(fn_metrics, step=step, commit=False)

        # Log reward statistics
        metrics = self.reward_monitor.get_metrics_dict()
        self.metrics_logger.log_step(metrics, step=step, commit=True)

        # Log distributions periodically
        if step % self.log_frequency == 0:
            self.metrics_logger.log_distributions(step=step, reset_after_log=True)

        # Log alerts if any
        recent_alerts = [
            a for a in self.reward_monitor.all_alerts
            if a["step"] >= step - self.log_frequency
        ]
        if recent_alerts:
            self.metrics_logger.log_alerts(recent_alerts, step=step)

        # Print summary periodically
        if step % self.summary_frequency == 0 and self.verbose:
            print(self.reward_monitor.get_summary())
            self.function_monitor.print_summary()

        # Create visualizations periodically
        if step % self.visualization_frequency == 0:
            self._create_visualizations(step)

    def _create_visualizations(self, step: int):
        """Create and save visualizations."""
        import os

        try:
            # Create dashboard
            dashboard_path = os.path.join(
                self.output_dir, f"dashboard_step_{step}.png"
            )
            self.visualizer.plot_reward_dashboard(
                self.reward_monitor,
                save_path=dashboard_path,
                show=False,
            )

            # Create HTML summary
            html_path = os.path.join(
                self.output_dir, f"summary_step_{step}.html"
            )
            create_monitoring_summary_html(
                self.reward_monitor,
                output_path=html_path,
            )

            if self.verbose:
                print(f"✓ Visualizations saved to {self.output_dir}")

        except Exception as e:
            warnings.warn(f"Failed to create visualizations: {e}")

    def finalize(self):
        """Finalize monitoring and create final reports."""
        if self.verbose:
            print("\n" + "=" * 80)
            print("Finalizing Reward Monitoring")
            print("=" * 80)

        # Print final summary
        print(self.reward_monitor.get_summary())
        self.function_monitor.print_summary()

        # Create final visualizations
        import os

        try:
            # Final dashboard
            final_dashboard = os.path.join(self.output_dir, "final_dashboard.png")
            self.visualizer.plot_reward_dashboard(
                self.reward_monitor,
                save_path=final_dashboard,
                show=False,
            )

            # Final HTML report
            final_html = os.path.join(self.output_dir, "final_summary.html")
            create_monitoring_summary_html(
                self.reward_monitor,
                output_path=final_html,
            )

            # Reward history plot
            history_plot = os.path.join(self.output_dir, "reward_history.png")
            self.visualizer.plot_reward_history(
                self.reward_monitor.reward_history,
                save_path=history_plot,
                show=False,
            )

            # Distribution plots
            dist_plot = os.path.join(self.output_dir, "reward_distributions.png")
            self.visualizer.plot_reward_distributions(
                self.reward_monitor.reward_history,
                save_path=dist_plot,
                show=False,
            )

            # Statistics plots
            stats_plot = os.path.join(self.output_dir, "reward_statistics.png")
            self.visualizer.plot_reward_statistics(
                self.reward_monitor.get_all_stats(),
                save_path=stats_plot,
                show=False,
            )

            # Save monitoring data
            self._save_monitoring_data()

            print(f"\n✓ Final reports saved to {self.output_dir}")

        except Exception as e:
            warnings.warn(f"Failed to create final reports: {e}")

        # Close logger
        self.metrics_logger.close()

        print("=" * 80 + "\n")

    def _save_monitoring_data(self):
        """Save monitoring data to JSON for later analysis."""
        import os
        import json

        # Convert alerts to JSON-serializable format
        alerts_serializable = []
        for alert in self.reward_monitor.all_alerts:
            alert_dict = {
                "step": alert.get("step"),
                "reward": alert.get("reward"),
                "message": alert.get("message"),
                "value": alert.get("value"),
            }
            # Convert stats if present
            if "stats" in alert and hasattr(alert["stats"], "mean"):
                stats = alert["stats"]
                alert_dict["stats"] = {
                    "mean": stats.mean,
                    "std": stats.std,
                    "ema_short": stats.ema_short,
                    "ema_long": stats.ema_long,
                }
            alerts_serializable.append(alert_dict)

        data = {
            "reward_names": self.reward_names,
            "global_step": self.reward_monitor.global_step,
            "stats": {
                name: {
                    "mean": s.mean,
                    "std": s.std,
                    "min": s.min,
                    "max": s.max,
                    "median": s.median,
                    "count": s.count,
                    "ema_short": s.ema_short,
                    "ema_long": s.ema_long,
                    "positive_fraction": s.positive_fraction,
                    "negative_fraction": s.negative_fraction,
                    "zeros_fraction": s.zeros_fraction,
                }
                for name, s in self.reward_monitor.stats.items()
            },
            "alerts": alerts_serializable,
            "function_stats": self.function_monitor.get_all_stats(),
        }

        json_path = os.path.join(self.output_dir, "monitoring_data.json")
        with open(json_path, "w") as f:
            json.dump(data, f, indent=2)

        if self.verbose:
            print(f"✓ Monitoring data saved to {json_path}")


def setup_grpo_monitoring(
    reward_functions: List[Callable],
    **kwargs,
) -> GRPOMonitoringIntegration:
    """
    Convenience function to set up GRPO monitoring.

    Args:
        reward_functions: List of reward functions
        **kwargs: Additional arguments to pass to GRPOMonitoringIntegration

    Returns:
        Configured monitoring integration
    """
    return GRPOMonitoringIntegration(reward_functions, **kwargs)
