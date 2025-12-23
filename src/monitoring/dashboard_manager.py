"""
Dashboard Manager

Manages multiple dashboard backends for visualizing reward signals and training metrics.
Supports W&B, TensorBoard, and custom web dashboards.
"""

import os
import json
from typing import Dict, List, Optional, Any
from pathlib import Path
import numpy as np

try:
    import wandb
    WANDB_AVAILABLE = True
except ImportError:
    WANDB_AVAILABLE = False

try:
    from torch.utils.tensorboard import SummaryWriter
    TENSORBOARD_AVAILABLE = True
except ImportError:
    TENSORBOARD_AVAILABLE = False

from .metrics_tracker import RewardMetrics, RewardMetricsTracker


class DashboardManager:
    """
    Manages multiple dashboard backends for monitoring.

    Supports:
    - Weights & Biases (W&B)
    - TensorBoard
    - JSON export for custom dashboards
    """

    def __init__(
        self,
        project_name: str = "grpo-reward-monitoring",
        run_name: Optional[str] = None,
        log_dir: str = "/tmp/tensorboard/grpo_rewards",
        enable_wandb: bool = True,
        enable_tensorboard: bool = True,
        enable_json_export: bool = True,
        json_export_dir: str = "./monitoring_data",
    ):
        """
        Initialize dashboard manager.

        Args:
            project_name: Project name for W&B
            run_name: Run name (auto-generated if None)
            log_dir: Directory for TensorBoard logs
            enable_wandb: Whether to enable W&B logging
            enable_tensorboard: Whether to enable TensorBoard logging
            enable_json_export: Whether to export JSON data
            json_export_dir: Directory for JSON exports
        """
        self.project_name = project_name
        self.run_name = run_name
        self.log_dir = log_dir
        self.json_export_dir = json_export_dir

        # Initialize backends
        self.wandb_enabled = enable_wandb and WANDB_AVAILABLE and os.getenv('WANDB_API_KEY')
        self.tensorboard_enabled = enable_tensorboard and TENSORBOARD_AVAILABLE
        self.json_export_enabled = enable_json_export

        self.wandb_run = None
        self.tensorboard_writer = None

        # Setup backends
        self._setup_wandb()
        self._setup_tensorboard()
        self._setup_json_export()

        # Track custom charts created
        self.custom_charts_defined = False

    def _setup_wandb(self):
        """Initialize W&B if enabled."""
        if not self.wandb_enabled:
            print("⚠️  W&B disabled (not available or no API key)")
            return

        try:
            # Initialize W&B run (or reuse existing)
            if wandb.run is None:
                self.wandb_run = wandb.init(
                    project=self.project_name,
                    name=self.run_name,
                    config={
                        'monitoring_enabled': True,
                        'reward_tracking': 'enhanced',
                    }
                )
            else:
                self.wandb_run = wandb.run
                print(f"✓ W&B: Reusing existing run: {wandb.run.name}")

            print(f"✓ W&B initialized: {self.wandb_run.url}")

        except Exception as e:
            print(f"⚠️  W&B initialization failed: {e}")
            self.wandb_enabled = False

    def _setup_tensorboard(self):
        """Initialize TensorBoard if enabled."""
        if not self.tensorboard_enabled:
            print("⚠️  TensorBoard disabled (not available)")
            return

        try:
            os.makedirs(self.log_dir, exist_ok=True)
            self.tensorboard_writer = SummaryWriter(log_dir=self.log_dir)
            print(f"✓ TensorBoard initialized: {self.log_dir}")

            # Add custom layout configuration
            self._configure_tensorboard_layout()

        except Exception as e:
            print(f"⚠️  TensorBoard initialization failed: {e}")
            self.tensorboard_enabled = False

    def _setup_json_export(self):
        """Setup JSON export directory."""
        if not self.json_export_enabled:
            return

        try:
            os.makedirs(self.json_export_dir, exist_ok=True)
            print(f"✓ JSON export directory: {self.json_export_dir}")
        except Exception as e:
            print(f"⚠️  JSON export setup failed: {e}")
            self.json_export_enabled = False

    def _configure_tensorboard_layout(self):
        """Configure custom TensorBoard layout for reward monitoring."""
        if not self.tensorboard_writer:
            return

        # Define custom scalar layout
        layout = {
            "Reward Signals": {
                "Total Rewards": ["Multiline", ["rewards/mean", "rewards/total", "rewards/max"]],
                "Per-Function Rewards": ["Multiline", [
                    "rewards/functions/format_exact",
                    "rewards/functions/format_approx",
                    "rewards/functions/check_answer",
                    "rewards/functions/check_numbers",
                ]],
                "Quality Metrics": ["Multiline", [
                    "rewards/signal_to_noise",
                    "rewards/entropy",
                    "rewards/variance",
                ]],
            },
            "Training Dynamics": {
                "Policy Metrics": ["Multiline", [
                    "training/kl_divergence",
                    "training/policy_entropy",
                ]],
                "Advantages": ["Multiline", [
                    "training/advantage_mean",
                    "training/advantage_std",
                ]],
            },
            "Contributions": {
                "Reward Contributions (%)": ["Multiline", [
                    "contributions/format_exact",
                    "contributions/format_approx",
                    "contributions/check_answer",
                    "contributions/check_numbers",
                ]],
            },
        }

        # TensorBoard doesn't have direct layout API, but we can structure our logging
        # to follow this pattern

    def define_wandb_charts(self, reward_function_names: List[str]):
        """
        Define custom W&B charts for reward monitoring.

        Args:
            reward_function_names: Names of reward functions to track
        """
        if not self.wandb_enabled or self.custom_charts_defined:
            return

        try:
            # Create custom panel layout
            # W&B will auto-detect metrics, but we can define custom charts

            # Log the chart definitions as config
            chart_config = {
                'reward_functions': reward_function_names,
                'custom_charts': {
                    'reward_breakdown': 'Per-function reward contributions over time',
                    'quality_metrics': 'Signal quality indicators (SNR, entropy, variance)',
                    'training_dynamics': 'KL divergence and policy entropy',
                    'anomaly_detection': 'Detected reward signal anomalies',
                }
            }

            if self.wandb_run:
                self.wandb_run.config.update({'chart_definitions': chart_config})

            self.custom_charts_defined = True
            print("✓ W&B custom charts defined")

        except Exception as e:
            print(f"⚠️  Failed to define W&B charts: {e}")

    def log_metrics(
        self,
        step: int,
        metrics: RewardMetrics,
        prefix: str = "",
    ):
        """
        Log metrics to all enabled backends.

        Args:
            step: Training step number
            metrics: RewardMetrics object
            prefix: Optional prefix for metric names
        """
        # Prepare metrics dict
        metrics_dict = self._prepare_metrics_dict(metrics, prefix)

        # Log to each backend
        if self.wandb_enabled:
            self._log_to_wandb(step, metrics_dict)

        if self.tensorboard_enabled:
            self._log_to_tensorboard(step, metrics_dict)

        if self.json_export_enabled:
            self._export_to_json(step, metrics, metrics_dict)

    def _prepare_metrics_dict(
        self,
        metrics: RewardMetrics,
        prefix: str = "",
    ) -> Dict[str, Any]:
        """Prepare metrics dictionary for logging."""
        p = f"{prefix}/" if prefix else ""

        metrics_dict = {
            # Aggregate rewards
            f"{p}rewards/mean": metrics.mean_reward,
            f"{p}rewards/std": metrics.std_reward,
            f"{p}rewards/min": metrics.min_reward,
            f"{p}rewards/max": metrics.max_reward,
            f"{p}rewards/total": metrics.total_reward,

            # Quality indicators
            f"{p}rewards/signal_to_noise": metrics.signal_to_noise,
            f"{p}rewards/entropy": metrics.reward_entropy,
            f"{p}rewards/variance": metrics.reward_variance,
        }

        # Per-function means
        for func_name, mean_val in metrics.function_means.items():
            clean_name = func_name.replace('_', ' ').title()
            metrics_dict[f"{p}rewards/functions/{func_name}"] = mean_val

        # Per-function standard deviations
        for func_name, std_val in metrics.function_stds.items():
            metrics_dict[f"{p}rewards/functions/{func_name}_std"] = std_val

        # Per-function contributions (percentage)
        for func_name, contrib in metrics.function_contributions.items():
            metrics_dict[f"{p}contributions/{func_name}"] = contrib

        # Training dynamics
        if metrics.kl_divergence is not None:
            metrics_dict[f"{p}training/kl_divergence"] = metrics.kl_divergence

        if metrics.policy_entropy is not None:
            metrics_dict[f"{p}training/policy_entropy"] = metrics.policy_entropy

        if metrics.value_estimate is not None:
            metrics_dict[f"{p}training/value_estimate"] = metrics.value_estimate

        if metrics.advantage_mean is not None:
            metrics_dict[f"{p}training/advantage_mean"] = metrics.advantage_mean
            metrics_dict[f"{p}training/advantage_std"] = metrics.advantage_std

        # Batch info
        metrics_dict[f"{p}batch/size"] = metrics.batch_size
        metrics_dict[f"{p}batch/num_generations"] = metrics.num_generations

        return metrics_dict

    def _log_to_wandb(self, step: int, metrics_dict: Dict[str, Any]):
        """Log metrics to W&B."""
        try:
            wandb.log(metrics_dict, step=step)
        except Exception as e:
            print(f"⚠️  W&B logging failed: {e}")

    def _log_to_tensorboard(self, step: int, metrics_dict: Dict[str, Any]):
        """Log metrics to TensorBoard."""
        try:
            for name, value in metrics_dict.items():
                if isinstance(value, (int, float)) and not np.isnan(value):
                    self.tensorboard_writer.add_scalar(name, value, step)
        except Exception as e:
            print(f"⚠️  TensorBoard logging failed: {e}")

    def _export_to_json(
        self,
        step: int,
        metrics: RewardMetrics,
        metrics_dict: Dict[str, Any],
    ):
        """Export metrics to JSON file."""
        try:
            # Export step-specific data
            step_file = Path(self.json_export_dir) / f"step_{step:06d}.json"

            step_data = {
                'step': step,
                'timestamp': metrics.timestamp,
                'metrics': metrics_dict,
                'reward_scores': metrics.reward_scores if metrics.reward_scores else {},
            }

            with open(step_file, 'w') as f:
                json.dump(step_data, f, indent=2)

        except Exception as e:
            print(f"⚠️  JSON export failed: {e}")

    def log_anomaly(
        self,
        step: int,
        anomaly_info: Dict[str, Any],
    ):
        """
        Log detected anomaly.

        Args:
            step: Training step
            anomaly_info: Dict with anomaly details
        """
        if self.wandb_enabled:
            try:
                wandb.log({
                    'anomalies/detected': 1,
                    'anomalies/z_score': anomaly_info.get('z_score', 0),
                    'anomalies/metric': anomaly_info.get('metric', 'unknown'),
                }, step=step)

                # Create alert table
                if 'anomaly_table' not in wandb.run.summary:
                    wandb.run.summary['anomaly_table'] = wandb.Table(
                        columns=['step', 'metric', 'value', 'z_score']
                    )

            except Exception as e:
                print(f"⚠️  Anomaly logging failed: {e}")

    def log_summary(
        self,
        tracker: RewardMetricsTracker,
        step: int,
    ):
        """
        Log comprehensive summary of tracking statistics.

        Args:
            tracker: RewardMetricsTracker instance
            step: Current training step
        """
        summary = tracker.get_summary()

        if self.wandb_enabled:
            try:
                # Log summary statistics
                wandb.log({
                    'summary/total_steps_tracked': summary['total_steps_tracked'],
                    'summary/elapsed_time': summary['elapsed_time'],
                    'summary/anomalies_detected': summary['anomalies_detected'],
                }, step=step)

                # Update run summary
                wandb.run.summary.update({
                    'final_mean_reward': summary['current_metrics']['mean_reward'],
                    'final_signal_to_noise': summary['current_metrics']['signal_to_noise'],
                })

            except Exception as e:
                print(f"⚠️  Summary logging failed: {e}")

        if self.json_export_enabled:
            try:
                summary_file = Path(self.json_export_dir) / "summary.json"
                with open(summary_file, 'w') as f:
                    json.dump(summary, f, indent=2)
            except Exception as e:
                print(f"⚠️  Summary export failed: {e}")

    def log_histogram(
        self,
        step: int,
        name: str,
        values: np.ndarray,
    ):
        """
        Log histogram of values.

        Args:
            step: Training step
            name: Histogram name
            values: Array of values
        """
        if self.wandb_enabled:
            try:
                wandb.log({name: wandb.Histogram(values)}, step=step)
            except Exception as e:
                print(f"⚠️  Histogram logging failed: {e}")

        if self.tensorboard_enabled:
            try:
                self.tensorboard_writer.add_histogram(name, values, step)
            except Exception as e:
                print(f"⚠️  TensorBoard histogram failed: {e}")

    def log_reward_distributions(
        self,
        step: int,
        metrics: RewardMetrics,
    ):
        """
        Log full reward distributions as histograms.

        Args:
            step: Training step
            metrics: RewardMetrics with reward_scores
        """
        if not metrics.reward_scores:
            return

        for func_name, scores in metrics.reward_scores.items():
            if scores:
                scores_array = np.array(scores)
                self.log_histogram(
                    step,
                    f"distributions/{func_name}",
                    scores_array,
                )

    def close(self):
        """Close all dashboard backends."""
        if self.wandb_enabled and self.wandb_run:
            try:
                self.wandb_run.finish()
                print("✓ W&B run finished")
            except Exception as e:
                print(f"⚠️  W&B close failed: {e}")

        if self.tensorboard_enabled and self.tensorboard_writer:
            try:
                self.tensorboard_writer.close()
                print("✓ TensorBoard writer closed")
            except Exception as e:
                print(f"⚠️  TensorBoard close failed: {e}")

    def __enter__(self):
        """Context manager entry."""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.close()
