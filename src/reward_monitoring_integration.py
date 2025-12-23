"""
Integration module for reward monitoring in GRPO training.

This module provides wrapper functions and utilities to integrate
the reward hack detection system into the GRPO training loop.
"""

import logging
from typing import List, Callable, Dict, Any, Optional
from dataclasses import dataclass
import numpy as np

from .reward_monitoring import RewardHackDetector, DetectionConfig, DetectionResult

logger = logging.getLogger(__name__)


@dataclass
class MonitoredRewardResult:
    """Result from a monitored reward computation."""
    reward: float
    component_rewards: Dict[str, float]
    response: str
    detections: List[DetectionResult]


class RewardFunctionMonitor:
    """
    Wrapper for reward functions that tracks metrics and detects anomalies.

    This class wraps the standard GRPO reward functions and integrates
    with the RewardHackDetector to monitor for problematic behaviors.
    """

    def __init__(
        self,
        reward_fns: List[Callable],
        reward_fn_names: List[str],
        detection_config: Optional[DetectionConfig] = None,
        wandb_enabled: bool = True,
    ):
        """
        Initialize the reward function monitor.

        Args:
            reward_fns: List of reward functions to wrap
            reward_fn_names: Names corresponding to each reward function
            detection_config: Configuration for detection thresholds
            wandb_enabled: Whether to log to W&B
        """
        self.reward_fns = reward_fns
        self.reward_fn_names = reward_fn_names
        self.detector = RewardHackDetector(detection_config)
        self.wandb_enabled = wandb_enabled

        self.step_count = 0
        self.total_rewards = []
        self.component_rewards_history = {name: [] for name in reward_fn_names}

        # Track detection statistics
        self.detection_counts = {
            'total': 0,
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0,
        }

    def compute_rewards(
        self,
        responses: List[str],
        questions: List[str],
        answers: List[str],
        step: int,
        kl_divergence: Optional[float] = None,
        gradient_norm: Optional[float] = None,
        loss: Optional[float] = None,
    ) -> List[MonitoredRewardResult]:
        """
        Compute rewards for a batch of responses with monitoring.

        Args:
            responses: Generated responses
            questions: Input questions
            answers: Ground truth answers
            step: Current training step
            kl_divergence: KL divergence from reference policy
            gradient_norm: Gradient norm
            loss: Training loss

        Returns:
            List of MonitoredRewardResult objects
        """
        self.step_count = step
        results = []

        for response, question, answer in zip(responses, questions, answers):
            # Compute individual reward components
            component_rewards = {}
            total_reward = 0.0

            for reward_fn, name in zip(self.reward_fns, self.reward_fn_names):
                try:
                    reward = reward_fn(response, question, answer)
                    component_rewards[name] = float(reward)
                    total_reward += float(reward)
                except Exception as e:
                    logger.error(f"Error computing {name} reward: {e}")
                    component_rewards[name] = 0.0

            # Analyze for anomalies
            detections = self.detector.analyze_step(
                response=response,
                total_reward=total_reward,
                reward_components=component_rewards,
                kl_divergence=kl_divergence,
                gradient_norm=gradient_norm,
                loss=loss,
            )

            # Update detection statistics
            for detection in detections:
                self.detection_counts['total'] += 1
                self.detection_counts[detection.severity] += 1

            # Log detections
            if detections:
                self.detector.log_detections(step, detections)

            # Store history
            self.total_rewards.append(total_reward)
            for name, value in component_rewards.items():
                self.component_rewards_history[name].append(value)

            results.append(MonitoredRewardResult(
                reward=total_reward,
                component_rewards=component_rewards,
                response=response,
                detections=detections,
            ))

        return results

    def get_metrics_for_logging(self) -> Dict[str, Any]:
        """
        Get metrics to log to W&B or TensorBoard.

        Returns:
            Dictionary of metrics
        """
        metrics = {}

        # Basic reward statistics
        if self.total_rewards:
            recent_rewards = self.total_rewards[-20:] if len(self.total_rewards) >= 20 else self.total_rewards
            metrics['reward/mean'] = float(np.mean(self.total_rewards))
            metrics['reward/std'] = float(np.std(self.total_rewards))
            metrics['reward/recent_mean'] = float(np.mean(recent_rewards))
            metrics['reward/recent_std'] = float(np.std(recent_rewards))
            metrics['reward/min'] = float(np.min(self.total_rewards))
            metrics['reward/max'] = float(np.max(self.total_rewards))

        # Component reward statistics
        for name, values in self.component_rewards_history.items():
            if values:
                recent_values = values[-20:] if len(values) >= 20 else values
                metrics[f'reward_component/{name}/mean'] = float(np.mean(values))
                metrics[f'reward_component/{name}/recent_mean'] = float(np.mean(recent_values))

        # Detection statistics
        metrics['detections/total'] = self.detection_counts['total']
        metrics['detections/critical'] = self.detection_counts['critical']
        metrics['detections/high'] = self.detection_counts['high']
        metrics['detections/medium'] = self.detection_counts['medium']
        metrics['detections/low'] = self.detection_counts['low']

        # Detection rate (per 100 steps)
        if self.step_count > 0:
            metrics['detections/rate'] = (self.detection_counts['total'] / max(1, self.step_count)) * 100

        # Get detector summary metrics
        detector_metrics = self.detector.get_summary_metrics()
        for key, value in detector_metrics.items():
            if key not in ['detections_by_type', 'detections_by_severity']:
                metrics[f'detector/{key}'] = value

        # Add detection type breakdown
        for det_type, count in detector_metrics.get('detections_by_type', {}).items():
            metrics[f'detection_type/{det_type}'] = count

        return metrics

    def get_summary_report(self) -> str:
        """
        Generate a human-readable summary report.

        Returns:
            String containing the summary report
        """
        lines = []
        lines.append("=" * 70)
        lines.append("REWARD MONITORING SUMMARY")
        lines.append("=" * 70)

        lines.append(f"\nTotal Steps: {self.step_count}")
        lines.append(f"Total Detections: {self.detection_counts['total']}")

        if self.detection_counts['total'] > 0:
            lines.append("\nDetections by Severity:")
            lines.append(f"  Critical: {self.detection_counts['critical']}")
            lines.append(f"  High:     {self.detection_counts['high']}")
            lines.append(f"  Medium:   {self.detection_counts['medium']}")
            lines.append(f"  Low:      {self.detection_counts['low']}")

            detection_rate = (self.detection_counts['total'] / max(1, self.step_count)) * 100
            lines.append(f"\nDetection Rate: {detection_rate:.2f} per 100 steps")

        if self.total_rewards:
            lines.append("\nReward Statistics:")
            lines.append(f"  Mean: {np.mean(self.total_rewards):.4f}")
            lines.append(f"  Std:  {np.std(self.total_rewards):.4f}")
            lines.append(f"  Min:  {np.min(self.total_rewards):.4f}")
            lines.append(f"  Max:  {np.max(self.total_rewards):.4f}")

        # Component breakdown
        lines.append("\nReward Components:")
        for name, values in self.component_rewards_history.items():
            if values:
                lines.append(f"  {name}: mean={np.mean(values):.4f}, std={np.std(values):.4f}")

        # Detection type breakdown
        detector_metrics = self.detector.get_summary_metrics()
        detections_by_type = detector_metrics.get('detections_by_type', {})
        if detections_by_type:
            lines.append("\nDetections by Type:")
            for det_type, count in sorted(detections_by_type.items(), key=lambda x: x[1], reverse=True):
                lines.append(f"  {det_type}: {count}")

        lines.append("=" * 70)
        return "\n".join(lines)

    def should_alert(self) -> bool:
        """
        Determine if training should be alerted or stopped.

        Returns:
            True if critical issues detected
        """
        # Alert if we have critical detections
        if self.detection_counts['critical'] > 0:
            return True

        # Alert if high severity detections exceed threshold
        if self.step_count > 0:
            high_severity_rate = (self.detection_counts['high'] / max(1, self.step_count)) * 100
            if high_severity_rate > 50:  # More than 50% of steps have high severity issues
                return True

        return False


def create_monitored_reward_wrapper(
    reward_fn: Callable,
    reward_fn_name: str,
    monitor: RewardFunctionMonitor,
) -> Callable:
    """
    Create a wrapper around a single reward function that integrates with monitoring.

    Args:
        reward_fn: The original reward function
        reward_fn_name: Name of the reward function
        monitor: The RewardFunctionMonitor instance

    Returns:
        Wrapped reward function
    """
    def wrapped_reward_fn(response: str, question: str, answer: str) -> float:
        """Wrapped reward function with monitoring."""
        try:
            reward = reward_fn(response, question, answer)
            return float(reward)
        except Exception as e:
            logger.error(f"Error in {reward_fn_name}: {e}")
            return 0.0

    return wrapped_reward_fn


def log_detections_to_wandb(detections: List[DetectionResult], step: int):
    """
    Log detection results to Weights & Biases.

    Args:
        detections: List of detection results
        step: Current training step
    """
    try:
        import wandb

        if not wandb.run:
            return

        # Log each detection
        for detection in detections:
            wandb.log({
                f'detection/{detection.detection_type}': 1,
                f'detection_severity/{detection.severity}': 1,
                **{f'detection_metric/{k}': v for k, v in detection.metrics.items()}
            }, step=step)

            # Log as alert if critical or high severity
            if detection.severity in ['critical', 'high']:
                wandb.alert(
                    title=f"{detection.severity.upper()}: {detection.detection_type}",
                    text=detection.message,
                    level=wandb.AlertLevel.WARN if detection.severity == 'high' else wandb.AlertLevel.ERROR
                )

    except ImportError:
        pass  # W&B not available
    except Exception as e:
        logger.warning(f"Failed to log to W&B: {e}")


def create_monitoring_callback(monitor: RewardFunctionMonitor, log_interval: int = 10):
    """
    Create a callback function for periodic monitoring reports.

    Args:
        monitor: The RewardFunctionMonitor instance
        log_interval: How often to log summary metrics (in steps)

    Returns:
        Callback function
    """
    def callback(step: int):
        """Callback to log monitoring metrics."""
        if step % log_interval == 0:
            # Get and log metrics
            metrics = monitor.get_metrics_for_logging()

            # Log to W&B if enabled
            if monitor.wandb_enabled:
                try:
                    import wandb
                    if wandb.run:
                        wandb.log(metrics, step=step)
                except ImportError:
                    pass

            # Print summary every 50 steps
            if step % 50 == 0 and step > 0:
                logger.info(f"\n{monitor.get_summary_report()}")

            # Check for alerts
            if monitor.should_alert():
                logger.critical(f"ALERT at step {step}: Critical issues detected during training!")

    return callback
