"""
Reward Quality Monitoring and Intervention System

Provides real-time monitoring, logging, and automatic intervention for reward quality issues.
Integrates with Weights & Biases for dashboard visualization and alerting.
"""

import json
import warnings
from typing import Dict, List, Any, Optional, Callable
from dataclasses import dataclass, asdict
from pathlib import Path
import numpy as np

from .reward_quality import (
    RewardQualityAssessor,
    QualityMetrics,
    PathologyAlert,
)


@dataclass
class InterventionConfig:
    """Configuration for automatic interventions when pathologies detected."""

    # Enable/disable interventions
    enable_interventions: bool = True

    # Severity thresholds for interventions
    intervention_severity: str = 'high'  # 'low', 'medium', 'high', 'critical'

    # Intervention actions
    log_to_console: bool = True
    log_to_file: bool = True
    send_to_wandb: bool = True

    # File paths
    alert_log_file: str = '/tmp/reward_quality_alerts.jsonl'
    metrics_log_file: str = '/tmp/reward_quality_metrics.jsonl'

    # Alert aggregation
    alert_aggregation_window: int = 100  # Steps
    max_alerts_per_window: int = 10  # Prevent spam

    # Callbacks
    custom_alert_callback: Optional[Callable[[PathologyAlert], None]] = None


class RewardQualityMonitor:
    """
    Monitoring system for reward quality with automatic interventions.

    Integrates with training loops to provide real-time quality assessment,
    logging, and alerting capabilities.
    """

    def __init__(
        self,
        assessor: RewardQualityAssessor,
        config: Optional[InterventionConfig] = None,
        wandb_run: Optional[Any] = None
    ):
        """
        Initialize the monitor.

        Args:
            assessor: RewardQualityAssessor instance
            config: Configuration for interventions
            wandb_run: Weights & Biases run object (optional)
        """
        self.assessor = assessor
        self.config = config or InterventionConfig()
        self.wandb_run = wandb_run

        # Alert tracking
        self.alert_count_by_severity = {'low': 0, 'medium': 0, 'high': 0, 'critical': 0}
        self.alert_count_by_type = {}
        self.recent_alert_steps = []

        # Setup logging
        self._setup_logging()

        # Severity ranking for intervention decisions
        self.severity_rank = {'low': 0, 'medium': 1, 'high': 2, 'critical': 3}

    def _setup_logging(self):
        """Setup log files and directories."""
        if self.config.log_to_file:
            Path(self.config.alert_log_file).parent.mkdir(parents=True, exist_ok=True)
            Path(self.config.metrics_log_file).parent.mkdir(parents=True, exist_ok=True)

    def monitor_batch(
        self,
        responses: List[str],
        rewards: Dict[str, List[float]],
        step: int,
        references: Optional[List[str]] = None,
        extra_metrics: Optional[Dict[str, Any]] = None
    ) -> QualityMetrics:
        """
        Monitor a batch of responses and rewards.

        Args:
            responses: List of model responses
            rewards: Dictionary of reward components and values
            step: Current training step
            references: Optional reference responses
            extra_metrics: Additional metrics to log

        Returns:
            QualityMetrics for the batch
        """
        # Assess quality
        metrics, alerts = self.assessor.assess_batch(responses, rewards, references)

        # Log metrics
        self._log_metrics(metrics, step, extra_metrics)

        # Handle alerts if any detected
        if alerts:
            self._handle_alerts(alerts, step)

        # Log to W&B
        if self.wandb_run and self.config.send_to_wandb:
            self._log_to_wandb(metrics, alerts, step, extra_metrics)

        return metrics

    def _log_metrics(
        self,
        metrics: QualityMetrics,
        step: int,
        extra_metrics: Optional[Dict[str, Any]] = None
    ):
        """Log quality metrics to file and console."""
        metrics_dict = metrics.to_dict()
        metrics_dict['step'] = step

        if extra_metrics:
            metrics_dict.update(extra_metrics)

        # Log to file
        if self.config.log_to_file:
            with open(self.config.metrics_log_file, 'a') as f:
                f.write(json.dumps(metrics_dict) + '\n')

    def _handle_alerts(self, alerts: List[PathologyAlert], step: int):
        """Handle detected pathology alerts."""
        # Check if we should throttle alerts
        self.recent_alert_steps.append(step)
        # Keep only recent window
        cutoff = step - self.config.alert_aggregation_window
        self.recent_alert_steps = [s for s in self.recent_alert_steps if s > cutoff]

        # Throttle if too many alerts
        if len(self.recent_alert_steps) > self.config.max_alerts_per_window:
            if self.config.log_to_console:
                print(f"[WARNING] Alert rate limit reached ({len(self.recent_alert_steps)} alerts in last {self.config.alert_aggregation_window} steps)")
            return

        # Process each alert
        for alert in alerts:
            # Update counts
            self.alert_count_by_severity[alert.severity] += 1
            self.alert_count_by_type[alert.pathology_type] = \
                self.alert_count_by_type.get(alert.pathology_type, 0) + 1

            # Check if intervention needed
            if self._should_intervene(alert):
                self._execute_intervention(alert, step)

    def _should_intervene(self, alert: PathologyAlert) -> bool:
        """Determine if an intervention should be triggered."""
        if not self.config.enable_interventions:
            return False

        alert_severity_rank = self.severity_rank.get(alert.severity, 0)
        threshold_rank = self.severity_rank.get(self.config.intervention_severity, 2)

        return alert_severity_rank >= threshold_rank

    def _execute_intervention(self, alert: PathologyAlert, step: int):
        """Execute intervention actions for an alert."""
        # Console logging
        if self.config.log_to_console:
            self._print_alert(alert, step)

        # File logging
        if self.config.log_to_file:
            self._log_alert_to_file(alert, step)

        # Custom callback
        if self.config.custom_alert_callback:
            try:
                self.config.custom_alert_callback(alert)
            except Exception as e:
                warnings.warn(f"Custom alert callback failed: {e}")

    def _print_alert(self, alert: PathologyAlert, step: int):
        """Print alert to console with formatting."""
        severity_colors = {
            'low': '\033[36m',      # Cyan
            'medium': '\033[33m',   # Yellow
            'high': '\033[91m',     # Red
            'critical': '\033[95m'  # Magenta
        }
        reset = '\033[0m'

        color = severity_colors.get(alert.severity, '')
        print(f"\n{color}{'='*80}")
        print(f"REWARD QUALITY ALERT [Step {step}]")
        print(f"Severity: {alert.severity.upper()}")
        print(f"Type: {alert.pathology_type}")
        print(f"Message: {alert.message}")
        if alert.metrics:
            print(f"Metrics: {json.dumps(alert.metrics, indent=2)}")
        print(f"{'='*80}{reset}\n")

    def _log_alert_to_file(self, alert: PathologyAlert, step: int):
        """Log alert to JSONL file."""
        alert_dict = {
            'step': step,
            'timestamp': alert.timestamp,
            'severity': alert.severity,
            'pathology_type': alert.pathology_type,
            'message': alert.message,
            'metrics': alert.metrics
        }

        with open(self.config.alert_log_file, 'a') as f:
            f.write(json.dumps(alert_dict) + '\n')

    def _log_to_wandb(
        self,
        metrics: QualityMetrics,
        alerts: List[PathologyAlert],
        step: int,
        extra_metrics: Optional[Dict[str, Any]] = None
    ):
        """Log metrics and alerts to Weights & Biases."""
        if not self.wandb_run:
            return

        # Prepare metrics dict
        log_dict = {}

        # Quality metrics with prefix
        metrics_dict = metrics.to_dict()
        for key, value in metrics_dict.items():
            log_dict[f'reward_quality/{key}'] = value

        # Alert counts
        for severity, count in self.alert_count_by_severity.items():
            log_dict[f'reward_quality/alerts_{severity}'] = count

        # Alert types
        for pathology_type, count in self.alert_count_by_type.items():
            log_dict[f'reward_quality/pathology_{pathology_type}'] = count

        # Current batch alerts
        log_dict['reward_quality/alerts_this_batch'] = len(alerts)

        # Extra metrics
        if extra_metrics:
            for key, value in extra_metrics.items():
                log_dict[f'reward_quality/{key}'] = value

        # Log alert messages as text (for recent alerts)
        if alerts:
            alert_messages = '\n'.join([f"[{a.severity.upper()}] {a.message}" for a in alerts])
            log_dict['reward_quality/recent_alerts'] = self.wandb_run.Html(
                f"<pre>{alert_messages}</pre>"
            )

        # Send to W&B
        try:
            self.wandb_run.log(log_dict, step=step)
        except Exception as e:
            warnings.warn(f"Failed to log to W&B: {e}")

    def get_monitoring_summary(self) -> Dict[str, Any]:
        """Get a summary of monitoring statistics."""
        summary = self.assessor.get_summary_statistics()

        # Add monitoring-specific stats
        summary['monitoring'] = {
            'total_alerts_by_severity': dict(self.alert_count_by_severity),
            'total_alerts_by_type': dict(self.alert_count_by_type),
            'recent_alert_rate': len(self.recent_alert_steps) / max(1, self.config.alert_aggregation_window)
        }

        return summary

    def export_metrics_history(self, output_path: str):
        """Export all logged metrics to a file."""
        import shutil
        if Path(self.config.metrics_log_file).exists():
            shutil.copy(self.config.metrics_log_file, output_path)
            print(f"Metrics history exported to {output_path}")
        else:
            warnings.warn("No metrics history file found")

    def export_alert_history(self, output_path: str):
        """Export all logged alerts to a file."""
        import shutil
        if Path(self.config.alert_log_file).exists():
            shutil.copy(self.config.alert_log_file, output_path)
            print(f"Alert history exported to {output_path}")
        else:
            warnings.warn("No alert history file found")


def create_default_monitor(
    wandb_run: Optional[Any] = None,
    window_size: int = 1000,
    enable_interventions: bool = True,
    alert_log_file: Optional[str] = None
) -> RewardQualityMonitor:
    """
    Create a monitor with sensible defaults.

    Args:
        wandb_run: Weights & Biases run object
        window_size: Size of rolling window for statistics
        enable_interventions: Whether to enable automatic interventions
        alert_log_file: Custom path for alert log file

    Returns:
        Configured RewardQualityMonitor
    """
    assessor = RewardQualityAssessor(
        window_size=window_size,
        min_samples_for_analysis=100
    )

    config = InterventionConfig(
        enable_interventions=enable_interventions,
        intervention_severity='medium',  # Alert on medium+ severity
        log_to_console=True,
        log_to_file=True,
        send_to_wandb=wandb_run is not None
    )

    if alert_log_file:
        config.alert_log_file = alert_log_file

    return RewardQualityMonitor(assessor, config, wandb_run)


class RewardQualityDashboard:
    """
    Utility for creating visualizations and reports of reward quality.

    Useful for post-training analysis and debugging.
    """

    def __init__(self, metrics_log_file: str, alert_log_file: str):
        """
        Initialize dashboard from log files.

        Args:
            metrics_log_file: Path to metrics JSONL file
            alert_log_file: Path to alerts JSONL file
        """
        self.metrics_log_file = metrics_log_file
        self.alert_log_file = alert_log_file

        self.metrics_history = self._load_metrics()
        self.alert_history = self._load_alerts()

    def _load_metrics(self) -> List[Dict[str, Any]]:
        """Load metrics from JSONL file."""
        metrics = []
        if Path(self.metrics_log_file).exists():
            with open(self.metrics_log_file, 'r') as f:
                for line in f:
                    metrics.append(json.loads(line))
        return metrics

    def _load_alerts(self) -> List[Dict[str, Any]]:
        """Load alerts from JSONL file."""
        alerts = []
        if Path(self.alert_log_file).exists():
            with open(self.alert_log_file, 'r') as f:
                for line in f:
                    alerts.append(json.loads(line))
        return alerts

    def generate_report(self, output_path: str):
        """
        Generate a comprehensive HTML report of reward quality.

        Args:
            output_path: Path to save HTML report
        """
        html = self._create_html_report()
        with open(output_path, 'w') as f:
            f.write(html)
        print(f"Report generated at {output_path}")

    def _create_html_report(self) -> str:
        """Create HTML report content."""
        # Calculate summary statistics
        total_steps = len(self.metrics_history)
        total_alerts = len(self.alert_history)

        alert_by_severity = {}
        alert_by_type = {}
        for alert in self.alert_history:
            severity = alert.get('severity', 'unknown')
            pathology_type = alert.get('pathology_type', 'unknown')
            alert_by_severity[severity] = alert_by_severity.get(severity, 0) + 1
            alert_by_type[pathology_type] = alert_by_type.get(pathology_type, 0) + 1

        # Create HTML
        html = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <title>Reward Quality Assessment Report</title>
            <style>
                body {{ font-family: Arial, sans-serif; margin: 20px; }}
                h1 {{ color: #333; }}
                h2 {{ color: #666; margin-top: 30px; }}
                table {{ border-collapse: collapse; width: 100%; margin-top: 10px; }}
                th, td {{ border: 1px solid #ddd; padding: 8px; text-align: left; }}
                th {{ background-color: #4CAF50; color: white; }}
                .alert-critical {{ background-color: #ffcccc; }}
                .alert-high {{ background-color: #ffddaa; }}
                .alert-medium {{ background-color: #ffffcc; }}
                .alert-low {{ background-color: #ddffdd; }}
            </style>
        </head>
        <body>
            <h1>Reward Quality Assessment Report</h1>

            <h2>Summary</h2>
            <p><strong>Total Steps Monitored:</strong> {total_steps}</p>
            <p><strong>Total Alerts:</strong> {total_alerts}</p>

            <h2>Alerts by Severity</h2>
            <table>
                <tr><th>Severity</th><th>Count</th></tr>
                {''.join([f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in sorted(alert_by_severity.items())])}
            </table>

            <h2>Alerts by Pathology Type</h2>
            <table>
                <tr><th>Pathology Type</th><th>Count</th></tr>
                {''.join([f'<tr><td>{k}</td><td>{v}</td></tr>' for k, v in sorted(alert_by_type.items(), key=lambda x: x[1], reverse=True)])}
            </table>

            <h2>Recent Alerts (Last 50)</h2>
            <table>
                <tr>
                    <th>Step</th>
                    <th>Severity</th>
                    <th>Type</th>
                    <th>Message</th>
                </tr>
                {''.join([
                    f'<tr class="alert-{alert.get("severity", "unknown")}">'
                    f'<td>{alert.get("step", "N/A")}</td>'
                    f'<td>{alert.get("severity", "N/A")}</td>'
                    f'<td>{alert.get("pathology_type", "N/A")}</td>'
                    f'<td>{alert.get("message", "N/A")}</td>'
                    f'</tr>'
                    for alert in self.alert_history[-50:]
                ])}
            </table>
        </body>
        </html>
        """
        return html

    def get_metrics_dataframe(self):
        """
        Get metrics as a pandas DataFrame (if pandas available).

        Returns:
            DataFrame with metrics history
        """
        try:
            import pandas as pd
            return pd.DataFrame(self.metrics_history)
        except ImportError:
            warnings.warn("pandas not available, returning raw metrics")
            return self.metrics_history
