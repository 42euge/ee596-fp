"""
Real-time Monitoring and Alerting

Provides live monitoring of reward signals with alert capabilities.
"""

import time
from typing import Dict, List, Optional, Callable, Any
from dataclasses import dataclass
from enum import Enum


class AlertLevel(Enum):
    """Alert severity levels."""
    INFO = "info"
    WARNING = "warning"
    ERROR = "error"
    CRITICAL = "critical"


@dataclass
class Alert:
    """Alert notification."""
    level: AlertLevel
    message: str
    step: int
    timestamp: float
    metric_name: str
    metric_value: Optional[float] = None
    threshold: Optional[float] = None


class RealtimeMonitor:
    """
    Real-time monitoring with alerting capabilities.

    Features:
    - Threshold-based alerts
    - Trend-based alerts
    - Custom alert conditions
    - Alert callbacks for custom actions
    """

    def __init__(
        self,
        alert_callback: Optional[Callable[[Alert], None]] = None,
        print_alerts: bool = True,
    ):
        """
        Initialize real-time monitor.

        Args:
            alert_callback: Optional callback function to handle alerts
            print_alerts: Whether to print alerts to console
        """
        self.alert_callback = alert_callback
        self.print_alerts = print_alerts

        # Alert history
        self.alerts: List[Alert] = []

        # Alert conditions
        self.threshold_conditions: Dict[str, Dict[str, Any]] = {}
        self.trend_conditions: Dict[str, Dict[str, Any]] = {}

        # Metrics buffer for trend detection
        self.metrics_buffer: Dict[str, List[float]] = {}
        self.buffer_size = 10

        # Alert cooldown (avoid spam)
        self.last_alert_time: Dict[str, float] = {}
        self.cooldown_seconds = 30

    def add_threshold_alert(
        self,
        metric_name: str,
        threshold: float,
        condition: str = "greater",  # "greater", "less", "equal"
        level: AlertLevel = AlertLevel.WARNING,
    ):
        """
        Add threshold-based alert condition.

        Args:
            metric_name: Name of metric to monitor
            threshold: Threshold value
            condition: Comparison condition
            level: Alert severity level
        """
        self.threshold_conditions[metric_name] = {
            'threshold': threshold,
            'condition': condition,
            'level': level,
        }

    def add_trend_alert(
        self,
        metric_name: str,
        min_slope: Optional[float] = None,
        max_slope: Optional[float] = None,
        window_size: int = 10,
        level: AlertLevel = AlertLevel.WARNING,
    ):
        """
        Add trend-based alert condition.

        Args:
            metric_name: Name of metric to monitor
            min_slope: Minimum acceptable slope (alert if below)
            max_slope: Maximum acceptable slope (alert if above)
            window_size: Number of steps to analyze for trend
            level: Alert severity level
        """
        self.trend_conditions[metric_name] = {
            'min_slope': min_slope,
            'max_slope': max_slope,
            'window_size': window_size,
            'level': level,
        }

    def check_metrics(
        self,
        step: int,
        metrics: Dict[str, float],
    ):
        """
        Check metrics against alert conditions.

        Args:
            step: Training step
            metrics: Dict of metric values
        """
        timestamp = time.time()

        # Check threshold conditions
        for metric_name, condition in self.threshold_conditions.items():
            if metric_name in metrics:
                value = metrics[metric_name]
                threshold = condition['threshold']
                comp = condition['condition']
                level = condition['level']

                triggered = False

                if comp == "greater" and value > threshold:
                    triggered = True
                elif comp == "less" and value < threshold:
                    triggered = True
                elif comp == "equal" and abs(value - threshold) < 1e-6:
                    triggered = True

                if triggered:
                    self._trigger_alert(
                        level=level,
                        message=f"{metric_name} ({value:.4f}) {comp} than threshold ({threshold:.4f})",
                        step=step,
                        timestamp=timestamp,
                        metric_name=metric_name,
                        metric_value=value,
                        threshold=threshold,
                    )

        # Update metrics buffer for trend detection
        for metric_name, value in metrics.items():
            if metric_name not in self.metrics_buffer:
                self.metrics_buffer[metric_name] = []

            self.metrics_buffer[metric_name].append(value)

            # Keep buffer size limited
            if len(self.metrics_buffer[metric_name]) > self.buffer_size:
                self.metrics_buffer[metric_name].pop(0)

        # Check trend conditions
        for metric_name, condition in self.trend_conditions.items():
            if metric_name in self.metrics_buffer:
                buffer = self.metrics_buffer[metric_name]
                window_size = condition['window_size']

                if len(buffer) >= window_size:
                    # Compute trend (simple linear fit)
                    import numpy as np
                    x = np.arange(len(buffer[-window_size:]))
                    y = np.array(buffer[-window_size:])

                    if len(x) > 1:
                        slope, _ = np.polyfit(x, y, 1)

                        min_slope = condition['min_slope']
                        max_slope = condition['max_slope']
                        level = condition['level']

                        if min_slope is not None and slope < min_slope:
                            self._trigger_alert(
                                level=level,
                                message=f"{metric_name} trend slope ({slope:.6f}) below minimum ({min_slope:.6f})",
                                step=step,
                                timestamp=timestamp,
                                metric_name=metric_name,
                                metric_value=slope,
                                threshold=min_slope,
                            )

                        if max_slope is not None and slope > max_slope:
                            self._trigger_alert(
                                level=level,
                                message=f"{metric_name} trend slope ({slope:.6f}) above maximum ({max_slope:.6f})",
                                step=step,
                                timestamp=timestamp,
                                metric_name=metric_name,
                                metric_value=slope,
                                threshold=max_slope,
                            )

    def _trigger_alert(
        self,
        level: AlertLevel,
        message: str,
        step: int,
        timestamp: float,
        metric_name: str,
        metric_value: Optional[float] = None,
        threshold: Optional[float] = None,
    ):
        """Trigger an alert."""
        # Check cooldown
        alert_key = f"{metric_name}_{level.value}"
        if alert_key in self.last_alert_time:
            if timestamp - self.last_alert_time[alert_key] < self.cooldown_seconds:
                return  # Skip due to cooldown

        self.last_alert_time[alert_key] = timestamp

        # Create alert
        alert = Alert(
            level=level,
            message=message,
            step=step,
            timestamp=timestamp,
            metric_name=metric_name,
            metric_value=metric_value,
            threshold=threshold,
        )

        self.alerts.append(alert)

        # Print if enabled
        if self.print_alerts:
            icon = {
                AlertLevel.INFO: "ℹ️",
                AlertLevel.WARNING: "⚠️",
                AlertLevel.ERROR: "❌",
                AlertLevel.CRITICAL: "🚨",
            }.get(level, "")

            print(f"{icon} [{level.value.upper()}] Step {step}: {message}")

        # Call custom callback
        if self.alert_callback:
            try:
                self.alert_callback(alert)
            except Exception as e:
                print(f"Alert callback failed: {e}")

    def get_recent_alerts(
        self,
        num_alerts: int = 10,
        level: Optional[AlertLevel] = None,
    ) -> List[Alert]:
        """
        Get recent alerts.

        Args:
            num_alerts: Number of recent alerts to return
            level: Optional filter by alert level

        Returns:
            List of recent alerts
        """
        alerts = self.alerts

        if level:
            alerts = [a for a in alerts if a.level == level]

        return alerts[-num_alerts:]

    def get_alert_summary(self) -> Dict[str, Any]:
        """Get summary of alerts."""
        summary = {
            'total_alerts': len(self.alerts),
            'by_level': {
                level.value: len([a for a in self.alerts if a.level == level])
                for level in AlertLevel
            },
            'recent_alerts': [
                {
                    'level': a.level.value,
                    'message': a.message,
                    'step': a.step,
                }
                for a in self.alerts[-5:]
            ],
        }

        return summary

    def setup_default_alerts(self):
        """Setup default alert conditions for GRPO training."""
        # Alert if mean reward drops significantly
        self.add_trend_alert(
            'mean_reward',
            min_slope=-0.01,  # Alert if reward declining
            window_size=10,
            level=AlertLevel.WARNING,
        )

        # Alert if signal-to-noise ratio is too low
        self.add_threshold_alert(
            'signal_to_noise',
            threshold=0.5,
            condition='less',
            level=AlertLevel.WARNING,
        )

        # Alert if KL divergence gets too high
        self.add_threshold_alert(
            'kl_divergence',
            threshold=0.5,
            condition='greater',
            level=AlertLevel.ERROR,
        )

        # Alert if reward variance explodes
        self.add_threshold_alert(
            'reward_variance',
            threshold=100.0,
            condition='greater',
            level=AlertLevel.CRITICAL,
        )

        print("✓ Default alerts configured")
