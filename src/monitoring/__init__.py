"""
Reward Monitoring and Dashboard System

This package provides comprehensive monitoring tools for tracking reward signal
quality across GRPO training runs.

Components:
- RewardMetricsTracker: Core metrics tracking and aggregation
- DashboardManager: Multi-backend dashboard management (W&B, TensorBoard, Web)
- RewardAnalyzer: Statistical analysis and quality metrics
- RealtimeMonitor: Live monitoring and alerting
"""

from .metrics_tracker import RewardMetricsTracker
from .dashboard_manager import DashboardManager
from .reward_analyzer import RewardAnalyzer
from .realtime_monitor import RealtimeMonitor

__all__ = [
    'RewardMetricsTracker',
    'DashboardManager',
    'RewardAnalyzer',
    'RealtimeMonitor',
]
