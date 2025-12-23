#!/usr/bin/env python3
"""
Test that monitoring modules can be imported.

This is a basic sanity check that doesn't require external dependencies.
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

def test_imports():
    """Test that all monitoring modules can be imported."""
    print("Testing monitoring module imports...")

    try:
        from src.monitoring import (
            RewardMetricsTracker,
            DashboardManager,
            RewardAnalyzer,
            RealtimeMonitor,
        )
        print("✓ All monitoring modules imported successfully")
        return True

    except ImportError as e:
        print(f"✗ Import failed: {e}")
        return False


def test_basic_functionality():
    """Test basic functionality without external dependencies."""
    print("\nTesting basic functionality...")

    try:
        # Test that we can create instances
        from src.monitoring.metrics_tracker import RewardMetricsTracker
        from src.monitoring.reward_analyzer import RewardAnalyzer
        from src.monitoring.realtime_monitor import RealtimeMonitor, AlertLevel

        # Create tracker (should work without numpy if we don't use it)
        tracker = RewardMetricsTracker(
            reward_function_names=['test_func1', 'test_func2'],
            window_size=10,
            track_distributions=False
        )
        print("✓ RewardMetricsTracker initialized")

        # Create analyzer
        analyzer = RewardAnalyzer()
        print("✓ RewardAnalyzer initialized")

        # Create monitor
        monitor = RealtimeMonitor(print_alerts=False)
        monitor.add_threshold_alert(
            metric_name='test_metric',
            threshold=1.0,
            condition='greater',
            level=AlertLevel.WARNING
        )
        print("✓ RealtimeMonitor initialized")

        return True

    except Exception as e:
        print(f"✗ Functionality test failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == "__main__":
    print("="*60)
    print("Monitoring System Import Tests")
    print("="*60)
    print()

    success = True

    success = test_imports() and success
    success = test_basic_functionality() and success

    print()
    print("="*60)
    if success:
        print("✓ All tests passed!")
    else:
        print("✗ Some tests failed")
        print("\nNote: Full functionality requires dependencies:")
        print("  pip install -r requirements-monitoring.txt")
    print("="*60)

    sys.exit(0 if success else 1)
