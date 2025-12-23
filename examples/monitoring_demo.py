#!/usr/bin/env python3
"""
Monitoring System Demo

Demonstrates the reward monitoring system with simulated training data.
This is useful for testing the monitoring infrastructure without running
a full training run.

Usage:
    python examples/monitoring_demo.py
"""

import sys
import os
import numpy as np
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.monitoring import (
    RewardMetricsTracker,
    DashboardManager,
    RewardAnalyzer,
    RealtimeMonitor,
    AlertLevel,
)


def simulate_training_step(step: int, num_samples: int = 4):
    """
    Simulate a training step with reward signals.

    Returns:
        Dict mapping function names to reward arrays
    """
    # Simulate improving performance over time
    progress = step / 100.0

    # Format exact: improves from 0% to 80% compliance
    format_exact = np.random.choice(
        [0.0, 3.0],
        size=num_samples,
        p=[max(0.2, 1.0 - progress * 0.8), min(0.8, progress * 0.8)]
    )

    # Format approximate: gradual improvement
    format_approx = np.random.normal(
        loc=progress * 2.0,  # Mean increases
        scale=0.5,  # Constant variance
        size=num_samples
    )
    format_approx = np.clip(format_approx, -2.5, 2.5)

    # Check answer: accuracy improves
    answer_choices = [-1.0, 0.25, 0.5, 1.5, 3.0]
    answer_probs = [
        max(0.1, 0.5 - progress * 0.4),  # Wrong: decreases
        0.1,  # 20% ratio
        0.1,  # 10% ratio
        0.2,  # Normalized
        min(0.5, progress * 0.5),  # Exact: increases
    ]
    answer_probs = np.array(answer_probs) / sum(answer_probs)

    check_answer_rewards = np.random.choice(
        answer_choices,
        size=num_samples,
        p=answer_probs
    )

    # Check numbers: binary, improves
    check_numbers = np.random.choice(
        [0.0, 1.5],
        size=num_samples,
        p=[max(0.2, 1.0 - progress * 0.7), min(0.8, progress * 0.7)]
    )

    # Simulate KL divergence (should stay bounded)
    kl_divergence = 0.05 + 0.02 * np.sin(step / 10.0) + np.random.normal(0, 0.01)
    kl_divergence = max(0, kl_divergence)

    # Policy entropy (should decrease over time)
    policy_entropy = 3.0 - progress * 0.5 + np.random.normal(0, 0.1)

    # Advantages
    advantages = np.random.normal(loc=0, scale=1.0, size=num_samples)

    return {
        'rewards': {
            'format_exact': format_exact,
            'format_approx': format_approx,
            'check_answer': check_answer_rewards,
            'check_numbers': check_numbers,
        },
        'kl_divergence': kl_divergence,
        'policy_entropy': policy_entropy,
        'advantages': advantages,
    }


def main():
    print("="*60)
    print("Reward Monitoring System Demo")
    print("="*60)
    print()

    # Configuration
    num_steps = 100
    batch_size = 4
    num_generations = 1
    monitoring_dir = "./demo_monitoring_data"

    # Initialize monitoring components
    print("[1/5] Initializing monitoring system...")

    reward_function_names = [
        'format_exact',
        'format_approx',
        'check_answer',
        'check_numbers',
    ]

    # Create metrics tracker
    tracker = RewardMetricsTracker(
        reward_function_names=reward_function_names,
        window_size=50,
        track_distributions=True,
    )

    # Create dashboard manager (with W&B disabled for demo)
    dashboard = DashboardManager(
        project_name='monitoring-demo',
        run_name='demo-run',
        enable_wandb=False,  # Disable for demo
        enable_tensorboard=True,
        json_export_dir=monitoring_dir,
    )

    dashboard.define_wandb_charts(reward_function_names)

    # Create analyzer
    analyzer = RewardAnalyzer()

    # Create monitor with alerts
    monitor = RealtimeMonitor(print_alerts=True)
    monitor.setup_default_alerts()

    # Custom alert
    monitor.add_threshold_alert(
        'mean_reward',
        threshold=5.0,
        condition='greater',
        level=AlertLevel.INFO,
    )

    print("  ✓ Monitoring system initialized")
    print(f"  ✓ Tracking {len(reward_function_names)} reward functions")
    print(f"  ✓ Export directory: {monitoring_dir}")
    print()

    # Simulate training
    print("[2/5] Simulating training...")
    print()

    reward_history = []

    for step in range(num_steps):
        # Simulate step
        step_data = simulate_training_step(step, num_samples=batch_size)

        # Track metrics
        metrics = tracker.track_step(
            step=step,
            rewards_by_function=step_data['rewards'],
            kl_divergence=step_data['kl_divergence'],
            policy_entropy=step_data['policy_entropy'],
            advantages=step_data['advantages'],
            batch_size=batch_size,
            num_generations=num_generations,
        )

        # Log to dashboards
        dashboard.log_metrics(step, metrics)

        # Log distributions every 10 steps
        if step % 10 == 0:
            dashboard.log_reward_distributions(step, metrics)

        # Check alerts
        metrics_dict = {
            'mean_reward': metrics.mean_reward,
            'signal_to_noise': metrics.signal_to_noise,
            'kl_divergence': metrics.kl_divergence,
            'reward_variance': metrics.reward_variance,
        }
        monitor.check_metrics(step, metrics_dict)

        # Store for analysis
        reward_history.append(step_data['rewards'])

        # Progress update
        if (step + 1) % 20 == 0:
            print(f"  Step {step + 1}/{num_steps}: "
                  f"mean_reward={metrics.mean_reward:.3f}, "
                  f"SNR={metrics.signal_to_noise:.3f}")

    print()
    print("  ✓ Training simulation complete")
    print()

    # Perform analysis
    print("[3/5] Performing quality analysis...")

    steps = list(range(num_steps))
    quality_report = analyzer.analyze_quality(reward_history, steps)

    print(f"\n  📊 Quality Score: {quality_report.quality_score:.1f}/100")
    print(f"     - Consistency: {quality_report.consistency_score:.1f}/100")
    print(f"     - Discriminability: {quality_report.discriminability_score:.1f}/100")
    print(f"     - Stability: {quality_report.stability_score:.1f}/100")
    print(f"     - Coverage: {quality_report.coverage_score:.1f}/100")

    if quality_report.issues:
        print("\n  ❌ Issues:")
        for issue in quality_report.issues:
            print(f"     - {issue}")

    if quality_report.warnings:
        print("\n  ⚠️  Warnings:")
        for warning in quality_report.warnings:
            print(f"     - {warning}")

    if quality_report.recommendations:
        print("\n  💡 Recommendations:")
        for rec in quality_report.recommendations:
            print(f"     - {rec}")

    print()

    # Function importance
    print("[4/5] Analyzing function importance...")

    importance = analyzer.analyze_function_importance(reward_history, reward_function_names)

    print("\n  📈 Reward Function Contributions:")
    for func_name in reward_function_names:
        if func_name in importance:
            metrics = importance[func_name]
            print(f"     {func_name:20s}: "
                  f"{metrics['contribution']:5.1f}% "
                  f"(mean={metrics['mean']:5.2f}, "
                  f"impact={metrics['impact_score']:5.1f})")

    print()

    # Summary and trends
    print("[5/5] Generating summary...")

    summary = tracker.get_summary()
    trends = tracker.get_recent_trends(num_steps=50)

    print(f"\n  📊 Training Summary:")
    print(f"     Total steps: {summary['total_steps_tracked']}")
    print(f"     Final mean reward: {summary['current_metrics']['mean_reward']:.3f}")
    print(f"     Final SNR: {summary['current_metrics']['signal_to_noise']:.3f}")
    print(f"     Anomalies detected: {summary['anomalies_detected']}")

    print(f"\n  📈 Recent Trends:")
    reward_trend = trends['mean_reward_trend']
    print(f"     Mean reward: {reward_trend['direction']} "
          f"({reward_trend['start']:.3f} → {reward_trend['current']:.3f})")

    for func_name, func_trend in trends['function_trends'].items():
        print(f"     {func_name:20s}: {func_trend['direction']:10s} "
              f"(current={func_trend['current']:5.2f})")

    # Alerts summary
    alert_summary = monitor.get_alert_summary()
    print(f"\n  🚨 Alerts Summary:")
    print(f"     Total alerts: {alert_summary['total_alerts']}")
    for level, count in alert_summary['by_level'].items():
        if count > 0:
            print(f"     {level:10s}: {count}")

    print()

    # Log final summary
    dashboard.log_summary(tracker, num_steps - 1)

    # Close dashboard
    dashboard.close()

    # Display instructions
    print("="*60)
    print("Demo Complete!")
    print("="*60)
    print()
    print("📂 Monitoring data saved to:", monitoring_dir)
    print()
    print("📊 View dashboards:")
    print()
    print("  TensorBoard:")
    print(f"    tensorboard --logdir=/tmp/tensorboard/grpo_rewards")
    print("    Open: http://localhost:6006")
    print()
    print("  Web Dashboard:")
    print(f"    streamlit run src/monitoring/web_dashboard.py -- --data-dir {monitoring_dir}")
    print("    Open: http://localhost:8501")
    print()
    print("="*60)


if __name__ == "__main__":
    main()
