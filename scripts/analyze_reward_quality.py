#!/usr/bin/env python3
"""
Standalone script for analyzing reward quality from training outputs.

This script can be used to perform post-training analysis of reward quality
when the monitoring system wasn't integrated during training.

Usage:
    # Analyze from checkpoint
    python scripts/analyze_reward_quality.py --checkpoint /path/to/checkpoint --test-data ./data/test

    # Analyze from saved responses file
    python scripts/analyze_reward_quality.py --responses responses.jsonl --rewards rewards.jsonl

    # Generate report
    python scripts/analyze_reward_quality.py --metrics-log /tmp/metrics.jsonl --alert-log /tmp/alerts.jsonl --report output.html
"""

import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def parse_args():
    parser = argparse.ArgumentParser(
        description="Analyze reward quality from training outputs"
    )

    # Input sources
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument(
        "--checkpoint",
        type=str,
        help="Path to model checkpoint to evaluate"
    )
    input_group.add_argument(
        "--responses",
        type=str,
        help="Path to JSONL file containing saved responses"
    )
    input_group.add_argument(
        "--metrics-log",
        type=str,
        help="Path to existing metrics log file (for report generation)"
    )

    # Supporting inputs
    parser.add_argument(
        "--test-data",
        type=str,
        default="./data/test",
        help="Path to test data directory"
    )
    parser.add_argument(
        "--rewards",
        type=str,
        help="Path to JSONL file containing rewards (if using --responses)"
    )
    parser.add_argument(
        "--alert-log",
        type=str,
        help="Path to alerts log file (for report generation)"
    )

    # Model parameters (for checkpoint evaluation)
    parser.add_argument(
        "--model-id",
        type=str,
        default="google/gemma-3-1b-it",
        help="HuggingFace model ID"
    )
    parser.add_argument(
        "--num-samples",
        type=int,
        default=500,
        help="Number of samples to generate and analyze"
    )
    parser.add_argument(
        "--batch-size",
        type=int,
        default=10,
        help="Batch size for generation"
    )

    # Output options
    parser.add_argument(
        "--report",
        type=str,
        default="reward_quality_report.html",
        help="Path to save HTML report"
    )
    parser.add_argument(
        "--output-metrics",
        type=str,
        help="Path to save metrics JSON"
    )
    parser.add_argument(
        "--output-alerts",
        type=str,
        help="Path to save alerts JSON"
    )

    # Analysis parameters
    parser.add_argument(
        "--window-size",
        type=int,
        default=1000,
        help="Window size for rolling statistics"
    )
    parser.add_argument(
        "--min-samples",
        type=int,
        default=100,
        help="Minimum samples before running statistical tests"
    )

    return parser.parse_args()


def load_responses_from_file(filepath: str) -> List[str]:
    """Load responses from JSONL file."""
    responses = []
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            if 'response' in data:
                responses.append(data['response'])
            elif 'text' in data:
                responses.append(data['text'])
            else:
                responses.append(data.get('output', str(data)))
    return responses


def load_rewards_from_file(filepath: str) -> Dict[str, List[float]]:
    """Load rewards from JSONL file."""
    all_rewards = {}
    with open(filepath, 'r') as f:
        for line in f:
            data = json.loads(line)
            for key, value in data.items():
                if key not in all_rewards:
                    all_rewards[key] = []
                all_rewards[key].append(float(value))
    return all_rewards


def evaluate_checkpoint(
    checkpoint_path: str,
    test_data_path: str,
    model_id: str,
    num_samples: int,
    batch_size: int
) -> tuple[List[str], Dict[str, List[float]]]:
    """
    Generate responses from checkpoint and compute rewards.

    Returns:
        Tuple of (responses, rewards)
    """
    print(f"Loading checkpoint from {checkpoint_path}...")

    # Import dependencies
    try:
        from src.model import GemmaModel
        from tunrex.datasets import (
            get_train_val_test_datasets,
            get_system_prompt,
            DEFAULT_TEMPLATE,
            match_format_exactly,
            check_answer,
            check_numbers,
        )
    except ImportError as e:
        print(f"Error importing dependencies: {e}")
        sys.exit(1)

    # Load model
    print(f"Loading model: {model_id}...")
    model = GemmaModel.load(model_id)

    # Load test data
    print(f"Loading test data from {test_data_path}...")
    _, _, test_dataset = get_train_val_test_datasets(
        test_data_dir=test_data_path,
        source="tfds",
        batch_size=batch_size,
        num_batches=num_samples // batch_size,
        template=DEFAULT_TEMPLATE,
        system_prompt=get_system_prompt(0),
    )

    # Generate responses
    print(f"Generating {num_samples} responses...")
    responses = []
    references = []

    for batch in test_dataset:
        prompts = batch['prompt']
        refs = batch.get('answer', [''] * len(prompts))

        batch_responses = model.generate_batch(prompts)
        responses.extend(batch_responses)
        references.extend(refs)

        if len(responses) >= num_samples:
            break

    responses = responses[:num_samples]
    references = references[:num_samples]

    # Compute rewards
    print("Computing rewards...")
    reward_functions = {
        'format': match_format_exactly,
        'answer': check_answer,
        'numbers': check_numbers,
    }

    rewards = {}
    for name, fn in reward_functions.items():
        rewards[name] = [fn(resp, ref) for resp, ref in zip(responses, references)]

    return responses, rewards


def analyze_quality(
    responses: List[str],
    rewards: Dict[str, List[float]],
    window_size: int,
    min_samples: int
) -> tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    Analyze quality of responses and rewards.

    Returns:
        Tuple of (metrics_summary, alerts_list)
    """
    from tunrex.datasets import RewardQualityAssessor

    print("\nAnalyzing reward quality...")

    assessor = RewardQualityAssessor(
        window_size=window_size,
        min_samples_for_analysis=min_samples
    )

    # Process in batches
    batch_size = 50
    all_alerts = []

    for i in range(0, len(responses), batch_size):
        batch_responses = responses[i:i+batch_size]
        batch_rewards = {
            name: values[i:i+batch_size]
            for name, values in rewards.items()
        }

        metrics, alerts = assessor.assess_batch(batch_responses, batch_rewards)

        # Store alerts
        for alert in alerts:
            all_alerts.append({
                'batch': i // batch_size,
                'severity': alert.severity,
                'pathology_type': alert.pathology_type,
                'message': alert.message,
                'metrics': alert.metrics
            })

    # Get summary
    summary = assessor.get_summary_statistics()
    summary['final_metrics'] = metrics.to_dict()

    return summary, all_alerts


def print_summary(summary: Dict[str, Any], alerts: List[Dict[str, Any]]):
    """Print analysis summary to console."""
    print("\n" + "=" * 80)
    print("REWARD QUALITY ANALYSIS SUMMARY")
    print("=" * 80)

    print(f"\nTotal Samples Analyzed: {summary['total_samples']}")
    print(f"Total Alerts: {summary['total_alerts']}")

    print("\nAlerts by Severity:")
    for severity, count in sorted(summary['alerts_by_severity'].items()):
        if count > 0:
            print(f"  {severity.upper()}: {count}")

    print("\nAlerts by Type:")
    for ptype, count in sorted(
        summary['alerts_by_type'].items(),
        key=lambda x: x[1],
        reverse=True
    ):
        print(f"  {ptype}: {count}")

    print("\nReward Statistics:")
    for reward_name, stats in summary.get('reward_statistics', {}).items():
        print(f"  {reward_name}:")
        print(f"    Mean: {stats['mean']:.3f}")
        print(f"    Std:  {stats['std']:.3f}")
        print(f"    Min:  {stats['min']:.3f}")
        print(f"    Max:  {stats['max']:.3f}")

    # Show critical/high alerts
    critical_alerts = [a for a in alerts if a['severity'] in ['critical', 'high']]
    if critical_alerts:
        print(f"\nCritical/High Severity Alerts ({len(critical_alerts)}):")
        for alert in critical_alerts[:10]:  # Show first 10
            print(f"  [{alert['severity'].upper()}] {alert['message']}")

    print("\n" + "=" * 80)


def generate_report(
    metrics_log: str,
    alert_log: str,
    output_path: str
):
    """Generate HTML report from log files."""
    from tunrex.datasets import RewardQualityDashboard

    print(f"Generating report from logs...")
    print(f"  Metrics: {metrics_log}")
    print(f"  Alerts: {alert_log}")

    dashboard = RewardQualityDashboard(
        metrics_log_file=metrics_log,
        alert_log_file=alert_log
    )

    dashboard.generate_report(output_path)
    print(f"\nReport saved to: {output_path}")


def main():
    args = parse_args()

    # Determine analysis mode
    if args.metrics_log and args.alert_log:
        # Report generation mode
        generate_report(args.metrics_log, args.alert_log, args.report)
        return

    # Load data
    if args.checkpoint:
        # Evaluate checkpoint
        responses, rewards = evaluate_checkpoint(
            checkpoint_path=args.checkpoint,
            test_data_path=args.test_data,
            model_id=args.model_id,
            num_samples=args.num_samples,
            batch_size=args.batch_size
        )
    elif args.responses:
        # Load from files
        print(f"Loading responses from {args.responses}...")
        responses = load_responses_from_file(args.responses)

        if args.rewards:
            print(f"Loading rewards from {args.rewards}...")
            rewards = load_rewards_from_file(args.rewards)
        else:
            # Compute rewards from responses
            print("Computing rewards from responses...")
            from tunrex.datasets import (
                match_format_exactly,
                check_answer,
                check_numbers,
            )

            reward_functions = {
                'format': match_format_exactly,
                'answer': check_answer,
                'numbers': check_numbers,
            }

            rewards = {}
            for name, fn in reward_functions.items():
                rewards[name] = [fn(resp, "") for resp in responses]

    # Analyze
    summary, alerts = analyze_quality(
        responses=responses,
        rewards=rewards,
        window_size=args.window_size,
        min_samples=args.min_samples
    )

    # Print summary
    print_summary(summary, alerts)

    # Save outputs
    if args.output_metrics:
        print(f"\nSaving metrics to {args.output_metrics}...")
        with open(args.output_metrics, 'w') as f:
            json.dump(summary, f, indent=2)

    if args.output_alerts:
        print(f"Saving alerts to {args.output_alerts}...")
        with open(args.output_alerts, 'w') as f:
            json.dump(alerts, f, indent=2)

    # Generate simple text report
    report_path = args.report.replace('.html', '.txt')
    print(f"\nGenerating text report to {report_path}...")
    with open(report_path, 'w') as f:
        f.write("REWARD QUALITY ANALYSIS REPORT\n")
        f.write("=" * 80 + "\n\n")
        f.write(f"Total Samples: {summary['total_samples']}\n")
        f.write(f"Total Alerts: {summary['total_alerts']}\n\n")

        f.write("Alerts by Severity:\n")
        for severity, count in sorted(summary['alerts_by_severity'].items()):
            f.write(f"  {severity.upper()}: {count}\n")

        f.write("\nAlerts by Type:\n")
        for ptype, count in sorted(
            summary['alerts_by_type'].items(),
            key=lambda x: x[1],
            reverse=True
        ):
            f.write(f"  {ptype}: {count}\n")

        f.write("\nAll Alerts:\n")
        for i, alert in enumerate(alerts, 1):
            f.write(f"\n{i}. [{alert['severity'].upper()}] {alert['pathology_type']}\n")
            f.write(f"   {alert['message']}\n")

    print(f"\nAnalysis complete!")
    print(f"  Summary: {report_path}")
    if args.output_metrics:
        print(f"  Metrics: {args.output_metrics}")
    if args.output_alerts:
        print(f"  Alerts: {args.output_alerts}")


if __name__ == "__main__":
    main()
