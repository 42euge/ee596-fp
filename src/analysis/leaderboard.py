"""Leaderboard generation for experiments."""

import json
import sqlite3
from pathlib import Path
from typing import Dict, List, Optional

from ..experiment_tracker import LocalBackend


def generate_leaderboard(
    db_path: str = "experiments.db",
    benchmark: Optional[str] = None,
    metric: str = "accuracy",
    top_k: int = 10,
    status_filter: Optional[str] = "completed"
) -> List[Dict]:
    """Generate leaderboard of experiments.

    Args:
        db_path: Path to experiments database
        benchmark: Filter by benchmark name (None = all)
        metric: Metric to rank by (default: accuracy)
        top_k: Number of top experiments to return
        status_filter: Filter by status (None = all)

    Returns:
        List of experiment dictionaries sorted by metric
    """
    backend = LocalBackend(db_path)

    # Get all experiments
    experiments = backend.get_all_experiments()

    # Filter by status
    if status_filter:
        experiments = [e for e in experiments if e.get("status") == status_filter]

    # Get evaluation results for each experiment
    leaderboard = []
    for exp in experiments:
        exp_id = exp["experiment_id"]

        # Get evaluation results
        eval_results = backend.get_evaluation_results(exp_id, benchmark)

        if not eval_results:
            continue

        # Extract metrics
        for eval_result in eval_results:
            benchmark_name = eval_result["benchmark_name"]
            metrics_json = eval_result["metrics_json"]

            # Get the target metric
            if metric in metrics_json:
                leaderboard.append({
                    "experiment_id": exp_id,
                    "benchmark": benchmark_name,
                    "metric_name": metric,
                    "metric_value": metrics_json[metric],
                    "accuracy": metrics_json.get("accuracy", 0),
                    "partial_accuracy": metrics_json.get("partial_accuracy", 0),
                    "format_accuracy": metrics_json.get("format_accuracy", 0),
                    "timestamp": eval_result["timestamp"],
                    "git_commit": exp["git_commit"][:7],
                    "git_branch": exp["git_branch"],
                    "config_json": exp["config_json"],
                })

    # Sort by metric value (descending)
    leaderboard.sort(key=lambda x: x["metric_value"], reverse=True)

    # Return top k
    return leaderboard[:top_k]


def print_leaderboard(
    db_path: str = "experiments.db",
    benchmark: Optional[str] = None,
    metric: str = "accuracy",
    top_k: int = 10
):
    """Print leaderboard to console.

    Args:
        db_path: Path to experiments database
        benchmark: Filter by benchmark name
        metric: Metric to rank by
        top_k: Number of top experiments to show
    """
    leaderboard = generate_leaderboard(db_path, benchmark, metric, top_k)

    if not leaderboard:
        print("No experiments found.")
        return

    # Print header
    benchmark_name = benchmark or "All Benchmarks"
    print(f"\n{'='*100}")
    print(f"{benchmark_name} Leaderboard - Top {top_k} by {metric}")
    print(f"{'='*100}")

    # Print table header
    print(f"{'Rank':<6} {'Experiment ID':<30} {metric.capitalize():<12} {'Partial':<12} {'Format':<12} {'Date':<12}")
    print(f"{'-'*6} {'-'*30} {'-'*12} {'-'*12} {'-'*12} {'-'*12}")

    # Print rows
    for i, entry in enumerate(leaderboard, 1):
        exp_id_short = entry["experiment_id"][:28]
        date = entry["timestamp"][:10]

        print(
            f"{i:<6} "
            f"{exp_id_short:<30} "
            f"{entry['accuracy']:>10.1%}  "
            f"{entry['partial_accuracy']:>10.1%}  "
            f"{entry['format_accuracy']:>10.1%}  "
            f"{date:<12}"
        )

    print(f"{'='*100}\n")


def print_experiment_details(experiment_id: str, db_path: str = "experiments.db"):
    """Print detailed information about an experiment.

    Args:
        experiment_id: Experiment ID
        db_path: Path to experiments database
    """
    backend = LocalBackend(db_path)

    # Get experiment
    exp = backend.get_experiment(experiment_id)
    if not exp:
        print(f"Experiment {experiment_id} not found.")
        return

    # Parse config
    config = json.loads(exp["config_json"])

    # Print metadata
    print(f"\n{'='*80}")
    print(f"Experiment: {experiment_id}")
    print(f"{'='*80}")

    print("\nMetadata:")
    print(f"  Status:        {exp['status']}")
    print(f"  Git Commit:    {exp['git_commit']}")
    print(f"  Git Branch:    {exp['git_branch']}")
    print(f"  Git Dirty:     {exp['git_dirty']}")
    print(f"  Timestamp:     {exp['timestamp']}")
    print(f"  User:          {exp['user']}")
    print(f"  Hostname:      {exp['hostname']}")
    print(f"  Device:        {exp['device_type']}")

    if exp.get('notes'):
        print(f"  Notes:         {exp['notes']}")

    # Print configuration
    print("\nConfiguration:")
    print(f"  Model:         {config.get('base_model', 'N/A')}")
    print(f"  LoRA:          rank={config.get('lora_rank', 'N/A')}, alpha={config.get('lora_alpha', 'N/A')}")
    print(f"  Learning Rate: {config.get('learning_rate', 'N/A')}")
    print(f"  Batch Size:    {config.get('batch_size', 'N/A')}")
    print(f"  Num Steps:     {config.get('num_steps', 'N/A')}")
    print(f"  GRPO Beta:     {config.get('beta', 'N/A')}")
    print(f"  Temperature:   {config.get('temperature', 'N/A')}")

    # Print evaluation results
    eval_results = backend.get_evaluation_results(experiment_id)

    if eval_results:
        print("\nEvaluation Results:")
        for eval_result in eval_results:
            benchmark = eval_result["benchmark_name"]
            metrics = eval_result["metrics_json"]

            print(f"\n  {benchmark}:")
            print(f"    Accuracy:         {metrics.get('accuracy', 0):.1%}")
            print(f"    Partial Accuracy: {metrics.get('partial_accuracy', 0):.1%}")
            print(f"    Format Accuracy:  {metrics.get('format_accuracy', 0):.1%}")
            print(f"    Avg Gen Time:     {metrics.get('avg_generation_time', 0):.2f}s")

    print(f"{'='*80}\n")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Generate experiment leaderboard")
    parser.add_argument("--db", default="experiments.db", help="Path to database")
    parser.add_argument("--benchmark", help="Filter by benchmark name")
    parser.add_argument("--metric", default="accuracy", help="Metric to rank by")
    parser.add_argument("--top", type=int, default=10, help="Number of top experiments")
    parser.add_argument("--details", help="Show details for specific experiment")

    args = parser.parse_args()

    if args.details:
        print_experiment_details(args.details, args.db)
    else:
        print_leaderboard(args.db, args.benchmark, args.metric, args.top)
