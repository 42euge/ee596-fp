"""Experiment comparison utilities."""

import json
from typing import Dict, List, Optional

from ..experiment_tracker import LocalBackend


def compare_experiments(
    experiment_ids: List[str],
    db_path: str = "experiments.db",
    benchmark: Optional[str] = None
) -> Dict:
    """Compare multiple experiments.

    Args:
        experiment_ids: List of experiment IDs to compare
        db_path: Path to experiments database
        benchmark: Benchmark to compare on (None = all)

    Returns:
        Comparison dictionary
    """
    backend = LocalBackend(db_path)

    comparison = {
        "experiments": {},
        "config_diffs": [],
        "metric_comparison": {},
    }

    # Load all experiments
    configs = {}
    for exp_id in experiment_ids:
        exp = backend.get_experiment(exp_id)
        if not exp:
            print(f"Warning: Experiment {exp_id} not found")
            continue

        # Parse config
        config = json.loads(exp["config_json"])
        configs[exp_id] = config

        # Get evaluation results
        eval_results = backend.get_evaluation_results(exp_id, benchmark)

        comparison["experiments"][exp_id] = {
            "metadata": {
                "git_commit": exp["git_commit"][:7],
                "git_branch": exp["git_branch"],
                "timestamp": exp["timestamp"],
                "status": exp["status"],
            },
            "config": config,
            "evaluation": {}
        }

        # Add evaluation results
        for eval_result in eval_results:
            benchmark_name = eval_result["benchmark_name"]
            metrics = eval_result["metrics_json"]
            comparison["experiments"][exp_id]["evaluation"][benchmark_name] = metrics

    # Find config differences
    if len(configs) > 1:
        # Get all config keys
        all_keys = set()
        for config in configs.values():
            all_keys.update(_flatten_dict(config).keys())

        # Find differences
        for key in sorted(all_keys):
            values = {}
            for exp_id, config in configs.items():
                flat_config = _flatten_dict(config)
                values[exp_id] = flat_config.get(key, None)

            # Check if values differ
            unique_values = set(values.values())
            if len(unique_values) > 1:
                comparison["config_diffs"].append({
                    "key": key,
                    "values": values
                })

    # Compare metrics
    for exp_id in experiment_ids:
        if exp_id not in comparison["experiments"]:
            continue

        for benchmark_name, metrics in comparison["experiments"][exp_id]["evaluation"].items():
            if benchmark_name not in comparison["metric_comparison"]:
                comparison["metric_comparison"][benchmark_name] = {}

            for metric_name, metric_value in metrics.items():
                if metric_name not in comparison["metric_comparison"][benchmark_name]:
                    comparison["metric_comparison"][benchmark_name][metric_name] = {}

                comparison["metric_comparison"][benchmark_name][metric_name][exp_id] = metric_value

    return comparison


def _flatten_dict(d: Dict, parent_key: str = "", sep: str = ".") -> Dict:
    """Flatten nested dictionary.

    Args:
        d: Dictionary to flatten
        parent_key: Parent key prefix
        sep: Separator

    Returns:
        Flattened dictionary
    """
    items = []
    for k, v in d.items():
        new_key = f"{parent_key}{sep}{k}" if parent_key else k
        if isinstance(v, dict):
            items.extend(_flatten_dict(v, new_key, sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def format_comparison(comparison: Dict, verbose: bool = True) -> str:
    """Format comparison results as string.

    Args:
        comparison: Comparison dictionary from compare_experiments()
        verbose: Include detailed information

    Returns:
        Formatted string
    """
    lines = []

    # Header
    num_exps = len(comparison["experiments"])
    lines.append(f"\n{'='*100}")
    lines.append(f"Comparing {num_exps} Experiments")
    lines.append(f"{'='*100}")

    # Experiment list
    lines.append("\nExperiments:")
    for i, exp_id in enumerate(comparison["experiments"].keys(), 1):
        metadata = comparison["experiments"][exp_id]["metadata"]
        lines.append(
            f"  [{i}] {exp_id[:40]:<40} "
            f"({metadata['git_commit']}, {metadata['timestamp'][:10]})"
        )

    # Configuration differences
    if comparison["config_diffs"]:
        lines.append(f"\n{'='*100}")
        lines.append("Configuration Differences:")
        lines.append(f"{'='*100}")

        # Create table
        exp_ids = list(comparison["experiments"].keys())
        exp_ids_short = [exp_id[:15] for exp_id in exp_ids]

        lines.append(f"\n{'Parameter':<30} " + " ".join(f"{exp_id:<20}" for exp_id in exp_ids_short))
        lines.append(f"{'-'*30} " + " ".join("-"*20 for _ in exp_ids))

        for diff in comparison["config_diffs"]:
            key = diff["key"]
            if len(key) > 28:
                key = "..." + key[-25:]

            values = [str(diff["values"].get(exp_id, "N/A"))[:18] for exp_id in exp_ids]
            lines.append(f"{key:<30} " + " ".join(f"{v:<20}" for v in values))

    # Metric comparison
    if comparison["metric_comparison"]:
        lines.append(f"\n{'='*100}")
        lines.append("Performance Comparison:")
        lines.append(f"{'='*100}")

        exp_ids = list(comparison["experiments"].keys())
        exp_ids_short = [exp_id[:15] for exp_id in exp_ids]

        for benchmark, metrics in comparison["metric_comparison"].items():
            lines.append(f"\n{benchmark}:")
            lines.append(f"{'Metric':<25} " + " ".join(f"{exp_id:<20}" for exp_id in exp_ids_short))
            lines.append(f"{'-'*25} " + " ".join("-"*20 for _ in exp_ids))

            # Show main metrics first
            main_metrics = ["accuracy", "partial_accuracy", "format_accuracy"]
            other_metrics = [m for m in metrics.keys() if m not in main_metrics]

            for metric_name in main_metrics + other_metrics:
                if metric_name not in metrics:
                    continue

                values = metrics[metric_name]
                value_strs = []
                max_value = max(values.values()) if values else 0

                for exp_id in exp_ids:
                    if exp_id in values:
                        value = values[exp_id]
                        # Format based on metric type
                        if isinstance(value, float) and 0 <= value <= 1:
                            value_str = f"{value:>10.1%}"
                        elif isinstance(value, float):
                            value_str = f"{value:>10.3f}"
                        else:
                            value_str = f"{value:>10}"

                        # Add star for best value
                        if value == max_value and len(values) > 1:
                            value_str += " ⭐"
                        else:
                            value_str += "   "
                    else:
                        value_str = "N/A".rjust(13)

                    value_strs.append(value_str)

                lines.append(f"{metric_name:<25} " + " ".join(f"{v:<20}" for v in value_strs))

        # Find winner
        lines.append(f"\n{'='*100}")

        # Determine winner by average accuracy
        avg_accuracies = {}
        for exp_id in exp_ids:
            accuracies = []
            for benchmark, metrics in comparison["metric_comparison"].items():
                if "accuracy" in metrics and exp_id in metrics["accuracy"]:
                    accuracies.append(metrics["accuracy"][exp_id])

            if accuracies:
                avg_accuracies[exp_id] = sum(accuracies) / len(accuracies)

        if avg_accuracies:
            winner = max(avg_accuracies, key=avg_accuracies.get)
            winner_acc = avg_accuracies[winner]

            lines.append(f"Winner: {winner}")
            lines.append(f"Average Accuracy: {winner_acc:.1%}")

            # Compare to others
            if len(avg_accuracies) > 1:
                other_accs = [acc for exp_id, acc in avg_accuracies.items() if exp_id != winner]
                avg_other = sum(other_accs) / len(other_accs)
                improvement = ((winner_acc - avg_other) / avg_other) * 100
                lines.append(f"Improvement: {improvement:+.1f}% vs average of others")

    lines.append(f"{'='*100}\n")

    return "\n".join(lines)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Compare experiments")
    parser.add_argument("experiment_ids", nargs="+", help="Experiment IDs to compare")
    parser.add_argument("--db", default="experiments.db", help="Path to database")
    parser.add_argument("--benchmark", help="Filter by benchmark")

    args = parser.parse_args()

    comparison = compare_experiments(args.experiment_ids, args.db, args.benchmark)
    print(format_comparison(comparison))
