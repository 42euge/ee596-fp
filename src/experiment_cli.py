"""
CLI tool for experiment management.

Provides commands for listing, comparing, and analyzing experiments.
"""

import argparse
import sys

from analysis.leaderboard import print_leaderboard, print_experiment_details
from analysis.compare import compare_experiments, format_comparison
from analysis.statistics import compute_significance, format_significance_result
from experiment_tracker import LocalBackend


def cmd_list(args):
    """List experiments."""
    backend = LocalBackend(args.db)
    experiments = backend.get_all_experiments(limit=args.limit)

    if not experiments:
        print("No experiments found.")
        return

    print(f"\n{'='*120}")
    print(f"Experiments (showing {len(experiments)})")
    print(f"{'='*120}")

    print(f"{'ID':<35} {'Status':<12} {'Branch':<25} {'Date':<12} {'Commit':<10}")
    print(f"{'-'*35} {'-'*12} {'-'*25} {'-'*12} {'-'*10}")

    for exp in experiments:
        exp_id = exp["experiment_id"][:33]
        status = exp["status"][:10]
        branch = exp["git_branch"][:23]
        date = exp["timestamp"][:10]
        commit = exp["git_commit"][:8]

        print(f"{exp_id:<35} {status:<12} {branch:<25} {date:<12} {commit:<10}")

    print(f"{'='*120}\n")


def cmd_show(args):
    """Show experiment details."""
    print_experiment_details(args.experiment_id, args.db)


def cmd_leaderboard(args):
    """Show leaderboard."""
    print_leaderboard(
        db_path=args.db,
        benchmark=args.benchmark,
        metric=args.metric,
        top_k=args.top
    )


def cmd_compare(args):
    """Compare experiments."""
    comparison = compare_experiments(
        args.experiment_ids,
        db_path=args.db,
        benchmark=args.benchmark
    )
    print(format_comparison(comparison))


def cmd_significance(args):
    """Test statistical significance."""
    result = compute_significance(
        args.exp1,
        args.exp2,
        db_path=args.db,
        benchmark=args.benchmark,
        metric=args.metric,
        method=args.method,
        n_bootstrap=args.n_bootstrap
    )
    print(format_significance_result(result))


def cmd_delete(args):
    """Delete an experiment."""
    import sqlite3

    if not args.confirm:
        response = input(f"Are you sure you want to delete experiment {args.experiment_id}? (yes/no): ")
        if response.lower() != "yes":
            print("Cancelled.")
            return

    conn = sqlite3.connect(args.db)
    cursor = conn.cursor()

    # Delete from all tables
    cursor.execute("DELETE FROM experiments WHERE experiment_id = ?", (args.experiment_id,))
    cursor.execute("DELETE FROM training_metrics WHERE experiment_id = ?", (args.experiment_id,))
    cursor.execute("DELETE FROM evaluation_results WHERE experiment_id = ?", (args.experiment_id,))
    cursor.execute("DELETE FROM checkpoints WHERE experiment_id = ?", (args.experiment_id,))

    conn.commit()
    deleted = cursor.rowcount
    conn.close()

    if deleted > 0:
        print(f"Deleted experiment: {args.experiment_id}")
    else:
        print(f"Experiment not found: {args.experiment_id}")


def cmd_export(args):
    """Export experiment data."""
    import json
    from pathlib import Path

    backend = LocalBackend(args.db)

    # Get experiment
    exp = backend.get_experiment(args.experiment_id)
    if not exp:
        print(f"Experiment {args.experiment_id} not found.")
        return

    # Get metrics
    metrics = backend.get_metrics(args.experiment_id)

    # Get evaluations
    evaluations = backend.get_evaluation_results(args.experiment_id)

    # Create export data
    export_data = {
        "experiment": exp,
        "metrics": metrics,
        "evaluations": evaluations,
    }

    # Write to file
    output_path = Path(args.output)
    output_path.write_text(json.dumps(export_data, indent=2))

    print(f"Exported experiment to: {output_path}")


def cmd_stats(args):
    """Show database statistics."""
    backend = LocalBackend(args.db)

    experiments = backend.get_all_experiments()

    # Count by status
    status_counts = {}
    branch_counts = {}
    benchmark_counts = {}

    for exp in experiments:
        status = exp["status"]
        status_counts[status] = status_counts.get(status, 0) + 1

        branch = exp["git_branch"]
        branch_counts[branch] = branch_counts.get(branch, 0) + 1

        # Get evaluations
        evals = backend.get_evaluation_results(exp["experiment_id"])
        for eval_result in evals:
            benchmark = eval_result["benchmark_name"]
            benchmark_counts[benchmark] = benchmark_counts.get(benchmark, 0) + 1

    print(f"\n{'='*80}")
    print("Database Statistics")
    print(f"{'='*80}")

    print(f"\nTotal Experiments: {len(experiments)}")

    print("\nBy Status:")
    for status, count in sorted(status_counts.items()):
        print(f"  {status}: {count}")

    print("\nBy Branch:")
    for branch, count in sorted(branch_counts.items(), key=lambda x: -x[1])[:10]:
        print(f"  {branch}: {count}")

    print("\nBy Benchmark:")
    for benchmark, count in sorted(benchmark_counts.items()):
        print(f"  {benchmark}: {count}")

    print(f"{'='*80}\n")


def main():
    """Main CLI entry point."""
    parser = argparse.ArgumentParser(
        description="Experiment tracking CLI",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # List all experiments
  %(prog)s list

  # Show leaderboard
  %(prog)s leaderboard --benchmark gsm8k --top 10

  # Show experiment details
  %(prog)s show exp_20250123_143022_a1b2c3

  # Compare two experiments
  %(prog)s compare exp_001 exp_002

  # Test statistical significance
  %(prog)s significance exp_001 exp_002 --benchmark gsm8k

  # Export experiment data
  %(prog)s export exp_001 --output exp_001.json

  # Show database statistics
  %(prog)s stats
        """
    )

    parser.add_argument("--db", default="experiments.db", help="Path to database")

    subparsers = parser.add_subparsers(dest="command", help="Command to run")

    # List command
    list_parser = subparsers.add_parser("list", help="List experiments")
    list_parser.add_argument("--limit", type=int, default=50, help="Maximum number to show")
    list_parser.set_defaults(func=cmd_list)

    # Show command
    show_parser = subparsers.add_parser("show", help="Show experiment details")
    show_parser.add_argument("experiment_id", help="Experiment ID")
    show_parser.set_defaults(func=cmd_show)

    # Leaderboard command
    leaderboard_parser = subparsers.add_parser("leaderboard", help="Show leaderboard")
    leaderboard_parser.add_argument("--benchmark", help="Filter by benchmark")
    leaderboard_parser.add_argument("--metric", default="accuracy", help="Metric to rank by")
    leaderboard_parser.add_argument("--top", type=int, default=10, help="Number of top experiments")
    leaderboard_parser.set_defaults(func=cmd_leaderboard)

    # Compare command
    compare_parser = subparsers.add_parser("compare", help="Compare experiments")
    compare_parser.add_argument("experiment_ids", nargs="+", help="Experiment IDs to compare")
    compare_parser.add_argument("--benchmark", help="Filter by benchmark")
    compare_parser.set_defaults(func=cmd_compare)

    # Significance command
    sig_parser = subparsers.add_parser("significance", help="Test statistical significance")
    sig_parser.add_argument("exp1", help="First experiment ID")
    sig_parser.add_argument("exp2", help="Second experiment ID")
    sig_parser.add_argument("--benchmark", default="gsm8k", help="Benchmark name")
    sig_parser.add_argument("--metric", default="accuracy", help="Metric to compare")
    sig_parser.add_argument("--method", default="bootstrap", choices=["bootstrap", "t-test", "permutation"])
    sig_parser.add_argument("--n_bootstrap", type=int, default=10000, help="Number of bootstrap samples")
    sig_parser.set_defaults(func=cmd_significance)

    # Delete command
    delete_parser = subparsers.add_parser("delete", help="Delete an experiment")
    delete_parser.add_argument("experiment_id", help="Experiment ID")
    delete_parser.add_argument("--confirm", action="store_true", help="Skip confirmation")
    delete_parser.set_defaults(func=cmd_delete)

    # Export command
    export_parser = subparsers.add_parser("export", help="Export experiment data")
    export_parser.add_argument("experiment_id", help="Experiment ID")
    export_parser.add_argument("--output", required=True, help="Output JSON file")
    export_parser.set_defaults(func=cmd_export)

    # Stats command
    stats_parser = subparsers.add_parser("stats", help="Show database statistics")
    stats_parser.set_defaults(func=cmd_stats)

    # Parse args
    args = parser.parse_args()

    if not args.command:
        parser.print_help()
        sys.exit(1)

    # Run command
    args.func(args)


if __name__ == "__main__":
    main()
