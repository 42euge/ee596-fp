"""
Command-line interface for reward robustness evaluation.

Usage:
    python -m src.reward_robustness.cli --help
    python -m src.reward_robustness.cli evaluate --data-source gsm8k --num-samples 100
    python -m src.reward_robustness.cli report --input results.json --format markdown
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Optional


def parse_args() -> argparse.Namespace:
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Evaluate reward model robustness across semantic-preserving perturbations",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate internal rewards on GSM8K
  python -m src.reward_robustness.cli evaluate \\
      --data-source gsm8k \\
      --num-samples 100 \\
      --rewards format_reward accuracy_reward

  # With external reward models
  python -m src.reward_robustness.cli evaluate \\
      --data-source gsm8k \\
      --external-rewards "RLHFlow/ArmoRM-Llama3-8B-v0.1" \\
      --perturbations synonym paraphrase

  # Generate report from results
  python -m src.reward_robustness.cli report \\
      --input ./robustness_results/eval.json \\
      --format markdown
        """,
    )

    subparsers = parser.add_subparsers(dest="mode", help="Operation mode")

    # Evaluate subcommand
    eval_parser = subparsers.add_parser(
        "evaluate", help="Run robustness evaluation"
    )
    eval_parser.add_argument(
        "--data-source",
        type=str,
        default="gsm8k",
        choices=["gsm8k", "openrubrics", "custom"],
        help="Data source for evaluation (default: gsm8k)",
    )
    eval_parser.add_argument(
        "--data-path",
        type=str,
        default="./data",
        help="Path to data directory (for gsm8k) or custom data file",
    )
    eval_parser.add_argument(
        "--completions-file",
        type=str,
        help="JSON file with pre-generated completions (for custom source)",
    )
    eval_parser.add_argument(
        "--num-samples",
        type=int,
        default=100,
        help="Number of samples to evaluate (default: 100)",
    )
    eval_parser.add_argument(
        "--rewards",
        nargs="+",
        default=["format_reward", "accuracy_reward"],
        help="Internal reward functions to evaluate",
    )
    eval_parser.add_argument(
        "--external-rewards",
        nargs="+",
        default=[],
        help="External HuggingFace reward model IDs",
    )
    eval_parser.add_argument(
        "--perturbations",
        nargs="+",
        default=["synonym", "paraphrase", "reorder"],
        choices=["synonym", "paraphrase", "reorder"],
        help="Perturbation types to apply",
    )
    eval_parser.add_argument(
        "--num-variants",
        type=int,
        default=5,
        help="Number of variants per perturbation type (default: 5)",
    )
    eval_parser.add_argument(
        "--output",
        type=str,
        default="./robustness_results",
        help="Output directory for results",
    )
    eval_parser.add_argument(
        "--device",
        type=str,
        default="auto",
        choices=["auto", "cuda", "mps", "cpu"],
        help="Device for model inference",
    )
    eval_parser.add_argument(
        "--no-details",
        action="store_true",
        help="Don't save per-sample details",
    )
    eval_parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress output",
    )

    # Report subcommand
    report_parser = subparsers.add_parser(
        "report", help="Generate report from results"
    )
    report_parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Input JSON results file",
    )
    report_parser.add_argument(
        "--format",
        type=str,
        default="markdown",
        choices=["markdown", "text", "json"],
        help="Output format (default: markdown)",
    )
    report_parser.add_argument(
        "--output",
        type=str,
        help="Output file (default: stdout)",
    )

    return parser.parse_args()


def load_data(
    source: str,
    data_path: str,
    completions_file: Optional[str],
    num_samples: int,
) -> tuple:
    """Load evaluation data.

    Returns:
        Tuple of (prompts, completions, answers, rubrics)
    """
    prompts: List[str] = []
    completions: List[str] = []
    answers: List[str] = []
    rubrics: List[str] = []

    if source == "gsm8k":
        try:
            from ..utils import load_gsm8k_dataset
        except ImportError:
            from src.utils import load_gsm8k_dataset

        data = load_gsm8k_dataset(data_path, split="test", max_examples=num_samples)
        prompts = [d["question"] for d in data]
        answers = [d["answer"] for d in data]

        # If completions file provided, use those
        if completions_file:
            with open(completions_file) as f:
                comp_data = json.load(f)
                completions = comp_data.get("completions", [])
        else:
            # Generate placeholder completions for testing
            print("Warning: No completions file provided. Using placeholder completions.")
            completions = [
                f"<reasoning>\nLet me solve this step by step.\n</reasoning>\n\n<answer>\n{ans}\n</answer>"
                for ans in answers
            ]

    elif source == "openrubrics":
        try:
            from ..utils import load_openrubrics_dataset
        except ImportError:
            from src.utils import load_openrubrics_dataset

        data = load_openrubrics_dataset(split="train", max_examples=num_samples)
        prompts = [d["question"] for d in data]
        rubrics = [d.get("rubric", "") for d in data]

        if completions_file:
            with open(completions_file) as f:
                comp_data = json.load(f)
                completions = comp_data.get("completions", [])
        else:
            # Use reference responses as completions
            completions = [d.get("reference_response", "") for d in data]

    elif source == "custom":
        if not completions_file:
            raise ValueError("--completions-file required for custom data source")

        with open(completions_file) as f:
            data = json.load(f)
            prompts = data.get("prompts", [])
            completions = data.get("completions", [])
            answers = data.get("answers", [])
            rubrics = data.get("rubrics", [])

    return prompts, completions, answers, rubrics


def run_evaluate(args: argparse.Namespace) -> int:
    """Run the evaluation command."""
    from .config import RobustnessConfig, PerturbationConfig, ExternalRewardConfig
    from .evaluator import RobustnessEvaluator

    # Load data
    if not args.quiet:
        print(f"Loading data from {args.data_source}...")

    prompts, completions, answers, rubrics = load_data(
        args.data_source,
        args.data_path,
        args.completions_file,
        args.num_samples,
    )

    if not prompts or not completions:
        print("Error: No data loaded")
        return 1

    if not args.quiet:
        print(f"Loaded {len(prompts)} samples")

    # Build configuration
    config = RobustnessConfig(
        internal_rewards=args.rewards,
        external_rewards=ExternalRewardConfig(
            model_ids=args.external_rewards,
            device=args.device,
        ),
        perturbations=PerturbationConfig(
            enabled_types=args.perturbations,
            num_variants=args.num_variants,
        ),
        num_samples=args.num_samples,
        output_dir=args.output,
        save_detailed=not args.no_details,
    )

    # Run evaluation
    evaluator = RobustnessEvaluator(config)
    results = evaluator.evaluate(
        prompts=prompts,
        completions=completions,
        answers=answers if answers else None,
        rubrics=rubrics if rubrics else None,
        verbose=not args.quiet,
    )

    # Print summary
    if not args.quiet:
        print("\n" + "=" * 60)
        print("RESULTS SUMMARY")
        print("=" * 60)
        for item in results.ranking:
            print(f"  {item['reward']}: stability = {item['stability_score']:.4f}")
        print("=" * 60)

    return 0


def run_report(args: argparse.Namespace) -> int:
    """Run the report command."""
    from .evaluator import RobustnessResults
    from .metrics import ConsistencyMetrics

    # Load results
    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: Input file not found: {input_path}")
        return 1

    with open(input_path) as f:
        data = json.load(f)

    # Reconstruct results object
    metrics = {}
    for name, m in data.get("results", {}).items():
        metrics[name] = ConsistencyMetrics(
            reward_name=name,
            mean_variance=m["mean_variance"],
            max_variance=m["max_variance"],
            median_variance=m["median_variance"],
            variance_std=m["variance_std"],
            mean_cv=m["mean_cv"],
            kendall_tau=m["kendall_tau"],
            spearman_rho=m["spearman_rho"],
            flip_rate=m["flip_rate"],
            max_deviation=m["max_deviation"],
            mean_deviation=m["mean_deviation"],
            stability_score=m["stability_score"],
            num_samples=m["num_samples"],
        )

    results = RobustnessResults(
        timestamp=data["meta"]["timestamp"],
        config=data["meta"]["config"],
        num_samples=data["meta"]["num_samples"],
        perturbation_types=data["meta"]["perturbation_types"],
        num_variants_per_sample=data["meta"]["num_variants_per_sample"],
        metrics=metrics,
        ranking=data["summary"]["ranking"],
        samples=None,
    )

    # Create a minimal evaluator just for report generation
    from .evaluator import RobustnessEvaluator

    evaluator = RobustnessEvaluator()
    report = evaluator.generate_report(results, format=args.format)

    # Output
    if args.output:
        with open(args.output, "w") as f:
            f.write(report)
        print(f"Report saved to: {args.output}")
    else:
        print(report)

    return 0


def main() -> int:
    """Main entry point."""
    args = parse_args()

    if args.mode == "evaluate":
        return run_evaluate(args)
    elif args.mode == "report":
        return run_report(args)
    else:
        print("Please specify a mode: evaluate or report")
        print("Use --help for more information")
        return 1


if __name__ == "__main__":
    sys.exit(main())
