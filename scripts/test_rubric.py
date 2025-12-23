#!/usr/bin/env python3
"""
CLI tool for rapid rubric testing and experimentation

This script allows researchers to quickly test new rubric designs against small models
before scaling up to full training runs.

Usage:
    # Test a single rubric
    python scripts/test_rubric.py --rubric keyword --samples 50

    # Compare multiple rubrics
    python scripts/test_rubric.py --compare keyword format length --samples 100

    # Use custom model checkpoint
    python scripts/test_rubric.py --rubric format --model-checkpoint ./checkpoints/lora

    # Generate detailed report
    python scripts/test_rubric.py --compare keyword format --report rubric_report.md
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.rubric_testing import (
    KeywordMatchRubric,
    LengthRubric,
    FormatComplianceRubric,
    RubricEvaluator,
    EvaluationConfig,
    RubricComparator,
    RubricReporter,
    create_rubric,
    CompositeRubric,
)


# Registry of built-in rubrics
BUILTIN_RUBRICS = {
    "keyword": lambda: KeywordMatchRubric(name="keyword_match"),
    "length": lambda: LengthRubric(name="length_check", target_length=200),
    "format": lambda: FormatComplianceRubric(name="format_compliance"),
    "composite": lambda: CompositeRubric(
        "composite",
        [
            KeywordMatchRubric(name="keywords"),
            FormatComplianceRubric(name="format"),
        ]
    ),
}


def parse_args():
    parser = argparse.ArgumentParser(
        description="Rapid rubric testing tool for model evaluation",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    # Rubric selection
    rubric_group = parser.add_mutually_exclusive_group(required=True)
    rubric_group.add_argument(
        "--rubric",
        choices=list(BUILTIN_RUBRICS.keys()),
        help="Single rubric to evaluate"
    )
    rubric_group.add_argument(
        "--compare",
        nargs="+",
        choices=list(BUILTIN_RUBRICS.keys()),
        help="Multiple rubrics to compare"
    )

    # Dataset settings
    parser.add_argument(
        "--dataset",
        default="openrubrics",
        help="Dataset to use for evaluation"
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=100,
        help="Number of samples to evaluate"
    )
    parser.add_argument(
        "--dataset-split",
        default="train",
        help="Dataset split to use"
    )

    # Model settings
    parser.add_argument(
        "--model",
        default="google/gemma-3-1b-it",
        help="Model name or path"
    )
    parser.add_argument(
        "--model-checkpoint",
        help="Path to LoRA checkpoint to load"
    )
    parser.add_argument(
        "--quantization",
        choices=["4bit", "8bit", "none"],
        default="none",
        help="Model quantization level"
    )
    parser.add_argument(
        "--device",
        default="auto",
        help="Device to use (auto, cuda, cpu, mps)"
    )

    # Generation settings
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.9,
        help="Sampling temperature"
    )
    parser.add_argument(
        "--max-length",
        type=int,
        default=512,
        help="Maximum generation length"
    )
    parser.add_argument(
        "--no-generate",
        action="store_true",
        help="Skip generation (use existing completions if available)"
    )

    # Output settings
    parser.add_argument(
        "--report",
        help="Path to save report (markdown format)"
    )
    parser.add_argument(
        "--report-format",
        choices=["markdown", "json", "html"],
        default="markdown",
        help="Report format"
    )
    parser.add_argument(
        "--output-dir",
        default="./rubric_test_results",
        help="Directory to save results"
    )
    parser.add_argument(
        "--plot",
        action="store_true",
        help="Generate comparison plots (requires matplotlib)"
    )

    # Advanced settings
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for sampling"
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.05,
        help="Significance level for statistical tests"
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Verbose output"
    )

    return parser.parse_args()


def create_evaluation_config(args):
    """Create EvaluationConfig from command-line arguments"""
    config = EvaluationConfig(
        num_samples=args.samples,
        sample_seed=args.seed,
        max_length=args.max_length,
        temperature=args.temperature,
        num_generations_per_prompt=0 if args.no_generate else 1,
        model_name=args.model,
        device=args.device,
        quantization=None if args.quantization == "none" else args.quantization,
        use_lora=bool(args.model_checkpoint),
        lora_checkpoint=args.model_checkpoint,
        dataset_name=args.dataset,
        dataset_split=args.dataset_split,
        save_outputs=True,
        output_dir=args.output_dir,
    )
    return config


def evaluate_single_rubric(args):
    """Evaluate a single rubric"""
    print(f"=== Evaluating Rubric: {args.rubric} ===\n")

    # Create rubric
    rubric = BUILTIN_RUBRICS[args.rubric]()

    # Create evaluator
    config = create_evaluation_config(args)
    evaluator = RubricEvaluator(config)

    # Evaluate
    print("Starting evaluation...")
    result = evaluator.evaluate(rubric)

    # Print results
    print("\n=== Results ===")
    print(f"Rubric: {result.rubric_name}")
    print(f"Samples: {result.num_samples}")
    print(f"Mean Score: {result.mean_score:.2f} ± {result.std_score:.2f}")
    print(f"Median Score: {result.median_score:.2f}")
    print(f"Score Range: [{result.min_score:.2f}, {result.max_score:.2f}]")
    print(f"Time: {result.total_time:.2f}s ({result.time_per_sample:.3f}s/sample)")

    if result.component_stats:
        print("\nComponent Statistics:")
        for comp, stats in result.component_stats.items():
            print(f"  {comp}: {stats['mean']:.2f} ± {stats['std']:.2f}")

    # Generate report if requested
    if args.report:
        reporter = RubricReporter(output_dir=args.output_dir)
        reporter.generate_report([result], output_path=args.report, format=args.report_format)
        print(f"\nReport saved to: {args.report}")

    return result


def compare_rubrics(args):
    """Compare multiple rubrics"""
    print(f"=== Comparing Rubrics: {', '.join(args.compare)} ===\n")

    # Create rubrics
    rubrics = [BUILTIN_RUBRICS[name]() for name in args.compare]

    # Create evaluator
    config = create_evaluation_config(args)
    evaluator = RubricEvaluator(config)

    # Evaluate all rubrics
    print("Evaluating rubrics...")
    results = evaluator.evaluate_multiple(rubrics)

    # Compare results
    print("\n=== Comparison Results ===")
    comparator = RubricComparator(alpha=args.alpha)
    comparison = comparator.compare(results)

    print(f"\nBest Rubric: {comparison.best_rubric} (score: {comparison.best_score:.2f})")
    print("\nRankings:")
    for rubric, rank in sorted(comparison.rankings.items(), key=lambda x: x[1]):
        result = next(r for r in results if r.rubric_name == rubric)
        rel_perf = comparison.relative_performance[rubric]
        print(f"  {rank}. {rubric}: {result.mean_score:.2f} ± {result.std_score:.2f} ({rel_perf:.1%})")

    # Statistical significance
    if "pairwise_ttests" in comparison.statistical_tests:
        print("\nStatistical Significance (α={}):")
        for pair, test in comparison.statistical_tests["pairwise_ttests"].items():
            if "error" not in test:
                sig = "✓" if test["significant"] else "✗"
                print(f"  {pair}: p={test['p_value']:.4f} {sig}")

    if "anova" in comparison.statistical_tests:
        anova = comparison.statistical_tests["anova"]
        if "error" not in anova:
            print(f"\nANOVA: F={anova['f_statistic']:.4f}, p={anova['p_value']:.4f}")
            print(f"  {anova['interpretation']}")

    # Generate report
    if args.report:
        reporter = RubricReporter(output_dir=args.output_dir)
        reporter.generate_report(results, comparison, args.report, args.report_format)
        print(f"\nReport saved to: {args.report}")

    # Generate plots
    if args.plot:
        try:
            from src.rubric_testing.reporter import plot_rubric_comparison
            plot_path = Path(args.output_dir) / "rubric_comparison.png"
            plot_rubric_comparison(results, output_path=str(plot_path))
            print(f"Plot saved to: {plot_path}")
        except ImportError:
            print("\nWarning: matplotlib not installed. Install with: pip install matplotlib")

    return results, comparison


def main():
    args = parse_args()

    # Create output directory
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)

    # Set verbosity
    if args.verbose:
        import logging
        logging.basicConfig(level=logging.INFO)

    try:
        if args.rubric:
            # Evaluate single rubric
            result = evaluate_single_rubric(args)
        else:
            # Compare multiple rubrics
            results, comparison = compare_rubrics(args)

        print("\n✓ Evaluation complete!")

    except Exception as e:
        print(f"\n✗ Error: {e}", file=sys.stderr)
        if args.verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
