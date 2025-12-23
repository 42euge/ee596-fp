#!/usr/bin/env python3
"""
CLI tool for comparing grading methodologies.

Usage:
    python scripts/compare_grading.py --dataset gsm8k --split train --samples 100
    python scripts/compare_grading.py --dataset openrubrics --methods format_reward rubric_reward
    python scripts/compare_grading.py --help
"""

import argparse
import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from grading_registry import create_standard_comparator, get_all_grading_methods, analyze_dataset
from utils import load_gsm8k_dataset, load_openrubrics_dataset


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Compare different grading methodologies on a dataset",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Compare all methods on 100 GSM8K examples
  python scripts/compare_grading.py --dataset gsm8k --samples 100

  # Compare specific methods on OpenRubrics
  python scripts/compare_grading.py --dataset openrubrics --methods format_reward rubric_reward

  # Full analysis with plots
  python scripts/compare_grading.py --dataset gsm8k --samples 500 --plot --output ./my_analysis

  # List available methods
  python scripts/compare_grading.py --list-methods
        """
    )

    parser.add_argument(
        '--dataset',
        choices=['gsm8k', 'openrubrics'],
        help='Dataset to use for comparison'
    )

    parser.add_argument(
        '--split',
        default='train',
        help='Dataset split to use (default: train)'
    )

    parser.add_argument(
        '--samples',
        type=int,
        default=100,
        help='Number of samples to use (default: 100)'
    )

    parser.add_argument(
        '--methods',
        nargs='+',
        help='Specific grading methods to compare (default: all applicable)'
    )

    parser.add_argument(
        '--output',
        default='./results/grading_comparison',
        help='Output directory for results (default: ./results/grading_comparison)'
    )

    parser.add_argument(
        '--plot',
        action='store_true',
        help='Generate visualization plots'
    )

    parser.add_argument(
        '--list-methods',
        action='store_true',
        help='List all available grading methods and exit'
    )

    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for sampling (default: 42)'
    )

    parser.add_argument(
        '--data-dir',
        default='./data',
        help='Directory containing dataset files (for GSM8K)'
    )

    return parser.parse_args()


def list_methods():
    """List all available grading methods."""
    print("\n" + "=" * 70)
    print("AVAILABLE GRADING METHODS")
    print("=" * 70)

    methods = get_all_grading_methods()

    for name, metadata in methods.items():
        print(f"\n{name}")
        print(f"  Description: {metadata['description']}")
        print(f"  Score range: {metadata['score_range']}")
        print(f"  Requires ground truth: {metadata['requires_ground_truth']}")
        print(f"  Requires rubric: {metadata['requires_rubric']}")

    print("\n" + "=" * 70)


def load_dataset(args):
    """Load dataset based on arguments."""
    if args.dataset == 'gsm8k':
        print(f"Loading GSM8K dataset ({args.split} split)...")
        try:
            dataset = load_gsm8k_dataset(
                data_dir=args.data_dir,
                split=args.split,
                max_examples=args.samples,
                seed=args.seed
            )
            print(f"Loaded {len(dataset)} examples")

            # Add synthetic completions for testing
            import random
            random.seed(args.seed)

            for item in dataset:
                # Generate completions with varying quality
                quality = random.choice(['good', 'good', 'partial', 'poor'])

                if quality == 'good':
                    item['completion'] = (
                        f"<reasoning>Let me solve this step by step. "
                        f"The answer is {item['answer']}</reasoning>"
                        f"<answer>{item['answer']}</answer>"
                    )
                elif quality == 'partial':
                    wrong = str(int(item['answer']) + random.randint(-5, 5))
                    item['completion'] = (
                        f"<reasoning>Working through this problem...</reasoning>"
                        f"<answer>{wrong}</answer>"
                    )
                else:
                    item['completion'] = f"I think the answer is {random.randint(0, 100)}"

            return dataset

        except FileNotFoundError as e:
            print(f"\nError: {e}")
            print("\nGSM8K dataset not found. Please download it from:")
            print("https://www.kaggle.com/datasets/thedevastator/grade-school-math-8k-q-a")
            print(f"and place the CSV files in {args.data_dir}/")
            sys.exit(1)

    elif args.dataset == 'openrubrics':
        print(f"Loading OpenRubrics dataset ({args.split} split)...")
        dataset = load_openrubrics_dataset(
            split=args.split,
            max_examples=args.samples,
            seed=args.seed
        )

        if not dataset:
            print("\nError: Could not load OpenRubrics dataset")
            print("Make sure you have internet connection and the 'datasets' package installed")
            sys.exit(1)

        print(f"Loaded {len(dataset)} examples")

        # Use reference responses as completions
        for item in dataset:
            item['completion'] = item.get('reference_response', '')

        return dataset

    else:
        print("Error: No dataset specified")
        sys.exit(1)


def main():
    """Main entry point."""
    args = parse_args()

    # Handle --list-methods
    if args.list_methods:
        list_methods()
        return

    # Validate arguments
    if not args.dataset:
        print("Error: --dataset is required (unless using --list-methods)")
        print("Use --help for usage information")
        sys.exit(1)

    print("\n" + "=" * 70)
    print("GRADING METHODOLOGY COMPARISON")
    print("=" * 70)

    # Load dataset
    dataset = load_dataset(args)

    # Determine which methods to use
    if args.methods:
        method_names = args.methods
        print(f"\nComparing methods: {', '.join(method_names)}")
    else:
        # Use all applicable methods based on dataset
        all_methods = get_all_grading_methods()
        method_names = []

        for name, metadata in all_methods.items():
            # Check if method is applicable
            if metadata['requires_ground_truth'] and 'answer' not in dataset[0]:
                continue
            if metadata['requires_rubric'] and 'rubric' not in dataset[0]:
                continue
            method_names.append(name)

        print(f"\nComparing all applicable methods: {', '.join(method_names)}")

    # Run analysis
    print(f"\nRunning analysis...")
    print(f"Output directory: {args.output}")

    try:
        results = analyze_dataset(
            dataset=dataset,
            method_names=method_names,
            output_dir=args.output,
            generate_plots=args.plot
        )

        # Print summary
        print("\n" + "=" * 70)
        print("RESULTS SUMMARY")
        print("=" * 70)

        print(f"\nAnalyzed {len(dataset)} examples")
        print(f"Compared {len(method_names)} grading methods")

        print("\nStatistical Summary:")
        for method in results.methods:
            stats = results.statistics[method]
            print(f"\n{method}:")
            print(f"  Mean: {stats['mean']:.3f}")
            print(f"  Median: {stats['median']:.3f}")
            print(f"  Std: {stats['std']:.3f}")
            print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")

        if len(results.methods) > 1:
            print("\nTop Correlations (Pearson):")
            pearson_corrs = {
                k: v for k, v in results.correlations.items()
                if 'pearson' in k
            }
            for pair, corr in sorted(pearson_corrs.items(),
                                    key=lambda x: abs(x[1]),
                                    reverse=True)[:5]:
                methods = pair.replace('_pearson', '').replace('_vs_', ' vs ')
                print(f"  {methods}: {corr:.3f}")

        print("\n" + "=" * 70)
        print(f"Full results saved to: {args.output}/")
        print("=" * 70)

    except Exception as e:
        print(f"\nError during analysis: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()
