#!/usr/bin/env python3
"""
Example: Comparing Different Grading Methodologies

This script demonstrates how to use the grading comparison framework to:
1. Compare multiple grading methods on the same dataset
2. Analyze their statistical properties
3. Identify where methods agree/disagree
4. Understand effects on model behavior
5. Generate visualizations and reports
"""

import sys
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from grading_registry import create_standard_comparator, analyze_dataset
from utils import load_gsm8k_dataset, load_openrubrics_dataset


def example_1_basic_comparison():
    """Basic comparison of format and accuracy rewards."""
    print("=" * 70)
    print("Example 1: Basic Comparison of Format and Accuracy Rewards")
    print("=" * 70)

    # Sample data
    prompts = [
        "What is 5 + 3?",
        "Calculate 10 - 4",
        "What is 2 * 6?",
    ]

    completions = [
        "<reasoning>Adding 5 and 3 gives us 8</reasoning><answer>8</answer>",
        "<reasoning>Subtracting 4 from 10</reasoning><answer>6</answer>",
        "The answer is 12",  # Missing format tags
    ]

    answers = ["8", "6", "12"]

    # Create comparator and run comparison
    comparator = create_standard_comparator()
    results = comparator.compare(
        prompts=prompts,
        completions=completions,
        method_names=['format_reward', 'accuracy_reward'],
        answers=answers
    )

    # Print results
    print("\nScores by method:")
    for method in results.methods:
        print(f"\n{method}:")
        for i, score in enumerate(results.scores[method]):
            print(f"  Example {i+1}: {score:.2f}")

    print("\nStatistics:")
    for method in results.methods:
        stats = results.statistics[method]
        print(f"\n{method}:")
        print(f"  Mean: {stats['mean']:.3f}")
        print(f"  Std:  {stats['std']:.3f}")
        print(f"  Range: [{stats['min']:.3f}, {stats['max']:.3f}]")

    print("\nCorrelations:")
    for pair, corr in results.correlations.items():
        if 'pearson' in pair:
            print(f"  {pair}: {corr:.3f}")


def example_2_rubric_analysis():
    """Compare rubric-based vs accuracy-based grading."""
    print("\n" + "=" * 70)
    print("Example 2: Rubric-Based vs Accuracy-Based Grading")
    print("=" * 70)

    # Load some data from OpenRubrics
    print("\nLoading OpenRubrics dataset (this may take a moment)...")
    try:
        dataset = load_openrubrics_dataset(split="train", max_examples=50, seed=42)

        if not dataset:
            print("OpenRubrics dataset not available, skipping this example")
            return

        print(f"Loaded {len(dataset)} examples")

        # Prepare data
        prompts = [item['question'] for item in dataset]
        completions = [item['reference_response'] for item in dataset]
        rubrics = [item['rubric'] for item in dataset]
        target_scores = [item['target_score'] for item in dataset]

        # Compare rubric vs format methods
        comparator = create_standard_comparator()
        results = comparator.compare(
            prompts=prompts,
            completions=completions,
            method_names=['format_reward', 'rubric_reward'],
            rubrics=rubrics,
            reference_responses=completions,
            target_scores=target_scores
        )

        print("\nStatistical Summary:")
        for method in results.methods:
            stats = results.statistics[method]
            print(f"\n{method}:")
            print(f"  Mean ± Std: {stats['mean']:.3f} ± {stats['std']:.3f}")
            print(f"  Median: {stats['median']:.3f}")
            print(f"  IQR: [{stats['q25']:.3f}, {stats['q75']:.3f}]")

        # Find disagreements
        print("\nTop 5 disagreements between methods:")
        disagreements = comparator.find_disagreements(
            'format_reward', 'rubric_reward', top_k=5, normalize=True
        )
        for i, ex in enumerate(disagreements, 1):
            print(f"\n{i}. Difference: {ex['difference']:.3f}")
            print(f"   Format score: {ex['format_reward_score']:.3f}")
            print(f"   Rubric score: {ex['rubric_reward_score']:.3f}")
            print(f"   Question: {ex['prompt'][:100]}...")

    except Exception as e:
        print(f"Error loading OpenRubrics: {e}")
        print("Skipping this example")


def example_3_behavior_analysis():
    """Analyze how grading methods affect model behavior."""
    print("\n" + "=" * 70)
    print("Example 3: Analyzing Effects on Model Behavior")
    print("=" * 70)

    # Load GSM8K data
    print("\nLoading GSM8K dataset...")
    try:
        dataset = load_gsm8k_dataset(split="train", max_examples=100, seed=42)
        print(f"Loaded {len(dataset)} examples")

        # Create synthetic completions with varying quality
        import random
        prompts = [item['question'] for item in dataset]
        answers = [item['answer'] for item in dataset]

        completions = []
        for i, (prompt, answer) in enumerate(zip(prompts, answers)):
            # Vary quality randomly
            quality = random.choice(['good', 'partial', 'poor'])

            if quality == 'good':
                comp = f"<reasoning>Let me solve this step by step. The answer is {answer}</reasoning><answer>{answer}</answer>"
            elif quality == 'partial':
                comp = f"<reasoning>Solving...</reasoning><answer>{int(answer) + random.randint(-2, 2)}</answer>"
            else:
                comp = f"The answer might be {random.randint(0, 100)}"

            completions.append(comp)

        # Run comparison
        comparator = create_standard_comparator()
        results = comparator.compare(
            prompts=prompts,
            completions=completions,
            method_names=['format_reward', 'accuracy_reward'],
            answers=answers
        )

        # Analyze behavior effects
        print("\nBehavior analysis for 'format_reward':")
        behavior = comparator.analyze_behavior_effects('format_reward')

        for group_name, analysis in behavior.items():
            print(f"\n{group_name.upper()} scores:")
            print(f"  Count: {analysis['count']}")
            print(f"  Avg length: {analysis['avg_length']:.1f} chars")
            print(f"  Avg words: {analysis['avg_word_count']:.1f}")

        # Save results
        output_dir = Path(__file__).parent.parent / "results" / "grading_analysis"
        output_dir.mkdir(parents=True, exist_ok=True)

        results.save(str(output_dir / "behavior_analysis.json"))
        comparator.generate_report(str(output_dir / "behavior_report.md"))

        print(f"\nResults saved to {output_dir}/")

    except FileNotFoundError as e:
        print(f"GSM8K dataset not found: {e}")
        print("Please download the dataset first")
    except Exception as e:
        print(f"Error: {e}")


def example_4_full_analysis():
    """Run a full analysis on a dataset with all visualizations."""
    print("\n" + "=" * 70)
    print("Example 4: Full Dataset Analysis with Visualizations")
    print("=" * 70)

    try:
        # Load dataset
        dataset = load_gsm8k_dataset(split="train", max_examples=200, seed=42)
        print(f"Loaded {len(dataset)} examples")

        # Create synthetic completions
        import random
        for item in dataset:
            quality = random.choice(['good', 'good', 'partial', 'poor'])  # Bias toward good

            if quality == 'good':
                item['completion'] = f"<reasoning>Step-by-step solution</reasoning><answer>{item['answer']}</answer>"
            elif quality == 'partial':
                wrong_answer = str(int(item['answer']) + random.randint(-5, 5))
                item['completion'] = f"<reasoning>Attempt</reasoning><answer>{wrong_answer}</answer>"
            else:
                item['completion'] = f"I think the answer is {random.randint(0, 100)}"

        # Run full analysis
        output_dir = Path(__file__).parent.parent / "results" / "full_grading_analysis"
        print(f"\nRunning full analysis (this may take a moment)...")

        results = analyze_dataset(
            dataset=dataset,
            method_names=['format_reward', 'accuracy_reward'],
            output_dir=str(output_dir),
            generate_plots=True
        )

        print("\nAnalysis complete!")
        print(f"Check {output_dir}/ for:")
        print("  - comparison_results.json: Full numerical results")
        print("  - comparison_report.md: Human-readable report")
        print("  - distributions.png: Score distribution plots")
        print("  - correlations.png: Correlation heatmap")
        print("  - scatter_*.png: Pairwise comparison plots")

    except FileNotFoundError as e:
        print(f"Dataset not found: {e}")
        print("Please download the GSM8K dataset first")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def main():
    """Run all examples."""
    print("\n" + "=" * 70)
    print("GRADING METHODOLOGY COMPARISON EXAMPLES")
    print("=" * 70)

    examples = [
        ("Basic Comparison", example_1_basic_comparison),
        ("Rubric Analysis", example_2_rubric_analysis),
        ("Behavior Analysis", example_3_behavior_analysis),
        ("Full Analysis", example_4_full_analysis),
    ]

    print("\nAvailable examples:")
    for i, (name, _) in enumerate(examples, 1):
        print(f"  {i}. {name}")

    print("\nRunning all examples...\n")

    for name, example_func in examples:
        try:
            example_func()
        except KeyboardInterrupt:
            print("\n\nInterrupted by user")
            break
        except Exception as e:
            print(f"\nError in {name}: {e}")
            import traceback
            traceback.print_exc()
            continue

    print("\n" + "=" * 70)
    print("Examples complete!")
    print("=" * 70)


if __name__ == "__main__":
    main()
