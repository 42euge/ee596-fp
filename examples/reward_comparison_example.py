#!/usr/bin/env python3
"""Example: Comparing different reward methodologies.

This script demonstrates how to use the reward comparison framework to:
1. Define multiple reward evaluators (programmatic, rubric-based, preference models)
2. Run them on the same dataset
3. Analyze correlations and agreements
4. Visualize the results
"""

import sys
from pathlib import Path

# Add project to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root / "TunRex" / "src"))

from tunrex.datasets.rewards import (
    match_format_exactly,
    match_format_approximately,
    check_answer,
    check_numbers,
)
from tunrex.datasets.config import (
    reasoning_start,
    reasoning_end,
    solution_start,
    solution_end,
)
from tunrex.reward_comparison import (
    ProgrammaticReward,
    RubricReward,
    RubricCriterion,
    RewardComparison,
    RewardAnalyzer,
    create_format_reward,
    create_length_reward,
)


def main():
    print("=" * 80)
    print("Reward Comparison Example")
    print("=" * 80)
    print()

    # ========================================================================
    # Step 1: Create sample data
    # ========================================================================
    print("[1/5] Creating sample data...")

    prompts = [
        "What is 2 + 2?",
        "Calculate 15 * 3",
        "Solve: 100 - 37",
        "What is 50 / 2?",
    ]

    completions = [
        f"{reasoning_start}2 + 2 equals 4{reasoning_end} {solution_start}4{solution_end}",
        f"{reasoning_start}15 times 3 is 45{reasoning_end} {solution_start}45{solution_end}",
        # Bad format - missing tags
        "The answer is 63",
        f"{reasoning_start}50 divided by 2{reasoning_end} {solution_start}25{solution_end}",
    ]

    answers = ["4", "45", "63", "25"]

    print(f"  Created {len(prompts)} prompts and completions")
    print()

    # ========================================================================
    # Step 2: Define reward evaluators
    # ========================================================================
    print("[2/5] Defining reward evaluators...")

    # Programmatic rewards from existing functions
    format_exact = ProgrammaticReward(
        match_format_exactly,
        name="FormatExact",
        description="Checks exact format match"
    )

    format_approx = ProgrammaticReward(
        match_format_approximately,
        name="FormatApprox",
        description="Checks approximate format match with partial credit"
    )

    answer_check = ProgrammaticReward(
        check_answer,
        name="AnswerCheck",
        description="Checks if answer is correct"
    )

    # Rubric-based reward
    format_criterion = RubricCriterion(
        name="Format",
        description="Has correct reasoning and answer tags",
        max_score=3.0,
        evaluator=lambda p, c, **kw: (
            3.0 if reasoning_start in c and reasoning_end in c and
            solution_start in c and solution_end in c else 0.0
        )
    )

    length_criterion = RubricCriterion(
        name="Length",
        description="Response is appropriate length",
        max_score=1.0,
        evaluator=lambda p, c, **kw: 1.0 if 50 <= len(c) <= 200 else 0.5
    )

    completeness_criterion = RubricCriterion(
        name="Completeness",
        description="Has both reasoning and answer",
        max_score=2.0,
        evaluator=lambda p, c, **kw: (
            2.0 if len(c) > 30 and solution_start in c else
            1.0 if solution_start in c else 0.0
        )
    )

    rubric = RubricReward(
        criteria=[format_criterion, length_criterion, completeness_criterion],
        name="DetailedRubric",
        aggregation="sum"
    )

    # Helper-created rewards
    length_reward = create_length_reward(
        min_length=30,
        max_length=300,
        max_score=1.0,
        name="LengthReward"
    )

    print(f"  Created {5} reward evaluators:")
    print(f"    - FormatExact (programmatic)")
    print(f"    - FormatApprox (programmatic)")
    print(f"    - AnswerCheck (programmatic)")
    print(f"    - DetailedRubric (rubric-based, 3 criteria)")
    print(f"    - LengthReward (programmatic)")
    print()

    # ========================================================================
    # Step 3: Run comparison
    # ========================================================================
    print("[3/5] Running reward comparison...")
    print()

    comparison = RewardComparison(
        evaluators=[format_exact, format_approx, answer_check, rubric, length_reward],
        name="Math QA Reward Comparison"
    )

    result = comparison.evaluate(
        prompts=prompts,
        completions=completions,
        answer=answers,
        question=prompts
    )

    print()
    print(comparison.summarize(result))
    print()

    # ========================================================================
    # Step 4: Analyze results
    # ========================================================================
    print("[4/5] Analyzing results...")
    print()

    analyzer = RewardAnalyzer(result)

    # Generate comprehensive report
    report = analyzer.generate_report(
        include_correlations=True,
        include_agreement=True,
        agreement_threshold=2.0,  # Binary threshold
        top_k=[2, 3]  # Top-k agreement for k=2,3
    )

    print(report)
    print()

    # Find disagreements
    print("-" * 80)
    print("Finding disagreements (threshold=2.0)...")
    disagreements = comparison.find_disagreements(threshold=2.0)

    if disagreements:
        print(f"\nFound {len(disagreements)} disagreement(s):")
        for i, dis in enumerate(disagreements[:3], 1):  # Show top 3
            print(f"\n  Disagreement #{i}:")
            print(f"    Sample: {dis['sample_idx']}")
            print(f"    Completion: {dis['completion'][:60]}...")
            print(f"    {dis['evaluator1']}: {dis['score1']:.2f}")
            print(f"    {dis['evaluator2']}: {dis['score2']:.2f}")
            print(f"    Difference: {dis['difference']:.2f}")
    else:
        print("  No significant disagreements found")

    print()

    # ========================================================================
    # Step 5: Export results
    # ========================================================================
    print("[5/5] Exporting results...")

    # Export to CSV
    output_dir = project_root / "outputs"
    output_dir.mkdir(exist_ok=True)

    csv_path = output_dir / "reward_comparison.csv"
    comparison.export_to_csv(str(csv_path), include_text=True)

    # Try to create visualizations (requires matplotlib)
    try:
        # Score distributions
        dist_path = output_dir / "score_distributions.png"
        analyzer.plot_score_distributions(str(dist_path))

        # Correlation heatmap
        heatmap_path = output_dir / "correlation_heatmap.png"
        analyzer.plot_correlation_heatmap(str(heatmap_path), method="pearson")

        # Pairwise comparison
        comparison_path = output_dir / "format_vs_answer.png"
        analyzer.plot_score_comparison(
            "FormatExact",
            "AnswerCheck",
            str(comparison_path)
        )

        print(f"  Visualizations saved to {output_dir}/")
    except ImportError:
        print("  Skipping visualizations (matplotlib not installed)")

    print()
    print("=" * 80)
    print("✓ Reward comparison complete!")
    print("=" * 80)


if __name__ == "__main__":
    main()
