#!/usr/bin/env python3
"""Quick test to verify reward comparison framework works correctly."""

import sys
from pathlib import Path

# Add TunRex to path
sys.path.insert(0, str(Path(__file__).parent / "TunRex" / "src"))

# Import directly from reward_comparison to avoid dependency issues
from tunrex.reward_comparison.evaluators import (
    ProgrammaticReward,
    RubricReward,
    RubricCriterion,
)
from tunrex.reward_comparison.comparison import RewardComparison
from tunrex.reward_comparison.analysis import RewardAnalyzer


def test_basic_functionality():
    """Test basic reward comparison functionality."""
    print("Testing Reward Comparison Framework...")
    print("-" * 60)

    # Sample data
    prompts = ["Q1", "Q2", "Q3"]
    completions = [
        "Answer 1 with reasoning",
        "Short answer",
        "Detailed answer with multiple steps and reasoning"
    ]

    # Define simple rewards
    def length_reward(prompts, completions, **kwargs):
        return [len(c) / 10.0 for c in completions]

    def word_count_reward(prompts, completions, **kwargs):
        return [float(len(c.split())) for c in completions]

    # Wrap as ProgrammaticReward
    length_eval = ProgrammaticReward(length_reward, name="Length")
    word_count_eval = ProgrammaticReward(word_count_reward, name="WordCount")

    # Create rubric
    detail_criterion = RubricCriterion(
        name="Detail",
        description="Checks for detailed response",
        max_score=5.0,
        evaluator=lambda p, c, **kw: 5.0 if len(c) > 30 else 2.0 if len(c) > 15 else 0.0
    )

    rubric = RubricReward(
        criteria=[detail_criterion],
        name="DetailRubric",
        aggregation="sum"
    )

    # Test 1: Individual evaluation
    print("\n1. Testing individual evaluators...")
    result1 = length_eval.evaluate(prompts, completions)
    print(f"   ✓ Length scores: {result1.scores}")
    assert len(result1.scores) == 3

    result2 = word_count_eval.evaluate(prompts, completions)
    print(f"   ✓ WordCount scores: {result2.scores}")
    assert len(result2.scores) == 3

    result3 = rubric.evaluate(prompts, completions)
    print(f"   ✓ Rubric scores: {result3.scores}")
    assert len(result3.scores) == 3

    # Test 2: Comparison
    print("\n2. Testing reward comparison...")
    comparison = RewardComparison([length_eval, word_count_eval, rubric])
    comp_result = comparison.evaluate(prompts, completions)
    print(f"   ✓ Comparison result: {len(comp_result.evaluators)} evaluators")
    assert len(comp_result.evaluators) == 3

    # Test 3: Summary
    print("\n3. Testing summary generation...")
    summary = comparison.summarize()
    print("   ✓ Summary generated")
    assert "Length" in summary
    assert "WordCount" in summary
    assert "DetailRubric" in summary

    # Test 4: Analysis
    print("\n4. Testing analysis...")
    analyzer = RewardAnalyzer(comp_result)

    # Correlations
    corr = analyzer.compute_correlations()
    print(f"   ✓ Correlation matrix shape: {corr.pearson_correlation.shape}")
    assert corr.pearson_correlation.shape == (3, 3)

    # Agreement
    agreement = analyzer.compute_agreement()
    print(f"   ✓ Agreement analysis: {len(agreement.pairwise_agreement)} pairs")

    # Report
    report = analyzer.generate_report()
    print("   ✓ Report generated")
    assert "REWARD COMPARISON ANALYSIS REPORT" in report

    # Test 5: Disagreements
    print("\n5. Testing disagreement detection...")
    disagreements = comparison.find_disagreements(threshold=1.0)
    print(f"   ✓ Found {len(disagreements)} disagreements")

    # Test 6: Export
    print("\n6. Testing CSV export...")
    import tempfile
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
        csv_path = f.name

    comparison.export_to_csv(csv_path, include_text=True)
    print(f"   ✓ Exported to {csv_path}")

    import os
    os.unlink(csv_path)

    print("\n" + "=" * 60)
    print("✓ All tests passed!")
    print("=" * 60)


if __name__ == "__main__":
    try:
        test_basic_functionality()
        sys.exit(0)
    except Exception as e:
        print(f"\n✗ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
