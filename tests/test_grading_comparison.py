#!/usr/bin/env python3
"""
Tests for grading comparison framework.
"""

import sys
from pathlib import Path

# Add src to path
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

# Import modules directly to avoid package initialization issues
import grading_comparison
import grading_registry
import utils

GradingComparator = grading_comparison.GradingComparator
GradingMethod = grading_comparison.GradingMethod
ComparisonResults = grading_comparison.ComparisonResults
get_all_grading_methods = grading_registry.get_all_grading_methods
create_standard_comparator = grading_registry.create_standard_comparator


def test_get_methods():
    """Test getting available methods."""
    print("Test 1: Get available methods")
    methods = get_all_grading_methods()
    assert len(methods) > 0, "Should have at least one method"
    print(f"  Found {len(methods)} grading methods:")
    for name in methods.keys():
        print(f"    - {name}")
    print("  ✓ Test 1 passed\n")


def test_create_comparator():
    """Test creating a standard comparator."""
    print("Test 2: Create standard comparator")
    comparator = create_standard_comparator()
    assert len(comparator.methods) > 0, "Should have registered methods"
    print(f"  Comparator has {len(comparator.methods)} registered methods")
    print("  ✓ Test 2 passed\n")


def test_basic_comparison():
    """Test basic comparison functionality."""
    print("Test 3: Basic comparison")

    prompts = ['What is 2+2?', 'What is 5+3?', 'Calculate 10-4']
    completions = [
        '<reasoning>2+2 equals 4</reasoning><answer>4</answer>',
        'The answer is 8',  # Missing format tags
        '<reasoning>10-4=6</reasoning><answer>6</answer>'
    ]
    answers = ['4', '8', '6']

    comparator = create_standard_comparator()
    results = comparator.compare(
        prompts=prompts,
        completions=completions,
        method_names=['format_reward', 'accuracy_reward'],
        answers=answers
    )

    # Verify results
    assert len(results.methods) == 2, "Should have 2 methods"
    assert len(results.examples) == 3, "Should have 3 examples"
    assert 'format_reward' in results.scores, "Should have format_reward scores"
    assert 'accuracy_reward' in results.scores, "Should have accuracy_reward scores"

    print(f"  Compared {len(results.methods)} methods on {len(prompts)} examples")
    for method in results.methods:
        stats = results.statistics[method]
        scores = results.scores[method]
        print(f"  {method}:")
        print(f"    scores: {[f'{s:.2f}' for s in scores]}")
        print(f"    mean={stats['mean']:.2f}, std={stats['std']:.2f}")

    print("  ✓ Test 3 passed\n")


def test_correlations():
    """Test correlation calculation."""
    print("Test 4: Correlation calculation")

    prompts = ['Q' + str(i) for i in range(10)]
    completions = [
        '<reasoning>R</reasoning><answer>A</answer>' if i % 2 == 0 else 'Wrong format'
        for i in range(10)
    ]
    answers = ['A' for _ in range(10)]

    comparator = create_standard_comparator()
    results = comparator.compare(
        prompts=prompts,
        completions=completions,
        method_names=['format_reward', 'accuracy_reward'],
        answers=answers
    )

    # Should have correlation results
    assert len(results.correlations) > 0, "Should have correlations"
    print(f"  Calculated {len(results.correlations)} correlation metrics")

    for key, value in results.correlations.items():
        print(f"    {key}: {value:.3f}")

    print("  ✓ Test 4 passed\n")


def test_disagreements():
    """Test finding disagreements."""
    print("Test 5: Find disagreements")

    # Create examples where methods should disagree
    prompts = [
        'What is 2+2?',  # Good format, correct answer
        'What is 5+3?',  # Bad format, correct answer
        'What is 10-4?',  # Good format, wrong answer
    ]
    completions = [
        '<reasoning>2+2=4</reasoning><answer>4</answer>',
        '8',  # Correct but no format
        '<reasoning>10-4=10</reasoning><answer>10</answer>',  # Format but wrong
    ]
    answers = ['4', '8', '6']

    comparator = create_standard_comparator()
    results = comparator.compare(
        prompts=prompts,
        completions=completions,
        method_names=['format_reward', 'accuracy_reward'],
        answers=answers
    )

    disagreements = comparator.find_disagreements(
        'format_reward', 'accuracy_reward',
        top_k=2,
        normalize=True
    )

    assert len(disagreements) <= 2, "Should return at most top_k disagreements"
    print(f"  Found {len(disagreements)} top disagreements")

    for i, ex in enumerate(disagreements, 1):
        print(f"  {i}. Difference: {ex['difference']:.3f}")

    print("  ✓ Test 5 passed\n")


def test_behavior_analysis():
    """Test behavior analysis."""
    print("Test 6: Behavior analysis")

    prompts = ['Q' + str(i) for i in range(20)]
    completions = [
        '<reasoning>' + 'x' * (i * 10) + '</reasoning><answer>A</answer>'
        for i in range(20)
    ]
    answers = ['A' for _ in range(20)]

    comparator = create_standard_comparator()
    results = comparator.compare(
        prompts=prompts,
        completions=completions,
        method_names=['format_reward'],
        answers=answers
    )

    behavior = comparator.analyze_behavior_effects('format_reward')

    assert len(behavior) > 0, "Should have behavior analysis"
    print(f"  Analyzed behavior across {len(behavior)} score groups")

    for group_name, analysis in behavior.items():
        if analysis['count'] > 0:
            print(f"  {group_name}: {analysis['count']} examples, "
                  f"avg_length={analysis['avg_length']:.1f}")

    print("  ✓ Test 6 passed\n")


def test_save_load():
    """Test saving and loading results."""
    print("Test 7: Save and load results")

    import tempfile
    import os

    prompts = ['Q1', 'Q2']
    completions = ['<reasoning>R</reasoning><answer>A</answer>'] * 2
    answers = ['A', 'A']

    comparator = create_standard_comparator()
    results = comparator.compare(
        prompts=prompts,
        completions=completions,
        method_names=['format_reward'],
        answers=answers
    )

    # Save
    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name

    try:
        results.save(temp_path)
        print(f"  Saved results to {temp_path}")

        # Load
        loaded_results = ComparisonResults.load(temp_path)
        print(f"  Loaded results from {temp_path}")

        assert loaded_results.methods == results.methods, "Methods should match"
        assert len(loaded_results.scores) == len(results.scores), "Scores should match"

        print("  ✓ Test 7 passed\n")

    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)


def run_all_tests():
    """Run all tests."""
    print("=" * 70)
    print("GRADING COMPARISON FRAMEWORK TESTS")
    print("=" * 70)
    print()

    tests = [
        test_get_methods,
        test_create_comparator,
        test_basic_comparison,
        test_correlations,
        test_disagreements,
        test_behavior_analysis,
        test_save_load,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            test()
            passed += 1
        except Exception as e:
            print(f"  ✗ Test failed: {e}\n")
            import traceback
            traceback.print_exc()
            failed += 1

    print("=" * 70)
    print(f"RESULTS: {passed} passed, {failed} failed")
    print("=" * 70)

    return failed == 0


if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)
