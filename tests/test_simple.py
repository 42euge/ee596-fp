#!/usr/bin/env python3
"""
Simple standalone test for grading comparison framework.
"""

import sys
from pathlib import Path

# Add src to path
src_path = str(Path(__file__).parent.parent / "src")
sys.path.insert(0, src_path)

# Import just grading_comparison module directly
import grading_comparison


def simple_grading_function(prompts, completions, **kwargs):
    """A simple test grading function."""
    return [1.0 if len(c) > 10 else 0.0 for c in completions]


def test_basic_functionality():
    """Test basic functionality of the grading comparison framework."""
    print("Testing Grading Comparison Framework")
    print("=" * 70)

    # Test 1: Create comparator
    print("\n1. Creating comparator...")
    comparator = grading_comparison.GradingComparator()
    print("   ✓ Created successfully")

    # Test 2: Register a grading method
    print("\n2. Registering grading method...")
    comparator.register_method(
        name='simple_test',
        function=simple_grading_function,
        description='Simple test grading function',
        score_range=(0.0, 1.0)
    )
    assert 'simple_test' in comparator.methods
    print("   ✓ Registered successfully")

    # Test 3: Run comparison
    print("\n3. Running comparison...")
    prompts = ['Question 1', 'Question 2', 'Question 3']
    completions = [
        'Short answer',  # len = 12 -> score 1.0
        'A',             # len = 1 -> score 0.0
        'Medium length answer'  # len = 20 -> score 1.0
    ]

    results = comparator.compare(
        prompts=prompts,
        completions=completions,
        method_names=['simple_test']
    )

    assert len(results.scores['simple_test']) == 3
    print(f"   Scores: {results.scores['simple_test']}")
    print("   ✓ Comparison ran successfully")

    # Test 4: Check statistics
    print("\n4. Checking statistics...")
    stats = results.statistics['simple_test']
    print(f"   Mean: {stats['mean']:.2f}")
    print(f"   Std: {stats['std']:.2f}")
    print(f"   Min: {stats['min']:.2f}, Max: {stats['max']:.2f}")
    assert 'mean' in stats
    assert 'std' in stats
    print("   ✓ Statistics calculated")

    # Test 5: Save and load
    print("\n5. Testing save/load...")
    import tempfile
    import os

    with tempfile.NamedTemporaryFile(mode='w', delete=False, suffix='.json') as f:
        temp_path = f.name

    try:
        results.save(temp_path)
        loaded = grading_comparison.ComparisonResults.load(temp_path)
        assert loaded.methods == results.methods
        print(f"   Saved to: {temp_path}")
        print("   ✓ Save/load works")
    finally:
        if os.path.exists(temp_path):
            os.remove(temp_path)

    # Test 6: Test with multiple methods
    print("\n6. Testing with multiple methods...")

    def another_grading_function(prompts, completions, **kwargs):
        return [2.0 if 'answer' in c.lower() else 0.0 for c in completions]

    comparator.register_method(
        name='keyword_test',
        function=another_grading_function,
        description='Checks for keyword',
        score_range=(0.0, 2.0)
    )

    results2 = comparator.compare(
        prompts=prompts,
        completions=completions,
        method_names=['simple_test', 'keyword_test']
    )

    assert len(results2.methods) == 2
    assert len(results2.correlations) > 0
    print(f"   Methods: {results2.methods}")
    print(f"   Correlations: {len(results2.correlations)} calculated")
    print("   ✓ Multiple methods work")

    # Test 7: Find disagreements
    print("\n7. Testing disagreement analysis...")
    disagreements = comparator.find_disagreements(
        'simple_test', 'keyword_test',
        top_k=2,
        normalize=True
    )
    assert len(disagreements) <= 2
    print(f"   Found {len(disagreements)} top disagreements")
    print("   ✓ Disagreement analysis works")

    print("\n" + "=" * 70)
    print("All tests passed!")
    print("=" * 70)


if __name__ == "__main__":
    try:
        test_basic_functionality()
        sys.exit(0)
    except Exception as e:
        print(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
