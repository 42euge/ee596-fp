#!/usr/bin/env python3
"""Test script for the rubrics module.

Run this script to verify the rubrics module works correctly:
    python -m src.rubrics.test_rubrics
"""


def test_models():
    """Test core model classes."""
    from src.rubrics import Rubric, RubricSet, Criterion

    # Test Criterion
    criterion = Criterion(
        name="accuracy",
        description="Tests accuracy of the answer",
        weight=2.0,
        keywords=["correct", "accurate"],
    )
    assert criterion.name == "accuracy"
    assert criterion.weight == 2.0
    print("  Criterion: OK")

    # Test Rubric
    rubric = Rubric(
        name="test_rubric",
        description="A test rubric",
        criteria=[criterion],
        question_types=["math"],
    )
    assert rubric.name == "test_rubric"
    assert len(rubric.criteria) == 1
    assert "accuracy" in rubric.to_text()
    print("  Rubric: OK")

    # Test RubricSet
    rubricset = RubricSet(name="test_set", rubrics=[rubric])
    assert len(rubricset) == 1
    assert rubricset["test_rubric"].name == "test_rubric"
    print("  RubricSet: OK")

    return True


def test_builder():
    """Test RubricBuilder."""
    from src.rubrics import RubricBuilder

    rubric = (
        RubricBuilder("builder_test", "Tests the builder")
        .with_criterion("format", "Uses correct format", weight=1.0)
        .with_criterion("accuracy", "Correct answer", weight=2.0)
        .with_question_types("math", "algebra")
        .with_reference("<answer>42</answer>", target_score=10.0)
        .with_metadata(author="test")
        .build()
    )

    assert rubric.name == "builder_test"
    assert len(rubric.criteria) == 2
    assert rubric.question_types == ["math", "algebra"]
    assert rubric.target_score == 10.0
    assert rubric.metadata["author"] == "test"
    print("  RubricBuilder: OK")

    return True


def test_yaml_io():
    """Test YAML loading and saving."""
    import tempfile
    import os
    from src.rubrics import (
        load_rubricset_from_yaml,
        save_rubricset_to_yaml,
        RubricBuilder,
        RubricSet,
    )

    # Test loading example file
    rubricset = load_rubricset_from_yaml("rubrics/example_math.yaml")
    assert len(rubricset) == 3
    assert "step_by_step_reasoning" in rubricset
    print("  YAML loading: OK")

    # Test round-trip
    rubric = RubricBuilder("roundtrip", "Test roundtrip").with_criterion("test", "test").build()
    test_set = RubricSet(name="test", rubrics=[rubric])

    with tempfile.TemporaryDirectory() as tmpdir:
        path = os.path.join(tmpdir, "test.yaml")
        save_rubricset_to_yaml(test_set, path)
        loaded = load_rubricset_from_yaml(path)
        assert len(loaded) == 1
        assert loaded[0].name == "roundtrip"
    print("  YAML round-trip: OK")

    return True


def test_modifiers():
    """Test modification utilities."""
    from src.rubrics import (
        RubricBuilder,
        clone_rubric,
        merge_rubrics,
        adjust_weights,
        normalize_weights,
    )

    rubric1 = (
        RubricBuilder("r1", "Rubric 1")
        .with_criterion("format", "Format", weight=1.0)
        .build()
    )
    rubric2 = (
        RubricBuilder("r2", "Rubric 2")
        .with_criterion("accuracy", "Accuracy", weight=2.0)
        .build()
    )

    # Test clone
    cloned = clone_rubric(rubric1, "cloned")
    assert cloned.name == "cloned"
    assert cloned.id != rubric1.id
    print("  clone_rubric: OK")

    # Test merge
    merged = merge_rubrics([rubric1, rubric2], "merged")
    assert len(merged.criteria) == 2
    print("  merge_rubrics: OK")

    # Test adjust_weights
    adjusted = adjust_weights(merged, {"format": 3.0})
    assert adjusted.get_criterion("format").weight == 3.0
    print("  adjust_weights: OK")

    # Test normalize_weights
    normalized = normalize_weights(merged, target_sum=1.0)
    total = sum(c.weight for c in normalized.criteria)
    assert abs(total - 1.0) < 0.01
    print("  normalize_weights: OK")

    return True


def test_scoring():
    """Test scoring functions (requires src.utils)."""
    from src.rubrics import (
        RubricBuilder,
        create_rubric_scorer,
        create_weighted_rubric_scorer,
    )

    rubric = (
        RubricBuilder("score_test", "Test scoring")
        .with_criterion("format", "Uses tags", keywords=["<reasoning>", "<answer>"])
        .with_criterion("steps", "Shows steps", keywords=["step", "first", "then"])
        .build()
    )

    scorer = create_rubric_scorer(rubric)
    scores = scorer(
        ["test prompt"],
        ["<reasoning>First step, then second step</reasoning><answer>42</answer>"],
    )
    assert len(scores) == 1
    assert scores[0] > 0
    print("  create_rubric_scorer: OK")

    weighted_scorer = create_weighted_rubric_scorer(rubric)
    weighted_scores = weighted_scorer(
        ["test"],
        ["<reasoning>step by step</reasoning><answer>done</answer>"],
    )
    assert len(weighted_scores) == 1
    print("  create_weighted_rubric_scorer: OK")

    return True


def test_evaluation_utils():
    """Test the evaluation utilities."""
    from src.rubrics import (
        RubricBuilder,
        RubricSet,
        evaluate_rubric_with_dataset,
        quick_test_rubric,
        RubricTestConfig,
    )

    rubric = (
        RubricBuilder("test_rubric", "For testing")
        .with_criterion("format", "Format check", keywords=["<answer>"])
        .build()
    )

    # Test quick_test_rubric
    result = quick_test_rubric(
        rubric,
        test_responses=[
            "<answer>correct</answer>",
            "no tags here",
        ],
    )
    assert "scores" in result
    assert len(result["scores"]) == 2
    assert result["scores"][0] > result["scores"][1]
    print("  quick_test_rubric: OK")

    # Test evaluate_rubric_with_dataset
    dataset = [
        {"question": "What is 2+2?", "response": "<answer>4</answer>"},
        {"question": "What is 3+3?", "response": "Six"},
    ]
    config = RubricTestConfig(num_examples=10, verbose=True)
    test_result = evaluate_rubric_with_dataset(rubric, dataset, config)
    assert test_result.rubric_name == "test_rubric"
    assert len(test_result.scores) == 2
    print("  evaluate_rubric_with_dataset: OK")

    return True


def main():
    """Run all tests."""
    print("\n=== Testing Rubrics Module ===\n")

    print("1. Testing core models...")
    test_models()

    print("\n2. Testing RubricBuilder...")
    test_builder()

    print("\n3. Testing YAML I/O...")
    test_yaml_io()

    print("\n4. Testing modifiers...")
    test_modifiers()

    print("\n5. Testing scoring functions...")
    test_scoring()

    print("\n6. Testing evaluation utilities...")
    test_evaluation_utils()

    print("\n=== All tests passed! ===\n")


if __name__ == "__main__":
    main()
