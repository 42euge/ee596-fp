#!/usr/bin/env python3
"""Example script for creating rubrics programmatically.

This script demonstrates how to:
1. Build rubrics using RubricBuilder
2. Test rubrics with sample responses
3. Save rubrics to YAML files
4. Load and modify existing rubrics

Run from the project root:
    python rubrics/create_rubric_example.py
"""

from src.rubrics import (
    RubricBuilder,
    RubricSet,
    Criterion,
    Rubric,
    save_rubricset_to_yaml,
    load_rubricset_from_yaml,
    quick_test_rubric,
    clone_rubric,
    adjust_weights,
    normalize_weights,
)


def create_math_reasoning_rubric() -> Rubric:
    """Create a comprehensive math reasoning rubric."""
    return (
        RubricBuilder("math_reasoning", "Evaluates mathematical problem-solving quality")
        .with_criterion(
            name="format_compliance",
            description="Uses required <reasoning> and <answer> tags correctly",
            weight=1.0,
            keywords=["<reasoning>", "</reasoning>", "<answer>", "</answer>"],
        )
        .with_criterion(
            name="step_by_step",
            description="Shows clear step-by-step reasoning process",
            weight=2.0,
            keywords=["first", "then", "next", "step", "calculate", "therefore"],
        )
        .with_criterion(
            name="mathematical_accuracy",
            description="Performs calculations correctly and arrives at right answer",
            weight=2.5,
            keywords=["equals", "result", "total", "sum", "product"],
        )
        .with_criterion(
            name="clarity",
            description="Explanation is clear and easy to follow",
            weight=1.5,
            keywords=["because", "since", "means", "shows"],
        )
        .with_question_types("math", "arithmetic", "algebra", "word_problem")
        .with_reference(
            response="""<reasoning>
Let me solve this step by step.
First, I identify that we need to calculate the total.
The calculation is: 5 + 10 = 15
Therefore, the answer is 15.
</reasoning>
<answer>15</answer>""",
            target_score=18.0,
        )
        .with_metadata(author="training_team", version="1.0")
        .build()
    )


def create_format_rubric() -> Rubric:
    """Create a simple format-checking rubric."""
    return (
        RubricBuilder("format_check", "Checks if response uses correct XML tags")
        .with_criterion(
            name="reasoning_tags",
            description="Contains <reasoning> tags with content",
            weight=1.0,
            keywords=["<reasoning>", "</reasoning>"],
        )
        .with_criterion(
            name="answer_tags",
            description="Contains <answer> tags with final answer",
            weight=1.0,
            keywords=["<answer>", "</answer>"],
        )
        .with_question_types("math", "reasoning", "general")
        .build()
    )


def test_rubric_quality(rubric: Rubric) -> None:
    """Test a rubric with sample responses to validate scoring."""
    print(f"\nTesting rubric: {rubric.name}")
    print("-" * 50)

    test_responses = [
        # Excellent response
        "<reasoning>First, I need to add the numbers. 2 + 2 equals 4. Therefore, the result is 4.</reasoning><answer>4</answer>",
        # Good response (less verbose)
        "<reasoning>2+2=4</reasoning><answer>4</answer>",
        # Poor response (no tags)
        "The answer is 4",
        # Partial response (missing answer tag)
        "<reasoning>I calculated 2+2=4</reasoning>",
    ]

    result = quick_test_rubric(rubric, test_responses)

    print("Scores:")
    for i, (response, score) in enumerate(zip(test_responses, result["scores"])):
        preview = response[:50] + "..." if len(response) > 50 else response
        print(f"  [{i+1}] {score:.2f} - {preview}")

    print(f"\nMean score: {result['mean']:.2f}")
    print(f"Rubric text preview:\n{result['rubric_text'][:200]}...")


def main():
    print("=" * 60)
    print("RUBRIC CREATION EXAMPLE")
    print("=" * 60)

    # 1. Create rubrics
    print("\n1. Creating rubrics...")
    math_rubric = create_math_reasoning_rubric()
    format_rubric = create_format_rubric()
    print(f"   Created: {math_rubric.name} ({len(math_rubric.criteria)} criteria)")
    print(f"   Created: {format_rubric.name} ({len(format_rubric.criteria)} criteria)")

    # 2. Test the rubrics
    print("\n2. Testing rubrics with sample responses...")
    test_rubric_quality(math_rubric)
    test_rubric_quality(format_rubric)

    # 3. Create a RubricSet
    print("\n3. Creating RubricSet...")
    rubricset = RubricSet(
        name="Training Rubrics",
        description="Rubrics for GRPO training evaluation",
        rubrics=[math_rubric, format_rubric],
        metadata={"version": "1.0", "created_by": "example_script"},
    )
    print(f"   RubricSet: {rubricset.name} with {len(rubricset)} rubrics")

    # 4. Save to YAML
    output_path = "rubrics/generated_rubrics.yaml"
    print(f"\n4. Saving to {output_path}...")
    save_rubricset_to_yaml(rubricset, output_path)
    print("   Saved successfully!")

    # 5. Load and modify
    print("\n5. Loading and modifying rubric...")
    loaded = load_rubricset_from_yaml(output_path)
    print(f"   Loaded: {loaded.name}")

    # Clone and modify the math rubric
    modified = clone_rubric(loaded["math_reasoning"], new_name="math_reasoning_v2")
    modified = adjust_weights(modified, {"step_by_step": 3.0})  # Increase weight
    modified = normalize_weights(modified, target_sum=10.0)  # Normalize to sum to 10

    print(f"   Modified rubric: {modified.name}")
    print("   New weights:")
    for c in modified.criteria:
        print(f"     - {c.name}: {c.weight:.2f}")

    # 6. Display rubric as text
    print("\n6. Rubric text representation:")
    print("-" * 40)
    print(math_rubric.to_text())

    print("\n" + "=" * 60)
    print("EXAMPLE COMPLETE")
    print("=" * 60)
    print(f"\nGenerated rubrics saved to: {output_path}")
    print("You can load them with: load_rubricset_from_yaml('rubrics/generated_rubrics.yaml')")


if __name__ == "__main__":
    main()
