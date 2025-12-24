"""
Rubric Development Module for GRPO Training.

This module provides a Python API for creating, loading, modifying,
and testing rubrics in a GRPO training context.

Example Usage:
    from src.rubrics import (
        Rubric, RubricSet, Criterion,
        RubricBuilder,
        load_rubricset_from_yaml,
        load_rubrics_from_openrubrics,
        create_grpo_reward_function,
        test_rubric_with_dataset,
    )

    # Create a rubric programmatically
    rubric = (
        RubricBuilder("math_reasoning", "Evaluates math problem solutions")
        .with_criterion("accuracy", "Correct numerical answer", weight=2.0)
        .with_criterion("reasoning", "Clear step-by-step explanation")
        .with_question_types("math")
        .build()
    )

    # Load from YAML
    rubricset = load_rubricset_from_yaml("rubrics/custom.yaml")

    # Test the rubric
    results = test_rubric_with_dataset(rubricset, test_data)
    print(results.summary())

    # Use in GRPO training
    reward_fn = create_grpo_reward_function(rubricset)
"""

from .models import Rubric, RubricSet, Criterion

from .loaders import (
    load_rubric_from_yaml,
    load_rubricset_from_yaml,
    load_rubrics_from_directory,
    load_rubrics_from_openrubrics,
)

from .writers import (
    save_rubric_to_yaml,
    save_rubricset_to_yaml,
)

from .modifiers import (
    RubricBuilder,
    clone_rubric,
    merge_rubrics,
    apply_transform,
    adjust_weights,
    add_keywords_to_criterion,
    scale_weights,
    normalize_weights,
)

from .scoring import (
    create_rubric_scorer,
    create_rubricset_scorer,
    create_weighted_rubric_scorer,
    create_multi_rubric_scorer,
    rubric_reward_adapter,
)

from .testing import (
    RubricTestConfig,
    RubricTestResult,
    test_rubric_with_dataset,
    create_grpo_reward_function,
    quick_test_rubric,
    compare_rubrics,
    print_comparison,
)

__all__ = [
    # Core Models
    "Rubric",
    "RubricSet",
    "Criterion",
    # Loaders
    "load_rubric_from_yaml",
    "load_rubricset_from_yaml",
    "load_rubrics_from_directory",
    "load_rubrics_from_openrubrics",
    # Writers
    "save_rubric_to_yaml",
    "save_rubricset_to_yaml",
    # Modifiers
    "RubricBuilder",
    "clone_rubric",
    "merge_rubrics",
    "apply_transform",
    "adjust_weights",
    "add_keywords_to_criterion",
    "scale_weights",
    "normalize_weights",
    # Scoring
    "create_rubric_scorer",
    "create_rubricset_scorer",
    "create_weighted_rubric_scorer",
    "create_multi_rubric_scorer",
    "rubric_reward_adapter",
    # Testing
    "RubricTestConfig",
    "RubricTestResult",
    "test_rubric_with_dataset",
    "create_grpo_reward_function",
    "quick_test_rubric",
    "compare_rubrics",
    "print_comparison",
]
