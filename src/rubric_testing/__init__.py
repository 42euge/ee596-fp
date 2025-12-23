"""
Rubric Testing Infrastructure

This module provides tools for rapidly testing new rubric designs against small models
before scaling up to full training runs.

Key Components:
- RubricDesigner: Base class for defining custom rubric scoring functions
- RubricEvaluator: Quick evaluation engine for testing rubrics on small samples
- RubricComparator: Tools for comparing multiple rubric designs
- RubricReporter: Generate performance reports and visualizations
"""

from .designer import (
    RubricDesigner,
    BaseRubric,
    CompositeRubric,
    WeightedRubric,
    create_rubric,
)
from .evaluator import (
    RubricEvaluator,
    EvaluationConfig,
    EvaluationResult,
)
from .comparator import (
    RubricComparator,
    ComparisonResult,
    compare_rubrics,
)
from .reporter import (
    RubricReporter,
    generate_report,
    plot_rubric_comparison,
)

__all__ = [
    # Designer
    "RubricDesigner",
    "BaseRubric",
    "CompositeRubric",
    "WeightedRubric",
    "create_rubric",
    # Evaluator
    "RubricEvaluator",
    "EvaluationConfig",
    "EvaluationResult",
    # Comparator
    "RubricComparator",
    "ComparisonResult",
    "compare_rubrics",
    # Reporter
    "RubricReporter",
    "generate_report",
    "plot_rubric_comparison",
]
