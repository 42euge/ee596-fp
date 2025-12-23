"""Reward comparison framework for TunRex.

This module provides tools for comparing different reward methodologies:
- Preference models (e.g., GPT-4 as judge, trained reward models)
- Rubric-based evaluation
- Programmatic rewards

Usage:
    >>> from tunrex.reward_comparison import RewardComparison, ProgrammaticReward
    >>>
    >>> # Create reward evaluators
    >>> programmatic = ProgrammaticReward(check_answer)
    >>> rubric = RubricReward(rubric_config)
    >>>
    >>> # Compare them
    >>> comparison = RewardComparison([programmatic, rubric])
    >>> results = comparison.evaluate(prompts, completions, metadata)
    >>> report = comparison.generate_report()
"""

from tunrex.reward_comparison.base import (
    BaseReward,
    RewardResult,
    RewardMetadata,
)
from tunrex.reward_comparison.evaluators import (
    ProgrammaticReward,
    RubricReward,
    PreferenceModelReward,
)
from tunrex.reward_comparison.comparison import (
    RewardComparison,
    ComparisonResult,
)
from tunrex.reward_comparison.analysis import (
    RewardAnalyzer,
    CorrelationAnalysis,
    AgreementAnalysis,
)

__all__ = [
    # Base classes
    "BaseReward",
    "RewardResult",
    "RewardMetadata",
    # Evaluators
    "ProgrammaticReward",
    "RubricReward",
    "PreferenceModelReward",
    # Comparison
    "RewardComparison",
    "ComparisonResult",
    # Analysis
    "RewardAnalyzer",
    "CorrelationAnalysis",
    "AgreementAnalysis",
]
