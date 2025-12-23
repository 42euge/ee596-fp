"""Different reward evaluator implementations."""

import re
import time
from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Optional, Tuple

from tunrex.reward_comparison.base import BaseReward, RewardResult


class ProgrammaticReward(BaseReward):
    """Wrapper for programmatic reward functions.

    This evaluator wraps existing programmatic reward functions that compute
    rewards based on pattern matching, format checking, or numerical comparisons.

    Example:
        >>> def check_format(prompts, completions, **kwargs):
        ...     return [1.0 if "<answer>" in c else 0.0 for c in completions]
        >>>
        >>> reward = ProgrammaticReward(check_format, name="FormatChecker")
        >>> result = reward.evaluate(prompts, completions)
    """

    def __init__(
        self,
        func: Callable[[List[str], List[str], Any], List[float]],
        name: Optional[str] = None,
        description: str = ""
    ):
        """Initialize programmatic reward evaluator.

        Args:
            func: Function with signature (prompts, completions, **kwargs) -> List[float]
            name: Optional name for the evaluator
            description: Optional description of what the reward measures
        """
        super().__init__(name=name or func.__name__)
        self.func = func
        self.description = description

    def evaluate(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs
    ) -> RewardResult:
        """Evaluate rewards using the programmatic function."""
        start_time = time.time()
        scores = self.func(prompts, completions, **kwargs)
        elapsed_ms = (time.time() - start_time) * 1000

        metadata = self._create_metadata(
            elapsed_ms,
            extra_metadata={"description": self.description} if self.description else None
        )

        return RewardResult(scores=scores, metadata=metadata)

    def get_evaluator_type(self) -> str:
        return "programmatic"


@dataclass
class RubricCriterion:
    """A single criterion in a rubric."""

    name: str
    description: str
    max_score: float
    evaluator: Callable[[str, str, Any], float]  # (prompt, completion, **kwargs) -> score

    def evaluate(self, prompt: str, completion: str, **kwargs) -> Tuple[float, str]:
        """Evaluate this criterion.

        Returns:
            Tuple of (score, feedback)
        """
        score = self.evaluator(prompt, completion, **kwargs)
        score = max(0.0, min(self.max_score, score))  # Clamp to [0, max_score]
        return score, f"{self.name}: {score}/{self.max_score}"


class RubricReward(BaseReward):
    """Rubric-based reward evaluator.

    This evaluator computes rewards based on multiple criteria defined in a rubric.
    Each criterion is evaluated independently and scores are combined.

    Example:
        >>> format_criterion = RubricCriterion(
        ...     name="Format",
        ...     description="Checks if answer uses correct format",
        ...     max_score=3.0,
        ...     evaluator=lambda p, c, **kw: 3.0 if "<answer>" in c else 0.0
        ... )
        >>>
        >>> rubric = RubricReward([format_criterion], name="AnswerRubric")
        >>> result = rubric.evaluate(prompts, completions)
    """

    def __init__(
        self,
        criteria: List[RubricCriterion],
        name: str = "RubricReward",
        aggregation: str = "sum"  # "sum" or "mean"
    ):
        """Initialize rubric-based evaluator.

        Args:
            criteria: List of rubric criteria
            name: Name for this evaluator
            aggregation: How to combine criterion scores ("sum" or "mean")
        """
        super().__init__(name=name)
        self.criteria = criteria
        self.aggregation = aggregation

        if aggregation not in ("sum", "mean"):
            raise ValueError(f"Invalid aggregation: {aggregation}. Must be 'sum' or 'mean'")

    def evaluate(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs
    ) -> RewardResult:
        """Evaluate rewards using the rubric."""
        start_time = time.time()

        scores = []
        details = []

        for prompt, completion in zip(prompts, completions):
            criterion_scores = []
            criterion_feedback = []

            for criterion in self.criteria:
                score, feedback = criterion.evaluate(prompt, completion, **kwargs)
                criterion_scores.append(score)
                criterion_feedback.append(feedback)

            # Aggregate scores
            if self.aggregation == "sum":
                total_score = sum(criterion_scores)
            else:  # mean
                total_score = sum(criterion_scores) / len(criterion_scores)

            scores.append(total_score)
            details.append({
                "criterion_scores": criterion_scores,
                "criterion_feedback": criterion_feedback,
                "total_score": total_score
            })

        elapsed_ms = (time.time() - start_time) * 1000

        metadata = self._create_metadata(
            elapsed_ms,
            extra_metadata={
                "num_criteria": len(self.criteria),
                "aggregation": self.aggregation,
                "max_possible_score": sum(c.max_score for c in self.criteria)
                if self.aggregation == "sum" else
                sum(c.max_score for c in self.criteria) / len(self.criteria)
            }
        )

        return RewardResult(scores=scores, metadata=metadata, details=details)

    def get_evaluator_type(self) -> str:
        return "rubric"


class PreferenceModelReward(BaseReward):
    """Preference model-based reward evaluator.

    This evaluator uses a trained preference model or LLM-as-judge to score completions.
    Can be used with:
    - Trained reward models (e.g., from RLHF)
    - LLM-as-judge (e.g., GPT-4 scoring outputs)
    - External APIs

    Example:
        >>> def gpt4_judge(prompts, completions, **kwargs):
        ...     # Call GPT-4 API to score each completion
        ...     scores = []
        ...     for prompt, completion in zip(prompts, completions):
        ...         score = call_gpt4_api(prompt, completion)
        ...         scores.append(score)
        ...     return scores
        >>>
        >>> reward = PreferenceModelReward(
        ...     model_fn=gpt4_judge,
        ...     name="GPT4Judge"
        ... )
    """

    def __init__(
        self,
        model_fn: Callable[[List[str], List[str], Any], List[float]],
        name: str = "PreferenceModel",
        model_type: str = "llm_judge",
        description: str = ""
    ):
        """Initialize preference model evaluator.

        Args:
            model_fn: Function that scores completions using a model
            name: Name for this evaluator
            model_type: Type of model ("llm_judge", "reward_model", "api")
            description: Description of the model
        """
        super().__init__(name=name)
        self.model_fn = model_fn
        self.model_type = model_type
        self.description = description

    def evaluate(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs
    ) -> RewardResult:
        """Evaluate rewards using the preference model."""
        start_time = time.time()
        scores = self.model_fn(prompts, completions, **kwargs)
        elapsed_ms = (time.time() - start_time) * 1000

        metadata = self._create_metadata(
            elapsed_ms,
            extra_metadata={
                "model_type": self.model_type,
                "description": self.description
            }
        )

        return RewardResult(scores=scores, metadata=metadata)

    def get_evaluator_type(self) -> str:
        return "preference_model"


# Helper functions for creating common reward evaluators

def create_format_reward(
    pattern: str,
    score_on_match: float = 1.0,
    score_on_miss: float = 0.0,
    name: str = "FormatReward"
) -> ProgrammaticReward:
    """Create a reward that checks format using regex.

    Args:
        pattern: Regex pattern to match
        score_on_match: Score when pattern matches
        score_on_miss: Score when pattern doesn't match
        name: Name for the reward

    Returns:
        ProgrammaticReward instance
    """
    compiled_pattern = re.compile(pattern, re.MULTILINE | re.DOTALL)

    def check_format(prompts, completions, **kwargs):
        return [
            score_on_match if compiled_pattern.search(c) else score_on_miss
            for c in completions
        ]

    return ProgrammaticReward(
        check_format,
        name=name,
        description=f"Checks if completion matches pattern: {pattern[:50]}..."
    )


def create_length_reward(
    min_length: int = 0,
    max_length: int = float('inf'),
    max_score: float = 1.0,
    name: str = "LengthReward"
) -> ProgrammaticReward:
    """Create a reward based on completion length.

    Args:
        min_length: Minimum acceptable length
        max_length: Maximum acceptable length
        max_score: Score for completions in range
        name: Name for the reward

    Returns:
        ProgrammaticReward instance
    """
    def check_length(prompts, completions, **kwargs):
        scores = []
        for c in completions:
            length = len(c)
            if min_length <= length <= max_length:
                scores.append(max_score)
            else:
                # Linear penalty based on distance from range
                if length < min_length:
                    penalty = (min_length - length) / min_length
                else:
                    penalty = (length - max_length) / max_length
                scores.append(max(0.0, max_score * (1 - penalty)))
        return scores

    return ProgrammaticReward(
        check_length,
        name=name,
        description=f"Checks if length is in range [{min_length}, {max_length}]"
    )


def create_keyword_reward(
    required_keywords: List[str],
    forbidden_keywords: List[str] = None,
    score_per_required: float = 0.5,
    penalty_per_forbidden: float = -0.5,
    name: str = "KeywordReward"
) -> ProgrammaticReward:
    """Create a reward based on keyword presence.

    Args:
        required_keywords: Keywords that should appear
        forbidden_keywords: Keywords that should not appear
        score_per_required: Score for each required keyword present
        penalty_per_forbidden: Penalty for each forbidden keyword present
        name: Name for the reward

    Returns:
        ProgrammaticReward instance
    """
    forbidden_keywords = forbidden_keywords or []

    def check_keywords(prompts, completions, **kwargs):
        scores = []
        for c in completions:
            c_lower = c.lower()
            score = 0.0

            # Check required keywords
            for keyword in required_keywords:
                if keyword.lower() in c_lower:
                    score += score_per_required

            # Check forbidden keywords
            for keyword in forbidden_keywords:
                if keyword.lower() in c_lower:
                    score += penalty_per_forbidden

            scores.append(score)
        return scores

    return ProgrammaticReward(
        check_keywords,
        name=name,
        description=f"Checks for {len(required_keywords)} required and "
                   f"{len(forbidden_keywords)} forbidden keywords"
    )
