"""Integration with existing rubric_reward() scoring system.

Provides factory functions to create scorers compatible with GRPO training.
"""

from typing import List, Callable

from .models import Rubric, RubricSet


def create_rubric_scorer(rubric: Rubric) -> Callable:
    """Create a scoring function for a specific rubric.

    Returns a function compatible with existing rubric_reward() interface.

    Args:
        rubric: Rubric to use for scoring

    Returns:
        Scoring function with signature (prompts, completions, **kwargs) -> List[float]
    """
    from src.utils import rubric_overlap_score

    rubric_text = rubric.to_text()

    def scorer(
        prompts: List[str],
        completions: List[str],
        **kwargs,
    ) -> List[float]:
        """Score completions using the rubric."""
        scores = []
        for completion in completions:
            score = rubric_overlap_score(completion, rubric_text)
            scores.append(score)
        return scores

    return scorer


def create_rubricset_scorer(
    rubricset: RubricSet,
    question_type_detector: Callable[[str], str] | None = None,
) -> Callable:
    """Create a scoring function that selects rubric based on question type.

    Args:
        rubricset: RubricSet containing multiple rubrics
        question_type_detector: Optional function to detect question type from prompt

    Returns:
        Scoring function compatible with GRPO reward functions
    """
    from src.utils import rubric_overlap_score

    def scorer(
        prompts: List[str],
        completions: List[str],
        **kwargs,
    ) -> List[float]:
        """Score completions using appropriate rubric for each question."""
        scores = []
        for prompt, completion in zip(prompts, completions):
            # Find matching rubric
            rubric = rubricset.get_for_question(prompt, question_type_detector)
            if rubric:
                rubric_text = rubric.to_text()
                score = rubric_overlap_score(completion, rubric_text)
            else:
                score = 0.0
            scores.append(score)
        return scores

    return scorer


def create_weighted_rubric_scorer(
    rubric: Rubric,
    normalize: bool = True,
) -> Callable:
    """Create scorer that respects criterion weights.

    Scores each criterion separately and combines with weights.

    Args:
        rubric: Rubric with weighted criteria
        normalize: Whether to normalize score to 0-10 range

    Returns:
        Weighted scoring function
    """
    from src.utils import rubric_overlap_score

    def scorer(
        prompts: List[str],
        completions: List[str],
        **kwargs,
    ) -> List[float]:
        scores = []
        total_weight = sum(c.weight for c in rubric.criteria)

        for completion in completions:
            weighted_score = 0.0
            for criterion in rubric.criteria:
                # Build criterion text from description + keywords
                criterion_text = criterion.description
                if criterion.keywords:
                    criterion_text += " " + " ".join(criterion.keywords)

                criterion_score = rubric_overlap_score(completion, criterion_text)
                weighted_score += criterion_score * criterion.weight

            if normalize and total_weight > 0:
                weighted_score = (weighted_score / total_weight)

            scores.append(weighted_score)

        return scores

    return scorer


def rubric_reward_adapter(rubricset: RubricSet | Rubric) -> Callable:
    """Adapter to use RubricSet/Rubric with existing rubric_reward() function.

    Creates wrapper that converts our rubric format to the format expected
    by src.utils.rubric_reward().

    Args:
        rubricset: RubricSet or single Rubric

    Returns:
        Reward function compatible with GRPO training
    """
    from src.utils import rubric_reward

    if isinstance(rubricset, Rubric):
        rubricset = RubricSet(name="single", rubrics=[rubricset])

    def adapter(
        prompts: List[str],
        completions: List[str],
        **kwargs,
    ) -> List[float]:
        """Adapted reward function."""
        rubrics_text = []
        references = []
        targets = []

        for prompt in prompts:
            rubric = rubricset.get_for_question(prompt)
            if rubric:
                rubrics_text.append(rubric.to_text())
                references.append(rubric.reference_response or "")
                targets.append(rubric.target_score)
            else:
                rubrics_text.append("")
                references.append("")
                targets.append(None)

        return rubric_reward(
            prompts=prompts,
            completions=completions,
            rubrics=rubrics_text,
            reference_responses=references,
            target_scores=targets,
            **kwargs,
        )

    return adapter


def create_multi_rubric_scorer(
    rubricset: RubricSet,
    aggregation: str = "max",
) -> Callable:
    """Create scorer that evaluates against multiple rubrics.

    Useful when a response could match multiple rubric types.

    Args:
        rubricset: RubricSet containing multiple rubrics
        aggregation: How to combine scores ("max", "mean", "sum")

    Returns:
        Scoring function
    """
    from src.utils import rubric_overlap_score

    def scorer(
        prompts: List[str],
        completions: List[str],
        **kwargs,
    ) -> List[float]:
        scores = []
        for completion in completions:
            rubric_scores = []
            for rubric in rubricset.rubrics:
                rubric_text = rubric.to_text()
                score = rubric_overlap_score(completion, rubric_text)
                rubric_scores.append(score)

            if not rubric_scores:
                final_score = 0.0
            elif aggregation == "max":
                final_score = max(rubric_scores)
            elif aggregation == "mean":
                final_score = sum(rubric_scores) / len(rubric_scores)
            elif aggregation == "sum":
                final_score = sum(rubric_scores)
            else:
                final_score = max(rubric_scores)

            scores.append(final_score)
        return scores

    return scorer
