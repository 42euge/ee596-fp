"""Utilities for testing rubrics with training runs.

Provides tools for offline rubric evaluation and GRPO training integration.
"""

from dataclasses import dataclass, field
from typing import List, Callable
import statistics
import time

from .models import Rubric, RubricSet
from .scoring import rubric_reward_adapter, create_rubricset_scorer


@dataclass
class RubricTestConfig:
    """Configuration for rubric testing.

    Attributes:
        num_examples: Maximum number of examples to test
        verbose: Whether to collect example outputs
        num_verbose_examples: Number of examples to include in results
    """

    num_examples: int = 50
    verbose: bool = True
    num_verbose_examples: int = 3


@dataclass
class RubricTestResult:
    """Results from a rubric test run.

    Attributes:
        rubric_name: Name of the rubric/rubricset tested
        mean_score: Mean score across all examples
        std_score: Standard deviation of scores
        min_score: Minimum score
        max_score: Maximum score
        scores: All individual scores
        examples: Sample examples with scores (if verbose)
        duration_seconds: Time taken for the test
    """

    rubric_name: str
    mean_score: float
    std_score: float
    min_score: float
    max_score: float
    scores: List[float]
    examples: List[dict] = field(default_factory=list)
    duration_seconds: float = 0.0

    def summary(self) -> str:
        """Generate text summary of results."""
        return (
            f"Rubric: {self.rubric_name}\n"
            f"  Mean: {self.mean_score:.2f} +/- {self.std_score:.2f}\n"
            f"  Range: [{self.min_score:.2f}, {self.max_score:.2f}]\n"
            f"  Samples: {len(self.scores)}\n"
            f"  Duration: {self.duration_seconds:.1f}s"
        )

    def __repr__(self) -> str:
        return self.summary()


def test_rubric_with_dataset(
    rubric: Rubric | RubricSet,
    dataset: List[dict],
    config: RubricTestConfig | None = None,
) -> RubricTestResult:
    """Test a rubric against a dataset without model generation.

    Uses existing responses in the dataset to compute scores.
    Useful for offline evaluation of rubric quality.

    Args:
        rubric: Rubric or RubricSet to test
        dataset: List of examples with 'question'/'prompt' and 'response'/'completion' keys
        config: Test configuration

    Returns:
        RubricTestResult with scoring statistics
    """
    config = config or RubricTestConfig()
    start_time = time.time()

    if isinstance(rubric, Rubric):
        rubricset = RubricSet(name=rubric.name, rubrics=[rubric])
    else:
        rubricset = rubric

    scorer = create_rubricset_scorer(rubricset)

    # Limit examples if needed
    test_data = dataset[: config.num_examples]

    # Extract prompts and completions with flexible key names
    prompts = []
    completions = []
    for ex in test_data:
        prompt = ex.get("question") or ex.get("prompt") or ex.get("input") or ""
        completion = ex.get("response") or ex.get("completion") or ex.get("output") or ex.get("answer") or ""
        prompts.append(prompt)
        completions.append(completion)

    scores = scorer(prompts, completions)

    # Collect examples
    examples = []
    if config.verbose:
        for i in range(min(config.num_verbose_examples, len(prompts))):
            examples.append({
                "prompt": prompts[i][:200] + "..." if len(prompts[i]) > 200 else prompts[i],
                "completion": completions[i][:200] + "..." if len(completions[i]) > 200 else completions[i],
                "score": scores[i],
            })

    duration = time.time() - start_time

    return RubricTestResult(
        rubric_name=rubricset.name,
        mean_score=statistics.mean(scores) if scores else 0.0,
        std_score=statistics.stdev(scores) if len(scores) > 1 else 0.0,
        min_score=min(scores) if scores else 0.0,
        max_score=max(scores) if scores else 0.0,
        scores=scores,
        examples=examples,
        duration_seconds=duration,
    )


def create_grpo_reward_function(
    rubricset: RubricSet | Rubric,
    combine_with: List[Callable] | None = None,
    weight: float = 1.0,
) -> Callable:
    """Create a reward function for GRPO training with rubrics.

    Can combine rubric scoring with other reward functions.

    Args:
        rubricset: RubricSet or Rubric to use for scoring
        combine_with: Other reward functions to combine
        weight: Weight for rubric score in combined function

    Returns:
        Combined reward function for GRPO training

    Example:
        from tunrex.datasets.rewards import match_format_exactly, check_answer

        reward_fn = create_grpo_reward_function(
            rubricset,
            combine_with=[match_format_exactly, check_answer],
            weight=0.5,
        )
    """
    rubric_scorer = rubric_reward_adapter(rubricset)

    if not combine_with:
        if weight == 1.0:
            return rubric_scorer

        def weighted_scorer(
            prompts: List[str],
            completions: List[str],
            **kwargs,
        ) -> List[float]:
            scores = rubric_scorer(prompts, completions, **kwargs)
            return [s * weight for s in scores]

        return weighted_scorer

    def combined_reward(
        prompts: List[str],
        completions: List[str],
        **kwargs,
    ) -> List[float]:
        """Combined reward from rubric and other functions."""
        rubric_scores = rubric_scorer(prompts, completions, **kwargs)

        combined = [s * weight for s in rubric_scores]

        for reward_fn in combine_with:
            other_scores = reward_fn(prompts, completions, **kwargs)
            for i, s in enumerate(other_scores):
                combined[i] += s

        return combined

    return combined_reward


def quick_test_rubric(
    rubric: Rubric,
    test_responses: List[str],
    expected_scores: List[float] | None = None,
) -> dict:
    """Quick test of a rubric with sample responses.

    Useful for validating rubric design in notebooks.

    Args:
        rubric: Rubric to test
        test_responses: Sample responses to score
        expected_scores: Optional expected scores for validation

    Returns:
        Dict with scores, mean, rubric_text, and optional comparison to expected
    """
    from src.rubrics.scoring import create_rubric_scorer

    scorer = create_rubric_scorer(rubric)
    prompts = ["test"] * len(test_responses)  # Dummy prompts

    actual_scores = scorer(prompts, test_responses)

    result = {
        "scores": actual_scores,
        "mean": sum(actual_scores) / len(actual_scores) if actual_scores else 0,
        "rubric_text": rubric.to_text(),
    }

    if expected_scores:
        diffs = [abs(a - e) for a, e in zip(actual_scores, expected_scores)]
        result["expected"] = expected_scores
        result["differences"] = diffs
        result["max_diff"] = max(diffs) if diffs else 0
        result["passed"] = all(d < 1.0 for d in diffs)  # Allow 1.0 tolerance

    return result


def compare_rubrics(
    rubrics: List[Rubric] | RubricSet,
    dataset: List[dict],
    config: RubricTestConfig | None = None,
) -> List[RubricTestResult]:
    """Compare multiple rubrics against the same dataset.

    Args:
        rubrics: List of Rubrics or a RubricSet to compare
        dataset: Test dataset
        config: Test configuration

    Returns:
        List of RubricTestResult, one per rubric
    """
    if isinstance(rubrics, RubricSet):
        rubric_list = rubrics.rubrics
    else:
        rubric_list = rubrics

    results = []
    for rubric in rubric_list:
        result = test_rubric_with_dataset(rubric, dataset, config)
        results.append(result)

    return results


def print_comparison(results: List[RubricTestResult]) -> None:
    """Print a formatted comparison of rubric test results.

    Args:
        results: List of RubricTestResult to compare
    """
    # Sort by mean score descending
    sorted_results = sorted(results, key=lambda r: r.mean_score, reverse=True)

    print("\n" + "=" * 60)
    print("RUBRIC COMPARISON")
    print("=" * 60)
    print(f"{'Rubric':<30} {'Mean':>8} {'Std':>8} {'Min':>8} {'Max':>8}")
    print("-" * 60)

    for r in sorted_results:
        print(f"{r.rubric_name[:30]:<30} {r.mean_score:>8.2f} {r.std_score:>8.2f} {r.min_score:>8.2f} {r.max_score:>8.2f}")

    print("=" * 60)
