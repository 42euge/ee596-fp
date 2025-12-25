"""GRPO-compatible reward functions using LLM-as-judge.

Provides reward functions that can be used with the GRPO training pipeline.
Compatible with the signature used in TunRex/src/tunrex/datasets/rewards.py.
"""

from typing import List, Optional, Callable

from .config import RubricGeneratorConfig, ResponseScorerConfig
from .rubric_generator import RubricGenerator, Rubric
from .response_scorer import ResponseScorer


# Module-level singletons for efficiency in training loops
_rubric_generator: Optional[RubricGenerator] = None
_response_scorer: Optional[ResponseScorer] = None


def get_rubric_generator(
    config: Optional[RubricGeneratorConfig] = None,
) -> RubricGenerator:
    """Get or create the singleton RubricGenerator.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        RubricGenerator instance
    """
    global _rubric_generator
    if _rubric_generator is None:
        _rubric_generator = RubricGenerator(config)
    return _rubric_generator


def get_response_scorer(
    config: Optional[ResponseScorerConfig] = None,
) -> ResponseScorer:
    """Get or create the singleton ResponseScorer.

    Args:
        config: Optional configuration (only used on first call)

    Returns:
        ResponseScorer instance
    """
    global _response_scorer
    if _response_scorer is None:
        _response_scorer = ResponseScorer(config)
    return _response_scorer


def configure(
    generator_config: Optional[RubricGeneratorConfig] = None,
    scorer_config: Optional[ResponseScorerConfig] = None,
) -> None:
    """Configure the module with custom configs.

    Call this before using any reward functions to set up backends.

    Args:
        generator_config: Configuration for rubric generation
        scorer_config: Configuration for response scoring
    """
    global _rubric_generator, _response_scorer
    if generator_config:
        _rubric_generator = RubricGenerator(generator_config)
    if scorer_config:
        _response_scorer = ResponseScorer(scorer_config)


def reset() -> None:
    """Reset the module singletons.

    Useful for testing or switching configurations.
    """
    global _rubric_generator, _response_scorer
    _rubric_generator = None
    _response_scorer = None


def llm_judge_reward(
    prompts: List[str],
    completions: List[str],
    rubrics: Optional[List[str]] = None,
    questions: Optional[List[str]] = None,
    **kwargs,
) -> List[float]:
    """GRPO-compatible reward function using LLM-as-judge.

    This is the main reward function for integration with GRPO training.
    Compatible with the signature used in TunRex/src/tunrex/datasets/rewards.py

    Args:
        prompts: Input prompts (may include system prompt + question)
        completions: Model completions to score
        rubrics: Optional pre-defined rubrics (if None, will be generated)
        questions: Optional questions extracted from prompts (for rubric generation)
        **kwargs: Additional metadata (answer, question, etc.)

    Returns:
        List of reward scores (normalized to 0-10 range)
    """
    scorer = get_response_scorer()
    generator = get_rubric_generator()

    # Extract questions if not provided
    if questions is None:
        questions = kwargs.get("question", prompts)
        if isinstance(questions, str):
            questions = [questions] * len(completions)

    # Ensure we have a list of questions
    if not isinstance(questions, list):
        questions = [questions] * len(completions)

    # Generate rubrics if not provided
    if rubrics is None:
        rubric_objs = generator.generate_batch(questions)
        rubric_texts = [r.to_text() for r in rubric_objs]
    elif isinstance(rubrics, str):
        rubric_texts = [rubrics] * len(completions)
    else:
        rubric_texts = rubrics

    # Score each completion
    scores = []
    for question, completion, rubric_text in zip(questions, completions, rubric_texts):
        try:
            result = scorer.score(
                question=question,
                response=completion,
                rubric_text=rubric_text,
            )
            # Normalize to 0-10 range for consistency with existing rewards
            scores.append(result.normalized_score * 10.0)
        except Exception as e:
            # Fallback to 0 on errors to avoid breaking training
            print(f"Warning: LLM judge error: {e}")
            scores.append(0.0)

    return scores


def generate_and_score_reward(
    prompts: List[str],
    completions: List[str],
    answers: Optional[List[str]] = None,
    **kwargs,
) -> List[float]:
    """Combined reward: generates rubrics and scores responses.

    Higher-level function that:
    1. Detects question type
    2. Generates appropriate rubric
    3. Scores response against rubric
    4. Optionally combines with accuracy check if answer provided

    Compatible with GRPO training reward function signature.

    Args:
        prompts: Input prompts
        completions: Model completions
        answers: Optional ground truth answers for accuracy scoring
        **kwargs: Additional metadata

    Returns:
        List of reward scores
    """
    questions = kwargs.get("question", prompts)
    if isinstance(questions, str):
        questions = [questions] * len(completions)
    if not isinstance(questions, list):
        questions = [questions] * len(completions)

    # Generate rubrics based on question type
    generator = get_rubric_generator()
    rubrics = generator.generate_batch(questions)

    # Get LLM judge scores
    llm_scores = llm_judge_reward(
        prompts=prompts,
        completions=completions,
        rubrics=[r.to_text() for r in rubrics],
        questions=questions,
        **kwargs,
    )

    # If answers provided, combine with accuracy
    if answers is not None:
        try:
            from ..utils import accuracy_reward

            acc_scores = accuracy_reward(prompts, completions, answers)
            # Weighted combination: 60% rubric, 40% accuracy
            # Scale accuracy (0-1.5) to (0-10) range
            combined = [
                0.6 * llm + 0.4 * (acc * 6.67)
                for llm, acc in zip(llm_scores, acc_scores)
            ]
            return combined
        except ImportError:
            pass

    return llm_scores


def create_rubric_reward_fn(
    generator_config: Optional[RubricGeneratorConfig] = None,
    scorer_config: Optional[ResponseScorerConfig] = None,
    weight: float = 1.0,
) -> Callable[[List[str], List[str]], List[float]]:
    """Factory function to create a configured rubric reward function.

    Use this to create reward functions with specific configurations
    for different training runs.

    Args:
        generator_config: Configuration for rubric generation
        scorer_config: Configuration for response scoring
        weight: Multiplier for the final scores

    Returns:
        Configured reward function with signature (prompts, completions, **kwargs) -> scores
    """
    generator = RubricGenerator(generator_config)
    scorer = ResponseScorer(scorer_config)

    def reward_fn(
        prompts: List[str],
        completions: List[str],
        **kwargs,
    ) -> List[float]:
        """Custom reward function with configured generator and scorer."""
        questions = kwargs.get("question", prompts)
        if isinstance(questions, str):
            questions = [questions] * len(completions)
        if not isinstance(questions, list):
            questions = [questions] * len(completions)

        rubrics = generator.generate_batch(questions)

        scores = []
        for q, c, r in zip(questions, completions, rubrics):
            try:
                result = scorer.score(question=q, response=c, rubric=r)
                scores.append(result.normalized_score * 10.0 * weight)
            except Exception as e:
                print(f"Warning: Scoring error: {e}")
                scores.append(0.0)

        return scores

    return reward_fn


def rubric_quality_reward(
    prompts: List[str],
    completions: List[str],
    rubric_texts: Optional[List[str]] = None,
    **kwargs,
) -> List[float]:
    """Reward function based on rubric coverage and quality.

    Scores how well a response addresses the criteria in a rubric,
    without requiring a full LLM judge call. Uses text matching
    and keyword overlap as a faster alternative.

    Args:
        prompts: Input prompts
        completions: Model completions
        rubric_texts: Rubric texts to score against
        **kwargs: Additional metadata

    Returns:
        List of reward scores (0-10 range)
    """
    import re
    from collections import Counter

    questions = kwargs.get("question", prompts)
    if isinstance(questions, str):
        questions = [questions] * len(completions)

    # Generate rubrics if not provided
    if rubric_texts is None:
        generator = get_rubric_generator()
        rubrics = generator.generate_batch(questions)
        rubric_texts = [r.to_text() for r in rubrics]
    elif isinstance(rubric_texts, str):
        rubric_texts = [rubric_texts] * len(completions)

    def tokenize(text: str) -> List[str]:
        """Simple tokenization."""
        text = text.lower()
        text = re.sub(r"[^\w\s]", " ", text)
        return [t for t in text.split() if len(t) > 2]

    def overlap_score(response: str, rubric: str) -> float:
        """Calculate weighted overlap between response and rubric."""
        rubric_tokens = tokenize(rubric)
        response_tokens = set(tokenize(response))

        if not rubric_tokens:
            return 5.0  # Neutral score

        # TF-IDF style: rare terms matter more
        token_counts = Counter(rubric_tokens)
        weighted_matches = sum(
            1.0 / token_counts[t] for t in response_tokens if t in token_counts
        )
        max_score = sum(1.0 / c for c in token_counts.values())

        coverage = weighted_matches / max_score if max_score > 0 else 0.0
        return round(coverage * 10.0, 4)

    scores = []
    for completion, rubric_text in zip(completions, rubric_texts):
        score = overlap_score(completion, rubric_text)
        scores.append(score)

    return scores
