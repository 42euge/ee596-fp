"""LLM-based rubric generation and response scoring module.

This module provides:
- RubricGenerator: Generate evaluation rubrics using LLMs
- ResponseScorer: Score responses using LLM-as-judge
- GRPO-compatible reward functions for training

Example usage:
    from src.llm_judge import (
        RubricGenerator,
        ResponseScorer,
        RubricGeneratorConfig,
        ResponseScorerConfig,
        llm_judge_reward,
        configure,
    )

    # Configure for local inference
    configure(
        generator_config=RubricGeneratorConfig.for_local(device="cuda"),
        scorer_config=ResponseScorerConfig.for_local(device="cuda"),
    )

    # Generate a rubric
    generator = RubricGenerator()
    rubric = generator.generate("What is 2 + 2?")

    # Score a response
    scorer = ResponseScorer()
    result = scorer.score(
        question="What is 2 + 2?",
        response="2 + 2 = 4",
        rubric=rubric,
    )
    print(f"Score: {result.score}/{result.max_score}")

    # Use as GRPO reward function
    scores = llm_judge_reward(
        prompts=["What is 2 + 2?"],
        completions=["2 + 2 = 4"],
    )
"""

# Configuration
from .config import (
    GenerationConfig,
    LLMResponse,
    RubricGeneratorConfig,
    ResponseScorerConfig,
    JudgeMode,
)

# Core classes
from .rubric_generator import RubricGenerator, Rubric
from .response_scorer import ResponseScorer, ScoreResult

# Backends
from .backends import get_backend, list_backends, LLMBackend

# Reward functions
from .rewards import (
    configure,
    reset,
    get_rubric_generator,
    get_response_scorer,
    llm_judge_reward,
    generate_and_score_reward,
    create_rubric_reward_fn,
    rubric_quality_reward,
)

# Caching
from .cache import RubricCache

__all__ = [
    # Configuration
    "GenerationConfig",
    "LLMResponse",
    "RubricGeneratorConfig",
    "ResponseScorerConfig",
    "JudgeMode",
    # Core classes
    "RubricGenerator",
    "Rubric",
    "ResponseScorer",
    "ScoreResult",
    # Backends
    "get_backend",
    "list_backends",
    "LLMBackend",
    # Reward functions
    "configure",
    "reset",
    "get_rubric_generator",
    "get_response_scorer",
    "llm_judge_reward",
    "generate_and_score_reward",
    "create_rubric_reward_fn",
    "rubric_quality_reward",
    # Caching
    "RubricCache",
]

__version__ = "0.1.0"
