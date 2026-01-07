"""
PRIME RL Reward Functions

GRPO-compatible reward functions implementing PRIME RL
(Process-based Reinforcement with Intermediate Model Evaluation).

These functions can be used as drop-in replacements or additions to
existing reward functions in the GRPO training pipeline.
"""

from typing import List, Optional, Callable, Dict, Any
import warnings

from .config import PRIMEConfig
from .process_reward import ProcessRewardCalculator, create_prime_reward_function


def prime_rl_reward(
    prompts: List[str],
    completions: List[str],
    answer: Optional[List[str]] = None,
    question: Optional[List[str]] = None,
    config: Optional[PRIMEConfig] = None,
    outcome_reward_fn: Optional[Callable] = None,
    **kwargs
) -> List[float]:
    """
    Main PRIME RL reward function for GRPO training.

    This function implements process-based reinforcement learning by:
    1. Parsing completions into intermediate reasoning steps
    2. Evaluating each step using multiple methods
    3. Aggregating step rewards with proper credit assignment
    4. Combining with outcome-based rewards

    Args:
        prompts: List of input prompts
        completions: List of model completions
        answer: Optional list of ground truth answers
        question: Optional list of questions
        config: PRIME RL configuration (uses defaults if None)
        outcome_reward_fn: Optional function for outcome-based rewards
        **kwargs: Additional context passed to evaluators

    Returns:
        List of reward values (one per completion)

    Example:
        >>> config = PRIMEConfig(
        ...     step_evaluation_method=StepEvaluationMethod.HYBRID,
        ...     reward_aggregation=RewardAggregation.DISCOUNTED_SUM,
        ...     gamma=0.95
        ... )
        >>> rewards = prime_rl_reward(
        ...     prompts=["Solve: 2 + 2 = ?"],
        ...     completions=["Step 1: Add 2 + 2\\nStep 2: Result is 4\\nAnswer: 4"],
        ...     answer=["4"],
        ...     config=config
        ... )
    """
    if config is None:
        config = PRIMEConfig()

    # Create reward function
    reward_fn = create_prime_reward_function(config, outcome_reward_fn)

    # Calculate rewards
    return reward_fn(prompts, completions, answer, question, **kwargs)


def prime_rl_with_accuracy(
    prompts: List[str],
    completions: List[str],
    answer: Optional[List[str]] = None,
    question: Optional[List[str]] = None,
    config: Optional[PRIMEConfig] = None,
    **kwargs
) -> List[float]:
    """
    PRIME RL reward function combined with accuracy checking.

    Uses answer accuracy as the outcome reward and combines it with
    process-based rewards.

    Args:
        prompts: List of input prompts
        completions: List of model completions
        answer: List of ground truth answers
        question: Optional list of questions
        config: PRIME RL configuration
        **kwargs: Additional context

    Returns:
        List of reward values
    """
    # Import accuracy reward function
    try:
        from ..utils import accuracy_reward
    except ImportError:
        warnings.warn("accuracy_reward not available, using simplified version")

        def accuracy_reward(prompts, completions, answer, **kw):
            # Simplified accuracy check
            from ..utils import extract_reasoning_and_answer
            rewards = []
            for comp, ans in zip(completions, answer):
                _, pred_answer = extract_reasoning_and_answer(comp)
                reward = 1.0 if pred_answer == ans else 0.0
                rewards.append(reward)
            return rewards

    return prime_rl_reward(
        prompts=prompts,
        completions=completions,
        answer=answer,
        question=question,
        config=config,
        outcome_reward_fn=accuracy_reward,
        **kwargs
    )


def prime_rl_with_format(
    prompts: List[str],
    completions: List[str],
    answer: Optional[List[str]] = None,
    question: Optional[List[str]] = None,
    config: Optional[PRIMEConfig] = None,
    **kwargs
) -> List[float]:
    """
    PRIME RL reward function combined with format checking.

    Uses format compliance as the outcome reward and combines it with
    process-based rewards.

    Args:
        prompts: List of input prompts
        completions: List of model completions
        answer: Optional list of ground truth answers
        question: Optional list of questions
        config: PRIME RL configuration
        **kwargs: Additional context

    Returns:
        List of reward values
    """
    # Import format reward function
    try:
        from ..utils import format_reward
    except ImportError:
        warnings.warn("format_reward not available, using simplified version")

        def format_reward(prompts, completions, **kw):
            # Simplified format check
            rewards = []
            for comp in completions:
                has_reasoning = "<reasoning>" in comp.lower()
                has_answer = "<answer>" in comp.lower()
                reward = 1.0 if (has_reasoning and has_answer) else 0.0
                rewards.append(reward)
            return rewards

    return prime_rl_reward(
        prompts=prompts,
        completions=completions,
        answer=answer,
        question=question,
        config=config,
        outcome_reward_fn=format_reward,
        **kwargs
    )


def create_prime_rl_reward_suite(
    config: Optional[PRIMEConfig] = None,
    include_format: bool = True,
    include_accuracy: bool = True,
    include_pure_prime: bool = True
) -> List[Callable]:
    """
    Create a suite of PRIME RL reward functions for GRPO training.

    Returns a list of reward functions that can be used together
    in the GRPO reward_fns parameter.

    Args:
        config: PRIME RL configuration
        include_format: Include format-based PRIME RL reward
        include_accuracy: Include accuracy-based PRIME RL reward
        include_pure_prime: Include pure process-based reward

    Returns:
        List of reward functions

    Example:
        >>> config = PRIMEConfig(gamma=0.95)
        >>> reward_fns = create_prime_rl_reward_suite(config)
        >>> # Use in GRPO training
        >>> trainer = GRPOLearner(..., reward_fns=reward_fns)
    """
    if config is None:
        config = PRIMEConfig()

    reward_fns = []

    if include_pure_prime:
        # Pure process-based reward (no outcome reward)
        pure_config = PRIMEConfig(**{
            **config.__dict__,
            "use_step_rewards_only": True
        })

        def pure_prime_reward(prompts, completions, answer=None, question=None, **kw):
            return prime_rl_reward(
                prompts, completions, answer, question,
                config=pure_config, **kw
            )

        reward_fns.append(pure_prime_reward)

    if include_format:
        def format_prime_reward(prompts, completions, answer=None, question=None, **kw):
            return prime_rl_with_format(
                prompts, completions, answer, question,
                config=config, **kw
            )

        reward_fns.append(format_prime_reward)

    if include_accuracy:
        def accuracy_prime_reward(prompts, completions, answer=None, question=None, **kw):
            return prime_rl_with_accuracy(
                prompts, completions, answer, question,
                config=config, **kw
            )

        reward_fns.append(accuracy_prime_reward)

    return reward_fns


# Convenience function for backward compatibility
def get_default_prime_rl_config() -> PRIMEConfig:
    """
    Get default PRIME RL configuration optimized for mathematical reasoning.

    Returns:
        PRIMEConfig with recommended settings
    """
    return PRIMEConfig(
        # Step parsing
        step_parsing_strategy="numbered",
        min_step_length=10,
        max_steps=20,

        # Evaluation
        step_evaluation_method="hybrid",
        enable_symbolic_solver=True,
        llm_judge_enabled=False,  # Disabled by default for cost

        # Reward aggregation
        reward_aggregation="discounted_sum",
        gamma=0.95,
        step_reward_scale=1.0,
        final_answer_weight=2.0,

        # Credit assignment
        normalize_rewards=True,
        baseline_subtraction=True,

        # Combination with outcome
        combine_with_outcome_rewards=True,
        outcome_reward_weight=0.3,

        # Process supervision
        penalize_incorrect_steps=True,
        incorrect_step_penalty=-0.5,
        reward_correct_process=True,
    )
