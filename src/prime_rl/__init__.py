"""
PRIME RL - Process-based Reinforcement with Intermediate Model Evaluation

A reinforcement learning paradigm for training reasoning models where rewards
are assigned to intermediate reasoning steps, not just the final answer.

Core Components:
    - PRIMEConfig: Configuration for PRIME RL training
    - StepParser: Extracts intermediate reasoning steps
    - StepEvaluator: Evaluates individual steps (rule-based, symbolic, LLM)
    - ProcessRewardCalculator: Aggregates step rewards with credit assignment
    - Reward Functions: GRPO-compatible reward functions

Usage:
    >>> from src.prime_rl import PRIMEConfig, prime_rl_reward
    >>> config = PRIMEConfig(gamma=0.95)
    >>> rewards = prime_rl_reward(
    ...     prompts=["Solve: 2 + 2"],
    ...     completions=["Step 1: ..."],
    ...     config=config
    ... )

For more details, see the documentation at docs/prime_rl.md
"""

from .config import (
    PRIMEConfig,
    StepEvaluationMethod,
    RewardAggregation,
    StepParsingStrategy,
    StepReward,
    TrajectoryReward,
)

from .step_parser import (
    StepParser,
    ParsedStep,
    parse_steps,
    format_steps_for_display,
)

from .step_evaluator import (
    StepEvaluator,
    RuleBasedEvaluator,
    SymbolicEvaluator,
    LLMJudgeStepEvaluator,
    StepEvaluation,
)

from .process_reward import (
    ProcessRewardCalculator,
    create_prime_reward_function,
    calculate_step_advantages,
    analyze_trajectory,
)

from .rewards import (
    prime_rl_reward,
    prime_rl_with_accuracy,
    prime_rl_with_format,
    create_prime_rl_reward_suite,
    get_default_prime_rl_config,
)

__all__ = [
    # Configuration
    "PRIMEConfig",
    "StepEvaluationMethod",
    "RewardAggregation",
    "StepParsingStrategy",
    "StepReward",
    "TrajectoryReward",

    # Step Parsing
    "StepParser",
    "ParsedStep",
    "parse_steps",
    "format_steps_for_display",

    # Step Evaluation
    "StepEvaluator",
    "RuleBasedEvaluator",
    "SymbolicEvaluator",
    "LLMJudgeStepEvaluator",
    "StepEvaluation",

    # Process Rewards
    "ProcessRewardCalculator",
    "create_prime_reward_function",
    "calculate_step_advantages",
    "analyze_trajectory",

    # Reward Functions
    "prime_rl_reward",
    "prime_rl_with_accuracy",
    "prime_rl_with_format",
    "create_prime_rl_reward_suite",
    "get_default_prime_rl_config",
]

__version__ = "0.1.0"
