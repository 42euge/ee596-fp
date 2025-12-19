"""
Gemma3-1B GRPO Fine-tuning Package

This package provides utilities for fine-tuning Gemma3-1B with GRPO
(Group Relative Policy Optimization) for improved reasoning capabilities.
"""

from .config import (
    Config,
    get_default_config,
    format_prompt,
    get_system_prompt,
    REASONING_START,
    REASONING_END,
    SOLUTION_START,
    SOLUTION_END,
)

from .model import GemmaModel, load_model, get_device

from .utils import (
    extract_reasoning_and_answer,
    extract_numerical_answer,
    load_gsm8k_dataset,
    load_openrubrics_dataset,
    format_reward,
    accuracy_reward,
    rubric_reward,
    detect_question_type,
    evaluate_accuracy,
)

from .hyperparam_search import (
    SearchSpace,
    ParameterSpec,
    TrialResult,
    SearchResults,
    HyperparameterSearch,
    GridSearchStrategy,
    RandomSearchStrategy,
    BayesianSearchStrategy,
    get_lora_search_space,
    get_training_search_space,
    get_grpo_search_space,
    get_generation_search_space,
    get_full_search_space,
    params_to_config,
    create_objective_from_eval_fn,
    analyze_results,
)

__version__ = "1.0.0"
__all__ = [
    # Config
    "Config",
    "get_default_config",
    "format_prompt",
    "get_system_prompt",
    "REASONING_START",
    "REASONING_END",
    "SOLUTION_START",
    "SOLUTION_END",
    # Model
    "GemmaModel",
    "load_model",
    "get_device",
    # Utils
    "extract_reasoning_and_answer",
    "extract_numerical_answer",
    "load_gsm8k_dataset",
    "load_openrubrics_dataset",
    "format_reward",
    "accuracy_reward",
    "rubric_reward",
    "detect_question_type",
    "evaluate_accuracy",
    # Hyperparameter Search
    "SearchSpace",
    "ParameterSpec",
    "TrialResult",
    "SearchResults",
    "HyperparameterSearch",
    "GridSearchStrategy",
    "RandomSearchStrategy",
    "BayesianSearchStrategy",
    "get_lora_search_space",
    "get_training_search_space",
    "get_grpo_search_space",
    "get_generation_search_space",
    "get_full_search_space",
    "params_to_config",
    "create_objective_from_eval_fn",
    "analyze_results",
]
