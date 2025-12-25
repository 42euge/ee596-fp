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

# Lazy imports for torch-dependent modules to avoid requiring torch
# when only importing lightweight submodules like rubrics
def __getattr__(name):
    """Lazy import for torch-dependent modules."""
    if name in ("GemmaModel", "load_model", "get_device"):
        from .model import GemmaModel, load_model, get_device
        return {"GemmaModel": GemmaModel, "load_model": load_model, "get_device": get_device}[name]

    if name in ("extract_reasoning_and_answer", "extract_numerical_answer",
                "load_gsm8k_dataset", "load_openrubrics_dataset", "format_reward",
                "accuracy_reward", "rubric_reward", "detect_question_type", "evaluate_accuracy"):
        from . import utils
        return getattr(utils, name)

    if name in ("SearchSpace", "ParameterSpec", "TrialResult", "SearchResults",
                "HyperparameterSearch", "GridSearchStrategy", "RandomSearchStrategy",
                "BayesianSearchStrategy", "get_lora_search_space", "get_training_search_space",
                "get_grpo_search_space", "get_generation_search_space", "get_full_search_space",
                "params_to_config", "create_objective_from_eval_fn", "analyze_results"):
        from . import hyperparam_search
        return getattr(hyperparam_search, name)

    if name in ("RobustnessConfig", "PerturbationConfig", "ExternalRewardConfig",
                "RobustnessEvaluator", "RobustnessResults", "ConsistencyMetrics",
                "compute_consistency_metrics", "PerturbationPipeline", "SynonymPerturbation",
                "ParaphrasePerturbation", "SentenceReorderPerturbation", "InternalReward",
                "HuggingFaceReward", "load_reward_model"):
        from . import reward_robustness
        return getattr(reward_robustness, name)

    if name == "get_default_robustness_config":
        from .reward_robustness import get_default_config
        return get_default_config
    if name == "get_quick_robustness_config":
        from .reward_robustness import get_quick_config
        return get_quick_config
    if name == "get_thorough_robustness_config":
        from .reward_robustness import get_thorough_config
        return get_thorough_config

    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")

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
    # Reward Robustness
    "RobustnessConfig",
    "PerturbationConfig",
    "ExternalRewardConfig",
    "get_default_robustness_config",
    "get_quick_robustness_config",
    "get_thorough_robustness_config",
    "RobustnessEvaluator",
    "RobustnessResults",
    "ConsistencyMetrics",
    "compute_consistency_metrics",
    "PerturbationPipeline",
    "SynonymPerturbation",
    "ParaphrasePerturbation",
    "SentenceReorderPerturbation",
    "InternalReward",
    "HuggingFaceReward",
    "load_reward_model",
]
