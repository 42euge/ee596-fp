"""
Reward Robustness Evaluation Module

Evaluates reward model consistency across semantic-preserving perturbations.
Tests whether reward functions give consistent scores when inputs are
paraphrased, synonyms replaced, or sentences reordered.
"""

from .config import (
    RobustnessConfig,
    PerturbationConfig,
    ExternalRewardConfig,
)
from .evaluator import RobustnessEvaluator, RobustnessResults
from .metrics import ConsistencyMetrics, compute_consistency_metrics
from .perturbations import (
    BasePerturbation,
    SynonymPerturbation,
    ParaphrasePerturbation,
    SentenceReorderPerturbation,
    PerturbationPipeline,
)
from .rewards import (
    RewardModel,
    InternalReward,
    HuggingFaceReward,
    load_reward_model,
)

__all__ = [
    # Config
    "RobustnessConfig",
    "PerturbationConfig",
    "ExternalRewardConfig",
    # Evaluator
    "RobustnessEvaluator",
    "RobustnessResults",
    # Metrics
    "ConsistencyMetrics",
    "compute_consistency_metrics",
    # Perturbations
    "BasePerturbation",
    "SynonymPerturbation",
    "ParaphrasePerturbation",
    "SentenceReorderPerturbation",
    "PerturbationPipeline",
    # Rewards
    "RewardModel",
    "InternalReward",
    "HuggingFaceReward",
    "load_reward_model",
]
