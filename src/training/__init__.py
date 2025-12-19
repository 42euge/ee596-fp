"""
Training utilities for GRPO fine-tuning in Google Colab.

This module provides a light orchestration API for training notebooks,
keeping heavy logic in reusable Python modules.
"""

from .colab_pipeline import (
    ColabTrainingConfig,
    ColabSession,
    TrainerState,
    prepare_colab_session,
    train_grpo,
    export_checkpoint,
    quick_test,
    run_full_pipeline,
    format_prompt,
    prepare_datasets,
    create_reward_functions,
)

__all__ = [
    "ColabTrainingConfig",
    "ColabSession",
    "TrainerState",
    "prepare_colab_session",
    "train_grpo",
    "export_checkpoint",
    "quick_test",
    "run_full_pipeline",
    "format_prompt",
    "prepare_datasets",
    "create_reward_functions",
]
