"""TunRex - A flexible TUI/CLI toolkit for ML dataset management."""

__version__ = "0.1.0"

# Dataset loading API
from tunrex.datasets import (
    TunRex,
    TunRexConfig,
    download_kaggle_dataset,
    extract_hash_answer,
    load_from_huggingface,
    load_from_kaggle,
    load_openrubrics,
)

# Model loading API
from tunrex.models import (
    DEFAULT_MESH,
    create_mesh,
    get_gemma_ref_model,
    get_lora_model,
    prepare_gemma_checkpoint,
    save_model_state,
)

# Reward comparison API
from tunrex.reward_comparison import (
    BaseReward,
    RewardResult,
    RewardMetadata,
    ProgrammaticReward,
    RubricReward,
    RubricCriterion,
    PreferenceModelReward,
    RewardComparison,
    ComparisonResult,
    RewardAnalyzer,
    CorrelationAnalysis,
    AgreementAnalysis,
)

__all__ = [
    "__version__",
    # Datasets
    "TunRex",
    "TunRexConfig",
    "download_kaggle_dataset",
    "extract_hash_answer",
    "load_from_huggingface",
    "load_from_kaggle",
    "load_openrubrics",
    # Models
    "DEFAULT_MESH",
    "create_mesh",
    "get_gemma_ref_model",
    "get_lora_model",
    "prepare_gemma_checkpoint",
    "save_model_state",
    # Reward Comparison
    "BaseReward",
    "RewardResult",
    "RewardMetadata",
    "ProgrammaticReward",
    "RubricReward",
    "RubricCriterion",
    "PreferenceModelReward",
    "RewardComparison",
    "ComparisonResult",
    "RewardAnalyzer",
    "CorrelationAnalysis",
    "AgreementAnalysis",
]
