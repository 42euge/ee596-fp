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
)

# Model loading API
from tunrex.models import (
    DEFAULT_MESH,
    get_gemma_ref_model,
    get_lora_model,
    save_model_state,
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
    # Models
    "DEFAULT_MESH",
    "get_gemma_ref_model",
    "get_lora_model",
    "save_model_state",
]
