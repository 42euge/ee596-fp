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

__all__ = [
    "__version__",
    "TunRex",
    "TunRexConfig",
    "download_kaggle_dataset",
    "extract_hash_answer",
    "load_from_huggingface",
    "load_from_kaggle",
]
