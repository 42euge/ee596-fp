"""TunRex datasets module for loading and preparing ML training datasets."""

from tunrex.datasets.config import (
    TunRexConfig,
    # Template tags
    reasoning_start,
    reasoning_end,
    solution_start,
    solution_end,
    REASONING_START,
    REASONING_END,
    SOLUTION_START,
    SOLUTION_END,
    # Templates
    DEFAULT_TEMPLATE,
    DEFAULT_SYSTEM_PROMPT,
    SYSTEM_PROMPTS,
    get_system_prompt,
)
from tunrex.datasets.core import TunRex
from tunrex.datasets.loaders import (
    download_kaggle_dataset,
    extract_hash_answer,
    get_dataset,
    get_train_val_test_datasets,
    load_from_huggingface,
    load_from_kaggle,
    load_from_tfds,
    load_openrubrics,
)
from tunrex.datasets.rewards import (
    match_format,
    match_numbers,
    match_format_exactly,
    match_format_approximately,
    check_answer,
    check_numbers,
)
from tunrex.datasets.evaluate import (
    generate,
    evaluate,
)

__all__ = [
    # Core classes
    "TunRex",
    "TunRexConfig",
    # Template tags
    "reasoning_start",
    "reasoning_end",
    "solution_start",
    "solution_end",
    "REASONING_START",
    "REASONING_END",
    "SOLUTION_START",
    "SOLUTION_END",
    # Templates
    "DEFAULT_TEMPLATE",
    "DEFAULT_SYSTEM_PROMPT",
    "SYSTEM_PROMPTS",
    "get_system_prompt",
    # Loaders
    "download_kaggle_dataset",
    "extract_hash_answer",
    "get_dataset",
    "get_train_val_test_datasets",
    "load_from_huggingface",
    "load_from_kaggle",
    "load_from_tfds",
    "load_openrubrics",
    # Rewards
    "match_format",
    "match_numbers",
    "match_format_exactly",
    "match_format_approximately",
    "check_answer",
    "check_numbers",
    # Evaluate
    "generate",
    "evaluate",
]
