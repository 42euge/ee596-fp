"""Test that TunRex can be imported without heavy dependencies."""

import pytest


def test_tunrex_import():
    """Test basic tunrex import works."""
    import tunrex
    assert hasattr(tunrex, "__version__")


def test_datasets_import():
    """Test datasets module imports."""
    from tunrex.datasets import (
        TunRex,
        TunRexConfig,
        load_openrubrics,
        load_from_huggingface,
    )
    assert TunRex is not None
    assert TunRexConfig is not None
    assert callable(load_openrubrics)
    assert callable(load_from_huggingface)


def test_models_import_without_tunix():
    """Test models module can be imported even without tunix installed.

    The module uses lazy imports so it should not fail at import time.
    """
    from tunrex.models import (
        DEFAULT_MESH,
        get_gemma_ref_model,
        get_lora_model,
        save_model_state,
    )
    assert DEFAULT_MESH == [(1, 1), ("fsdp", "tp")]
    assert callable(get_gemma_ref_model)
    assert callable(get_lora_model)
    assert callable(save_model_state)


def test_top_level_exports():
    """Test all expected symbols are exported from top-level."""
    import tunrex

    # Datasets
    assert hasattr(tunrex, "TunRex")
    assert hasattr(tunrex, "TunRexConfig")
    assert hasattr(tunrex, "load_openrubrics")
    assert hasattr(tunrex, "load_from_huggingface")
    assert hasattr(tunrex, "load_from_kaggle")
    assert hasattr(tunrex, "download_kaggle_dataset")
    assert hasattr(tunrex, "extract_hash_answer")

    # Models
    assert hasattr(tunrex, "DEFAULT_MESH")
    assert hasattr(tunrex, "get_gemma_ref_model")
    assert hasattr(tunrex, "get_lora_model")
    assert hasattr(tunrex, "prepare_gemma_checkpoint")
    assert hasattr(tunrex, "save_model_state")
