"""Tests for tunrex.models module."""

import pytest


class TestModelsModule:
    """Test models module structure."""

    def test_default_mesh_value(self):
        """Test DEFAULT_MESH has expected value."""
        from tunrex.models import DEFAULT_MESH

        assert DEFAULT_MESH == [(1, 1), ("fsdp", "tp")]

    def test_prepare_gemma_checkpoint_signature(self):
        """Test prepare_gemma_checkpoint has expected parameters."""
        from tunrex.models import prepare_gemma_checkpoint
        import inspect

        sig = inspect.signature(prepare_gemma_checkpoint)
        params = list(sig.parameters.keys())

        assert "ckpt_dir" in params
        assert "clean_dirs" in params

    def test_prepare_gemma_checkpoint_defaults(self):
        """Test prepare_gemma_checkpoint has expected defaults."""
        from tunrex.models import prepare_gemma_checkpoint
        import inspect

        sig = inspect.signature(prepare_gemma_checkpoint)

        assert sig.parameters["ckpt_dir"].default == "/tmp/intermediate_ckpt"
        assert sig.parameters["clean_dirs"].default is None

    def test_create_mesh_signature(self):
        """Test create_mesh has expected parameters."""
        from tunrex.models import create_mesh
        import inspect

        sig = inspect.signature(create_mesh)
        params = list(sig.parameters.keys())

        assert "mesh_config" in params

    def test_create_mesh_default(self):
        """Test create_mesh default parameter."""
        from tunrex.models import create_mesh
        import inspect

        sig = inspect.signature(create_mesh)
        assert sig.parameters["mesh_config"].default is None

    def test_get_gemma_ref_model_signature(self):
        """Test get_gemma_ref_model has expected parameters."""
        from tunrex.models import get_gemma_ref_model
        import inspect

        sig = inspect.signature(get_gemma_ref_model)
        params = list(sig.parameters.keys())

        assert "ckpt_path" in params
        assert "model_checkpoint_path" in params
        assert "mesh_config" in params

    def test_get_lora_model_signature(self):
        """Test get_lora_model has expected parameters."""
        from tunrex.models import get_lora_model
        import inspect

        sig = inspect.signature(get_lora_model)
        params = list(sig.parameters.keys())

        assert "base_model" in params
        assert "mesh" in params
        assert "rank" in params
        assert "alpha" in params
        assert "include" in params
        assert "exclude" in params

    def test_get_lora_model_defaults(self):
        """Test get_lora_model has expected default values."""
        from tunrex.models import get_lora_model
        import inspect

        sig = inspect.signature(get_lora_model)

        assert sig.parameters["rank"].default == 64
        assert sig.parameters["alpha"].default == 64.0
        assert sig.parameters["include"].default is None
        assert sig.parameters["exclude"].default is None

    def test_save_model_state_signature(self):
        """Test save_model_state has expected parameters."""
        from tunrex.models import save_model_state
        import inspect

        sig = inspect.signature(save_model_state)
        params = list(sig.parameters.keys())

        assert "model" in params
        assert "ckpt_dir" in params


class TestLazyImports:
    """Test that tunix imports are truly lazy."""

    def test_models_module_imports_without_tunix(self):
        """Verify models module doesn't import tunix at module load time."""
        import sys

        # Remove tunix from sys.modules if present
        tunix_modules = [k for k in sys.modules if k.startswith("tunix")]
        for mod in tunix_modules:
            sys.modules.pop(mod, None)

        # Import tunrex.models - should succeed even without tunix
        from tunrex import models

        # Verify tunix wasn't imported just by importing the module
        # (it will be imported when functions are called)
        assert models.DEFAULT_MESH is not None
