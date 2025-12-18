"""Tests for tunrex.datasets module."""

import pytest


class TestLoaders:
    """Test dataset loader functions."""

    def test_extract_hash_answer_valid(self):
        """Test extracting answer after #### delimiter."""
        from tunrex.datasets import extract_hash_answer

        result = extract_hash_answer("Some reasoning #### 42")
        assert result == "42"

    def test_extract_hash_answer_with_whitespace(self):
        """Test extracting answer handles whitespace."""
        from tunrex.datasets import extract_hash_answer

        result = extract_hash_answer("Calculation steps ####   answer here  ")
        assert result == "answer here"

    def test_extract_hash_answer_missing_delimiter(self):
        """Test returns None when delimiter is missing."""
        from tunrex.datasets import extract_hash_answer

        result = extract_hash_answer("No delimiter here")
        assert result is None

    def test_extract_hash_answer_empty_after_delimiter(self):
        """Test handles empty string after delimiter."""
        from tunrex.datasets import extract_hash_answer

        result = extract_hash_answer("Text #### ")
        assert result == ""


class TestTunRexConfig:
    """Test TunRexConfig class."""

    def test_default_config(self):
        """Test default configuration values."""
        from tunrex.datasets import TunRexConfig

        config = TunRexConfig()
        assert config.train_data_dir is not None
        assert config.shuffle_seed == 42
        assert config.batch_size == 4

    def test_custom_config(self):
        """Test custom configuration."""
        from tunrex.datasets import TunRexConfig

        config = TunRexConfig(train_data_dir="/custom/path", shuffle_seed=123)
        assert config.train_data_dir == "/custom/path"
        assert config.shuffle_seed == 123

    def test_gsm8k_preset(self):
        """Test GSM8K preset configuration."""
        from tunrex.datasets import TunRexConfig

        config = TunRexConfig.gsm8k()
        assert config.dataset_name == "gsm8k"
        assert config.hf_subset == "main"

    def test_openrubrics_preset(self):
        """Test OpenRubrics preset configuration."""
        from tunrex.datasets import TunRexConfig

        config = TunRexConfig.openrubrics(max_examples=100)
        assert config.dataset_name == "OpenRubrics/OpenRubrics"
        assert config.max_examples == 100


class TestLoadOpenRubrics:
    """Test load_openrubrics function."""

    def test_function_signature(self):
        """Test function has expected parameters."""
        from tunrex.datasets import load_openrubrics
        import inspect

        sig = inspect.signature(load_openrubrics)
        params = list(sig.parameters.keys())

        assert "split" in params
        assert "max_examples" in params

    @pytest.mark.skip(reason="Requires network access")
    def test_load_openrubrics_returns_list(self):
        """Test that load_openrubrics returns a list of dicts."""
        from tunrex.datasets import load_openrubrics

        data = load_openrubrics(max_examples=5)
        assert isinstance(data, list)
        if len(data) > 0:
            assert isinstance(data[0], dict)
            assert "question" in data[0]
