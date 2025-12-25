"""Tests for tunrex.datasets.config module."""

import pytest
import sys
from pathlib import Path

# Import directly from the module file to avoid grain dependency in __init__.py
tunrex_src = Path(__file__).parent.parent / "src" / "tunrex" / "datasets"
import importlib.util
config_spec = importlib.util.spec_from_file_location("config", tunrex_src / "config.py")
config_module = importlib.util.module_from_spec(config_spec)
sys.modules["tunrex.datasets.config"] = config_module
config_spec.loader.exec_module(config_module)

from tunrex.datasets.config import (
    reasoning_start,
    reasoning_end,
    solution_start,
    solution_end,
    REASONING_START,
    REASONING_END,
    SOLUTION_START,
    SOLUTION_END,
    DEFAULT_TEMPLATE,
    DEFAULT_SYSTEM_PROMPT,
    SYSTEM_PROMPTS,
    get_system_prompt,
    TunRexConfig,
)


class TestTagConstants:
    """Tests for tag constant definitions."""

    def test_lowercase_tags(self):
        """Test lowercase tag constants."""
        assert reasoning_start == "<reasoning>"
        assert reasoning_end == "</reasoning>"
        assert solution_start == "<answer>"
        assert solution_end == "</answer>"

    def test_uppercase_aliases(self):
        """Test uppercase aliases match lowercase."""
        assert REASONING_START == reasoning_start
        assert REASONING_END == reasoning_end
        assert SOLUTION_START == solution_start
        assert SOLUTION_END == solution_end


class TestDefaultTemplate:
    """Tests for DEFAULT_TEMPLATE."""

    def test_contains_placeholders(self):
        """Test template contains required placeholders."""
        assert "{system_prompt}" in DEFAULT_TEMPLATE
        assert "{question}" in DEFAULT_TEMPLATE

    def test_contains_turn_markers(self):
        """Test template contains Gemma turn markers."""
        assert "<start_of_turn>user" in DEFAULT_TEMPLATE
        assert "<end_of_turn>" in DEFAULT_TEMPLATE
        assert "<start_of_turn>model" in DEFAULT_TEMPLATE

    def test_template_formatting(self):
        """Test template can be formatted."""
        result = DEFAULT_TEMPLATE.format(
            system_prompt="Be helpful",
            question="What is 2+2?"
        )
        assert "Be helpful" in result
        assert "What is 2+2?" in result


class TestDefaultSystemPrompt:
    """Tests for DEFAULT_SYSTEM_PROMPT."""

    def test_contains_reasoning_tags(self):
        """Test system prompt mentions reasoning tags."""
        assert reasoning_start in DEFAULT_SYSTEM_PROMPT
        assert reasoning_end in DEFAULT_SYSTEM_PROMPT

    def test_contains_answer_tags(self):
        """Test system prompt mentions answer tags."""
        assert solution_start in DEFAULT_SYSTEM_PROMPT
        assert solution_end in DEFAULT_SYSTEM_PROMPT


class TestSystemPrompts:
    """Tests for SYSTEM_PROMPTS dictionary."""

    def test_all_versions_exist(self):
        """Test all 7 versions exist (0-6)."""
        for version in range(7):
            assert version in SYSTEM_PROMPTS
            assert isinstance(SYSTEM_PROMPTS[version], str)
            assert len(SYSTEM_PROMPTS[version]) > 0

    def test_version_0_is_default(self):
        """Test version 0 matches DEFAULT_SYSTEM_PROMPT."""
        assert SYSTEM_PROMPTS[0] == DEFAULT_SYSTEM_PROMPT

    def test_all_versions_contain_tags(self):
        """Test all versions reference reasoning and answer tags."""
        for version, prompt in SYSTEM_PROMPTS.items():
            assert reasoning_start in prompt or "reasoning" in prompt.lower()
            assert solution_start in prompt or "answer" in prompt.lower()


class TestGetSystemPrompt:
    """Tests for get_system_prompt function."""

    def test_valid_versions(self):
        """Test getting valid version prompts."""
        for version in range(7):
            prompt = get_system_prompt(version)
            assert prompt == SYSTEM_PROMPTS[version]

    def test_default_is_version_0(self):
        """Test default returns version 0."""
        prompt = get_system_prompt()
        assert prompt == SYSTEM_PROMPTS[0]

    def test_invalid_version_returns_default(self):
        """Test invalid version returns version 0."""
        prompt = get_system_prompt(999)
        assert prompt == SYSTEM_PROMPTS[0]

    def test_negative_version_returns_default(self):
        """Test negative version returns version 0."""
        prompt = get_system_prompt(-1)
        assert prompt == SYSTEM_PROMPTS[0]


class TestTunRexConfig:
    """Tests for TunRexConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        config = TunRexConfig()

        assert config.source == "huggingface"
        assert config.dataset_name == "gsm8k"
        assert config.hf_subset == "main"
        assert config.train_data_dir == "./data/train"
        assert config.test_data_dir == "./data/test"
        assert config.batch_size == 4
        assert config.train_fraction == 0.8
        assert config.val_fraction == 0.1
        assert config.num_epochs == 1
        assert config.shuffle_seed == 42
        assert config.max_train_batches is None
        assert config.max_test_batches == 100
        assert config.max_examples is None
        assert config.apply_template is True
        assert config.answer_extractor is None

    def test_custom_values(self):
        """Test custom configuration values."""
        config = TunRexConfig(
            source="kaggle",
            dataset_name="my-dataset",
            batch_size=8,
            train_fraction=0.9,
        )
        assert config.source == "kaggle"
        assert config.dataset_name == "my-dataset"
        assert config.batch_size == 8
        assert config.train_fraction == 0.9

    def test_template_defaults(self):
        """Test template defaults."""
        config = TunRexConfig()
        assert config.template == DEFAULT_TEMPLATE
        assert config.system_prompt == DEFAULT_SYSTEM_PROMPT


class TestTunRexConfigValidation:
    """Tests for TunRexConfig validation in __post_init__."""

    def test_fraction_sum_validation(self):
        """Test that train + val fraction cannot exceed 1.0."""
        with pytest.raises(ValueError, match="cannot exceed 1.0"):
            TunRexConfig(train_fraction=0.8, val_fraction=0.3)

    def test_fraction_sum_at_limit(self):
        """Test that train + val fraction can equal 1.0."""
        config = TunRexConfig(train_fraction=0.8, val_fraction=0.2)
        assert config.train_fraction + config.val_fraction == 1.0

    def test_invalid_source(self):
        """Test that invalid source raises error."""
        with pytest.raises(ValueError, match="Invalid source"):
            TunRexConfig(source="invalid_source")

    def test_valid_sources(self):
        """Test all valid source types."""
        for source in ["tfds", "kaggle", "huggingface"]:
            config = TunRexConfig(source=source)
            assert config.source == source


class TestTunRexConfigGSM8KPreset:
    """Tests for TunRexConfig.gsm8k() preset."""

    def test_default_gsm8k(self):
        """Test default GSM8K preset."""
        config = TunRexConfig.gsm8k()

        assert config.source == "huggingface"
        assert config.dataset_name == "gsm8k"
        assert config.hf_subset == "main"

    def test_gsm8k_with_kaggle_source(self):
        """Test GSM8K preset with Kaggle source."""
        config = TunRexConfig.gsm8k(source="kaggle")

        assert config.source == "kaggle"
        assert config.dataset_name == "gsm8k"

    def test_gsm8k_with_custom_params(self):
        """Test GSM8K preset with custom parameters."""
        config = TunRexConfig.gsm8k(batch_size=16, max_examples=1000)

        assert config.dataset_name == "gsm8k"
        assert config.batch_size == 16
        assert config.max_examples == 1000


class TestTunRexConfigOpenRubricsPreset:
    """Tests for TunRexConfig.openrubrics() preset."""

    def test_default_openrubrics(self):
        """Test default OpenRubrics preset."""
        config = TunRexConfig.openrubrics()

        assert config.source == "huggingface"
        assert config.dataset_name == "OpenRubrics/OpenRubrics"
        assert config.hf_subset == "train"
        assert config.max_examples == 2000

    def test_openrubrics_with_custom_split(self):
        """Test OpenRubrics preset with custom split."""
        config = TunRexConfig.openrubrics(split="test")

        assert config.hf_subset == "test"

    def test_openrubrics_with_custom_max_examples(self):
        """Test OpenRubrics preset with custom max_examples."""
        config = TunRexConfig.openrubrics(max_examples=500)

        assert config.max_examples == 500

    def test_openrubrics_with_additional_params(self):
        """Test OpenRubrics preset with additional parameters."""
        config = TunRexConfig.openrubrics(
            split="validation",
            max_examples=100,
            batch_size=8,
        )

        assert config.hf_subset == "validation"
        assert config.max_examples == 100
        assert config.batch_size == 8
