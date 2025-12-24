"""Tests for src/config.py module.

Note: We mock torch before importing to avoid requiring it in CI.
"""

import pytest
import sys
from unittest.mock import MagicMock

# Mock torch and its submodules before importing src (which imports src.model which needs torch)
sys.modules["torch"] = MagicMock()
sys.modules["torch.cuda"] = MagicMock()
sys.modules["torch.backends"] = MagicMock()
sys.modules["torch.backends.mps"] = MagicMock()
sys.modules["transformers"] = MagicMock()

from src.config import (
    LoRAConfig,
    GRPOConfig,
    TrainingConfig,
    DataConfig,
    InferenceConfig,
    Config,
    REASONING_START,
    REASONING_END,
    SOLUTION_START,
    SOLUTION_END,
    SYSTEM_PROMPTS,
    get_default_config,
    get_system_prompt,
    format_prompt,
)


class TestLoRAConfig:
    """Tests for LoRAConfig dataclass."""

    def test_default_values(self):
        """Test default LoRA configuration values."""
        config = LoRAConfig()
        assert config.rank == 64
        assert config.alpha == 64.0

    def test_custom_values(self):
        """Test custom LoRA configuration."""
        config = LoRAConfig(rank=32, alpha=16.0)
        assert config.rank == 32
        assert config.alpha == 16.0


class TestGRPOConfig:
    """Tests for GRPOConfig dataclass."""

    def test_default_values(self):
        """Test default GRPO configuration values."""
        config = GRPOConfig()
        assert config.max_prompt_length == 256
        assert config.total_generation_steps == 512
        assert config.temperature == 0.9
        assert config.top_p == 1.0
        assert config.top_k == 50
        assert config.num_generations == 2
        assert config.num_iterations == 1
        assert config.beta == 0.08
        assert config.epsilon == 0.2

    def test_custom_values(self):
        """Test custom GRPO configuration."""
        config = GRPOConfig(temperature=0.7, top_k=100, beta=0.1)
        assert config.temperature == 0.7
        assert config.top_k == 100
        assert config.beta == 0.1


class TestTrainingConfig:
    """Tests for TrainingConfig dataclass."""

    def test_default_values(self):
        """Test default training configuration values."""
        config = TrainingConfig()
        assert config.train_micro_batch_size == 2
        assert config.num_batches == 1000
        assert config.num_test_batches == 100
        assert config.eval_every_n_steps == 10
        assert config.num_epochs == 1
        assert config.learning_rate == 3e-6
        assert config.beta1 == 0.9
        assert config.beta2 == 0.99
        assert config.weight_decay == 0.1
        assert config.warmup_ratio == 0.1
        assert config.max_grad_norm == 0.1
        assert config.save_interval_steps == 500
        assert config.max_checkpoints_to_keep == 4


class TestDataConfig:
    """Tests for DataConfig dataclass."""

    def test_default_values(self):
        """Test default data configuration values."""
        config = DataConfig()
        assert config.train_data_dir == "./data/train"
        assert config.test_data_dir == "./data/test"
        assert config.train_fraction == 0.942
        assert config.eval_fraction == 0.22
        assert config.use_gsm8k is False
        assert config.use_openrubrics is True
        assert config.openrubrics_max_examples == 2000
        assert config.openrubrics_split == "train"


class TestInferenceConfig:
    """Tests for InferenceConfig dataclass."""

    def test_default_values(self):
        """Test default inference configuration values."""
        config = InferenceConfig()

        # Check greedy config
        assert config.greedy["temperature"] == 1e-4
        assert config.greedy["top_k"] == 1
        assert config.greedy["top_p"] == 1.0

        # Check standard config
        assert config.standard["temperature"] == 0.7
        assert config.standard["top_k"] == 50
        assert config.standard["top_p"] == 0.95

        # Check liberal config
        assert config.liberal["temperature"] == 0.85
        assert config.liberal["top_k"] == 2000
        assert config.liberal["top_p"] == 1.0


class TestConfig:
    """Tests for the master Config dataclass."""

    def test_default_values(self):
        """Test default master configuration values."""
        config = Config()

        assert isinstance(config.lora, LoRAConfig)
        assert isinstance(config.grpo, GRPOConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.inference, InferenceConfig)

        assert config.checkpoint_dir == "./checkpoints"
        assert config.intermediate_checkpoint_dir == "./checkpoints/intermediate"
        assert config.results_dir == "./results"
        assert config.system_prompt_version == 2
        assert config.use_rubric_evaluation is True
        assert config.device == "auto"

    def test_nested_config_access(self):
        """Test accessing nested configuration values."""
        config = Config()
        assert config.lora.rank == 64
        assert config.grpo.temperature == 0.9
        assert config.training.learning_rate == 3e-6


class TestPromptConstants:
    """Tests for prompt-related constants."""

    def test_tag_constants(self):
        """Test that tag constants are defined correctly."""
        assert REASONING_START == "<reasoning>"
        assert REASONING_END == "</reasoning>"
        assert SOLUTION_START == "<answer>"
        assert SOLUTION_END == "</answer>"

    def test_system_prompts_exist(self):
        """Test that all system prompt versions exist."""
        for version in range(7):
            assert version in SYSTEM_PROMPTS
            assert isinstance(SYSTEM_PROMPTS[version], str)
            assert len(SYSTEM_PROMPTS[version]) > 0

    def test_system_prompts_contain_tags(self):
        """Test that system prompts reference the correct tags."""
        for version, prompt in SYSTEM_PROMPTS.items():
            assert REASONING_START in prompt or "reasoning" in prompt.lower()
            assert SOLUTION_START in prompt or "answer" in prompt.lower()


class TestGetDefaultConfig:
    """Tests for get_default_config function."""

    def test_returns_config_instance(self):
        """Test that get_default_config returns a Config instance."""
        config = get_default_config()
        assert isinstance(config, Config)

    def test_returns_defaults(self):
        """Test that returned config has default values."""
        config = get_default_config()
        assert config.lora.rank == 64
        assert config.device == "auto"


class TestGetSystemPrompt:
    """Tests for get_system_prompt function."""

    def test_valid_versions(self):
        """Test getting prompts for all valid versions."""
        for version in range(7):
            prompt = get_system_prompt(version)
            assert isinstance(prompt, str)
            assert len(prompt) > 0

    def test_default_version(self):
        """Test default version is 2."""
        prompt = get_system_prompt()
        assert prompt == SYSTEM_PROMPTS[2]

    def test_invalid_version_returns_default(self):
        """Test that invalid version returns version 2."""
        prompt = get_system_prompt(999)
        assert prompt == SYSTEM_PROMPTS[2]

    def test_negative_version_returns_default(self):
        """Test that negative version returns version 2."""
        prompt = get_system_prompt(-1)
        assert prompt == SYSTEM_PROMPTS[2]


class TestFormatPrompt:
    """Tests for format_prompt function."""

    def test_basic_prompt(self):
        """Test basic prompt formatting."""
        question = "What is 2 + 2?"
        result = format_prompt(question)

        assert "<start_of_turn>user" in result
        assert question in result
        assert "<end_of_turn>" in result
        assert "<start_of_turn>model" in result

    def test_with_custom_system_prompt(self):
        """Test prompt with custom system prompt."""
        question = "What is 2 + 2?"
        system = "You are a helpful math tutor."
        result = format_prompt(question, system_prompt=system)

        assert system in result
        assert question in result

    def test_with_rubric(self):
        """Test prompt with rubric."""
        question = "Explain photosynthesis."
        rubric = "Must mention: sunlight, chlorophyll"
        result = format_prompt(question, rubric=rubric)

        assert "Rubric:" in result
        assert rubric in result
        assert question in result

    def test_with_all_parameters(self):
        """Test prompt with all parameters."""
        question = "What is 2 + 2?"
        system = "You are a math tutor."
        rubric = "Show your work."
        result = format_prompt(question, system_prompt=system, rubric=rubric)

        assert system in result
        assert "Rubric:" in result
        assert rubric in result
        assert question in result

    def test_default_system_prompt(self):
        """Test that default system prompt is used when None."""
        question = "What is 2 + 2?"
        result = format_prompt(question, system_prompt=None)

        # Should contain content from version 2 system prompt
        assert "<reasoning>" in result or "reasoning" in result.lower()
