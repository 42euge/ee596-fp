"""Unit tests for reward_robustness/config.py."""

import pytest
import sys
from pathlib import Path
from dataclasses import fields

# Add src to path to allow direct imports without going through src/__init__.py
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestPerturbationConfig:
    """Tests for PerturbationConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from reward_robustness.config import PerturbationConfig

        config = PerturbationConfig()

        assert config.enabled_types == ["synonym", "paraphrase", "reorder"]
        assert config.num_variants == 5
        assert config.synonym_probability == 0.3
        assert config.synonym_max_replacements == 5
        assert config.paraphrase_model == "humarin/chatgpt_paraphraser_on_T5_base"
        assert config.paraphrase_max_length == 256
        assert config.seed == 42
        assert config.preserve_tags is True

    def test_custom_values(self):
        """Test creating config with custom values."""
        from reward_robustness.config import PerturbationConfig

        config = PerturbationConfig(
            enabled_types=["synonym"],
            num_variants=10,
            synonym_probability=0.5,
            seed=123,
        )

        assert config.enabled_types == ["synonym"]
        assert config.num_variants == 10
        assert config.synonym_probability == 0.5
        assert config.seed == 123

    def test_empty_perturbation_types(self):
        """Test with empty perturbation types list."""
        from reward_robustness.config import PerturbationConfig

        config = PerturbationConfig(enabled_types=[])
        assert config.enabled_types == []


class TestExternalRewardConfig:
    """Tests for ExternalRewardConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from reward_robustness.config import ExternalRewardConfig

        config = ExternalRewardConfig()

        assert config.model_ids == []
        assert config.device == "auto"
        assert config.load_in_8bit is True
        assert config.load_in_4bit is False
        assert config.batch_size == 8
        assert config.max_length == 2048
        assert config.cache_dir is None

    def test_with_model_ids(self):
        """Test with external model IDs specified."""
        from reward_robustness.config import ExternalRewardConfig

        config = ExternalRewardConfig(
            model_ids=["model/a", "model/b"],
            device="cuda",
            load_in_8bit=False,
        )

        assert config.model_ids == ["model/a", "model/b"]
        assert config.device == "cuda"
        assert config.load_in_8bit is False


class TestRobustnessConfig:
    """Tests for main RobustnessConfig dataclass."""

    def test_default_values(self):
        """Test default configuration values."""
        from reward_robustness.config import RobustnessConfig

        config = RobustnessConfig()

        assert config.internal_rewards == ["format_reward", "accuracy_reward"]
        assert config.num_samples == 100
        assert config.output_dir == "./robustness_results"
        assert config.save_detailed is True
        assert config.variance_threshold == 0.5
        assert config.flip_threshold == 0.0

    def test_nested_configs(self):
        """Test that nested configs are properly initialized."""
        from reward_robustness.config import (
            RobustnessConfig,
            PerturbationConfig,
            ExternalRewardConfig,
        )

        config = RobustnessConfig()

        assert isinstance(config.perturbations, PerturbationConfig)
        assert isinstance(config.external_rewards, ExternalRewardConfig)

    def test_custom_nested_configs(self):
        """Test creating config with custom nested configs."""
        from reward_robustness.config import (
            RobustnessConfig,
            PerturbationConfig,
            ExternalRewardConfig,
        )

        config = RobustnessConfig(
            internal_rewards=["format_reward"],
            external_rewards=ExternalRewardConfig(model_ids=["test/model"]),
            perturbations=PerturbationConfig(num_variants=3),
            num_samples=50,
        )

        assert config.internal_rewards == ["format_reward"]
        assert config.external_rewards.model_ids == ["test/model"]
        assert config.perturbations.num_variants == 3
        assert config.num_samples == 50


class TestConfigFactoryFunctions:
    """Tests for configuration factory functions."""

    def test_get_default_config(self):
        """Test get_default_config returns valid config."""
        from reward_robustness.config import get_default_config, RobustnessConfig

        config = get_default_config()

        assert isinstance(config, RobustnessConfig)
        assert config.num_samples == 100

    def test_get_quick_config(self):
        """Test get_quick_config returns smaller evaluation config."""
        from reward_robustness.config import get_quick_config

        config = get_quick_config()

        assert config.num_samples == 20
        assert config.perturbations.num_variants == 3
        assert "synonym" in config.perturbations.enabled_types

    def test_get_thorough_config(self):
        """Test get_thorough_config returns comprehensive config."""
        from reward_robustness.config import get_thorough_config

        config = get_thorough_config()

        assert config.num_samples == 500
        assert config.perturbations.num_variants == 10
        assert len(config.perturbations.enabled_types) == 3
        assert config.save_detailed is True


class TestConfigSerialization:
    """Tests for config serialization compatibility."""

    def test_config_to_dict(self):
        """Test that configs can be converted to dictionaries."""
        from dataclasses import asdict
        from reward_robustness.config import RobustnessConfig

        config = RobustnessConfig()
        config_dict = asdict(config)

        assert isinstance(config_dict, dict)
        assert "internal_rewards" in config_dict
        assert "perturbations" in config_dict
        assert isinstance(config_dict["perturbations"], dict)

    def test_config_fields(self):
        """Test that all expected fields are present."""
        from reward_robustness.config import RobustnessConfig

        field_names = {f.name for f in fields(RobustnessConfig)}

        expected_fields = {
            "internal_rewards",
            "external_rewards",
            "perturbations",
            "num_samples",
            "output_dir",
            "save_detailed",
            "variance_threshold",
            "flip_threshold",
        }

        assert expected_fields.issubset(field_names)
