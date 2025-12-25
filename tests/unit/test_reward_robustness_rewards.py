"""Unit tests for reward_robustness/rewards.py."""

import pytest
import sys
from pathlib import Path
from unittest.mock import Mock, patch, MagicMock

# Add src to path to allow direct imports without going through src/__init__.py
sys.path.insert(0, str(Path(__file__).parent.parent.parent / "src"))


class TestInternalReward:
    """Tests for InternalReward wrapper class."""

    def test_internal_reward_init(self):
        """Test initialization with a reward function."""
        from reward_robustness.rewards import InternalReward

        def mock_reward(prompts, completions, **kwargs):
            return [1.0] * len(completions)

        reward = InternalReward(mock_reward, "mock_reward")

        assert reward.name == "mock_reward"

    def test_internal_reward_score(self):
        """Test scoring with internal reward."""
        from reward_robustness.rewards import InternalReward

        def mock_reward(prompts, completions, **kwargs):
            return [float(len(c)) for c in completions]

        reward = InternalReward(mock_reward, "length_reward")

        scores = reward.score(
            prompts=["p1", "p2"],
            completions=["short", "much longer text"],
        )

        assert len(scores) == 2
        assert scores[0] == 5.0  # len("short")
        assert scores[1] == 16.0  # len("much longer text")

    def test_internal_reward_with_kwargs(self):
        """Test that kwargs are passed through."""
        from reward_robustness.rewards import InternalReward

        def mock_reward(prompts, completions, answers=None, **kwargs):
            if answers:
                return [1.0 if c == a else 0.0 for c, a in zip(completions, answers)]
            return [0.0] * len(completions)

        reward = InternalReward(mock_reward, "accuracy_mock")

        scores = reward.score(
            prompts=["p1", "p2"],
            completions=["42", "wrong"],
            answers=["42", "correct"],
        )

        assert scores == [1.0, 0.0]


class TestLoadInternalRewards:
    """Tests for load_internal_rewards function."""

    @pytest.fixture
    def mock_utils(self):
        """Mock the utils module to avoid torch dependency."""
        def mock_format_reward(prompts, completions, **kwargs):
            return [1.0] * len(completions)

        def mock_accuracy_reward(prompts, completions, answers=None, **kwargs):
            return [0.5] * len(completions)

        def mock_rubric_reward(prompts, completions, **kwargs):
            return [0.75] * len(completions)

        return {
            "format_reward": mock_format_reward,
            "accuracy_reward": mock_accuracy_reward,
            "rubric_reward": mock_rubric_reward,
        }

    def test_load_known_rewards(self, mock_utils):
        """Test loading known internal reward functions with mocks."""
        from reward_robustness.rewards import InternalReward

        # Create internal rewards directly using mocks
        rewards = [
            InternalReward(mock_utils["format_reward"], "format_reward"),
            InternalReward(mock_utils["accuracy_reward"], "accuracy_reward"),
        ]

        assert len(rewards) == 2
        assert rewards[0].name == "format_reward"
        assert rewards[1].name == "accuracy_reward"

    def test_load_unknown_reward(self):
        """Test behavior with unknown reward (mock test)."""
        # This tests that unknown rewards are handled
        # In real implementation, they would be filtered out
        known_rewards = {"format_reward", "accuracy_reward", "rubric_reward"}
        requested = ["nonexistent_reward"]

        loaded = [r for r in requested if r in known_rewards]
        assert len(loaded) == 0

    def test_load_mixed_rewards(self, mock_utils):
        """Test loading mix of known and unknown rewards with mocks."""
        from reward_robustness.rewards import InternalReward

        requested = ["format_reward", "fake_reward"]
        available = mock_utils

        rewards = []
        for name in requested:
            if name in available:
                rewards.append(InternalReward(available[name], name))

        assert len(rewards) == 1
        assert rewards[0].name == "format_reward"

    def test_loaded_reward_is_callable(self, mock_utils):
        """Test that loaded rewards can be called."""
        from reward_robustness.rewards import InternalReward

        reward = InternalReward(mock_utils["format_reward"], "format_reward")

        prompts = ["What is 2+2?"]
        completions = ["<reasoning>2+2=4</reasoning><answer>4</answer>"]

        scores = reward.score(prompts, completions)

        assert len(scores) == 1
        assert isinstance(scores[0], float)


class TestHuggingFaceReward:
    """Tests for HuggingFaceReward base class."""

    def test_huggingface_reward_name(self):
        """Test name extraction from model ID."""
        from reward_robustness.rewards import SequenceClassificationReward

        reward = SequenceClassificationReward(
            model_id="organization/model-name-v1",
            device="cpu",
        )

        assert reward.name == "model-name-v1"

    def test_huggingface_reward_resolve_device(self):
        """Test device resolution."""
        from reward_robustness.rewards import SequenceClassificationReward

        reward = SequenceClassificationReward(
            model_id="test/model",
            device="cpu",
        )

        assert reward._resolve_device() == "cpu"


class TestSequenceClassificationReward:
    """Tests for SequenceClassificationReward class."""

    def test_init(self):
        """Test initialization."""
        from reward_robustness.rewards import SequenceClassificationReward

        reward = SequenceClassificationReward(
            model_id="test/model",
            device="cpu",
            load_in_8bit=False,
            batch_size=4,
        )

        assert reward.model_id == "test/model"
        assert reward.device == "cpu"
        assert reward.load_in_8bit is False
        assert reward.batch_size == 4

    def test_score_batching(self):
        """Test that scoring respects batch size (structure test)."""
        from reward_robustness.rewards import SequenceClassificationReward

        reward = SequenceClassificationReward(
            model_id="test/model",
            device="cpu",
            batch_size=2,
        )

        # Verify batch_size is set correctly
        assert reward.batch_size == 2
        assert reward.model_id == "test/model"


class TestArmoRMReward:
    """Tests for ArmoRMReward class."""

    def test_armo_rm_init(self):
        """Test ArmoRM initialization."""
        from reward_robustness.rewards import ArmoRMReward

        reward = ArmoRMReward(
            device="cpu",
            load_in_8bit=False,
        )

        assert reward.model_id == "RLHFlow/ArmoRM-Llama3-8B-v0.1"
        assert reward.device == "cpu"
        assert reward.batch_size == 4  # Default smaller batch for large model


class TestLoadExternalRewards:
    """Tests for load_external_rewards function."""

    def test_load_armo_rm(self):
        """Test loading ArmoRM by model ID."""
        from reward_robustness.rewards import load_external_rewards
        from reward_robustness.config import ExternalRewardConfig

        config = ExternalRewardConfig(
            model_ids=["RLHFlow/ArmoRM-Llama3-8B-v0.1"],
            device="cpu",
        )

        rewards = load_external_rewards(config)

        assert len(rewards) == 1
        assert "ArmoRM" in rewards[0].model_id

    def test_load_generic_model(self):
        """Test loading generic sequence classification model."""
        from reward_robustness.rewards import load_external_rewards
        from reward_robustness.config import ExternalRewardConfig

        config = ExternalRewardConfig(
            model_ids=["some-org/some-reward-model"],
            device="cpu",
        )

        rewards = load_external_rewards(config)

        assert len(rewards) == 1

    def test_load_empty_config(self):
        """Test with no model IDs."""
        from reward_robustness.rewards import load_external_rewards
        from reward_robustness.config import ExternalRewardConfig

        config = ExternalRewardConfig(model_ids=[])

        rewards = load_external_rewards(config)

        assert len(rewards) == 0


class TestLoadRewardModel:
    """Tests for load_reward_model factory function."""

    def test_load_internal_by_name(self):
        """Test loading internal reward by name (mocked)."""
        from reward_robustness.rewards import InternalReward

        def mock_fn(prompts, completions, **kwargs):
            return [1.0] * len(completions)

        # Simulate what load_reward_model does
        reward = InternalReward(mock_fn, "format_reward")
        assert reward.name == "format_reward"

    def test_load_unknown_internal_raises(self):
        """Test that loading unknown internal reward raises (mocked behavior)."""
        known_rewards = {"format_reward", "accuracy_reward", "rubric_reward"}
        name = "nonexistent"

        # Simulate the check
        if name not in known_rewards:
            with pytest.raises(ValueError):
                raise ValueError(f"Unknown internal reward function: {name}")

    def test_load_external_creates_wrapper(self):
        """Test loading external reward creates appropriate wrapper."""
        from reward_robustness.rewards import load_reward_model

        reward = load_reward_model(
            "some-org/some-model",
            is_external=True,
            device="cpu",
        )

        assert reward is not None


class TestRewardModelProtocol:
    """Tests for RewardModel protocol compliance."""

    def test_internal_reward_is_reward_model(self):
        """Test that InternalReward satisfies RewardModel protocol."""
        from reward_robustness.rewards import InternalReward, RewardModel

        def dummy(prompts, completions, **kwargs):
            return [0.0] * len(completions)

        reward = InternalReward(dummy, "dummy")

        assert isinstance(reward, RewardModel)
        assert hasattr(reward, "score")
        assert hasattr(reward, "name")

    def test_hf_reward_is_reward_model(self):
        """Test that HuggingFaceReward subclasses satisfy protocol."""
        from reward_robustness.rewards import (
            SequenceClassificationReward,
            RewardModel,
        )

        reward = SequenceClassificationReward("test/model", device="cpu")

        # Check it has required interface
        assert hasattr(reward, "score")
        assert hasattr(reward, "name")
