"""
Reward model wrappers for robustness evaluation.

Provides a unified interface for both internal reward functions
and external HuggingFace reward models.
"""

from abc import ABC, abstractmethod
from typing import List, Dict, Any, Optional, Callable, Protocol, runtime_checkable
from dataclasses import dataclass

from .config import ExternalRewardConfig


@runtime_checkable
class RewardModel(Protocol):
    """Protocol for reward model interface."""

    def score(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs: Any,
    ) -> List[float]:
        """Score a batch of prompt-completion pairs.

        Args:
            prompts: List of input prompts
            completions: List of model completions
            **kwargs: Additional arguments (e.g., answers, rubrics)

        Returns:
            List of reward scores
        """
        ...

    @property
    def name(self) -> str:
        """Name identifier for this reward model."""
        ...


class InternalReward:
    """Wrapper for internal reward functions from src/utils.py."""

    def __init__(self, reward_fn: Callable, name: str):
        """Initialize with a reward function.

        Args:
            reward_fn: The reward function to wrap
            name: Name identifier for this reward
        """
        self._reward_fn = reward_fn
        self._name = name

    def score(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs: Any,
    ) -> List[float]:
        """Score completions using the wrapped reward function."""
        return self._reward_fn(prompts, completions, **kwargs)

    @property
    def name(self) -> str:
        return self._name


class HuggingFaceReward(ABC):
    """Base class for HuggingFace reward models."""

    def __init__(
        self,
        model_id: str,
        device: str = "auto",
        load_in_8bit: bool = True,
        load_in_4bit: bool = False,
        max_length: int = 2048,
        batch_size: int = 8,
    ):
        self.model_id = model_id
        self.device = device
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = None
        self._tokenizer = None

    @abstractmethod
    def _load_model(self) -> None:
        """Load the model and tokenizer."""
        pass

    @abstractmethod
    def score(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs: Any,
    ) -> List[float]:
        """Score completions."""
        pass

    @property
    def name(self) -> str:
        # Return last part of model ID as name
        return self.model_id.split("/")[-1]

    def _resolve_device(self) -> str:
        """Resolve 'auto' device to actual device."""
        if self.device != "auto":
            return self.device

        try:
            import torch

            if torch.cuda.is_available():
                return "cuda"
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return "mps"
        except ImportError:
            pass

        return "cpu"


class SequenceClassificationReward(HuggingFaceReward):
    """Reward model using HuggingFace sequence classification models.

    Works with models like:
    - OpenAssistant/reward-model-deberta-v3-large-v2
    - Other AutoModelForSequenceClassification compatible models
    """

    def _load_model(self) -> None:
        """Load the sequence classification model."""
        if self._model is not None:
            return

        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch are required. "
                "Install with: pip install transformers torch"
            )

        device = self._resolve_device()

        # Quantization config
        quantization_config = None
        if self.load_in_8bit or self.load_in_4bit:
            try:
                from transformers import BitsAndBytesConfig

                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=self.load_in_8bit,
                    load_in_4bit=self.load_in_4bit,
                )
            except ImportError:
                print("Warning: bitsandbytes not available, loading without quantization")

        self._tokenizer = AutoTokenizer.from_pretrained(self.model_id)

        model_kwargs = {"device_map": device if device == "cuda" else None}
        if quantization_config:
            model_kwargs["quantization_config"] = quantization_config

        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, **model_kwargs
        )

        if device != "cuda" or quantization_config is None:
            self._model = self._model.to(device)

        self._model.eval()
        self._device = device

    def score(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs: Any,
    ) -> List[float]:
        """Score using sequence classification logits."""
        self._load_model()

        import torch

        scores = []

        # Process in batches
        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i : i + self.batch_size]
            batch_completions = completions[i : i + self.batch_size]

            # Format as conversation
            texts = [
                f"{prompt}\n\n{completion}"
                for prompt, completion in zip(batch_prompts, batch_completions)
            ]

            inputs = self._tokenizer(
                texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=self.max_length,
            ).to(self._device)

            with torch.no_grad():
                outputs = self._model(**inputs)
                # Get the reward score (usually logits[:, 0] for binary classification)
                if outputs.logits.shape[-1] == 1:
                    batch_scores = outputs.logits.squeeze(-1).cpu().numpy().tolist()
                else:
                    # For multi-class, use first class logit or softmax
                    batch_scores = outputs.logits[:, 0].cpu().numpy().tolist()

            scores.extend(batch_scores)

        return scores


class ArmoRMReward(HuggingFaceReward):
    """ArmoRM reward model from RLHFlow.

    Model: RLHFlow/ArmoRM-Llama3-8B-v0.1
    """

    MODEL_ID = "RLHFlow/ArmoRM-Llama3-8B-v0.1"

    def __init__(
        self,
        device: str = "auto",
        load_in_8bit: bool = True,
        load_in_4bit: bool = False,
        max_length: int = 2048,
        batch_size: int = 4,  # Smaller batch for larger model
    ):
        super().__init__(
            model_id=self.MODEL_ID,
            device=device,
            load_in_8bit=load_in_8bit,
            load_in_4bit=load_in_4bit,
            max_length=max_length,
            batch_size=batch_size,
        )

    def _load_model(self) -> None:
        """Load ArmoRM model."""
        if self._model is not None:
            return

        try:
            from transformers import AutoTokenizer, AutoModelForSequenceClassification
            import torch
        except ImportError:
            raise ImportError(
                "transformers and torch are required. "
                "Install with: pip install transformers torch"
            )

        device = self._resolve_device()

        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id, trust_remote_code=True
        )

        # Quantization config
        model_kwargs = {"trust_remote_code": True}
        if device == "cuda":
            model_kwargs["device_map"] = "auto"
            if self.load_in_8bit:
                model_kwargs["load_in_8bit"] = True
            elif self.load_in_4bit:
                model_kwargs["load_in_4bit"] = True
            else:
                model_kwargs["torch_dtype"] = torch.float16

        self._model = AutoModelForSequenceClassification.from_pretrained(
            self.model_id, **model_kwargs
        )

        if device != "cuda":
            self._model = self._model.to(device)

        self._model.eval()
        self._device = device

    def score(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs: Any,
    ) -> List[float]:
        """Score using ArmoRM."""
        self._load_model()

        import torch

        scores = []

        for i in range(0, len(prompts), self.batch_size):
            batch_prompts = prompts[i : i + self.batch_size]
            batch_completions = completions[i : i + self.batch_size]

            batch_scores = []
            for prompt, completion in zip(batch_prompts, batch_completions):
                # Format as chat message for ArmoRM
                messages = [
                    {"role": "user", "content": prompt},
                    {"role": "assistant", "content": completion},
                ]

                input_ids = self._tokenizer.apply_chat_template(
                    messages,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=self.max_length,
                ).to(self._device)

                with torch.no_grad():
                    outputs = self._model(input_ids)
                    score = outputs.logits[0].item()
                    batch_scores.append(score)

            scores.extend(batch_scores)

        return scores


def load_internal_rewards(reward_names: List[str]) -> List[InternalReward]:
    """Load internal reward functions by name.

    Args:
        reward_names: List of reward function names from src/utils.py

    Returns:
        List of InternalReward wrappers
    """
    # Import reward functions from utils module directly (avoid src.__init__ which requires torch)
    import sys
    from pathlib import Path

    # Ensure src is in path
    src_path = str(Path(__file__).parent.parent)
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    # Import directly from utils module to avoid torch dependency
    from utils import format_reward, accuracy_reward, rubric_reward

    available = {
        "format_reward": format_reward,
        "accuracy_reward": accuracy_reward,
        "rubric_reward": rubric_reward,
    }

    rewards = []
    for name in reward_names:
        if name in available:
            rewards.append(InternalReward(available[name], name))
        else:
            print(f"Warning: Unknown internal reward function: {name}")

    return rewards


def load_external_rewards(
    config: ExternalRewardConfig,
) -> List[HuggingFaceReward]:
    """Load external HuggingFace reward models.

    Args:
        config: External reward configuration

    Returns:
        List of HuggingFaceReward instances
    """
    rewards = []

    for model_id in config.model_ids:
        # Check for known model types
        if "ArmoRM" in model_id:
            rewards.append(
                ArmoRMReward(
                    device=config.device,
                    load_in_8bit=config.load_in_8bit,
                    load_in_4bit=config.load_in_4bit,
                    max_length=config.max_length,
                    batch_size=config.batch_size,
                )
            )
        else:
            # Use generic sequence classification
            rewards.append(
                SequenceClassificationReward(
                    model_id=model_id,
                    device=config.device,
                    load_in_8bit=config.load_in_8bit,
                    load_in_4bit=config.load_in_4bit,
                    max_length=config.max_length,
                    batch_size=config.batch_size,
                )
            )

    return rewards


def load_reward_model(
    name_or_id: str,
    is_external: bool = False,
    **kwargs: Any,
) -> RewardModel:
    """Factory function to load a reward model by name or ID.

    Args:
        name_or_id: Internal function name or HuggingFace model ID
        is_external: Whether this is an external HuggingFace model
        **kwargs: Additional arguments for model initialization

    Returns:
        RewardModel instance
    """
    if is_external:
        config = ExternalRewardConfig(model_ids=[name_or_id], **kwargs)
        rewards = load_external_rewards(config)
        if rewards:
            return rewards[0]
        raise ValueError(f"Failed to load external reward model: {name_or_id}")
    else:
        rewards = load_internal_rewards([name_or_id])
        if rewards:
            return rewards[0]
        raise ValueError(f"Unknown internal reward function: {name_or_id}")
