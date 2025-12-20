"""Configuration classes for TunRex dataset loading."""

from dataclasses import dataclass, field
from typing import Callable, Literal, Optional


# Reasoning and answer tag definitions
reasoning_start = "<reasoning>"
reasoning_end = "</reasoning>"
solution_start = "<answer>"
solution_end = "</answer>"

# Uppercase aliases
REASONING_START = reasoning_start
REASONING_END = reasoning_end
SOLUTION_START = solution_start
SOLUTION_END = solution_end

# Default template for Gemma-style models
DEFAULT_TEMPLATE = """\
<start_of_turn>user
{system_prompt}

{question}<end_of_turn>
<start_of_turn>model
"""

# Baseline system prompt (version 0) - exact match for Gemma GRPO training
DEFAULT_SYSTEM_PROMPT = f"""\
You are given a problem. First, think about the problem and provide your reasoning. \
Place it between {reasoning_start} and {reasoning_end}. \
Then, provide the final answer (i.e., just one numerical value) between {solution_start} and {solution_end}."""

# System prompt variants
SYSTEM_PROMPTS = {
    0: DEFAULT_SYSTEM_PROMPT,
    1: f"You are given a problem. Think about the problem and provide your reasoning. Place it between {reasoning_start} and {reasoning_end}. Then, provide the final answer between {solution_start} and {solution_end}.",
    2: f"You are given a problem. Think carefully and show your detailed reasoning step-by-step. Place your reasoning between {reasoning_start} and {reasoning_end}. After completing your reasoning, provide the final answer between {solution_start} and {solution_end}.",
    3: f"You are given a problem. Let's think step by step. Provide your reasoning process between {reasoning_start} and {reasoning_end}. Then provide the final answer between {solution_start} and {solution_end}.",
    4: f"You are given a problem. First, understand what is being asked. Then, work through your reasoning carefully. Place your reasoning between {reasoning_start} and {reasoning_end}. Finally, provide your answer between {solution_start} and {solution_end}.",
    5: f"You are given a problem. Consider different approaches to solve it. Think through your reasoning, exploring multiple paths if helpful. Place your complete reasoning between {reasoning_start} and {reasoning_end}. Then provide the best final answer between {solution_start} and {solution_end}.",
    6: f"Solve the problem below. Show your work in {reasoning_start}{reasoning_end}. Give your answer in {solution_start}{solution_end}.",
}


def get_system_prompt(version: int = 0) -> str:
    """Get system prompt by version number."""
    return SYSTEM_PROMPTS.get(version, SYSTEM_PROMPTS[0])


@dataclass
class TunRexConfig:
    """Configuration for TunRex dataset loading and preparation.

    Attributes:
        source: Data source type ("tfds", "kaggle", or "huggingface")
        dataset_name: Name of the dataset to load (e.g., "gsm8k", "OpenRubrics/OpenRubrics")
        train_data_dir: Directory for training data cache
        test_data_dir: Directory for test data cache
        batch_size: Number of examples per batch
        train_fraction: Fraction of data for training (0.0 to 1.0)
        val_fraction: Fraction of data for validation (0.0 to 1.0)
        num_epochs: Number of epochs to repeat training data
        shuffle_seed: Random seed for shuffling
        max_train_batches: Maximum number of training batches (None for unlimited)
        max_test_batches: Maximum number of test batches
        max_examples: Maximum total examples to load (None for unlimited)
        template: Prompt template string with {system_prompt} and {question} placeholders
        system_prompt: System prompt to use in template
        answer_extractor: Function to extract answer from raw answer text
        apply_template: Whether to apply template formatting to prompts
    """

    # Data source configuration
    source: Literal["tfds", "kaggle", "huggingface"] = "huggingface"
    dataset_name: str = "gsm8k"
    hf_subset: str | None = "main"  # HuggingFace dataset subset (e.g., "main" for gsm8k)

    # Paths
    train_data_dir: str = "./data/train"
    test_data_dir: str = "./data/test"

    # Batching & splitting
    batch_size: int = 4
    train_fraction: float = 0.8
    val_fraction: float = 0.1
    num_epochs: int = 1
    shuffle_seed: int = 42

    # Limits
    max_train_batches: int | None = None
    max_test_batches: int = 100
    max_examples: int | None = None

    # Template & prompt configuration
    template: str = field(default_factory=lambda: DEFAULT_TEMPLATE)
    system_prompt: str = field(default_factory=lambda: DEFAULT_SYSTEM_PROMPT)
    apply_template: bool = True

    # Custom answer extraction (defaults to GSM8K hash extraction)
    answer_extractor: Callable[[str], str | None] | None = None

    def __post_init__(self):
        """Validate configuration after initialization."""
        if self.train_fraction + self.val_fraction > 1.0:
            raise ValueError(
                f"train_fraction ({self.train_fraction}) + val_fraction ({self.val_fraction}) "
                f"cannot exceed 1.0"
            )
        if self.source not in ("tfds", "kaggle", "huggingface"):
            raise ValueError(f"Invalid source: {self.source}. Must be 'tfds', 'kaggle', or 'huggingface'")

    @classmethod
    def gsm8k(
        cls,
        source: Literal["tfds", "kaggle", "huggingface"] = "huggingface",
        **kwargs,
    ) -> "TunRexConfig":
        """Create a configuration for GSM8K dataset."""
        return cls(
            source=source,
            dataset_name="gsm8k",
            hf_subset="main",
            **kwargs,
        )

    @classmethod
    def openrubrics(
        cls,
        split: str = "train",
        max_examples: int = 2000,
        **kwargs,
    ) -> "TunRexConfig":
        """Create a configuration for OpenRubrics dataset."""
        return cls(
            source="huggingface",
            dataset_name="OpenRubrics/OpenRubrics",
            hf_subset=split,
            max_examples=max_examples,
            **kwargs,
        )
