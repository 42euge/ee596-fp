"""
Configuration for Reward Robustness Evaluation

Dataclass-based configuration following patterns from src/config.py.
"""

from dataclasses import dataclass, field
from typing import List, Optional


@dataclass
class PerturbationConfig:
    """Configuration for semantic-preserving perturbations."""

    # Which perturbation types to apply
    enabled_types: List[str] = field(
        default_factory=lambda: ["synonym", "paraphrase", "reorder"]
    )

    # Number of perturbed variants to generate per sample
    num_variants: int = 5

    # Synonym replacement settings
    synonym_probability: float = 0.3  # Probability of replacing each eligible word
    synonym_max_replacements: int = 5  # Max words to replace per text

    # Paraphrase model settings
    paraphrase_model: str = "humarin/chatgpt_paraphraser_on_T5_base"
    paraphrase_max_length: int = 256
    paraphrase_num_beams: int = 5
    paraphrase_temperature: float = 0.7

    # Sentence reordering settings
    reorder_preserve_first: bool = True  # Keep first sentence in place
    reorder_preserve_last: bool = True   # Keep last sentence in place

    # General settings
    seed: int = 42
    preserve_tags: bool = True  # Preserve <reasoning> and <answer> XML tags


@dataclass
class ExternalRewardConfig:
    """Configuration for external HuggingFace reward models."""

    # List of HuggingFace model IDs to evaluate
    model_ids: List[str] = field(default_factory=list)

    # Device settings
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"

    # Quantization
    load_in_8bit: bool = True
    load_in_4bit: bool = False

    # Inference settings
    batch_size: int = 8
    max_length: int = 2048

    # Cache directory for models
    cache_dir: Optional[str] = None


@dataclass
class RobustnessConfig:
    """Master configuration for reward robustness evaluation."""

    # Internal reward functions to test (from src/utils.py)
    internal_rewards: List[str] = field(
        default_factory=lambda: ["format_reward", "accuracy_reward"]
    )

    # External reward model configuration
    external_rewards: ExternalRewardConfig = field(
        default_factory=ExternalRewardConfig
    )

    # Perturbation configuration
    perturbations: PerturbationConfig = field(
        default_factory=PerturbationConfig
    )

    # Evaluation settings
    num_samples: int = 100  # Number of samples to evaluate

    # Output settings
    output_dir: str = "./robustness_results"
    save_detailed: bool = True  # Save per-sample details

    # Consistency thresholds for reporting
    variance_threshold: float = 0.5  # Flag high variance samples
    flip_threshold: float = 0.0  # Threshold for sign flips


def get_default_config() -> RobustnessConfig:
    """Get default robustness evaluation configuration."""
    return RobustnessConfig()


def get_quick_config() -> RobustnessConfig:
    """Get a quick evaluation config for testing."""
    return RobustnessConfig(
        num_samples=20,
        perturbations=PerturbationConfig(
            enabled_types=["synonym"],
            num_variants=3,
        ),
    )


def get_thorough_config() -> RobustnessConfig:
    """Get a thorough evaluation config."""
    return RobustnessConfig(
        num_samples=500,
        perturbations=PerturbationConfig(
            enabled_types=["synonym", "paraphrase", "reorder"],
            num_variants=10,
        ),
        save_detailed=True,
    )
