"""
Configuration for Gemma3-1B GRPO Fine-tuning

This file contains all hyperparameters and configuration settings for:
- Model architecture (LoRA rank, alpha)
- GRPO training parameters
- Data processing settings
- Inference settings
"""

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class LoRAConfig:
    """LoRA (Low-Rank Adaptation) configuration."""
    rank: int = 64
    alpha: float = 64.0


@dataclass
class GRPOConfig:
    """GRPO (Group Relative Policy Optimization) configuration."""
    max_prompt_length: int = 256
    total_generation_steps: int = 512
    temperature: float = 0.9
    top_p: float = 1.0
    top_k: int = 50
    num_generations: int = 2  # Reduced for lower RAM usage
    num_iterations: int = 1
    beta: float = 0.08  # KL divergence penalty coefficient
    epsilon: float = 0.2  # Clipping parameter


@dataclass
class TrainingConfig:
    """Training configuration."""
    train_micro_batch_size: int = 2
    num_batches: int = 1000
    num_test_batches: int = 100
    eval_every_n_steps: int = 10
    num_epochs: int = 1
    learning_rate: float = 3e-6
    beta1: float = 0.9
    beta2: float = 0.99
    weight_decay: float = 0.1
    warmup_ratio: float = 0.1
    max_grad_norm: float = 0.1
    save_interval_steps: int = 500
    max_checkpoints_to_keep: int = 4


@dataclass
class DataConfig:
    """Data configuration."""
    train_data_dir: str = "./data/train"
    test_data_dir: str = "./data/test"
    train_fraction: float = 0.942
    eval_fraction: float = 0.22
    use_gsm8k: bool = False
    use_openrubrics: bool = True
    openrubrics_max_examples: int = 2000
    openrubrics_split: str = "train"


@dataclass
class InferenceConfig:
    """Inference configuration."""
    # Generation configs for different modes
    greedy: Dict = field(default_factory=lambda: {
        "temperature": 1e-4,
        "top_k": 1,
        "top_p": 1.0
    })
    standard: Dict = field(default_factory=lambda: {
        "temperature": 0.7,
        "top_k": 50,
        "top_p": 0.95
    })
    liberal: Dict = field(default_factory=lambda: {
        "temperature": 0.85,
        "top_k": 2000,
        "top_p": 1.0
    })


@dataclass
class MonitoringConfig:
    """Configuration for reward hack detection and behavior monitoring."""
    enabled: bool = True  # Enable/disable monitoring

    # Statistical anomaly detection
    reward_zscore_threshold: float = 3.0
    reward_variance_threshold: float = 5.0
    min_samples_for_detection: int = 10

    # Response length monitoring
    min_response_length: int = 10
    max_response_length: int = 2048
    length_outlier_threshold: float = 3.0

    # Repetition detection
    max_ngram_repetition_ratio: float = 0.3
    ngram_size: int = 3
    max_token_repetition_ratio: float = 0.5

    # Format gaming detection
    min_reasoning_length: int = 20
    format_quality_ratio_threshold: float = 0.3

    # Diversity monitoring
    min_unique_responses_ratio: float = 0.5
    similarity_threshold: float = 0.9

    # KL divergence monitoring
    kl_divergence_min: float = 0.001
    kl_divergence_max: float = 5.0

    # Gradient monitoring
    gradient_norm_max: float = 10.0
    gradient_norm_min: float = 1e-6

    # Reward component balance
    max_component_imbalance: float = 0.9

    # Moving average window sizes
    short_window_size: int = 20
    long_window_size: int = 100

    # Logging configuration
    log_detections_to_wandb: bool = True
    log_detections_to_console: bool = True
    detection_log_interval: int = 1  # Log every N steps


@dataclass
class Config:
    """Master configuration combining all sub-configs."""
    lora: LoRAConfig = field(default_factory=LoRAConfig)
    grpo: GRPOConfig = field(default_factory=GRPOConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    inference: InferenceConfig = field(default_factory=InferenceConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)

    # Paths
    checkpoint_dir: str = "./checkpoints"
    intermediate_checkpoint_dir: str = "./checkpoints/intermediate"
    results_dir: str = "./results"

    # Prompt configuration
    system_prompt_version: int = 2  # Use explicit reasoning prompt
    use_rubric_evaluation: bool = True

    # Device settings
    device: str = "auto"  # "auto", "cuda", "mps", "cpu"


# Prompt templates
REASONING_START = "<reasoning>"
REASONING_END = "</reasoning>"
SOLUTION_START = "<answer>"
SOLUTION_END = "</answer>"

# System prompts (version 0-6)
SYSTEM_PROMPTS = {
    0: f"""You are given a problem. Think about the problem and provide your reasoning. Place it between {REASONING_START} and {REASONING_END}. Then, provide the final answer (i.e., just one numerical value) between {SOLUTION_START} and {SOLUTION_END}.""",

    1: f"""You are given a problem. Think about the problem and provide your reasoning. Place it between {REASONING_START} and {REASONING_END}. Then, provide the final answer between {SOLUTION_START} and {SOLUTION_END}.""",

    2: f"""You are given a problem. Think carefully and show your detailed reasoning step-by-step. Place your reasoning between {REASONING_START} and {REASONING_END}. After completing your reasoning, provide the final answer between {SOLUTION_START} and {SOLUTION_END}.""",

    3: f"""You are given a problem. Let's think step by step. Provide your reasoning process between {REASONING_START} and {REASONING_END}. Then provide the final answer between {SOLUTION_START} and {SOLUTION_END}.""",

    4: f"""You are given a problem. First, understand what is being asked. Then, work through your reasoning carefully. Place your reasoning between {REASONING_START} and {REASONING_END}. Finally, provide your answer between {SOLUTION_START} and {SOLUTION_END}.""",

    5: f"""You are given a problem. Consider different approaches to solve it. Think through your reasoning, exploring multiple paths if helpful. Place your complete reasoning between {REASONING_START} and {REASONING_END}. Then provide the best final answer between {SOLUTION_START} and {SOLUTION_END}.""",

    6: f"""Solve the problem below. Show your work in {REASONING_START}{REASONING_END}. Give your answer in {SOLUTION_START}{SOLUTION_END}.""",
}

# Chat template for Gemma3
CHAT_TEMPLATE = """<start_of_turn>user
{system_prompt}

{question}<end_of_turn>
<start_of_turn>model"""


def get_default_config() -> Config:
    """Get default configuration."""
    return Config()


def get_system_prompt(version: int = 2) -> str:
    """Get system prompt by version number."""
    return SYSTEM_PROMPTS.get(version, SYSTEM_PROMPTS[2])


def format_prompt(question: str, system_prompt: Optional[str] = None, rubric: Optional[str] = None) -> str:
    """Format a question into the chat template.

    Args:
        question: The question/problem to solve
        system_prompt: Optional system prompt (uses default if None)
        rubric: Optional rubric to include in the prompt

    Returns:
        Formatted prompt string
    """
    if system_prompt is None:
        system_prompt = get_system_prompt(2)

    rubric_block = ""
    if rubric:
        rubric_block = f"\nRubric:\n{rubric}\n\n"

    return CHAT_TEMPLATE.format(
        system_prompt=system_prompt,
        question=f"{rubric_block}{question}",
    )
