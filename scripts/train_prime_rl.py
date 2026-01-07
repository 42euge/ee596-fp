#!/usr/bin/env python3
"""
Train with PRIME RL (Process-based Reinforcement with Intermediate Model Evaluation)

This script demonstrates how to train a reasoning model using PRIME RL,
which assigns rewards to intermediate reasoning steps rather than just
the final answer.

Usage:
    python scripts/train_prime_rl.py --config configs/prime_rl_config.yaml

Example:
    # Train with default PRIME RL configuration
    python scripts/train_prime_rl.py

    # Train with custom configuration
    python scripts/train_prime_rl.py \\
        --gamma 0.95 \\
        --step_evaluation_method hybrid \\
        --reward_aggregation discounted_sum

    # Train with LLM judge for step evaluation
    python scripts/train_prime_rl.py \\
        --llm_judge_enabled \\
        --llm_judge_model gpt-4o-mini
"""

import os
import sys
from pathlib import Path
import argparse
from typing import List, Optional

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

# PRIME RL imports
from src.prime_rl import (
    PRIMEConfig,
    StepEvaluationMethod,
    RewardAggregation,
    StepParsingStrategy,
    create_prime_rl_reward_suite,
    get_default_prime_rl_config,
)

# Existing imports
from src.config import Config, GRPOConfig, LoRAConfig, TrainingConfig, DataConfig
from src.model import GemmaModel
from TunRex.src.tunrex.datasets import TunRex, TunRexConfig
from TunRex.src.tunrex.datasets.rewards import (
    match_format_exactly,
    match_format_approximately,
    check_answer,
    check_numbers,
)

# Training utilities
from scripts.training_config import (
    create_cluster_config,
    create_grpo_config,
    create_optimizer,
)


def create_prime_rl_config_from_args(args) -> PRIMEConfig:
    """
    Create PRIME RL configuration from command-line arguments.

    Args:
        args: Parsed arguments

    Returns:
        PRIMEConfig
    """
    # Start with defaults
    config = get_default_prime_rl_config()

    # Override with command-line arguments
    if args.gamma is not None:
        config.gamma = args.gamma

    if args.step_evaluation_method is not None:
        config.step_evaluation_method = StepEvaluationMethod(args.step_evaluation_method)

    if args.reward_aggregation is not None:
        config.reward_aggregation = RewardAggregation(args.reward_aggregation)

    if args.step_parsing_strategy is not None:
        config.step_parsing_strategy = StepParsingStrategy(args.step_parsing_strategy)

    if args.llm_judge_enabled is not None:
        config.llm_judge_enabled = args.llm_judge_enabled

    if args.llm_judge_model is not None:
        config.llm_judge_model = args.llm_judge_model

    if args.final_answer_weight is not None:
        config.final_answer_weight = args.final_answer_weight

    if args.outcome_reward_weight is not None:
        config.outcome_reward_weight = args.outcome_reward_weight

    if args.penalize_incorrect_steps is not None:
        config.penalize_incorrect_steps = args.penalize_incorrect_steps

    if args.enable_symbolic_solver is not None:
        config.enable_symbolic_solver = args.enable_symbolic_solver

    return config


def setup_reward_functions(
    prime_config: PRIMEConfig,
    include_baseline_rewards: bool = True,
    include_prime_rewards: bool = True
) -> List:
    """
    Setup reward functions for GRPO training.

    Args:
        prime_config: PRIME RL configuration
        include_baseline_rewards: Include baseline format/accuracy rewards
        include_prime_rewards: Include PRIME RL rewards

    Returns:
        List of reward functions
    """
    reward_fns = []

    # Baseline rewards (from existing TunRex)
    if include_baseline_rewards:
        reward_fns.extend([
            match_format_exactly,
            match_format_approximately,
            check_answer,
            check_numbers,
        ])

    # PRIME RL rewards
    if include_prime_rewards:
        prime_rewards = create_prime_rl_reward_suite(
            config=prime_config,
            include_format=True,
            include_accuracy=True,
            include_pure_prime=True,
        )
        reward_fns.extend(prime_rewards)

    return reward_fns


def train_prime_rl(
    model_name: str = "google/gemma-3-1b-it",
    dataset_name: str = "gsm8k",
    prime_config: Optional[PRIMEConfig] = None,
    lora_rank: int = 64,
    learning_rate: float = 3e-6,
    num_epochs: int = 3,
    batch_size: int = 2,
    output_dir: str = "./checkpoints/prime_rl",
    wandb_project: Optional[str] = "prime-rl-training",
    include_baseline_rewards: bool = True,
    **kwargs
):
    """
    Main training function for PRIME RL.

    Args:
        model_name: HuggingFace model name
        dataset_name: Dataset name (gsm8k, etc.)
        prime_config: PRIME RL configuration
        lora_rank: LoRA rank
        learning_rate: Learning rate
        num_epochs: Number of training epochs
        batch_size: Batch size
        output_dir: Output directory for checkpoints
        wandb_project: Weights & Biases project name
        include_baseline_rewards: Include baseline rewards alongside PRIME RL
        **kwargs: Additional training arguments
    """
    print("=" * 80)
    print("PRIME RL Training")
    print("=" * 80)

    # Setup PRIME RL configuration
    if prime_config is None:
        prime_config = get_default_prime_rl_config()

    print("\nPRIME RL Configuration:")
    print(f"  Step Parsing: {prime_config.step_parsing_strategy.value}")
    print(f"  Step Evaluation: {prime_config.step_evaluation_method.value}")
    print(f"  Reward Aggregation: {prime_config.reward_aggregation.value}")
    print(f"  Gamma (discount factor): {prime_config.gamma}")
    print(f"  LLM Judge Enabled: {prime_config.llm_judge_enabled}")
    print(f"  Symbolic Solver Enabled: {prime_config.enable_symbolic_solver}")
    print(f"  Outcome Reward Weight: {prime_config.outcome_reward_weight}")
    print()

    # Create training configuration
    config = Config(
        lora=LoRAConfig(rank=lora_rank, alpha=float(lora_rank)),
        grpo=GRPOConfig(num_generations=2, beta=0.08, epsilon=0.2),
        training=TrainingConfig(
            batch_size=batch_size,
            learning_rate=learning_rate,
            num_epochs=num_epochs,
            gradient_clip_norm=0.1,
        ),
        data=DataConfig(
            dataset_name=dataset_name,
            train_split="train",
            val_split="test",
        ),
    )

    print("Loading model and dataset...")

    # Load dataset
    tunrex_config = TunRexConfig(
        dataset_name=dataset_name,
        batch_size=batch_size,
        train_split="train[:1000]",  # Subset for faster experimentation
        val_split="test[:100]",
    )
    dataset = TunRex(tunrex_config)
    train_ds, val_ds, test_ds = dataset.load()

    print(f"Dataset loaded: {len(train_ds)} training examples")

    # Setup reward functions
    print("\nSetting up reward functions...")
    reward_fns = setup_reward_functions(
        prime_config=prime_config,
        include_baseline_rewards=include_baseline_rewards,
        include_prime_rewards=True,
    )

    print(f"  Total reward functions: {len(reward_fns)}")
    print(f"  Baseline rewards: {4 if include_baseline_rewards else 0}")
    print(f"  PRIME RL rewards: {3}")

    # Initialize Weights & Biases (optional)
    if wandb_project:
        try:
            import wandb
            wandb.init(
                project=wandb_project,
                config={
                    "model": model_name,
                    "dataset": dataset_name,
                    "prime_rl": prime_config.__dict__,
                    "training": config.training.__dict__,
                    "lora": config.lora.__dict__,
                    "grpo": config.grpo.__dict__,
                }
            )
            print(f"\nW&B initialized: {wandb_project}")
        except Exception as e:
            print(f"\nW&B initialization failed: {e}")

    print("\n" + "=" * 80)
    print("Starting PRIME RL Training")
    print("=" * 80)

    # Note: The actual training loop would be implemented here
    # using the existing GRPO training infrastructure from train_grpo.py
    # This is a demonstration of how to set up PRIME RL

    print("\n⚠️  Full training integration requires:")
    print("1. Setting up GRPO learner with PRIME RL reward functions")
    print("2. Configuring trajectory collection for step rewards")
    print("3. Setting up logging for step-wise metrics")
    print("4. Integrating with existing training loop in train_grpo.py")

    print("\nFor full training, use the existing train_grpo.py script and")
    print("replace reward functions with PRIME RL suite:")
    print("  reward_fns = create_prime_rl_reward_suite(config)")

    print("\n✓ Configuration and reward functions successfully set up!")

    return config, prime_config, reward_fns


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Train with PRIME RL (Process-based Reinforcement Learning)"
    )

    # Model and data
    parser.add_argument(
        "--model_name",
        type=str,
        default="google/gemma-3-1b-it",
        help="HuggingFace model name"
    )
    parser.add_argument(
        "--dataset_name",
        type=str,
        default="gsm8k",
        help="Dataset name"
    )

    # PRIME RL configuration
    parser.add_argument(
        "--gamma",
        type=float,
        default=0.95,
        help="Discount factor for credit assignment"
    )
    parser.add_argument(
        "--step_evaluation_method",
        type=str,
        choices=["rule_based", "symbolic", "llm_judge", "hybrid"],
        default="hybrid",
        help="Method for evaluating steps"
    )
    parser.add_argument(
        "--reward_aggregation",
        type=str,
        choices=["sum", "discounted_sum", "mean", "weighted_mean", "min", "product"],
        default="discounted_sum",
        help="Strategy for aggregating step rewards"
    )
    parser.add_argument(
        "--step_parsing_strategy",
        type=str,
        choices=["numbered", "line_based", "sentence_based", "semantic"],
        default="numbered",
        help="Strategy for parsing reasoning steps"
    )
    parser.add_argument(
        "--llm_judge_enabled",
        action="store_true",
        help="Enable LLM judge for step evaluation"
    )
    parser.add_argument(
        "--llm_judge_model",
        type=str,
        default="gpt-4o-mini",
        help="LLM model for step evaluation"
    )
    parser.add_argument(
        "--final_answer_weight",
        type=float,
        default=2.0,
        help="Weight for final answer correctness"
    )
    parser.add_argument(
        "--outcome_reward_weight",
        type=float,
        default=0.3,
        help="Weight for outcome-based rewards (vs process rewards)"
    )
    parser.add_argument(
        "--penalize_incorrect_steps",
        action="store_true",
        help="Penalize incorrect intermediate steps"
    )
    parser.add_argument(
        "--enable_symbolic_solver",
        action="store_true",
        help="Enable symbolic solver for math verification"
    )

    # Training configuration
    parser.add_argument(
        "--lora_rank",
        type=int,
        default=64,
        help="LoRA rank"
    )
    parser.add_argument(
        "--learning_rate",
        type=float,
        default=3e-6,
        help="Learning rate"
    )
    parser.add_argument(
        "--num_epochs",
        type=int,
        default=3,
        help="Number of training epochs"
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=2,
        help="Batch size"
    )

    # Output and logging
    parser.add_argument(
        "--output_dir",
        type=str,
        default="./checkpoints/prime_rl",
        help="Output directory for checkpoints"
    )
    parser.add_argument(
        "--wandb_project",
        type=str,
        default="prime-rl-training",
        help="Weights & Biases project name"
    )
    parser.add_argument(
        "--no_baseline_rewards",
        action="store_true",
        help="Exclude baseline rewards (use PRIME RL only)"
    )

    return parser.parse_args()


def main():
    """Main entry point."""
    args = parse_args()

    # Create PRIME RL configuration
    prime_config = create_prime_rl_config_from_args(args)

    # Train
    config, prime_config, reward_fns = train_prime_rl(
        model_name=args.model_name,
        dataset_name=args.dataset_name,
        prime_config=prime_config,
        lora_rank=args.lora_rank,
        learning_rate=args.learning_rate,
        num_epochs=args.num_epochs,
        batch_size=args.batch_size,
        output_dir=args.output_dir,
        wandb_project=args.wandb_project,
        include_baseline_rewards=not args.no_baseline_rewards,
    )

    print("\n✓ PRIME RL setup complete!")


if __name__ == "__main__":
    main()
