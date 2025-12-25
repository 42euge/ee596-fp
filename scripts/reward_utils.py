"""Reward function setup utilities for GRPO training."""


def get_base_reward_fns():
    """Get base reward functions from TunRex."""
    from tunrex.datasets import (
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    )

    reward_fns = [
        match_format_exactly,
        match_format_approximately,
        check_answer,
        check_numbers,
    ]
    reward_names = [
        "format_exact",
        "format_approx",
        "answer_check",
        "number_check",
    ]

    return reward_fns, reward_names


def load_rubricset(rubric_file: str):
    """Load rubricset from YAML file."""
    from src.rubrics import load_rubricset_from_yaml

    print(f"  Loading rubric from: {rubric_file}")
    rubricset = load_rubricset_from_yaml(rubric_file)
    print(f"  Loaded rubric: {rubricset.name} with {len(rubricset.rubrics)} rubric(s)")
    return rubricset


def setup_rewards(args, wandb_enabled: bool):
    """Set up reward functions with optional W&B logging and rubrics.

    Args:
        args: Training arguments with rubric_file, rubric_weight
        wandb_enabled: Whether W&B logging is enabled

    Returns:
        Tuple of (reward_fns, reward_logger or None)
    """
    base_reward_fns, reward_names = get_base_reward_fns()

    # Load rubric if specified
    rubricset = None
    if args.rubric_file:
        rubricset = load_rubricset(args.rubric_file)

    # Wrap with W&B logging if enabled
    reward_logger = None
    if wandb_enabled:
        try:
            from src.wandb_rewards import create_logged_reward_fns

            reward_fns, reward_logger = create_logged_reward_fns(
                base_reward_fns,
                reward_names,
                rubricset=rubricset,
                rubric_weight=args.rubric_weight if args.rubric_file else 1.0,
                log_every_n_steps=10,
            )
            print(f"  W&B reward logging enabled for {len(reward_fns)} reward functions")
            return reward_fns, reward_logger
        except ImportError as e:
            print(f"  WARNING: Could not enable W&B reward logging: {e}")

    # Fallback: use base rewards + rubric without logging
    reward_fns = list(base_reward_fns)
    if rubricset:
        from src.rubrics import create_grpo_reward_function
        reward_fns.append(create_grpo_reward_function(rubricset, weight=args.rubric_weight))

    return reward_fns, None
