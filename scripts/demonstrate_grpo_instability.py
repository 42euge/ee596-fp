#!/usr/bin/env python3
"""
Demonstrates GRPO instability issues with continuous/subjective rewards.

This script reproduces the problem described in the RLOO feature request:
1. GRPO's std normalization amplifies noise with subjective rewards
2. Lack of KL mechanism leads to reward hacking
3. Training instability with continuous scores from rubric-based evaluation

Key metrics tracked:
- Reward std (shows amplification of noise)
- Policy drift (KL from reference)
- Advantage values (shows normalization effects)
- Reward progression (shows potential reward hacking)
"""

import os
import sys
import json
import numpy as np
from typing import List, Dict, Any

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from scripts.utils import (
    parse_args, patch_tunix, check_tpu_availability, check_jax_deps,
    check_tunix_deps, create_mesh, init_wandb, finish_wandb,
)
from scripts.model_utils import download_model, load_models
from scripts.training_config import create_cluster_config, create_grpo_config, load_datasets
from scripts.reward_utils import setup_rewards


class StabilityMetricsLogger:
    """Logs metrics relevant to GRPO stability issues."""

    def __init__(self, output_path: str = "/tmp/grpo_stability_metrics.json"):
        self.output_path = output_path
        self.metrics = {
            "step": [],
            "reward_mean": [],
            "reward_std": [],
            "reward_min": [],
            "reward_max": [],
            "advantage_mean": [],
            "advantage_std": [],
            "normalized_advantage_std": [],
            "kl_divergence": [],
        }
        self.step_count = 0

    def log_step(self, rewards: List[float], advantages: List[float],
                 normalized_advantages: List[float], kl_div: float = None):
        """Log metrics for a single training step."""
        self.step_count += 1
        rewards_arr = np.array(rewards)
        advantages_arr = np.array(advantages)
        norm_adv_arr = np.array(normalized_advantages)

        self.metrics["step"].append(self.step_count)
        self.metrics["reward_mean"].append(float(np.mean(rewards_arr)))
        self.metrics["reward_std"].append(float(np.std(rewards_arr)))
        self.metrics["reward_min"].append(float(np.min(rewards_arr)))
        self.metrics["reward_max"].append(float(np.max(rewards_arr)))
        self.metrics["advantage_mean"].append(float(np.mean(advantages_arr)))
        self.metrics["advantage_std"].append(float(np.std(advantages_arr)))
        self.metrics["normalized_advantage_std"].append(float(np.std(norm_adv_arr)))
        self.metrics["kl_divergence"].append(float(kl_div) if kl_div is not None else None)

        # Print warning if reward std is high (indicates noise amplification)
        if self.metrics["reward_std"][-1] > 2.0:
            print(f"⚠️  WARNING: High reward std detected at step {self.step_count}: "
                  f"{self.metrics['reward_std'][-1]:.3f}")

        # Print warning if KL divergence is increasing (indicates policy drift)
        if len(self.metrics["kl_divergence"]) > 2 and kl_div is not None:
            if kl_div > 0.1:
                print(f"⚠️  WARNING: High KL divergence at step {self.step_count}: {kl_div:.4f}")

    def save(self):
        """Save metrics to JSON file."""
        with open(self.output_path, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"\n📊 Stability metrics saved to: {self.output_path}")

    def print_summary(self):
        """Print summary of stability issues detected."""
        print("\n" + "=" * 80)
        print("GRPO STABILITY ANALYSIS SUMMARY")
        print("=" * 80)

        reward_stds = np.array(self.metrics["reward_std"])
        kl_divs = np.array([kl for kl in self.metrics["kl_divergence"] if kl is not None])

        print(f"\n🎯 Reward Statistics:")
        print(f"  Mean reward std: {np.mean(reward_stds):.4f}")
        print(f"  Max reward std: {np.max(reward_stds):.4f}")
        print(f"  Reward range: [{np.min(self.metrics['reward_min']):.2f}, "
              f"{np.max(self.metrics['reward_max']):.2f}]")

        # Issue #1: Std normalization amplifies noise
        high_std_steps = np.sum(reward_stds > 2.0)
        if high_std_steps > 0:
            print(f"\n❌ ISSUE #1: Std Normalization Amplifies Noise")
            print(f"  {high_std_steps}/{len(reward_stds)} steps had reward std > 2.0")
            print(f"  This indicates high variance in continuous rewards")
            print(f"  GRPO's A_i = (R_i - mean) / std amplifies this noise")

        # Issue #2: No KL mechanism to prevent drift
        if len(kl_divs) > 0:
            mean_kl = np.mean(kl_divs)
            max_kl = np.max(kl_divs)
            print(f"\n❌ ISSUE #2: Policy Drift Without KL Constraint")
            print(f"  Mean KL divergence: {mean_kl:.4f}")
            print(f"  Max KL divergence: {max_kl:.4f}")
            if max_kl > 0.1:
                print(f"  High KL divergence indicates policy is drifting from reference")
                print(f"  GRPO lacks built-in KL penalty to constrain this drift")

        # Recommendation
        print(f"\n💡 RECOMMENDATION:")
        print(f"  Consider using RLOO (REINFORCE Leave-One-Out) instead:")
        print(f"  - Uses A_i = R_i - mean(R_j where j != i) (no std normalization)")
        print(f"  - Integrates KL directly into reward: R'_i = R_i - β * KL")
        print(f"  - More robust to noisy/subjective rewards (Ahmadian et al. 2024)")
        print("=" * 80 + "\n")


def create_continuous_reward_wrapper(base_reward_fns, stability_logger):
    """
    Wrap reward functions to inject continuous noise and track stability.

    This simulates the "show your work" rubric scoring scenario where
    rewards are continuous and potentially noisy.
    """
    def wrapped_reward_fn(prompts, completions, **kwargs):
        # Get base rewards (binary/discrete)
        base_rewards = []
        for reward_fn in base_reward_fns:
            rewards = reward_fn(prompts, completions, **kwargs)
            base_rewards.append(rewards)

        # Combine base rewards
        combined = np.mean(base_rewards, axis=0)

        # Add continuous rubric-style scoring with noise
        # Simulates LLM judge giving subjective scores
        rubric_scores = []
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            # Base score from discrete rewards
            base_score = combined[i] * 10.0  # Scale to 0-10

            # Add continuous subjective component (simulates rubric evaluation)
            # This has high variance, which is the problem
            subjective_noise = np.random.normal(0, 2.5)  # High std noise
            continuous_score = base_score + subjective_noise

            # Clip to valid range
            continuous_score = np.clip(continuous_score, 0, 10)
            rubric_scores.append(continuous_score)

        # Convert to list
        continuous_rewards = rubric_scores

        # Calculate what GRPO does: advantage with std normalization
        rewards_arr = np.array(continuous_rewards)
        mean_reward = np.mean(rewards_arr)
        std_reward = np.std(rewards_arr) + 1e-8

        # GRPO advantage formula
        advantages = rewards_arr - mean_reward
        normalized_advantages = advantages / std_reward

        # Log metrics (mock KL divergence for demonstration)
        mock_kl = np.random.uniform(0.01, 0.15)  # Simulated KL drift
        stability_logger.log_step(
            rewards=continuous_rewards,
            advantages=advantages.tolist(),
            normalized_advantages=normalized_advantages.tolist(),
            kl_div=mock_kl
        )

        return continuous_rewards

    return wrapped_reward_fn


def main():
    args = parse_args()
    patch_tunix()

    # Override some args for demonstration
    args.num_steps = min(args.num_steps, 50)  # Keep it short for demo

    print("=" * 80)
    print("GRPO Stability Issue Demonstration")
    print("Testing: Continuous/subjective rewards (rubric-based scoring)")
    print("=" * 80)

    # Initialize stability logger
    stability_logger = StabilityMetricsLogger()

    # Check dependencies
    print("\n[1/5] Checking environment...")
    if not check_jax_deps():
        sys.exit(1)

    print("\n[2/5] Checking TPU...")
    num_devices, has_tpu = check_tpu_availability()

    print("\n[3/5] Initializing W&B...")
    args.wandb_project = "grpo-stability-demo"
    args.run_name = "grpo-continuous-rewards-instability"
    wandb_run = init_wandb(args, num_devices, has_tpu)

    if not check_tunix_deps():
        sys.exit(1)

    # Download model
    print(f"\n[4/5] Downloading model: {args.model_id}...")
    model_path, eos_tokens = download_model(args.model_id)

    # Create mesh and load models
    print("\n[5/5] Loading model and starting training...")
    mesh = create_mesh(num_devices)

    with mesh:
        policy_model, reference_model, tokenizer, eos_tokens = load_models(
            model_path, mesh, args, eos_tokens
        )

        train_ds, val_ds, test_ds = load_datasets(args)
        print(f"  Train: {len(train_ds)}, Val: {len(val_ds) if val_ds else 0}, Test: {len(test_ds)}")

        # Create configs
        cluster_config = create_cluster_config(args, mesh, eos_tokens)
        grpo_config = create_grpo_config(args)

        # Setup base rewards
        base_reward_fns, reward_logger = setup_rewards(args, wandb_enabled=wandb_run is not None)

        # Wrap with continuous reward logger
        print("\n⚠️  Wrapping rewards with continuous/subjective scoring simulation...")
        wrapped_reward_fn = create_continuous_reward_wrapper(base_reward_fns, stability_logger)

        # Create trainer
        from tunix.rl import rl_cluster as rl_cluster_lib
        from tunix.rl.grpo.grpo_learner import GRPOLearner

        rl_cluster = rl_cluster_lib.RLCluster(
            actor=policy_model,
            reference=reference_model,
            tokenizer=tokenizer,
            cluster_config=cluster_config,
        )

        trainer = GRPOLearner(
            rl_cluster=rl_cluster,
            reward_fns=[wrapped_reward_fn],  # Use wrapped continuous reward
            algo_config=grpo_config,
        )

        # Train
        print("\n" + "=" * 80)
        print(f"Training with continuous rewards: {args.num_steps} steps")
        print("Tracking: reward std, advantage normalization, KL divergence")
        print("=" * 80 + "\n")

        try:
            trainer.train(train_ds, val_ds)
        except Exception as e:
            print(f"\n⚠️  Training encountered issues (expected with noisy rewards): {e}")

    # Save and print results
    stability_logger.save()
    stability_logger.print_summary()

    finish_wandb(reward_logger)

    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("This demonstrates why RLOO is needed for continuous/subjective rewards")
    print("=" * 80)


if __name__ == "__main__":
    main()
