"""
Process Reward Calculator for PRIME RL

Implements trajectory-based reward calculation with:
- Step-wise credit assignment
- Reward aggregation strategies
- Combination with outcome-based rewards
- Advantage estimation for variance reduction
"""

from typing import List, Optional, Dict, Callable
import numpy as np

from .config import (
    PRIMEConfig,
    StepReward,
    TrajectoryReward,
    RewardAggregation
)
from .step_parser import StepParser, ParsedStep
from .step_evaluator import StepEvaluator


class ProcessRewardCalculator:
    """
    Calculator for process-based rewards in PRIME RL.

    Aggregates step-level rewards into trajectory rewards using
    various strategies (discounted sum, mean, weighted, etc.).
    """

    def __init__(self, config: PRIMEConfig):
        self.config = config
        self.step_parser = StepParser(config)
        self.step_evaluator = StepEvaluator(config)

    def calculate_trajectory_reward(
        self,
        prompt: str,
        completion: str,
        final_answer_reward: float,
        question: Optional[str] = None,
        answer: Optional[str] = None,
        **kwargs
    ) -> TrajectoryReward:
        """
        Calculate complete trajectory reward with step-wise supervision.

        Args:
            prompt: Input prompt
            completion: Model completion
            final_answer_reward: Reward for final answer correctness
            question: Optional question text
            answer: Optional ground truth answer
            **kwargs: Additional context

        Returns:
            TrajectoryReward with step rewards and aggregated reward
        """
        # Parse steps from completion
        steps = self.step_parser.parse(completion)

        # Build context for step evaluation
        context = {
            "question": question or prompt,
            "answer": answer,
            **kwargs
        }

        # Evaluate each step
        step_rewards = self.step_evaluator.evaluate_steps(steps, context)

        # Aggregate step rewards
        aggregated_reward = self._aggregate_rewards(step_rewards)

        # Combine with final answer reward
        total_reward = self._combine_with_outcome_reward(
            aggregated_reward,
            final_answer_reward
        )

        # Build trajectory reward object
        trajectory = TrajectoryReward(
            prompt=prompt,
            completion=completion,
            steps=[s.text for s in steps],
            step_rewards=step_rewards,
            aggregated_reward=aggregated_reward,
            final_answer_reward=final_answer_reward,
            total_reward=total_reward,
            metadata={
                "num_steps": len(steps),
                "aggregation_method": self.config.reward_aggregation.value,
                "gamma": self.config.gamma,
                **context
            }
        )

        return trajectory

    def _aggregate_rewards(self, step_rewards: List[StepReward]) -> float:
        """
        Aggregate step rewards using configured strategy.

        Args:
            step_rewards: List of step rewards

        Returns:
            Aggregated reward value
        """
        if not step_rewards:
            return 0.0

        rewards = np.array([sr.reward for sr in step_rewards])

        strategy = self.config.reward_aggregation

        if strategy == RewardAggregation.SUM:
            return float(np.sum(rewards))

        elif strategy == RewardAggregation.DISCOUNTED_SUM:
            # Discounted sum: ∑ γ^t r_t
            gamma = self.config.gamma
            discounts = np.array([gamma ** t for t in range(len(rewards))])
            return float(np.sum(rewards * discounts))

        elif strategy == RewardAggregation.MEAN:
            return float(np.mean(rewards))

        elif strategy == RewardAggregation.WEIGHTED_MEAN:
            # Later steps weighted more heavily
            weights = np.linspace(0.5, 1.0, len(rewards))
            weights = weights / weights.sum()
            return float(np.sum(rewards * weights))

        elif strategy == RewardAggregation.MIN:
            # Strictest evaluation (weakest link)
            return float(np.min(rewards))

        elif strategy == RewardAggregation.PRODUCT:
            # Product of normalized rewards
            # Normalize to [0.5, 1.0] to avoid zeros
            normalized = (rewards + 1.0) / 2.0  # Map [-1, 1] to [0, 1]
            normalized = np.clip(normalized, 0.5, 1.0)  # Clip to [0.5, 1.0]
            product = np.prod(normalized)
            # Map back to [0, 1]
            return float(product)

        else:
            raise ValueError(f"Unknown aggregation strategy: {strategy}")

    def _combine_with_outcome_reward(
        self,
        process_reward: float,
        outcome_reward: float
    ) -> float:
        """
        Combine process reward with outcome reward.

        Args:
            process_reward: Aggregated step reward
            outcome_reward: Final answer reward

        Returns:
            Combined reward
        """
        if self.config.use_step_rewards_only:
            return process_reward

        if not self.config.combine_with_outcome_rewards:
            return outcome_reward

        # Weighted combination
        alpha = self.config.outcome_reward_weight
        beta = 1.0 - alpha

        combined = alpha * outcome_reward + beta * process_reward

        # Add bonus for correct final answer
        if outcome_reward > 0.9:  # Correct answer threshold
            combined += self.config.final_answer_weight * 0.1

        return combined

    def normalize_rewards(
        self,
        trajectories: List[TrajectoryReward]
    ) -> List[TrajectoryReward]:
        """
        Normalize rewards across trajectories for variance reduction.

        Args:
            trajectories: List of trajectory rewards

        Returns:
            Trajectories with normalized rewards
        """
        if not self.config.normalize_rewards or not trajectories:
            return trajectories

        # Extract total rewards
        rewards = np.array([t.total_reward for t in trajectories])

        # Normalize to zero mean, unit variance
        mean = np.mean(rewards)
        std = np.std(rewards)

        if std < 1e-6:  # Avoid division by zero
            return trajectories

        # Create normalized trajectories
        normalized = []
        for i, traj in enumerate(trajectories):
            normalized_reward = (rewards[i] - mean) / std

            # Create new trajectory with normalized reward
            new_traj = TrajectoryReward(
                prompt=traj.prompt,
                completion=traj.completion,
                steps=traj.steps,
                step_rewards=traj.step_rewards,
                aggregated_reward=traj.aggregated_reward,
                final_answer_reward=traj.final_answer_reward,
                total_reward=normalized_reward,
                metadata={
                    **traj.metadata,
                    "original_reward": rewards[i],
                    "normalized": True,
                    "mean": mean,
                    "std": std
                }
            )
            normalized.append(new_traj)

        return normalized


def create_prime_reward_function(
    config: PRIMEConfig,
    outcome_reward_fn: Optional[Callable] = None
) -> Callable:
    """
    Create a GRPO-compatible reward function for PRIME RL.

    Args:
        config: PRIME RL configuration
        outcome_reward_fn: Optional outcome-based reward function
                          (e.g., accuracy checker)

    Returns:
        Reward function with signature:
        (prompts, completions, answer, question, **kwargs) -> List[float]
    """
    calculator = ProcessRewardCalculator(config)

    def prime_reward_fn(
        prompts: List[str],
        completions: List[str],
        answer: Optional[List[str]] = None,
        question: Optional[List[str]] = None,
        **kwargs
    ) -> List[float]:
        """
        PRIME RL reward function.

        Args:
            prompts: List of prompts
            completions: List of completions
            answer: Optional list of ground truth answers
            question: Optional list of questions
            **kwargs: Additional context

        Returns:
            List of reward values
        """
        rewards = []

        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            # Get outcome reward if function provided
            if outcome_reward_fn:
                outcome_rewards = outcome_reward_fn(
                    prompts=[prompt],
                    completions=[completion],
                    answer=[answer[i]] if answer else None,
                    question=[question[i]] if question else None,
                    **kwargs
                )
                final_answer_reward = outcome_rewards[0]
            else:
                final_answer_reward = 0.0

            # Calculate trajectory reward
            trajectory = calculator.calculate_trajectory_reward(
                prompt=prompt,
                completion=completion,
                final_answer_reward=final_answer_reward,
                question=question[i] if question else None,
                answer=answer[i] if answer else None,
                **kwargs
            )

            rewards.append(trajectory.total_reward)

        return rewards

    return prime_reward_fn


def calculate_step_advantages(
    step_rewards: List[StepReward],
    gamma: float = 0.95,
    baseline: Optional[float] = None
) -> List[float]:
    """
    Calculate advantages for step rewards (for policy gradient).

    Implements:
    A_t = r_t + γ * V(s_{t+1}) - V(s_t)

    Simplified version using reward-to-go:
    A_t = (∑_{t'=t}^T γ^{t'-t} r_{t'}) - baseline

    Args:
        step_rewards: List of step rewards
        gamma: Discount factor
        baseline: Optional baseline value (mean reward)

    Returns:
        List of advantage values
    """
    if not step_rewards:
        return []

    rewards = [sr.reward for sr in step_rewards]
    T = len(rewards)

    # Calculate reward-to-go for each step
    reward_to_go = []
    running_sum = 0.0

    for t in range(T - 1, -1, -1):
        running_sum = rewards[t] + gamma * running_sum
        reward_to_go.insert(0, running_sum)

    # Subtract baseline
    if baseline is None:
        baseline = np.mean(reward_to_go)

    advantages = [rtg - baseline for rtg in reward_to_go]

    return advantages


def analyze_trajectory(trajectory: TrajectoryReward) -> Dict:
    """
    Analyze a trajectory for debugging and logging.

    Args:
        trajectory: Trajectory reward object

    Returns:
        Dictionary with analysis statistics
    """
    step_rewards = [sr.reward for sr in trajectory.step_rewards]

    analysis = {
        "num_steps": trajectory.num_steps,
        "mean_step_reward": trajectory.mean_step_reward,
        "std_step_reward": np.std(step_rewards) if step_rewards else 0.0,
        "min_step_reward": min(step_rewards) if step_rewards else 0.0,
        "max_step_reward": max(step_rewards) if step_rewards else 0.0,
        "aggregated_reward": trajectory.aggregated_reward,
        "final_answer_reward": trajectory.final_answer_reward,
        "total_reward": trajectory.total_reward,
        "step_rewards": step_rewards,
    }

    # Calculate step quality distribution
    if step_rewards:
        analysis["step_quality"] = {
            "excellent": sum(1 for r in step_rewards if r > 0.8),
            "good": sum(1 for r in step_rewards if 0.6 < r <= 0.8),
            "fair": sum(1 for r in step_rewards if 0.4 < r <= 0.6),
            "poor": sum(1 for r in step_rewards if r <= 0.4),
        }

    return analysis
