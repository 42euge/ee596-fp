"""
Utility functions for PRIME RL
"""

from typing import List, Dict, Any, Optional
import json
import numpy as np

from .config import TrajectoryReward, StepReward


def format_trajectory_summary(trajectory: TrajectoryReward) -> str:
    """
    Format trajectory for human-readable display.

    Args:
        trajectory: TrajectoryReward object

    Returns:
        Formatted string summary
    """
    lines = [
        "=" * 80,
        "TRAJECTORY SUMMARY",
        "=" * 80,
        f"Total Reward: {trajectory.total_reward:.3f}",
        f"Steps: {trajectory.num_steps}",
        f"Mean Step Reward: {trajectory.mean_step_reward:.3f}",
        f"Final Answer Reward: {trajectory.final_answer_reward:.3f}",
        f"Aggregated Process Reward: {trajectory.aggregated_reward:.3f}",
        "",
        "STEP-BY-STEP BREAKDOWN:",
        "-" * 80,
    ]

    for i, (step_text, step_reward) in enumerate(zip(trajectory.steps, trajectory.step_rewards)):
        status = "✓" if step_reward.reward > 0.6 else "✗" if step_reward.reward < 0.4 else "~"
        lines.append(f"\nStep {i + 1}: [{status}] (reward: {step_reward.reward:.3f})")
        lines.append(f"  {step_text[:200]}{'...' if len(step_text) > 200 else ''}")

    lines.append("\n" + "=" * 80)

    return "\n".join(lines)


def export_trajectories_to_json(
    trajectories: List[TrajectoryReward],
    filepath: str
) -> None:
    """
    Export trajectories to JSON file for analysis.

    Args:
        trajectories: List of trajectory rewards
        filepath: Output file path
    """
    data = []

    for traj in trajectories:
        traj_dict = {
            "prompt": traj.prompt,
            "completion": traj.completion,
            "num_steps": traj.num_steps,
            "total_reward": traj.total_reward,
            "final_answer_reward": traj.final_answer_reward,
            "aggregated_reward": traj.aggregated_reward,
            "mean_step_reward": traj.mean_step_reward,
            "steps": [
                {
                    "index": sr.step_index,
                    "text": sr.step_text,
                    "reward": sr.reward,
                    "method": sr.evaluation_method,
                }
                for sr in traj.step_rewards
            ],
            "metadata": traj.metadata,
        }
        data.append(traj_dict)

    with open(filepath, 'w') as f:
        json.dump(data, f, indent=2)


def compute_trajectory_statistics(
    trajectories: List[TrajectoryReward]
) -> Dict[str, Any]:
    """
    Compute aggregate statistics across trajectories.

    Args:
        trajectories: List of trajectory rewards

    Returns:
        Dictionary with statistics
    """
    if not trajectories:
        return {}

    total_rewards = [t.total_reward for t in trajectories]
    step_rewards_all = [sr.reward for t in trajectories for sr in t.step_rewards]
    num_steps = [t.num_steps for t in trajectories]

    stats = {
        "num_trajectories": len(trajectories),
        "total_reward": {
            "mean": np.mean(total_rewards),
            "std": np.std(total_rewards),
            "min": np.min(total_rewards),
            "max": np.max(total_rewards),
            "median": np.median(total_rewards),
        },
        "step_reward": {
            "mean": np.mean(step_rewards_all),
            "std": np.std(step_rewards_all),
            "min": np.min(step_rewards_all),
            "max": np.max(step_rewards_all),
            "median": np.median(step_rewards_all),
        },
        "num_steps": {
            "mean": np.mean(num_steps),
            "std": np.std(num_steps),
            "min": np.min(num_steps),
            "max": np.max(num_steps),
        },
        "step_quality_distribution": _compute_quality_distribution(step_rewards_all),
    }

    return stats


def _compute_quality_distribution(rewards: List[float]) -> Dict[str, int]:
    """Compute distribution of step quality."""
    return {
        "excellent (>0.8)": sum(1 for r in rewards if r > 0.8),
        "good (0.6-0.8)": sum(1 for r in rewards if 0.6 < r <= 0.8),
        "fair (0.4-0.6)": sum(1 for r in rewards if 0.4 < r <= 0.6),
        "poor (<0.4)": sum(1 for r in rewards if r <= 0.4),
    }


def visualize_trajectory_rewards(
    trajectories: List[TrajectoryReward],
    max_display: int = 10
) -> str:
    """
    Create ASCII visualization of trajectory rewards.

    Args:
        trajectories: List of trajectories
        max_display: Maximum number of trajectories to display

    Returns:
        ASCII visualization string
    """
    lines = ["TRAJECTORY REWARDS VISUALIZATION", "=" * 80, ""]

    for i, traj in enumerate(trajectories[:max_display]):
        lines.append(f"Trajectory {i + 1} (Total: {traj.total_reward:.3f}):")

        # Create bar chart for step rewards
        step_rewards = [sr.reward for sr in traj.step_rewards]
        max_reward = max(step_rewards) if step_rewards else 1.0
        min_reward = min(step_rewards) if step_rewards else 0.0

        for j, reward in enumerate(step_rewards):
            # Normalize to 0-50 characters
            normalized = (reward - min_reward) / (max_reward - min_reward + 1e-6)
            bar_length = int(normalized * 50)
            bar = "█" * bar_length

            lines.append(f"  Step {j + 1:2d}: {bar} {reward:.3f}")

        lines.append("")

    return "\n".join(lines)


def filter_high_quality_trajectories(
    trajectories: List[TrajectoryReward],
    min_reward: float = 0.6,
    min_mean_step_reward: Optional[float] = None
) -> List[TrajectoryReward]:
    """
    Filter trajectories by quality criteria.

    Args:
        trajectories: List of trajectories
        min_reward: Minimum total reward
        min_mean_step_reward: Optional minimum mean step reward

    Returns:
        Filtered list of high-quality trajectories
    """
    filtered = []

    for traj in trajectories:
        if traj.total_reward < min_reward:
            continue

        if min_mean_step_reward is not None:
            if traj.mean_step_reward < min_mean_step_reward:
                continue

        filtered.append(traj)

    return filtered


def create_step_reward_heatmap_data(
    trajectories: List[TrajectoryReward],
    max_steps: int = 20
) -> np.ndarray:
    """
    Create heatmap data for step rewards across trajectories.

    Args:
        trajectories: List of trajectories
        max_steps: Maximum number of steps to include

    Returns:
        2D numpy array (trajectories × steps) of rewards
    """
    heatmap_data = []

    for traj in trajectories:
        step_rewards = [sr.reward for sr in traj.step_rewards]

        # Pad or truncate to max_steps
        if len(step_rewards) < max_steps:
            step_rewards += [0.0] * (max_steps - len(step_rewards))
        else:
            step_rewards = step_rewards[:max_steps]

        heatmap_data.append(step_rewards)

    return np.array(heatmap_data)
