"""
Wrapper for reward functions to enable quality monitoring.

This module provides utilities to wrap existing reward functions
with quality monitoring capabilities.
"""

from typing import Callable, Dict, List, Any, Optional
import numpy as np
from functools import wraps

from .reward_quality import RewardQualityAssessor
from .reward_monitor import RewardQualityMonitor, InterventionConfig


class MonitoredRewardFunction:
    """
    Wrapper for reward functions that enables quality monitoring.

    This wrapper intercepts reward computations to track quality metrics
    and detect pathologies in real-time during training.
    """

    def __init__(
        self,
        reward_fn: Callable,
        reward_name: str,
        monitor: Optional[RewardQualityMonitor] = None,
        enable_monitoring: bool = True
    ):
        """
        Initialize monitored reward function.

        Args:
            reward_fn: Original reward function to wrap
            reward_name: Name identifier for this reward component
            monitor: Optional shared monitor instance
            enable_monitoring: Whether to enable monitoring
        """
        self.reward_fn = reward_fn
        self.reward_name = reward_name
        self.monitor = monitor
        self.enable_monitoring = enable_monitoring

        # Track calls for batch aggregation
        self.pending_responses = []
        self.pending_rewards = []
        self.call_count = 0
        self.batch_interval = 10  # Monitor every N calls

    def __call__(self, response: str, reference: str = "") -> float:
        """
        Compute reward and track for monitoring.

        Args:
            response: Model-generated response
            reference: Reference/ground-truth response

        Returns:
            Reward value
        """
        # Compute original reward
        reward = self.reward_fn(response, reference)

        # Track for monitoring
        if self.enable_monitoring and self.monitor:
            self.pending_responses.append(response)
            self.pending_rewards.append(reward)
            self.call_count += 1

            # Periodically assess batch
            if len(self.pending_responses) >= self.batch_interval:
                self._assess_pending_batch()

        return reward

    def _assess_pending_batch(self):
        """Assess accumulated responses and rewards."""
        if not self.pending_responses:
            return

        # Prepare rewards dict
        rewards = {self.reward_name: self.pending_rewards}

        # Monitor batch
        try:
            self.monitor.monitor_batch(
                responses=self.pending_responses,
                rewards=rewards,
                step=self.call_count
            )
        except Exception as e:
            import warnings
            warnings.warn(f"Monitoring failed: {e}")

        # Clear pending
        self.pending_responses = []
        self.pending_rewards = []

    def flush(self):
        """Flush any pending monitoring data."""
        if self.pending_responses:
            self._assess_pending_batch()


class RewardFunctionRegistry:
    """
    Registry for reward functions with unified monitoring.

    Manages multiple reward functions and provides centralized
    quality monitoring across all reward components.
    """

    def __init__(
        self,
        monitor: Optional[RewardQualityMonitor] = None,
        enable_monitoring: bool = True
    ):
        """
        Initialize registry.

        Args:
            monitor: Shared monitor for all reward functions
            enable_monitoring: Whether to enable monitoring
        """
        self.monitor = monitor
        self.enable_monitoring = enable_monitoring
        self.reward_functions: Dict[str, MonitoredRewardFunction] = {}

    def register(
        self,
        name: str,
        reward_fn: Callable,
        enable_monitoring: Optional[bool] = None
    ) -> MonitoredRewardFunction:
        """
        Register a reward function.

        Args:
            name: Name identifier for the reward function
            reward_fn: The reward function to register
            enable_monitoring: Override global monitoring setting

        Returns:
            Monitored reward function wrapper
        """
        monitoring_enabled = (
            enable_monitoring if enable_monitoring is not None
            else self.enable_monitoring
        )

        monitored_fn = MonitoredRewardFunction(
            reward_fn=reward_fn,
            reward_name=name,
            monitor=self.monitor,
            enable_monitoring=monitoring_enabled
        )

        self.reward_functions[name] = monitored_fn
        return monitored_fn

    def get(self, name: str) -> Optional[MonitoredRewardFunction]:
        """Get a registered reward function by name."""
        return self.reward_functions.get(name)

    def get_all_functions(self) -> List[Callable]:
        """Get all registered reward functions as a list."""
        return list(self.reward_functions.values())

    def flush_all(self):
        """Flush monitoring data for all reward functions."""
        for reward_fn in self.reward_functions.values():
            reward_fn.flush()

    def assess_combined_batch(
        self,
        responses: List[str],
        all_rewards: Dict[str, List[float]],
        step: int,
        references: Optional[List[str]] = None
    ):
        """
        Assess a batch with multiple reward components.

        This is useful when you have access to all responses and rewards
        at once (e.g., from a custom training loop).

        Args:
            responses: List of responses
            all_rewards: Dict mapping reward names to reward values
            step: Current training step
            references: Optional reference responses
        """
        if self.monitor and self.enable_monitoring:
            self.monitor.monitor_batch(
                responses=responses,
                rewards=all_rewards,
                step=step,
                references=references
            )


def create_monitored_reward_functions(
    reward_functions: List[Callable],
    reward_names: Optional[List[str]] = None,
    wandb_run: Optional[Any] = None,
    enable_interventions: bool = True
) -> tuple[List[MonitoredRewardFunction], RewardQualityMonitor]:
    """
    Convenience function to create monitored versions of reward functions.

    Args:
        reward_functions: List of reward functions to monitor
        reward_names: Optional names for the reward functions
        wandb_run: Optional W&B run for logging
        enable_interventions: Whether to enable automatic interventions

    Returns:
        Tuple of (list of monitored functions, monitor instance)
    """
    from .reward_monitor import create_default_monitor

    # Create monitor
    monitor = create_default_monitor(
        wandb_run=wandb_run,
        enable_interventions=enable_interventions
    )

    # Create registry
    registry = RewardFunctionRegistry(monitor=monitor)

    # Auto-generate names if not provided
    if reward_names is None:
        reward_names = [f"reward_{i}" for i in range(len(reward_functions))]
    elif len(reward_names) != len(reward_functions):
        raise ValueError("Number of reward names must match number of reward functions")

    # Register all functions
    monitored_functions = []
    for name, fn in zip(reward_names, reward_functions):
        monitored_fn = registry.register(name, fn)
        monitored_functions.append(monitored_fn)

    return monitored_functions, monitor


def batch_compute_rewards(
    responses: List[str],
    reward_functions: List[Callable],
    reward_names: List[str],
    references: Optional[List[str]] = None,
    monitor: Optional[RewardQualityMonitor] = None,
    step: int = 0
) -> Dict[str, List[float]]:
    """
    Compute rewards for a batch and optionally monitor quality.

    This is a utility function for computing multiple reward functions
    on a batch of responses with integrated quality monitoring.

    Args:
        responses: List of model responses
        reward_functions: List of reward functions
        reward_names: Names for each reward function
        references: Optional reference responses
        monitor: Optional quality monitor
        step: Current training step

    Returns:
        Dictionary mapping reward names to lists of reward values
    """
    if len(reward_functions) != len(reward_names):
        raise ValueError("Number of functions must match number of names")

    if references is None:
        references = [""] * len(responses)
    elif len(references) != len(responses):
        raise ValueError("Number of references must match number of responses")

    # Compute all rewards
    all_rewards = {}
    for fn, name in zip(reward_functions, reward_names):
        rewards = [fn(resp, ref) for resp, ref in zip(responses, references)]
        all_rewards[name] = rewards

    # Monitor if enabled
    if monitor:
        monitor.monitor_batch(
            responses=responses,
            rewards=all_rewards,
            step=step,
            references=references
        )

    return all_rewards


def extract_rewards_from_grpo_output(
    grpo_output: Any,
    reward_names: List[str]
) -> Dict[str, List[float]]:
    """
    Extract reward values from GRPO trainer output.

    This is a helper function to parse reward values from the
    GRPO trainer's output format.

    Args:
        grpo_output: Output from GRPO trainer
        reward_names: Names of reward components

    Returns:
        Dictionary of reward values
    """
    # This will need to be customized based on the actual GRPO output format
    # For now, return a placeholder structure
    rewards = {}

    # Try to extract rewards from output
    if hasattr(grpo_output, 'rewards'):
        if isinstance(grpo_output.rewards, dict):
            rewards = grpo_output.rewards
        elif isinstance(grpo_output.rewards, (list, np.ndarray)):
            # Assume rewards are concatenated, split by reward_names
            reward_array = np.array(grpo_output.rewards)
            if len(reward_names) > 0:
                split_size = len(reward_array) // len(reward_names)
                for i, name in enumerate(reward_names):
                    rewards[name] = reward_array[i*split_size:(i+1)*split_size].tolist()

    return rewards
