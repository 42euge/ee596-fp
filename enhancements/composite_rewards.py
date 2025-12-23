"""
Composite Reward System

Multi-signal reward composition with configurable weights.
Based on research: CARD Framework and multi-objective RL for LLMs (2024-2025)

Usage:
    from enhancements.composite_rewards import CompositeReward

    reward = CompositeReward(
        components={'format': format_fn, 'answer': answer_fn},
        weights={'format': 0.3, 'answer': 0.7}
    )
"""

import numpy as np
from typing import Dict, Callable, List, Optional
from dataclasses import dataclass, field
import logging

logger = logging.getLogger(__name__)


@dataclass
class RewardComponent:
    """Represents a single reward component."""
    name: str
    function: Callable
    weight: float = 1.0
    enabled: bool = True


class CompositeReward:
    """
    Composable reward system that combines multiple reward signals.

    Features:
    - Weighted combination of multiple reward functions
    - Per-component logging and analysis
    - Dynamic weight adjustment
    - Component enable/disable

    Research basis:
    - CARD Framework (arXiv:2410.14660)
    - Multi-objective RLHF (2024-2025 surveys)
    """

    def __init__(
        self,
        components: Dict[str, Callable],
        weights: Optional[Dict[str, float]] = None,
        normalize: bool = True,
        log_components: bool = True
    ):
        """
        Initialize composite reward.

        Args:
            components: Dict mapping component names to reward functions
            weights: Dict mapping component names to weights (default: equal weights)
            normalize: Whether to normalize weights to sum to 1.0
            log_components: Whether to log individual component scores
        """
        self.components = {}
        self.log_components = log_components
        self.component_history = {name: [] for name in components.keys()}

        # Initialize weights
        if weights is None:
            weights = {name: 1.0 for name in components.keys()}

        # Normalize weights if requested
        if normalize:
            total = sum(weights.values())
            weights = {name: w / total for name, w in weights.items()}

        # Create RewardComponent objects
        for name, func in components.items():
            weight = weights.get(name, 1.0)
            self.components[name] = RewardComponent(
                name=name,
                function=func,
                weight=weight
            )

        logger.info(f"Initialized CompositeReward with {len(self.components)} components")
        for name, comp in self.components.items():
            logger.info(f"  - {name}: weight={comp.weight:.3f}")

    def __call__(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs
    ) -> List[float]:
        """
        Compute composite reward scores.

        Args:
            prompts: List of input prompts
            completions: List of model completions
            **kwargs: Additional arguments passed to component reward functions

        Returns:
            List of composite reward scores (one per completion)
        """
        n_samples = len(completions)
        composite_scores = np.zeros(n_samples)
        component_scores_dict = {}

        # Compute each component
        for name, component in self.components.items():
            if not component.enabled:
                continue

            try:
                # Call component reward function
                scores = component.function(prompts, completions, **kwargs)
                scores = np.array(scores)

                # Store for logging
                component_scores_dict[name] = scores

                # Add weighted contribution
                composite_scores += scores * component.weight

                # Log to history
                if self.log_components:
                    self.component_history[name].append({
                        'mean': float(np.mean(scores)),
                        'std': float(np.std(scores)),
                        'min': float(np.min(scores)),
                        'max': float(np.max(scores))
                    })

            except Exception as e:
                logger.error(f"Error computing reward component '{name}': {e}")
                # Continue with other components
                continue

        # Store latest component scores for external logging
        self.latest_component_scores = component_scores_dict

        return composite_scores.tolist()

    def get_component_scores(self) -> Dict[str, np.ndarray]:
        """Get the most recent component scores."""
        return self.latest_component_scores

    def update_weights(self, new_weights: Dict[str, float], normalize: bool = True):
        """
        Update component weights dynamically.

        Args:
            new_weights: Dict mapping component names to new weights
            normalize: Whether to normalize weights to sum to 1.0
        """
        if normalize:
            total = sum(new_weights.values())
            new_weights = {name: w / total for name, w in new_weights.items()}

        for name, weight in new_weights.items():
            if name in self.components:
                old_weight = self.components[name].weight
                self.components[name].weight = weight
                logger.info(f"Updated weight for '{name}': {old_weight:.3f} -> {weight:.3f}")

    def enable_component(self, name: str):
        """Enable a reward component."""
        if name in self.components:
            self.components[name].enabled = True
            logger.info(f"Enabled component: {name}")

    def disable_component(self, name: str):
        """Disable a reward component."""
        if name in self.components:
            self.components[name].enabled = False
            logger.info(f"Disabled component: {name}")

    def get_statistics(self) -> Dict[str, Dict[str, float]]:
        """
        Get statistics for all components across history.

        Returns:
            Dict mapping component names to statistics
        """
        stats = {}
        for name, history in self.component_history.items():
            if not history:
                continue

            means = [h['mean'] for h in history]
            stats[name] = {
                'overall_mean': float(np.mean(means)),
                'overall_std': float(np.std(means)),
                'trajectory_length': len(history),
                'latest_mean': history[-1]['mean'],
                'latest_std': history[-1]['std']
            }

        return stats


def create_default_composite_reward(
    format_fn: Callable,
    answer_fn: Callable,
    format_weight: float = 0.3,
    answer_weight: float = 0.7
) -> CompositeReward:
    """
    Create a default composite reward with format and answer components.

    Args:
        format_fn: Format checking reward function
        answer_fn: Answer correctness reward function
        format_weight: Weight for format component
        answer_weight: Weight for answer component

    Returns:
        CompositeReward instance
    """
    return CompositeReward(
        components={
            'format': format_fn,
            'answer': answer_fn
        },
        weights={
            'format': format_weight,
            'answer': answer_weight
        }
    )


# Example usage
if __name__ == "__main__":
    # Mock reward functions for demonstration
    def mock_format_reward(prompts, completions, **kwargs):
        return [1.0 if '<reasoning>' in c else 0.0 for c in completions]

    def mock_answer_reward(prompts, completions, **kwargs):
        answers = kwargs.get('answer', [])
        return [3.0 if a in c else 0.0 for a, c in zip(answers, completions)]

    # Create composite reward
    reward = CompositeReward(
        components={
            'format': mock_format_reward,
            'answer': mock_answer_reward
        },
        weights={
            'format': 0.3,
            'answer': 0.7
        }
    )

    # Test
    prompts = ["What is 2+2?"]
    completions = ["<reasoning>2+2=4</reasoning><answer>4</answer>"]
    answers = ["4"]

    scores = reward(prompts, completions, answer=answers)
    print(f"Composite scores: {scores}")
    print(f"Component scores: {reward.get_component_scores()}")
    print(f"Statistics: {reward.get_statistics()}")
