"""W&B logging wrappers for reward functions.

Wraps GRPO reward functions to log detailed metrics to Weights & Biases.
"""

from typing import List, Callable, Dict, Any
import statistics


class WandbRewardLogger:
    """Wrapper that logs reward function outputs to W&B."""

    def __init__(self, log_every_n_steps: int = 10):
        """Initialize the logger.

        Args:
            log_every_n_steps: How often to log detailed metrics
        """
        self.log_every_n_steps = log_every_n_steps
        self.step = 0
        self.reward_history: Dict[str, List[float]] = {}
        self._wandb = None

    def _get_wandb(self):
        """Lazy import wandb."""
        if self._wandb is None:
            try:
                import wandb
                self._wandb = wandb
            except ImportError:
                self._wandb = False
        return self._wandb if self._wandb else None

    def wrap_reward_fn(
        self,
        reward_fn: Callable,
        name: str,
    ) -> Callable:
        """Wrap a reward function to log its outputs.

        Args:
            reward_fn: Original reward function
            name: Name for logging (e.g., "format_exact", "rubric")

        Returns:
            Wrapped reward function that logs to W&B
        """
        if name not in self.reward_history:
            self.reward_history[name] = []

        def wrapped(
            prompts: List[str],
            completions: List[str],
            **kwargs,
        ) -> List[float]:
            scores = reward_fn(prompts, completions, **kwargs)

            # Store scores
            self.reward_history[name].extend(scores)

            # Log to W&B
            wandb = self._get_wandb()
            if wandb and wandb.run:
                # Log mean score for this batch
                if scores:
                    wandb.log({
                        f"rewards/{name}_mean": statistics.mean(scores),
                        f"rewards/{name}_max": max(scores),
                        f"rewards/{name}_min": min(scores),
                    }, commit=False)

                    # Log histogram periodically
                    if self.step % self.log_every_n_steps == 0 and len(self.reward_history[name]) > 10:
                        wandb.log({
                            f"rewards/{name}_hist": wandb.Histogram(self.reward_history[name][-100:]),
                        }, commit=False)

            return scores

        return wrapped

    def wrap_all_rewards(
        self,
        reward_fns: List[Callable],
        names: List[str],
    ) -> List[Callable]:
        """Wrap multiple reward functions.

        Args:
            reward_fns: List of reward functions
            names: Names for each function

        Returns:
            List of wrapped functions
        """
        wrapped = []
        for fn, name in zip(reward_fns, names):
            wrapped.append(self.wrap_reward_fn(fn, name))
        return wrapped

    def log_step(self, step: int, extra_metrics: Dict[str, Any] | None = None):
        """Log aggregated metrics for a training step.

        Args:
            step: Current training step
            extra_metrics: Additional metrics to log
        """
        self.step = step
        wandb = self._get_wandb()
        if not wandb or not wandb.run:
            return

        metrics = {"train/step": step}

        # Compute total reward
        total_rewards = []
        for name, scores in self.reward_history.items():
            if scores:
                # Get recent scores (last batch)
                recent = scores[-10:] if len(scores) >= 10 else scores
                metrics[f"rewards/{name}_recent_mean"] = statistics.mean(recent)

                # Accumulate for total
                if not total_rewards:
                    total_rewards = list(recent)
                else:
                    for i, s in enumerate(recent[:len(total_rewards)]):
                        total_rewards[i] += s

        if total_rewards:
            metrics["rewards/total_mean"] = statistics.mean(total_rewards)

        if extra_metrics:
            metrics.update(extra_metrics)

        wandb.log(metrics)

    def log_summary(self):
        """Log final summary statistics."""
        wandb = self._get_wandb()
        if not wandb or not wandb.run:
            return

        summary = {}
        for name, scores in self.reward_history.items():
            if scores:
                summary[f"final/{name}_mean"] = statistics.mean(scores)
                summary[f"final/{name}_std"] = statistics.stdev(scores) if len(scores) > 1 else 0
                summary[f"final/{name}_max"] = max(scores)
                summary[f"final/{name}_min"] = min(scores)
                summary[f"final/{name}_total_samples"] = len(scores)

        wandb.log(summary)


class RubricRewardLogger(WandbRewardLogger):
    """Extended logger for rubric-based rewards with per-criterion tracking."""

    def __init__(self, log_every_n_steps: int = 10):
        super().__init__(log_every_n_steps)
        self.criterion_scores: Dict[str, List[float]] = {}

    def wrap_rubric_reward(
        self,
        rubricset,
        name: str = "rubric",
        weight: float = 1.0,
    ) -> Callable:
        """Wrap a rubric reward function with detailed criterion logging.

        Args:
            rubricset: RubricSet or Rubric to use
            name: Base name for logging
            weight: Weight multiplier for scores

        Returns:
            Wrapped reward function
        """
        from src.rubrics import RubricSet, Rubric
        from src.utils import rubric_overlap_score

        if isinstance(rubricset, Rubric):
            rubricset = RubricSet(name=rubricset.name, rubrics=[rubricset])

        # Initialize criterion tracking
        for rubric in rubricset.rubrics:
            for criterion in rubric.criteria:
                key = f"{name}/{rubric.name}/{criterion.name}"
                self.criterion_scores[key] = []

        def scorer(
            prompts: List[str],
            completions: List[str],
            **kwargs,
        ) -> List[float]:
            scores = []
            wandb = self._get_wandb()

            for prompt, completion in zip(prompts, completions):
                rubric = rubricset.get_for_question(prompt)
                total_score = 0.0

                if rubric:
                    for criterion in rubric.criteria:
                        criterion_text = criterion.description
                        if criterion.keywords:
                            criterion_text += " " + " ".join(criterion.keywords)

                        criterion_score = rubric_overlap_score(completion, criterion_text)
                        weighted_criterion = criterion_score * criterion.weight
                        total_score += weighted_criterion

                        # Track criterion score
                        key = f"{name}/{rubric.name}/{criterion.name}"
                        if key in self.criterion_scores:
                            self.criterion_scores[key].append(criterion_score)

                scores.append(total_score * weight)

            # Store in history
            if name not in self.reward_history:
                self.reward_history[name] = []
            self.reward_history[name].extend(scores)

            # Log to W&B
            if wandb and wandb.run and scores:
                log_data = {
                    f"rewards/{name}_mean": statistics.mean(scores),
                    f"rewards/{name}_max": max(scores),
                    f"rewards/{name}_min": min(scores),
                }

                # Log criterion scores periodically
                if self.step % self.log_every_n_steps == 0:
                    for key, crit_scores in self.criterion_scores.items():
                        if crit_scores:
                            recent = crit_scores[-20:]
                            log_data[f"criteria/{key}_mean"] = statistics.mean(recent)

                wandb.log(log_data, commit=False)

            return scores

        return scorer

    def log_rubric_summary(self):
        """Log detailed rubric summary."""
        wandb = self._get_wandb()
        if not wandb or not wandb.run:
            return

        # Create a table for criterion performance
        columns = ["criterion", "mean", "std", "min", "max", "samples"]
        data = []

        for key, scores in self.criterion_scores.items():
            if scores:
                data.append([
                    key,
                    round(statistics.mean(scores), 3),
                    round(statistics.stdev(scores) if len(scores) > 1 else 0, 3),
                    round(min(scores), 3),
                    round(max(scores), 3),
                    len(scores),
                ])

        if data:
            table = wandb.Table(columns=columns, data=data)
            wandb.log({"rubric_criteria_summary": table})

        # Also log summary metrics
        self.log_summary()


def create_logged_reward_fns(
    reward_fns: List[Callable],
    names: List[str] | None = None,
    rubricset=None,
    rubric_weight: float = 1.0,
    log_every_n_steps: int = 10,
) -> tuple[List[Callable], WandbRewardLogger]:
    """Create logged versions of reward functions.

    Args:
        reward_fns: List of base reward functions
        names: Names for each function (auto-generated if None)
        rubricset: Optional rubricset for rubric reward
        rubric_weight: Weight for rubric reward
        log_every_n_steps: How often to log histograms

    Returns:
        Tuple of (wrapped_functions, logger)
    """
    if names is None:
        names = [f"reward_{i}" for i in range(len(reward_fns))]

    if rubricset:
        logger = RubricRewardLogger(log_every_n_steps)
    else:
        logger = WandbRewardLogger(log_every_n_steps)

    wrapped = logger.wrap_all_rewards(reward_fns, names)

    if rubricset:
        rubric_fn = logger.wrap_rubric_reward(rubricset, "rubric", rubric_weight)
        wrapped.append(rubric_fn)

    return wrapped, logger
