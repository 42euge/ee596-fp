"""
Hyperparameter Search Module for Gemma3-1B GRPO Fine-tuning

Supports multiple search strategies:
- Grid Search: Exhaustive search over parameter combinations
- Random Search: Random sampling from parameter distributions
- Bayesian Optimization: Sequential model-based optimization (via Optuna)

Usage:
    from src.hyperparam_search import HyperparameterSearch, SearchSpace

    # Define search space
    search_space = SearchSpace()
    search_space.add_continuous("learning_rate", 1e-6, 1e-4, log_scale=True)
    search_space.add_discrete("lora_rank", [16, 32, 64, 128])
    search_space.add_continuous("grpo_beta", 0.01, 0.2)

    # Run search
    searcher = HyperparameterSearch(
        search_space=search_space,
        objective_fn=my_objective,
        strategy="bayesian",
        n_trials=50,
    )
    best_params, best_score = searcher.search()
"""

import json
import os
import time
import random
import itertools
import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field, asdict
from datetime import datetime
from pathlib import Path
from typing import (
    Any, Callable, Dict, List, Optional, Tuple, Union, Iterator
)

import numpy as np

from .config import (
    Config, LoRAConfig, GRPOConfig, TrainingConfig, DataConfig,
    get_default_config
)


# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Search Space Definition
# =============================================================================

@dataclass
class ParameterSpec:
    """Specification for a single hyperparameter."""
    name: str
    param_type: str  # "continuous", "discrete", "categorical", "integer"
    low: Optional[float] = None
    high: Optional[float] = None
    choices: Optional[List[Any]] = None
    log_scale: bool = False
    default: Optional[Any] = None
    config_path: Optional[str] = None  # e.g., "lora.rank" or "training.learning_rate"

    def sample(self, rng: Optional[np.random.Generator] = None) -> Any:
        """Sample a value from this parameter's distribution."""
        if rng is None:
            rng = np.random.default_rng()

        if self.param_type == "continuous":
            if self.log_scale:
                log_low = np.log(self.low)
                log_high = np.log(self.high)
                return float(np.exp(rng.uniform(log_low, log_high)))
            return float(rng.uniform(self.low, self.high))

        elif self.param_type == "integer":
            return int(rng.integers(self.low, self.high + 1))

        elif self.param_type in ("discrete", "categorical"):
            return rng.choice(self.choices)

        raise ValueError(f"Unknown parameter type: {self.param_type}")

    def grid_values(self, n_points: int = 5) -> List[Any]:
        """Get grid values for this parameter."""
        if self.param_type == "continuous":
            if self.log_scale:
                return list(np.logspace(
                    np.log10(self.low),
                    np.log10(self.high),
                    n_points
                ))
            return list(np.linspace(self.low, self.high, n_points))

        elif self.param_type == "integer":
            step = max(1, (self.high - self.low) // (n_points - 1))
            return list(range(int(self.low), int(self.high) + 1, step))

        elif self.param_type in ("discrete", "categorical"):
            return list(self.choices)

        raise ValueError(f"Unknown parameter type: {self.param_type}")


class SearchSpace:
    """Defines the hyperparameter search space."""

    def __init__(self):
        self.parameters: Dict[str, ParameterSpec] = {}

    def add_continuous(
        self,
        name: str,
        low: float,
        high: float,
        log_scale: bool = False,
        default: Optional[float] = None,
        config_path: Optional[str] = None,
    ) -> "SearchSpace":
        """Add a continuous parameter to the search space."""
        self.parameters[name] = ParameterSpec(
            name=name,
            param_type="continuous",
            low=low,
            high=high,
            log_scale=log_scale,
            default=default,
            config_path=config_path,
        )
        return self

    def add_integer(
        self,
        name: str,
        low: int,
        high: int,
        default: Optional[int] = None,
        config_path: Optional[str] = None,
    ) -> "SearchSpace":
        """Add an integer parameter to the search space."""
        self.parameters[name] = ParameterSpec(
            name=name,
            param_type="integer",
            low=low,
            high=high,
            default=default,
            config_path=config_path,
        )
        return self

    def add_discrete(
        self,
        name: str,
        choices: List[Any],
        default: Optional[Any] = None,
        config_path: Optional[str] = None,
    ) -> "SearchSpace":
        """Add a discrete parameter (numeric choices) to the search space."""
        self.parameters[name] = ParameterSpec(
            name=name,
            param_type="discrete",
            choices=choices,
            default=default,
            config_path=config_path,
        )
        return self

    def add_categorical(
        self,
        name: str,
        choices: List[Any],
        default: Optional[Any] = None,
        config_path: Optional[str] = None,
    ) -> "SearchSpace":
        """Add a categorical parameter (non-numeric choices) to the search space."""
        self.parameters[name] = ParameterSpec(
            name=name,
            param_type="categorical",
            choices=choices,
            default=default,
            config_path=config_path,
        )
        return self

    def sample(self, rng: Optional[np.random.Generator] = None) -> Dict[str, Any]:
        """Sample a random configuration from the search space."""
        if rng is None:
            rng = np.random.default_rng()
        return {name: spec.sample(rng) for name, spec in self.parameters.items()}

    def grid_iterator(self, n_points_per_param: int = 5) -> Iterator[Dict[str, Any]]:
        """Iterate over all grid combinations."""
        param_grids = {
            name: spec.grid_values(n_points_per_param)
            for name, spec in self.parameters.items()
        }

        keys = list(param_grids.keys())
        values = [param_grids[k] for k in keys]

        for combo in itertools.product(*values):
            yield dict(zip(keys, combo))

    def get_defaults(self) -> Dict[str, Any]:
        """Get default values for all parameters."""
        return {
            name: spec.default
            for name, spec in self.parameters.items()
            if spec.default is not None
        }

    def __len__(self) -> int:
        return len(self.parameters)

    def __repr__(self) -> str:
        return f"SearchSpace({list(self.parameters.keys())})"


# =============================================================================
# Trial Results
# =============================================================================

@dataclass
class TrialResult:
    """Result of a single hyperparameter trial."""
    trial_id: int
    params: Dict[str, Any]
    score: float
    metrics: Dict[str, float] = field(default_factory=dict)
    duration_seconds: float = 0.0
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    status: str = "completed"  # "completed", "failed", "pruned"
    error_message: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for serialization."""
        return asdict(self)

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "TrialResult":
        """Create from dictionary."""
        return cls(**data)


@dataclass
class SearchResults:
    """Collection of all trial results from a search."""
    trials: List[TrialResult] = field(default_factory=list)
    best_trial: Optional[TrialResult] = None
    search_strategy: str = ""
    search_space_config: Dict[str, Any] = field(default_factory=dict)
    total_time_seconds: float = 0.0
    n_completed: int = 0
    n_failed: int = 0

    def add_trial(self, trial: TrialResult) -> None:
        """Add a trial result and update best if necessary."""
        self.trials.append(trial)

        if trial.status == "completed":
            self.n_completed += 1
            if self.best_trial is None or trial.score > self.best_trial.score:
                self.best_trial = trial
        else:
            self.n_failed += 1

    def get_best_params(self) -> Optional[Dict[str, Any]]:
        """Get parameters from the best trial."""
        return self.best_trial.params if self.best_trial else None

    def get_best_score(self) -> Optional[float]:
        """Get score from the best trial."""
        return self.best_trial.score if self.best_trial else None

    def to_dataframe(self):
        """Convert to pandas DataFrame (if pandas available)."""
        try:
            import pandas as pd
            records = []
            for trial in self.trials:
                record = {"trial_id": trial.trial_id, "score": trial.score}
                record.update(trial.params)
                record.update(trial.metrics)
                records.append(record)
            return pd.DataFrame(records)
        except ImportError:
            logger.warning("pandas not available, returning list of dicts")
            return [t.to_dict() for t in self.trials]

    def save(self, path: str) -> None:
        """Save results to JSON file."""
        data = {
            "trials": [t.to_dict() for t in self.trials],
            "best_trial": self.best_trial.to_dict() if self.best_trial else None,
            "search_strategy": self.search_strategy,
            "search_space_config": self.search_space_config,
            "total_time_seconds": self.total_time_seconds,
            "n_completed": self.n_completed,
            "n_failed": self.n_failed,
        }
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Results saved to {path}")

    @classmethod
    def load(cls, path: str) -> "SearchResults":
        """Load results from JSON file."""
        with open(path) as f:
            data = json.load(f)

        results = cls(
            search_strategy=data["search_strategy"],
            search_space_config=data["search_space_config"],
            total_time_seconds=data["total_time_seconds"],
            n_completed=data["n_completed"],
            n_failed=data["n_failed"],
        )
        results.trials = [TrialResult.from_dict(t) for t in data["trials"]]
        if data["best_trial"]:
            results.best_trial = TrialResult.from_dict(data["best_trial"])
        return results


# =============================================================================
# Search Strategies
# =============================================================================

class SearchStrategy(ABC):
    """Abstract base class for search strategies."""

    @abstractmethod
    def suggest(self, trial_id: int) -> Dict[str, Any]:
        """Suggest parameters for the next trial."""
        pass

    @abstractmethod
    def report(self, trial_id: int, score: float, params: Dict[str, Any]) -> None:
        """Report the result of a trial."""
        pass

    @abstractmethod
    def should_stop(self) -> bool:
        """Check if search should stop early."""
        pass


class GridSearchStrategy(SearchStrategy):
    """Exhaustive grid search over parameter combinations."""

    def __init__(
        self,
        search_space: SearchSpace,
        n_points_per_param: int = 5,
    ):
        self.search_space = search_space
        self.n_points_per_param = n_points_per_param
        self._grid_iter = None
        self._exhausted = False
        self._reset_iterator()

    def _reset_iterator(self):
        self._grid_iter = self.search_space.grid_iterator(self.n_points_per_param)

    def suggest(self, trial_id: int) -> Dict[str, Any]:
        try:
            return next(self._grid_iter)
        except StopIteration:
            self._exhausted = True
            raise StopIteration("Grid search exhausted all combinations")

    def report(self, trial_id: int, score: float, params: Dict[str, Any]) -> None:
        pass  # Grid search doesn't adapt based on results

    def should_stop(self) -> bool:
        return self._exhausted

    def total_combinations(self) -> int:
        """Calculate total number of grid combinations."""
        total = 1
        for spec in self.search_space.parameters.values():
            total *= len(spec.grid_values(self.n_points_per_param))
        return total


class RandomSearchStrategy(SearchStrategy):
    """Random search over parameter distributions."""

    def __init__(
        self,
        search_space: SearchSpace,
        n_trials: int = 100,
        seed: Optional[int] = None,
    ):
        self.search_space = search_space
        self.n_trials = n_trials
        self.rng = np.random.default_rng(seed)
        self._trial_count = 0

    def suggest(self, trial_id: int) -> Dict[str, Any]:
        if self._trial_count >= self.n_trials:
            raise StopIteration("Random search completed all trials")
        self._trial_count += 1
        return self.search_space.sample(self.rng)

    def report(self, trial_id: int, score: float, params: Dict[str, Any]) -> None:
        pass  # Random search doesn't adapt

    def should_stop(self) -> bool:
        return self._trial_count >= self.n_trials


class BayesianSearchStrategy(SearchStrategy):
    """Bayesian optimization using Optuna."""

    def __init__(
        self,
        search_space: SearchSpace,
        n_trials: int = 100,
        seed: Optional[int] = None,
        pruner: Optional[Any] = None,
        sampler: Optional[Any] = None,
    ):
        try:
            import optuna
            self.optuna = optuna
        except ImportError:
            raise ImportError(
                "Optuna required for Bayesian search. Install with: pip install optuna"
            )

        self.search_space = search_space
        self.n_trials = n_trials
        self.seed = seed
        self._trial_count = 0

        # Create Optuna study
        if sampler is None:
            sampler = optuna.samplers.TPESampler(seed=seed)
        if pruner is None:
            pruner = optuna.pruners.MedianPruner()

        self.study = optuna.create_study(
            direction="maximize",
            sampler=sampler,
            pruner=pruner,
        )
        self._current_trial = None

    def suggest(self, trial_id: int) -> Dict[str, Any]:
        if self._trial_count >= self.n_trials:
            raise StopIteration("Bayesian search completed all trials")

        # Create Optuna trial
        self._current_trial = self.study.ask()
        params = {}

        for name, spec in self.search_space.parameters.items():
            if spec.param_type == "continuous":
                if spec.log_scale:
                    params[name] = self._current_trial.suggest_float(
                        name, spec.low, spec.high, log=True
                    )
                else:
                    params[name] = self._current_trial.suggest_float(
                        name, spec.low, spec.high
                    )
            elif spec.param_type == "integer":
                params[name] = self._current_trial.suggest_int(
                    name, int(spec.low), int(spec.high)
                )
            elif spec.param_type in ("discrete", "categorical"):
                params[name] = self._current_trial.suggest_categorical(
                    name, spec.choices
                )

        self._trial_count += 1
        return params

    def report(self, trial_id: int, score: float, params: Dict[str, Any]) -> None:
        if self._current_trial is not None:
            self.study.tell(self._current_trial, score)

    def should_stop(self) -> bool:
        return self._trial_count >= self.n_trials

    def get_best_params(self) -> Dict[str, Any]:
        """Get the best parameters found by Optuna."""
        return self.study.best_params

    def get_optimization_history(self) -> List[Tuple[int, float]]:
        """Get optimization history (trial number, best value so far)."""
        history = []
        best_so_far = float("-inf")
        for i, trial in enumerate(self.study.trials):
            if trial.value is not None and trial.value > best_so_far:
                best_so_far = trial.value
            history.append((i, best_so_far))
        return history


# =============================================================================
# Main Hyperparameter Search Class
# =============================================================================

class HyperparameterSearch:
    """Main class for conducting hyperparameter searches."""

    def __init__(
        self,
        search_space: SearchSpace,
        objective_fn: Callable[[Dict[str, Any]], Tuple[float, Dict[str, float]]],
        strategy: str = "random",
        n_trials: int = 100,
        n_grid_points: int = 5,
        seed: Optional[int] = 42,
        output_dir: str = "./hyperparam_results",
        checkpoint_every: int = 10,
        verbose: bool = True,
    ):
        """
        Initialize hyperparameter search.

        Args:
            search_space: SearchSpace defining parameters to optimize
            objective_fn: Function that takes params dict, returns (score, metrics_dict)
            strategy: Search strategy ("grid", "random", "bayesian")
            n_trials: Number of trials for random/bayesian search
            n_grid_points: Points per parameter for grid search
            seed: Random seed for reproducibility
            output_dir: Directory to save results
            checkpoint_every: Save checkpoint every N trials
            verbose: Print progress updates
        """
        self.search_space = search_space
        self.objective_fn = objective_fn
        self.strategy_name = strategy
        self.n_trials = n_trials
        self.n_grid_points = n_grid_points
        self.seed = seed
        self.output_dir = Path(output_dir)
        self.checkpoint_every = checkpoint_every
        self.verbose = verbose

        # Create output directory
        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Initialize strategy
        self._strategy = self._create_strategy()

        # Initialize results
        self.results = SearchResults(
            search_strategy=strategy,
            search_space_config={
                name: {
                    "type": spec.param_type,
                    "low": spec.low,
                    "high": spec.high,
                    "choices": spec.choices,
                    "log_scale": spec.log_scale,
                }
                for name, spec in search_space.parameters.items()
            },
        )

    def _create_strategy(self) -> SearchStrategy:
        """Create the appropriate search strategy."""
        if self.strategy_name == "grid":
            return GridSearchStrategy(self.search_space, self.n_grid_points)
        elif self.strategy_name == "random":
            return RandomSearchStrategy(self.search_space, self.n_trials, self.seed)
        elif self.strategy_name == "bayesian":
            return BayesianSearchStrategy(self.search_space, self.n_trials, self.seed)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy_name}")

    def search(self) -> Tuple[Dict[str, Any], float]:
        """
        Run the hyperparameter search.

        Returns:
            Tuple of (best_params, best_score)
        """
        start_time = time.time()
        trial_id = 0

        if self.verbose:
            logger.info(f"Starting {self.strategy_name} search")
            logger.info(f"Search space: {self.search_space}")
            if self.strategy_name == "grid":
                total = self._strategy.total_combinations()
                logger.info(f"Total grid combinations: {total}")

        while not self._strategy.should_stop():
            try:
                # Get next parameters to try
                params = self._strategy.suggest(trial_id)
            except StopIteration:
                break

            if self.verbose:
                logger.info(f"\nTrial {trial_id + 1}: {params}")

            # Run objective function
            trial_start = time.time()
            try:
                score, metrics = self.objective_fn(params)
                trial_duration = time.time() - trial_start

                trial_result = TrialResult(
                    trial_id=trial_id,
                    params=params,
                    score=score,
                    metrics=metrics,
                    duration_seconds=trial_duration,
                    status="completed",
                )

                if self.verbose:
                    logger.info(f"  Score: {score:.6f} (took {trial_duration:.1f}s)")

            except Exception as e:
                trial_duration = time.time() - trial_start
                trial_result = TrialResult(
                    trial_id=trial_id,
                    params=params,
                    score=float("-inf"),
                    duration_seconds=trial_duration,
                    status="failed",
                    error_message=str(e),
                )
                logger.error(f"  Trial failed: {e}")

            # Report result to strategy
            self._strategy.report(trial_id, trial_result.score, params)

            # Add to results
            self.results.add_trial(trial_result)

            # Checkpoint
            if (trial_id + 1) % self.checkpoint_every == 0:
                self._save_checkpoint()

            trial_id += 1

        # Final save
        self.results.total_time_seconds = time.time() - start_time
        self._save_final_results()

        if self.verbose:
            self._print_summary()

        best_params = self.results.get_best_params() or {}
        best_score = self.results.get_best_score() or float("-inf")
        return best_params, best_score

    def _save_checkpoint(self) -> None:
        """Save intermediate checkpoint."""
        checkpoint_path = self.output_dir / "checkpoint.json"
        self.results.save(str(checkpoint_path))

    def _save_final_results(self) -> None:
        """Save final results."""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        final_path = self.output_dir / f"results_{timestamp}.json"
        self.results.save(str(final_path))

        # Also save as latest
        latest_path = self.output_dir / "results_latest.json"
        self.results.save(str(latest_path))

    def _print_summary(self) -> None:
        """Print search summary."""
        logger.info("\n" + "=" * 60)
        logger.info("HYPERPARAMETER SEARCH COMPLETE")
        logger.info("=" * 60)
        logger.info(f"Strategy: {self.strategy_name}")
        logger.info(f"Completed trials: {self.results.n_completed}")
        logger.info(f"Failed trials: {self.results.n_failed}")
        logger.info(f"Total time: {self.results.total_time_seconds:.1f}s")

        if self.results.best_trial:
            logger.info(f"\nBest Score: {self.results.best_trial.score:.6f}")
            logger.info("Best Parameters:")
            for k, v in self.results.best_trial.params.items():
                logger.info(f"  {k}: {v}")

            if self.results.best_trial.metrics:
                logger.info("Best Metrics:")
                for k, v in self.results.best_trial.metrics.items():
                    logger.info(f"  {k}: {v:.4f}")


# =============================================================================
# Predefined Search Spaces
# =============================================================================

def get_lora_search_space() -> SearchSpace:
    """Get search space for LoRA hyperparameters."""
    space = SearchSpace()
    space.add_discrete("lora_rank", [8, 16, 32, 64, 128], default=64, config_path="lora.rank")
    space.add_discrete("lora_alpha", [8, 16, 32, 64, 128, 256], default=64, config_path="lora.alpha")
    return space


def get_training_search_space() -> SearchSpace:
    """Get search space for training hyperparameters."""
    space = SearchSpace()
    space.add_continuous(
        "learning_rate", 1e-7, 1e-4, log_scale=True, default=3e-6,
        config_path="training.learning_rate"
    )
    space.add_discrete(
        "batch_size", [1, 2, 4, 8], default=2,
        config_path="training.train_micro_batch_size"
    )
    space.add_continuous(
        "warmup_ratio", 0.0, 0.2, default=0.1,
        config_path="training.warmup_ratio"
    )
    space.add_continuous(
        "max_grad_norm", 0.01, 1.0, log_scale=True, default=0.1,
        config_path="training.max_grad_norm"
    )
    space.add_continuous(
        "weight_decay", 0.0, 0.3, default=0.1,
        config_path="training.weight_decay"
    )
    return space


def get_grpo_search_space() -> SearchSpace:
    """Get search space for GRPO hyperparameters."""
    space = SearchSpace()
    space.add_continuous(
        "grpo_temperature", 0.5, 1.2, default=0.9,
        config_path="grpo.temperature"
    )
    space.add_continuous(
        "grpo_beta", 0.01, 0.2, default=0.08,
        config_path="grpo.beta"
    )
    space.add_continuous(
        "grpo_epsilon", 0.1, 0.3, default=0.2,
        config_path="grpo.epsilon"
    )
    space.add_discrete(
        "num_generations", [1, 2, 4, 8], default=2,
        config_path="grpo.num_generations"
    )
    return space


def get_generation_search_space() -> SearchSpace:
    """Get search space for generation/inference hyperparameters."""
    space = SearchSpace()
    space.add_continuous(
        "temperature", 0.1, 1.5, default=0.7,
        config_path="inference.temperature"
    )
    space.add_discrete(
        "top_k", [1, 10, 50, 100, 500], default=50,
        config_path="inference.top_k"
    )
    space.add_continuous(
        "top_p", 0.5, 1.0, default=0.95,
        config_path="inference.top_p"
    )
    return space


def get_full_search_space() -> SearchSpace:
    """Get comprehensive search space combining all hyperparameters."""
    space = SearchSpace()

    # LoRA parameters
    space.add_discrete("lora_rank", [16, 32, 64, 128], config_path="lora.rank")
    space.add_discrete("lora_alpha", [16, 32, 64, 128], config_path="lora.alpha")

    # Training parameters
    space.add_continuous(
        "learning_rate", 1e-7, 1e-4, log_scale=True,
        config_path="training.learning_rate"
    )
    space.add_continuous(
        "warmup_ratio", 0.0, 0.2,
        config_path="training.warmup_ratio"
    )
    space.add_continuous(
        "max_grad_norm", 0.01, 1.0, log_scale=True,
        config_path="training.max_grad_norm"
    )

    # GRPO parameters
    space.add_continuous(
        "grpo_beta", 0.01, 0.2,
        config_path="grpo.beta"
    )
    space.add_continuous(
        "grpo_epsilon", 0.1, 0.3,
        config_path="grpo.epsilon"
    )

    return space


# =============================================================================
# Utility Functions
# =============================================================================

def params_to_config(params: Dict[str, Any], base_config: Optional[Config] = None) -> Config:
    """
    Convert a parameters dictionary to a Config object.

    Args:
        params: Dictionary of parameter values
        base_config: Base configuration to modify (creates default if None)

    Returns:
        Modified Config object
    """
    config = base_config or get_default_config()

    # Map parameter names to config paths
    param_mapping = {
        # LoRA
        "lora_rank": ("lora", "rank"),
        "lora_alpha": ("lora", "alpha"),
        # Training
        "learning_rate": ("training", "learning_rate"),
        "batch_size": ("training", "train_micro_batch_size"),
        "warmup_ratio": ("training", "warmup_ratio"),
        "max_grad_norm": ("training", "max_grad_norm"),
        "weight_decay": ("training", "weight_decay"),
        "num_epochs": ("training", "num_epochs"),
        # GRPO
        "grpo_temperature": ("grpo", "temperature"),
        "grpo_beta": ("grpo", "beta"),
        "grpo_epsilon": ("grpo", "epsilon"),
        "num_generations": ("grpo", "num_generations"),
        "grpo_top_p": ("grpo", "top_p"),
        "grpo_top_k": ("grpo", "top_k"),
        # Generation
        "temperature": None,  # Handled specially
        "top_k": None,
        "top_p": None,
    }

    for param_name, value in params.items():
        if param_name in param_mapping and param_mapping[param_name] is not None:
            section, attr = param_mapping[param_name]
            sub_config = getattr(config, section)
            setattr(sub_config, attr, value)

    return config


def create_objective_from_eval_fn(
    eval_fn: Callable[[Config], Dict[str, float]],
    metric_name: str = "accuracy",
    base_config: Optional[Config] = None,
) -> Callable[[Dict[str, Any]], Tuple[float, Dict[str, float]]]:
    """
    Create an objective function from an evaluation function.

    Args:
        eval_fn: Function that takes Config and returns metrics dict
        metric_name: Name of metric to optimize
        base_config: Base configuration to use

    Returns:
        Objective function for hyperparameter search
    """
    def objective(params: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        config = params_to_config(params, base_config)
        metrics = eval_fn(config)
        score = metrics.get(metric_name, 0.0)
        return score, metrics

    return objective


# =============================================================================
# Analysis Functions
# =============================================================================

def analyze_results(results: SearchResults) -> Dict[str, Any]:
    """
    Analyze hyperparameter search results.

    Args:
        results: SearchResults object

    Returns:
        Dictionary with analysis metrics
    """
    if not results.trials:
        return {"error": "No trials to analyze"}

    scores = [t.score for t in results.trials if t.status == "completed"]

    if not scores:
        return {"error": "No completed trials"}

    analysis = {
        "n_trials": len(results.trials),
        "n_completed": results.n_completed,
        "n_failed": results.n_failed,
        "best_score": max(scores),
        "worst_score": min(scores),
        "mean_score": np.mean(scores),
        "std_score": np.std(scores),
        "median_score": np.median(scores),
    }

    # Parameter importance (correlation with score)
    if results.n_completed >= 5:
        param_importance = {}
        for param_name in results.trials[0].params.keys():
            values = []
            scores_for_corr = []
            for t in results.trials:
                if t.status == "completed":
                    val = t.params.get(param_name)
                    if isinstance(val, (int, float)):
                        values.append(val)
                        scores_for_corr.append(t.score)

            if len(values) >= 3:
                try:
                    corr = np.corrcoef(values, scores_for_corr)[0, 1]
                    param_importance[param_name] = abs(corr) if not np.isnan(corr) else 0.0
                except Exception:
                    param_importance[param_name] = 0.0

        analysis["parameter_importance"] = dict(
            sorted(param_importance.items(), key=lambda x: x[1], reverse=True)
        )

    return analysis


def plot_optimization_history(results: SearchResults, save_path: Optional[str] = None):
    """
    Plot the optimization history.

    Args:
        results: SearchResults object
        save_path: Path to save the plot (optional)
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError:
        logger.warning("matplotlib not available for plotting")
        return

    scores = [t.score for t in results.trials if t.status == "completed"]
    best_so_far = []
    current_best = float("-inf")
    for s in scores:
        if s > current_best:
            current_best = s
        best_so_far.append(current_best)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

    # Score history
    ax1.plot(scores, "b.", alpha=0.5, label="Trial scores")
    ax1.plot(best_so_far, "r-", linewidth=2, label="Best so far")
    ax1.set_xlabel("Trial")
    ax1.set_ylabel("Score")
    ax1.set_title("Optimization History")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Score distribution
    ax2.hist(scores, bins=20, edgecolor="black", alpha=0.7)
    ax2.axvline(max(scores), color="r", linestyle="--", label=f"Best: {max(scores):.4f}")
    ax2.set_xlabel("Score")
    ax2.set_ylabel("Count")
    ax2.set_title("Score Distribution")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches="tight")
        logger.info(f"Plot saved to {save_path}")
    else:
        plt.show()

    plt.close()


# =============================================================================
# CLI Interface
# =============================================================================

def main():
    """Command-line interface for hyperparameter search."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Hyperparameter search for Gemma3-1B GRPO fine-tuning"
    )
    parser.add_argument(
        "--strategy",
        choices=["grid", "random", "bayesian"],
        default="random",
        help="Search strategy to use",
    )
    parser.add_argument(
        "--n-trials",
        type=int,
        default=50,
        help="Number of trials (for random/bayesian)",
    )
    parser.add_argument(
        "--n-grid-points",
        type=int,
        default=3,
        help="Points per parameter (for grid search)",
    )
    parser.add_argument(
        "--search-space",
        choices=["lora", "training", "grpo", "generation", "full"],
        default="full",
        help="Predefined search space to use",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="./hyperparam_results",
        help="Output directory for results",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print search space and exit without running",
    )

    args = parser.parse_args()

    # Get search space
    space_map = {
        "lora": get_lora_search_space,
        "training": get_training_search_space,
        "grpo": get_grpo_search_space,
        "generation": get_generation_search_space,
        "full": get_full_search_space,
    }
    search_space = space_map[args.search_space]()

    if args.dry_run:
        print(f"\nSearch Space: {args.search_space}")
        print(f"Strategy: {args.strategy}")
        print(f"Parameters ({len(search_space)}):")
        for name, spec in search_space.parameters.items():
            if spec.param_type == "continuous":
                scale = "log" if spec.log_scale else "linear"
                print(f"  {name}: [{spec.low}, {spec.high}] ({scale})")
            elif spec.param_type == "integer":
                print(f"  {name}: [{spec.low}, {spec.high}] (int)")
            else:
                print(f"  {name}: {spec.choices}")

        if args.strategy == "grid":
            strategy = GridSearchStrategy(search_space, args.n_grid_points)
            print(f"\nTotal grid combinations: {strategy.total_combinations()}")
        else:
            print(f"\nTrials to run: {args.n_trials}")
        return

    # Create a dummy objective for demonstration
    def dummy_objective(params: Dict[str, Any]) -> Tuple[float, Dict[str, float]]:
        """Dummy objective function for testing."""
        # Simulate some computation
        time.sleep(0.1)

        # Create a score based on params (for demo purposes)
        score = 0.5
        if "learning_rate" in params:
            # Prefer learning rates around 1e-5
            lr = params["learning_rate"]
            score += 0.2 * np.exp(-((np.log10(lr) + 5) ** 2))

        if "lora_rank" in params:
            # Prefer rank 64
            score += 0.1 * (1 - abs(params["lora_rank"] - 64) / 64)

        # Add some noise
        score += np.random.normal(0, 0.05)

        metrics = {
            "accuracy": score * 100,
            "format_accuracy": score * 95,
        }

        return score, metrics

    print("\n" + "=" * 60)
    print("HYPERPARAMETER SEARCH (Demo Mode)")
    print("=" * 60)
    print("Note: Using dummy objective function for demonstration.")
    print("To use real training, provide your own objective function.\n")

    # Run search
    searcher = HyperparameterSearch(
        search_space=search_space,
        objective_fn=dummy_objective,
        strategy=args.strategy,
        n_trials=args.n_trials,
        n_grid_points=args.n_grid_points,
        seed=args.seed,
        output_dir=args.output_dir,
    )

    best_params, best_score = searcher.search()

    # Analyze and plot results
    analysis = analyze_results(searcher.results)
    print("\nAnalysis:")
    for k, v in analysis.items():
        if k != "parameter_importance":
            print(f"  {k}: {v}")

    if "parameter_importance" in analysis:
        print("\nParameter Importance (correlation with score):")
        for param, importance in analysis["parameter_importance"].items():
            print(f"  {param}: {importance:.4f}")


if __name__ == "__main__":
    main()
