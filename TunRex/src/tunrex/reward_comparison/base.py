"""Base classes for reward evaluation."""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Callable
import time


@dataclass
class RewardMetadata:
    """Metadata about reward computation."""

    evaluator_name: str
    evaluator_type: str  # "programmatic", "rubric", "preference_model"
    evaluation_time_ms: float
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        return (
            f"RewardMetadata(name={self.evaluator_name}, "
            f"type={self.evaluator_type}, time={self.evaluation_time_ms:.2f}ms)"
        )


@dataclass
class RewardResult:
    """Result from a reward evaluation."""

    scores: List[float]  # Reward scores for each completion
    metadata: RewardMetadata
    details: Optional[List[Dict[str, Any]]] = None  # Per-sample details

    def __post_init__(self):
        """Validate the result."""
        if self.details is not None and len(self.details) != len(self.scores):
            raise ValueError(
                f"Length mismatch: {len(self.scores)} scores vs "
                f"{len(self.details)} details"
            )

    @property
    def mean_score(self) -> float:
        """Average reward score."""
        return sum(self.scores) / len(self.scores) if self.scores else 0.0

    @property
    def std_score(self) -> float:
        """Standard deviation of reward scores."""
        if not self.scores:
            return 0.0
        mean = self.mean_score
        variance = sum((x - mean) ** 2 for x in self.scores) / len(self.scores)
        return variance ** 0.5

    def __repr__(self) -> str:
        return (
            f"RewardResult(n={len(self.scores)}, "
            f"mean={self.mean_score:.3f}, std={self.std_score:.3f}, "
            f"metadata={self.metadata})"
        )


class BaseReward(ABC):
    """Base class for all reward evaluators.

    Subclasses should implement the `evaluate` method to compute rewards
    for a batch of (prompt, completion) pairs.
    """

    def __init__(self, name: Optional[str] = None):
        """Initialize the reward evaluator.

        Args:
            name: Optional name for this evaluator. If not provided,
                  defaults to the class name.
        """
        self.name = name or self.__class__.__name__

    @abstractmethod
    def evaluate(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs
    ) -> RewardResult:
        """Evaluate rewards for a batch of completions.

        Args:
            prompts: List of prompts
            completions: List of completions (same length as prompts)
            **kwargs: Additional metadata (e.g., answers, questions)

        Returns:
            RewardResult containing scores and metadata
        """
        pass

    @abstractmethod
    def get_evaluator_type(self) -> str:
        """Return the type of evaluator.

        Returns:
            One of: "programmatic", "rubric", "preference_model"
        """
        pass

    def _create_metadata(
        self,
        evaluation_time_ms: float,
        extra_metadata: Optional[Dict[str, Any]] = None
    ) -> RewardMetadata:
        """Helper to create reward metadata.

        Args:
            evaluation_time_ms: Time taken to evaluate (milliseconds)
            extra_metadata: Optional additional metadata

        Returns:
            RewardMetadata instance
        """
        metadata_dict = extra_metadata or {}
        return RewardMetadata(
            evaluator_name=self.name,
            evaluator_type=self.get_evaluator_type(),
            evaluation_time_ms=evaluation_time_ms,
            metadata=metadata_dict,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.name})"


class RewardFunction(BaseReward):
    """Wrapper for callable reward functions.

    This is a convenience class for wrapping existing reward functions
    that follow the signature: (prompts, completions, **kwargs) -> List[float]
    """

    def __init__(
        self,
        func: Callable[[List[str], List[str], Any], List[float]],
        name: Optional[str] = None,
        evaluator_type: str = "programmatic"
    ):
        """Initialize the reward function wrapper.

        Args:
            func: Reward function to wrap
            name: Optional name (defaults to function name)
            evaluator_type: Type of evaluator
        """
        self.func = func
        self._evaluator_type = evaluator_type
        super().__init__(name=name or func.__name__)

    def evaluate(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs
    ) -> RewardResult:
        """Evaluate using the wrapped function."""
        start_time = time.time()
        scores = self.func(prompts, completions, **kwargs)
        elapsed_ms = (time.time() - start_time) * 1000

        metadata = self._create_metadata(elapsed_ms)
        return RewardResult(scores=scores, metadata=metadata)

    def get_evaluator_type(self) -> str:
        """Return the evaluator type."""
        return self._evaluator_type
