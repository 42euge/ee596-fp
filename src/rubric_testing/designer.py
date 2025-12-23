"""
Rubric Designer - Base classes and tools for creating custom rubric scoring functions

This module provides a flexible framework for defining custom rubric designs that can be
quickly tested and iterated upon.
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Tuple
import numpy as np


@dataclass
class RubricScore:
    """Container for rubric scoring results"""
    total: float
    components: Dict[str, float] = field(default_factory=dict)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __repr__(self) -> str:
        comp_str = ", ".join(f"{k}={v:.2f}" for k, v in self.components.items())
        return f"RubricScore(total={self.total:.2f}, {comp_str})"


class BaseRubric(ABC):
    """
    Base class for all rubric designs.

    Subclass this to create custom rubric scoring functions.
    """

    def __init__(self, name: str, weight: float = 1.0):
        self.name = name
        self.weight = weight
        self._score_range = (0.0, 10.0)  # Default range

    @abstractmethod
    def score(
        self,
        prompt: str,
        completion: str,
        rubric: str,
        reference_response: Optional[str] = None,
        target_score: Optional[float] = None,
        **kwargs
    ) -> RubricScore:
        """
        Score a completion according to this rubric design.

        Args:
            prompt: The input prompt/question
            completion: The model's generated response
            rubric: The rubric criteria text
            reference_response: Optional reference answer
            target_score: Optional target quality score
            **kwargs: Additional context-specific parameters

        Returns:
            RubricScore with total score and component breakdown
        """
        pass

    @property
    def score_range(self) -> Tuple[float, float]:
        """Return the (min, max) score range for this rubric"""
        return self._score_range

    def normalize_score(self, score: float) -> float:
        """Normalize score to [0, 1] range"""
        min_score, max_score = self._score_range
        if max_score == min_score:
            return 0.0
        return (score - min_score) / (max_score - min_score)


class RubricDesigner:
    """
    Helper class for designing and registering custom rubric functions.

    Example:
        designer = RubricDesigner()

        @designer.register("keyword_match")
        def keyword_rubric(prompt, completion, rubric, **kwargs):
            # Custom scoring logic
            keywords = rubric.lower().split()
            matches = sum(1 for kw in keywords if kw in completion.lower())
            return RubricScore(total=matches, components={"matches": matches})
    """

    def __init__(self):
        self._registry: Dict[str, BaseRubric] = {}

    def register(self, name: str, weight: float = 1.0):
        """Decorator to register a rubric function"""
        def decorator(func: Callable) -> Callable:
            rubric = FunctionRubric(name, func, weight)
            self._registry[name] = rubric
            return func
        return decorator

    def get(self, name: str) -> Optional[BaseRubric]:
        """Get a registered rubric by name"""
        return self._registry.get(name)

    def list_rubrics(self) -> List[str]:
        """List all registered rubric names"""
        return list(self._registry.keys())


class FunctionRubric(BaseRubric):
    """Wrapper to convert a function into a rubric"""

    def __init__(self, name: str, func: Callable, weight: float = 1.0):
        super().__init__(name, weight)
        self.func = func

    def score(
        self,
        prompt: str,
        completion: str,
        rubric: str,
        reference_response: Optional[str] = None,
        target_score: Optional[float] = None,
        **kwargs
    ) -> RubricScore:
        result = self.func(
            prompt=prompt,
            completion=completion,
            rubric=rubric,
            reference_response=reference_response,
            target_score=target_score,
            **kwargs
        )

        # Convert to RubricScore if needed
        if isinstance(result, RubricScore):
            return result
        elif isinstance(result, (int, float)):
            return RubricScore(total=float(result))
        elif isinstance(result, dict):
            total = result.get("total", sum(result.values()))
            return RubricScore(total=total, components=result)
        else:
            raise ValueError(f"Rubric function returned invalid type: {type(result)}")


class CompositeRubric(BaseRubric):
    """
    Combine multiple rubrics into a composite scoring function.

    Example:
        composite = CompositeRubric("combined", [
            keyword_rubric,
            format_rubric,
            similarity_rubric
        ])
    """

    def __init__(self, name: str, rubrics: List[BaseRubric], normalize: bool = True):
        super().__init__(name)
        self.rubrics = rubrics
        self.normalize = normalize

        # Calculate total range
        if normalize:
            self._score_range = (0.0, len(rubrics))
        else:
            min_total = sum(r.score_range[0] for r in rubrics)
            max_total = sum(r.score_range[1] for r in rubrics)
            self._score_range = (min_total, max_total)

    def score(
        self,
        prompt: str,
        completion: str,
        rubric: str,
        reference_response: Optional[str] = None,
        target_score: Optional[float] = None,
        **kwargs
    ) -> RubricScore:
        total = 0.0
        components = {}
        metadata = {"rubric_scores": {}}

        for r in self.rubrics:
            result = r.score(prompt, completion, rubric, reference_response, target_score, **kwargs)

            if self.normalize:
                score = r.normalize_score(result.total)
            else:
                score = result.total

            total += score * r.weight
            components[r.name] = score
            metadata["rubric_scores"][r.name] = result

        return RubricScore(total=total, components=components, metadata=metadata)


class WeightedRubric(CompositeRubric):
    """
    Composite rubric with explicit weights for each component.

    Example:
        weighted = WeightedRubric("weighted_combo", {
            keyword_rubric: 0.3,
            format_rubric: 0.2,
            similarity_rubric: 0.5
        })
    """

    def __init__(self, name: str, rubric_weights: Dict[BaseRubric, float]):
        rubrics = list(rubric_weights.keys())
        super().__init__(name, rubrics, normalize=True)

        # Set weights
        for rubric, weight in rubric_weights.items():
            rubric.weight = weight

        # Normalize weights to sum to 1.0
        total_weight = sum(rubric_weights.values())
        if total_weight > 0:
            for rubric in rubrics:
                rubric.weight /= total_weight


def create_rubric(
    name: str,
    score_func: Optional[Callable] = None,
    components: Optional[List[BaseRubric]] = None,
    weights: Optional[Dict[BaseRubric, float]] = None,
) -> BaseRubric:
    """
    Factory function to create rubric instances.

    Args:
        name: Rubric name
        score_func: Optional scoring function (creates FunctionRubric)
        components: Optional list of sub-rubrics (creates CompositeRubric)
        weights: Optional rubric weights (creates WeightedRubric)

    Returns:
        BaseRubric instance

    Examples:
        # Simple function rubric
        rubric = create_rubric("simple", score_func=lambda **kw: 5.0)

        # Composite rubric
        rubric = create_rubric("combo", components=[rubric1, rubric2])

        # Weighted rubric
        rubric = create_rubric("weighted", weights={rubric1: 0.7, rubric2: 0.3})
    """
    if weights is not None:
        return WeightedRubric(name, weights)
    elif components is not None:
        return CompositeRubric(name, components)
    elif score_func is not None:
        return FunctionRubric(name, score_func)
    else:
        raise ValueError("Must provide score_func, components, or weights")


# Built-in rubric implementations

class KeywordMatchRubric(BaseRubric):
    """Scores based on keyword overlap with rubric criteria"""

    def __init__(self, name: str = "keyword_match", weight: float = 1.0, case_sensitive: bool = False):
        super().__init__(name, weight)
        self.case_sensitive = case_sensitive
        self._score_range = (0.0, 10.0)

    def score(
        self,
        prompt: str,
        completion: str,
        rubric: str,
        reference_response: Optional[str] = None,
        target_score: Optional[float] = None,
        **kwargs
    ) -> RubricScore:
        # Extract keywords from rubric
        rubric_text = rubric if self.case_sensitive else rubric.lower()
        completion_text = completion if self.case_sensitive else completion.lower()

        # Simple keyword extraction (split on whitespace and common punctuation)
        import re
        keywords = set(re.findall(r'\b\w+\b', rubric_text))
        keywords = {kw for kw in keywords if len(kw) > 3}  # Filter short words

        if not keywords:
            return RubricScore(total=0.0, components={"matches": 0, "total_keywords": 0})

        # Count matches
        matches = sum(1 for kw in keywords if kw in completion_text)
        match_ratio = matches / len(keywords)

        # Score from 0-10 based on match ratio
        score = match_ratio * 10.0

        return RubricScore(
            total=score,
            components={
                "matches": matches,
                "total_keywords": len(keywords),
                "match_ratio": match_ratio
            }
        )


class LengthRubric(BaseRubric):
    """Scores based on response length relative to target"""

    def __init__(
        self,
        name: str = "length",
        weight: float = 1.0,
        target_length: int = 200,
        tolerance: float = 0.5
    ):
        super().__init__(name, weight)
        self.target_length = target_length
        self.tolerance = tolerance
        self._score_range = (0.0, 10.0)

    def score(
        self,
        prompt: str,
        completion: str,
        rubric: str,
        reference_response: Optional[str] = None,
        target_score: Optional[float] = None,
        **kwargs
    ) -> RubricScore:
        length = len(completion)

        # Calculate deviation from target
        deviation = abs(length - self.target_length) / self.target_length

        # Score decreases with deviation
        if deviation <= self.tolerance:
            score = 10.0 * (1.0 - deviation / self.tolerance)
        else:
            score = 0.0

        return RubricScore(
            total=score,
            components={
                "length": length,
                "target_length": self.target_length,
                "deviation": deviation
            }
        )


class FormatComplianceRubric(BaseRubric):
    """Scores based on format compliance (reasoning/answer tags)"""

    def __init__(self, name: str = "format_compliance", weight: float = 1.0):
        super().__init__(name, weight)
        self._score_range = (0.0, 10.0)

    def score(
        self,
        prompt: str,
        completion: str,
        rubric: str,
        reference_response: Optional[str] = None,
        target_score: Optional[float] = None,
        **kwargs
    ) -> RubricScore:
        score = 0.0
        components = {}

        # Check for reasoning tags
        has_reasoning_start = "<reasoning>" in completion
        has_reasoning_end = "</reasoning>" in completion
        components["has_reasoning_tags"] = float(has_reasoning_start and has_reasoning_end)

        # Check for answer tags
        has_answer_start = "<answer>" in completion
        has_answer_end = "</answer>" in completion
        components["has_answer_tags"] = float(has_answer_start and has_answer_end)

        # Extract content lengths
        if has_reasoning_start and has_reasoning_end:
            try:
                reasoning = completion.split("<reasoning>")[1].split("</reasoning>")[0]
                components["reasoning_length"] = len(reasoning.strip())
                if len(reasoning.strip()) > 50:
                    score += 3.0
                elif len(reasoning.strip()) > 0:
                    score += 1.5
            except:
                pass

        if has_answer_start and has_answer_end:
            try:
                answer = completion.split("<answer>")[1].split("</answer>")[0]
                components["answer_length"] = len(answer.strip())
                if 0 < len(answer.strip()) < 200:
                    score += 3.0
                elif len(answer.strip()) > 0:
                    score += 1.5
            except:
                pass

        # Tag presence bonus
        if has_reasoning_start and has_reasoning_end:
            score += 2.0
        if has_answer_start and has_answer_end:
            score += 2.0

        return RubricScore(total=min(score, 10.0), components=components)
