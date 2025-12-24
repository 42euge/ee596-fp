"""Programmatic rubric modification utilities.

Provides tools for creating and modifying rubrics programmatically,
including a fluent builder API.
"""

from copy import deepcopy
from typing import List, Callable
import uuid

from .models import Rubric, RubricSet, Criterion


class RubricBuilder:
    """Fluent builder for creating rubrics programmatically.

    Example:
        rubric = (
            RubricBuilder("math_eval", "Evaluates math solutions")
            .with_criterion("accuracy", "Correct answer", weight=2.0)
            .with_criterion("reasoning", "Clear steps shown")
            .with_question_types("math", "algebra")
            .with_reference("<reasoning>...</reasoning><answer>42</answer>")
            .build()
        )
    """

    def __init__(self, name: str, description: str):
        """Initialize builder with required rubric fields.

        Args:
            name: Name of the rubric
            description: Description of what this rubric evaluates
        """
        self._rubric = Rubric(name=name, description=description)

    def with_id(self, id: str) -> "RubricBuilder":
        """Set the rubric ID (auto-generated if not set)."""
        self._rubric.id = id
        return self

    def with_criterion(
        self,
        name: str,
        description: str,
        weight: float = 1.0,
        keywords: List[str] | None = None,
        score_range: tuple[float, float] = (0.0, 10.0),
        examples: dict[int, str] | None = None,
    ) -> "RubricBuilder":
        """Add a criterion to the rubric.

        Args:
            name: Criterion name/identifier
            description: What this criterion evaluates
            weight: Relative weight for scoring
            keywords: Keywords that indicate criterion is satisfied
            score_range: Min/max score for this criterion
            examples: Example responses by score level

        Returns:
            Self for chaining
        """
        self._rubric.add_criterion(
            Criterion(
                name=name,
                description=description,
                weight=weight,
                keywords=keywords or [],
                score_range=score_range,
                examples=examples or {},
            )
        )
        return self

    def with_question_types(self, *types: str) -> "RubricBuilder":
        """Set applicable question types.

        Args:
            *types: Question type strings (e.g., "math", "reasoning")

        Returns:
            Self for chaining
        """
        self._rubric.question_types = list(types)
        return self

    def with_reference(
        self,
        response: str,
        target_score: float | None = None,
    ) -> "RubricBuilder":
        """Set reference response and target score.

        Args:
            response: Ideal/reference response
            target_score: Expected score for this response

        Returns:
            Self for chaining
        """
        self._rubric.reference_response = response
        self._rubric.target_score = target_score
        return self

    def with_metadata(self, **kwargs) -> "RubricBuilder":
        """Add metadata key-value pairs.

        Args:
            **kwargs: Metadata key-value pairs

        Returns:
            Self for chaining
        """
        self._rubric.metadata.update(kwargs)
        return self

    def build(self) -> Rubric:
        """Build and return the rubric.

        Returns:
            The constructed Rubric object
        """
        return self._rubric


def clone_rubric(rubric: Rubric, new_name: str | None = None) -> Rubric:
    """Create a deep copy of a rubric for modification.

    Args:
        rubric: Rubric to clone
        new_name: Optional new name for the clone

    Returns:
        Deep copy of the rubric
    """
    cloned = deepcopy(rubric)
    if new_name:
        cloned.name = new_name
        cloned.id = str(uuid.uuid4())[:8]
    return cloned


def merge_rubrics(rubrics: List[Rubric], name: str, description: str = "") -> Rubric:
    """Merge multiple rubrics into one, combining criteria.

    Args:
        rubrics: List of rubrics to merge
        name: Name for the merged rubric
        description: Optional description (auto-generated if not provided)

    Returns:
        New rubric with combined criteria
    """
    if not description:
        description = f"Merged from: {', '.join(r.name for r in rubrics)}"

    merged = Rubric(
        name=name,
        description=description,
        metadata={"merged_from": [r.id for r in rubrics]},
    )

    seen_criteria = set()
    for rubric in rubrics:
        for criterion in rubric.criteria:
            if criterion.name not in seen_criteria:
                merged.add_criterion(deepcopy(criterion))
                seen_criteria.add(criterion.name)
        merged.question_types.extend(rubric.question_types)

    merged.question_types = list(set(merged.question_types))
    return merged


def apply_transform(
    rubricset: RubricSet,
    transform: Callable[[Rubric], Rubric],
) -> RubricSet:
    """Apply a transformation to all rubrics in a set.

    Args:
        rubricset: RubricSet to transform
        transform: Function that takes a Rubric and returns a modified Rubric

    Returns:
        New RubricSet with transformed rubrics
    """
    return RubricSet(
        name=rubricset.name,
        rubrics=[transform(r) for r in rubricset.rubrics],
        description=rubricset.description,
        metadata=rubricset.metadata,
    )


def adjust_weights(rubric: Rubric, adjustments: dict[str, float]) -> Rubric:
    """Adjust criterion weights by name.

    Args:
        rubric: Rubric to modify (creates a copy)
        adjustments: Dict of criterion_name -> new_weight

    Returns:
        Modified rubric copy
    """
    modified = clone_rubric(rubric)
    for name, weight in adjustments.items():
        modified.update_criterion(name, weight=weight)
    return modified


def add_keywords_to_criterion(
    rubric: Rubric,
    criterion_name: str,
    new_keywords: List[str],
) -> Rubric:
    """Add keywords to a specific criterion.

    Args:
        rubric: Rubric to modify (creates a copy)
        criterion_name: Name of criterion to update
        new_keywords: Keywords to add

    Returns:
        Modified rubric copy
    """
    modified = clone_rubric(rubric)
    for c in modified.criteria:
        if c.name == criterion_name:
            c.keywords = list(set(c.keywords + new_keywords))
            break
    return modified


def scale_weights(rubric: Rubric, factor: float) -> Rubric:
    """Scale all criterion weights by a factor.

    Args:
        rubric: Rubric to modify (creates a copy)
        factor: Multiplication factor for weights

    Returns:
        Modified rubric copy
    """
    modified = clone_rubric(rubric)
    for c in modified.criteria:
        c.weight *= factor
    return modified


def normalize_weights(rubric: Rubric, target_sum: float = 1.0) -> Rubric:
    """Normalize criterion weights to sum to a target value.

    Args:
        rubric: Rubric to modify (creates a copy)
        target_sum: Target sum for all weights (default: 1.0)

    Returns:
        Modified rubric copy with normalized weights
    """
    modified = clone_rubric(rubric)
    total = sum(c.weight for c in modified.criteria)
    if total > 0:
        factor = target_sum / total
        for c in modified.criteria:
            c.weight *= factor
    return modified
