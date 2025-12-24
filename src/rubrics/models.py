"""Core dataclasses for rubric representation.

This module defines the data structures for rubrics, criteria, and rubric sets
used in GRPO training evaluation.
"""

from dataclasses import dataclass, field
from typing import Optional, List, Iterator, Callable, Any
import uuid


@dataclass
class Criterion:
    """A single evaluation criterion within a rubric.

    Attributes:
        name: Short identifier for the criterion (e.g., "clarity", "accuracy")
        description: Detailed description of what this criterion evaluates
        weight: Relative weight for scoring (default 1.0)
        score_range: Tuple of (min_score, max_score) for this criterion
        keywords: Key terms that indicate this criterion is satisfied
        examples: Optional example responses for each score level
    """

    name: str
    description: str
    weight: float = 1.0
    score_range: tuple[float, float] = (0.0, 10.0)
    keywords: List[str] = field(default_factory=list)
    examples: dict[int, str] = field(default_factory=dict)

    def to_text(self) -> str:
        """Convert criterion to text format for embedding in prompts."""
        return f"- {self.name}: {self.description}"

    def to_dict(self) -> dict[str, Any]:
        """Convert criterion to dictionary for serialization."""
        d = {
            "name": self.name,
            "description": self.description,
        }
        if self.weight != 1.0:
            d["weight"] = self.weight
        if self.score_range != (0.0, 10.0):
            d["score_range"] = list(self.score_range)
        if self.keywords:
            d["keywords"] = self.keywords
        if self.examples:
            d["examples"] = self.examples
        return d


@dataclass
class Rubric:
    """A complete rubric for evaluating model responses.

    Attributes:
        name: Human-readable name
        description: What type of responses this rubric evaluates
        criteria: List of evaluation criteria
        id: Unique identifier for the rubric (auto-generated if not provided)
        question_types: Types of questions this rubric applies to
        reference_response: Optional ideal/reference response
        target_score: Expected score for reference response
        metadata: Additional metadata (source, version, etc.)
    """

    name: str
    description: str
    criteria: List[Criterion] = field(default_factory=list)
    id: str = field(default_factory=lambda: str(uuid.uuid4())[:8])
    question_types: List[str] = field(default_factory=list)
    reference_response: Optional[str] = None
    target_score: Optional[float] = None
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        """Convert rubric to text format for use in prompts/scoring."""
        parts = [f"Rubric: {self.name}", self.description, "Criteria:"]
        for criterion in self.criteria:
            parts.append(criterion.to_text())
        return "\n".join(parts)

    def get_keywords(self) -> List[str]:
        """Extract all keywords from criteria for TF-IDF scoring."""
        keywords = []
        for criterion in self.criteria:
            keywords.extend(criterion.keywords)
        return keywords

    def add_criterion(self, criterion: Criterion) -> "Rubric":
        """Add a criterion (returns self for chaining)."""
        self.criteria.append(criterion)
        return self

    def remove_criterion(self, name: str) -> bool:
        """Remove a criterion by name. Returns True if found."""
        for i, c in enumerate(self.criteria):
            if c.name == name:
                self.criteria.pop(i)
                return True
        return False

    def update_criterion(self, name: str, **kwargs) -> bool:
        """Update a criterion's attributes by name."""
        for criterion in self.criteria:
            if criterion.name == name:
                for key, value in kwargs.items():
                    if hasattr(criterion, key):
                        setattr(criterion, key, value)
                return True
        return False

    def get_criterion(self, name: str) -> Optional[Criterion]:
        """Get a criterion by name."""
        for criterion in self.criteria:
            if criterion.name == name:
                return criterion
        return None

    def to_dict(self) -> dict[str, Any]:
        """Convert rubric to dictionary for serialization."""
        d = {
            "name": self.name,
            "description": self.description,
            "criteria": [c.to_dict() for c in self.criteria],
        }
        if self.id:
            d["id"] = self.id
        if self.question_types:
            d["question_types"] = self.question_types
        if self.reference_response:
            d["reference_response"] = self.reference_response
        if self.target_score is not None:
            d["target_score"] = self.target_score
        if self.metadata:
            d["metadata"] = self.metadata
        return d


@dataclass
class RubricSet:
    """A collection of rubrics for training/evaluation.

    Supports iteration, indexing, and question-type matching.

    Attributes:
        name: Name of the rubric set
        rubrics: List of rubrics in the set
        description: Description of the rubric set
        metadata: Additional metadata
    """

    name: str
    rubrics: List[Rubric] = field(default_factory=list)
    description: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.rubrics)

    def __iter__(self) -> Iterator[Rubric]:
        return iter(self.rubrics)

    def __getitem__(self, key: int | str) -> Rubric:
        if isinstance(key, int):
            return self.rubrics[key]
        # String lookup by id or name
        for rubric in self.rubrics:
            if rubric.id == key or rubric.name == key:
                return rubric
        raise KeyError(f"Rubric not found: {key}")

    def __contains__(self, key: str) -> bool:
        """Check if rubric exists by id or name."""
        for rubric in self.rubrics:
            if rubric.id == key or rubric.name == key:
                return True
        return False

    def add(self, rubric: Rubric) -> "RubricSet":
        """Add a rubric (returns self for chaining)."""
        self.rubrics.append(rubric)
        return self

    def remove(self, id_or_name: str) -> bool:
        """Remove a rubric by id or name."""
        for i, r in enumerate(self.rubrics):
            if r.id == id_or_name or r.name == id_or_name:
                self.rubrics.pop(i)
                return True
        return False

    def filter(self, question_type: str) -> List[Rubric]:
        """Get rubrics matching a question type."""
        return [
            r
            for r in self.rubrics
            if not r.question_types or question_type in r.question_types
        ]

    def get_for_question(
        self,
        question: str,
        type_detector: Callable[[str], str] | None = None,
    ) -> Rubric | None:
        """Find best matching rubric for a question.

        Args:
            question: The question text
            type_detector: Optional function to detect question type

        Returns:
            Matching rubric or first rubric if no match
        """
        if type_detector is not None:
            qtype = type_detector(question)
            matches = self.filter(qtype)
            if matches:
                return matches[0]

        # Fall back to first rubric
        return self.rubrics[0] if self.rubrics else None

    def to_dict(self) -> dict[str, Any]:
        """Convert rubric set to dictionary for serialization."""
        return {
            "name": self.name,
            "description": self.description,
            "rubrics": [r.to_dict() for r in self.rubrics],
            "metadata": self.metadata,
        }
