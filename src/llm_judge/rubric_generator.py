"""Rubric generation using LLMs.

Generates evaluation rubrics for questions/tasks using configurable
LLM backends. Supports caching and question type detection.
"""

import hashlib
from dataclasses import dataclass, field
from typing import Optional, List, Dict, Any, Tuple

from .config import RubricGeneratorConfig, GenerationConfig
from .backends import get_backend, LLMBackend
from .prompts import get_rubric_generation_prompt
from .parsing import parse_rubric
from .cache import RubricCache


@dataclass
class Rubric:
    """Structured rubric for evaluation.

    Contains criteria with names, descriptions, weights, and scoring levels.
    """

    question_hash: str
    question_type: str
    criteria: List[Dict[str, Any]]
    score_range: Tuple[int, int]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_text(self) -> str:
        """Convert rubric to text format for prompts.

        Returns:
            Human-readable rubric text
        """
        lines = []
        for i, criterion in enumerate(self.criteria, 1):
            weight = criterion.get("weight", 1.0)
            lines.append(f"{i}. {criterion['name']} (weight: {weight:.2f})")
            if criterion.get("description"):
                lines.append(f"   {criterion['description']}")
            if "levels" in criterion:
                for level in criterion["levels"]:
                    lines.append(f"   - {level['score']}: {level['description']}")
        return "\n".join(lines)

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "question_hash": self.question_hash,
            "question_type": self.question_type,
            "criteria": self.criteria,
            "score_range": list(self.score_range),
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "Rubric":
        """Deserialize from dictionary.

        Args:
            data: Dictionary with rubric data

        Returns:
            Rubric instance
        """
        score_range = data.get("score_range", [0, 10])
        if isinstance(score_range, list):
            score_range = tuple(score_range)
        return cls(
            question_hash=data["question_hash"],
            question_type=data["question_type"],
            criteria=data.get("criteria", []),
            score_range=score_range,
            metadata=data.get("metadata", {}),
        )

    @property
    def max_score(self) -> int:
        """Get maximum score from score range."""
        return self.score_range[1]

    @property
    def min_score(self) -> int:
        """Get minimum score from score range."""
        return self.score_range[0]

    def __repr__(self) -> str:
        return (
            f"Rubric(type={self.question_type}, "
            f"criteria={len(self.criteria)}, "
            f"range={self.score_range})"
        )


class RubricGenerator:
    """Generates evaluation rubrics for questions using LLMs.

    Supports multiple backends (HuggingFace, OpenAI, Anthropic, vLLM)
    with caching and question type detection.
    """

    def __init__(self, config: Optional[RubricGeneratorConfig] = None):
        """Initialize the generator.

        Args:
            config: Configuration for rubric generation
        """
        self.config = config or RubricGeneratorConfig()
        self._backend: Optional[LLMBackend] = None
        self._cache: Optional[RubricCache] = None

    def _get_backend(self) -> LLMBackend:
        """Get or create the LLM backend.

        Returns:
            Initialized LLMBackend
        """
        if self._backend is None:
            self._backend = get_backend(
                self.config.backend,
                model_id=self.config.model_id,
                **self.config.backend_config,
            )
        return self._backend

    def _get_cache(self) -> Optional[RubricCache]:
        """Get or create the cache.

        Returns:
            RubricCache if caching is enabled, None otherwise
        """
        if self._cache is None and self.config.cache_enabled:
            self._cache = RubricCache(self.config.cache_dir)
        return self._cache

    def _hash_question(self, question: str) -> str:
        """Create a stable hash for a question.

        Args:
            question: The question text

        Returns:
            16-character hex hash
        """
        normalized = question.strip().lower()
        return hashlib.sha256(normalized.encode()).hexdigest()[:16]

    def _detect_question_type(self, question: str) -> str:
        """Detect the type of question.

        Reuses the detect_question_type function from src/utils.py if available,
        otherwise uses a simple keyword-based detection.

        Args:
            question: The question text

        Returns:
            Question type string
        """
        try:
            from ..utils import detect_question_type

            return detect_question_type(question)
        except ImportError:
            pass

        # Fallback: simple keyword detection
        question_lower = question.lower()

        keywords = {
            "math": [
                "calculate",
                "compute",
                "solve",
                "equation",
                "how many",
                "how much",
                "sum",
                "difference",
                "multiply",
                "divide",
            ],
            "coding": [
                "code",
                "program",
                "function",
                "implement",
                "algorithm",
                "write a",
                "debug",
                "fix the",
            ],
            "creative": [
                "write",
                "story",
                "creative",
                "imagine",
                "describe",
                "narrative",
            ],
            "science": [
                "explain",
                "why does",
                "how does",
                "scientific",
                "experiment",
                "hypothesis",
            ],
            "summarization": [
                "summarize",
                "summary",
                "main points",
                "key ideas",
                "tldr",
            ],
        }

        for qtype, kw_list in keywords.items():
            if any(kw in question_lower for kw in kw_list):
                return qtype

        return "default"

    def generate(
        self,
        question: str,
        question_type: Optional[str] = None,
        context: Optional[str] = None,
        force_regenerate: bool = False,
    ) -> Rubric:
        """Generate a rubric for a question.

        Args:
            question: The question/task to create a rubric for
            question_type: Type of question ("math", "creative", etc.)
                          If None and auto_detect_question_type is True, will be detected
            context: Optional additional context
            force_regenerate: If True, bypass cache

        Returns:
            Generated Rubric object
        """
        question_hash = self._hash_question(question)

        # Check cache first
        if not force_regenerate and self.config.cache_enabled:
            cache = self._get_cache()
            if cache:
                cached = cache.get(question_hash)
                if cached:
                    return cached

        # Detect question type if needed
        if question_type is None:
            if self.config.auto_detect_question_type:
                question_type = self._detect_question_type(question)
            else:
                question_type = self.config.default_question_type

        # Generate rubric
        prompt = get_rubric_generation_prompt(
            question=question,
            question_type=question_type,
            context=context,
            num_criteria=self.config.num_criteria,
            score_range=self.config.score_range,
            include_examples=self.config.include_examples,
        )

        backend = self._get_backend()
        gen_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )
        response = backend.generate(prompt, gen_config)

        # Parse response into structured rubric
        rubric = parse_rubric(
            response.text,
            question_hash=question_hash,
            question_type=question_type,
            score_range=self.config.score_range,
        )

        # Cache the rubric
        if self.config.cache_enabled:
            cache = self._get_cache()
            if cache:
                cache.set(question_hash, rubric)

        return rubric

    def generate_batch(
        self,
        questions: List[str],
        question_types: Optional[List[str]] = None,
        **kwargs,
    ) -> List[Rubric]:
        """Generate rubrics for multiple questions.

        Args:
            questions: List of questions
            question_types: Optional list of question types
            **kwargs: Additional arguments passed to generate()

        Returns:
            List of Rubric objects
        """
        if question_types is None:
            question_types = [None] * len(questions)

        rubrics = []
        for question, qtype in zip(questions, question_types):
            rubric = self.generate(question, question_type=qtype, **kwargs)
            rubrics.append(rubric)

        return rubrics

    def clear_cache(self) -> int:
        """Clear the rubric cache.

        Returns:
            Number of items cleared
        """
        cache = self._get_cache()
        if cache:
            return cache.clear()
        return 0

    def cache_stats(self) -> Optional[dict]:
        """Get cache statistics.

        Returns:
            Cache stats dict or None if caching disabled
        """
        cache = self._get_cache()
        if cache:
            return cache.stats()
        return None
