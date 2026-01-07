"""
Step-Aware Rubric Generation for PRIME RL

Extends the LLM Judge module to generate and apply rubrics for
individual reasoning steps, enabling fine-grained process supervision.
"""

from typing import List, Dict, Optional, Any
from dataclasses import dataclass

from .rubric_generator import RubricGenerator
from .response_scorer import ResponseScorer
from ..rubrics.models import Rubric, Criterion


@dataclass
class StepRubric:
    """Rubric for evaluating a single reasoning step."""

    step_index: int
    step_type: str  # "calculation", "inference", "conclusion", etc.
    criteria: List[Criterion]
    max_score: float = 10.0

    def to_rubric(self) -> Rubric:
        """Convert to standard Rubric object."""
        return Rubric(
            criteria=self.criteria,
            target_score=self.max_score,
            question_types=[self.step_type]
        )


class StepRubricGenerator:
    """
    Generator for step-specific rubrics.

    Creates evaluation criteria tailored to individual reasoning steps
    based on step type, position, and context.
    """

    def __init__(self, backend=None, cache_dir: Optional[str] = None):
        """
        Initialize step rubric generator.

        Args:
            backend: LLM backend for generation (OpenAI, Anthropic, etc.)
            cache_dir: Optional cache directory for generated rubrics
        """
        self.rubric_generator = RubricGenerator(backend=backend, cache_dir=cache_dir)
        self.backend = backend

    def generate_step_rubric(
        self,
        question: str,
        step_text: str,
        step_index: int,
        step_type: Optional[str] = None,
        previous_steps: Optional[List[str]] = None,
        **kwargs
    ) -> StepRubric:
        """
        Generate rubric for a single reasoning step.

        Args:
            question: Original question/problem
            step_text: Text of the current step
            step_index: Position of step in sequence
            step_type: Type of step (calculation, inference, etc.)
            previous_steps: Optional list of previous steps for context
            **kwargs: Additional context

        Returns:
            StepRubric with evaluation criteria
        """
        # Infer step type if not provided
        if step_type is None:
            step_type = self._infer_step_type(step_text)

        # Build context-aware prompt
        prompt = self._build_step_rubric_prompt(
            question=question,
            step_text=step_text,
            step_index=step_index,
            step_type=step_type,
            previous_steps=previous_steps,
            **kwargs
        )

        # Generate rubric using base generator
        rubric = self.rubric_generator.generate(
            question=question,
            reference_response=step_text,
            additional_context=prompt
        )

        # Convert to StepRubric
        step_rubric = StepRubric(
            step_index=step_index,
            step_type=step_type,
            criteria=rubric.criteria,
            max_score=rubric.target_score
        )

        return step_rubric

    def generate_trajectory_rubrics(
        self,
        question: str,
        steps: List[str],
        step_types: Optional[List[str]] = None,
        **kwargs
    ) -> List[StepRubric]:
        """
        Generate rubrics for all steps in a reasoning trajectory.

        Args:
            question: Original question
            steps: List of reasoning steps
            step_types: Optional list of step types
            **kwargs: Additional context

        Returns:
            List of StepRubric objects
        """
        if step_types is None:
            step_types = [self._infer_step_type(step) for step in steps]

        rubrics = []
        previous_steps = []

        for i, (step, step_type) in enumerate(zip(steps, step_types)):
            rubric = self.generate_step_rubric(
                question=question,
                step_text=step,
                step_index=i,
                step_type=step_type,
                previous_steps=previous_steps.copy(),
                **kwargs
            )
            rubrics.append(rubric)
            previous_steps.append(step)

        return rubrics

    def _infer_step_type(self, step_text: str) -> str:
        """
        Infer the type of reasoning step from its text.

        Returns one of:
        - "calculation": Mathematical calculation
        - "inference": Logical inference
        - "definition": Variable or term definition
        - "formula_application": Applying a formula
        - "conclusion": Final conclusion
        - "explanation": Explanatory step
        """
        import re

        step_lower = step_text.lower()

        # Check for calculations
        if re.search(r"=\s*[\d\.\+\-\*/\(\)]", step_text):
            return "calculation"

        # Check for formulas
        if re.search(r"(using|apply|substitute)", step_lower):
            return "formula_application"

        # Check for definitions
        if re.search(r"(let|define|assume|denote)", step_lower):
            return "definition"

        # Check for conclusions
        if re.search(r"(therefore|thus|hence|so|consequently)", step_lower):
            return "conclusion"

        # Check for explanations
        if re.search(r"(because|since|this means|which shows)", step_lower):
            return "explanation"

        # Default to inference
        return "inference"

    def _build_step_rubric_prompt(
        self,
        question: str,
        step_text: str,
        step_index: int,
        step_type: str,
        previous_steps: Optional[List[str]] = None,
        **kwargs
    ) -> str:
        """Build prompt for step-specific rubric generation."""
        previous_steps = previous_steps or []

        prompt_parts = [
            f"This is step {step_index + 1} in a multi-step reasoning process.",
            f"Step type: {step_type}",
            "",
            "Previous steps context:"
        ]

        if previous_steps:
            for i, prev_step in enumerate(previous_steps[-3:]):  # Last 3 steps
                prompt_parts.append(f"  Step {i + 1}: {prev_step[:100]}...")
        else:
            prompt_parts.append("  (This is the first step)")

        prompt_parts.extend([
            "",
            f"Current step: {step_text}",
            "",
            "Generate evaluation criteria specific to this type of step.",
            f"For a {step_type} step, focus on:",
        ])

        # Add type-specific guidance
        type_guidance = {
            "calculation": [
                "- Mathematical correctness",
                "- Proper use of operations",
                "- Arithmetic accuracy",
            ],
            "inference": [
                "- Logical validity",
                "- Connection to previous steps",
                "- Soundness of reasoning",
            ],
            "definition": [
                "- Clarity of definition",
                "- Consistency with problem",
                "- Proper variable usage",
            ],
            "formula_application": [
                "- Correct formula selection",
                "- Proper substitution",
                "- Valid application",
            ],
            "conclusion": [
                "- Logical follow-through",
                "- Completeness",
                "- Accuracy of conclusion",
            ],
            "explanation": [
                "- Clarity of explanation",
                "- Relevance to problem",
                "- Logical coherence",
            ],
        }

        guidance = type_guidance.get(step_type, ["- Quality of reasoning"])
        prompt_parts.extend(guidance)

        return "\n".join(prompt_parts)


class StepRubricScorer:
    """
    Scorer for evaluating reasoning steps against step-specific rubrics.
    """

    def __init__(self, backend=None):
        """
        Initialize step rubric scorer.

        Args:
            backend: LLM backend for scoring
        """
        self.scorer = ResponseScorer(backend=backend)
        self.backend = backend

    def score_step(
        self,
        step_text: str,
        step_rubric: StepRubric,
        question: Optional[str] = None,
        previous_steps: Optional[List[str]] = None,
        **kwargs
    ) -> Dict[str, Any]:
        """
        Score a reasoning step against its rubric.

        Args:
            step_text: Text of the step to score
            step_rubric: Rubric for this step
            question: Optional original question
            previous_steps: Optional previous steps for context
            **kwargs: Additional context

        Returns:
            Dictionary with score and details
        """
        # Build context
        context = {
            "question": question,
            "step_index": step_rubric.step_index,
            "step_type": step_rubric.step_type,
            "previous_steps": previous_steps or [],
            **kwargs
        }

        # Score using base scorer
        result = self.scorer.score(
            response=step_text,
            rubric=step_rubric.to_rubric(),
            question=question,
            mode="rubric_based",
            **context
        )

        return result

    def score_trajectory(
        self,
        steps: List[str],
        step_rubrics: List[StepRubric],
        question: Optional[str] = None,
        **kwargs
    ) -> List[Dict[str, Any]]:
        """
        Score all steps in a trajectory against their rubrics.

        Args:
            steps: List of reasoning steps
            step_rubrics: List of rubrics (one per step)
            question: Optional original question
            **kwargs: Additional context

        Returns:
            List of score dictionaries
        """
        assert len(steps) == len(step_rubrics), \
            "Must have same number of steps and rubrics"

        scores = []
        previous_steps = []

        for step, rubric in zip(steps, step_rubrics):
            score = self.score_step(
                step_text=step,
                step_rubric=rubric,
                question=question,
                previous_steps=previous_steps.copy(),
                **kwargs
            )
            scores.append(score)
            previous_steps.append(step)

        return scores


def create_step_evaluation_pipeline(
    backend=None,
    cache_dir: Optional[str] = None
) -> tuple[StepRubricGenerator, StepRubricScorer]:
    """
    Create a complete step evaluation pipeline.

    Args:
        backend: LLM backend
        cache_dir: Optional cache directory

    Returns:
        Tuple of (generator, scorer)
    """
    generator = StepRubricGenerator(backend=backend, cache_dir=cache_dir)
    scorer = StepRubricScorer(backend=backend)

    return generator, scorer
