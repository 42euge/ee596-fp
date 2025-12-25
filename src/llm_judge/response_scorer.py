"""Response scoring using LLM-as-judge.

Scores responses against rubrics or reference answers using
configurable LLM backends.
"""

from dataclasses import dataclass
from typing import Optional, List, Dict, Any

from .config import ResponseScorerConfig, JudgeMode, GenerationConfig
from .backends import get_backend, LLMBackend
from .prompts import get_scoring_prompt, get_reference_scoring_prompt
from .parsing import parse_score
from .rubric_generator import Rubric


@dataclass
class ScoreResult:
    """Result of scoring a response.

    Contains the score, optional reasoning, and per-criterion scores.
    """

    score: float
    max_score: float
    normalized_score: float  # 0.0 to 1.0
    reasoning: Optional[str] = None
    criterion_scores: Optional[Dict[str, float]] = None
    raw_output: Optional[str] = None

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary.

        Returns:
            Dictionary representation
        """
        return {
            "score": self.score,
            "max_score": self.max_score,
            "normalized_score": self.normalized_score,
            "reasoning": self.reasoning,
            "criterion_scores": self.criterion_scores,
        }

    def __repr__(self) -> str:
        return (
            f"ScoreResult(score={self.score}/{self.max_score}, "
            f"normalized={self.normalized_score:.2f})"
        )


class ResponseScorer:
    """Scores responses using LLM-as-judge.

    Supports multiple scoring modes:
    - RUBRIC_BASED: Score against a provided rubric
    - REFERENCE_BASED: Score against a reference answer
    - COMPARATIVE: Compare two responses (not yet implemented)
    """

    def __init__(self, config: Optional[ResponseScorerConfig] = None):
        """Initialize the scorer.

        Args:
            config: Configuration for response scoring
        """
        self.config = config or ResponseScorerConfig()
        self._backend: Optional[LLMBackend] = None

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

    def score(
        self,
        question: str,
        response: str,
        rubric: Optional[Rubric] = None,
        rubric_text: Optional[str] = None,
        reference_answer: Optional[str] = None,
    ) -> ScoreResult:
        """Score a single response.

        Args:
            question: The original question
            response: The response to score
            rubric: Rubric object (for RUBRIC_BASED mode)
            rubric_text: Raw rubric text (alternative to Rubric object)
            reference_answer: Reference answer (for REFERENCE_BASED mode)

        Returns:
            ScoreResult with score and optional reasoning

        Raises:
            ValueError: If required parameters for the mode are missing
        """
        # Build scoring prompt based on mode
        if self.config.judge_mode == JudgeMode.RUBRIC_BASED:
            if rubric is not None:
                rubric_str = rubric.to_text()
                max_score = rubric.max_score
            elif rubric_text is not None:
                rubric_str = rubric_text
                max_score = 10  # Default
            else:
                raise ValueError(
                    "rubric or rubric_text required for RUBRIC_BASED mode"
                )

            prompt = get_scoring_prompt(
                question=question,
                response=response,
                rubric=rubric_str,
                output_format=self.config.output_format,
                include_reasoning=self.config.include_reasoning,
                max_score=max_score,
            )

        elif self.config.judge_mode == JudgeMode.REFERENCE_BASED:
            if reference_answer is None:
                raise ValueError(
                    "reference_answer required for REFERENCE_BASED mode"
                )
            prompt = get_reference_scoring_prompt(
                question=question,
                response=response,
                reference=reference_answer,
                output_format=self.config.output_format,
            )
            max_score = 10

        elif self.config.judge_mode == JudgeMode.COMPARATIVE:
            raise NotImplementedError(
                "COMPARATIVE mode not yet implemented. "
                "Use RUBRIC_BASED or REFERENCE_BASED mode."
            )

        else:
            raise ValueError(f"Unknown judge mode: {self.config.judge_mode}")

        # Generate score
        backend = self._get_backend()
        gen_config = GenerationConfig(
            max_new_tokens=self.config.max_new_tokens,
            temperature=self.config.temperature,
        )

        # Retry logic for robustness
        last_error = None
        for attempt in range(self.config.max_retries):
            try:
                llm_response = backend.generate(prompt, gen_config)

                # Parse score from response
                score, reasoning, criterion_scores = parse_score(
                    llm_response.text,
                    output_format=self.config.output_format,
                    max_score=max_score,
                )

                return ScoreResult(
                    score=score,
                    max_score=max_score,
                    normalized_score=score / max_score if max_score > 0 else 0.0,
                    reasoning=reasoning if self.config.include_reasoning else None,
                    criterion_scores=criterion_scores,
                    raw_output=llm_response.text,
                )

            except Exception as e:
                last_error = e
                if attempt < self.config.max_retries - 1:
                    import time

                    time.sleep(self.config.retry_delay * (attempt + 1))

        # All retries failed
        raise RuntimeError(
            f"Failed to score response after {self.config.max_retries} attempts: "
            f"{last_error}"
        )

    def score_batch(
        self,
        questions: List[str],
        responses: List[str],
        rubrics: Optional[List[Rubric]] = None,
        rubric_texts: Optional[List[str]] = None,
        reference_answers: Optional[List[str]] = None,
        **kwargs,
    ) -> List[ScoreResult]:
        """Score multiple responses.

        Args:
            questions: List of questions
            responses: List of responses to score
            rubrics: Optional list of Rubric objects
            rubric_texts: Optional list of rubric texts
            reference_answers: Optional list of reference answers
            **kwargs: Additional arguments passed to score()

        Returns:
            List of ScoreResult objects
        """
        n = len(questions)
        if len(responses) != n:
            raise ValueError(
                f"Length mismatch: {n} questions vs {len(responses)} responses"
            )

        # Prepare optional lists
        if rubrics is None:
            rubrics = [None] * n
        if rubric_texts is None:
            rubric_texts = [None] * n
        if reference_answers is None:
            reference_answers = [None] * n

        results = []
        for i in range(n):
            try:
                result = self.score(
                    question=questions[i],
                    response=responses[i],
                    rubric=rubrics[i],
                    rubric_text=rubric_texts[i],
                    reference_answer=reference_answers[i],
                    **kwargs,
                )
            except Exception as e:
                # Return zero score on error
                result = ScoreResult(
                    score=0.0,
                    max_score=10.0,
                    normalized_score=0.0,
                    reasoning=f"Error: {str(e)}",
                    raw_output=None,
                )
            results.append(result)

        return results

    def score_with_generated_rubric(
        self,
        question: str,
        response: str,
        rubric_generator: "RubricGenerator",
        question_type: Optional[str] = None,
    ) -> ScoreResult:
        """Score a response, generating a rubric first if needed.

        Convenience method that combines rubric generation and scoring.

        Args:
            question: The original question
            response: The response to score
            rubric_generator: RubricGenerator instance to use
            question_type: Optional question type for rubric generation

        Returns:
            ScoreResult with score and reasoning
        """
        from .rubric_generator import RubricGenerator

        # Generate rubric
        rubric = rubric_generator.generate(
            question=question,
            question_type=question_type,
        )

        # Score against rubric
        return self.score(
            question=question,
            response=response,
            rubric=rubric,
        )
