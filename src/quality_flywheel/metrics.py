"""
Quality Metrics Module

Comprehensive quality scoring system that aggregates existing metrics
and adds new quality dimensions for reasoning transcripts.
"""

import re
import json
from dataclasses import dataclass, asdict
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import numpy as np


@dataclass
class TranscriptQuality:
    """Complete quality assessment for a reasoning transcript"""

    # Identifiers
    transcript_id: str
    question: str
    response: str
    expected_answer: Optional[str] = None
    timestamp: str = None

    # Format Quality (0-1 scale)
    has_reasoning_tags: bool = False
    has_answer_tags: bool = False
    format_score: float = 0.0

    # Answer Quality (0-1 scale)
    answer_correct: bool = False
    answer_extracted: Optional[str] = None
    answer_similarity: float = 0.0

    # Reasoning Quality (0-1 scale)
    reasoning_length: int = 0
    reasoning_completeness: float = 0.0
    reasoning_coherence: float = 0.0
    has_step_markers: bool = False
    num_steps: int = 0

    # Content Quality (0-1 scale)
    contains_numbers: bool = False
    contains_calculations: bool = False
    logical_flow_score: float = 0.0

    # Overall Quality
    overall_score: float = 0.0
    confidence: float = 0.0

    # Issue Flags
    issues: List[str] = None
    severity: str = "none"  # none, low, medium, high, critical

    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now().isoformat()
        if self.issues is None:
            self.issues = []

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return asdict(self)

    def to_json(self) -> str:
        """Convert to JSON string"""
        return json.dumps(self.to_dict(), indent=2)


class QualityMetrics:
    """
    Comprehensive quality metrics calculator for reasoning transcripts.

    Integrates with existing reward functions and adds new quality dimensions.
    """

    # Regex patterns for format validation
    REASONING_PATTERN = r'<reasoning>(.*?)</reasoning>'
    ANSWER_PATTERN = r'<answer>(.*?)</answer>'
    NUMBER_PATTERN = r'-?\d+(?:\.\d+)?(?:,\d{3})*'
    CALCULATION_PATTERN = r'[\+\-\*/=]'
    STEP_PATTERNS = [
        r'step\s*\d+',
        r'first[,\s]',
        r'second[,\s]',
        r'then[,\s]',
        r'next[,\s]',
        r'finally[,\s]',
    ]

    def __init__(self):
        """Initialize quality metrics calculator"""
        self.calculation_keywords = [
            'multiply', 'divide', 'add', 'subtract',
            'sum', 'total', 'difference', 'product',
            'quotient', 'calculate', 'compute'
        ]

    def evaluate(
        self,
        transcript_id: str,
        question: str,
        response: str,
        expected_answer: Optional[str] = None,
        rubric: Optional[str] = None,
    ) -> TranscriptQuality:
        """
        Comprehensive quality evaluation of a reasoning transcript.

        Args:
            transcript_id: Unique identifier for the transcript
            question: Input question/problem
            response: Model-generated response
            expected_answer: Ground truth answer (optional)
            rubric: Quality rubric for evaluation (optional)

        Returns:
            TranscriptQuality object with all metrics
        """
        quality = TranscriptQuality(
            transcript_id=transcript_id,
            question=question,
            response=response,
            expected_answer=expected_answer,
        )

        # Evaluate format quality
        self._evaluate_format(quality)

        # Evaluate answer quality
        if expected_answer:
            self._evaluate_answer(quality)

        # Evaluate reasoning quality
        self._evaluate_reasoning(quality)

        # Evaluate content quality
        self._evaluate_content(quality)

        # Calculate overall score and detect issues
        self._calculate_overall_score(quality)
        self._detect_issues(quality)

        return quality

    def _evaluate_format(self, quality: TranscriptQuality):
        """Evaluate format compliance"""
        response = quality.response

        # Check for required tags
        quality.has_reasoning_tags = bool(re.search(self.REASONING_PATTERN, response, re.DOTALL))
        quality.has_answer_tags = bool(re.search(self.ANSWER_PATTERN, response, re.DOTALL))

        # Calculate format score
        if quality.has_reasoning_tags and quality.has_answer_tags:
            quality.format_score = 1.0
        elif quality.has_reasoning_tags or quality.has_answer_tags:
            quality.format_score = 0.5
        else:
            quality.format_score = 0.0

    def _evaluate_answer(self, quality: TranscriptQuality):
        """Evaluate answer correctness"""
        response = quality.response
        expected = quality.expected_answer

        # Extract answer from tags
        answer_match = re.search(self.ANSWER_PATTERN, response, re.DOTALL)
        if answer_match:
            quality.answer_extracted = answer_match.group(1).strip()
        else:
            quality.answer_extracted = None
            quality.answer_correct = False
            quality.answer_similarity = 0.0
            return

        # Compare with expected answer
        extracted = quality.answer_extracted

        # Extract numbers from both
        expected_nums = self._extract_numbers(expected)
        extracted_nums = self._extract_numbers(extracted)

        # Check exact match
        if extracted.strip() == expected.strip():
            quality.answer_correct = True
            quality.answer_similarity = 1.0
        elif expected_nums and extracted_nums:
            # Compare numerical values
            try:
                expected_val = float(expected_nums[0].replace(',', ''))
                extracted_val = float(extracted_nums[0].replace(',', ''))

                ratio = extracted_val / expected_val if expected_val != 0 else 0

                if 0.99 <= ratio <= 1.01:
                    quality.answer_correct = True
                    quality.answer_similarity = 1.0
                elif 0.9 <= ratio <= 1.1:
                    quality.answer_similarity = 0.8
                elif 0.8 <= ratio <= 1.2:
                    quality.answer_similarity = 0.5
                else:
                    quality.answer_similarity = 0.2
            except (ValueError, ZeroDivisionError):
                quality.answer_similarity = 0.0
        else:
            quality.answer_similarity = 0.0

    def _evaluate_reasoning(self, quality: TranscriptQuality):
        """Evaluate reasoning quality"""
        response = quality.response

        # Extract reasoning section
        reasoning_match = re.search(self.REASONING_PATTERN, response, re.DOTALL)
        if not reasoning_match:
            quality.reasoning_length = 0
            quality.reasoning_completeness = 0.0
            quality.reasoning_coherence = 0.0
            quality.has_step_markers = False
            quality.num_steps = 0
            return

        reasoning = reasoning_match.group(1).strip()
        quality.reasoning_length = len(reasoning)

        # Check for step markers
        num_steps = 0
        for pattern in self.STEP_PATTERNS:
            matches = re.findall(pattern, reasoning.lower())
            num_steps += len(matches)

        quality.num_steps = num_steps
        quality.has_step_markers = num_steps > 0

        # Estimate completeness based on length and structure
        if quality.reasoning_length < 50:
            quality.reasoning_completeness = 0.3
        elif quality.reasoning_length < 150:
            quality.reasoning_completeness = 0.6
        else:
            quality.reasoning_completeness = 0.9

        # Boost completeness if has steps
        if quality.has_step_markers:
            quality.reasoning_completeness = min(1.0, quality.reasoning_completeness + 0.2)

        # Estimate coherence based on sentence structure
        sentences = reasoning.split('.')
        quality.reasoning_coherence = min(1.0, len([s for s in sentences if len(s.strip()) > 10]) / 5)

    def _evaluate_content(self, quality: TranscriptQuality):
        """Evaluate content quality"""
        response = quality.response

        # Check for numbers
        quality.contains_numbers = bool(re.search(self.NUMBER_PATTERN, response))

        # Check for calculations
        quality.contains_calculations = bool(re.search(self.CALCULATION_PATTERN, response))
        if not quality.contains_calculations:
            # Check for calculation keywords
            response_lower = response.lower()
            quality.contains_calculations = any(
                keyword in response_lower for keyword in self.calculation_keywords
            )

        # Estimate logical flow
        reasoning_match = re.search(self.REASONING_PATTERN, response, re.DOTALL)
        if reasoning_match:
            reasoning = reasoning_match.group(1).lower()

            # Check for logical connectors
            connectors = ['therefore', 'thus', 'so', 'because', 'since', 'then', 'hence']
            connector_count = sum(1 for c in connectors if c in reasoning)

            quality.logical_flow_score = min(1.0, connector_count / 3)
        else:
            quality.logical_flow_score = 0.0

    def _calculate_overall_score(self, quality: TranscriptQuality):
        """Calculate overall quality score with weighted components"""

        # Weights for different quality dimensions
        weights = {
            'format': 0.2,
            'answer': 0.3,
            'reasoning_completeness': 0.2,
            'reasoning_coherence': 0.15,
            'content': 0.15,
        }

        # Calculate weighted score
        score = 0.0
        score += weights['format'] * quality.format_score

        if quality.expected_answer:
            score += weights['answer'] * (1.0 if quality.answer_correct else quality.answer_similarity)
        else:
            # Redistribute answer weight if no expected answer
            weights['reasoning_completeness'] += weights['answer'] * 0.5
            weights['content'] += weights['answer'] * 0.5

        score += weights['reasoning_completeness'] * quality.reasoning_completeness
        score += weights['reasoning_coherence'] * quality.reasoning_coherence

        content_score = (
            quality.logical_flow_score * 0.5 +
            (0.25 if quality.contains_numbers else 0.0) +
            (0.25 if quality.contains_calculations else 0.0)
        )
        score += weights['content'] * content_score

        quality.overall_score = score

        # Calculate confidence based on available information
        confidence_factors = []
        confidence_factors.append(1.0 if quality.has_reasoning_tags else 0.5)
        confidence_factors.append(1.0 if quality.has_answer_tags else 0.5)
        if quality.expected_answer:
            confidence_factors.append(1.0)

        quality.confidence = np.mean(confidence_factors)

    def _detect_issues(self, quality: TranscriptQuality):
        """Detect and flag quality issues"""
        issues = []

        # Critical issues
        if not quality.has_reasoning_tags:
            issues.append("missing_reasoning_tags")
        if not quality.has_answer_tags:
            issues.append("missing_answer_tags")
        if quality.expected_answer and not quality.answer_correct:
            issues.append("incorrect_answer")

        # High severity issues
        if quality.reasoning_length < 50:
            issues.append("insufficient_reasoning")
        if not quality.contains_numbers and quality.expected_answer:
            issues.append("missing_numerical_content")

        # Medium severity issues
        if quality.reasoning_coherence < 0.3:
            issues.append("low_coherence")
        if not quality.has_step_markers and quality.reasoning_length > 100:
            issues.append("missing_step_markers")
        if quality.logical_flow_score < 0.3:
            issues.append("weak_logical_flow")

        # Low severity issues
        if not quality.contains_calculations and quality.expected_answer:
            issues.append("missing_calculation_keywords")

        quality.issues = issues

        # Determine severity
        if any(i in ["missing_reasoning_tags", "missing_answer_tags"] for i in issues):
            quality.severity = "critical"
        elif "incorrect_answer" in issues:
            quality.severity = "high"
        elif len(issues) >= 3:
            quality.severity = "medium"
        elif len(issues) > 0:
            quality.severity = "low"
        else:
            quality.severity = "none"

    def _extract_numbers(self, text: str) -> List[str]:
        """Extract all numbers from text"""
        return re.findall(self.NUMBER_PATTERN, text)

    def batch_evaluate(
        self,
        transcripts: List[Dict],
        id_field: str = 'id',
        question_field: str = 'question',
        response_field: str = 'response',
        answer_field: str = 'answer',
    ) -> List[TranscriptQuality]:
        """
        Evaluate a batch of transcripts.

        Args:
            transcripts: List of transcript dictionaries
            id_field: Field name for transcript ID
            question_field: Field name for question
            response_field: Field name for response
            answer_field: Field name for expected answer

        Returns:
            List of TranscriptQuality objects
        """
        results = []
        for i, transcript in enumerate(transcripts):
            transcript_id = transcript.get(id_field, f"transcript_{i}")
            question = transcript.get(question_field, "")
            response = transcript.get(response_field, "")
            expected_answer = transcript.get(answer_field, None)

            quality = self.evaluate(
                transcript_id=transcript_id,
                question=question,
                response=response,
                expected_answer=expected_answer,
            )
            results.append(quality)

        return results
