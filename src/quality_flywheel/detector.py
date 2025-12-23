"""
Issue Detection Module

Automated detection and classification of problematic transcripts
with configurable thresholds and detection rules.
"""

from enum import Enum
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Callable
import json

from .metrics import TranscriptQuality, QualityMetrics


class IssueType(Enum):
    """Classification of transcript issues"""

    # Format Issues
    MISSING_REASONING_TAGS = "missing_reasoning_tags"
    MISSING_ANSWER_TAGS = "missing_answer_tags"
    MALFORMED_STRUCTURE = "malformed_structure"

    # Answer Issues
    INCORRECT_ANSWER = "incorrect_answer"
    MISSING_ANSWER_CONTENT = "missing_answer_content"
    ANSWER_MISMATCH = "answer_mismatch"

    # Reasoning Issues
    INSUFFICIENT_REASONING = "insufficient_reasoning"
    LOW_COHERENCE = "low_coherence"
    MISSING_STEP_MARKERS = "missing_step_markers"
    WEAK_LOGICAL_FLOW = "weak_logical_flow"

    # Content Issues
    MISSING_NUMERICAL_CONTENT = "missing_numerical_content"
    MISSING_CALCULATIONS = "missing_calculation_keywords"
    INCOMPLETE_SOLUTION = "incomplete_solution"

    # Quality Issues
    LOW_OVERALL_SCORE = "low_overall_score"
    LOW_CONFIDENCE = "low_confidence"


@dataclass
class DetectedIssue:
    """Represents a detected issue in a transcript"""

    issue_type: IssueType
    severity: str  # critical, high, medium, low
    description: str
    transcript_id: str
    confidence: float  # 0-1 scale
    auto_flagged: bool = True
    requires_review: bool = False

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['issue_type'] = self.issue_type.value
        return data


@dataclass
class DetectionResult:
    """Result of issue detection for a transcript"""

    transcript_id: str
    quality: TranscriptQuality
    detected_issues: List[DetectedIssue]
    should_flag: bool
    priority: int  # 1 (highest) to 5 (lowest)

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        return {
            'transcript_id': self.transcript_id,
            'quality': self.quality.to_dict(),
            'detected_issues': [issue.to_dict() for issue in self.detected_issues],
            'should_flag': self.should_flag,
            'priority': self.priority,
        }


class IssueDetector:
    """
    Automated issue detection system for reasoning transcripts.

    Detects problematic transcripts based on quality metrics and
    configurable detection rules.
    """

    def __init__(
        self,
        quality_threshold: float = 0.6,
        confidence_threshold: float = 0.7,
        auto_flag_critical: bool = True,
        auto_flag_incorrect_answers: bool = True,
    ):
        """
        Initialize issue detector.

        Args:
            quality_threshold: Minimum overall quality score (0-1)
            confidence_threshold: Minimum confidence for auto-flagging
            auto_flag_critical: Automatically flag critical issues
            auto_flag_incorrect_answers: Automatically flag incorrect answers
        """
        self.quality_threshold = quality_threshold
        self.confidence_threshold = confidence_threshold
        self.auto_flag_critical = auto_flag_critical
        self.auto_flag_incorrect_answers = auto_flag_incorrect_answers

        self.metrics_calculator = QualityMetrics()

        # Detection rules (issue_type -> detection function)
        self.detection_rules: Dict[IssueType, Callable] = {
            IssueType.MISSING_REASONING_TAGS: self._detect_missing_reasoning_tags,
            IssueType.MISSING_ANSWER_TAGS: self._detect_missing_answer_tags,
            IssueType.INCORRECT_ANSWER: self._detect_incorrect_answer,
            IssueType.INSUFFICIENT_REASONING: self._detect_insufficient_reasoning,
            IssueType.LOW_COHERENCE: self._detect_low_coherence,
            IssueType.MISSING_STEP_MARKERS: self._detect_missing_step_markers,
            IssueType.WEAK_LOGICAL_FLOW: self._detect_weak_logical_flow,
            IssueType.MISSING_NUMERICAL_CONTENT: self._detect_missing_numerical_content,
            IssueType.MISSING_CALCULATIONS: self._detect_missing_calculations,
            IssueType.LOW_OVERALL_SCORE: self._detect_low_overall_score,
            IssueType.LOW_CONFIDENCE: self._detect_low_confidence,
        }

    def detect(self, quality: TranscriptQuality) -> DetectionResult:
        """
        Detect issues in a transcript based on quality metrics.

        Args:
            quality: TranscriptQuality object

        Returns:
            DetectionResult with detected issues and flagging decision
        """
        detected_issues = []

        # Run all detection rules
        for issue_type, detection_func in self.detection_rules.items():
            issue = detection_func(quality)
            if issue:
                detected_issues.append(issue)

        # Determine if transcript should be flagged
        should_flag = self._should_flag_transcript(quality, detected_issues)

        # Calculate priority (1 = highest, 5 = lowest)
        priority = self._calculate_priority(quality, detected_issues)

        return DetectionResult(
            transcript_id=quality.transcript_id,
            quality=quality,
            detected_issues=detected_issues,
            should_flag=should_flag,
            priority=priority,
        )

    def detect_batch(
        self,
        transcripts: List[Dict],
        id_field: str = 'id',
        question_field: str = 'question',
        response_field: str = 'response',
        answer_field: str = 'answer',
    ) -> List[DetectionResult]:
        """
        Detect issues in a batch of transcripts.

        Args:
            transcripts: List of transcript dictionaries
            id_field: Field name for transcript ID
            question_field: Field name for question
            response_field: Field name for response
            answer_field: Field name for expected answer

        Returns:
            List of DetectionResult objects
        """
        # First, evaluate quality metrics
        quality_results = self.metrics_calculator.batch_evaluate(
            transcripts,
            id_field=id_field,
            question_field=question_field,
            response_field=response_field,
            answer_field=answer_field,
        )

        # Then detect issues
        detection_results = []
        for quality in quality_results:
            result = self.detect(quality)
            detection_results.append(result)

        return detection_results

    def _should_flag_transcript(
        self,
        quality: TranscriptQuality,
        issues: List[DetectedIssue]
    ) -> bool:
        """Determine if a transcript should be flagged for review"""

        # Auto-flag critical issues
        if self.auto_flag_critical and quality.severity == "critical":
            return True

        # Auto-flag incorrect answers
        if self.auto_flag_incorrect_answers and quality.expected_answer and not quality.answer_correct:
            return True

        # Flag if overall quality is below threshold
        if quality.overall_score < self.quality_threshold:
            return True

        # Flag if multiple high/medium severity issues
        high_medium_issues = [
            i for i in issues
            if i.severity in ["high", "medium"]
        ]
        if len(high_medium_issues) >= 2:
            return True

        return False

    def _calculate_priority(
        self,
        quality: TranscriptQuality,
        issues: List[DetectedIssue]
    ) -> int:
        """Calculate review priority (1 = highest, 5 = lowest)"""

        # Priority 1: Critical issues
        if quality.severity == "critical":
            return 1

        # Priority 2: Incorrect answers or high severity
        if quality.severity == "high" or (quality.expected_answer and not quality.answer_correct):
            return 2

        # Priority 3: Multiple medium issues or low quality
        if quality.severity == "medium" or quality.overall_score < 0.4:
            return 3

        # Priority 4: Low severity or marginal quality
        if quality.severity == "low" or quality.overall_score < 0.6:
            return 4

        # Priority 5: Minor issues or high quality
        return 5

    # Detection rule implementations

    def _detect_missing_reasoning_tags(self, quality: TranscriptQuality) -> Optional[DetectedIssue]:
        """Detect missing reasoning tags"""
        if not quality.has_reasoning_tags:
            return DetectedIssue(
                issue_type=IssueType.MISSING_REASONING_TAGS,
                severity="critical",
                description="Response is missing <reasoning> tags",
                transcript_id=quality.transcript_id,
                confidence=1.0,
                requires_review=True,
            )
        return None

    def _detect_missing_answer_tags(self, quality: TranscriptQuality) -> Optional[DetectedIssue]:
        """Detect missing answer tags"""
        if not quality.has_answer_tags:
            return DetectedIssue(
                issue_type=IssueType.MISSING_ANSWER_TAGS,
                severity="critical",
                description="Response is missing <answer> tags",
                transcript_id=quality.transcript_id,
                confidence=1.0,
                requires_review=True,
            )
        return None

    def _detect_incorrect_answer(self, quality: TranscriptQuality) -> Optional[DetectedIssue]:
        """Detect incorrect answers"""
        if quality.expected_answer and not quality.answer_correct:
            confidence = 1.0 if quality.answer_similarity < 0.5 else 0.8
            return DetectedIssue(
                issue_type=IssueType.INCORRECT_ANSWER,
                severity="high",
                description=f"Answer '{quality.answer_extracted}' does not match expected '{quality.expected_answer}'",
                transcript_id=quality.transcript_id,
                confidence=confidence,
                requires_review=True,
            )
        return None

    def _detect_insufficient_reasoning(self, quality: TranscriptQuality) -> Optional[DetectedIssue]:
        """Detect insufficient reasoning"""
        if quality.reasoning_length > 0 and quality.reasoning_length < 50:
            return DetectedIssue(
                issue_type=IssueType.INSUFFICIENT_REASONING,
                severity="high",
                description=f"Reasoning section is too short ({quality.reasoning_length} chars)",
                transcript_id=quality.transcript_id,
                confidence=0.9,
                requires_review=True,
            )
        return None

    def _detect_low_coherence(self, quality: TranscriptQuality) -> Optional[DetectedIssue]:
        """Detect low reasoning coherence"""
        if quality.reasoning_coherence < 0.3:
            return DetectedIssue(
                issue_type=IssueType.LOW_COHERENCE,
                severity="medium",
                description=f"Reasoning has low coherence score ({quality.reasoning_coherence:.2f})",
                transcript_id=quality.transcript_id,
                confidence=0.7,
                requires_review=False,
            )
        return None

    def _detect_missing_step_markers(self, quality: TranscriptQuality) -> Optional[DetectedIssue]:
        """Detect missing step markers in long reasoning"""
        if quality.reasoning_length > 100 and not quality.has_step_markers:
            return DetectedIssue(
                issue_type=IssueType.MISSING_STEP_MARKERS,
                severity="medium",
                description="Long reasoning without clear step markers",
                transcript_id=quality.transcript_id,
                confidence=0.8,
                requires_review=False,
            )
        return None

    def _detect_weak_logical_flow(self, quality: TranscriptQuality) -> Optional[DetectedIssue]:
        """Detect weak logical flow"""
        if quality.logical_flow_score < 0.3 and quality.reasoning_length > 50:
            return DetectedIssue(
                issue_type=IssueType.WEAK_LOGICAL_FLOW,
                severity="medium",
                description=f"Weak logical flow in reasoning ({quality.logical_flow_score:.2f})",
                transcript_id=quality.transcript_id,
                confidence=0.6,
                requires_review=False,
            )
        return None

    def _detect_missing_numerical_content(self, quality: TranscriptQuality) -> Optional[DetectedIssue]:
        """Detect missing numerical content in math problems"""
        if quality.expected_answer and not quality.contains_numbers:
            return DetectedIssue(
                issue_type=IssueType.MISSING_NUMERICAL_CONTENT,
                severity="high",
                description="Math problem response lacks numerical content",
                transcript_id=quality.transcript_id,
                confidence=0.9,
                requires_review=True,
            )
        return None

    def _detect_missing_calculations(self, quality: TranscriptQuality) -> Optional[DetectedIssue]:
        """Detect missing calculation indicators"""
        if quality.expected_answer and not quality.contains_calculations:
            return DetectedIssue(
                issue_type=IssueType.MISSING_CALCULATIONS,
                severity="low",
                description="Response lacks calculation keywords or operators",
                transcript_id=quality.transcript_id,
                confidence=0.5,
                requires_review=False,
            )
        return None

    def _detect_low_overall_score(self, quality: TranscriptQuality) -> Optional[DetectedIssue]:
        """Detect low overall quality score"""
        if quality.overall_score < self.quality_threshold:
            severity = "high" if quality.overall_score < 0.4 else "medium"
            return DetectedIssue(
                issue_type=IssueType.LOW_OVERALL_SCORE,
                severity=severity,
                description=f"Overall quality score is low ({quality.overall_score:.2f})",
                transcript_id=quality.transcript_id,
                confidence=quality.confidence,
                requires_review=True,
            )
        return None

    def _detect_low_confidence(self, quality: TranscriptQuality) -> Optional[DetectedIssue]:
        """Detect low confidence in quality assessment"""
        if quality.confidence < self.confidence_threshold:
            return DetectedIssue(
                issue_type=IssueType.LOW_CONFIDENCE,
                severity="low",
                description=f"Low confidence in quality assessment ({quality.confidence:.2f})",
                transcript_id=quality.transcript_id,
                confidence=quality.confidence,
                requires_review=False,
            )
        return None

    def get_flagged_transcripts(
        self,
        detection_results: List[DetectionResult],
        min_priority: int = 5,
    ) -> List[DetectionResult]:
        """
        Filter detection results to get flagged transcripts.

        Args:
            detection_results: List of DetectionResult objects
            min_priority: Minimum priority level to include (1-5, lower is higher priority)

        Returns:
            Filtered list of flagged transcripts sorted by priority
        """
        flagged = [
            result for result in detection_results
            if result.should_flag and result.priority <= min_priority
        ]

        # Sort by priority (ascending) and overall score (ascending)
        flagged.sort(key=lambda r: (r.priority, r.quality.overall_score))

        return flagged

    def get_statistics(self, detection_results: List[DetectionResult]) -> Dict:
        """
        Get statistics about detected issues.

        Args:
            detection_results: List of DetectionResult objects

        Returns:
            Dictionary with statistics
        """
        total = len(detection_results)
        flagged = len([r for r in detection_results if r.should_flag])

        # Count by severity
        severity_counts = {
            'critical': 0,
            'high': 0,
            'medium': 0,
            'low': 0,
            'none': 0,
        }
        for result in detection_results:
            severity_counts[result.quality.severity] += 1

        # Count by issue type
        issue_type_counts = {}
        for result in detection_results:
            for issue in result.detected_issues:
                issue_type = issue.issue_type.value
                issue_type_counts[issue_type] = issue_type_counts.get(issue_type, 0) + 1

        # Count by priority
        priority_counts = {i: 0 for i in range(1, 6)}
        for result in detection_results:
            if result.should_flag:
                priority_counts[result.priority] += 1

        # Quality score distribution
        scores = [r.quality.overall_score for r in detection_results]
        avg_score = sum(scores) / len(scores) if scores else 0

        return {
            'total_transcripts': total,
            'flagged_count': flagged,
            'flagged_percentage': (flagged / total * 100) if total > 0 else 0,
            'severity_counts': severity_counts,
            'issue_type_counts': issue_type_counts,
            'priority_counts': priority_counts,
            'average_quality_score': avg_score,
            'quality_score_distribution': {
                'min': min(scores) if scores else 0,
                'max': max(scores) if scores else 0,
                'median': sorted(scores)[len(scores) // 2] if scores else 0,
            }
        }
