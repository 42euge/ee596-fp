"""
Data Quality Flywheel System

A comprehensive system for identifying problematic reasoning transcripts,
collecting researcher feedback, and feeding improvements back into the training pipeline.

Components:
- metrics: Quality metric calculation and tracking
- detector: Automated detection of problematic transcripts
- feedback: Annotation and feedback storage
- analytics: Quality trend analysis and visualization
- improvement: Feedback loop integration with training
- review_ui: Researcher interface for reviewing flagged transcripts
"""

from .metrics import QualityMetrics, TranscriptQuality
from .detector import IssueDetector, IssueType
from .feedback import FeedbackStore, Annotation, AnnotationType
from .analytics import QualityAnalytics, QualityDashboard
from .improvement import ImprovementLoop

__all__ = [
    'QualityMetrics',
    'TranscriptQuality',
    'IssueDetector',
    'IssueType',
    'FeedbackStore',
    'Annotation',
    'AnnotationType',
    'QualityAnalytics',
    'QualityDashboard',
    'ImprovementLoop',
]

__version__ = '1.0.0'
