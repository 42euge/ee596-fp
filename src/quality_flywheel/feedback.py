"""
Feedback Storage Module

SQLite-based storage system for researcher annotations and feedback
on problematic transcripts.
"""

import sqlite3
import json
from dataclasses import dataclass, asdict
from typing import List, Dict, Optional, Tuple
from datetime import datetime
from enum import Enum
from pathlib import Path


class AnnotationType(Enum):
    """Types of annotations researchers can make"""

    # Issue confirmation
    CONFIRMED_ISSUE = "confirmed_issue"
    FALSE_POSITIVE = "false_positive"

    # Correction types
    CORRECTED_ANSWER = "corrected_answer"
    CORRECTED_REASONING = "corrected_reasoning"
    CORRECTED_FORMAT = "corrected_format"

    # Quality assessment
    QUALITY_RATING = "quality_rating"
    USABILITY_RATING = "usability_rating"

    # Categorization
    CATEGORIZED = "categorized"
    TAGGED = "tagged"

    # Action items
    NEEDS_RETRAINING = "needs_retraining"
    NEEDS_PROMPT_UPDATE = "needs_prompt_update"
    EDGE_CASE = "edge_case"

    # General
    COMMENT = "comment"


class ResolutionStatus(Enum):
    """Status of issue resolution"""

    OPEN = "open"
    IN_PROGRESS = "in_progress"
    RESOLVED = "resolved"
    WONT_FIX = "wont_fix"
    DUPLICATE = "duplicate"


@dataclass
class Annotation:
    """Represents a researcher annotation on a transcript"""

    id: Optional[int] = None
    transcript_id: str = ""
    annotation_type: AnnotationType = AnnotationType.COMMENT
    researcher_id: str = ""
    timestamp: str = ""

    # Annotation content
    content: str = ""
    rating: Optional[float] = None  # For quality/usability ratings (0-1 scale)
    tags: List[str] = None
    metadata: Dict = None

    # Resolution tracking
    resolution_status: ResolutionStatus = ResolutionStatus.OPEN
    resolution_notes: str = ""
    resolved_at: Optional[str] = None
    resolved_by: Optional[str] = None

    def __post_init__(self):
        if not self.timestamp:
            self.timestamp = datetime.now().isoformat()
        if self.tags is None:
            self.tags = []
        if self.metadata is None:
            self.metadata = {}

    def to_dict(self) -> Dict:
        """Convert to dictionary"""
        data = asdict(self)
        data['annotation_type'] = self.annotation_type.value
        data['resolution_status'] = self.resolution_status.value
        return data


class FeedbackStore:
    """
    SQLite-based storage for transcript quality feedback and annotations.

    Stores:
    - Researcher annotations
    - Quality assessments
    - Issue resolutions
    - Improvement suggestions
    """

    def __init__(self, db_path: str = "./data/quality_feedback.db"):
        """
        Initialize feedback store.

        Args:
            db_path: Path to SQLite database file
        """
        self.db_path = db_path

        # Ensure parent directory exists
        Path(db_path).parent.mkdir(parents=True, exist_ok=True)

        self.conn = sqlite3.connect(db_path, check_same_thread=False)
        self.conn.row_factory = sqlite3.Row
        self._create_tables()

    def _create_tables(self):
        """Create database tables if they don't exist"""
        cursor = self.conn.cursor()

        # Annotations table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS annotations (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transcript_id TEXT NOT NULL,
                annotation_type TEXT NOT NULL,
                researcher_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                content TEXT,
                rating REAL,
                tags TEXT,
                metadata TEXT,
                resolution_status TEXT DEFAULT 'open',
                resolution_notes TEXT,
                resolved_at TEXT,
                resolved_by TEXT,
                FOREIGN KEY (transcript_id) REFERENCES transcripts(id)
            )
        """)

        # Transcripts table (for tracking which transcripts have been reviewed)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS transcripts (
                id TEXT PRIMARY KEY,
                question TEXT,
                response TEXT,
                expected_answer TEXT,
                quality_score REAL,
                severity TEXT,
                flagged_at TEXT,
                reviewed_at TEXT,
                reviewed_by TEXT,
                review_status TEXT DEFAULT 'pending',
                metadata TEXT
            )
        """)

        # Quality history table (for tracking quality over time)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS quality_history (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transcript_id TEXT NOT NULL,
                timestamp TEXT NOT NULL,
                quality_score REAL,
                format_score REAL,
                answer_correct INTEGER,
                reasoning_completeness REAL,
                overall_score REAL,
                severity TEXT,
                issues TEXT,
                metadata TEXT,
                FOREIGN KEY (transcript_id) REFERENCES transcripts(id)
            )
        """)

        # Improvement suggestions table
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS improvement_suggestions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                transcript_id TEXT,
                suggestion_type TEXT NOT NULL,
                description TEXT NOT NULL,
                priority INTEGER DEFAULT 3,
                suggested_by TEXT,
                suggested_at TEXT NOT NULL,
                status TEXT DEFAULT 'pending',
                implemented_at TEXT,
                implemented_by TEXT,
                notes TEXT,
                FOREIGN KEY (transcript_id) REFERENCES transcripts(id)
            )
        """)

        # Create indexes for faster queries
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_annotations_transcript ON annotations(transcript_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_annotations_type ON annotations(annotation_type)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_annotations_researcher ON annotations(researcher_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_severity ON transcripts(severity)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_transcripts_status ON transcripts(review_status)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_quality_history_transcript ON quality_history(transcript_id)")
        cursor.execute("CREATE INDEX IF NOT EXISTS idx_suggestions_status ON improvement_suggestions(status)")

        self.conn.commit()

    # Annotation operations

    def add_annotation(self, annotation: Annotation) -> int:
        """
        Add a new annotation.

        Args:
            annotation: Annotation object

        Returns:
            ID of the created annotation
        """
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO annotations (
                transcript_id, annotation_type, researcher_id, timestamp,
                content, rating, tags, metadata,
                resolution_status, resolution_notes, resolved_at, resolved_by
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            annotation.transcript_id,
            annotation.annotation_type.value,
            annotation.researcher_id,
            annotation.timestamp,
            annotation.content,
            annotation.rating,
            json.dumps(annotation.tags),
            json.dumps(annotation.metadata),
            annotation.resolution_status.value,
            annotation.resolution_notes,
            annotation.resolved_at,
            annotation.resolved_by,
        ))

        self.conn.commit()
        return cursor.lastrowid

    def get_annotation(self, annotation_id: int) -> Optional[Annotation]:
        """Get annotation by ID"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM annotations WHERE id = ?", (annotation_id,))
        row = cursor.fetchone()

        if row:
            return self._row_to_annotation(row)
        return None

    def get_annotations_for_transcript(
        self,
        transcript_id: str,
        annotation_type: Optional[AnnotationType] = None,
    ) -> List[Annotation]:
        """
        Get all annotations for a transcript.

        Args:
            transcript_id: Transcript ID
            annotation_type: Optional filter by annotation type

        Returns:
            List of Annotation objects
        """
        cursor = self.conn.cursor()

        if annotation_type:
            cursor.execute(
                "SELECT * FROM annotations WHERE transcript_id = ? AND annotation_type = ? ORDER BY timestamp DESC",
                (transcript_id, annotation_type.value)
            )
        else:
            cursor.execute(
                "SELECT * FROM annotations WHERE transcript_id = ? ORDER BY timestamp DESC",
                (transcript_id,)
            )

        rows = cursor.fetchall()
        return [self._row_to_annotation(row) for row in rows]

    def update_annotation(self, annotation: Annotation):
        """Update an existing annotation"""
        cursor = self.conn.cursor()

        cursor.execute("""
            UPDATE annotations SET
                content = ?,
                rating = ?,
                tags = ?,
                metadata = ?,
                resolution_status = ?,
                resolution_notes = ?,
                resolved_at = ?,
                resolved_by = ?
            WHERE id = ?
        """, (
            annotation.content,
            annotation.rating,
            json.dumps(annotation.tags),
            json.dumps(annotation.metadata),
            annotation.resolution_status.value,
            annotation.resolution_notes,
            annotation.resolved_at,
            annotation.resolved_by,
            annotation.id,
        ))

        self.conn.commit()

    def delete_annotation(self, annotation_id: int):
        """Delete an annotation"""
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM annotations WHERE id = ?", (annotation_id,))
        self.conn.commit()

    # Transcript operations

    def add_transcript(
        self,
        transcript_id: str,
        question: str,
        response: str,
        expected_answer: Optional[str] = None,
        quality_score: float = 0.0,
        severity: str = "none",
        metadata: Optional[Dict] = None,
    ):
        """Add a transcript to the feedback database"""
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT OR REPLACE INTO transcripts (
                id, question, response, expected_answer,
                quality_score, severity, flagged_at, review_status, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            transcript_id,
            question,
            response,
            expected_answer,
            quality_score,
            severity,
            datetime.now().isoformat(),
            'pending',
            json.dumps(metadata or {}),
        ))

        self.conn.commit()

    def get_transcript(self, transcript_id: str) -> Optional[Dict]:
        """Get transcript information"""
        cursor = self.conn.cursor()
        cursor.execute("SELECT * FROM transcripts WHERE id = ?", (transcript_id,))
        row = cursor.fetchone()

        if row:
            return dict(row)
        return None

    def mark_transcript_reviewed(
        self,
        transcript_id: str,
        reviewer_id: str,
        status: str = "reviewed",
    ):
        """Mark a transcript as reviewed"""
        cursor = self.conn.cursor()

        cursor.execute("""
            UPDATE transcripts SET
                reviewed_at = ?,
                reviewed_by = ?,
                review_status = ?
            WHERE id = ?
        """, (
            datetime.now().isoformat(),
            reviewer_id,
            status,
            transcript_id,
        ))

        self.conn.commit()

    def get_pending_transcripts(
        self,
        limit: int = 100,
        severity_filter: Optional[str] = None,
    ) -> List[Dict]:
        """
        Get transcripts pending review.

        Args:
            limit: Maximum number of transcripts to return
            severity_filter: Optional filter by severity level

        Returns:
            List of transcript dictionaries
        """
        cursor = self.conn.cursor()

        if severity_filter:
            cursor.execute("""
                SELECT * FROM transcripts
                WHERE review_status = 'pending' AND severity = ?
                ORDER BY flagged_at ASC
                LIMIT ?
            """, (severity_filter, limit))
        else:
            cursor.execute("""
                SELECT * FROM transcripts
                WHERE review_status = 'pending'
                ORDER BY severity DESC, flagged_at ASC
                LIMIT ?
            """, (limit,))

        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    # Quality history operations

    def add_quality_record(
        self,
        transcript_id: str,
        quality_score: float,
        format_score: float,
        answer_correct: bool,
        reasoning_completeness: float,
        overall_score: float,
        severity: str,
        issues: List[str],
        metadata: Optional[Dict] = None,
    ):
        """Add a quality record to history"""
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO quality_history (
                transcript_id, timestamp, quality_score, format_score,
                answer_correct, reasoning_completeness, overall_score,
                severity, issues, metadata
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
        """, (
            transcript_id,
            datetime.now().isoformat(),
            quality_score,
            format_score,
            1 if answer_correct else 0,
            reasoning_completeness,
            overall_score,
            severity,
            json.dumps(issues),
            json.dumps(metadata or {}),
        ))

        self.conn.commit()

    def get_quality_history(
        self,
        transcript_id: Optional[str] = None,
        start_date: Optional[str] = None,
        end_date: Optional[str] = None,
    ) -> List[Dict]:
        """
        Get quality history records.

        Args:
            transcript_id: Optional filter by transcript ID
            start_date: Optional start date (ISO format)
            end_date: Optional end date (ISO format)

        Returns:
            List of quality history records
        """
        cursor = self.conn.cursor()

        query = "SELECT * FROM quality_history WHERE 1=1"
        params = []

        if transcript_id:
            query += " AND transcript_id = ?"
            params.append(transcript_id)

        if start_date:
            query += " AND timestamp >= ?"
            params.append(start_date)

        if end_date:
            query += " AND timestamp <= ?"
            params.append(end_date)

        query += " ORDER BY timestamp DESC"

        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    # Improvement suggestions

    def add_improvement_suggestion(
        self,
        suggestion_type: str,
        description: str,
        transcript_id: Optional[str] = None,
        priority: int = 3,
        suggested_by: str = "system",
    ) -> int:
        """Add an improvement suggestion"""
        cursor = self.conn.cursor()

        cursor.execute("""
            INSERT INTO improvement_suggestions (
                transcript_id, suggestion_type, description,
                priority, suggested_by, suggested_at, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?)
        """, (
            transcript_id,
            suggestion_type,
            description,
            priority,
            suggested_by,
            datetime.now().isoformat(),
            'pending',
        ))

        self.conn.commit()
        return cursor.lastrowid

    def get_improvement_suggestions(
        self,
        status: Optional[str] = None,
        suggestion_type: Optional[str] = None,
        limit: int = 100,
    ) -> List[Dict]:
        """Get improvement suggestions"""
        cursor = self.conn.cursor()

        query = "SELECT * FROM improvement_suggestions WHERE 1=1"
        params = []

        if status:
            query += " AND status = ?"
            params.append(status)

        if suggestion_type:
            query += " AND suggestion_type = ?"
            params.append(suggestion_type)

        query += " ORDER BY priority ASC, suggested_at DESC LIMIT ?"
        params.append(limit)

        cursor.execute(query, params)
        rows = cursor.fetchall()
        return [dict(row) for row in rows]

    def mark_suggestion_implemented(
        self,
        suggestion_id: int,
        implemented_by: str,
        notes: Optional[str] = None,
    ):
        """Mark an improvement suggestion as implemented"""
        cursor = self.conn.cursor()

        cursor.execute("""
            UPDATE improvement_suggestions SET
                status = 'implemented',
                implemented_at = ?,
                implemented_by = ?,
                notes = ?
            WHERE id = ?
        """, (
            datetime.now().isoformat(),
            implemented_by,
            notes,
            suggestion_id,
        ))

        self.conn.commit()

    # Statistics and analytics

    def get_statistics(self) -> Dict:
        """Get overall statistics from the feedback store"""
        cursor = self.conn.cursor()

        stats = {}

        # Total transcripts
        cursor.execute("SELECT COUNT(*) as count FROM transcripts")
        stats['total_transcripts'] = cursor.fetchone()['count']

        # Pending reviews
        cursor.execute("SELECT COUNT(*) as count FROM transcripts WHERE review_status = 'pending'")
        stats['pending_reviews'] = cursor.fetchone()['count']

        # Reviewed transcripts
        cursor.execute("SELECT COUNT(*) as count FROM transcripts WHERE review_status != 'pending'")
        stats['reviewed_transcripts'] = cursor.fetchone()['count']

        # Total annotations
        cursor.execute("SELECT COUNT(*) as count FROM annotations")
        stats['total_annotations'] = cursor.fetchone()['count']

        # Annotations by type
        cursor.execute("""
            SELECT annotation_type, COUNT(*) as count
            FROM annotations
            GROUP BY annotation_type
        """)
        stats['annotations_by_type'] = {row['annotation_type']: row['count'] for row in cursor.fetchall()}

        # Transcripts by severity
        cursor.execute("""
            SELECT severity, COUNT(*) as count
            FROM transcripts
            GROUP BY severity
        """)
        stats['transcripts_by_severity'] = {row['severity']: row['count'] for row in cursor.fetchall()}

        # Average quality score
        cursor.execute("SELECT AVG(quality_score) as avg_score FROM transcripts")
        stats['average_quality_score'] = cursor.fetchone()['avg_score'] or 0

        # Improvement suggestions
        cursor.execute("SELECT COUNT(*) as count FROM improvement_suggestions WHERE status = 'pending'")
        stats['pending_suggestions'] = cursor.fetchone()['count']

        cursor.execute("SELECT COUNT(*) as count FROM improvement_suggestions WHERE status = 'implemented'")
        stats['implemented_suggestions'] = cursor.fetchone()['count']

        return stats

    # Helper methods

    def _row_to_annotation(self, row) -> Annotation:
        """Convert database row to Annotation object"""
        return Annotation(
            id=row['id'],
            transcript_id=row['transcript_id'],
            annotation_type=AnnotationType(row['annotation_type']),
            researcher_id=row['researcher_id'],
            timestamp=row['timestamp'],
            content=row['content'],
            rating=row['rating'],
            tags=json.loads(row['tags']) if row['tags'] else [],
            metadata=json.loads(row['metadata']) if row['metadata'] else {},
            resolution_status=ResolutionStatus(row['resolution_status']),
            resolution_notes=row['resolution_notes'],
            resolved_at=row['resolved_at'],
            resolved_by=row['resolved_by'],
        )

    def close(self):
        """Close database connection"""
        self.conn.close()

    def __enter__(self):
        """Context manager entry"""
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit"""
        self.close()
