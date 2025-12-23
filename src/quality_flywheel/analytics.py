"""
Analytics and Visualization Module

Quality trend analysis, visualization, and dashboard generation
for the data quality flywheel.
"""

import json
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timedelta
from collections import defaultdict, Counter
import numpy as np

from .feedback import FeedbackStore
from .detector import DetectionResult, IssueType


class QualityAnalytics:
    """
    Analytics engine for quality metrics and trends.

    Provides insights into:
    - Quality trends over time
    - Common failure patterns
    - Improvement tracking
    - Researcher feedback patterns
    """

    def __init__(self, feedback_store: FeedbackStore):
        """
        Initialize analytics engine.

        Args:
            feedback_store: FeedbackStore instance
        """
        self.store = feedback_store

    def get_quality_trends(
        self,
        days: int = 30,
        granularity: str = 'day',
    ) -> Dict:
        """
        Get quality trends over time.

        Args:
            days: Number of days to analyze
            granularity: 'day', 'week', or 'hour'

        Returns:
            Dictionary with trend data
        """
        start_date = (datetime.now() - timedelta(days=days)).isoformat()
        history = self.store.get_quality_history(start_date=start_date)

        if not history:
            return {
                'timestamps': [],
                'overall_scores': [],
                'format_scores': [],
                'answer_accuracy': [],
                'reasoning_completeness': [],
                'severity_counts': {},
            }

        # Group by time period
        grouped = defaultdict(list)
        for record in history:
            timestamp = datetime.fromisoformat(record['timestamp'])

            if granularity == 'day':
                key = timestamp.strftime('%Y-%m-%d')
            elif granularity == 'week':
                key = timestamp.strftime('%Y-W%U')
            elif granularity == 'hour':
                key = timestamp.strftime('%Y-%m-%d %H:00')
            else:
                key = timestamp.strftime('%Y-%m-%d')

            grouped[key].append(record)

        # Calculate averages for each time period
        timestamps = sorted(grouped.keys())
        overall_scores = []
        format_scores = []
        answer_accuracy = []
        reasoning_completeness = []
        severity_counts = {s: [] for s in ['critical', 'high', 'medium', 'low', 'none']}

        for ts in timestamps:
            records = grouped[ts]

            # Calculate averages
            overall_scores.append(np.mean([r['overall_score'] for r in records]))
            format_scores.append(np.mean([r['format_score'] for r in records]))
            answer_accuracy.append(np.mean([r['answer_correct'] for r in records]))
            reasoning_completeness.append(np.mean([r['reasoning_completeness'] for r in records]))

            # Count by severity
            severity_counter = Counter(r['severity'] for r in records)
            for severity in severity_counts:
                severity_counts[severity].append(severity_counter.get(severity, 0))

        return {
            'timestamps': timestamps,
            'overall_scores': overall_scores,
            'format_scores': format_scores,
            'answer_accuracy': answer_accuracy,
            'reasoning_completeness': reasoning_completeness,
            'severity_counts': severity_counts,
            'total_records': len(history),
        }

    def get_issue_patterns(
        self,
        detection_results: List[DetectionResult],
    ) -> Dict:
        """
        Analyze common failure patterns in detected issues.

        Args:
            detection_results: List of DetectionResult objects

        Returns:
            Dictionary with pattern analysis
        """
        # Count issue types
        issue_type_counts = Counter()
        issue_type_severities = defaultdict(Counter)
        co_occurring_issues = defaultdict(Counter)

        for result in detection_results:
            issue_types = [issue.issue_type for issue in result.detected_issues]

            # Count individual issues
            for issue in result.detected_issues:
                issue_type_counts[issue.issue_type.value] += 1
                issue_type_severities[issue.issue_type.value][issue.severity] += 1

            # Track co-occurring issues
            for i, issue1 in enumerate(issue_types):
                for issue2 in issue_types[i+1:]:
                    pair = tuple(sorted([issue1.value, issue2.value]))
                    co_occurring_issues[pair[0]][pair[1]] += 1

        # Find most common patterns
        most_common_issues = issue_type_counts.most_common(10)

        # Find strongest correlations
        correlations = []
        for issue1, co_occurs in co_occurring_issues.items():
            for issue2, count in co_occurs.most_common(5):
                correlation_strength = count / issue_type_counts.get(issue1, 1)
                correlations.append({
                    'issue1': issue1,
                    'issue2': issue2,
                    'co_occurrence_count': count,
                    'correlation_strength': correlation_strength,
                })

        correlations.sort(key=lambda x: x['correlation_strength'], reverse=True)

        return {
            'issue_type_counts': dict(issue_type_counts),
            'issue_type_severities': {k: dict(v) for k, v in issue_type_severities.items()},
            'most_common_issues': most_common_issues,
            'correlations': correlations[:10],
            'total_issues': sum(issue_type_counts.values()),
        }

    def get_researcher_activity(self) -> Dict:
        """
        Analyze researcher annotation activity.

        Returns:
            Dictionary with researcher activity stats
        """
        cursor = self.store.conn.cursor()

        # Annotations per researcher
        cursor.execute("""
            SELECT researcher_id, COUNT(*) as count
            FROM annotations
            GROUP BY researcher_id
            ORDER BY count DESC
        """)
        annotations_per_researcher = {
            row['researcher_id']: row['count']
            for row in cursor.fetchall()
        }

        # Reviews per researcher
        cursor.execute("""
            SELECT reviewed_by, COUNT(*) as count
            FROM transcripts
            WHERE reviewed_by IS NOT NULL
            GROUP BY reviewed_by
            ORDER BY count DESC
        """)
        reviews_per_researcher = {
            row['reviewed_by']: row['count']
            for row in cursor.fetchall()
        }

        # Recent activity (last 7 days)
        week_ago = (datetime.now() - timedelta(days=7)).isoformat()
        cursor.execute("""
            SELECT researcher_id, COUNT(*) as count
            FROM annotations
            WHERE timestamp >= ?
            GROUP BY researcher_id
        """, (week_ago,))
        recent_activity = {
            row['researcher_id']: row['count']
            for row in cursor.fetchall()
        }

        # Annotation types distribution
        cursor.execute("""
            SELECT annotation_type, COUNT(*) as count
            FROM annotations
            GROUP BY annotation_type
        """)
        annotation_type_distribution = {
            row['annotation_type']: row['count']
            for row in cursor.fetchall()
        }

        return {
            'annotations_per_researcher': annotations_per_researcher,
            'reviews_per_researcher': reviews_per_researcher,
            'recent_activity': recent_activity,
            'annotation_type_distribution': annotation_type_distribution,
            'total_researchers': len(annotations_per_researcher),
        }

    def get_improvement_impact(self) -> Dict:
        """
        Analyze the impact of improvements over time.

        Returns:
            Dictionary with improvement impact metrics
        """
        # Get all quality history
        history = self.store.get_quality_history()

        if not history:
            return {
                'quality_improvement': 0.0,
                'format_improvement': 0.0,
                'answer_accuracy_improvement': 0.0,
                'issue_reduction': 0.0,
            }

        # Sort by timestamp
        history.sort(key=lambda x: x['timestamp'])

        # Split into before/after periods (first half vs second half)
        midpoint = len(history) // 2
        first_half = history[:midpoint]
        second_half = history[midpoint:]

        # Calculate averages for each period
        def avg_metrics(records):
            if not records:
                return {
                    'overall_score': 0,
                    'format_score': 0,
                    'answer_correct': 0,
                    'num_issues': 0,
                }

            return {
                'overall_score': np.mean([r['overall_score'] for r in records]),
                'format_score': np.mean([r['format_score'] for r in records]),
                'answer_correct': np.mean([r['answer_correct'] for r in records]),
                'num_issues': np.mean([len(json.loads(r['issues'])) for r in records]),
            }

        before = avg_metrics(first_half)
        after = avg_metrics(second_half)

        # Calculate improvements
        quality_improvement = after['overall_score'] - before['overall_score']
        format_improvement = after['format_score'] - before['format_score']
        answer_accuracy_improvement = after['answer_correct'] - before['answer_correct']
        issue_reduction = before['num_issues'] - after['num_issues']

        # Get implemented suggestions
        implemented = self.store.get_improvement_suggestions(status='implemented')

        return {
            'quality_improvement': quality_improvement,
            'format_improvement': format_improvement,
            'answer_accuracy_improvement': answer_accuracy_improvement,
            'issue_reduction': issue_reduction,
            'before_metrics': before,
            'after_metrics': after,
            'implemented_suggestions': len(implemented),
            'period_start': first_half[0]['timestamp'] if first_half else None,
            'period_mid': history[midpoint]['timestamp'] if midpoint < len(history) else None,
            'period_end': history[-1]['timestamp'] if history else None,
        }

    def get_dashboard_summary(self) -> Dict:
        """
        Get a comprehensive dashboard summary.

        Returns:
            Dictionary with all key metrics for a dashboard
        """
        stats = self.store.get_statistics()
        trends = self.get_quality_trends(days=30)
        researcher_activity = self.get_researcher_activity()
        improvement_impact = self.get_improvement_impact()

        # Calculate review velocity
        if stats['reviewed_transcripts'] > 0:
            cursor = self.store.conn.cursor()
            cursor.execute("""
                SELECT AVG(
                    CAST((julianday(reviewed_at) - julianday(flagged_at)) AS REAL)
                ) as avg_days
                FROM transcripts
                WHERE reviewed_at IS NOT NULL AND flagged_at IS NOT NULL
            """)
            avg_review_time = cursor.fetchone()['avg_days']
        else:
            avg_review_time = 0

        return {
            'summary': {
                'total_transcripts': stats['total_transcripts'],
                'pending_reviews': stats['pending_reviews'],
                'reviewed_transcripts': stats['reviewed_transcripts'],
                'average_quality_score': stats['average_quality_score'],
                'average_review_time_days': avg_review_time,
            },
            'quality_trends': {
                'latest_overall_score': trends['overall_scores'][-1] if trends['overall_scores'] else 0,
                'latest_format_score': trends['format_scores'][-1] if trends['format_scores'] else 0,
                'latest_answer_accuracy': trends['answer_accuracy'][-1] if trends['answer_accuracy'] else 0,
                'trend_direction': self._calculate_trend_direction(trends['overall_scores']),
            },
            'severity_distribution': stats['transcripts_by_severity'],
            'annotations': {
                'total': stats['total_annotations'],
                'by_type': stats['annotations_by_type'],
            },
            'researcher_activity': researcher_activity,
            'improvements': {
                'pending_suggestions': stats['pending_suggestions'],
                'implemented_suggestions': stats['implemented_suggestions'],
                'impact': improvement_impact,
            },
        }

    def _calculate_trend_direction(self, scores: List[float]) -> str:
        """Calculate if trend is improving, declining, or stable"""
        if len(scores) < 2:
            return 'insufficient_data'

        # Compare recent average to older average
        midpoint = len(scores) // 2
        old_avg = np.mean(scores[:midpoint]) if scores[:midpoint] else 0
        recent_avg = np.mean(scores[midpoint:]) if scores[midpoint:] else 0

        diff = recent_avg - old_avg

        if diff > 0.05:
            return 'improving'
        elif diff < -0.05:
            return 'declining'
        else:
            return 'stable'


class QualityDashboard:
    """
    Dashboard generator for quality metrics visualization.

    Generates text-based dashboards and reports.
    """

    def __init__(self, analytics: QualityAnalytics):
        """
        Initialize dashboard generator.

        Args:
            analytics: QualityAnalytics instance
        """
        self.analytics = analytics

    def generate_text_dashboard(self) -> str:
        """
        Generate a text-based dashboard.

        Returns:
            Formatted dashboard string
        """
        summary = self.analytics.get_dashboard_summary()

        dashboard = []
        dashboard.append("=" * 80)
        dashboard.append("DATA QUALITY FLYWHEEL - DASHBOARD")
        dashboard.append("=" * 80)
        dashboard.append("")

        # Summary section
        dashboard.append("SUMMARY")
        dashboard.append("-" * 80)
        dashboard.append(f"Total Transcripts:        {summary['summary']['total_transcripts']:,}")
        dashboard.append(f"Pending Reviews:          {summary['summary']['pending_reviews']:,}")
        dashboard.append(f"Reviewed:                 {summary['summary']['reviewed_transcripts']:,}")
        dashboard.append(f"Average Quality Score:    {summary['summary']['average_quality_score']:.2%}")
        dashboard.append(f"Avg Review Time:          {summary['summary']['average_review_time_days']:.1f} days")
        dashboard.append("")

        # Quality trends section
        dashboard.append("QUALITY TRENDS")
        dashboard.append("-" * 80)
        dashboard.append(f"Overall Score:            {summary['quality_trends']['latest_overall_score']:.2%}")
        dashboard.append(f"Format Compliance:        {summary['quality_trends']['latest_format_score']:.2%}")
        dashboard.append(f"Answer Accuracy:          {summary['quality_trends']['latest_answer_accuracy']:.2%}")
        dashboard.append(f"Trend Direction:          {summary['quality_trends']['trend_direction'].upper()}")
        dashboard.append("")

        # Severity distribution
        dashboard.append("SEVERITY DISTRIBUTION")
        dashboard.append("-" * 80)
        severity_dist = summary['severity_distribution']
        total = sum(severity_dist.values()) if severity_dist else 1
        for severity in ['critical', 'high', 'medium', 'low', 'none']:
            count = severity_dist.get(severity, 0)
            pct = (count / total * 100) if total > 0 else 0
            bar = self._make_bar(pct, width=30)
            dashboard.append(f"{severity.capitalize():12s}  {bar}  {count:4d} ({pct:5.1f}%)")
        dashboard.append("")

        # Annotations
        dashboard.append("ANNOTATIONS")
        dashboard.append("-" * 80)
        dashboard.append(f"Total Annotations:        {summary['annotations']['total']:,}")
        if summary['annotations']['by_type']:
            top_types = sorted(
                summary['annotations']['by_type'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            dashboard.append("Top Types:")
            for ann_type, count in top_types:
                dashboard.append(f"  - {ann_type:30s}  {count:4d}")
        dashboard.append("")

        # Researcher activity
        dashboard.append("RESEARCHER ACTIVITY")
        dashboard.append("-" * 80)
        activity = summary['researcher_activity']
        dashboard.append(f"Active Researchers:       {activity['total_researchers']}")
        if activity['annotations_per_researcher']:
            top_annotators = sorted(
                activity['annotations_per_researcher'].items(),
                key=lambda x: x[1],
                reverse=True
            )[:5]
            dashboard.append("Top Annotators:")
            for researcher, count in top_annotators:
                dashboard.append(f"  - {researcher:30s}  {count:4d} annotations")
        dashboard.append("")

        # Improvements
        dashboard.append("IMPROVEMENTS")
        dashboard.append("-" * 80)
        improvements = summary['improvements']
        dashboard.append(f"Pending Suggestions:      {improvements['pending_suggestions']:,}")
        dashboard.append(f"Implemented:              {improvements['implemented_suggestions']:,}")

        impact = improvements['impact']
        dashboard.append(f"\nImpact Metrics:")
        dashboard.append(f"  Quality Improvement:    {impact['quality_improvement']:+.2%}")
        dashboard.append(f"  Format Improvement:     {impact['format_improvement']:+.2%}")
        dashboard.append(f"  Accuracy Improvement:   {impact['answer_accuracy_improvement']:+.2%}")
        dashboard.append(f"  Issue Reduction:        {impact['issue_reduction']:+.1f} issues/transcript")
        dashboard.append("")

        dashboard.append("=" * 80)

        return "\n".join(dashboard)

    def generate_report(
        self,
        detection_results: Optional[List[DetectionResult]] = None,
    ) -> str:
        """
        Generate a detailed quality report.

        Args:
            detection_results: Optional list of DetectionResult objects

        Returns:
            Formatted report string
        """
        report = []
        report.append("=" * 80)
        report.append("DATA QUALITY FLYWHEEL - DETAILED REPORT")
        report.append(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        report.append("=" * 80)
        report.append("")

        # Dashboard summary
        report.append(self.generate_text_dashboard())
        report.append("")

        # Issue patterns (if detection results provided)
        if detection_results:
            patterns = self.analytics.get_issue_patterns(detection_results)

            report.append("ISSUE PATTERNS")
            report.append("-" * 80)
            report.append(f"Total Issues Detected:    {patterns['total_issues']:,}")
            report.append("")

            report.append("Most Common Issues:")
            for issue_type, count in patterns['most_common_issues']:
                report.append(f"  - {issue_type:40s}  {count:5d}")
            report.append("")

            if patterns['correlations']:
                report.append("Strongly Correlated Issues:")
                for corr in patterns['correlations'][:5]:
                    report.append(
                        f"  - {corr['issue1']} + {corr['issue2']}: "
                        f"{corr['correlation_strength']:.1%} correlation "
                        f"({corr['co_occurrence_count']} cases)"
                    )
            report.append("")

        report.append("=" * 80)

        return "\n".join(report)

    def _make_bar(self, percentage: float, width: int = 30) -> str:
        """Create a text-based bar chart"""
        filled = int(percentage / 100 * width)
        return f"[{'█' * filled}{'░' * (width - filled)}]"

    def export_json(self, filepath: str):
        """Export dashboard data to JSON file"""
        summary = self.analytics.get_dashboard_summary()

        with open(filepath, 'w') as f:
            json.dump(summary, f, indent=2)
