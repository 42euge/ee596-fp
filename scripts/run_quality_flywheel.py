#!/usr/bin/env python3
"""
Quality Flywheel Integration Script

Integrates the data quality flywheel with the GRPO training pipeline.

This script:
1. Evaluates model-generated transcripts for quality issues
2. Detects and flags problematic transcripts
3. Stores feedback in the feedback database
4. Generates quality analytics and reports
5. Creates improvement suggestions for the next training iteration
"""

import argparse
import json
import sys
from pathlib import Path
from typing import List, Dict, Optional

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from quality_flywheel import (
    QualityMetrics,
    IssueDetector,
    FeedbackStore,
    QualityAnalytics,
    QualityDashboard,
    ImprovementLoop,
)


def load_transcripts(file_path: str) -> List[Dict]:
    """
    Load transcripts from a JSON or JSONL file.

    Args:
        file_path: Path to transcripts file

    Returns:
        List of transcript dictionaries
    """
    transcripts = []

    with open(file_path, 'r') as f:
        # Try JSON first
        try:
            data = json.load(f)
            if isinstance(data, list):
                transcripts = data
            elif isinstance(data, dict):
                # Might be a wrapped format
                if 'examples' in data:
                    transcripts = data['examples']
                elif 'transcripts' in data:
                    transcripts = data['transcripts']
                else:
                    transcripts = [data]
        except json.JSONDecodeError:
            # Try JSONL
            f.seek(0)
            for line in f:
                line = line.strip()
                if line:
                    transcripts.append(json.loads(line))

    return transcripts


def evaluate_transcripts(
    transcripts: List[Dict],
    store: FeedbackStore,
    detector: IssueDetector,
    save_to_db: bool = True,
) -> List:
    """
    Evaluate transcripts and detect issues.

    Args:
        transcripts: List of transcript dictionaries
        store: FeedbackStore instance
        detector: IssueDetector instance
        save_to_db: Whether to save results to database

    Returns:
        List of DetectionResult objects
    """
    print(f"\nEvaluating {len(transcripts)} transcripts...")

    # Run detection
    detection_results = detector.detect_batch(
        transcripts,
        id_field='id',
        question_field='question',
        response_field='response',
        answer_field='answer',
    )

    # Save to database if requested
    if save_to_db:
        print("Saving results to database...")
        for result in detection_results:
            quality = result.quality

            # Add transcript
            store.add_transcript(
                transcript_id=quality.transcript_id,
                question=quality.question,
                response=quality.response,
                expected_answer=quality.expected_answer,
                quality_score=quality.overall_score,
                severity=quality.severity,
                metadata={
                    'format_score': quality.format_score,
                    'answer_correct': quality.answer_correct,
                    'reasoning_completeness': quality.reasoning_completeness,
                    'issues': quality.issues,
                },
            )

            # Add quality history
            store.add_quality_record(
                transcript_id=quality.transcript_id,
                quality_score=quality.overall_score,
                format_score=quality.format_score,
                answer_correct=quality.answer_correct,
                reasoning_completeness=quality.reasoning_completeness,
                overall_score=quality.overall_score,
                severity=quality.severity,
                issues=quality.issues,
            )

    # Get statistics
    stats = detector.get_statistics(detection_results)

    print(f"\n✓ Evaluation complete!")
    print(f"  - Total transcripts: {stats['total_transcripts']}")
    print(f"  - Flagged: {stats['flagged_count']} ({stats['flagged_percentage']:.1f}%)")
    print(f"  - Average quality: {stats['average_quality_score']:.2%}")

    return detection_results


def generate_reports(
    detection_results: List,
    store: FeedbackStore,
    output_dir: str,
):
    """
    Generate quality reports and analytics.

    Args:
        detection_results: List of DetectionResult objects
        store: FeedbackStore instance
        output_dir: Directory to save reports
    """
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print(f"\nGenerating reports in {output_dir}...")

    # Create analytics
    analytics = QualityAnalytics(store)
    dashboard = QualityDashboard(analytics)

    # Generate text dashboard
    dashboard_text = dashboard.generate_text_dashboard()
    with open(output_path / 'dashboard.txt', 'w') as f:
        f.write(dashboard_text)
    print("  ✓ dashboard.txt")

    # Generate detailed report
    report_text = dashboard.generate_report(detection_results)
    with open(output_path / 'detailed_report.txt', 'w') as f:
        f.write(report_text)
    print("  ✓ detailed_report.txt")

    # Export JSON summary
    dashboard.export_json(str(output_path / 'dashboard.json'))
    print("  ✓ dashboard.json")

    # Get issue patterns
    patterns = analytics.get_issue_patterns(detection_results)
    with open(output_path / 'issue_patterns.json', 'w') as f:
        json.dump(patterns, f, indent=2)
    print("  ✓ issue_patterns.json")

    # Get flagged transcripts
    flagged = [r for r in detection_results if r.should_flag]
    flagged_data = {
        'total_flagged': len(flagged),
        'by_priority': {},
        'transcripts': [r.to_dict() for r in flagged],
    }

    for result in flagged:
        priority = result.priority
        flagged_data['by_priority'][priority] = flagged_data['by_priority'].get(priority, 0) + 1

    with open(output_path / 'flagged_transcripts.json', 'w') as f:
        json.dump(flagged_data, f, indent=2)
    print("  ✓ flagged_transcripts.json")

    print("\n✓ Reports generated!")


def generate_improvements(
    detection_results: List,
    store: FeedbackStore,
    iteration_name: str,
    output_dir: str,
):
    """
    Generate improvement suggestions and datasets.

    Args:
        detection_results: List of DetectionResult objects
        store: FeedbackStore instance
        iteration_name: Name for this improvement iteration
        output_dir: Directory to save outputs
    """
    print(f"\nGenerating improvement iteration: {iteration_name}...")

    improvement_loop = ImprovementLoop(store)

    # Create improvement iteration
    summary = improvement_loop.create_improvement_iteration(
        iteration_name=iteration_name,
        detection_results=detection_results,
        output_dir=output_dir,
    )

    print(f"\n✓ Improvement iteration created!")
    print(f"  - Output directory: {summary['output_directory']}")
    print(f"  - Targeted examples: {summary['dataset']['num_examples']}")
    print(f"  - Prompt suggestions: {summary['num_prompt_suggestions']}")
    print(f"  - Reward suggestions: {summary['num_reward_suggestions']}")

    return summary


def main():
    parser = argparse.ArgumentParser(
        description='Run the data quality flywheel on transcript data'
    )
    parser.add_argument(
        'transcripts_file',
        help='Path to transcripts file (JSON or JSONL)',
    )
    parser.add_argument(
        '--db',
        default='./data/quality_feedback.db',
        help='Path to feedback database (default: ./data/quality_feedback.db)',
    )
    parser.add_argument(
        '--output-dir',
        default='./quality_reports',
        help='Directory for output reports (default: ./quality_reports)',
    )
    parser.add_argument(
        '--quality-threshold',
        type=float,
        default=0.6,
        help='Quality score threshold for flagging (default: 0.6)',
    )
    parser.add_argument(
        '--no-save',
        action='store_true',
        help='Do not save results to database',
    )
    parser.add_argument(
        '--generate-improvements',
        action='store_true',
        help='Generate improvement iteration',
    )
    parser.add_argument(
        '--iteration-name',
        help='Name for improvement iteration (default: auto-generated)',
    )

    args = parser.parse_args()

    # Load transcripts
    print(f"Loading transcripts from {args.transcripts_file}...")
    try:
        transcripts = load_transcripts(args.transcripts_file)
        print(f"✓ Loaded {len(transcripts)} transcripts")
    except Exception as e:
        print(f"✗ Error loading transcripts: {e}")
        return 1

    # Initialize components
    print("\nInitializing quality flywheel components...")
    store = FeedbackStore(args.db)
    detector = IssueDetector(quality_threshold=args.quality_threshold)
    print("✓ Components initialized")

    try:
        # Evaluate transcripts
        detection_results = evaluate_transcripts(
            transcripts=transcripts,
            store=store,
            detector=detector,
            save_to_db=not args.no_save,
        )

        # Generate reports
        generate_reports(
            detection_results=detection_results,
            store=store,
            output_dir=args.output_dir,
        )

        # Generate improvements if requested
        if args.generate_improvements:
            from datetime import datetime
            iteration_name = args.iteration_name or f"iteration_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

            improvement_dir = Path(args.output_dir) / 'improvements'
            generate_improvements(
                detection_results=detection_results,
                store=store,
                iteration_name=iteration_name,
                output_dir=str(improvement_dir),
            )

        print("\n" + "=" * 80)
        print("QUALITY FLYWHEEL COMPLETE!")
        print("=" * 80)
        print(f"\nResults saved to: {args.output_dir}")
        print(f"Database: {args.db}")
        print("\nNext steps:")
        print("  1. Review flagged transcripts using the review interface:")
        print(f"     python -m quality_flywheel.review_ui --db {args.db}")
        print("  2. View the dashboard and reports in the output directory")
        print("  3. Use improvement suggestions to enhance your training pipeline")
        print()

    finally:
        store.close()

    return 0


if __name__ == '__main__':
    sys.exit(main())
