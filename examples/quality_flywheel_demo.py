#!/usr/bin/env python3
"""
Quality Flywheel Demo

Demonstrates the complete data quality flywheel workflow with sample data.
"""

import sys
import json
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / 'src'))

from quality_flywheel import (
    QualityMetrics,
    IssueDetector,
    FeedbackStore,
    Annotation,
    AnnotationType,
    QualityAnalytics,
    QualityDashboard,
    ImprovementLoop,
)


# Sample transcripts with various quality issues
SAMPLE_TRANSCRIPTS = [
    {
        "id": "good_1",
        "question": "If Sarah has 12 apples and gives 4 to her friend, how many does she have left?",
        "response": "<reasoning>Sarah starts with 12 apples. She gives away 4 apples. To find how many she has left, I need to subtract: 12 - 4 = 8 apples.</reasoning><answer>8</answer>",
        "answer": "8",
    },
    {
        "id": "missing_reasoning_1",
        "question": "What is 15 + 7?",
        "response": "<answer>22</answer>",
        "answer": "22",
    },
    {
        "id": "missing_answer_1",
        "question": "Calculate 10 × 5",
        "response": "<reasoning>To multiply 10 by 5, I get 50.</reasoning>",
        "answer": "50",
    },
    {
        "id": "wrong_answer_1",
        "question": "If a book costs $25 and is discounted by 20%, what is the sale price?",
        "response": "<reasoning>20% of $25 is $5. So the discount is $5. Sale price = $25 - $5 = $20.</reasoning><answer>$15</answer>",
        "answer": "20",
    },
    {
        "id": "insufficient_reasoning_1",
        "question": "A train travels 120 miles in 2 hours. What is its average speed?",
        "response": "<reasoning>Easy.</reasoning><answer>60 mph</answer>",
        "answer": "60",
    },
    {
        "id": "good_2",
        "question": "What is 100 divided by 4?",
        "response": "<reasoning>To divide 100 by 4, I can think of it as how many 4s fit into 100. First, 4 × 20 = 80, and 4 × 5 = 20. So 4 × 25 = 100. Therefore, 100 ÷ 4 = 25.</reasoning><answer>25</answer>",
        "answer": "25",
    },
    {
        "id": "low_coherence_1",
        "question": "If eggs cost $3 per dozen and you buy 3 dozen, how much do you pay?",
        "response": "<reasoning>Eggs dozen three. Cost three dollars. Buy three. Money payment total.</reasoning><answer>$9</answer>",
        "answer": "9",
    },
]


def main():
    print("=" * 80)
    print("DATA QUALITY FLYWHEEL - DEMONSTRATION")
    print("=" * 80)
    print()

    # Create demo directory
    demo_dir = Path("./demo_output")
    demo_dir.mkdir(exist_ok=True)

    # Initialize components
    print("1. Initializing Quality Flywheel Components")
    print("-" * 80)

    db_path = demo_dir / "demo_feedback.db"
    store = FeedbackStore(str(db_path))
    metrics_calculator = QualityMetrics()
    detector = IssueDetector(
        quality_threshold=0.6,
        auto_flag_critical=True,
        auto_flag_incorrect_answers=True,
    )
    analytics = QualityAnalytics(store)
    dashboard = QualityDashboard(analytics)
    improvement = ImprovementLoop(store, analytics)

    print(f"✓ Components initialized")
    print(f"  Database: {db_path}")
    print()

    # Step 1: Evaluate Quality
    print("2. Evaluating Transcript Quality")
    print("-" * 80)

    detection_results = detector.detect_batch(SAMPLE_TRANSCRIPTS)

    print(f"✓ Evaluated {len(SAMPLE_TRANSCRIPTS)} transcripts")
    print()

    # Display individual results
    for result in detection_results:
        quality = result.quality
        print(f"  {quality.transcript_id:25s}  Score: {quality.overall_score:.2%}  "
              f"Severity: {quality.severity:8s}  Flagged: {result.should_flag}")

    print()

    # Step 2: Store Results
    print("3. Storing Results in Database")
    print("-" * 80)

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

    print(f"✓ Stored {len(detection_results)} transcripts in database")
    print()

    # Step 3: Analyze Issues
    print("4. Analyzing Issue Patterns")
    print("-" * 80)

    stats = detector.get_statistics(detection_results)
    patterns = analytics.get_issue_patterns(detection_results)

    print(f"Total Transcripts:    {stats['total_transcripts']}")
    print(f"Flagged:              {stats['flagged_count']} ({stats['flagged_percentage']:.1f}%)")
    print(f"Average Quality:      {stats['average_quality_score']:.2%}")
    print()

    print("Issue Type Counts:")
    for issue_type, count in sorted(patterns['issue_type_counts'].items(), key=lambda x: x[1], reverse=True):
        print(f"  {issue_type:40s}  {count}")
    print()

    # Step 4: Simulate Researcher Annotations
    print("5. Simulating Researcher Annotations")
    print("-" * 80)

    # Add some sample annotations
    annotations = [
        Annotation(
            transcript_id="wrong_answer_1",
            annotation_type=AnnotationType.CORRECTED_ANSWER,
            researcher_id="demo_researcher",
            content="$20",
        ),
        Annotation(
            transcript_id="wrong_answer_1",
            annotation_type=AnnotationType.CONFIRMED_ISSUE,
            researcher_id="demo_researcher",
            content="Answer calculation is correct but final answer is wrong. Likely typo.",
        ),
        Annotation(
            transcript_id="insufficient_reasoning_1",
            annotation_type=AnnotationType.CORRECTED_REASONING,
            researcher_id="demo_researcher",
            content="To find average speed, divide total distance by total time. The train travels 120 miles in 2 hours. Speed = 120 miles ÷ 2 hours = 60 miles per hour.",
        ),
        Annotation(
            transcript_id="good_1",
            annotation_type=AnnotationType.QUALITY_RATING,
            researcher_id="demo_researcher",
            content="Excellent reasoning and correct answer",
            rating=0.95,
        ),
    ]

    for ann in annotations:
        store.add_annotation(ann)

    print(f"✓ Added {len(annotations)} researcher annotations")
    print()

    # Step 5: Generate Dashboard
    print("6. Generating Quality Dashboard")
    print("-" * 80)

    dashboard_text = dashboard.generate_text_dashboard()
    print(dashboard_text)

    # Save dashboard
    with open(demo_dir / "dashboard.txt", 'w') as f:
        f.write(dashboard_text)
    print(f"\n✓ Saved to {demo_dir / 'dashboard.txt'}")
    print()

    # Step 6: Generate Improvements
    print("7. Generating Improvement Suggestions")
    print("-" * 80)

    # Prompt suggestions
    prompt_suggestions = improvement.suggest_prompt_improvements(detection_results)
    print(f"\nPrompt Improvement Suggestions ({len(prompt_suggestions)}):")
    for i, sugg in enumerate(prompt_suggestions, 1):
        print(f"\n  {i}. [{sugg['type']}] Priority {sugg['priority']}")
        print(f"     Issue: {sugg['issue']}")
        print(f"     {sugg['description']}")
        print(f"     Suggested: {sugg['suggested_change']}")

    # Reward function suggestions
    reward_suggestions = improvement.suggest_reward_function_updates(detection_results)
    print(f"\nReward Function Suggestions ({len(reward_suggestions)}):")
    for i, sugg in enumerate(reward_suggestions, 1):
        print(f"\n  {i}. [{sugg['type']}] Priority {sugg['priority']}")
        print(f"     Issue: {sugg['issue']}")
        print(f"     {sugg['description']}")
        print(f"     Suggested: {sugg['suggested_change']}")

    print()

    # Step 7: Generate Targeted Dataset
    print("8. Generating Targeted Training Dataset")
    print("-" * 80)

    dataset_path = demo_dir / "targeted_dataset.json"
    dataset_stats = improvement.generate_targeted_dataset(
        output_path=str(dataset_path),
        max_examples=10,
        include_corrections=True,
    )

    print(f"✓ Generated targeted dataset")
    print(f"  Output: {dataset_path}")
    print(f"  Examples: {dataset_stats['num_examples']}")
    print(f"  With corrections: {dataset_stats['num_with_corrections']}")
    print()

    # Step 8: Create Improvement Iteration
    print("9. Creating Improvement Iteration Package")
    print("-" * 80)

    iteration_summary = improvement.create_improvement_iteration(
        iteration_name="demo_iteration",
        detection_results=detection_results,
        output_dir=str(demo_dir / "improvements"),
    )

    print(f"✓ Improvement iteration created")
    print(f"  Directory: {iteration_summary['output_directory']}")
    print(f"  Targeted examples: {iteration_summary['dataset']['num_examples']}")
    print(f"  Prompt suggestions: {iteration_summary['num_prompt_suggestions']}")
    print(f"  Reward suggestions: {iteration_summary['num_reward_suggestions']}")
    print()

    # Step 9: Summary
    print("10. Summary and Next Steps")
    print("-" * 80)

    improvement_summary = improvement.get_improvement_summary()

    print(f"Pending Suggestions:      {improvement_summary['pending_suggestions']['total']}")
    print(f"Implemented Suggestions:  {improvement_summary['implemented_suggestions']['total']}")
    print()

    print("Generated Files:")
    for file in sorted(demo_dir.rglob("*")):
        if file.is_file():
            print(f"  {file.relative_to(demo_dir)}")
    print()

    print("Next Steps:")
    print("  1. Review the generated dashboard and reports")
    print("  2. Examine flagged transcripts in the database")
    print("  3. Run the review interface:")
    print(f"     python -m quality_flywheel.review_ui --db {db_path}")
    print("  4. Incorporate improvement suggestions into your training pipeline")
    print("  5. Use the targeted dataset for retraining")
    print()

    print("=" * 80)
    print("DEMONSTRATION COMPLETE!")
    print("=" * 80)
    print()

    # Cleanup
    store.close()


if __name__ == '__main__':
    main()
