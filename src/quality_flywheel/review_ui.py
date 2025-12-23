"""
Researcher Review Interface

Interactive CLI for researchers to review and annotate
flagged transcripts.
"""

import sys
from typing import Optional, List, Dict
from datetime import datetime

from .feedback import FeedbackStore, Annotation, AnnotationType, ResolutionStatus
from .analytics import QualityAnalytics, QualityDashboard


class ReviewInterface:
    """
    Interactive command-line interface for reviewing transcripts.

    Allows researchers to:
    - View flagged transcripts
    - Add annotations and corrections
    - Rate quality
    - Mark issues as resolved
    - View statistics
    """

    def __init__(
        self,
        feedback_store: FeedbackStore,
        researcher_id: str = "researcher",
    ):
        """
        Initialize review interface.

        Args:
            feedback_store: FeedbackStore instance
            researcher_id: Identifier for the researcher
        """
        self.store = feedback_store
        self.researcher_id = researcher_id
        self.analytics = QualityAnalytics(feedback_store)
        self.dashboard = QualityDashboard(self.analytics)
        self.current_transcript = None

    def run(self):
        """Run the interactive review interface"""
        print("=" * 80)
        print("DATA QUALITY FLYWHEEL - TRANSCRIPT REVIEW INTERFACE")
        print("=" * 80)
        print()

        while True:
            self._show_main_menu()
            choice = input("\nEnter your choice: ").strip()

            if choice == '1':
                self._review_next_transcript()
            elif choice == '2':
                self._search_transcript()
            elif choice == '3':
                self._view_statistics()
            elif choice == '4':
                self._view_dashboard()
            elif choice == '5':
                self._view_suggestions()
            elif choice == 'q':
                print("\nExiting review interface. Goodbye!")
                break
            else:
                print("\nInvalid choice. Please try again.")

    def _show_main_menu(self):
        """Display main menu"""
        print("\n" + "-" * 80)
        print("MAIN MENU")
        print("-" * 80)
        print("1. Review next pending transcript")
        print("2. Search for specific transcript")
        print("3. View statistics")
        print("4. View quality dashboard")
        print("5. View improvement suggestions")
        print("q. Quit")

    def _review_next_transcript(self):
        """Review the next pending transcript"""
        # Get pending transcripts
        transcripts = self.store.get_pending_transcripts(limit=1)

        if not transcripts:
            print("\n✓ No pending transcripts to review!")
            return

        self.current_transcript = transcripts[0]
        self._review_transcript(self.current_transcript)

    def _search_transcript(self):
        """Search for a specific transcript"""
        transcript_id = input("\nEnter transcript ID: ").strip()

        transcript = self.store.get_transcript(transcript_id)

        if not transcript:
            print(f"\n✗ Transcript '{transcript_id}' not found.")
            return

        self.current_transcript = transcript
        self._review_transcript(transcript)

    def _review_transcript(self, transcript: Dict):
        """Review a single transcript"""
        print("\n" + "=" * 80)
        print("TRANSCRIPT REVIEW")
        print("=" * 80)

        # Display transcript details
        self._display_transcript(transcript)

        # Show existing annotations
        annotations = self.store.get_annotations_for_transcript(transcript['id'])
        if annotations:
            print("\n" + "-" * 80)
            print("EXISTING ANNOTATIONS")
            print("-" * 80)
            for i, ann in enumerate(annotations, 1):
                print(f"\n{i}. [{ann.annotation_type.value}] by {ann.researcher_id}")
                print(f"   {ann.content}")
                if ann.rating:
                    print(f"   Rating: {ann.rating:.2f}")

        # Review menu
        while True:
            print("\n" + "-" * 80)
            print("REVIEW ACTIONS")
            print("-" * 80)
            print("1. Add comment")
            print("2. Correct answer")
            print("3. Correct reasoning")
            print("4. Rate quality")
            print("5. Confirm issue")
            print("6. Mark as false positive")
            print("7. Mark as reviewed")
            print("8. Skip to next")
            print("b. Back to main menu")

            choice = input("\nEnter your choice: ").strip()

            if choice == '1':
                self._add_comment(transcript['id'])
            elif choice == '2':
                self._correct_answer(transcript['id'])
            elif choice == '3':
                self._correct_reasoning(transcript['id'])
            elif choice == '4':
                self._rate_quality(transcript['id'])
            elif choice == '5':
                self._confirm_issue(transcript['id'])
            elif choice == '6':
                self._mark_false_positive(transcript['id'])
            elif choice == '7':
                self._mark_reviewed(transcript['id'])
                break
            elif choice == '8':
                break
            elif choice == 'b':
                break
            else:
                print("\nInvalid choice. Please try again.")

    def _display_transcript(self, transcript: Dict):
        """Display transcript details"""
        print(f"\nID: {transcript['id']}")
        print(f"Quality Score: {transcript['quality_score']:.2%}")
        print(f"Severity: {transcript['severity'].upper()}")
        print(f"Status: {transcript['review_status']}")

        print("\n" + "-" * 80)
        print("QUESTION")
        print("-" * 80)
        print(transcript['question'])

        print("\n" + "-" * 80)
        print("RESPONSE")
        print("-" * 80)
        print(transcript['response'])

        if transcript['expected_answer']:
            print("\n" + "-" * 80)
            print("EXPECTED ANSWER")
            print("-" * 80)
            print(transcript['expected_answer'])

    def _add_comment(self, transcript_id: str):
        """Add a general comment"""
        print("\n" + "-" * 40)
        print("ADD COMMENT")
        print("-" * 40)
        comment = input("Enter your comment: ").strip()

        if comment:
            annotation = Annotation(
                transcript_id=transcript_id,
                annotation_type=AnnotationType.COMMENT,
                researcher_id=self.researcher_id,
                content=comment,
            )
            self.store.add_annotation(annotation)
            print("✓ Comment added successfully!")

    def _correct_answer(self, transcript_id: str):
        """Add corrected answer"""
        print("\n" + "-" * 40)
        print("CORRECT ANSWER")
        print("-" * 40)
        corrected = input("Enter corrected answer: ").strip()

        if corrected:
            annotation = Annotation(
                transcript_id=transcript_id,
                annotation_type=AnnotationType.CORRECTED_ANSWER,
                researcher_id=self.researcher_id,
                content=corrected,
            )
            self.store.add_annotation(annotation)
            print("✓ Corrected answer saved!")

    def _correct_reasoning(self, transcript_id: str):
        """Add corrected reasoning"""
        print("\n" + "-" * 40)
        print("CORRECT REASONING")
        print("-" * 40)
        print("Enter corrected reasoning (empty line to finish):")

        lines = []
        while True:
            line = input()
            if not line:
                break
            lines.append(line)

        corrected = '\n'.join(lines)

        if corrected:
            annotation = Annotation(
                transcript_id=transcript_id,
                annotation_type=AnnotationType.CORRECTED_REASONING,
                researcher_id=self.researcher_id,
                content=corrected,
            )
            self.store.add_annotation(annotation)
            print("✓ Corrected reasoning saved!")

    def _rate_quality(self, transcript_id: str):
        """Rate transcript quality"""
        print("\n" + "-" * 40)
        print("RATE QUALITY")
        print("-" * 40)
        print("Enter rating (0-10):")

        try:
            rating = float(input().strip())
            if 0 <= rating <= 10:
                annotation = Annotation(
                    transcript_id=transcript_id,
                    annotation_type=AnnotationType.QUALITY_RATING,
                    researcher_id=self.researcher_id,
                    content=f"Quality rating: {rating}/10",
                    rating=rating / 10,  # Normalize to 0-1
                )
                self.store.add_annotation(annotation)
                print("✓ Quality rating saved!")
            else:
                print("✗ Rating must be between 0 and 10.")
        except ValueError:
            print("✗ Invalid rating. Please enter a number.")

    def _confirm_issue(self, transcript_id: str):
        """Confirm an issue"""
        print("\n" + "-" * 40)
        print("CONFIRM ISSUE")
        print("-" * 40)
        description = input("Describe the confirmed issue: ").strip()

        if description:
            annotation = Annotation(
                transcript_id=transcript_id,
                annotation_type=AnnotationType.CONFIRMED_ISSUE,
                researcher_id=self.researcher_id,
                content=description,
            )
            self.store.add_annotation(annotation)
            print("✓ Issue confirmed!")

    def _mark_false_positive(self, transcript_id: str):
        """Mark as false positive"""
        annotation = Annotation(
            transcript_id=transcript_id,
            annotation_type=AnnotationType.FALSE_POSITIVE,
            researcher_id=self.researcher_id,
            content="Marked as false positive - no actual issue found",
        )
        self.store.add_annotation(annotation)
        self.store.mark_transcript_reviewed(transcript_id, self.researcher_id, "false_positive")
        print("✓ Marked as false positive!")

    def _mark_reviewed(self, transcript_id: str):
        """Mark transcript as reviewed"""
        self.store.mark_transcript_reviewed(transcript_id, self.researcher_id, "reviewed")
        print("✓ Transcript marked as reviewed!")

    def _view_statistics(self):
        """View statistics"""
        print("\n" + "=" * 80)
        print("STATISTICS")
        print("=" * 80)

        stats = self.store.get_statistics()

        print(f"\nTotal Transcripts:        {stats['total_transcripts']:,}")
        print(f"Pending Reviews:          {stats['pending_reviews']:,}")
        print(f"Reviewed:                 {stats['reviewed_transcripts']:,}")
        print(f"Average Quality Score:    {stats['average_quality_score']:.2%}")
        print(f"Total Annotations:        {stats['total_annotations']:,}")

        print("\nTranscripts by Severity:")
        for severity, count in sorted(stats['transcripts_by_severity'].items()):
            print(f"  {severity.capitalize():12s}  {count:5d}")

        if stats['annotations_by_type']:
            print("\nAnnotations by Type:")
            for ann_type, count in sorted(stats['annotations_by_type'].items()):
                print(f"  {ann_type:30s}  {count:5d}")

    def _view_dashboard(self):
        """View quality dashboard"""
        print("\n")
        print(self.dashboard.generate_text_dashboard())

    def _view_suggestions(self):
        """View improvement suggestions"""
        print("\n" + "=" * 80)
        print("IMPROVEMENT SUGGESTIONS")
        print("=" * 80)

        pending = self.store.get_improvement_suggestions(status='pending', limit=20)

        if not pending:
            print("\nNo pending suggestions.")
            return

        for i, sugg in enumerate(pending, 1):
            print(f"\n{i}. [{sugg['suggestion_type']}] Priority: {sugg['priority']}")
            print(f"   {sugg['description']}")
            print(f"   Suggested by: {sugg['suggested_by']} at {sugg['suggested_at']}")


def main():
    """Main entry point for the review interface"""
    import argparse

    parser = argparse.ArgumentParser(description='Transcript Review Interface')
    parser.add_argument(
        '--db',
        default='./data/quality_feedback.db',
        help='Path to feedback database',
    )
    parser.add_argument(
        '--researcher',
        default='researcher',
        help='Researcher ID',
    )

    args = parser.parse_args()

    # Initialize feedback store
    store = FeedbackStore(args.db)

    # Create interface
    interface = ReviewInterface(store, researcher_id=args.researcher)

    try:
        interface.run()
    except KeyboardInterrupt:
        print("\n\nInterrupted by user. Exiting...")
    finally:
        store.close()


if __name__ == '__main__':
    main()
