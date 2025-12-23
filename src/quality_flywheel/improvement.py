"""
Improvement Loop Module

Feeds quality insights and researcher feedback back into the training pipeline
to create a continuous improvement cycle.
"""

import json
from typing import List, Dict, Optional, Callable
from pathlib import Path
from datetime import datetime
import random

from .feedback import FeedbackStore, Annotation, AnnotationType
from .detector import DetectionResult, IssueType
from .analytics import QualityAnalytics


class ImprovementLoop:
    """
    Continuous improvement system that uses quality feedback
    to enhance model training.

    Capabilities:
    - Generate targeted training examples from flagged transcripts
    - Update prompts based on common issues
    - Adjust reward functions based on feedback
    - Create synthetic edge cases for training
    - Track improvement iterations
    """

    def __init__(
        self,
        feedback_store: FeedbackStore,
        analytics: Optional[QualityAnalytics] = None,
    ):
        """
        Initialize improvement loop.

        Args:
            feedback_store: FeedbackStore instance
            analytics: Optional QualityAnalytics instance
        """
        self.store = feedback_store
        self.analytics = analytics or QualityAnalytics(feedback_store)

    def generate_targeted_dataset(
        self,
        output_path: str,
        issue_types: Optional[List[IssueType]] = None,
        min_priority: int = 3,
        max_examples: int = 1000,
        include_corrections: bool = True,
    ) -> Dict:
        """
        Generate a targeted training dataset from flagged transcripts.

        Args:
            output_path: Path to save the dataset
            issue_types: Specific issue types to focus on (None = all)
            min_priority: Minimum priority level (1-5)
            max_examples: Maximum number of examples
            include_corrections: Include researcher corrections if available

        Returns:
            Dictionary with dataset statistics
        """
        # Get flagged transcripts
        transcripts = self.store.get_pending_transcripts(limit=max_examples)

        # Also get reviewed transcripts with annotations
        cursor = self.store.conn.cursor()
        cursor.execute("""
            SELECT t.*, GROUP_CONCAT(a.annotation_type) as annotation_types
            FROM transcripts t
            LEFT JOIN annotations a ON t.id = a.transcript_id
            WHERE t.review_status != 'pending'
            GROUP BY t.id
            LIMIT ?
        """, (max_examples,))
        reviewed = cursor.fetchall()

        # Combine and deduplicate
        all_transcripts = {t['id']: dict(t) for t in transcripts}
        for t in reviewed:
            if t['id'] not in all_transcripts:
                all_transcripts[t['id']] = dict(t)

        # Build targeted examples
        targeted_examples = []

        for transcript_id, transcript in all_transcripts.items():
            # Get annotations
            annotations = self.store.get_annotations_for_transcript(transcript_id)

            # Build example
            example = {
                'id': transcript_id,
                'question': transcript['question'],
                'original_response': transcript['response'],
                'expected_answer': transcript['expected_answer'],
                'quality_score': transcript['quality_score'],
                'severity': transcript['severity'],
                'issues': [],
                'corrections': [],
            }

            # Add corrections from annotations if available
            if include_corrections:
                for ann in annotations:
                    if ann.annotation_type == AnnotationType.CORRECTED_ANSWER:
                        example['corrections'].append({
                            'type': 'answer',
                            'corrected_value': ann.content,
                            'researcher': ann.researcher_id,
                        })
                    elif ann.annotation_type == AnnotationType.CORRECTED_REASONING:
                        example['corrections'].append({
                            'type': 'reasoning',
                            'corrected_value': ann.content,
                            'researcher': ann.researcher_id,
                        })

            targeted_examples.append(example)

        # Limit to max_examples
        targeted_examples = targeted_examples[:max_examples]

        # Save dataset
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        dataset = {
            'metadata': {
                'created_at': datetime.now().isoformat(),
                'num_examples': len(targeted_examples),
                'issue_types_filter': [it.value for it in issue_types] if issue_types else 'all',
                'min_priority': min_priority,
                'include_corrections': include_corrections,
            },
            'examples': targeted_examples,
        }

        with open(output_path, 'w') as f:
            json.dump(dataset, f, indent=2)

        # Record suggestion
        self.store.add_improvement_suggestion(
            suggestion_type='targeted_dataset',
            description=f'Generated targeted dataset with {len(targeted_examples)} examples',
            priority=2,
            suggested_by='improvement_loop',
        )

        return {
            'output_path': output_path,
            'num_examples': len(targeted_examples),
            'num_with_corrections': len([e for e in targeted_examples if e['corrections']]),
        }

    def suggest_prompt_improvements(
        self,
        detection_results: List[DetectionResult],
    ) -> List[Dict]:
        """
        Suggest prompt improvements based on detected issues.

        Args:
            detection_results: List of DetectionResult objects

        Returns:
            List of prompt improvement suggestions
        """
        patterns = self.analytics.get_issue_patterns(detection_results)

        suggestions = []

        # Check for format issues
        format_issues = [
            'missing_reasoning_tags',
            'missing_answer_tags',
            'malformed_structure',
        ]
        format_count = sum(
            patterns['issue_type_counts'].get(issue, 0)
            for issue in format_issues
        )

        if format_count > len(detection_results) * 0.2:  # More than 20%
            suggestions.append({
                'type': 'prompt_update',
                'issue': 'high_format_violations',
                'description': 'High rate of format violations detected. Consider emphasizing format requirements in the system prompt.',
                'priority': 1,
                'suggested_change': 'Add explicit examples of correct format with <reasoning> and <answer> tags. Emphasize that tags are REQUIRED.',
            })

        # Check for reasoning quality issues
        reasoning_issues = [
            'insufficient_reasoning',
            'low_coherence',
            'missing_step_markers',
            'weak_logical_flow',
        ]
        reasoning_count = sum(
            patterns['issue_type_counts'].get(issue, 0)
            for issue in reasoning_issues
        )

        if reasoning_count > len(detection_results) * 0.3:  # More than 30%
            suggestions.append({
                'type': 'prompt_update',
                'issue': 'poor_reasoning_quality',
                'description': 'High rate of poor reasoning quality. Consider adding more guidance on step-by-step problem solving.',
                'priority': 1,
                'suggested_change': 'Add instruction to break down the problem into clear steps. Request explicit numbering or "First, Second, Then" structure.',
            })

        # Check for calculation issues
        calc_issues = [
            'missing_numerical_content',
            'missing_calculation_keywords',
        ]
        calc_count = sum(
            patterns['issue_type_counts'].get(issue, 0)
            for issue in calc_issues
        )

        if calc_count > len(detection_results) * 0.25:  # More than 25%
            suggestions.append({
                'type': 'prompt_update',
                'issue': 'missing_calculations',
                'description': 'Model often omits explicit calculations. Consider requesting detailed arithmetic.',
                'priority': 2,
                'suggested_change': 'Add instruction to "show all calculations explicitly" and "state each arithmetic operation clearly".',
            })

        # Record suggestions in database
        for suggestion in suggestions:
            self.store.add_improvement_suggestion(
                suggestion_type=suggestion['type'],
                description=suggestion['description'],
                priority=suggestion['priority'],
                suggested_by='improvement_loop',
            )

        return suggestions

    def suggest_reward_function_updates(
        self,
        detection_results: List[DetectionResult],
    ) -> List[Dict]:
        """
        Suggest reward function updates based on detected issues.

        Args:
            detection_results: List of DetectionResult objects

        Returns:
            List of reward function update suggestions
        """
        patterns = self.analytics.get_issue_patterns(detection_results)

        suggestions = []

        # Analyze incorrect answer patterns
        incorrect_count = patterns['issue_type_counts'].get('incorrect_answer', 0)
        if incorrect_count > 0:
            # Calculate how many incorrect answers were close
            close_answers = 0
            for result in detection_results:
                if result.quality.expected_answer and not result.quality.answer_correct:
                    if result.quality.answer_similarity > 0.5:
                        close_answers += 1

            if close_answers > incorrect_count * 0.3:  # More than 30% were close
                suggestions.append({
                    'type': 'reward_function_update',
                    'issue': 'close_but_incorrect_answers',
                    'description': f'{close_answers}/{incorrect_count} incorrect answers were numerically close. Consider partial credit.',
                    'priority': 2,
                    'suggested_change': 'Increase partial credit for numerically close answers (currently gives 0.5 for 90-110% ratio).',
                })

        # Analyze format compliance
        format_count = patterns['issue_type_counts'].get('missing_reasoning_tags', 0)
        format_count += patterns['issue_type_counts'].get('missing_answer_tags', 0)

        if format_count > len(detection_results) * 0.15:  # More than 15%
            suggestions.append({
                'type': 'reward_function_update',
                'issue': 'format_violations',
                'description': 'High rate of format violations. Consider stronger penalties.',
                'priority': 1,
                'suggested_change': 'Increase penalty for missing format tags from current value to -2.0 or -3.0.',
            })

        # Analyze reasoning quality
        low_coherence_count = patterns['issue_type_counts'].get('low_coherence', 0)

        if low_coherence_count > len(detection_results) * 0.2:  # More than 20%
            suggestions.append({
                'type': 'reward_function_update',
                'issue': 'low_reasoning_coherence',
                'description': 'Many transcripts have low reasoning coherence.',
                'priority': 2,
                'suggested_change': 'Add a coherence-based reward component that checks for logical connectors and flow.',
            })

        # Record suggestions
        for suggestion in suggestions:
            self.store.add_improvement_suggestion(
                suggestion_type=suggestion['type'],
                description=suggestion['description'],
                priority=suggestion['priority'],
                suggested_by='improvement_loop',
            )

        return suggestions

    def generate_synthetic_examples(
        self,
        issue_type: IssueType,
        num_examples: int = 10,
        base_examples: Optional[List[Dict]] = None,
    ) -> List[Dict]:
        """
        Generate synthetic examples for specific issue types.

        Args:
            issue_type: Type of issue to generate examples for
            num_examples: Number of examples to generate
            base_examples: Optional base examples to use as templates

        Returns:
            List of synthetic examples
        """
        synthetic_examples = []

        if issue_type == IssueType.MISSING_REASONING_TAGS:
            # Generate examples with missing reasoning tags
            for i in range(num_examples):
                synthetic_examples.append({
                    'id': f'synthetic_missing_reasoning_{i}',
                    'question': 'Sample math problem',
                    'response': '<answer>42</answer>',  # Missing reasoning
                    'expected_answer': '42',
                    'issue_type': issue_type.value,
                    'synthetic': True,
                })

        elif issue_type == IssueType.MISSING_ANSWER_TAGS:
            # Generate examples with missing answer tags
            for i in range(num_examples):
                synthetic_examples.append({
                    'id': f'synthetic_missing_answer_{i}',
                    'question': 'Sample math problem',
                    'response': '<reasoning>Step 1: Calculate...</reasoning>',  # Missing answer
                    'expected_answer': '42',
                    'issue_type': issue_type.value,
                    'synthetic': True,
                })

        elif issue_type == IssueType.INSUFFICIENT_REASONING:
            # Generate examples with insufficient reasoning
            for i in range(num_examples):
                synthetic_examples.append({
                    'id': f'synthetic_insufficient_reasoning_{i}',
                    'question': 'Sample complex math problem',
                    'response': '<reasoning>Simple.</reasoning><answer>42</answer>',  # Too short
                    'expected_answer': '42',
                    'issue_type': issue_type.value,
                    'synthetic': True,
                })

        return synthetic_examples

    def create_improvement_iteration(
        self,
        iteration_name: str,
        detection_results: List[DetectionResult],
        output_dir: str = './data/improvements',
    ) -> Dict:
        """
        Create a complete improvement iteration package.

        Includes:
        - Targeted training dataset
        - Prompt improvement suggestions
        - Reward function update suggestions
        - Analytics report

        Args:
            iteration_name: Name for this improvement iteration
            detection_results: Detection results to analyze
            output_dir: Directory to save outputs

        Returns:
            Dictionary with iteration summary
        """
        output_path = Path(output_dir) / iteration_name
        output_path.mkdir(parents=True, exist_ok=True)

        # Generate targeted dataset
        dataset_path = output_path / 'targeted_dataset.json'
        dataset_stats = self.generate_targeted_dataset(
            output_path=str(dataset_path),
            max_examples=500,
            include_corrections=True,
        )

        # Generate prompt suggestions
        prompt_suggestions = self.suggest_prompt_improvements(detection_results)
        with open(output_path / 'prompt_suggestions.json', 'w') as f:
            json.dump(prompt_suggestions, f, indent=2)

        # Generate reward function suggestions
        reward_suggestions = self.suggest_reward_function_updates(detection_results)
        with open(output_path / 'reward_suggestions.json', 'w') as f:
            json.dump(reward_suggestions, f, indent=2)

        # Generate analytics report
        patterns = self.analytics.get_issue_patterns(detection_results)
        with open(output_path / 'issue_patterns.json', 'w') as f:
            json.dump(patterns, f, indent=2)

        # Create summary
        summary = {
            'iteration_name': iteration_name,
            'created_at': datetime.now().isoformat(),
            'dataset': dataset_stats,
            'num_prompt_suggestions': len(prompt_suggestions),
            'num_reward_suggestions': len(reward_suggestions),
            'total_issues_analyzed': patterns['total_issues'],
            'output_directory': str(output_path),
        }

        with open(output_path / 'iteration_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)

        # Record in database
        self.store.add_improvement_suggestion(
            suggestion_type='improvement_iteration',
            description=f'Created improvement iteration: {iteration_name}',
            priority=1,
            suggested_by='improvement_loop',
        )

        return summary

    def export_for_retraining(
        self,
        output_path: str,
        min_quality_score: float = 0.0,
        max_quality_score: float = 0.7,
        include_corrections: bool = True,
    ) -> Dict:
        """
        Export data specifically formatted for retraining.

        Args:
            output_path: Path to save the retraining dataset
            min_quality_score: Minimum quality score to include
            max_quality_score: Maximum quality score to include
            include_corrections: Use corrected versions if available

        Returns:
            Dictionary with export statistics
        """
        cursor = self.store.conn.cursor()

        # Get transcripts in quality range
        cursor.execute("""
            SELECT * FROM transcripts
            WHERE quality_score >= ? AND quality_score <= ?
            ORDER BY severity DESC, quality_score ASC
        """, (min_quality_score, max_quality_score))

        transcripts = cursor.fetchall()

        # Build retraining examples
        retraining_examples = []

        for transcript in transcripts:
            transcript_id = transcript['id']
            annotations = self.store.get_annotations_for_transcript(transcript_id)

            # Use corrected version if available
            corrected_answer = None
            corrected_reasoning = None

            if include_corrections:
                for ann in annotations:
                    if ann.annotation_type == AnnotationType.CORRECTED_ANSWER:
                        corrected_answer = ann.content
                    elif ann.annotation_type == AnnotationType.CORRECTED_REASONING:
                        corrected_reasoning = ann.content

            # Build example in TunRex format
            example = {
                'question': transcript['question'],
                'answer': corrected_answer or transcript['expected_answer'],
            }

            # Add corrected reasoning as reference if available
            if corrected_reasoning:
                example['reference_response'] = corrected_reasoning

            retraining_examples.append(example)

        # Save in JSON Lines format (common for training data)
        output_dir = Path(output_path).parent
        output_dir.mkdir(parents=True, exist_ok=True)

        with open(output_path, 'w') as f:
            for example in retraining_examples:
                f.write(json.dumps(example) + '\n')

        return {
            'output_path': output_path,
            'num_examples': len(retraining_examples),
            'num_with_corrected_answer': len([e for e in retraining_examples if 'reference_response' in e]),
            'quality_range': [min_quality_score, max_quality_score],
        }

    def get_improvement_summary(self) -> Dict:
        """
        Get a summary of all improvement activities.

        Returns:
            Dictionary with improvement metrics
        """
        pending_suggestions = self.store.get_improvement_suggestions(status='pending')
        implemented_suggestions = self.store.get_improvement_suggestions(status='implemented')

        # Group by type
        pending_by_type = {}
        for sugg in pending_suggestions:
            stype = sugg['suggestion_type']
            pending_by_type[stype] = pending_by_type.get(stype, 0) + 1

        implemented_by_type = {}
        for sugg in implemented_suggestions:
            stype = sugg['suggestion_type']
            implemented_by_type[stype] = implemented_by_type.get(stype, 0) + 1

        # Get impact metrics
        impact = self.analytics.get_improvement_impact()

        return {
            'pending_suggestions': {
                'total': len(pending_suggestions),
                'by_type': pending_by_type,
            },
            'implemented_suggestions': {
                'total': len(implemented_suggestions),
                'by_type': implemented_by_type,
            },
            'impact_metrics': impact,
            'completion_rate': (
                len(implemented_suggestions) / (len(pending_suggestions) + len(implemented_suggestions))
                if (len(pending_suggestions) + len(implemented_suggestions)) > 0
                else 0
            ),
        }
