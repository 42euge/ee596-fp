"""
Grading Method Registry

Provides convenience functions to register all available grading methods
from the project for comparison.
"""

from typing import Dict, Callable
import sys
import os

# Handle both package and direct imports
if __package__:
    from .grading_comparison import GradingComparator
    from . import utils
else:
    # Add current directory to path for direct imports
    sys.path.insert(0, os.path.dirname(__file__))
    import grading_comparison
    import utils as utils_module
    GradingComparator = grading_comparison.GradingComparator
    utils = utils_module


def get_all_grading_methods() -> Dict[str, Dict]:
    """Get metadata for all available grading methods.

    Returns:
        Dictionary mapping method names to their metadata
    """
    methods = {
        'format_reward': {
            'function': utils.format_reward,
            'description': 'Rewards proper format usage (reasoning + answer tags). '
                          'Scores from -2 to +2.',
            'score_range': (-2.0, 2.0),
            'requires_ground_truth': False,
            'requires_rubric': False,
        },
        'accuracy_reward': {
            'function': utils.accuracy_reward,
            'description': 'Rewards answer accuracy for verifiable tasks (e.g., math). '
                          'Exact match gives 1.5, incorrect gives 0.0.',
            'score_range': (0.0, 1.5),
            'requires_ground_truth': True,
            'requires_rubric': False,
        },
        'rubric_reward': {
            'function': utils.rubric_reward,
            'description': 'Rubric-as-Reward (RaR) scoring. Combines rubric overlap (0-10), '
                          'reference similarity (0-5), and target score alignment (0-5). '
                          'Total range: 0-20.',
            'score_range': (0.0, 20.0),
            'requires_ground_truth': False,
            'requires_rubric': True,
        },
    }

    # Try to import TunRex methods if available
    try:
        from TunRex.src.tunrex.datasets import rewards as tunrex_rewards

        methods['match_format_exactly'] = {
            'function': tunrex_rewards.match_format_exactly,
            'description': 'Strict format matching. Returns 3.0 if format is exactly correct, '
                          '0.0 otherwise.',
            'score_range': (0.0, 3.0),
            'requires_ground_truth': False,
            'requires_rubric': False,
        }

        methods['match_format_approximately'] = {
            'function': tunrex_rewards.match_format_approximately,
            'description': 'Approximate format matching with partial credit. '
                          'Scores based on presence and position of format tags.',
            'score_range': (-2.5, 2.5),
            'requires_ground_truth': False,
            'requires_rubric': False,
        }

        methods['check_answer'] = {
            'function': tunrex_rewards.check_answer,
            'description': 'Answer checking with partial credit. Exact match: 3.0, '
                          'close match: 1.5, within 10% ratio: 0.5, wrong: -1.0.',
            'score_range': (-1.0, 3.0),
            'requires_ground_truth': True,
            'requires_rubric': False,
        }

        methods['check_numbers'] = {
            'function': tunrex_rewards.check_numbers,
            'description': 'Numerical answer extraction and comparison. '
                          'Exact match: 1.5, incorrect: 0.0.',
            'score_range': (0.0, 1.5),
            'requires_ground_truth': True,
            'requires_rubric': False,
        }

    except ImportError:
        pass  # TunRex methods not available

    return methods


def create_standard_comparator() -> GradingComparator:
    """Create a GradingComparator with all available methods registered.

    Returns:
        GradingComparator instance with all methods registered
    """
    comparator = GradingComparator()
    methods = get_all_grading_methods()

    for name, metadata in methods.items():
        comparator.register_method(
            name=name,
            function=metadata['function'],
            description=metadata['description'],
            score_range=metadata['score_range'],
            requires_ground_truth=metadata.get('requires_ground_truth', False),
            requires_rubric=metadata.get('requires_rubric', False),
        )

    return comparator


def compare_methods_quick(
    prompts,
    completions,
    method_names=None,
    **kwargs
):
    """Quick comparison of grading methods.

    Convenience function for one-off comparisons.

    Args:
        prompts: List of input prompts
        completions: List of model completions
        method_names: Specific methods to compare (None = all available)
        **kwargs: Additional arguments (answers, rubrics, etc.)

    Returns:
        ComparisonResults object

    Example:
        >>> results = compare_methods_quick(
        ...     prompts=["What is 2+2?"],
        ...     completions=["<reasoning>2 plus 2 is 4</reasoning><answer>4</answer>"],
        ...     method_names=['format_reward', 'accuracy_reward'],
        ...     answers=["4"]
        ... )
    """
    comparator = create_standard_comparator()
    return comparator.compare(prompts, completions, method_names, **kwargs)


def analyze_dataset(
    dataset,
    method_names=None,
    output_dir='./analysis',
    generate_plots=True,
):
    """Analyze a full dataset with multiple grading methods.

    Args:
        dataset: List of dicts with 'question' and optionally 'answer', 'rubric' keys
        method_names: Methods to compare (None = all applicable)
        output_dir: Directory to save analysis results
        generate_plots: Whether to generate visualization plots

    Returns:
        ComparisonResults object
    """
    import os
    from pathlib import Path

    # Create output directory
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Extract data
    prompts = [item['question'] for item in dataset]

    # Generate completions if not present
    if 'completion' in dataset[0]:
        completions = [item['completion'] for item in dataset]
    else:
        # Use reference response or answer as completion for testing
        completions = [
            item.get('reference_response', item.get('answer', ''))
            for item in dataset
        ]

    # Prepare kwargs
    kwargs = {}
    if 'answer' in dataset[0]:
        kwargs['answers'] = [item['answer'] for item in dataset]
    if 'rubric' in dataset[0]:
        kwargs['rubrics'] = [item['rubric'] for item in dataset]
    if 'reference_response' in dataset[0]:
        kwargs['reference_responses'] = [item['reference_response'] for item in dataset]
    if 'target_score' in dataset[0]:
        kwargs['target_scores'] = [item.get('target_score') for item in dataset]

    # Run comparison
    comparator = create_standard_comparator()
    results = comparator.compare(prompts, completions, method_names, **kwargs)

    # Save results
    results.save(str(output_path / 'comparison_results.json'))
    comparator.generate_report(str(output_path / 'comparison_report.md'))

    # Generate plots
    if generate_plots:
        try:
            comparator.plot_distributions(str(output_path / 'distributions.png'))
            comparator.plot_correlation_heatmap(str(output_path / 'correlations.png'))

            # Generate pairwise scatter plots for top correlations
            method_pairs = []
            for i, m1 in enumerate(results.methods):
                for m2 in results.methods[i+1:]:
                    method_pairs.append((m1, m2))

            for m1, m2 in method_pairs[:3]:  # Top 3 pairs
                filename = f'scatter_{m1}_vs_{m2}.png'
                comparator.plot_pairwise_scatter(m1, m2, str(output_path / filename))

        except Exception as e:
            print(f"Warning: Could not generate plots: {e}")

    print(f"\nAnalysis complete! Results saved to {output_dir}/")
    print(f"- comparison_results.json: Full results data")
    print(f"- comparison_report.md: Human-readable report")
    if generate_plots:
        print(f"- *.png: Visualization plots")

    return results
