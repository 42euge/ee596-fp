"""
Grading Methodology Comparison Framework

This module provides tools for comparing different grading/reward methodologies
and analyzing their effects on model behavior during GRPO training.

Key features:
- Compare multiple grading methods on the same dataset
- Statistical analysis of reward distributions
- Correlation analysis between methods
- Identify agreement/disagreement patterns
- Visualize differences and generate reports
"""

import json
import numpy as np
from typing import List, Dict, Callable, Any, Optional, Tuple
from dataclasses import dataclass, field, asdict
from collections import defaultdict
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats


@dataclass
class GradingMethod:
    """Represents a single grading methodology."""
    name: str
    function: Callable
    description: str
    score_range: Tuple[float, float]
    requires_ground_truth: bool = False
    requires_rubric: bool = False


@dataclass
class ComparisonResults:
    """Stores results from comparing multiple grading methods."""
    methods: List[str] = field(default_factory=list)
    scores: Dict[str, List[float]] = field(default_factory=dict)
    statistics: Dict[str, Dict[str, float]] = field(default_factory=dict)
    correlations: Dict[str, float] = field(default_factory=dict)
    agreement_matrix: Optional[np.ndarray] = None
    examples: List[Dict[str, Any]] = field(default_factory=list)

    def to_dict(self):
        """Convert to dictionary for JSON serialization."""
        result = asdict(self)
        if self.agreement_matrix is not None:
            result['agreement_matrix'] = self.agreement_matrix.tolist()
        return result

    def save(self, filepath: str):
        """Save results to JSON file."""
        with open(filepath, 'w') as f:
            json.dump(self.to_dict(), f, indent=2)

    @classmethod
    def load(cls, filepath: str):
        """Load results from JSON file."""
        with open(filepath, 'r') as f:
            data = json.load(f)
        if 'agreement_matrix' in data and data['agreement_matrix'] is not None:
            data['agreement_matrix'] = np.array(data['agreement_matrix'])
        return cls(**data)


class GradingComparator:
    """Framework for comparing different grading methodologies."""

    def __init__(self):
        self.methods: Dict[str, GradingMethod] = {}
        self.results: Optional[ComparisonResults] = None

    def register_method(
        self,
        name: str,
        function: Callable,
        description: str,
        score_range: Tuple[float, float],
        requires_ground_truth: bool = False,
        requires_rubric: bool = False,
    ):
        """Register a grading method for comparison.

        Args:
            name: Unique identifier for the method
            function: Callable that takes (prompts, completions, **kwargs) and returns scores
            description: Human-readable description of the method
            score_range: Tuple of (min, max) possible scores
            requires_ground_truth: Whether method needs ground truth answers
            requires_rubric: Whether method needs rubric information
        """
        self.methods[name] = GradingMethod(
            name=name,
            function=function,
            description=description,
            score_range=score_range,
            requires_ground_truth=requires_ground_truth,
            requires_rubric=requires_rubric,
        )

    def compare(
        self,
        prompts: List[str],
        completions: List[str],
        method_names: Optional[List[str]] = None,
        **kwargs
    ) -> ComparisonResults:
        """Run comparison across multiple grading methods.

        Args:
            prompts: List of input prompts
            completions: List of model completions
            method_names: Specific methods to compare (None = all registered)
            **kwargs: Additional arguments (answers, rubrics, etc.)

        Returns:
            ComparisonResults object with scores, statistics, and analysis
        """
        if method_names is None:
            method_names = list(self.methods.keys())

        results = ComparisonResults()
        results.methods = method_names

        # Store examples for detailed analysis
        for i, (prompt, completion) in enumerate(zip(prompts, completions)):
            example = {
                'id': i,
                'prompt': prompt,
                'completion': completion,
                'scores': {}
            }
            results.examples.append(example)

        # Run each grading method
        for name in method_names:
            if name not in self.methods:
                print(f"Warning: Method '{name}' not registered, skipping")
                continue

            method = self.methods[name]

            # Check requirements
            if method.requires_ground_truth and 'answers' not in kwargs:
                print(f"Warning: Method '{name}' requires ground truth, skipping")
                continue
            if method.requires_rubric and 'rubrics' not in kwargs:
                print(f"Warning: Method '{name}' requires rubrics, skipping")
                continue

            try:
                # Run the grading method
                scores = method.function(prompts, completions, **kwargs)
                results.scores[name] = scores

                # Store scores in examples
                for i, score in enumerate(scores):
                    results.examples[i]['scores'][name] = score

                # Calculate statistics
                results.statistics[name] = self._calculate_statistics(scores)

            except Exception as e:
                print(f"Error running method '{name}': {e}")
                continue

        # Calculate correlations
        results.correlations = self._calculate_correlations(results.scores)

        # Calculate agreement matrix
        results.agreement_matrix = self._calculate_agreement_matrix(results.scores)

        self.results = results
        return results

    def _calculate_statistics(self, scores: List[float]) -> Dict[str, float]:
        """Calculate statistical metrics for a set of scores."""
        scores_array = np.array(scores)
        return {
            'mean': float(np.mean(scores_array)),
            'median': float(np.median(scores_array)),
            'std': float(np.std(scores_array)),
            'min': float(np.min(scores_array)),
            'max': float(np.max(scores_array)),
            'q25': float(np.percentile(scores_array, 25)),
            'q75': float(np.percentile(scores_array, 75)),
            'count': len(scores),
        }

    def _calculate_correlations(self, scores_dict: Dict[str, List[float]]) -> Dict[str, float]:
        """Calculate pairwise correlations between methods."""
        correlations = {}
        method_names = list(scores_dict.keys())

        for i, name1 in enumerate(method_names):
            for name2 in method_names[i+1:]:
                scores1 = np.array(scores_dict[name1])
                scores2 = np.array(scores_dict[name2])

                # Pearson correlation
                pearson_r, _ = stats.pearsonr(scores1, scores2)
                correlations[f'{name1}_vs_{name2}_pearson'] = float(pearson_r)

                # Spearman rank correlation
                spearman_r, _ = stats.spearmanr(scores1, scores2)
                correlations[f'{name1}_vs_{name2}_spearman'] = float(spearman_r)

        return correlations

    def _calculate_agreement_matrix(self, scores_dict: Dict[str, List[float]]) -> np.ndarray:
        """Calculate pairwise agreement matrix.

        Agreement is measured as the proportion of examples where both methods
        give scores in the same quartile.
        """
        if not scores_dict:
            return np.array([])

        method_names = list(scores_dict.keys())
        n_methods = len(method_names)
        agreement_matrix = np.zeros((n_methods, n_methods))

        # Calculate quartile assignments for each method
        quartiles = {}
        for name, scores in scores_dict.items():
            scores_array = np.array(scores)
            q25, q50, q75 = np.percentile(scores_array, [25, 50, 75])
            quartile_assignment = np.zeros(len(scores), dtype=int)
            quartile_assignment[scores_array >= q75] = 3
            quartile_assignment[(scores_array >= q50) & (scores_array < q75)] = 2
            quartile_assignment[(scores_array >= q25) & (scores_array < q50)] = 1
            quartiles[name] = quartile_assignment

        # Calculate agreement
        for i, name1 in enumerate(method_names):
            for j, name2 in enumerate(method_names):
                if i == j:
                    agreement_matrix[i, j] = 1.0
                else:
                    agreement = np.mean(quartiles[name1] == quartiles[name2])
                    agreement_matrix[i, j] = agreement

        return agreement_matrix

    def find_disagreements(
        self,
        method1: str,
        method2: str,
        top_k: int = 10,
        normalize: bool = True
    ) -> List[Dict[str, Any]]:
        """Find examples where two methods disagree most.

        Args:
            method1: First method name
            method2: Second method name
            top_k: Number of top disagreements to return
            normalize: Normalize scores to [0,1] before comparing

        Returns:
            List of examples with largest score differences
        """
        if self.results is None:
            raise ValueError("No comparison results available. Run compare() first.")

        if method1 not in self.results.scores or method2 not in self.results.scores:
            raise ValueError(f"Methods must be in {list(self.results.scores.keys())}")

        scores1 = np.array(self.results.scores[method1])
        scores2 = np.array(self.results.scores[method2])

        # Normalize if requested
        if normalize:
            m1 = self.methods[method1]
            m2 = self.methods[method2]
            scores1 = (scores1 - m1.score_range[0]) / (m1.score_range[1] - m1.score_range[0])
            scores2 = (scores2 - m2.score_range[0]) / (m2.score_range[1] - m2.score_range[0])

        # Calculate absolute differences
        differences = np.abs(scores1 - scores2)

        # Get top-k indices
        top_indices = np.argsort(differences)[-top_k:][::-1]

        disagreements = []
        for idx in top_indices:
            example = self.results.examples[idx].copy()
            example['difference'] = float(differences[idx])
            example[f'{method1}_score'] = float(scores1[idx])
            example[f'{method2}_score'] = float(scores2[idx])
            disagreements.append(example)

        return disagreements

    def analyze_behavior_effects(
        self,
        method_name: str,
        thresholds: Optional[List[float]] = None
    ) -> Dict[str, Any]:
        """Analyze how a grading method affects model behavior.

        Groups completions by score ranges and analyzes patterns.

        Args:
            method_name: Name of the method to analyze
            thresholds: Score thresholds for grouping (default: quartiles)

        Returns:
            Dictionary with behavior analysis by score range
        """
        if self.results is None:
            raise ValueError("No comparison results available. Run compare() first.")

        if method_name not in self.results.scores:
            raise ValueError(f"Method '{method_name}' not in results")

        scores = np.array(self.results.scores[method_name])

        # Default to quartiles
        if thresholds is None:
            thresholds = [
                np.percentile(scores, 25),
                np.percentile(scores, 50),
                np.percentile(scores, 75)
            ]

        # Group examples by score ranges
        groups = {
            'low': [],
            'mid_low': [],
            'mid_high': [],
            'high': []
        }

        for i, score in enumerate(scores):
            example = self.results.examples[i]
            if score < thresholds[0]:
                groups['low'].append(example)
            elif score < thresholds[1]:
                groups['mid_low'].append(example)
            elif score < thresholds[2]:
                groups['mid_high'].append(example)
            else:
                groups['high'].append(example)

        # Analyze each group
        analysis = {}
        for group_name, examples in groups.items():
            if not examples:
                continue

            completions = [ex['completion'] for ex in examples]

            analysis[group_name] = {
                'count': len(examples),
                'avg_length': np.mean([len(c) for c in completions]),
                'avg_word_count': np.mean([len(c.split()) for c in completions]),
                'examples': examples[:3]  # Sample examples
            }

        return analysis

    def plot_distributions(self, save_path: Optional[str] = None):
        """Plot score distributions for all methods.

        Args:
            save_path: Path to save the figure (None = display only)
        """
        if self.results is None:
            raise ValueError("No comparison results available. Run compare() first.")

        n_methods = len(self.results.methods)
        fig, axes = plt.subplots(1, n_methods, figsize=(5*n_methods, 4))

        if n_methods == 1:
            axes = [axes]

        for ax, method_name in zip(axes, self.results.methods):
            scores = self.results.scores[method_name]
            stats_dict = self.results.statistics[method_name]

            ax.hist(scores, bins=30, alpha=0.7, edgecolor='black')
            ax.axvline(stats_dict['mean'], color='red', linestyle='--',
                      label=f'Mean: {stats_dict["mean"]:.2f}')
            ax.axvline(stats_dict['median'], color='green', linestyle='--',
                      label=f'Median: {stats_dict["median"]:.2f}')
            ax.set_xlabel('Score')
            ax.set_ylabel('Frequency')
            ax.set_title(method_name)
            ax.legend()
            ax.grid(alpha=0.3)

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Distribution plot saved to {save_path}")
        else:
            plt.show()

    def plot_correlation_heatmap(self, save_path: Optional[str] = None):
        """Plot correlation heatmap between methods.

        Args:
            save_path: Path to save the figure (None = display only)
        """
        if self.results is None:
            raise ValueError("No comparison results available. Run compare() first.")

        method_names = self.results.methods
        n_methods = len(method_names)

        # Build correlation matrix
        corr_matrix = np.zeros((n_methods, n_methods))
        for i, name1 in enumerate(method_names):
            for j, name2 in enumerate(method_names):
                if i == j:
                    corr_matrix[i, j] = 1.0
                else:
                    key = f'{name1}_vs_{name2}_pearson'
                    reverse_key = f'{name2}_vs_{name1}_pearson'
                    if key in self.results.correlations:
                        corr_matrix[i, j] = self.results.correlations[key]
                    elif reverse_key in self.results.correlations:
                        corr_matrix[i, j] = self.results.correlations[reverse_key]

        # Plot heatmap
        plt.figure(figsize=(10, 8))
        sns.heatmap(corr_matrix, annot=True, fmt='.3f', cmap='coolwarm',
                   xticklabels=method_names, yticklabels=method_names,
                   vmin=-1, vmax=1, center=0)
        plt.title('Pearson Correlation Between Grading Methods')
        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Correlation heatmap saved to {save_path}")
        else:
            plt.show()

    def plot_pairwise_scatter(
        self,
        method1: str,
        method2: str,
        save_path: Optional[str] = None
    ):
        """Plot scatter plot comparing two methods.

        Args:
            method1: First method name
            method2: Second method name
            save_path: Path to save the figure (None = display only)
        """
        if self.results is None:
            raise ValueError("No comparison results available. Run compare() first.")

        if method1 not in self.results.scores or method2 not in self.results.scores:
            raise ValueError(f"Methods must be in {list(self.results.scores.keys())}")

        scores1 = self.results.scores[method1]
        scores2 = self.results.scores[method2]

        # Get correlation
        key = f'{method1}_vs_{method2}_pearson'
        reverse_key = f'{method2}_vs_{method1}_pearson'
        if key in self.results.correlations:
            corr = self.results.correlations[key]
        elif reverse_key in self.results.correlations:
            corr = self.results.correlations[reverse_key]
        else:
            corr = np.corrcoef(scores1, scores2)[0, 1]

        # Plot
        plt.figure(figsize=(8, 8))
        plt.scatter(scores1, scores2, alpha=0.5, s=20)
        plt.xlabel(method1)
        plt.ylabel(method2)
        plt.title(f'{method1} vs {method2}\nPearson r = {corr:.3f}')
        plt.grid(alpha=0.3)

        # Add diagonal line
        all_scores = scores1 + scores2
        min_score, max_score = min(all_scores), max(all_scores)
        plt.plot([min_score, max_score], [min_score, max_score],
                'r--', alpha=0.5, label='Perfect agreement')
        plt.legend()

        plt.tight_layout()

        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Scatter plot saved to {save_path}")
        else:
            plt.show()

    def generate_report(self, output_path: str):
        """Generate a comprehensive comparison report.

        Args:
            output_path: Path to save the report (markdown format)
        """
        if self.results is None:
            raise ValueError("No comparison results available. Run compare() first.")

        report_lines = [
            "# Grading Methodology Comparison Report\n",
            f"**Number of examples analyzed:** {len(self.results.examples)}\n",
            f"**Number of methods compared:** {len(self.results.methods)}\n",
            "\n## Methods Compared\n"
        ]

        for name in self.results.methods:
            method = self.methods[name]
            report_lines.append(f"### {name}\n")
            report_lines.append(f"- **Description:** {method.description}\n")
            report_lines.append(f"- **Score range:** {method.score_range}\n")
            report_lines.append(f"- **Requires ground truth:** {method.requires_ground_truth}\n")
            report_lines.append(f"- **Requires rubric:** {method.requires_rubric}\n\n")

        report_lines.append("\n## Statistical Summary\n\n")
        report_lines.append("| Method | Mean | Median | Std | Min | Max | Q25 | Q75 |\n")
        report_lines.append("|--------|------|--------|-----|-----|-----|-----|-----|\n")

        for name in self.results.methods:
            stats = self.results.statistics[name]
            report_lines.append(
                f"| {name} | {stats['mean']:.3f} | {stats['median']:.3f} | "
                f"{stats['std']:.3f} | {stats['min']:.3f} | {stats['max']:.3f} | "
                f"{stats['q25']:.3f} | {stats['q75']:.3f} |\n"
            )

        report_lines.append("\n## Correlations\n\n")
        report_lines.append("### Pearson Correlations\n\n")

        pearson_corrs = {k: v for k, v in self.results.correlations.items()
                        if 'pearson' in k}
        for pair, corr in sorted(pearson_corrs.items(), key=lambda x: abs(x[1]), reverse=True):
            methods = pair.replace('_pearson', '').replace('_vs_', ' vs ')
            report_lines.append(f"- **{methods}:** {corr:.3f}\n")

        report_lines.append("\n### Spearman Rank Correlations\n\n")

        spearman_corrs = {k: v for k, v in self.results.correlations.items()
                         if 'spearman' in k}
        for pair, corr in sorted(spearman_corrs.items(), key=lambda x: abs(x[1]), reverse=True):
            methods = pair.replace('_spearman', '').replace('_vs_', ' vs ')
            report_lines.append(f"- **{methods}:** {corr:.3f}\n")

        # Write report
        with open(output_path, 'w') as f:
            f.writelines(report_lines)

        print(f"Report saved to {output_path}")
