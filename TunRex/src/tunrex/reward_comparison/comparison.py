"""Reward comparison utilities for comparing different reward methodologies."""

import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

from tunrex.reward_comparison.base import BaseReward, RewardResult


@dataclass
class ComparisonResult:
    """Results from comparing multiple reward evaluators.

    Attributes:
        evaluators: List of evaluator names
        results: Dict mapping evaluator name to RewardResult
        prompts: The prompts used for evaluation
        completions: The completions that were evaluated
        metadata: Additional metadata about the comparison
    """

    evaluators: List[str]
    results: Dict[str, RewardResult]
    prompts: List[str]
    completions: List[str]
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __post_init__(self):
        """Validate the comparison result."""
        if len(self.results) != len(self.evaluators):
            raise ValueError(
                f"Length mismatch: {len(self.evaluators)} evaluators vs "
                f"{len(self.results)} results"
            )

        # Check all results have same number of scores
        num_samples = len(self.completions)
        for name, result in self.results.items():
            if len(result.scores) != num_samples:
                raise ValueError(
                    f"Result for {name} has {len(result.scores)} scores, "
                    f"expected {num_samples}"
                )

    def get_scores_matrix(self) -> np.ndarray:
        """Get scores as a matrix (n_samples x n_evaluators).

        Returns:
            Numpy array of shape (n_samples, n_evaluators)
        """
        scores_list = [self.results[name].scores for name in self.evaluators]
        return np.array(scores_list).T

    def get_mean_scores(self) -> Dict[str, float]:
        """Get mean score for each evaluator.

        Returns:
            Dict mapping evaluator name to mean score
        """
        return {name: self.results[name].mean_score for name in self.evaluators}

    def get_std_scores(self) -> Dict[str, float]:
        """Get standard deviation for each evaluator.

        Returns:
            Dict mapping evaluator name to std score
        """
        return {name: self.results[name].std_score for name in self.evaluators}

    def get_evaluation_times(self) -> Dict[str, float]:
        """Get evaluation time for each evaluator.

        Returns:
            Dict mapping evaluator name to evaluation time (ms)
        """
        return {
            name: self.results[name].metadata.evaluation_time_ms
            for name in self.evaluators
        }

    def __repr__(self) -> str:
        return (
            f"ComparisonResult(n_evaluators={len(self.evaluators)}, "
            f"n_samples={len(self.completions)})"
        )


class RewardComparison:
    """Compare multiple reward evaluators on the same data.

    This class allows researchers to:
    - Run multiple reward methods on the same completions
    - Compare reward distributions
    - Analyze agreement and correlation
    - Identify edge cases where methods disagree

    Example:
        >>> from tunrex.reward_comparison import (
        ...     RewardComparison, ProgrammaticReward
        ... )
        >>>
        >>> # Create evaluators
        >>> format_reward = ProgrammaticReward(check_format, name="Format")
        >>> answer_reward = ProgrammaticReward(check_answer, name="Answer")
        >>>
        >>> # Compare them
        >>> comparison = RewardComparison([format_reward, answer_reward])
        >>> result = comparison.evaluate(prompts, completions, answer=answers)
        >>>
        >>> # Get summary
        >>> summary = comparison.summarize()
        >>> print(summary)
    """

    def __init__(
        self,
        evaluators: List[BaseReward],
        name: Optional[str] = None
    ):
        """Initialize reward comparison.

        Args:
            evaluators: List of reward evaluators to compare
            name: Optional name for this comparison
        """
        if len(evaluators) < 2:
            raise ValueError("Need at least 2 evaluators to compare")

        self.evaluators = evaluators
        self.name = name or "RewardComparison"
        self._last_result: Optional[ComparisonResult] = None

    def evaluate(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs
    ) -> ComparisonResult:
        """Evaluate all reward methods on the same data.

        Args:
            prompts: List of prompts
            completions: List of completions
            **kwargs: Additional metadata (e.g., answers, questions)

        Returns:
            ComparisonResult containing all evaluations
        """
        if len(prompts) != len(completions):
            raise ValueError(
                f"Length mismatch: {len(prompts)} prompts vs "
                f"{len(completions)} completions"
            )

        print(f"Running {len(self.evaluators)} reward evaluators on "
              f"{len(completions)} samples...")

        results = {}
        evaluator_names = []

        for i, evaluator in enumerate(self.evaluators):
            print(f"  [{i+1}/{len(self.evaluators)}] Evaluating {evaluator.name}...")
            result = evaluator.evaluate(prompts, completions, **kwargs)
            results[evaluator.name] = result
            evaluator_names.append(evaluator.name)

        # Create comparison result
        self._last_result = ComparisonResult(
            evaluators=evaluator_names,
            results=results,
            prompts=prompts,
            completions=completions,
            metadata=kwargs
        )

        print(f"✓ Comparison complete!")
        return self._last_result

    def evaluate_parallel(
        self,
        prompts: List[str],
        completions: List[str],
        **kwargs
    ) -> ComparisonResult:
        """Evaluate all reward methods in parallel (if possible).

        Note: This is a placeholder for future parallel implementation.
        Currently falls back to sequential evaluation.

        Args:
            prompts: List of prompts
            completions: List of completions
            **kwargs: Additional metadata

        Returns:
            ComparisonResult containing all evaluations
        """
        # TODO: Implement parallel evaluation using threading/multiprocessing
        return self.evaluate(prompts, completions, **kwargs)

    def summarize(self, result: Optional[ComparisonResult] = None) -> str:
        """Generate a text summary of the comparison.

        Args:
            result: ComparisonResult to summarize (uses last result if None)

        Returns:
            Formatted summary string
        """
        if result is None:
            result = self._last_result
            if result is None:
                raise ValueError("No comparison result available. Run evaluate() first.")

        lines = []
        lines.append("=" * 70)
        lines.append(f"Reward Comparison: {self.name}")
        lines.append("=" * 70)
        lines.append(f"Samples evaluated: {len(result.completions)}")
        lines.append(f"Evaluators: {len(result.evaluators)}")
        lines.append("")

        # Summary table
        lines.append("Evaluator Performance:")
        lines.append("-" * 70)
        lines.append(f"{'Evaluator':<25} {'Type':<18} {'Mean':<10} {'Std':<10} {'Time (ms)':<10}")
        lines.append("-" * 70)

        for name in result.evaluators:
            res = result.results[name]
            evaluator_type = res.metadata.evaluator_type
            mean_score = res.mean_score
            std_score = res.std_score
            eval_time = res.metadata.evaluation_time_ms

            lines.append(
                f"{name:<25} {evaluator_type:<18} "
                f"{mean_score:<10.3f} {std_score:<10.3f} {eval_time:<10.1f}"
            )

        lines.append("=" * 70)
        return "\n".join(lines)

    def find_disagreements(
        self,
        result: Optional[ComparisonResult] = None,
        threshold: float = 0.5,
        evaluator_pairs: Optional[List[Tuple[str, str]]] = None
    ) -> List[Dict[str, Any]]:
        """Find samples where evaluators disagree significantly.

        Args:
            result: ComparisonResult to analyze
            threshold: Score difference threshold for disagreement
            evaluator_pairs: Optional list of (eval1, eval2) pairs to compare.
                           If None, compares all pairs.

        Returns:
            List of disagreement cases with details
        """
        if result is None:
            result = self._last_result
            if result is None:
                raise ValueError("No comparison result available.")

        disagreements = []

        if evaluator_pairs is None:
            # Compare all pairs
            evaluator_pairs = [
                (result.evaluators[i], result.evaluators[j])
                for i in range(len(result.evaluators))
                for j in range(i + 1, len(result.evaluators))
            ]

        for eval1_name, eval2_name in evaluator_pairs:
            eval1_scores = result.results[eval1_name].scores
            eval2_scores = result.results[eval2_name].scores

            for idx, (score1, score2) in enumerate(zip(eval1_scores, eval2_scores)):
                diff = abs(score1 - score2)
                if diff >= threshold:
                    disagreements.append({
                        "sample_idx": idx,
                        "prompt": result.prompts[idx],
                        "completion": result.completions[idx],
                        "evaluator1": eval1_name,
                        "score1": score1,
                        "evaluator2": eval2_name,
                        "score2": score2,
                        "difference": diff,
                    })

        # Sort by difference (largest first)
        disagreements.sort(key=lambda x: x["difference"], reverse=True)
        return disagreements

    def get_score_ranges(
        self,
        result: Optional[ComparisonResult] = None
    ) -> Dict[str, Tuple[float, float]]:
        """Get min and max scores for each evaluator.

        Args:
            result: ComparisonResult to analyze

        Returns:
            Dict mapping evaluator name to (min, max) tuple
        """
        if result is None:
            result = self._last_result
            if result is None:
                raise ValueError("No comparison result available.")

        ranges = {}
        for name in result.evaluators:
            scores = result.results[name].scores
            ranges[name] = (min(scores), max(scores))

        return ranges

    def export_to_csv(
        self,
        filepath: str,
        result: Optional[ComparisonResult] = None,
        include_text: bool = False
    ) -> None:
        """Export comparison results to CSV.

        Args:
            filepath: Path to save CSV file
            result: ComparisonResult to export
            include_text: Whether to include prompt/completion text
        """
        import csv

        if result is None:
            result = self._last_result
            if result is None:
                raise ValueError("No comparison result available.")

        with open(filepath, 'w', newline='', encoding='utf-8') as f:
            fieldnames = ['sample_idx']

            if include_text:
                fieldnames.extend(['prompt', 'completion'])

            # Add score columns
            for name in result.evaluators:
                fieldnames.append(f"{name}_score")

            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()

            for idx in range(len(result.completions)):
                row = {'sample_idx': idx}

                if include_text:
                    row['prompt'] = result.prompts[idx]
                    row['completion'] = result.completions[idx]

                for name in result.evaluators:
                    row[f"{name}_score"] = result.results[name].scores[idx]

                writer.writerow(row)

        print(f"✓ Exported comparison to {filepath}")
