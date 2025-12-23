"""Metrics computation and formatting utilities."""

from typing import Dict, List, Any
from .benchmarks.base import EvaluationResult, SampleResult


def compute_metrics(results: List[SampleResult]) -> Dict[str, float]:
    """Compute aggregate metrics from sample results.

    Args:
        results: List of sample results

    Returns:
        Dictionary of metrics
    """
    if not results:
        return {
            "accuracy": 0.0,
            "partial_accuracy": 0.0,
            "format_accuracy": 0.0,
            "avg_generation_time": 0.0,
        }

    num_samples = len(results)
    correct = sum(1 for r in results if r.is_correct)
    format_correct = sum(1 for r in results if r.format_correct)
    total_time = sum(r.generation_time for r in results)

    # Compute partial correctness (within 10% for numerical answers)
    partial_correct = correct
    for r in results:
        if not r.is_correct:
            if isinstance(r.predicted_answer, (int, float)) and isinstance(r.gold_answer, (int, float)):
                if r.gold_answer != 0:
                    ratio = r.predicted_answer / r.gold_answer
                    if 0.9 <= ratio <= 1.1:
                        partial_correct += 1

    return {
        "accuracy": correct / num_samples,
        "partial_accuracy": partial_correct / num_samples,
        "format_accuracy": format_correct / num_samples,
        "avg_generation_time": total_time / num_samples,
        "total_time": total_time,
        "num_samples": num_samples,
        "num_correct": correct,
        "num_partial_correct": partial_correct,
        "num_format_correct": format_correct,
    }


def format_evaluation_results(result: EvaluationResult, verbose: bool = True) -> str:
    """Format evaluation results as a string.

    Args:
        result: Evaluation result
        verbose: Include per-sample details

    Returns:
        Formatted string
    """
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append(f"Benchmark: {result.benchmark_name}")
    lines.append(f"Samples: {result.num_samples}")
    lines.append(f"{'='*70}")

    # Main metrics
    lines.append("\nMetrics:")
    lines.append(f"  Accuracy:         {result.metrics['accuracy']:.1%}")
    lines.append(f"  Partial Accuracy: {result.metrics['partial_accuracy']:.1%}")
    lines.append(f"  Format Accuracy:  {result.metrics['format_accuracy']:.1%}")
    lines.append(f"  Avg Gen Time:     {result.metrics['avg_generation_time']:.2f}s")

    # Additional metrics if available
    for key, value in result.metrics.items():
        if key not in ['accuracy', 'partial_accuracy', 'format_accuracy', 'avg_generation_time']:
            if isinstance(value, float):
                lines.append(f"  {key}: {value:.3f}")
            else:
                lines.append(f"  {key}: {value}")

    if verbose and result.per_sample_results:
        lines.append(f"\n{'='*70}")
        lines.append("Sample Results (first 10):")
        lines.append(f"{'='*70}")

        for i, sample in enumerate(result.per_sample_results[:10]):
            lines.append(f"\n[{i+1}] {sample.sample_id}")
            lines.append(f"  Question: {sample.question[:100]}...")
            lines.append(f"  Gold:     {sample.gold_answer}")
            lines.append(f"  Predicted: {sample.predicted_answer}")
            lines.append(f"  Correct:  {sample.is_correct} | Format: {sample.format_correct}")
            lines.append(f"  Time:     {sample.generation_time:.2f}s")

        if len(result.per_sample_results) > 10:
            lines.append(f"\n... and {len(result.per_sample_results) - 10} more samples")

    lines.append(f"{'='*70}\n")

    return "\n".join(lines)


def compare_results(results: Dict[str, EvaluationResult]) -> str:
    """Compare results across multiple benchmarks.

    Args:
        results: Dictionary mapping benchmark names to results

    Returns:
        Formatted comparison string
    """
    lines = []
    lines.append(f"\n{'='*70}")
    lines.append("Multi-Benchmark Comparison")
    lines.append(f"{'='*70}")

    # Create table header
    lines.append(f"\n{'Benchmark':<20} {'Accuracy':<12} {'Partial':<12} {'Format':<12}")
    lines.append(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12}")

    # Add rows
    for benchmark_name, result in results.items():
        lines.append(
            f"{benchmark_name:<20} "
            f"{result.metrics['accuracy']:>10.1%}  "
            f"{result.metrics['partial_accuracy']:>10.1%}  "
            f"{result.metrics['format_accuracy']:>10.1%}"
        )

    # Compute averages
    if results:
        avg_acc = sum(r.metrics['accuracy'] for r in results.values()) / len(results)
        avg_partial = sum(r.metrics['partial_accuracy'] for r in results.values()) / len(results)
        avg_format = sum(r.metrics['format_accuracy'] for r in results.values()) / len(results)

        lines.append(f"{'-'*20} {'-'*12} {'-'*12} {'-'*12}")
        lines.append(
            f"{'Average':<20} "
            f"{avg_acc:>10.1%}  "
            f"{avg_partial:>10.1%}  "
            f"{avg_format:>10.1%}"
        )

    lines.append(f"{'='*70}\n")

    return "\n".join(lines)
