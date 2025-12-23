"""
Rubric Reporter - Generate reports and visualizations for rubric evaluation

This module provides reporting and visualization tools for rubric testing results.
"""

from typing import Dict, List, Optional, Any
from pathlib import Path
import json

from .evaluator import EvaluationResult
from .comparator import ComparisonResult


class RubricReporter:
    """
    Generate reports and visualizations for rubric evaluation.

    This class provides:
    - Markdown report generation
    - Performance visualizations
    - Comparison tables
    - Export to various formats

    Example:
        reporter = RubricReporter()
        reporter.generate_report(results, output_path="rubric_report.md")
    """

    def __init__(self, output_dir: Optional[str] = None):
        self.output_dir = Path(output_dir) if output_dir else Path("./rubric_reports")
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def generate_report(
        self,
        results: List[EvaluationResult],
        comparison: Optional[ComparisonResult] = None,
        output_path: Optional[str] = None,
        format: str = "markdown",
    ) -> str:
        """
        Generate a comprehensive report.

        Args:
            results: List of evaluation results
            comparison: Optional comparison result
            output_path: Path to save report (if None, prints to stdout)
            format: Report format ("markdown", "json", "html")

        Returns:
            Report content as string
        """
        if format == "markdown":
            report = self._generate_markdown_report(results, comparison)
        elif format == "json":
            report = self._generate_json_report(results, comparison)
        elif format == "html":
            report = self._generate_html_report(results, comparison)
        else:
            raise ValueError(f"Unsupported format: {format}")

        if output_path:
            with open(output_path, "w") as f:
                f.write(report)
            print(f"Report saved to {output_path}")

        return report

    def _generate_markdown_report(
        self,
        results: List[EvaluationResult],
        comparison: Optional[ComparisonResult] = None,
    ) -> str:
        """Generate a markdown report"""
        lines = ["# Rubric Evaluation Report\n"]

        # Summary section
        lines.append("## Summary\n")
        lines.append(f"- **Total Rubrics Evaluated**: {len(results)}")
        if results:
            lines.append(f"- **Samples per Rubric**: {results[0].num_samples}")
            lines.append(f"- **Dataset**: {results[0].config.dataset_name}")
        lines.append("")

        # Comparison section
        if comparison:
            lines.append("## Comparison Results\n")
            lines.append(f"**Best Rubric**: {comparison.best_rubric} (score: {comparison.best_score:.2f})\n")

            lines.append("### Rankings\n")
            for rubric, rank in sorted(comparison.rankings.items(), key=lambda x: x[1]):
                result = next(r for r in results if r.rubric_name == rubric)
                lines.append(
                    f"{rank}. **{rubric}**: {result.mean_score:.2f} ± {result.std_score:.2f} "
                    f"(relative: {comparison.relative_performance[rubric]:.2%})"
                )
            lines.append("")

            # Statistical tests
            if "pairwise_ttests" in comparison.statistical_tests:
                lines.append("### Statistical Significance (Pairwise t-tests)\n")
                ttests = comparison.statistical_tests["pairwise_ttests"]
                for pair, test_result in ttests.items():
                    if "error" not in test_result:
                        sig = "✓" if test_result["significant"] else "✗"
                        lines.append(
                            f"- {pair}: p={test_result['p_value']:.4f} "
                            f"(diff={test_result['mean_diff']:.2f}) {sig}"
                        )
                lines.append("")

            if "anova" in comparison.statistical_tests:
                anova = comparison.statistical_tests["anova"]
                if "error" not in anova:
                    lines.append("### ANOVA Test\n")
                    lines.append(f"- F-statistic: {anova['f_statistic']:.4f}")
                    lines.append(f"- p-value: {anova['p_value']:.4f}")
                    lines.append(f"- **{anova['interpretation']}**\n")

        # Individual rubric details
        lines.append("## Individual Rubric Results\n")
        for result in results:
            lines.append(f"### {result.rubric_name}\n")
            lines.append(f"- **Mean Score**: {result.mean_score:.2f}")
            lines.append(f"- **Std Dev**: {result.std_score:.2f}")
            lines.append(f"- **Median**: {result.median_score:.2f}")
            lines.append(f"- **Range**: [{result.min_score:.2f}, {result.max_score:.2f}]")
            lines.append(f"- **Samples**: {result.num_samples}")
            lines.append(f"- **Time**: {result.total_time:.2f}s ({result.time_per_sample:.3f}s/sample)")

            if result.component_stats:
                lines.append("\n**Component Statistics**:\n")
                for comp, stats in result.component_stats.items():
                    lines.append(
                        f"- {comp}: {stats['mean']:.2f} ± {stats['std']:.2f} "
                        f"(range: [{stats['min']:.2f}, {stats['max']:.2f}])"
                    )

            lines.append("")

        # Configuration details
        if results:
            lines.append("## Configuration\n")
            config = results[0].config
            lines.append(f"- **Model**: {config.model_name}")
            lines.append(f"- **Temperature**: {config.temperature}")
            lines.append(f"- **Max Length**: {config.max_length}")
            lines.append(f"- **Quantization**: {config.quantization or 'None'}")
            if config.use_lora and config.lora_checkpoint:
                lines.append(f"- **LoRA Checkpoint**: {config.lora_checkpoint}")
            lines.append("")

        return "\n".join(lines)

    def _generate_json_report(
        self,
        results: List[EvaluationResult],
        comparison: Optional[ComparisonResult] = None,
    ) -> str:
        """Generate a JSON report"""
        report_data = {
            "summary": {
                "total_rubrics": len(results),
                "num_samples": results[0].num_samples if results else 0,
            },
            "results": [r.summary() for r in results],
        }

        if comparison:
            report_data["comparison"] = comparison.summary()

        return json.dumps(report_data, indent=2)

    def _generate_html_report(
        self,
        results: List[EvaluationResult],
        comparison: Optional[ComparisonResult] = None,
    ) -> str:
        """Generate an HTML report"""
        # Convert markdown to HTML
        markdown_report = self._generate_markdown_report(results, comparison)

        html = f"""<!DOCTYPE html>
<html>
<head>
    <meta charset="UTF-8">
    <title>Rubric Evaluation Report</title>
    <style>
        body {{
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, 'Helvetica Neue', Arial, sans-serif;
            max-width: 1200px;
            margin: 0 auto;
            padding: 20px;
            line-height: 1.6;
        }}
        h1, h2, h3 {{ color: #333; }}
        table {{ border-collapse: collapse; width: 100%; margin: 20px 0; }}
        th, td {{ border: 1px solid #ddd; padding: 12px; text-align: left; }}
        th {{ background-color: #f5f5f5; }}
        .metric {{ font-weight: bold; color: #0066cc; }}
        .significant {{ color: #00aa00; }}
        .not-significant {{ color: #999; }}
    </style>
</head>
<body>
    <pre>{markdown_report}</pre>
</body>
</html>
"""
        return html


def generate_report(
    results: List[EvaluationResult],
    comparison: Optional[ComparisonResult] = None,
    output_path: Optional[str] = None,
    format: str = "markdown",
) -> str:
    """
    Convenience function to generate a report.

    Args:
        results: List of evaluation results
        comparison: Optional comparison result
        output_path: Path to save report
        format: Report format ("markdown", "json", "html")

    Returns:
        Report content as string
    """
    reporter = RubricReporter()
    return reporter.generate_report(results, comparison, output_path, format)


def plot_rubric_comparison(
    results: List[EvaluationResult],
    output_path: Optional[str] = None,
    metric: str = "mean_score",
) -> None:
    """
    Create a visualization comparing rubric scores.

    Args:
        results: List of evaluation results
        output_path: Path to save plot (if None, displays)
        metric: Metric to visualize

    Note: Requires matplotlib to be installed
    """
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("Warning: matplotlib not installed. Install with: pip install matplotlib")
        return

    # Extract data
    rubric_names = [r.rubric_name for r in results]
    means = [getattr(r, metric) for r in results]
    stds = [r.std_score for r in results]

    # Create bar plot
    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(rubric_names))
    bars = ax.bar(x, means, yerr=stds, capsize=5, alpha=0.7)

    # Color bars by rank
    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(rubric_names)))
    sorted_indices = np.argsort(means)[::-1]
    for i, idx in enumerate(sorted_indices):
        bars[idx].set_color(colors[i])

    ax.set_xlabel('Rubric', fontsize=12)
    ax.set_ylabel(metric.replace('_', ' ').title(), fontsize=12)
    ax.set_title('Rubric Performance Comparison', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(rubric_names, rotation=45, ha='right')
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def plot_score_distributions(
    results: List[EvaluationResult],
    output_path: Optional[str] = None,
) -> None:
    """
    Create violin plots of score distributions.

    Args:
        results: List of evaluation results
        output_path: Path to save plot

    Note: Requires matplotlib and seaborn
    """
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
        import pandas as pd
    except ImportError:
        print("Warning: matplotlib/seaborn not installed. Install with: pip install matplotlib seaborn")
        return

    # Prepare data
    data = []
    for result in results:
        for score in result.scores:
            data.append({
                "Rubric": result.rubric_name,
                "Score": score.total,
            })

    df = pd.DataFrame(data)

    # Create violin plot
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.violinplot(data=df, x="Rubric", y="Score", ax=ax)

    ax.set_xlabel('Rubric', fontsize=12)
    ax.set_ylabel('Score', fontsize=12)
    ax.set_title('Score Distribution by Rubric', fontsize=14, fontweight='bold')
    ax.tick_params(axis='x', rotation=45)
    ax.grid(axis='y', alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        print(f"Plot saved to {output_path}")
    else:
        plt.show()

    plt.close()


def export_to_csv(
    results: List[EvaluationResult],
    output_path: str,
) -> None:
    """
    Export results to CSV format.

    Args:
        results: List of evaluation results
        output_path: Path to save CSV file
    """
    try:
        import pandas as pd
    except ImportError:
        print("Warning: pandas not installed. Install with: pip install pandas")
        return

    # Prepare data
    data = []
    for result in results:
        for i, score in enumerate(result.scores):
            row = {
                "rubric_name": result.rubric_name,
                "sample_id": i,
                "total_score": score.total,
            }
            # Add component scores
            for comp_name, comp_value in score.components.items():
                row[f"component_{comp_name}"] = comp_value

            data.append(row)

    df = pd.DataFrame(data)
    df.to_csv(output_path, index=False)
    print(f"Results exported to {output_path}")
