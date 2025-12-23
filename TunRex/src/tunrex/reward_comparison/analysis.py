"""Analysis and visualization tools for reward comparison."""

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np

from tunrex.reward_comparison.comparison import ComparisonResult


@dataclass
class CorrelationAnalysis:
    """Results from correlation analysis between reward evaluators.

    Attributes:
        evaluators: List of evaluator names
        pearson_correlation: Pearson correlation matrix
        spearman_correlation: Spearman rank correlation matrix
        kendall_correlation: Kendall tau correlation matrix
    """

    evaluators: List[str]
    pearson_correlation: np.ndarray
    spearman_correlation: np.ndarray
    kendall_correlation: np.ndarray

    def get_pairwise_correlation(
        self,
        eval1: str,
        eval2: str,
        method: str = "pearson"
    ) -> float:
        """Get correlation between two evaluators.

        Args:
            eval1: First evaluator name
            eval2: Second evaluator name
            method: Correlation method ("pearson", "spearman", "kendall")

        Returns:
            Correlation coefficient
        """
        idx1 = self.evaluators.index(eval1)
        idx2 = self.evaluators.index(eval2)

        if method == "pearson":
            return self.pearson_correlation[idx1, idx2]
        elif method == "spearman":
            return self.spearman_correlation[idx1, idx2]
        elif method == "kendall":
            return self.kendall_correlation[idx1, idx2]
        else:
            raise ValueError(f"Unknown method: {method}")

    def format_matrix(self, method: str = "pearson") -> str:
        """Format correlation matrix as a string.

        Args:
            method: Correlation method to display

        Returns:
            Formatted string representation
        """
        if method == "pearson":
            matrix = self.pearson_correlation
            title = "Pearson Correlation"
        elif method == "spearman":
            matrix = self.spearman_correlation
            title = "Spearman Correlation"
        elif method == "kendall":
            matrix = self.kendall_correlation
            title = "Kendall Tau Correlation"
        else:
            raise ValueError(f"Unknown method: {method}")

        lines = [title + ":"]
        lines.append("")

        # Header
        header = "  " + "".join(f"{name[:8]:>10}" for name in self.evaluators)
        lines.append(header)

        # Rows
        for i, name in enumerate(self.evaluators):
            row = f"{name[:8]:<8} "
            row += "".join(f"{matrix[i, j]:>10.3f}" for j in range(len(self.evaluators)))
            lines.append(row)

        return "\n".join(lines)


@dataclass
class AgreementAnalysis:
    """Results from agreement analysis between reward evaluators.

    Attributes:
        evaluators: List of evaluator names
        pairwise_agreement: Dict of (eval1, eval2) -> agreement rate
        threshold: Threshold used for binary classification
        top_k_agreement: Agreement on top-k ranked samples
    """

    evaluators: List[str]
    pairwise_agreement: Dict[Tuple[str, str], float]
    threshold: Optional[float] = None
    top_k_agreement: Optional[Dict[int, Dict[Tuple[str, str], float]]] = None

    def get_agreement_rate(self, eval1: str, eval2: str) -> float:
        """Get agreement rate between two evaluators.

        Args:
            eval1: First evaluator name
            eval2: Second evaluator name

        Returns:
            Agreement rate (0.0 to 1.0)
        """
        key = (eval1, eval2) if (eval1, eval2) in self.pairwise_agreement else (eval2, eval1)
        return self.pairwise_agreement.get(key, 0.0)

    def format_summary(self) -> str:
        """Format agreement analysis as a string.

        Returns:
            Formatted string representation
        """
        lines = ["Agreement Analysis:"]
        lines.append("")

        if self.threshold is not None:
            lines.append(f"Binary threshold: {self.threshold}")
            lines.append("")

        lines.append("Pairwise Agreement:")
        for (eval1, eval2), rate in sorted(
            self.pairwise_agreement.items(),
            key=lambda x: x[1],
            reverse=True
        ):
            lines.append(f"  {eval1} <-> {eval2}: {rate:.3f}")

        if self.top_k_agreement:
            lines.append("")
            lines.append("Top-K Agreement:")
            for k in sorted(self.top_k_agreement.keys()):
                lines.append(f"  Top-{k}:")
                for (eval1, eval2), rate in sorted(
                    self.top_k_agreement[k].items(),
                    key=lambda x: x[1],
                    reverse=True
                ):
                    lines.append(f"    {eval1} <-> {eval2}: {rate:.3f}")

        return "\n".join(lines)


class RewardAnalyzer:
    """Analyzer for comparing reward methodologies.

    This class provides statistical analysis and visualization tools
    for understanding how different reward methods relate to each other.
    """

    def __init__(self, comparison_result: ComparisonResult):
        """Initialize analyzer with comparison results.

        Args:
            comparison_result: Results from RewardComparison.evaluate()
        """
        self.result = comparison_result
        self.scores_matrix = comparison_result.get_scores_matrix()

    def compute_correlations(self) -> CorrelationAnalysis:
        """Compute correlation matrices between evaluators.

        Returns:
            CorrelationAnalysis with Pearson, Spearman, and Kendall correlations
        """
        from scipy.stats import pearsonr, spearmanr, kendalltau

        n_evaluators = len(self.result.evaluators)
        pearson_matrix = np.eye(n_evaluators)
        spearman_matrix = np.eye(n_evaluators)
        kendall_matrix = np.eye(n_evaluators)

        for i in range(n_evaluators):
            for j in range(i + 1, n_evaluators):
                scores_i = self.scores_matrix[:, i]
                scores_j = self.scores_matrix[:, j]

                # Pearson correlation
                pearson_corr, _ = pearsonr(scores_i, scores_j)
                pearson_matrix[i, j] = pearson_corr
                pearson_matrix[j, i] = pearson_corr

                # Spearman correlation
                spearman_corr, _ = spearmanr(scores_i, scores_j)
                spearman_matrix[i, j] = spearman_corr
                spearman_matrix[j, i] = spearman_corr

                # Kendall tau
                kendall_corr, _ = kendalltau(scores_i, scores_j)
                kendall_matrix[i, j] = kendall_corr
                kendall_matrix[j, i] = kendall_corr

        return CorrelationAnalysis(
            evaluators=self.result.evaluators,
            pearson_correlation=pearson_matrix,
            spearman_correlation=spearman_matrix,
            kendall_correlation=kendall_matrix,
        )

    def compute_agreement(
        self,
        threshold: Optional[float] = None,
        top_k: Optional[List[int]] = None
    ) -> AgreementAnalysis:
        """Compute agreement rates between evaluators.

        Args:
            threshold: Optional threshold for binary classification.
                      If provided, computes agreement on pass/fail.
            top_k: Optional list of k values for top-k agreement analysis.

        Returns:
            AgreementAnalysis with pairwise agreement rates
        """
        pairwise_agreement = {}

        # Binary agreement (if threshold provided)
        if threshold is not None:
            binary_scores = self.scores_matrix >= threshold

            for i in range(len(self.result.evaluators)):
                for j in range(i + 1, len(self.result.evaluators)):
                    eval1 = self.result.evaluators[i]
                    eval2 = self.result.evaluators[j]

                    agreement = np.mean(binary_scores[:, i] == binary_scores[:, j])
                    pairwise_agreement[(eval1, eval2)] = agreement
        else:
            # Compute agreement based on score similarity
            for i in range(len(self.result.evaluators)):
                for j in range(i + 1, len(self.result.evaluators)):
                    eval1 = self.result.evaluators[i]
                    eval2 = self.result.evaluators[j]

                    scores_i = self.scores_matrix[:, i]
                    scores_j = self.scores_matrix[:, j]

                    # Normalize scores to [0, 1]
                    if scores_i.max() > scores_i.min():
                        norm_i = (scores_i - scores_i.min()) / (scores_i.max() - scores_i.min())
                    else:
                        norm_i = scores_i

                    if scores_j.max() > scores_j.min():
                        norm_j = (scores_j - scores_j.min()) / (scores_j.max() - scores_j.min())
                    else:
                        norm_j = scores_j

                    # Agreement = 1 - mean absolute difference
                    agreement = 1 - np.mean(np.abs(norm_i - norm_j))
                    pairwise_agreement[(eval1, eval2)] = agreement

        # Top-k agreement
        top_k_agreement = None
        if top_k:
            top_k_agreement = {}
            for k in top_k:
                k_agreement = {}
                for i in range(len(self.result.evaluators)):
                    for j in range(i + 1, len(self.result.evaluators)):
                        eval1 = self.result.evaluators[i]
                        eval2 = self.result.evaluators[j]

                        # Get top-k indices
                        top_k_i = set(np.argsort(self.scores_matrix[:, i])[-k:])
                        top_k_j = set(np.argsort(self.scores_matrix[:, j])[-k:])

                        # Compute overlap
                        overlap = len(top_k_i & top_k_j)
                        k_agreement[(eval1, eval2)] = overlap / k

                top_k_agreement[k] = k_agreement

        return AgreementAnalysis(
            evaluators=self.result.evaluators,
            pairwise_agreement=pairwise_agreement,
            threshold=threshold,
            top_k_agreement=top_k_agreement,
        )

    def generate_report(
        self,
        include_correlations: bool = True,
        include_agreement: bool = True,
        agreement_threshold: Optional[float] = None,
        top_k: Optional[List[int]] = None
    ) -> str:
        """Generate a comprehensive analysis report.

        Args:
            include_correlations: Include correlation analysis
            include_agreement: Include agreement analysis
            agreement_threshold: Threshold for binary agreement
            top_k: List of k values for top-k agreement

        Returns:
            Formatted report string
        """
        lines = []
        lines.append("=" * 80)
        lines.append("REWARD COMPARISON ANALYSIS REPORT")
        lines.append("=" * 80)
        lines.append("")

        # Basic statistics
        lines.append("Dataset Statistics:")
        lines.append(f"  Number of samples: {len(self.result.completions)}")
        lines.append(f"  Number of evaluators: {len(self.result.evaluators)}")
        lines.append("")

        # Score statistics
        lines.append("Score Statistics:")
        for name in self.result.evaluators:
            res = self.result.results[name]
            lines.append(f"  {name}:")
            lines.append(f"    Mean: {res.mean_score:.3f}")
            lines.append(f"    Std:  {res.std_score:.3f}")
            lines.append(f"    Min:  {min(res.scores):.3f}")
            lines.append(f"    Max:  {max(res.scores):.3f}")
        lines.append("")

        # Correlation analysis
        if include_correlations:
            lines.append("-" * 80)
            corr_analysis = self.compute_correlations()
            lines.append(corr_analysis.format_matrix("pearson"))
            lines.append("")
            lines.append(corr_analysis.format_matrix("spearman"))
            lines.append("")

        # Agreement analysis
        if include_agreement:
            lines.append("-" * 80)
            agreement_analysis = self.compute_agreement(
                threshold=agreement_threshold,
                top_k=top_k
            )
            lines.append(agreement_analysis.format_summary())
            lines.append("")

        lines.append("=" * 80)
        return "\n".join(lines)

    def plot_score_distributions(
        self,
        filepath: Optional[str] = None,
        figsize: Tuple[int, int] = (12, 6)
    ) -> None:
        """Plot score distributions for each evaluator.

        Args:
            filepath: Optional path to save the plot
            figsize: Figure size (width, height)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: matplotlib not installed. Skipping plot.")
            return

        fig, axes = plt.subplots(1, len(self.result.evaluators), figsize=figsize)
        if len(self.result.evaluators) == 1:
            axes = [axes]

        for i, name in enumerate(self.result.evaluators):
            scores = self.result.results[name].scores
            axes[i].hist(scores, bins=20, alpha=0.7, edgecolor='black')
            axes[i].set_title(name)
            axes[i].set_xlabel('Score')
            axes[i].set_ylabel('Frequency')
            axes[i].grid(True, alpha=0.3)

        plt.tight_layout()

        if filepath:
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"✓ Saved plot to {filepath}")
        else:
            plt.show()

    def plot_correlation_heatmap(
        self,
        filepath: Optional[str] = None,
        method: str = "pearson",
        figsize: Tuple[int, int] = (10, 8)
    ) -> None:
        """Plot correlation heatmap.

        Args:
            filepath: Optional path to save the plot
            method: Correlation method ("pearson", "spearman", "kendall")
            figsize: Figure size (width, height)
        """
        try:
            import matplotlib.pyplot as plt
            import seaborn as sns
        except ImportError:
            print("Warning: matplotlib/seaborn not installed. Skipping plot.")
            return

        corr_analysis = self.compute_correlations()

        if method == "pearson":
            matrix = corr_analysis.pearson_correlation
            title = "Pearson Correlation"
        elif method == "spearman":
            matrix = corr_analysis.spearman_correlation
            title = "Spearman Correlation"
        elif method == "kendall":
            matrix = corr_analysis.kendall_correlation
            title = "Kendall Tau Correlation"
        else:
            raise ValueError(f"Unknown method: {method}")

        fig, ax = plt.subplots(figsize=figsize)
        sns.heatmap(
            matrix,
            annot=True,
            fmt='.3f',
            cmap='coolwarm',
            center=0,
            vmin=-1,
            vmax=1,
            xticklabels=self.result.evaluators,
            yticklabels=self.result.evaluators,
            ax=ax,
            square=True
        )
        ax.set_title(f'{title} Between Reward Evaluators')

        plt.tight_layout()

        if filepath:
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"✓ Saved heatmap to {filepath}")
        else:
            plt.show()

    def plot_score_comparison(
        self,
        eval1: str,
        eval2: str,
        filepath: Optional[str] = None,
        figsize: Tuple[int, int] = (8, 8)
    ) -> None:
        """Plot scatter plot comparing two evaluators.

        Args:
            eval1: First evaluator name
            eval2: Second evaluator name
            filepath: Optional path to save the plot
            figsize: Figure size (width, height)
        """
        try:
            import matplotlib.pyplot as plt
        except ImportError:
            print("Warning: matplotlib not installed. Skipping plot.")
            return

        idx1 = self.result.evaluators.index(eval1)
        idx2 = self.result.evaluators.index(eval2)

        scores1 = self.scores_matrix[:, idx1]
        scores2 = self.scores_matrix[:, idx2]

        # Compute correlation
        corr_analysis = self.compute_correlations()
        pearson_corr = corr_analysis.pearson_correlation[idx1, idx2]

        fig, ax = plt.subplots(figsize=figsize)
        ax.scatter(scores1, scores2, alpha=0.5, edgecolors='black', linewidths=0.5)
        ax.set_xlabel(eval1)
        ax.set_ylabel(eval2)
        ax.set_title(f'{eval1} vs {eval2}\n(Pearson r = {pearson_corr:.3f})')
        ax.grid(True, alpha=0.3)

        # Add diagonal line
        min_score = min(scores1.min(), scores2.min())
        max_score = max(scores1.max(), scores2.max())
        ax.plot([min_score, max_score], [min_score, max_score],
                'r--', alpha=0.5, label='Perfect agreement')
        ax.legend()

        plt.tight_layout()

        if filepath:
            plt.savefig(filepath, dpi=150, bbox_inches='tight')
            print(f"✓ Saved comparison plot to {filepath}")
        else:
            plt.show()
