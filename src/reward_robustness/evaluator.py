"""
Main evaluator for reward robustness assessment.

Orchestrates perturbation generation, reward scoring, and metric computation.
"""

import json
import os
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Any, Optional, Union
from pathlib import Path

from .config import RobustnessConfig
from .perturbations import PerturbationPipeline, PerturbedText
from .metrics import ConsistencyMetrics, compute_consistency_metrics, compare_metrics
from .rewards import (
    RewardModel,
    load_internal_rewards,
    load_external_rewards,
)


@dataclass
class SampleResult:
    """Results for a single sample."""

    sample_idx: int
    prompt: str
    original_completion: str
    perturbations: List[Dict[str, Any]]  # type, text, changes
    scores: Dict[str, Dict[str, float]]  # reward_name -> {original, perturbed_0, ...}


@dataclass
class RobustnessResults:
    """Complete results from robustness evaluation."""

    # Metadata
    timestamp: str
    config: Dict[str, Any]
    num_samples: int
    perturbation_types: List[str]
    num_variants_per_sample: int

    # Results per reward function
    metrics: Dict[str, ConsistencyMetrics]

    # Ranking by stability
    ranking: List[Dict[str, Any]]  # [{reward, stability_score}, ...]

    # Detailed per-sample data (optional)
    samples: Optional[List[SampleResult]] = None

    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for JSON serialization."""
        result = {
            "meta": {
                "timestamp": self.timestamp,
                "config": self.config,
                "num_samples": self.num_samples,
                "perturbation_types": self.perturbation_types,
                "num_variants_per_sample": self.num_variants_per_sample,
            },
            "summary": {
                "ranking": self.ranking,
            },
            "results": {
                name: {
                    "mean_variance": m.mean_variance,
                    "max_variance": m.max_variance,
                    "median_variance": m.median_variance,
                    "variance_std": m.variance_std,
                    "mean_cv": m.mean_cv,
                    "kendall_tau": m.kendall_tau,
                    "spearman_rho": m.spearman_rho,
                    "flip_rate": m.flip_rate,
                    "max_deviation": m.max_deviation,
                    "mean_deviation": m.mean_deviation,
                    "stability_score": m.stability_score,
                    "num_samples": m.num_samples,
                }
                for name, m in self.metrics.items()
            },
        }

        if self.samples:
            result["samples"] = [
                {
                    "idx": s.sample_idx,
                    "prompt": s.prompt[:200] + "..." if len(s.prompt) > 200 else s.prompt,
                    "original_completion": (
                        s.original_completion[:200] + "..."
                        if len(s.original_completion) > 200
                        else s.original_completion
                    ),
                    "perturbations": s.perturbations,
                    "scores": s.scores,
                }
                for s in self.samples
            ]

        return result

    def to_json(self, indent: int = 2) -> str:
        """Convert to JSON string."""
        return json.dumps(self.to_dict(), indent=indent)

    def save(self, path: Union[str, Path]) -> None:
        """Save results to JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w") as f:
            f.write(self.to_json())


class RobustnessEvaluator:
    """Main class for evaluating reward robustness."""

    def __init__(self, config: Optional[RobustnessConfig] = None):
        """Initialize the evaluator.

        Args:
            config: Robustness evaluation configuration
        """
        self.config = config or RobustnessConfig()
        self._perturbation_pipeline: Optional[PerturbationPipeline] = None
        self._reward_models: List[RewardModel] = []
        self._is_setup = False

    def setup(self) -> None:
        """Initialize perturbation pipeline and load reward models."""
        if self._is_setup:
            return

        # Setup perturbation pipeline
        self._perturbation_pipeline = PerturbationPipeline(self.config.perturbations)

        # Load internal reward functions
        internal_rewards = load_internal_rewards(self.config.internal_rewards)
        self._reward_models.extend(internal_rewards)

        # Load external reward models
        if self.config.external_rewards.model_ids:
            external_rewards = load_external_rewards(self.config.external_rewards)
            self._reward_models.extend(external_rewards)

        self._is_setup = True
        print(f"Loaded {len(self._reward_models)} reward models")
        print(f"Perturbation types: {self._perturbation_pipeline.perturbation_names}")

    def evaluate(
        self,
        prompts: List[str],
        completions: List[str],
        answers: Optional[List[str]] = None,
        rubrics: Optional[List[str]] = None,
        reference_responses: Optional[List[str]] = None,
        target_scores: Optional[List[float]] = None,
        verbose: bool = True,
    ) -> RobustnessResults:
        """Run full robustness evaluation.

        Args:
            prompts: List of input prompts
            completions: List of model completions to evaluate
            answers: Ground truth answers (for accuracy_reward)
            rubrics: Rubric texts (for rubric_reward)
            reference_responses: Reference responses (for rubric_reward)
            target_scores: Target scores (for rubric_reward)
            verbose: Print progress information

        Returns:
            RobustnessResults with all metrics
        """
        if not self._is_setup:
            self.setup()

        num_samples = min(len(prompts), self.config.num_samples)
        prompts = prompts[:num_samples]
        completions = completions[:num_samples]

        if verbose:
            print(f"Evaluating {num_samples} samples...")

        # Prepare optional kwargs for reward functions
        reward_kwargs = {}
        if answers:
            reward_kwargs["answers"] = answers[:num_samples]
        if rubrics:
            reward_kwargs["rubrics"] = rubrics[:num_samples]
        if reference_responses:
            reward_kwargs["reference_responses"] = reference_responses[:num_samples]
        if target_scores:
            reward_kwargs["target_scores"] = target_scores[:num_samples]

        # Generate perturbations for all samples
        if verbose:
            print("Generating perturbations...")

        all_perturbations: List[List[PerturbedText]] = []
        for i, completion in enumerate(completions):
            variants = self._perturbation_pipeline.generate_variants(completion)
            all_perturbations.append(variants)
            if verbose and (i + 1) % 10 == 0:
                print(f"  Generated perturbations for {i + 1}/{num_samples} samples")

        # Score all samples with all reward models
        if verbose:
            print("Scoring with reward models...")

        all_scores: Dict[str, Dict[str, List[float]]] = {}
        # Structure: {reward_name: {"original": [...], "perturbed": [[...], ...]}}

        for reward_model in self._reward_models:
            if verbose:
                print(f"  Scoring with {reward_model.name}...")

            # Score original completions
            original_scores = reward_model.score(
                prompts, completions, **reward_kwargs
            )

            # Score all perturbed completions
            perturbed_scores: List[List[float]] = []

            for sample_idx, variants in enumerate(all_perturbations):
                sample_perturbed_scores = []

                for variant in variants:
                    # Create single-item lists for scoring
                    score = reward_model.score(
                        [prompts[sample_idx]],
                        [variant.perturbed],
                        **{
                            k: [v[sample_idx]] if v else None
                            for k, v in reward_kwargs.items()
                        },
                    )[0]
                    sample_perturbed_scores.append(score)

                perturbed_scores.append(sample_perturbed_scores)

            all_scores[reward_model.name] = {
                "original": original_scores,
                "perturbed": perturbed_scores,
            }

        # Compute consistency metrics for each reward model
        if verbose:
            print("Computing consistency metrics...")

        metrics: Dict[str, ConsistencyMetrics] = {}

        for reward_name, scores in all_scores.items():
            metrics[reward_name] = compute_consistency_metrics(
                reward_name=reward_name,
                original_scores=scores["original"],
                perturbed_scores=scores["perturbed"],
                threshold=self.config.flip_threshold,
                include_details=self.config.save_detailed,
            )

        # Compute ranking
        ranking = compare_metrics(list(metrics.values()))
        ranking_dicts = [
            {"reward": name, "stability_score": round(score, 4)}
            for name, score in ranking
        ]

        # Prepare sample results if detailed output requested
        sample_results = None
        if self.config.save_detailed:
            sample_results = []
            for i in range(num_samples):
                # Collect scores for this sample
                sample_scores = {}
                for reward_name, scores in all_scores.items():
                    sample_scores[reward_name] = {
                        "original": scores["original"][i],
                        **{
                            f"perturbed_{j}": scores["perturbed"][i][j]
                            for j in range(len(scores["perturbed"][i]))
                        },
                    }

                sample_results.append(
                    SampleResult(
                        sample_idx=i,
                        prompt=prompts[i],
                        original_completion=completions[i],
                        perturbations=[
                            {
                                "type": v.perturbation_type,
                                "text": v.perturbed[:500],  # Truncate for storage
                                "changes": v.changes,
                            }
                            for v in all_perturbations[i]
                        ],
                        scores=sample_scores,
                    )
                )

        # Build results
        results = RobustnessResults(
            timestamp=datetime.now().isoformat(),
            config=asdict(self.config),
            num_samples=num_samples,
            perturbation_types=self._perturbation_pipeline.perturbation_names,
            num_variants_per_sample=len(all_perturbations[0]) if all_perturbations else 0,
            metrics=metrics,
            ranking=ranking_dicts,
            samples=sample_results,
        )

        # Save results if output directory specified
        if self.config.output_dir:
            output_path = Path(self.config.output_dir) / f"robustness_{results.timestamp.replace(':', '-')}.json"
            results.save(output_path)
            if verbose:
                print(f"Results saved to: {output_path}")

        return results

    def generate_report(
        self,
        results: RobustnessResults,
        format: str = "markdown",
    ) -> str:
        """Generate a human-readable report from results.

        Args:
            results: RobustnessResults from evaluate()
            format: Output format ("markdown", "text", "json")

        Returns:
            Formatted report string
        """
        if format == "json":
            return results.to_json()

        lines = []

        if format == "markdown":
            lines.append("# Reward Robustness Evaluation Report")
            lines.append("")
            lines.append(f"**Timestamp**: {results.timestamp}")
            lines.append(f"**Samples**: {results.num_samples}")
            lines.append(f"**Perturbation Types**: {', '.join(results.perturbation_types)}")
            lines.append(f"**Variants per Sample**: {results.num_variants_per_sample}")
            lines.append("")

            # Summary table
            lines.append("## Summary")
            lines.append("")
            lines.append("| Reward | Stability | Flip Rate | Kendall's τ | Mean Variance |")
            lines.append("|--------|-----------|-----------|-------------|---------------|")

            for item in results.ranking:
                name = item["reward"]
                m = results.metrics[name]
                lines.append(
                    f"| {name} | {m.stability_score:.3f} | {m.flip_rate:.1%} | "
                    f"{m.kendall_tau:.3f} | {m.mean_variance:.4f} |"
                )

            lines.append("")

            # Detailed metrics
            lines.append("## Detailed Metrics")
            lines.append("")

            for name, m in results.metrics.items():
                lines.append(f"### {name}")
                lines.append("")
                lines.append(f"- **Stability Score**: {m.stability_score:.4f}")
                lines.append(f"- **Mean Variance**: {m.mean_variance:.4f}")
                lines.append(f"- **Max Variance**: {m.max_variance:.4f}")
                lines.append(f"- **Mean CV**: {m.mean_cv:.4f}")
                lines.append(f"- **Kendall's τ**: {m.kendall_tau:.4f}")
                lines.append(f"- **Spearman's ρ**: {m.spearman_rho:.4f}")
                lines.append(f"- **Flip Rate**: {m.flip_rate:.2%}")
                lines.append(f"- **Max Deviation**: {m.max_deviation:.4f}")
                lines.append(f"- **Mean Deviation**: {m.mean_deviation:.4f}")
                lines.append("")

        else:  # text format
            lines.append("=" * 60)
            lines.append("REWARD ROBUSTNESS EVALUATION REPORT")
            lines.append("=" * 60)
            lines.append("")
            lines.append(f"Timestamp: {results.timestamp}")
            lines.append(f"Samples: {results.num_samples}")
            lines.append(f"Perturbations: {', '.join(results.perturbation_types)}")
            lines.append("")
            lines.append("-" * 60)
            lines.append("RANKING BY STABILITY")
            lines.append("-" * 60)

            for i, item in enumerate(results.ranking, 1):
                name = item["reward"]
                m = results.metrics[name]
                lines.append(f"{i}. {name}: {m.stability_score:.4f} (flip rate: {m.flip_rate:.1%})")

            lines.append("")

        return "\n".join(lines)
