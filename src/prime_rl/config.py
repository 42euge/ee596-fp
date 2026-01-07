"""
PRIME RL Configuration

Configuration dataclasses for PRIME RL (Process-based Reinforcement with
Intermediate Model Evaluation) training and evaluation.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, List


class StepEvaluationMethod(Enum):
    """Method for evaluating intermediate reasoning steps."""

    RULE_BASED = "rule_based"          # Rule-based checks (regex, pattern matching)
    SYMBOLIC = "symbolic"              # Symbolic solvers (for math)
    VERIFIER = "verifier"              # Learned verifier model
    LLM_JUDGE = "llm_judge"            # LLM-based step evaluation
    HYBRID = "hybrid"                  # Combination of multiple methods


class RewardAggregation(Enum):
    """Strategy for aggregating step rewards into trajectory reward."""

    SUM = "sum"                        # Simple sum: ∑ r_t
    DISCOUNTED_SUM = "discounted_sum"  # Discounted sum: ∑ γ^t r_t
    MEAN = "mean"                      # Average reward: (∑ r_t) / T
    WEIGHTED_MEAN = "weighted_mean"    # Position-weighted mean
    MIN = "min"                        # Minimum step reward (strictest)
    PRODUCT = "product"                # Product of normalized step rewards


class StepParsingStrategy(Enum):
    """Strategy for parsing reasoning into steps."""

    LINE_BASED = "line_based"          # Each line is a step
    SENTENCE_BASED = "sentence_based"  # Each sentence is a step
    NUMBERED = "numbered"              # Detect numbered steps (Step 1:, etc.)
    SEMANTIC = "semantic"              # Semantic chunking (paragraphs/ideas)
    CUSTOM_DELIMITER = "custom_delimiter"  # Custom delimiter (e.g., "---")


@dataclass
class PRIMEConfig:
    """
    Configuration for PRIME RL training.

    PRIME RL decomposes reasoning into explicit intermediate steps and applies
    step-level rewards, enabling the model to learn how to reason, not just
    what to answer.
    """

    # Step Parsing Configuration
    step_parsing_strategy: StepParsingStrategy = StepParsingStrategy.NUMBERED
    custom_delimiter: Optional[str] = None
    min_step_length: int = 10  # Minimum characters per step
    max_steps: int = 20  # Maximum number of steps to extract

    # Step Evaluation Configuration
    step_evaluation_method: StepEvaluationMethod = StepEvaluationMethod.HYBRID
    evaluators: List[StepEvaluationMethod] = field(
        default_factory=lambda: [
            StepEvaluationMethod.RULE_BASED,
            StepEvaluationMethod.LLM_JUDGE
        ]
    )

    # Reward Configuration
    reward_aggregation: RewardAggregation = RewardAggregation.DISCOUNTED_SUM
    gamma: float = 0.95  # Discount factor for credit assignment
    step_reward_scale: float = 1.0  # Scale factor for step rewards
    final_answer_weight: float = 2.0  # Additional weight for final answer correctness

    # Credit Assignment
    normalize_rewards: bool = True  # Normalize rewards to [-1, 1] or [0, 1]
    baseline_subtraction: bool = True  # Subtract baseline for variance reduction
    advantage_normalization: bool = False  # Normalize advantages

    # Step Quality Criteria
    step_quality_weights: dict = field(default_factory=lambda: {
        "logical_coherence": 0.3,
        "mathematical_correctness": 0.4,
        "relevance": 0.2,
        "clarity": 0.1
    })

    # Rule-Based Evaluator Settings
    rule_based_patterns: dict = field(default_factory=lambda: {
        "calculation": r"=\s*[\d\.\+\-\*/\(\)]+",
        "conclusion": r"(therefore|thus|hence|so|consequently)",
        "formula_usage": r"(using|applying|substituting)",
        "variable_definition": r"(let|define|assume)",
    })

    # Symbolic Evaluator Settings (for math)
    enable_symbolic_solver: bool = True
    symbolic_solver_timeout: float = 1.0  # Seconds

    # LLM Judge Settings for Step Evaluation
    llm_judge_enabled: bool = True
    llm_judge_model: str = "gpt-4o-mini"  # Fast model for step evaluation
    llm_judge_temperature: float = 0.0
    llm_judge_max_tokens: int = 100
    step_evaluation_batch_size: int = 16

    # Process Supervision
    penalize_incorrect_steps: bool = True
    incorrect_step_penalty: float = -0.5
    reward_correct_process: bool = True  # Reward correct process even if final answer wrong

    # Training Integration
    use_step_rewards_only: bool = False  # Use only step rewards (ignore final answer)
    combine_with_outcome_rewards: bool = True  # Combine step + outcome rewards
    outcome_reward_weight: float = 0.3  # Weight for outcome-based rewards

    # Debugging & Logging
    log_step_rewards: bool = True
    log_step_evaluations: bool = False  # Detailed step-by-step logs
    save_trajectories: bool = False  # Save full trajectories for analysis

    def __post_init__(self):
        """Validate configuration."""
        assert 0.0 <= self.gamma <= 1.0, "Gamma must be in [0, 1]"
        assert self.step_reward_scale > 0, "Step reward scale must be positive"
        assert self.final_answer_weight >= 0, "Final answer weight must be non-negative"

        if self.step_parsing_strategy == StepParsingStrategy.CUSTOM_DELIMITER:
            assert self.custom_delimiter is not None, \
                "Must provide custom_delimiter when using CUSTOM_DELIMITER strategy"

        # Ensure weights sum to reasonable value
        if self.combine_with_outcome_rewards:
            total_weight = self.outcome_reward_weight + (1.0 - self.outcome_reward_weight)
            assert abs(total_weight - 1.0) < 1e-6, \
                "Outcome and process weights should sum to 1.0"


@dataclass
class StepReward:
    """Reward information for a single reasoning step."""

    step_index: int
    step_text: str
    reward: float
    evaluation_method: str
    details: dict = field(default_factory=dict)

    def __repr__(self):
        return f"StepReward(step={self.step_index}, reward={self.reward:.3f}, method={self.evaluation_method})"


@dataclass
class TrajectoryReward:
    """Complete reward information for a reasoning trajectory."""

    prompt: str
    completion: str
    steps: List[str]
    step_rewards: List[StepReward]
    aggregated_reward: float
    final_answer_reward: float
    total_reward: float
    metadata: dict = field(default_factory=dict)

    @property
    def num_steps(self) -> int:
        return len(self.steps)

    @property
    def mean_step_reward(self) -> float:
        if not self.step_rewards:
            return 0.0
        return sum(sr.reward for sr in self.step_rewards) / len(self.step_rewards)

    def __repr__(self):
        return (f"TrajectoryReward(steps={self.num_steps}, "
                f"mean_step_reward={self.mean_step_reward:.3f}, "
                f"total_reward={self.total_reward:.3f})")
