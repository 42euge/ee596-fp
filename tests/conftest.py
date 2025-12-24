"""Shared pytest fixtures for testing."""

import pytest
import sys
from pathlib import Path

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# =============================================================================
# Sample Completion Fixtures
# =============================================================================

@pytest.fixture
def valid_completion():
    """A properly formatted completion with reasoning and answer tags."""
    return """<reasoning>
Let me work through this step by step.
First, I need to calculate the total.
5 + 10 = 15
Then multiply by 2.
15 * 2 = 30
</reasoning>
<answer>30</answer>"""


@pytest.fixture
def valid_completion_short():
    """A short but valid completion."""
    return "<reasoning>Quick calc: 5+5=10</reasoning><answer>10</answer>"


@pytest.fixture
def completion_missing_reasoning():
    """Completion without reasoning tags."""
    return "The answer is <answer>42</answer>"


@pytest.fixture
def completion_missing_answer():
    """Completion without answer tags."""
    return "<reasoning>I calculated 42</reasoning>\nThe answer is 42."


@pytest.fixture
def completion_no_tags():
    """Completion with no tags at all."""
    return "The total cost is $42.50 based on my calculations."


@pytest.fixture
def completion_with_numbers():
    """Completion with various number formats."""
    return """<reasoning>
The price is $1,234.56 which includes tax.
After discount: 1000.00
</reasoning>
<answer>1000</answer>"""


# =============================================================================
# Sample Data Fixtures
# =============================================================================

@pytest.fixture
def sample_prompts():
    """Sample prompts for testing."""
    return [
        "What is 5 + 10?",
        "Calculate the total cost of 3 items at $5 each.",
        "If John has 10 apples and gives away 3, how many remain?",
    ]


@pytest.fixture
def sample_answers():
    """Corresponding answers for sample prompts."""
    return ["15", "15", "7"]


@pytest.fixture
def sample_completions(valid_completion, valid_completion_short, completion_missing_answer):
    """Sample completions for batch testing."""
    return [valid_completion, valid_completion_short, completion_missing_answer]


@pytest.fixture
def sample_rubric():
    """Sample rubric for testing rubric-based scoring."""
    return """
    Scoring Criteria:
    - Shows step-by-step calculation (2 points)
    - Provides clear reasoning (2 points)
    - Gives correct final answer (4 points)
    - Uses proper mathematical notation (2 points)
    """


@pytest.fixture
def sample_reference_response():
    """Sample reference response for similarity scoring."""
    return """First, I'll add the numbers: 5 + 10 = 15.
Then multiply by 2 to get 30.
The final answer is 30."""


# =============================================================================
# Math Question Fixtures
# =============================================================================

@pytest.fixture
def math_question():
    """A typical math word problem."""
    return "If a store sells 15 apples at $2 each, how much total revenue is generated?"


@pytest.fixture
def creative_question():
    """A creative writing prompt."""
    return "Imagine a world where animals could talk. Write a short story about a dog's first day at school."


@pytest.fixture
def science_question():
    """A science question."""
    return "Explain the hypothesis behind why chemical reactions produce energy."


@pytest.fixture
def summarization_question():
    """A summarization task."""
    return "Summarize the main ideas of the following article in brief bullet points."


# =============================================================================
# GSM8K-style Data Fixtures
# =============================================================================

@pytest.fixture
def gsm8k_sample_data():
    """Sample GSM8K-style data entries."""
    return [
        {
            "question": "Tom has 5 apples. He buys 3 more. How many apples does Tom have?",
            "answer": "8",
            "source": "gsm8k",
        },
        {
            "question": "A book costs $12. If you buy 4 books, how much do you spend?",
            "answer": "48",
            "source": "gsm8k",
        },
    ]


@pytest.fixture
def openrubrics_sample_data():
    """Sample OpenRubrics-style data entries."""
    return [
        {
            "question": "Explain the concept of photosynthesis.",
            "rubric": "Must mention: sunlight, chlorophyll, glucose, oxygen",
            "reference_response": "Photosynthesis is the process by which plants convert sunlight into energy.",
            "target_score": 8,
            "source": "openrubrics",
        },
    ]


# =============================================================================
# Rubrics Module Fixtures
# =============================================================================

@pytest.fixture
def sample_criterion():
    """A sample Criterion for testing."""
    from src.rubrics.models import Criterion
    return Criterion(
        name="accuracy",
        description="Tests the accuracy of the answer",
        weight=2.0,
        keywords=["correct", "accurate", "right"],
    )


@pytest.fixture
def sample_rubric_obj():
    """A sample Rubric object for testing."""
    from src.rubrics.models import Criterion, Rubric
    return Rubric(
        name="math_evaluation",
        description="Evaluates mathematical problem solving",
        criteria=[
            Criterion(
                name="reasoning",
                description="Shows clear step-by-step reasoning",
                weight=2.0,
                keywords=["step", "first", "then", "therefore"],
            ),
            Criterion(
                name="format",
                description="Uses proper format with tags",
                weight=1.0,
                keywords=["<reasoning>", "</reasoning>", "<answer>", "</answer>"],
            ),
            Criterion(
                name="accuracy",
                description="Provides correct final answer",
                weight=2.0,
                keywords=["correct", "equals", "result"],
            ),
        ],
        question_types=["math", "arithmetic"],
        reference_response="<reasoning>Step by step solution</reasoning><answer>42</answer>",
        target_score=15.0,
    )


@pytest.fixture
def sample_rubricset():
    """A sample RubricSet for testing."""
    from src.rubrics.models import Criterion, Rubric, RubricSet
    return RubricSet(
        name="test_rubricset",
        description="A test rubric set",
        rubrics=[
            Rubric(
                name="math_rubric",
                description="For math questions",
                criteria=[Criterion(name="calc", description="Shows calculation")],
                question_types=["math"],
            ),
            Rubric(
                name="general_rubric",
                description="For general questions",
                criteria=[Criterion(name="clarity", description="Clear explanation")],
                question_types=["general"],
            ),
        ],
    )


@pytest.fixture
def rubric_test_dataset():
    """A dataset for testing rubrics."""
    return [
        {
            "question": "What is 2+2?",
            "response": "<reasoning>2+2=4</reasoning><answer>4</answer>",
        },
        {
            "question": "What is 3*4?",
            "response": "<reasoning>3 times 4 equals 12</reasoning><answer>12</answer>",
        },
        {
            "question": "Explain gravity",
            "response": "Gravity is a force that attracts objects toward each other.",
        },
    ]


# =============================================================================
# Reward Robustness Fixtures
# =============================================================================

@pytest.fixture
def robustness_config():
    """Basic robustness evaluation configuration."""
    from reward_robustness.config import (
        RobustnessConfig,
        PerturbationConfig,
    )

    return RobustnessConfig(
        internal_rewards=["format_reward"],
        perturbations=PerturbationConfig(
            enabled_types=["reorder"],
            num_variants=3,
        ),
        num_samples=5,
        save_detailed=True,
    )


@pytest.fixture
def robustness_config_quick():
    """Quick robustness evaluation configuration for fast tests."""
    from reward_robustness.config import get_quick_config

    return get_quick_config()


@pytest.fixture
def robustness_sample_completions():
    """Sample completions with multi-sentence reasoning for perturbation testing."""
    return [
        """<reasoning>
First, I need to understand the problem. Then, I will identify the key numbers.
Next, I perform the calculation. After that, I verify my result.
Finally, I provide the answer.
</reasoning>
<answer>42</answer>""",
        """<reasoning>
Let me break this down step by step. The first step is to add the numbers.
The second step is to multiply. Then I subtract the discount.
The final step is to round to the nearest dollar.
</reasoning>
<answer>100</answer>""",
        """<reasoning>
I'll solve this systematically. Start with the given values.
Apply the formula. Calculate intermediate results.
Combine everything for the final answer.
</reasoning>
<answer>256</answer>""",
    ]


@pytest.fixture
def robustness_sample_prompts():
    """Sample prompts for robustness testing."""
    return [
        "What is 6 * 7?",
        "Calculate the total after a 20% discount on $125.",
        "What is 2^8?",
    ]


@pytest.fixture
def robustness_sample_answers():
    """Answers for robustness sample prompts."""
    return ["42", "100", "256"]


@pytest.fixture
def consistency_metrics_sample():
    """Sample ConsistencyMetrics for testing."""
    from reward_robustness.metrics import ConsistencyMetrics

    return ConsistencyMetrics(
        reward_name="test_reward",
        mean_variance=0.15,
        max_variance=0.45,
        median_variance=0.12,
        variance_std=0.08,
        mean_cv=0.1,
        kendall_tau=0.85,
        spearman_rho=0.88,
        flip_rate=0.05,
        max_deviation=0.3,
        mean_deviation=0.12,
        stability_score=0.82,
        num_samples=100,
    )


@pytest.fixture
def perturbation_pipeline_reorder():
    """Perturbation pipeline with only reorder (no external deps)."""
    from reward_robustness.perturbations import PerturbationPipeline
    from reward_robustness.config import PerturbationConfig

    config = PerturbationConfig(
        enabled_types=["reorder"],
        num_variants=3,
    )
    return PerturbationPipeline(config)


@pytest.fixture
def mock_reward_scores():
    """Mock reward scores for metrics testing."""
    import numpy as np

    np.random.seed(42)

    # Original scores
    original = np.array([1.0, 2.0, 1.5, 2.5, 3.0])

    # Perturbed scores with slight variations
    perturbed = np.array([
        [1.1, 0.9, 1.0, 1.05, 0.95],
        [2.1, 1.9, 2.0, 2.05, 1.95],
        [1.6, 1.4, 1.5, 1.55, 1.45],
        [2.6, 2.4, 2.5, 2.55, 2.45],
        [3.1, 2.9, 3.0, 3.05, 2.95],
    ])

    return original, perturbed
