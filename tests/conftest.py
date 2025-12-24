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
