"""Tests for tunrex.datasets.rewards module."""

import pytest
import sys
from pathlib import Path

# Add TunRex src to path and import directly to avoid grain dependency
tunrex_src = Path(__file__).parent.parent / "src" / "tunrex" / "datasets"
sys.path.insert(0, str(tunrex_src))

# Import directly from the module file to avoid package __init__.py
import importlib.util
spec = importlib.util.spec_from_file_location("rewards", tunrex_src / "rewards.py")
rewards_module = importlib.util.module_from_spec(spec)

# First load config module (rewards depends on it)
config_spec = importlib.util.spec_from_file_location("config", tunrex_src / "config.py")
config_module = importlib.util.module_from_spec(config_spec)
sys.modules["tunrex.datasets.config"] = config_module
config_spec.loader.exec_module(config_module)

# Now load rewards module
sys.modules["tunrex.datasets.rewards"] = rewards_module
spec.loader.exec_module(rewards_module)

from tunrex.datasets.rewards import (
    match_format,
    match_numbers,
    match_format_exactly,
    match_format_approximately,
    check_answer,
    check_numbers,
)


class TestMatchFormatRegex:
    """Tests for match_format regex pattern."""

    def test_valid_format(self):
        """Test that valid format matches."""
        text = "<reasoning>Some reasoning here</reasoning><answer>42</answer>"
        match = match_format.search(text)
        assert match is not None
        assert match.group(1) == "42"

    def test_multiline_format(self):
        """Test multiline content matches."""
        text = """<reasoning>
        Line 1
        Line 2
        </reasoning>
        <answer>100</answer>"""
        match = match_format.search(text)
        assert match is not None
        assert match.group(1) == "100"

    def test_invalid_format_missing_reasoning(self):
        """Test missing reasoning doesn't match."""
        text = "<answer>42</answer>"
        match = match_format.search(text)
        assert match is None

    def test_invalid_format_missing_answer(self):
        """Test missing answer doesn't match."""
        text = "<reasoning>reasoning</reasoning>"
        match = match_format.search(text)
        assert match is None

    def test_format_with_whitespace(self):
        """Test format with leading/trailing whitespace."""
        text = "  <reasoning>r</reasoning><answer>1</answer>  "
        match = match_format.search(text)
        assert match is not None


class TestMatchNumbersRegex:
    """Tests for match_numbers regex pattern."""

    def test_integer(self):
        """Test integer extraction."""
        text = "<answer>42</answer>"
        match = match_numbers.search(text)
        assert match is not None
        assert match.group(1) == "42"

    def test_decimal(self):
        """Test decimal extraction."""
        text = "<answer>3.14159</answer>"
        match = match_numbers.search(text)
        assert match is not None
        assert match.group(1) == "3.14159"

    def test_number_with_text(self):
        """Test number extraction when text is present."""
        text = "<answer>The answer is 100</answer>"
        match = match_numbers.search(text)
        assert match is not None
        assert match.group(1) == "100"


class TestMatchFormatExactly:
    """Tests for match_format_exactly function."""

    def test_valid_format_returns_3(self):
        """Test valid format returns 3.0 score."""
        completions = ["<reasoning>r</reasoning><answer>42</answer>"]
        scores = match_format_exactly(["prompt"], completions)
        assert scores == [3.0]

    def test_invalid_format_returns_0(self):
        """Test invalid format returns 0 score."""
        completions = ["Just some text with no tags"]
        scores = match_format_exactly(["prompt"], completions)
        assert scores == [0]

    def test_partial_format_returns_0(self):
        """Test partial format (missing reasoning) returns 0."""
        completions = ["<answer>42</answer>"]
        scores = match_format_exactly(["prompt"], completions)
        assert scores == [0]

    def test_batch_scoring(self):
        """Test batch scoring."""
        completions = [
            "<reasoning>r</reasoning><answer>1</answer>",
            "no tags",
            "<reasoning>r</reasoning><answer>2</answer>",
        ]
        scores = match_format_exactly(["p1", "p2", "p3"], completions)
        assert scores == [3.0, 0, 3.0]


class TestMatchFormatApproximately:
    """Tests for match_format_approximately function."""

    def test_perfect_format(self):
        """Test perfect format gets high score."""
        completions = ["<reasoning>reasoning</reasoning><answer>42</answer>"]
        scores = match_format_approximately(["prompt"], completions)
        assert len(scores) == 1
        assert scores[0] > 0  # Should be positive

    def test_no_tags_negative(self):
        """Test no tags gets negative score."""
        completions = ["Just plain text"]
        scores = match_format_approximately(["prompt"], completions)
        assert scores[0] < 0

    def test_duplicate_tags_penalty(self):
        """Test duplicate tags get penalized."""
        completions = ["<reasoning>r1</reasoning><reasoning>r2</reasoning><answer>42</answer>"]
        scores = match_format_approximately(["prompt"], completions)
        assert scores[0] < 2.5  # Less than perfect score

    def test_reasoning_not_at_start_penalty(self):
        """Test reasoning not at start gets penalized."""
        completions = ["text before<reasoning>r</reasoning><answer>42</answer>"]
        scores = match_format_approximately(["prompt"], completions)
        # Should lose points for reasoning not at position 0
        assert scores[0] < 2.5

    def test_batch_approximate_scoring(self):
        """Test batch approximate scoring."""
        completions = [
            "<reasoning>r</reasoning><answer>1</answer>",
            "no tags here",
        ]
        scores = match_format_approximately(["p1", "p2"], completions)
        assert len(scores) == 2
        assert scores[0] > scores[1]


class TestCheckAnswer:
    """Tests for check_answer function."""

    def test_exact_match(self):
        """Test exact answer match gets 3 points."""
        completions = ["<reasoning>r</reasoning><answer>42</answer>"]
        answers = ["42"]
        scores = check_answer(["prompt"], completions, answers)
        assert scores[0] == 3.0

    def test_match_with_whitespace(self):
        """Test match with stripped whitespace gets 1.5 points."""
        completions = ["<reasoning>r</reasoning><answer> 42 </answer>"]
        answers = ["42"]
        scores = check_answer(["prompt"], completions, answers)
        assert scores[0] == 1.5

    def test_close_ratio(self):
        """Test close ratio (within 10%) gets 0.5 points."""
        completions = ["<reasoning>r</reasoning><answer>105</answer>"]
        answers = ["100"]
        scores = check_answer(["prompt"], completions, answers)
        assert scores[0] == 0.5

    def test_medium_ratio(self):
        """Test medium ratio (within 20%) gets 0.25 points."""
        completions = ["<reasoning>r</reasoning><answer>115</answer>"]
        answers = ["100"]
        scores = check_answer(["prompt"], completions, answers)
        assert scores[0] == 0.25

    def test_wrong_answer_penalty(self):
        """Test wrong answer gets penalty."""
        completions = ["<reasoning>r</reasoning><answer>999</answer>"]
        answers = ["100"]
        scores = check_answer(["prompt"], completions, answers)
        assert scores[0] == -1.0

    def test_no_format_match_zero(self):
        """Test no format match returns 0."""
        completions = ["just text 42"]
        answers = ["42"]
        scores = check_answer(["prompt"], completions, answers)
        assert scores[0] == 0

    def test_batch_check_answer(self):
        """Test batch answer checking."""
        completions = [
            "<reasoning>r</reasoning><answer>10</answer>",
            "<reasoning>r</reasoning><answer>20</answer>",
        ]
        answers = ["10", "20"]
        scores = check_answer(["p1", "p2"], completions, answers)
        assert len(scores) == 2
        assert all(s == 3.0 for s in scores)


class TestCheckNumbers:
    """Tests for check_numbers function."""

    def test_correct_number(self, capsys):
        """Test correct number match."""
        completions = ["<answer>42</answer>"]
        answers = ["42"]
        question = ["What is the answer?"]
        scores = check_numbers(["prompt"], completions, answers, question=question)
        assert scores[0] == 1.5

    def test_incorrect_number(self, capsys):
        """Test incorrect number match."""
        completions = ["<answer>99</answer>"]
        answers = ["42"]
        question = ["What is the answer?"]
        scores = check_numbers(["prompt"], completions, answers, question=question)
        assert scores[0] == 0.0

    def test_no_number_found(self, capsys):
        """Test no number found returns 0."""
        completions = ["no answer tag here"]
        answers = ["42"]
        question = ["What is the answer?"]
        scores = check_numbers(["prompt"], completions, answers, question=question)
        assert scores[0] == 0

    def test_decimal_comparison(self, capsys):
        """Test decimal number comparison."""
        completions = ["<answer>3.14</answer>"]
        answers = ["3.14"]
        question = ["What is pi?"]
        scores = check_numbers(["prompt"], completions, answers, question=question)
        assert scores[0] == 1.5

    def test_batch_check_numbers(self, capsys):
        """Test batch number checking."""
        completions = [
            "<answer>10</answer>",
            "<answer>20</answer>",
        ]
        answers = ["10", "20"]
        question = ["Q1", "Q2"]
        scores = check_numbers(["p1", "p2"], completions, answers, question=question)
        assert len(scores) == 2
        assert all(s == 1.5 for s in scores)
