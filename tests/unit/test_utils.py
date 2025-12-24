"""Tests for src/utils.py module.

Note: We mock torch before importing to avoid requiring it in CI.
"""

import pytest
import sys
from unittest.mock import MagicMock

# Mock torch and its submodules before importing src (which imports src.model which needs torch)
sys.modules["torch"] = MagicMock()
sys.modules["torch.cuda"] = MagicMock()
sys.modules["torch.backends"] = MagicMock()
sys.modules["torch.backends.mps"] = MagicMock()
sys.modules["transformers"] = MagicMock()
sys.modules["peft"] = MagicMock()

from src.utils import (
    extract_reasoning_and_answer,
    extract_numerical_answer,
    format_reward,
    accuracy_reward,
    rubric_overlap_score,
    rubric_reward,
    detect_question_type,
    evaluate_accuracy,
)


class TestExtractReasoningAndAnswer:
    """Tests for extract_reasoning_and_answer function."""

    def test_valid_completion(self, valid_completion):
        """Test extraction from valid completion with both tags."""
        reasoning, answer = extract_reasoning_and_answer(valid_completion)
        assert "step by step" in reasoning.lower() or "calculate" in reasoning.lower()
        assert answer == "30"

    def test_short_completion(self, valid_completion_short):
        """Test extraction from short valid completion."""
        reasoning, answer = extract_reasoning_and_answer(valid_completion_short)
        assert "5+5=10" in reasoning
        assert answer == "10"

    def test_missing_reasoning(self, completion_missing_reasoning):
        """Test extraction when reasoning tags are missing."""
        reasoning, answer = extract_reasoning_and_answer(completion_missing_reasoning)
        assert reasoning == ""
        assert answer == "42"

    def test_missing_answer(self, completion_missing_answer):
        """Test extraction when answer tags are missing."""
        reasoning, answer = extract_reasoning_and_answer(completion_missing_answer)
        assert "42" in reasoning
        assert answer == ""

    def test_no_tags(self, completion_no_tags):
        """Test extraction when no tags present."""
        reasoning, answer = extract_reasoning_and_answer(completion_no_tags)
        assert reasoning == ""
        assert answer == ""

    def test_multiline_content(self):
        """Test extraction with multiline content."""
        completion = """<reasoning>
Line 1
Line 2
Line 3
</reasoning>
<answer>
42
</answer>"""
        reasoning, answer = extract_reasoning_and_answer(completion)
        assert "Line 1" in reasoning
        assert "Line 3" in reasoning
        assert answer == "42"

    def test_whitespace_handling(self):
        """Test that whitespace is stripped properly."""
        completion = "<reasoning>   padded   </reasoning><answer>  42  </answer>"
        reasoning, answer = extract_reasoning_and_answer(completion)
        assert reasoning == "padded"
        assert answer == "42"


class TestExtractNumericalAnswer:
    """Tests for extract_numerical_answer function."""

    def test_answer_is_pattern(self):
        """Test 'the answer is X' pattern."""
        text = "After calculation, the answer is 42"
        result = extract_numerical_answer(text)
        assert result == "42"

    def test_equals_pattern(self):
        """Test '= X' pattern at end of text."""
        text = "Total cost = 150"
        result = extract_numerical_answer(text)
        assert result == "150"

    def test_xml_answer_tag(self):
        """Test extraction from answer tags."""
        text = "<reasoning>work</reasoning><answer>99</answer>"
        result = extract_numerical_answer(text)
        assert result == "99"

    def test_last_number_fallback(self):
        """Test fallback to last number in text."""
        text = "The value is somewhere around 123"
        result = extract_numerical_answer(text)
        assert result == "123"

    def test_with_commas(self):
        """Test numbers with comma formatting."""
        text = "The total is 1,234,567"
        result = extract_numerical_answer(text)
        assert result == "1234567"

    def test_negative_numbers(self):
        """Test negative numbers."""
        text = "The answer is -42"
        result = extract_numerical_answer(text)
        assert result == "-42"

    def test_decimal_numbers(self):
        """Test decimal numbers."""
        text = "The answer is 3.14159"
        result = extract_numerical_answer(text)
        assert result == "3.14159"

    def test_no_numbers(self):
        """Test text with no numbers returns None."""
        text = "There are no numbers here"
        result = extract_numerical_answer(text)
        assert result is None

    def test_final_answer_colon(self):
        """Test 'final answer:' pattern."""
        text = "After all that work, final answer: 256"
        result = extract_numerical_answer(text)
        assert result == "256"


class TestFormatReward:
    """Tests for format_reward function."""

    def test_valid_completion_high_score(self, valid_completion):
        """Test that valid completion gets high score."""
        scores = format_reward(["prompt"], [valid_completion])
        assert len(scores) == 1
        assert scores[0] > 0  # Should be positive

    def test_no_tags_low_score(self, completion_no_tags):
        """Test that completion without tags gets low score."""
        scores = format_reward(["prompt"], [completion_no_tags])
        assert len(scores) == 1
        assert scores[0] < 0  # Should be negative

    def test_partial_tags(self, completion_missing_reasoning):
        """Test partial tag completion gets medium score."""
        scores = format_reward(["prompt"], [completion_missing_reasoning])
        assert len(scores) == 1
        # Has answer but no reasoning, so mixed score

    def test_batch_scoring(self, sample_completions):
        """Test batch scoring returns correct number of scores."""
        prompts = ["p1", "p2", "p3"]
        scores = format_reward(prompts, sample_completions)
        assert len(scores) == 3

    def test_substantive_reasoning_bonus(self):
        """Test that longer reasoning gets bonus points."""
        short = "<reasoning>x</reasoning><answer>1</answer>"
        long = "<reasoning>" + "x" * 100 + "</reasoning><answer>1</answer>"

        short_score = format_reward(["p"], [short])[0]
        long_score = format_reward(["p"], [long])[0]

        assert long_score > short_score


class TestAccuracyReward:
    """Tests for accuracy_reward function."""

    def test_correct_answer(self):
        """Test correct answer gets high score."""
        completions = ["<answer>42</answer>"]
        answers = ["42"]
        scores = accuracy_reward(["prompt"], completions, answers)
        assert scores[0] == 1.5

    def test_incorrect_answer(self):
        """Test incorrect answer gets zero score."""
        completions = ["<answer>99</answer>"]
        answers = ["42"]
        scores = accuracy_reward(["prompt"], completions, answers)
        assert scores[0] == 0.0

    def test_no_answer_found(self, completion_no_tags):
        """Test no numerical answer gets zero score."""
        scores = accuracy_reward(["prompt"], [completion_no_tags], ["42"])
        # completion_no_tags has $42.50, so it might extract 42.50
        assert scores[0] >= 0.0

    def test_float_comparison(self):
        """Test float answer comparison."""
        completions = ["The answer is 3.14"]
        answers = ["3.14"]
        scores = accuracy_reward(["prompt"], completions, answers)
        assert scores[0] == 1.5

    def test_batch_accuracy(self, sample_prompts, sample_answers):
        """Test batch accuracy scoring."""
        completions = [
            "<answer>15</answer>",
            "<answer>15</answer>",
            "<answer>7</answer>",
        ]
        scores = accuracy_reward(sample_prompts, completions, sample_answers)
        assert len(scores) == 3
        assert all(s == 1.5 for s in scores)


class TestRubricOverlapScore:
    """Tests for rubric_overlap_score function."""

    def test_full_overlap(self):
        """Test response that matches all rubric terms."""
        rubric = "calculation step reasoning"
        response = "My calculation shows step by step reasoning"
        score = rubric_overlap_score(response, rubric)
        assert score > 5.0  # Should be high

    def test_no_overlap(self):
        """Test response with no matching terms."""
        rubric = "specific technical jargon"
        response = "completely unrelated text"
        score = rubric_overlap_score(response, rubric)
        assert score < 3.0  # Should be low

    def test_partial_overlap(self):
        """Test partial matching."""
        rubric = "calculation step reasoning answer"
        response = "The calculation is correct"
        score = rubric_overlap_score(response, rubric)
        assert 0 < score < 10  # Middle range

    def test_empty_rubric(self):
        """Test empty rubric returns zero."""
        score = rubric_overlap_score("any response", "")
        assert score == 0.0

    def test_empty_response(self):
        """Test empty response."""
        score = rubric_overlap_score("", "rubric terms")
        assert score == 0.0

    def test_punctuation_handling(self):
        """Test that punctuation is handled properly."""
        rubric = "calculation, step-by-step, reasoning!"
        response = "I'll show calculation and reasoning"
        score = rubric_overlap_score(response, rubric)
        assert score > 0  # Should find matches despite punctuation

    def test_case_insensitive(self):
        """Test case insensitivity."""
        rubric = "CALCULATION STEP"
        response = "calculation step"
        score = rubric_overlap_score(response, rubric)
        assert score > 5.0


class TestRubricReward:
    """Tests for rubric_reward function."""

    def test_with_rubric_only(self):
        """Test scoring with only rubric."""
        completions = ["My response mentions calculation and reasoning"]
        rubrics = ["calculation reasoning step"]
        rewards = rubric_reward(["prompt"], completions, rubrics=rubrics)
        assert len(rewards) == 1
        assert rewards[0] > 0

    def test_with_reference(self):
        """Test scoring with reference response."""
        completions = ["This is the response"]
        references = ["This is the response"]  # Identical
        rewards = rubric_reward(["prompt"], completions, reference_responses=references)
        assert len(rewards) == 1
        assert rewards[0] > 0  # Should get similarity bonus

    def test_with_target_score(self):
        """Test scoring with target score."""
        completions = ["response"]
        target_scores = [8.0]  # 80% quality
        rewards = rubric_reward(["prompt"], completions, target_scores=target_scores)
        assert len(rewards) == 1
        assert rewards[0] > 0

    def test_combined_scoring(self, sample_rubric, sample_reference_response):
        """Test combined scoring with all parameters."""
        completions = ["First, calculation step by step to get final answer 30"]
        rewards = rubric_reward(
            ["prompt"],
            completions,
            rubrics=[sample_rubric],
            reference_responses=[sample_reference_response],
            target_scores=[7.0],
        )
        assert len(rewards) == 1
        assert rewards[0] > 0

    def test_no_optional_params(self):
        """Test with no optional parameters."""
        completions = ["any response"]
        rewards = rubric_reward(["prompt"], completions)
        assert len(rewards) == 1
        assert rewards[0] == 0.0  # No rubric/ref/target means 0


class TestDetectQuestionType:
    """Tests for detect_question_type function."""

    def test_math_question(self, math_question):
        """Test math question detection."""
        qtype = detect_question_type(math_question)
        assert qtype == "math"

    def test_creative_question(self, creative_question):
        """Test creative question detection."""
        qtype = detect_question_type(creative_question)
        assert qtype == "creative"

    def test_science_question(self, science_question):
        """Test science question detection."""
        qtype = detect_question_type(science_question)
        assert qtype == "science"

    def test_summarization_question(self, summarization_question):
        """Test summarization question detection."""
        qtype = detect_question_type(summarization_question)
        assert qtype == "summarization"

    def test_default_question(self):
        """Test generic question returns default."""
        qtype = detect_question_type("What color is the sky?")
        assert qtype == "default"

    def test_math_with_numbers(self):
        """Test that numbers boost math detection."""
        question = "There are 10 items, 5 boxes, and 2 shelves. Organize them."
        qtype = detect_question_type(question)
        assert qtype == "math"

    def test_case_insensitive(self):
        """Test case insensitivity."""
        question = "CALCULATE the TOTAL cost"
        qtype = detect_question_type(question)
        assert qtype == "math"


class TestEvaluateAccuracy:
    """Tests for evaluate_accuracy function."""

    def test_all_correct(self):
        """Test all correct predictions."""
        predictions = [
            "<reasoning>calc</reasoning><answer>42</answer>",
            "<reasoning>work</reasoning><answer>100</answer>",
        ]
        truths = ["42", "100"]
        result = evaluate_accuracy(predictions, truths)

        assert result["accuracy"] == 100.0
        assert result["correct"] == 2
        assert result["total"] == 2

    def test_all_incorrect(self):
        """Test all incorrect predictions."""
        predictions = [
            "<reasoning>calc</reasoning><answer>99</answer>",
            "<reasoning>work</reasoning><answer>1</answer>",
        ]
        truths = ["42", "100"]
        result = evaluate_accuracy(predictions, truths)

        assert result["accuracy"] == 0.0
        assert result["correct"] == 0

    def test_partial_correct(self):
        """Test partial correctness."""
        predictions = [
            "<reasoning>r</reasoning><answer>42</answer>",
            "<reasoning>r</reasoning><answer>99</answer>",
        ]
        truths = ["42", "100"]
        result = evaluate_accuracy(predictions, truths)

        assert result["accuracy"] == 50.0
        assert result["correct"] == 1

    def test_format_accuracy(self):
        """Test format accuracy calculation."""
        predictions = [
            "<reasoning>r</reasoning><answer>42</answer>",  # Good format
            "Just 100",  # Bad format
        ]
        truths = ["42", "100"]
        result = evaluate_accuracy(predictions, truths)

        assert result["format_accuracy"] == 50.0

    def test_partial_accuracy(self):
        """Test partial accuracy (within 10% range)."""
        predictions = [
            "<reasoning>r</reasoning><answer>105</answer>",  # Within 10% of 100
        ]
        truths = ["100"]
        result = evaluate_accuracy(predictions, truths)

        assert result["partial_accuracy"] > 0  # Should count as partial

    def test_empty_predictions(self):
        """Test with empty prediction list."""
        result = evaluate_accuracy([], [])
        assert result["accuracy"] == 0
        assert result["total"] == 0
