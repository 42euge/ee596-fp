"""
Utility functions for data loading, evaluation, and reward computation.

Includes:
- Dataset loading (GSM8K, OpenRubrics)
- Reward functions for GRPO training
- Rubric-based evaluation
- Text processing utilities
"""

import os
import re
import csv
import random
import string
import difflib
from typing import List, Dict, Any, Optional, Tuple
from collections import Counter
from pathlib import Path

from .config import (
    REASONING_START, REASONING_END, SOLUTION_START, SOLUTION_END,
    format_prompt, get_system_prompt
)


# =============================================================================
# Text Processing Utilities
# =============================================================================

def extract_reasoning_and_answer(completion: str) -> Tuple[str, str]:
    """Extract reasoning and answer sections from a completion.

    Args:
        completion: The model's full response

    Returns:
        Tuple of (reasoning, answer) strings
    """
    reasoning_match = re.search(
        rf"{REASONING_START}(.*?){REASONING_END}",
        completion,
        flags=re.DOTALL | re.MULTILINE
    )
    answer_match = re.search(
        rf"{SOLUTION_START}(.*?){SOLUTION_END}",
        completion,
        flags=re.DOTALL | re.MULTILINE
    )

    reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
    answer = answer_match.group(1).strip() if answer_match else ""

    return reasoning, answer


def extract_numerical_answer(text: str) -> Optional[str]:
    """Extract a numerical answer from text.

    Tries multiple patterns to find numbers in various formats.

    Args:
        text: Text to search for numbers

    Returns:
        Extracted number as string, or None if not found
    """
    patterns = [
        # Common answer patterns
        r"(?:the answer is|answer:|final answer:?)\s*([\d,.-]+)",
        r"(?:=|equals)\s*([\d,.-]+)\s*$",
        # XML-style answer tags
        rf"{SOLUTION_START}\s*([\d,.-]+)\s*{SOLUTION_END}",
        # Last number in text (fallback)
        r"([\d,.-]+)\s*$",
    ]

    text_lower = text.lower()
    for pattern in patterns:
        match = re.search(pattern, text_lower, re.IGNORECASE | re.MULTILINE)
        if match:
            # Clean up the number (remove commas)
            num_str = match.group(1).replace(",", "")
            try:
                float(num_str)  # Validate it's a number
                return num_str
            except ValueError:
                continue

    return None


# =============================================================================
# Dataset Loading
# =============================================================================

def load_gsm8k_dataset(
    data_dir: str = "./data",
    split: str = "train",
    max_examples: Optional[int] = None,
    seed: int = 42,
) -> List[Dict[str, str]]:
    """Load GSM8K dataset from CSV files.

    Args:
        data_dir: Directory containing the CSV files
        split: Dataset split ("train" or "test")
        max_examples: Maximum number of examples to load
        seed: Random seed for sampling

    Returns:
        List of dictionaries with 'question', 'answer', 'source' keys
    """
    csv_path = os.path.join(data_dir, f"main_{split}.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"GSM8K dataset not found at {csv_path}. "
            "Please download from Kaggle: thedevastator/grade-school-math-8k-q-a"
        )

    data = []
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            answer = row["answer"]
            # Extract numerical answer after ####
            if "####" in answer:
                answer = answer.split("####")[1].strip()

            data.append({
                "question": row["question"],
                "answer": answer,
                "source": "gsm8k",
            })

    # Sample if max_examples specified
    if max_examples and len(data) > max_examples:
        random.seed(seed)
        data = random.sample(data, max_examples)

    return data


def load_openrubrics_dataset(
    split: str = "train",
    max_examples: int = 2000,
    seed: int = 42,
) -> List[Dict[str, Any]]:
    """Load OpenRubrics dataset from HuggingFace.

    Args:
        split: Dataset split
        max_examples: Maximum examples to load
        seed: Random seed

    Returns:
        List of dictionaries with question, rubric, response data
    """
    try:
        from datasets import load_dataset
    except ImportError:
        raise ImportError("Please install datasets: pip install datasets")

    try:
        raw_ds = load_dataset("OpenRubrics/OpenRubrics", split=split)
    except Exception as e:
        print(f"Warning: Could not load OpenRubrics: {e}")
        return []

    columns = list(raw_ds.column_names)

    def _normalize_entry(value):
        if value is None:
            return ""
        if isinstance(value, (list, tuple)):
            return "\n".join(str(v) for v in value if v)
        if isinstance(value, dict):
            parts = [f"{k}: {v}" for k, v in value.items() if v]
            return "\n".join(parts)
        return str(value)

    def _pick_column(options, default=""):
        for name in options:
            if name in columns:
                return name
        return default

    # Map column names
    question_col = _pick_column(["instruction", "prompt", "question", "input"])
    rubric_col = _pick_column(["rubric", "scoring_rubric", "criteria"])
    response_col = _pick_column(["response", "model_answer", "answer", "output"])
    score_col = _pick_column(["score", "rating", "label"])

    data = []
    for row in raw_ds:
        item = {
            "question": _normalize_entry(row.get(question_col, "")),
            "rubric": _normalize_entry(row.get(rubric_col, "")),
            "reference_response": _normalize_entry(row.get(response_col, "")),
            "target_score": row.get(score_col),
            "source": "openrubrics",
        }
        if item["question"]:
            data.append(item)

    # Sample
    if max_examples and len(data) > max_examples:
        random.seed(seed)
        data = random.sample(data, max_examples)

    return data


# =============================================================================
# Reward Functions
# =============================================================================

def format_reward(prompts: List[str], completions: List[str], **kwargs) -> List[float]:
    """Reward based on proper format usage (reasoning and answer tags).

    Returns scores from -2 to +2:
    - -2: No tags found
    - -1: Partial tags
    - +1: Both tags present
    - +2: Both tags with good content

    Args:
        prompts: Input prompts
        completions: Model completions

    Returns:
        List of format scores
    """
    scores = []
    for completion in completions:
        reasoning, answer = extract_reasoning_and_answer(completion)

        score = 0.0
        if reasoning:
            score += 1.0
            if len(reasoning) > 50:  # Bonus for substantive reasoning
                score += 0.5
        else:
            score -= 1.0

        if answer:
            score += 1.0
            if len(answer.strip()) > 0 and len(answer) < 100:
                score += 0.5
        else:
            score -= 1.0

        scores.append(score)

    return scores


def accuracy_reward(
    prompts: List[str],
    completions: List[str],
    answers: List[str],
    **kwargs
) -> List[float]:
    """Reward based on answer accuracy (for verifiable tasks like math).

    Args:
        prompts: Input prompts
        completions: Model completions
        answers: Ground truth answers

    Returns:
        List of accuracy scores (0.0 or 1.5)
    """
    scores = []
    for completion, true_answer in zip(completions, answers):
        guess = extract_numerical_answer(completion)

        if guess is None:
            scores.append(0.0)
            continue

        try:
            true_val = float(true_answer.strip())
            guess_val = float(guess.strip())
            scores.append(1.5 if guess_val == true_val else 0.0)
        except (ValueError, AttributeError):
            scores.append(0.0)

    return scores


def rubric_overlap_score(response: str, rubric_text: str) -> float:
    """Calculate rubric overlap with TF-IDF-style weighting.

    Rare rubric terms are weighted higher than common ones.

    Args:
        response: Model response
        rubric_text: Rubric text

    Returns:
        Score from 0 to 10
    """
    def tokenize(text):
        text = text.lower()
        for ch in string.punctuation:
            text = text.replace(ch, " ")
        return [t for t in text.split() if len(t) > 2]

    rubric_tokens = tokenize(rubric_text)
    response_tokens = set(tokenize(response))

    if not rubric_tokens:
        return 0.0

    # TF-IDF style: rare terms matter more
    token_counts = Counter(rubric_tokens)
    weighted_matches = sum(
        1.0 / token_counts[t] for t in response_tokens if t in token_counts
    )
    max_score = sum(1.0 / c for c in token_counts.values())

    coverage = weighted_matches / max_score if max_score > 0 else 0.0
    return round(coverage * 10.0, 4)


def rubric_reward(
    prompts: List[str],
    completions: List[str],
    rubrics: Optional[List[str]] = None,
    reference_responses: Optional[List[str]] = None,
    target_scores: Optional[List[float]] = None,
    **kwargs
) -> List[float]:
    """Rubric-as-Reward (RaR) scoring function.

    Combines:
    - Rubric overlap (0-10)
    - Reference response similarity (0-5)
    - Target score alignment (0-5)

    Total: 0-20 range

    Args:
        prompts: Input prompts
        completions: Model completions
        rubrics: Rubric texts
        reference_responses: Reference model responses
        target_scores: Target quality scores

    Returns:
        List of total rewards
    """
    rubrics = rubrics or [""] * len(completions)
    reference_responses = reference_responses or [""] * len(completions)
    target_scores = target_scores or [None] * len(completions)

    rewards = []
    for response, rubric, ref, target in zip(
        completions, rubrics, reference_responses, target_scores
    ):
        # Rubric overlap score (0-10)
        r_score = rubric_overlap_score(response, rubric)

        # Reference similarity score (0-5)
        f_score = 0.0
        if ref:
            similarity = difflib.SequenceMatcher(None, ref, response).ratio()
            f_score = similarity * 5.0

        # Target score alignment (0-5)
        l_score = 0.0
        if target is not None:
            try:
                l_score = max(0.0, min(1.0, float(target) / 10.0)) * 5.0
            except (ValueError, TypeError):
                pass

        rewards.append(r_score + f_score + l_score)

    return rewards


# =============================================================================
# Question Type Detection
# =============================================================================

def detect_question_type(prompt: str) -> str:
    """Detect question type for rubric selection.

    Args:
        prompt: The question/prompt

    Returns:
        One of: 'math', 'creative', 'summarization', 'science', 'default'
    """
    prompt_lower = prompt.lower()

    # Keyword lists for each type
    keywords = {
        "math": [
            "calculate", "compute", "solve", "equation", "how many", "how much",
            "total", "sum", "difference", "product", "divide", "multiply",
            "percentage", "fraction", "ratio", "average", "cost", "price",
        ],
        "creative": [
            "imagine", "story", "creative", "narrative", "describe", "write about",
            "what would happen", "suppose", "pretend", "character", "scenario",
        ],
        "summarization": [
            "summarize", "summary", "main idea", "key points", "in brief",
            "tldr", "explain in short",
        ],
        "science": [
            "experiment", "hypothesis", "theory", "phenomenon", "chemical",
            "physics", "biology", "scientific", "causes", "why does",
            "reaction", "energy", "force",
        ],
    }

    # Calculate normalized scores
    scores = {}
    for qtype, kw_list in keywords.items():
        matches = sum(1 for kw in kw_list if kw in prompt_lower)
        scores[qtype] = matches / len(kw_list)

    # Boost math score for numeric content
    numbers = len(re.findall(r'\d+', prompt))
    scores["math"] += min(numbers * 0.05, 0.3)

    max_score = max(scores.values())
    if max_score < 0.05:
        return "default"

    for qtype, score in scores.items():
        if score == max_score:
            return qtype

    return "default"


# =============================================================================
# Evaluation Utilities
# =============================================================================

def evaluate_accuracy(
    predictions: List[str],
    ground_truths: List[str],
) -> Dict[str, float]:
    """Evaluate prediction accuracy.

    Args:
        predictions: Model predictions
        ground_truths: Ground truth answers

    Returns:
        Dictionary with accuracy metrics
    """
    correct = 0
    partially_correct = 0
    format_ok = 0
    total = len(predictions)

    for pred, truth in zip(predictions, ground_truths):
        reasoning, answer = extract_reasoning_and_answer(pred)

        # Check format
        if reasoning and answer:
            format_ok += 1

        # Check accuracy
        pred_num = extract_numerical_answer(pred)
        if pred_num is not None:
            try:
                pred_val = float(pred_num)
                truth_val = float(truth)

                if pred_val == truth_val:
                    correct += 1
                    partially_correct += 1
                elif 0.9 <= pred_val / truth_val <= 1.1:
                    partially_correct += 1
            except (ValueError, ZeroDivisionError):
                pass

    return {
        "accuracy": correct / total * 100 if total > 0 else 0,
        "partial_accuracy": partially_correct / total * 100 if total > 0 else 0,
        "format_accuracy": format_ok / total * 100 if total > 0 else 0,
        "total": total,
        "correct": correct,
    }
