"""Structured output parsing for rubrics and scores.

Handles parsing of LLM outputs in various formats:
- JSON (preferred)
- XML
- Plain text (fallback)
"""

import re
import json
from typing import Dict, Any, Optional, Tuple, List, TYPE_CHECKING

if TYPE_CHECKING:
    from .rubric_generator import Rubric


def parse_rubric(
    text: str,
    question_hash: str,
    question_type: str,
    score_range: Tuple[int, int] = (0, 10),
) -> "Rubric":
    """Parse LLM output into a structured Rubric.

    Handles multiple formats:
    - JSON blocks (```json ... ```)
    - Plain JSON
    - Markdown-style lists
    - Plain text (fallback)

    Args:
        text: Raw LLM output text
        question_hash: Hash identifier for the question
        question_type: Type of question
        score_range: Valid score range tuple

    Returns:
        Parsed Rubric object
    """
    from .rubric_generator import Rubric

    # Try JSON in code block first
    json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if json_match:
        try:
            data = json.loads(json_match.group(1))
            return Rubric(
                question_hash=question_hash,
                question_type=question_type,
                criteria=data.get("criteria", []),
                score_range=score_range,
                metadata={"parse_method": "json_codeblock"},
            )
        except json.JSONDecodeError:
            pass

    # Try plain JSON
    try:
        # Find JSON object in text
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = text[json_start:json_end]
            data = json.loads(json_str)
            return Rubric(
                question_hash=question_hash,
                question_type=question_type,
                criteria=data.get("criteria", []),
                score_range=score_range,
                metadata={"parse_method": "json_plain"},
            )
    except json.JSONDecodeError:
        pass

    # Fallback: parse as plain text
    criteria = _parse_plain_text_rubric(text, score_range)
    return Rubric(
        question_hash=question_hash,
        question_type=question_type,
        criteria=criteria,
        score_range=score_range,
        metadata={"parse_method": "plain_text"},
    )


def _parse_plain_text_rubric(
    text: str, score_range: Tuple[int, int]
) -> List[Dict[str, Any]]:
    """Parse plain text rubric format.

    Attempts to extract criteria from numbered lists or bullet points.

    Args:
        text: Plain text rubric
        score_range: Score range for generating levels

    Returns:
        List of criterion dictionaries
    """
    criteria = []
    min_score, max_score = score_range
    mid_score = (min_score + max_score) // 2

    # Pattern: "1. Criterion Name (weight: X.XX)" or "1. Criterion Name"
    pattern = r"(\d+)\.\s*([^(\n]+?)(?:\s*\(weight:\s*([\d.]+)\))?"
    matches = re.findall(pattern, text)

    for idx, name, weight in matches:
        weight_val = float(weight) if weight else 1.0 / max(len(matches), 1)
        criteria.append(
            {
                "name": name.strip(),
                "description": "",
                "weight": weight_val,
                "levels": [
                    {"score": min_score, "description": "Poor"},
                    {"score": mid_score, "description": "Average"},
                    {"score": max_score, "description": "Excellent"},
                ],
            }
        )

    # If no matches, try bullet point pattern
    if not criteria:
        bullet_pattern = r"[-•*]\s*([^:\n]+?)(?::\s*(.+))?"
        bullet_matches = re.findall(bullet_pattern, text)
        for name, desc in bullet_matches[:5]:  # Limit to 5 criteria
            criteria.append(
                {
                    "name": name.strip(),
                    "description": desc.strip() if desc else "",
                    "weight": 1.0 / max(len(bullet_matches), 1),
                    "levels": [
                        {"score": min_score, "description": "Poor"},
                        {"score": mid_score, "description": "Average"},
                        {"score": max_score, "description": "Excellent"},
                    ],
                }
            )

    return criteria


def parse_score(
    text: str,
    output_format: str = "json",
    max_score: float = 10.0,
) -> Tuple[float, Optional[str], Optional[Dict[str, float]]]:
    """Parse LLM scoring output.

    Args:
        text: Raw LLM output text
        output_format: Expected format ("json", "xml", "plain")
        max_score: Maximum valid score

    Returns:
        Tuple of (score, reasoning, criterion_scores)
    """
    if output_format == "json":
        return _parse_json_score(text, max_score)
    elif output_format == "xml":
        return _parse_xml_score(text, max_score)
    else:
        return _parse_plain_score(text, max_score)


def _parse_json_score(
    text: str, max_score: float
) -> Tuple[float, Optional[str], Optional[Dict[str, float]]]:
    """Parse JSON format score.

    Args:
        text: Raw text containing JSON
        max_score: Maximum valid score

    Returns:
        Tuple of (score, reasoning, criterion_scores)
    """
    # Try to find JSON in code block
    json_match = re.search(r"```json\s*(.*?)\s*```", text, re.DOTALL)
    if json_match:
        text = json_match.group(1)

    try:
        # Find JSON object
        json_start = text.find("{")
        json_end = text.rfind("}") + 1
        if json_start >= 0 and json_end > json_start:
            json_str = text[json_start:json_end]
            data = json.loads(json_str)

            score = float(data.get("total_score", 0))
            score = max(0, min(score, max_score))  # Clamp to valid range
            reasoning = data.get("reasoning")
            criterion_scores = data.get("criterion_scores")

            return score, reasoning, criterion_scores
    except (json.JSONDecodeError, ValueError):
        pass

    # Fallback to plain parsing
    return _parse_plain_score(text, max_score)


def _parse_xml_score(
    text: str, max_score: float
) -> Tuple[float, Optional[str], Optional[Dict[str, float]]]:
    """Parse XML format score.

    Args:
        text: Raw text containing XML
        max_score: Maximum valid score

    Returns:
        Tuple of (score, reasoning, criterion_scores)
    """
    score = 0.0
    reasoning = None
    criterion_scores = None

    # Extract total score
    score_match = re.search(r"<total_score>\s*([\d.]+)\s*</total_score>", text)
    if score_match:
        score = float(score_match.group(1))
        score = max(0, min(score, max_score))

    # Extract reasoning
    reasoning_match = re.search(
        r"<reasoning>\s*(.*?)\s*</reasoning>", text, re.DOTALL
    )
    if reasoning_match:
        reasoning = reasoning_match.group(1).strip()

    # Extract criterion scores
    criterion_matches = re.findall(
        r'<criterion\s+name="([^"]+)">\s*([\d.]+)\s*</criterion>', text
    )
    if criterion_matches:
        criterion_scores = {name: float(score) for name, score in criterion_matches}

    return score, reasoning, criterion_scores


def _parse_plain_score(
    text: str, max_score: float
) -> Tuple[float, Optional[str], Optional[Dict[str, float]]]:
    """Parse plain text score (fallback).

    Looks for patterns like:
    - "Score: 8"
    - "8/10"
    - "Total score: 7.5"

    Args:
        text: Raw text
        max_score: Maximum valid score

    Returns:
        Tuple of (score, reasoning, criterion_scores)
    """
    patterns = [
        r"(?:score|total|rating|total_score)[:\s]*(\d+(?:\.\d+)?)",
        r"(\d+(?:\.\d+)?)\s*/\s*\d+",
        r"(\d+(?:\.\d+)?)\s*(?:out of|/)\s*\d+",
        r"^(\d+(?:\.\d+)?)\s*$",
    ]

    for pattern in patterns:
        match = re.search(pattern, text, re.IGNORECASE | re.MULTILINE)
        if match:
            score = float(match.group(1))
            score = max(0, min(score, max_score))

            # Try to extract reasoning
            reasoning = None
            reasoning_match = re.search(
                r"(?:reasoning|explanation|because)[:\s]*(.+)",
                text,
                re.IGNORECASE | re.DOTALL,
            )
            if reasoning_match:
                reasoning = reasoning_match.group(1).strip()[:500]  # Limit length

            return score, reasoning, None

    # No score found, return 0
    return 0.0, text[:500] if text else None, None
