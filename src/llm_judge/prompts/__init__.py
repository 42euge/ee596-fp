"""Prompt templates for rubric generation and scoring."""

from .templates import (
    get_rubric_generation_prompt,
    get_scoring_prompt,
    get_reference_scoring_prompt,
    QUESTION_TYPE_RUBRIC_HINTS,
)

__all__ = [
    "get_rubric_generation_prompt",
    "get_scoring_prompt",
    "get_reference_scoring_prompt",
    "QUESTION_TYPE_RUBRIC_HINTS",
]
