"""Transform functions for dataset processing."""

from __future__ import annotations

from typing import Any, Callable

from tunrex.datasets.loaders import extract_hash_answer


def _as_text(v: Any) -> str:
    """Convert value to text, handling bytes."""
    return v if isinstance(v, str) else v.decode("utf-8")


def create_grpo_transform(
    template: str,
    system_prompt: str,
    answer_extractor: Callable[[str], str | None] | None = None,
    include_rubric: bool = False,
) -> Callable[[dict], dict]:
    """Create a transform function for GRPO training.

    Args:
        template: Prompt template with {system_prompt} and {question} placeholders
        system_prompt: System prompt to insert into template
        answer_extractor: Function to extract answer from raw answer text
            Defaults to GSM8K hash extraction if None
        include_rubric: Whether to include rubric fields in output

    Returns:
        Transform function that takes a raw example and returns processed example
    """
    if answer_extractor is None:
        answer_extractor = extract_hash_answer

    def transform(x: dict) -> dict:
        question = _as_text(x.get("question", ""))
        raw_answer = _as_text(x.get("answer", ""))

        result = {
            "prompts": template.format(
                system_prompt=system_prompt,
                question=question,
            ),
            "question": question,
            "answer": answer_extractor(raw_answer) if answer_extractor else raw_answer,
        }

        if include_rubric:
            result["rubric"] = x.get("rubric", "")
            result["reference_response"] = x.get("reference_response", "")
            result["target_score"] = x.get("target_score")

        return result

    return transform


def create_raw_transform() -> Callable[[dict], dict]:
    """Create a transform that passes through raw data with minimal processing.

    Returns:
        Transform function that normalizes keys without applying templates
    """

    def transform(x: dict) -> dict:
        return {
            "question": _as_text(x.get("question", "")),
            "answer": _as_text(x.get("answer", "")),
            "rubric": x.get("rubric", ""),
            "reference_response": x.get("reference_response", ""),
            "target_score": x.get("target_score"),
        }

    return transform
