"""Functions for loading rubrics from various sources.

Supports loading from YAML files, directories, and the OpenRubrics dataset.
"""

from pathlib import Path
from typing import List, Any
import string
import uuid

import yaml

from .models import Rubric, RubricSet, Criterion


def load_rubric_from_yaml(file_path: str | Path) -> Rubric:
    """Load a single rubric from a YAML file.

    Args:
        file_path: Path to YAML file containing a single rubric

    Returns:
        Rubric object

    Raises:
        FileNotFoundError: If file doesn't exist
        ValueError: If YAML structure is invalid
    """
    path = Path(file_path)
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    return _parse_rubric(data)


def load_rubricset_from_yaml(file_path: str | Path) -> RubricSet:
    """Load a RubricSet from a YAML file.

    The file should contain a 'rubrics' list or be a single rubric.

    Args:
        file_path: Path to YAML file containing rubrics

    Returns:
        RubricSet object
    """
    path = Path(file_path)
    with open(path, "r") as f:
        data = yaml.safe_load(f)

    # Handle single rubric files
    if "rubrics" not in data and "criteria" in data:
        rubric = _parse_rubric(data)
        return RubricSet(
            name=rubric.name,
            rubrics=[rubric],
            metadata={"source_file": str(path)},
        )

    return _parse_rubricset(data, source_file=str(path))


def load_rubrics_from_directory(
    directory: str | Path,
    pattern: str = "*.yaml",
) -> RubricSet:
    """Load all rubrics from YAML files in a directory.

    Args:
        directory: Directory path containing YAML files
        pattern: Glob pattern for matching files (default: *.yaml)

    Returns:
        RubricSet containing all loaded rubrics
    """
    dir_path = Path(directory)
    rubrics = []

    for yaml_file in sorted(dir_path.glob(pattern)):
        try:
            data = yaml.safe_load(yaml_file.read_text())
            if "rubrics" in data:
                # File contains a RubricSet
                rubricset = _parse_rubricset(data)
                rubrics.extend(rubricset.rubrics)
            else:
                # File contains a single Rubric
                rubrics.append(_parse_rubric(data))
        except Exception as e:
            print(f"Warning: Could not load {yaml_file}: {e}")

    return RubricSet(
        name=f"rubrics_from_{dir_path.name}",
        rubrics=rubrics,
        metadata={"source_dir": str(dir_path), "file_count": len(rubrics)},
    )


def load_rubrics_from_openrubrics(
    split: str = "train",
    max_examples: int | None = None,
    question_types: List[str] | None = None,
) -> RubricSet:
    """Load rubrics from the OpenRubrics HuggingFace dataset.

    Wraps existing load_openrubrics() and converts to RubricSet format.

    Args:
        split: Dataset split ("train", "test")
        max_examples: Maximum examples to load
        question_types: Filter to specific question types (not implemented yet)

    Returns:
        RubricSet with rubrics from OpenRubrics
    """
    # Use existing TunRex loader
    try:
        from tunrex.datasets import load_openrubrics
    except ImportError:
        # Fallback to direct import
        from TunRex.src.tunrex.datasets.loaders import load_openrubrics

    raw_data = load_openrubrics(split=split, max_examples=max_examples)

    rubrics = []
    for i, item in enumerate(raw_data):
        rubric_text = item.get("rubric", "")
        question = item.get("question", "")

        rubric = Rubric(
            id=f"openrubrics_{i}",
            name=f"OpenRubrics #{i}",
            description=question[:100] if question else "OpenRubrics item",
            criteria=[
                Criterion(
                    name="rubric_criteria",
                    description=rubric_text,
                    keywords=_extract_keywords(rubric_text),
                )
            ],
            reference_response=item.get("reference_response"),
            target_score=item.get("target_score"),
            metadata={
                "source": "OpenRubrics",
                "original_question": question,
                "original_answer": item.get("answer"),
            },
        )
        rubrics.append(rubric)

    return RubricSet(
        name="OpenRubrics",
        rubrics=rubrics,
        description=f"Loaded from OpenRubrics dataset ({split})",
        metadata={"split": split, "count": len(rubrics)},
    )


def _parse_criterion(data: dict[str, Any]) -> Criterion:
    """Parse a criterion from dict."""
    score_range = data.get("score_range", [0.0, 10.0])
    if isinstance(score_range, list) and len(score_range) >= 2:
        score_range = (float(score_range[0]), float(score_range[1]))
    else:
        score_range = (0.0, 10.0)

    return Criterion(
        name=data["name"],
        description=data["description"],
        weight=float(data.get("weight", 1.0)),
        score_range=score_range,
        keywords=data.get("keywords", []),
        examples=data.get("examples", {}),
    )


def _parse_rubric(data: dict[str, Any]) -> Rubric:
    """Parse a rubric from dict."""
    criteria = [_parse_criterion(c) for c in data.get("criteria", [])]

    return Rubric(
        id=data.get("id", str(uuid.uuid4())[:8]),
        name=data["name"],
        description=data.get("description", ""),
        criteria=criteria,
        question_types=data.get("question_types", []),
        reference_response=data.get("reference_response"),
        target_score=data.get("target_score"),
        metadata=data.get("metadata", {}),
    )


def _parse_rubricset(
    data: dict[str, Any],
    source_file: str | None = None,
) -> RubricSet:
    """Parse a rubricset from dict."""
    rubrics = [_parse_rubric(r) for r in data.get("rubrics", [])]

    metadata = data.get("metadata", {})
    if source_file:
        metadata["source_file"] = source_file

    return RubricSet(
        name=data.get("name", "Unnamed RubricSet"),
        rubrics=rubrics,
        description=data.get("description", ""),
        metadata=metadata,
    )


def _extract_keywords(text: str, min_length: int = 3, max_keywords: int = 20) -> List[str]:
    """Extract potential keywords from rubric text.

    Args:
        text: Text to extract keywords from
        min_length: Minimum word length to include
        max_keywords: Maximum number of keywords to return

    Returns:
        List of extracted keywords
    """
    text = text.lower()
    for ch in string.punctuation:
        text = text.replace(ch, " ")

    words = [w.strip() for w in text.split() if len(w.strip()) >= min_length]

    # Filter common stopwords
    stopwords = {
        "the", "and", "for", "that", "this", "with", "are", "was", "were",
        "have", "has", "had", "been", "being", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "not", "but",
        "from", "they", "them", "their", "there", "here", "where", "when",
        "what", "which", "who", "whom", "how", "why", "all", "any", "both",
        "each", "few", "more", "most", "other", "some", "such", "than",
        "too", "very", "just", "also", "now", "only", "then", "about",
    }

    keywords = [w for w in words if w not in stopwords]
    # Remove duplicates while preserving order
    seen = set()
    unique = []
    for kw in keywords:
        if kw not in seen:
            seen.add(kw)
            unique.append(kw)

    return unique[:max_keywords]
