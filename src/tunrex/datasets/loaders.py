"""Data loaders for various sources (Kaggle, HuggingFace, TFDS)."""

from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:
    from datasets import Dataset


def extract_hash_answer(text: str) -> str | None:
    """Extract answer after #### delimiter in GSM8K format.

    Args:
        text: Raw answer text containing #### delimiter

    Returns:
        Extracted answer string or None if delimiter not found
    """
    if "####" not in text:
        return None
    return text.split("####")[1].strip()


def download_kaggle_dataset(target_dir: str = "./data/gsm8k") -> str:
    """Download GSM8K dataset from Kaggle.

    Args:
        target_dir: Directory to save the downloaded files

    Returns:
        Path to the directory containing the dataset
    """
    import kagglehub

    os.makedirs(target_dir, exist_ok=True)
    src = kagglehub.dataset_download("thedevastator/grade-school-math-8k-q-a")
    src = Path(src)
    dst = Path(target_dir)

    for csv_file in src.glob("*.csv"):
        shutil.copy2(csv_file, dst / csv_file.name)
        print(f"Copied {csv_file.name} -> {dst/csv_file.name}")

    return target_dir


def load_from_kaggle(data_dir: str, split: str = "train") -> list[dict[str, Any]]:
    """Load GSM8K dataset from Kaggle CSV files.

    Args:
        data_dir: Directory containing the CSV files
        split: Dataset split ("train" or "test")

    Returns:
        List of dictionaries with "question" and "answer" keys
    """
    kaggle_dir = download_kaggle_dataset(data_dir)
    file_name = f"main_{split}.csv"
    csv_path = os.path.join(kaggle_dir, file_name)

    data = []
    with open(csv_path, newline="", encoding="utf-8") as csvfile:
        reader = csv.DictReader(csvfile)
        for row in reader:
            data.append({
                "question": row["question"],
                "answer": row["answer"],
            })

    return data


def load_from_huggingface(
    dataset_name: str,
    split: str = "train",
    subset: str | None = None,
) -> "Dataset":
    """Load dataset from HuggingFace Hub.

    Args:
        dataset_name: Name of the dataset (e.g., "gsm8k", "OpenRubrics/OpenRubrics")
        split: Dataset split to load
        subset: Dataset subset/config (e.g., "main" for gsm8k)

    Returns:
        HuggingFace Dataset object
    """
    from datasets import load_dataset

    os.environ["HF_HUB_DISABLE_XET"] = "1"

    if subset:
        return load_dataset(dataset_name, subset, split=split)
    return load_dataset(dataset_name, split=split)


def load_from_tfds(data_dir: str, split: str = "train"):
    """Load GSM8K dataset from TensorFlow Datasets.

    Args:
        data_dir: Directory for dataset cache
        split: Dataset split to load

    Returns:
        TFDS data source
    """
    import tensorflow_datasets as tfds
    import tensorflow_datasets.text.gsm8k  # noqa: F401 - Required for registration

    return tfds.data_source(
        "gsm8k",
        split=split,
        data_dir=data_dir,
        builder_kwargs={"file_format": tfds.core.FileFormat.ARRAY_RECORD},
        download=True,
    )


def _normalize_value(value: Any) -> str:
    """Normalize various value types to string format.

    Used for OpenRubrics dataset field normalization.
    """
    if value is None:
        return ""
    if isinstance(value, (list, tuple)):
        return "\n".join(str(v) for v in value if v)
    if isinstance(value, dict):
        parts = []
        for k, v in value.items():
            if isinstance(v, (dict, list, tuple)):
                parts.append(f"{k}: {_normalize_value(v)}")
            elif v:
                parts.append(f"{k}: {v}")
        return "\n".join(parts)
    return str(value)


def _pick_column(columns: list[str], options: list[str], default: str = "") -> str:
    """Pick first available column name from options."""
    for name in options:
        if name in columns:
            return name
    return default


def load_openrubrics(
    split: str = "train",
    max_examples: int | None = None,
) -> list[dict[str, Any]]:
    """Load OpenRubrics dataset from HuggingFace with flexible column mapping.

    Args:
        split: Dataset split to load
        max_examples: Maximum number of examples to load

    Returns:
        List of processed examples with standardized keys
    """
    from datasets import load_dataset

    try:
        raw_ds = load_dataset("OpenRubrics/OpenRubrics", split=split)
    except Exception as e:
        print(f"Could not load OpenRubrics split {split}: {e}")
        return []

    columns = list(raw_ds.column_names)

    # Flexible column mapping
    question_col = _pick_column(
        columns,
        ["prompt", "question", "instruction", "query", "student_input"],
        "prompt",
    )
    rubric_col = _pick_column(
        columns,
        ["rubric", "rubrics", "grading_rubric", "rubric_text"],
        "rubric",
    )
    reference_col = _pick_column(
        columns,
        ["reference_answer", "target", "gold_answer", "reference", "ideal_answer"],
        "",
    )
    response_col = _pick_column(
        columns,
        ["response", "model_answer", "answer", "output", "student_answer"],
        "",
    )
    score_col = _pick_column(
        columns,
        ["score", "rubric_score", "rating", "grade", "normalized_score"],
        "",
    )

    processed = []
    for row in raw_ds:
        question = str(row.get(question_col, "")).strip()
        rubric_text = _normalize_value(row.get(rubric_col, ""))
        reference = str(row.get(reference_col, "") or row.get(response_col, "") or "").strip()
        target_score = row.get(score_col) if score_col else None

        if not question:
            continue

        processed.append({
            "question": question,
            "answer": reference,
            "rubric": rubric_text,
            "reference_response": reference,
            "target_score": target_score,
        })

        if max_examples and len(processed) >= max_examples:
            break

    print(f"Loaded {len(processed)} examples from OpenRubrics ({split})")
    return processed
