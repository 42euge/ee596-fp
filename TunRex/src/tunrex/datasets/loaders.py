"""Data loaders for various sources (Kaggle, HuggingFace, TFDS)."""
# TODO: refactor into separate files per source if this grows too large
from __future__ import annotations

import csv
import os
import shutil
from pathlib import Path
from typing import TYPE_CHECKING, Any

import grain

if TYPE_CHECKING:
    from datasets import Dataset

# TODO: move to gsm8k module
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

# TODO: move to a config module
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


def get_dataset(
    data_dir: str,
    split: str = "train",
    source: str = "tfds",
    template: str | None = None,
    system_prompt: str | None = None,
    seed: int = 42,
) -> grain.MapDataset:
    """Load and preprocess GSM8K dataset as a grain MapDataset.

    Args:
        data_dir: Directory to store/load the dataset
        split: Dataset split ("train" or "test")
        source: Data source ("tfds" or "kaggle")
        template: Prompt template with {system_prompt} and {question} placeholders.
            If None, uses a default Gemma-style template.
        system_prompt: System prompt for the model. If None, uses a default
            reasoning-focused prompt.
        seed: Random seed for shuffling

    Returns:
        grain.MapDataset with "prompts", "question", and "answer" keys
    """
    if not os.path.exists(data_dir):
        os.makedirs(data_dir)

    if template is None:
        template = """<start_of_turn>user
{system_prompt}

{question}<end_of_turn>
<start_of_turn>model
"""

    if system_prompt is None:
        system_prompt = (
            "You are given a problem. First, think about the problem "
            "and provide your reasoning. Place it between <reasoning> and "
            "</reasoning>. Then, provide the final answer (i.e., just one "
            "numerical value) between <answer> and </answer>."
        )

    if source == "tfds":
        data = load_from_tfds(data_dir, split)
    elif source == "kaggle":
        data = load_from_kaggle(data_dir, split)
    else:
        raise ValueError(f"Unknown source: {source}")

    def _as_text(v: Any) -> str:
        return v if isinstance(v, str) else v.decode("utf-8")

    dataset = (
        grain.MapDataset.source(data)
        .shuffle(seed=seed)
        .map(
            lambda x: {
                # passed to model forward pass
                "prompts": template.format(
                    system_prompt=system_prompt,
                    question=_as_text(x["question"]),
                ),
                # passed to reward functions
                "question": _as_text(x["question"]),
                # passed to reward functions
                "answer": extract_hash_answer(_as_text(x["answer"])),
            }
        )
    )
    return dataset


def get_train_val_test_datasets(
    train_data_dir: str,
    test_data_dir: str,
    source: str = "tfds",
    batch_size: int = 1,
    num_batches: int | None = None,
    num_test_batches: int | None = None,
    train_fraction: float = 1.0,
    num_epochs: int = 1,
    template: str | None = None,
    system_prompt: str | None = None,
    seed: int = 42,
) -> tuple[grain.MapDataset, grain.MapDataset | None, grain.MapDataset]:
    """Load and prepare train, validation, and test datasets.

    Args:
        train_data_dir: Directory for training data
        test_data_dir: Directory for test data
        source: Data source ("tfds" or "kaggle")
        batch_size: Batch size for all datasets
        num_batches: Max number of training batches (None for all)
        num_test_batches: Max number of test batches (None for all)
        train_fraction: Fraction of data for training (rest goes to validation).
            If 1.0, no validation set is created.
        num_epochs: Number of epochs to repeat training data
        template: Prompt template (see get_dataset for details)
        system_prompt: System prompt (see get_dataset for details)
        seed: Random seed for shuffling

    Returns:
        Tuple of (train_dataset, val_dataset, test_dataset).
        val_dataset is None if train_fraction == 1.0
    """
    dataset = get_dataset(
        train_data_dir,
        split="train",
        source=source,
        template=template,
        system_prompt=system_prompt,
        seed=seed,
    ).batch(batch_size)

    if num_batches is not None:
        dataset = dataset[:num_batches]

    if train_fraction == 1.0:
        train_dataset = dataset.repeat(num_epochs)
        val_dataset = None
    else:
        split_idx = int(len(dataset) * train_fraction)
        train_dataset = dataset[:split_idx].repeat(num_epochs)
        val_dataset = dataset[split_idx:].repeat(num_epochs)

    test_dataset = get_dataset(
        test_data_dir,
        split="test",
        source=source,
        template=template,
        system_prompt=system_prompt,
        seed=seed,
    ).batch(batch_size)

    if num_test_batches is not None:
        test_dataset = test_dataset[:num_test_batches]

    return train_dataset, val_dataset, test_dataset
