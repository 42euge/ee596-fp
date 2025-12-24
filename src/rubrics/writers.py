"""Functions for saving rubrics to YAML files."""

from pathlib import Path

import yaml

from .models import Rubric, RubricSet


def save_rubric_to_yaml(rubric: Rubric, file_path: str | Path) -> Path:
    """Save a single rubric to a YAML file.

    Args:
        rubric: Rubric object to save
        file_path: Destination path

    Returns:
        Path to saved file
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = rubric.to_dict()

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return path


def save_rubricset_to_yaml(rubricset: RubricSet, file_path: str | Path) -> Path:
    """Save a RubricSet to a YAML file.

    Args:
        rubricset: RubricSet object to save
        file_path: Destination path

    Returns:
        Path to saved file
    """
    path = Path(file_path)
    path.parent.mkdir(parents=True, exist_ok=True)

    data = rubricset.to_dict()

    with open(path, "w") as f:
        yaml.dump(data, f, default_flow_style=False, sort_keys=False, allow_unicode=True)

    return path
