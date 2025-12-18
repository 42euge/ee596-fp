"""
Data formatting utilities for TUI display
"""

import json
import numpy as np
from typing import Any, Dict


def truncate_text(text: str, max_length: int = 50) -> str:
    """Truncate text to max_length with ellipsis"""
    if len(text) <= max_length:
        return text
    return text[:max_length - 3] + "..."


def format_value(value: Any, max_length: int = 50) -> str:
    """
    Format a value for display in a table cell

    Args:
        value: Value to format
        max_length: Maximum length of formatted string

    Returns:
        Formatted string
    """
    if value is None:
        return ""

    # Handle NumPy arrays
    if isinstance(value, np.ndarray):
        # Convert to list for JSON serialization
        value = value.tolist()

    if isinstance(value, (dict, list)):
        # Convert complex types to JSON string
        json_str = json.dumps(value, ensure_ascii=False)
        return truncate_text(json_str, max_length)

    if isinstance(value, bool):
        return "✓" if value else "✗"

    if isinstance(value, (int, float)):
        return str(value)

    # Default: convert to string and truncate
    return truncate_text(str(value), max_length)


def format_file_size(size_bytes: int) -> str:
    """Format byte size to human-readable format"""
    for unit in ['B', 'KB', 'MB', 'GB', 'TB']:
        if size_bytes < 1024.0:
            return f"{size_bytes:.1f} {unit}"
        size_bytes /= 1024.0
    return f"{size_bytes:.1f} PB"


def extract_reasoning_answer(text: str) -> Dict[str, str]:
    """
    Extract reasoning and answer from formatted text

    Args:
        text: Text containing <reasoning> and <answer> tags

    Returns:
        Dict with 'reasoning' and 'answer' keys
    """
    result = {'reasoning': '', 'answer': ''}

    # Extract reasoning
    if '<reasoning>' in text and '</reasoning>' in text:
        start = text.index('<reasoning>') + len('<reasoning>')
        end = text.index('</reasoning>')
        result['reasoning'] = text[start:end].strip()

    # Extract answer
    if '<answer>' in text and '</answer>' in text:
        start = text.index('<answer>') + len('<answer>')
        end = text.index('</answer>')
        result['answer'] = text[start:end].strip()

    return result


def format_json_pretty(data: Any, indent: int = 2) -> str:
    """Format data as pretty-printed JSON"""
    def numpy_encoder(obj):
        """Custom encoder for NumPy types"""
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        if isinstance(obj, np.bool_):
            return bool(obj)
        raise TypeError(f"Object of type {type(obj)} is not JSON serializable")

    return json.dumps(data, indent=indent, ensure_ascii=False, default=numpy_encoder)
