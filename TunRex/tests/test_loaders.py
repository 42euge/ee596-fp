"""Tests for tunrex.datasets.loaders module.

Note: These tests mock the grain dependency to avoid import errors.
"""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path

# Mock grain before importing loaders to avoid import error
sys.modules["grain"] = MagicMock()

# Import directly from the module file
tunrex_src = Path(__file__).parent.parent / "src" / "tunrex" / "datasets"
import importlib.util
loaders_spec = importlib.util.spec_from_file_location("loaders", tunrex_src / "loaders.py")
loaders_module = importlib.util.module_from_spec(loaders_spec)
sys.modules["tunrex.datasets.loaders"] = loaders_module
loaders_spec.loader.exec_module(loaders_module)

from tunrex.datasets.loaders import (
    extract_hash_answer,
    load_from_huggingface,
    load_openrubrics,
)


class TestExtractHashAnswer:
    """Tests for extract_hash_answer function."""
    # TODO: add some examples from real datasets as docstrings so we can see real use cases

    def test_valid_hash_answer(self):
        """Test extracting answer after #### delimiter."""
        result = extract_hash_answer("Some reasoning #### 42")
        assert result == "42"

    def test_with_whitespace(self):
        """Test extracting answer handles whitespace."""
        result = extract_hash_answer("Calculation steps ####   answer here  ")
        assert result == "answer here"

    def test_missing_delimiter(self):
        """Test returns None when delimiter is missing."""
        result = extract_hash_answer("No delimiter here")
        assert result is None

    def test_empty_after_delimiter(self):
        """Test handles empty string after delimiter."""
        result = extract_hash_answer("Text #### ")
        assert result == ""

    def test_multiple_hashes(self):
        """Test splits on first #### and takes second part."""
        # Note: The function uses split("####")[1] which splits all and takes second part
        result = extract_hash_answer("Step 1 #### 42 #### extra")
        assert result == "42"

    def test_complex_reasoning(self):
        """Test with complex reasoning text."""
        text = """
        Step 1: Calculate 5 * 3 = 15
        Step 2: Add 7 to get 22
        Step 3: Divide by 2 = 11
        #### 11
        """
        result = extract_hash_answer(text)
        assert result == "11"

    def test_numeric_answer(self):
        """Test with numeric answer."""
        result = extract_hash_answer("The total is #### 1234567")
        assert result == "1234567"

    def test_negative_number(self):
        """Test with negative number answer."""
        result = extract_hash_answer("Result #### -42")
        assert result == "-42"

    def test_decimal_number(self):
        """Test with decimal number answer."""
        result = extract_hash_answer("Price #### 19.99")
        assert result == "19.99"


class TestLoadFromHuggingface:
    """Tests for load_from_huggingface function."""

    @pytest.mark.skip(reason="Requires tunrex package installed for mock patch path")
    def test_basic_loading(self):
        """Test basic dataset loading."""
        pass

    @pytest.mark.skip(reason="Requires tunrex package installed for mock patch path")
    def test_with_subset(self):
        """Test loading with subset specified."""
        pass

    def test_function_signature(self):
        """Test function has expected parameters."""
        import inspect
        sig = inspect.signature(load_from_huggingface)
        params = list(sig.parameters.keys())

        assert "dataset_name" in params
        assert "split" in params


class TestLoadOpenRubrics:
    """Tests for load_openrubrics function."""

    def test_function_signature(self):
        """Test function has expected parameters."""
        import inspect
        sig = inspect.signature(load_openrubrics)
        params = list(sig.parameters.keys())

        assert "split" in params
        assert "max_examples" in params

    def test_default_parameters(self):
        """Test default parameter values."""
        import inspect
        sig = inspect.signature(load_openrubrics)
        params = sig.parameters

        # Check split default
        if "split" in params:
            assert params["split"].default in ("train", inspect.Parameter.empty)

        # Check max_examples has a default
        if "max_examples" in params:
            assert params["max_examples"].default != inspect.Parameter.empty

    @pytest.mark.skip(reason="Requires network access")
    def test_load_openrubrics_returns_list(self):
        """Test that load_openrubrics returns a list of dicts."""
        data = load_openrubrics(max_examples=5)
        assert isinstance(data, list)
        if len(data) > 0:
            assert isinstance(data[0], dict)
            assert "question" in data[0] or "instruction" in data[0]

    @pytest.mark.skip(reason="Requires tunrex package installed for mock patch path")
    def test_openrubrics_with_mock(self):
        """Test OpenRubrics loading with mocked HuggingFace."""
        pass


class TestLoaderEdgeCases:
    """Tests for edge cases in loader functions."""

    def test_extract_hash_answer_unicode(self):
        """Test extract_hash_answer with unicode."""
        result = extract_hash_answer("计算 #### 42")
        assert result == "42"

    def test_extract_hash_answer_special_chars(self):
        """Test extract_hash_answer with special characters."""
        result = extract_hash_answer("Result: $100.50 total #### $100.50")
        assert result == "$100.50"

    def test_extract_hash_answer_empty_input(self):
        """Test extract_hash_answer with empty input."""
        result = extract_hash_answer("")
        assert result is None

    def test_extract_hash_answer_only_delimiter(self):
        """Test extract_hash_answer with only delimiter."""
        result = extract_hash_answer("####")
        assert result == ""

    def test_extract_hash_answer_newlines(self):
        """Test extract_hash_answer preserves newlines in answer."""
        result = extract_hash_answer("Work #### Line1\nLine2")
        assert "Line1" in result
        assert "Line2" in result
