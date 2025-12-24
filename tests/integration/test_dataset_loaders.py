"""Integration tests for dataset loaders with mocked dependencies.

Note: We mock torch before importing to avoid requiring it in CI.
"""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock, mock_open
import os
import csv
import io

# Mock torch and its submodules before importing src (which imports src.model which needs torch)
sys.modules["torch"] = MagicMock()
sys.modules["torch.cuda"] = MagicMock()
sys.modules["torch.backends"] = MagicMock()
sys.modules["torch.backends.mps"] = MagicMock()
sys.modules["transformers"] = MagicMock()

from src.utils import load_gsm8k_dataset, load_openrubrics_dataset


class TestLoadGSM8KDataset:
    """Tests for load_gsm8k_dataset function with mocked file system."""

    def test_load_basic(self, tmp_path):
        """Test basic CSV loading."""
        # Create a temporary CSV file
        csv_file = tmp_path / "main_train.csv"
        csv_content = "question,answer\n" \
                      "What is 2+2?,The sum is 4 #### 4\n" \
                      "What is 3+3?,Total: 6 #### 6\n"
        csv_file.write_text(csv_content)

        result = load_gsm8k_dataset(
            data_dir=str(tmp_path),
            split="train",
        )

        assert len(result) == 2
        assert result[0]["question"] == "What is 2+2?"
        assert result[0]["answer"] == "4"
        assert result[0]["source"] == "gsm8k"

    def test_load_with_max_examples(self, tmp_path):
        """Test loading with max_examples limit."""
        csv_file = tmp_path / "main_train.csv"
        rows = [f"Q{i}?,Answer #### {i}" for i in range(10)]
        csv_content = "question,answer\n" + "\n".join(rows)
        csv_file.write_text(csv_content)

        result = load_gsm8k_dataset(
            data_dir=str(tmp_path),
            split="train",
            max_examples=3,
        )

        assert len(result) == 3

    def test_load_different_splits(self, tmp_path):
        """Test loading train vs test splits."""
        # Create train file
        train_file = tmp_path / "main_train.csv"
        train_file.write_text("question,answer\nTrain Q?,A #### 1\n")

        # Create test file
        test_file = tmp_path / "main_test.csv"
        test_file.write_text("question,answer\nTest Q?,A #### 2\n")

        train_result = load_gsm8k_dataset(data_dir=str(tmp_path), split="train")
        test_result = load_gsm8k_dataset(data_dir=str(tmp_path), split="test")

        assert train_result[0]["question"] == "Train Q?"
        assert test_result[0]["question"] == "Test Q?"

    def test_file_not_found(self, tmp_path):
        """Test FileNotFoundError when CSV doesn't exist."""
        with pytest.raises(FileNotFoundError, match="GSM8K dataset not found"):
            load_gsm8k_dataset(data_dir=str(tmp_path), split="train")

    def test_answer_without_hash(self, tmp_path):
        """Test handling answers without #### delimiter."""
        csv_file = tmp_path / "main_train.csv"
        csv_file.write_text("question,answer\nQ?,42\n")

        result = load_gsm8k_dataset(data_dir=str(tmp_path), split="train")

        assert result[0]["answer"] == "42"

    def test_deterministic_sampling(self, tmp_path):
        """Test that sampling is deterministic with same seed."""
        csv_file = tmp_path / "main_train.csv"
        rows = [f"Q{i}?,A #### {i}" for i in range(100)]
        csv_content = "question,answer\n" + "\n".join(rows)
        csv_file.write_text(csv_content)

        result1 = load_gsm8k_dataset(
            data_dir=str(tmp_path),
            split="train",
            max_examples=10,
            seed=42,
        )
        result2 = load_gsm8k_dataset(
            data_dir=str(tmp_path),
            split="train",
            max_examples=10,
            seed=42,
        )

        assert result1 == result2

    def test_different_seeds_different_samples(self, tmp_path):
        """Test that different seeds produce different samples."""
        csv_file = tmp_path / "main_train.csv"
        rows = [f"Q{i}?,A #### {i}" for i in range(100)]
        csv_content = "question,answer\n" + "\n".join(rows)
        csv_file.write_text(csv_content)

        result1 = load_gsm8k_dataset(
            data_dir=str(tmp_path),
            split="train",
            max_examples=10,
            seed=42,
        )
        result2 = load_gsm8k_dataset(
            data_dir=str(tmp_path),
            split="train",
            max_examples=10,
            seed=123,
        )

        # Very unlikely to be the same with different seeds
        assert result1 != result2


class TestLoadOpenRubricsDataset:
    """Tests for load_openrubrics_dataset function with mocked HuggingFace."""

    @patch("datasets.load_dataset")
    def test_load_basic(self, mock_load_dataset):
        """Test basic OpenRubrics loading."""
        mock_data = Mock()
        mock_data.column_names = ["instruction", "rubric", "response", "score"]
        mock_data.__iter__ = Mock(return_value=iter([
            {
                "instruction": "Question 1",
                "rubric": "Rubric 1",
                "response": "Response 1",
                "score": 8,
            },
            {
                "instruction": "Question 2",
                "rubric": "Rubric 2",
                "response": "Response 2",
                "score": 9,
            },
        ]))
        mock_load_dataset.return_value = mock_data

        result = load_openrubrics_dataset(split="train", max_examples=10)

        assert len(result) == 2
        assert result[0]["source"] == "openrubrics"
        mock_load_dataset.assert_called_once()

    @patch("datasets.load_dataset")
    def test_load_with_max_examples(self, mock_load_dataset):
        """Test loading with max_examples limit."""
        mock_data = Mock()
        mock_data.column_names = ["instruction", "rubric", "response", "score"]
        mock_data.__iter__ = Mock(return_value=iter([
            {"instruction": f"Q{i}", "rubric": "", "response": "", "score": i}
            for i in range(100)
        ]))
        mock_load_dataset.return_value = mock_data

        result = load_openrubrics_dataset(max_examples=5)

        assert len(result) == 5

    @patch("datasets.load_dataset")
    def test_column_mapping(self, mock_load_dataset):
        """Test that various column names are handled."""
        mock_data = Mock()
        # Use alternative column names
        mock_data.column_names = ["prompt", "scoring_rubric", "model_answer", "rating"]
        mock_data.__iter__ = Mock(return_value=iter([
            {
                "prompt": "My Question",
                "scoring_rubric": "My Rubric",
                "model_answer": "My Answer",
                "rating": 7,
            },
        ]))
        mock_load_dataset.return_value = mock_data

        result = load_openrubrics_dataset()

        assert len(result) == 1
        assert result[0]["question"] == "My Question"

    @patch("datasets.load_dataset")
    def test_handles_load_error(self, mock_load_dataset):
        """Test graceful handling of load errors."""
        mock_load_dataset.side_effect = Exception("Network error")

        result = load_openrubrics_dataset()

        assert result == []

    @patch("datasets.load_dataset")
    def test_skips_empty_questions(self, mock_load_dataset):
        """Test that entries with empty questions are skipped."""
        mock_data = Mock()
        mock_data.column_names = ["instruction", "rubric", "response", "score"]
        mock_data.__iter__ = Mock(return_value=iter([
            {"instruction": "", "rubric": "R1", "response": "A1", "score": 5},
            {"instruction": "Valid Q", "rubric": "R2", "response": "A2", "score": 6},
        ]))
        mock_load_dataset.return_value = mock_data

        result = load_openrubrics_dataset()

        # Only the non-empty question should be included
        assert len(result) == 1
        assert result[0]["question"] == "Valid Q"

    @patch("datasets.load_dataset")
    def test_normalizes_list_values(self, mock_load_dataset):
        """Test that list values are normalized to strings."""
        mock_data = Mock()
        mock_data.column_names = ["instruction", "rubric", "response", "score"]
        mock_data.__iter__ = Mock(return_value=iter([
            {
                "instruction": "Question",
                "rubric": ["Point 1", "Point 2"],
                "response": "Answer",
                "score": 5,
            },
        ]))
        mock_load_dataset.return_value = mock_data

        result = load_openrubrics_dataset()

        assert len(result) == 1
        assert "Point 1" in result[0]["rubric"]
        assert "Point 2" in result[0]["rubric"]

    @patch("datasets.load_dataset")
    def test_normalizes_dict_values(self, mock_load_dataset):
        """Test that dict values are normalized to strings."""
        mock_data = Mock()
        mock_data.column_names = ["instruction", "rubric", "response", "score"]
        mock_data.__iter__ = Mock(return_value=iter([
            {
                "instruction": "Question",
                "rubric": {"criterion1": "Value1", "criterion2": "Value2"},
                "response": "Answer",
                "score": 5,
            },
        ]))
        mock_load_dataset.return_value = mock_data

        result = load_openrubrics_dataset()

        assert len(result) == 1
        assert "criterion1" in result[0]["rubric"]
        assert "Value1" in result[0]["rubric"]


class TestDataLoaderEdgeCases:
    """Tests for edge cases in data loading."""

    def test_gsm8k_unicode(self, tmp_path):
        """Test GSM8K loading with unicode characters."""
        csv_file = tmp_path / "main_train.csv"
        csv_file.write_text("question,answer\nWhat is π?,3.14159 #### 3.14159\n", encoding="utf-8")

        result = load_gsm8k_dataset(data_dir=str(tmp_path), split="train")

        assert "π" in result[0]["question"]

    def test_gsm8k_empty_file(self, tmp_path):
        """Test GSM8K loading with empty file (headers only)."""
        csv_file = tmp_path / "main_train.csv"
        csv_file.write_text("question,answer\n")

        result = load_gsm8k_dataset(data_dir=str(tmp_path), split="train")

        assert result == []

    def test_gsm8k_whitespace_answer(self, tmp_path):
        """Test GSM8K loading with whitespace in answers."""
        csv_file = tmp_path / "main_train.csv"
        csv_file.write_text("question,answer\nQ?,Work ####   42   \n")

        result = load_gsm8k_dataset(data_dir=str(tmp_path), split="train")

        assert result[0]["answer"] == "42"
