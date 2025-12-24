"""Integration tests for src/model.py with mocked dependencies.

Note: We create a mock torch module that provides basic tensor functionality
needed for the tests, since torch is not installed in CI.
"""

import pytest
import sys
from unittest.mock import Mock, patch, MagicMock

# Create a mock torch module with tensor support
class MockTensor:
    def __init__(self, data):
        self.data = data
    def __getitem__(self, idx):
        return self.data[idx] if isinstance(self.data, list) else self.data

class MockTorch:
    @staticmethod
    def tensor(data):
        return MockTensor(data)

    class cuda:
        @staticmethod
        def is_available():
            return False

    class backends:
        class mps:
            @staticmethod
            def is_available():
                return False

# Install mock torch before any imports
mock_torch = MockTorch()
mock_torch.cuda = MockTorch.cuda
mock_torch.backends = MockTorch.backends
sys.modules["torch"] = mock_torch
sys.modules["torch.cuda"] = mock_torch.cuda
sys.modules["torch.backends"] = mock_torch.backends
sys.modules["torch.backends.mps"] = mock_torch.backends.mps
sys.modules["transformers"] = MagicMock()
sys.modules["peft"] = MagicMock()

# Now we can import - use the mock torch for tensor creation in tests
import torch

from src.model import get_device, GemmaModel, load_model


class TestGetDevice:
    """Tests for get_device function."""

    def test_explicit_cuda(self):
        """Test explicit CUDA device selection."""
        result = get_device("cuda")
        assert result == "cuda"

    def test_explicit_mps(self):
        """Test explicit MPS device selection."""
        result = get_device("mps")
        assert result == "mps"

    def test_explicit_cpu(self):
        """Test explicit CPU device selection."""
        result = get_device("cpu")
        assert result == "cpu"

    @patch("torch.cuda.is_available")
    def test_auto_selects_cuda(self, mock_cuda):
        """Test auto selects CUDA when available."""
        mock_cuda.return_value = True
        result = get_device("auto")
        assert result == "cuda"

    @patch("torch.cuda.is_available")
    @patch("torch.backends.mps.is_available")
    def test_auto_selects_mps(self, mock_mps, mock_cuda):
        """Test auto selects MPS when CUDA unavailable."""
        mock_cuda.return_value = False
        mock_mps.return_value = True
        result = get_device("auto")
        assert result == "mps"

    @patch("torch.cuda.is_available")
    @patch("torch.backends.mps.is_available")
    def test_auto_fallback_cpu(self, mock_mps, mock_cuda):
        """Test auto falls back to CPU."""
        mock_cuda.return_value = False
        mock_mps.return_value = False
        result = get_device("auto")
        assert result == "cpu"


class TestGemmaModelInit:
    """Tests for GemmaModel initialization."""

    def test_default_init(self):
        """Test default initialization."""
        model = GemmaModel()

        assert model.checkpoint_path is None
        assert model.load_in_8bit is False
        assert model.load_in_4bit is False
        assert model.model is None
        assert model.tokenizer is None
        assert model._loaded is False

    def test_custom_init(self):
        """Test custom initialization."""
        model = GemmaModel(
            checkpoint_path="/path/to/checkpoint",
            device="cpu",
            load_in_8bit=True,
        )

        assert model.checkpoint_path == "/path/to/checkpoint"
        assert model.device == "cpu"
        assert model.load_in_8bit is True

    @patch("torch.cuda.is_available", return_value=False)
    @patch("torch.backends.mps.is_available", return_value=False)
    def test_auto_device_cpu(self, mock_mps, mock_cuda):
        """Test auto device selection falls back to CPU."""
        model = GemmaModel(device="auto")
        assert model.device == "cpu"


class TestGemmaModelLoad:
    """Tests for GemmaModel.load() with mocked HuggingFace."""

    @patch("src.model.HAS_GEMMA3_CLASS", False)
    @patch("src.model.AutoTokenizer")
    @patch("src.model.AutoModelForCausalLM")
    def test_load_basic(self, mock_model_class, mock_tokenizer_class):
        """Test basic model loading."""
        # Setup mocks
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model

        # Create and load model
        model = GemmaModel(device="cpu")
        model.load()

        # Verify
        assert model._loaded is True
        assert model.tokenizer is not None
        assert model.model is not None
        mock_tokenizer_class.from_pretrained.assert_called_once()
        mock_model_class.from_pretrained.assert_called_once()

    @patch("src.model.AutoTokenizer")
    @patch("src.model.AutoModelForCausalLM")
    def test_load_sets_pad_token(self, mock_model_class, mock_tokenizer_class):
        """Test that pad_token is set to eos_token if None."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = None
        mock_tokenizer.eos_token = "<eos>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model

        model = GemmaModel(device="cpu")
        model.load()

        assert model.tokenizer.pad_token == "<eos>"

    @patch("src.model.AutoTokenizer")
    @patch("src.model.AutoModelForCausalLM")
    def test_load_idempotent(self, mock_model_class, mock_tokenizer_class):
        """Test that load() is idempotent."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model

        model = GemmaModel(device="cpu")
        model.load()
        model.load()  # Second call should be no-op

        # Should only be called once
        assert mock_tokenizer_class.from_pretrained.call_count == 1

    @patch("src.model.AutoTokenizer")
    def test_load_gated_model_error(self, mock_tokenizer_class):
        """Test handling of gated model access error."""
        mock_tokenizer_class.from_pretrained.side_effect = OSError(
            "gated repo - need access"
        )

        model = GemmaModel(device="cpu")

        with pytest.raises(RuntimeError, match="gated model"):
            model.load()


class TestGemmaModelGenerate:
    """Tests for GemmaModel.generate() with mocked model."""

    @patch("src.model.HAS_GEMMA3_CLASS", False)
    @patch("src.model.AutoTokenizer")
    @patch("src.model.AutoModelForCausalLM")
    def test_generate_basic(self, mock_model_class, mock_tokenizer_class):
        """Test basic generation."""
        # Setup tokenizer mock
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_tokenizer.decode.return_value = "Generated response"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Setup model mock
        mock_model = MagicMock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6]])
        mock_model_class.from_pretrained.return_value = mock_model

        # Create model and generate
        model = GemmaModel(device="cpu")
        result = model.generate("Test prompt")

        assert result == "Generated response"
        # Verify the model's generate was called (the actual model object after .to() might differ)
        assert model.model.generate.called or mock_model.generate.called

    @patch("src.model.AutoTokenizer")
    @patch("src.model.AutoModelForCausalLM")
    def test_generate_auto_loads(self, mock_model_class, mock_tokenizer_class):
        """Test that generate() auto-loads model if not loaded."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_tokenizer.decode.return_value = "Response"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6]])
        mock_model_class.from_pretrained.return_value = mock_model

        model = GemmaModel(device="cpu")
        assert model._loaded is False

        model.generate("Test prompt")

        assert model._loaded is True


class TestGemmaModelGenerateBatch:
    """Tests for GemmaModel.generate_batch() with mocked model."""

    @patch("src.model.AutoTokenizer")
    @patch("src.model.AutoModelForCausalLM")
    def test_generate_batch(self, mock_model_class, mock_tokenizer_class):
        """Test batch generation."""
        # Setup tokenizer mock
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3], [1, 2, 0]]),
            "attention_mask": torch.tensor([[1, 1, 1], [1, 1, 0]]),
        }
        mock_tokenizer.decode.side_effect = ["Response 1", "Response 2"]
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Setup model mock
        mock_model = MagicMock()
        mock_model.generate.return_value = torch.tensor([
            [1, 2, 3, 4, 5, 6],
            [1, 2, 0, 7, 8, 9],
        ])
        mock_model_class.from_pretrained.return_value = mock_model

        # Create model and generate batch
        model = GemmaModel(device="cpu")
        results = model.generate_batch(["Prompt 1", "Prompt 2"])

        assert len(results) == 2
        assert results[0] == "Response 1"
        assert results[1] == "Response 2"


class TestGemmaModelSolve:
    """Tests for GemmaModel.solve() with mocked model."""

    @patch("src.model.AutoTokenizer")
    @patch("src.model.AutoModelForCausalLM")
    def test_solve_extracts_reasoning_and_answer(self, mock_model_class, mock_tokenizer_class):
        """Test that solve() extracts reasoning and answer."""
        # Setup tokenizer mock
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_tokenizer.decode.return_value = (
            "<reasoning>Step 1: 5+5=10</reasoning><answer>10</answer>"
        )
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        # Setup model mock
        mock_model = MagicMock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6]])
        mock_model_class.from_pretrained.return_value = mock_model

        # Create model and solve
        model = GemmaModel(device="cpu")
        result = model.solve("What is 5 + 5?")

        assert "reasoning" in result
        assert "answer" in result
        assert "prompt" in result
        assert "response" in result
        assert result["reasoning"] == "Step 1: 5+5=10"
        assert result["answer"] == "10"

    @patch("src.model.AutoTokenizer")
    @patch("src.model.AutoModelForCausalLM")
    def test_solve_handles_missing_tags(self, mock_model_class, mock_tokenizer_class):
        """Test that solve() handles missing tags gracefully."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer.pad_token_id = 0
        mock_tokenizer.eos_token_id = 1
        mock_tokenizer.return_value = {
            "input_ids": torch.tensor([[1, 2, 3]]),
            "attention_mask": torch.tensor([[1, 1, 1]]),
        }
        mock_tokenizer.decode.return_value = "Just a plain response without tags"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model.generate.return_value = torch.tensor([[1, 2, 3, 4, 5, 6]])
        mock_model_class.from_pretrained.return_value = mock_model

        model = GemmaModel(device="cpu")
        result = model.solve("What is 5 + 5?")

        assert result["reasoning"] == ""
        assert result["answer"] == ""


class TestLoadModelFunction:
    """Tests for load_model convenience function."""

    @patch("src.model.AutoTokenizer")
    @patch("src.model.AutoModelForCausalLM")
    def test_load_model_returns_loaded_model(self, mock_model_class, mock_tokenizer_class):
        """Test that load_model returns a loaded model."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model

        model = load_model(device="cpu")

        assert isinstance(model, GemmaModel)
        assert model._loaded is True

    @patch("src.model.AutoTokenizer")
    @patch("src.model.AutoModelForCausalLM")
    def test_load_model_with_checkpoint(self, mock_model_class, mock_tokenizer_class):
        """Test load_model with checkpoint path."""
        mock_tokenizer = MagicMock()
        mock_tokenizer.pad_token = "<pad>"
        mock_tokenizer_class.from_pretrained.return_value = mock_tokenizer

        mock_model = MagicMock()
        mock_model_class.from_pretrained.return_value = mock_model

        model = load_model(checkpoint_path="/fake/path", device="cpu")

        assert model.checkpoint_path == "/fake/path"
