"""HuggingFace local model backend.

Reuses patterns from src/model.py (GemmaModel class).
Supports CUDA, MPS (Apple Silicon), and CPU devices.
"""

import os
from typing import List, Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from .base import LLMBackend
from ..config import GenerationConfig, LLMResponse


def get_device(preferred: str = "auto") -> str:
    """Determine the best available device.

    Args:
        preferred: Preferred device ("auto", "cuda", "mps", "cpu")

    Returns:
        Device string for PyTorch
    """
    if preferred != "auto":
        return preferred

    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class HuggingFaceBackend(LLMBackend):
    """Local HuggingFace model backend.

    Supports:
    - CUDA (NVIDIA GPUs)
    - MPS (Apple Silicon)
    - CPU (fallback)
    - 8-bit and 4-bit quantization (CUDA only)
    """

    def __init__(
        self,
        model_id: str = "google/gemma-3-1b-it",
        device: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
        **kwargs,
    ):
        """Initialize the HuggingFace backend.

        Args:
            model_id: HuggingFace model ID or local path
            device: Device to use ("auto", "cuda", "mps", "cpu")
            load_in_8bit: Use 8-bit quantization (requires bitsandbytes, CUDA only)
            load_in_4bit: Use 4-bit quantization (requires bitsandbytes, CUDA only)
        """
        self.model_id = model_id
        self._device = get_device(device)
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit

        self._model = None
        self._tokenizer = None
        self._loaded = False

    def _load(self) -> None:
        """Lazy load model and tokenizer."""
        if self._loaded:
            return

        print(f"Loading {self.model_id} on device: {self._device}")

        # Load tokenizer
        self._tokenizer = AutoTokenizer.from_pretrained(
            self.model_id,
            token=os.getenv("HF_TOKEN"),
        )
        if self._tokenizer.pad_token is None:
            self._tokenizer.pad_token = self._tokenizer.eos_token

        # Configure model loading
        model_kwargs = {
            "torch_dtype": torch.float16 if self._device != "cpu" else torch.float32,
            "device_map": "auto" if self._device == "cuda" else None,
            "token": os.getenv("HF_TOKEN"),
            "trust_remote_code": True,
        }

        # Add quantization config if needed (CUDA only)
        if (self.load_in_8bit or self.load_in_4bit) and self._device == "cuda":
            try:
                from transformers import BitsAndBytesConfig

                if self.load_in_4bit:
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_4bit=True,
                        bnb_4bit_compute_dtype=torch.float16,
                        bnb_4bit_quant_type="nf4",
                        bnb_4bit_use_double_quant=True,
                    )
                else:
                    model_kwargs["quantization_config"] = BitsAndBytesConfig(
                        load_in_8bit=True
                    )
            except ImportError:
                print("Warning: bitsandbytes not available, skipping quantization")

        self._model = AutoModelForCausalLM.from_pretrained(
            self.model_id,
            **model_kwargs,
        )

        # Move to device if not using device_map="auto"
        if self._device != "cuda" and hasattr(self._model, "to"):
            self._model = self._model.to(self._device)

        self._model.eval()
        self._loaded = True
        print(f"Model loaded successfully on {self._device}")

    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> LLMResponse:
        """Generate a single response.

        Args:
            prompt: Input prompt
            config: Generation configuration

        Returns:
            LLMResponse with generated text
        """
        self._load()
        config = config or GenerationConfig()

        inputs = self._tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self._device) for k, v in inputs.items()}
        input_length = inputs["input_ids"].shape[1]

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=config.max_new_tokens,
                temperature=config.temperature if config.temperature > 0 else 1.0,
                top_p=config.top_p,
                top_k=config.top_k,
                do_sample=config.temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
            )

        generated = outputs[0][input_length:]
        text = self._tokenizer.decode(generated, skip_special_tokens=True)

        return LLMResponse(
            text=text,
            finish_reason="stop",
            usage={
                "prompt_tokens": input_length,
                "completion_tokens": len(generated),
            },
        )

    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
    ) -> List[LLMResponse]:
        """Generate responses for multiple prompts.

        Note: For HuggingFace, this uses sequential generation.
        For true batch generation with padding, consider vLLM backend.

        Args:
            prompts: List of input prompts
            config: Generation configuration

        Returns:
            List of LLMResponse objects
        """
        return [self.generate(p, config) for p in prompts]

    def is_available(self) -> bool:
        """Check if this backend is available."""
        try:
            self._load()
            return True
        except Exception as e:
            print(f"HuggingFace backend not available: {e}")
            return False

    @property
    def backend_name(self) -> str:
        """Return the name of this backend."""
        return f"huggingface:{self.model_id}"
