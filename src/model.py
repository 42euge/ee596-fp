"""
Model loading and inference for Gemma3-1B

Supports:
- Local inference with PyTorch on CUDA/MPS/CPU
- HuggingFace transformers backend for cross-platform compatibility

For training, the original tunix library is used (requires TPU/Colab).
For inference, we use transformers for better local device support.
"""

import os
from typing import Optional, Tuple, List, Dict, Any
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
try:
    from transformers import Gemma3ForCausalLM
    HAS_GEMMA3_CLASS = True
except ImportError:
    HAS_GEMMA3_CLASS = False
from peft import PeftModel, LoraConfig, get_peft_model

# Import ecolab for Colab/Kaggle environment support
try:
    from etils import ecolab
    HAS_ECOLAB = True
except ImportError:
    HAS_ECOLAB = False
    ecolab = None

from .config import (
    Config, get_default_config, format_prompt, get_system_prompt,
    REASONING_START, REASONING_END, SOLUTION_START, SOLUTION_END
)


def setup_notebook_env() -> None:
    """Setup notebook environment (Colab/Kaggle) with ecolab.

    Configures:
    - Auto-plotting for better visualization in notebooks
    - Environment detection and reporting
    """
    if not HAS_ECOLAB:
        return

    # Auto-configure plotting for notebook environments
    try:
        ecolab.auto_plot_config()
        if ecolab.is_notebook():
            print("✅ Notebook environment configured with ecolab")
    except Exception as e:
        print(f"⚠️  Warning: Could not configure notebook environment: {e}")


def get_device(preferred: str = "auto") -> str:
    """Determine the best available device.

    Supports auto-detection for:
    - Google Colab (GPU/TPU)
    - Kaggle notebooks (GPU)
    - Local CUDA, MPS (Apple Silicon), or CPU

    Args:
        preferred: Preferred device ("auto", "cuda", "mps", "cpu")

    Returns:
        Device string for PyTorch
    """
    if preferred != "auto":
        return preferred

    # Use ecolab for environment detection if available
    if HAS_ECOLAB:
        # Check if we're in Colab or Kaggle
        if ecolab.is_notebook():
            print(f"🔍 Detected notebook environment: {ecolab.get_notebook_name()}")

        # ecolab can help detect Colab-specific features
        if hasattr(ecolab, 'is_colab') and ecolab.is_colab():
            print("📓 Running in Google Colab")

        if hasattr(ecolab, 'is_kaggle') and ecolab.is_kaggle():
            print("📊 Running in Kaggle")

    # Standard device detection
    if torch.cuda.is_available():
        return "cuda"
    elif torch.backends.mps.is_available():
        return "mps"
    else:
        return "cpu"


class GemmaModel:
    """Wrapper for Gemma3-1B model with LoRA support."""

    # Base model ID on HuggingFace
    BASE_MODEL_ID = "google/gemma-3-1b-it"

    def __init__(
        self,
        checkpoint_path: Optional[str] = None,
        device: str = "auto",
        load_in_8bit: bool = False,
        load_in_4bit: bool = False,
    ):
        """Initialize the Gemma model.

        Args:
            checkpoint_path: Path to fine-tuned LoRA weights (optional)
            device: Device to use ("auto", "cuda", "mps", "cpu")
            load_in_8bit: Use 8-bit quantization (requires bitsandbytes)
            load_in_4bit: Use 4-bit quantization (requires bitsandbytes)
        """
        self.device = get_device(device)
        self.checkpoint_path = checkpoint_path
        self.load_in_8bit = load_in_8bit
        self.load_in_4bit = load_in_4bit

        self.model = None
        self.tokenizer = None
        self._loaded = False

    def load(self) -> None:
        """Load the model and tokenizer.

        Note: Gemma models are gated on HuggingFace. You must:
        1. Accept the license at https://huggingface.co/google/gemma-3-1b-it
        2. Login with: huggingface-cli login
        """
        if self._loaded:
            return

        print(f"Loading Gemma3-1B on device: {self.device}")

        # Configure quantization if requested
        quantization_config = None
        if self.load_in_4bit:
            quantization_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_compute_dtype=torch.float16,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_use_double_quant=True,
            )
        elif self.load_in_8bit:
            quantization_config = BitsAndBytesConfig(load_in_8bit=True)

        # Load tokenizer
        try:
            self.tokenizer = AutoTokenizer.from_pretrained(self.BASE_MODEL_ID)
        except OSError as e:
            if "gated repo" in str(e).lower() or "401" in str(e):
                raise RuntimeError(
                    "Gemma is a gated model. To access it:\n"
                    "1. Visit https://huggingface.co/google/gemma-3-1b-it\n"
                    "2. Accept the license agreement\n"
                    "3. Run: huggingface-cli login\n"
                    "4. Enter your HuggingFace token"
                ) from e
            raise
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load base model
        model_kwargs = {
            "torch_dtype": torch.float16 if self.device != "cpu" else torch.float32,
            "device_map": "auto" if self.device == "cuda" else None,
            "trust_remote_code": True,
        }

        if quantization_config and self.device == "cuda":
            model_kwargs["quantization_config"] = quantization_config
        elif self.device == "mps":
            # MPS doesn't support quantization, use float16
            model_kwargs["torch_dtype"] = torch.float16

        # Try using specific Gemma3 class first, fall back to Auto
        if HAS_GEMMA3_CLASS:
            self.model = Gemma3ForCausalLM.from_pretrained(
                self.BASE_MODEL_ID,
                **model_kwargs
            )
        else:
            self.model = AutoModelForCausalLM.from_pretrained(
                self.BASE_MODEL_ID,
                **model_kwargs
            )

        # Load LoRA weights if checkpoint provided
        if self.checkpoint_path and os.path.exists(self.checkpoint_path):
            print(f"Loading LoRA weights from: {self.checkpoint_path}")
            self.model = PeftModel.from_pretrained(
                self.model,
                self.checkpoint_path,
            )

        # Move to device if not using device_map="auto"
        if self.device != "cuda" and hasattr(self.model, "to"):
            self.model = self.model.to(self.device)

        self.model.eval()
        self._loaded = True
        print("Model loaded successfully!")

    def generate(
        self,
        prompt: str,
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> str:
        """Generate a response for a single prompt.

        Args:
            prompt: Input prompt (already formatted with template)
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling (False for greedy)

        Returns:
            Generated text response
        """
        if not self._loaded:
            self.load()

        inputs = self.tokenizer(prompt, return_tensors="pt")
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_k=top_k if do_sample else 1,
                top_p=top_p if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        # Decode only the generated part (exclude input)
        input_length = inputs["input_ids"].shape[1]
        generated_tokens = outputs[0][input_length:]
        response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)

        return response

    def generate_batch(
        self,
        prompts: List[str],
        max_new_tokens: int = 512,
        temperature: float = 0.7,
        top_k: int = 50,
        top_p: float = 0.95,
        do_sample: bool = True,
    ) -> List[str]:
        """Generate responses for multiple prompts.

        Args:
            prompts: List of input prompts
            max_new_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            top_k: Top-k sampling parameter
            top_p: Top-p (nucleus) sampling parameter
            do_sample: Whether to use sampling

        Returns:
            List of generated responses
        """
        if not self._loaded:
            self.load()

        # Tokenize with padding
        inputs = self.tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=512,
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_new_tokens=max_new_tokens,
                temperature=temperature if do_sample else 1.0,
                top_k=top_k if do_sample else 1,
                top_p=top_p if do_sample else 1.0,
                do_sample=do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id,
            )

        responses = []
        for i, output in enumerate(outputs):
            input_length = (inputs["attention_mask"][i] == 1).sum().item()
            generated_tokens = output[input_length:]
            response = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
            responses.append(response)

        return responses

    def solve(
        self,
        question: str,
        system_prompt_version: int = 2,
        **generation_kwargs,
    ) -> Dict[str, Any]:
        """Solve a problem and extract reasoning and answer.

        Args:
            question: The problem/question to solve
            system_prompt_version: Version of system prompt to use (0-6)
            **generation_kwargs: Additional arguments for generate()

        Returns:
            Dictionary with 'prompt', 'response', 'reasoning', 'answer'
        """
        system_prompt = get_system_prompt(system_prompt_version)
        prompt = format_prompt(question, system_prompt=system_prompt)

        response = self.generate(prompt, **generation_kwargs)

        # Extract reasoning and answer
        import re
        reasoning_match = re.search(
            rf"{REASONING_START}(.*?){REASONING_END}",
            response,
            flags=re.DOTALL
        )
        answer_match = re.search(
            rf"{SOLUTION_START}(.*?){SOLUTION_END}",
            response,
            flags=re.DOTALL
        )

        reasoning = reasoning_match.group(1).strip() if reasoning_match else ""
        answer = answer_match.group(1).strip() if answer_match else ""

        return {
            "prompt": prompt,
            "response": response,
            "reasoning": reasoning,
            "answer": answer,
        }


def load_model(
    checkpoint_path: Optional[str] = None,
    device: str = "auto",
    **kwargs
) -> GemmaModel:
    """Convenience function to load a Gemma model.

    Args:
        checkpoint_path: Path to fine-tuned LoRA weights
        device: Device to use
        **kwargs: Additional arguments for GemmaModel

    Returns:
        Loaded GemmaModel instance
    """
    model = GemmaModel(
        checkpoint_path=checkpoint_path,
        device=device,
        **kwargs
    )
    model.load()
    return model
