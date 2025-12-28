"""vLLM backend for efficient local serving.

Provides high-throughput inference using vLLM's optimized serving.
"""

from typing import List, Optional

from .base import LLMBackend
from ..config import GenerationConfig, LLMResponse


class VLLMBackend(LLMBackend):
    """vLLM backend for efficient local inference.

    Uses vLLM's optimized inference engine for high-throughput
    generation on GPU. Supports tensor parallelism for multi-GPU setups.
    """

    def __init__(
        self,
        model_id: str = "google/gemma-3-1b-it",
        tensor_parallel_size: int = 1,
        gpu_memory_utilization: float = 0.9,
        max_model_len: Optional[int] = None,
        trust_remote_code: bool = True,
        **kwargs,
    ):
        """Initialize the vLLM backend.

        Args:
            model_id: HuggingFace model ID or local path
            tensor_parallel_size: Number of GPUs for tensor parallelism
            gpu_memory_utilization: Fraction of GPU memory to use
            max_model_len: Maximum model context length
            trust_remote_code: Whether to trust remote code in model
        """
        self.model_id = model_id
        self.tensor_parallel_size = tensor_parallel_size
        self.gpu_memory_utilization = gpu_memory_utilization
        self.max_model_len = max_model_len
        self.trust_remote_code = trust_remote_code

        self._llm = None
        self._sampling_params_class = None

    def _load(self):
        """Load the vLLM model.

        Raises:
            ImportError: If vllm package is not installed
        """
        if self._llm is not None:
            return

        try:
            from vllm import LLM, SamplingParams
        except ImportError:
            raise ImportError(
                "vLLM required. Install with: pip install vllm\n"
                "Note: vLLM requires CUDA-capable GPU."
            )

        print(f"Loading {self.model_id} with vLLM...")

        llm_kwargs = {
            "model": self.model_id,
            "tensor_parallel_size": self.tensor_parallel_size,
            "gpu_memory_utilization": self.gpu_memory_utilization,
            "trust_remote_code": self.trust_remote_code,
        }

        if self.max_model_len:
            llm_kwargs["max_model_len"] = self.max_model_len

        self._llm = LLM(**llm_kwargs)
        self._sampling_params_class = SamplingParams

        print("vLLM model loaded successfully!")

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
        responses = self.generate_batch([prompt], config)
        return responses[0]

    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
    ) -> List[LLMResponse]:
        """Generate responses for multiple prompts.

        This is the efficient path for vLLM - batched generation
        takes advantage of continuous batching for high throughput.

        Args:
            prompts: List of input prompts
            config: Generation configuration

        Returns:
            List of LLMResponse objects
        """
        self._load()
        config = config or GenerationConfig()

        # Build sampling parameters
        sampling_kwargs = {
            "max_tokens": config.max_new_tokens,
            "temperature": config.temperature if config.temperature > 0 else 0.0,
            "top_p": config.top_p,
            "top_k": config.top_k if config.top_k > 0 else -1,
        }

        if config.stop_sequences:
            sampling_kwargs["stop"] = config.stop_sequences

        sampling_params = self._sampling_params_class(**sampling_kwargs)

        # Generate
        outputs = self._llm.generate(prompts, sampling_params)

        # Convert to LLMResponse objects
        responses = []
        for output in outputs:
            # Get the first (and usually only) output
            generated_output = output.outputs[0]
            text = generated_output.text

            responses.append(
                LLMResponse(
                    text=text,
                    finish_reason=generated_output.finish_reason,
                    usage={
                        "prompt_tokens": len(output.prompt_token_ids),
                        "completion_tokens": len(generated_output.token_ids),
                    },
                    raw_response=output,
                )
            )

        return responses

    def is_available(self) -> bool:
        """Check if this backend is available."""
        try:
            import vllm

            return True
        except ImportError:
            return False

    @property
    def backend_name(self) -> str:
        """Return the name of this backend."""
        return f"vllm:{self.model_id}"
