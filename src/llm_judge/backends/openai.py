"""OpenAI API backend.

Supports OpenAI's chat completion API for rubric generation and scoring.
"""

import os
import time
from typing import List, Optional

from .base import LLMBackend
from ..config import GenerationConfig, LLMResponse


class OpenAIBackend(LLMBackend):
    """OpenAI API backend.

    Uses OpenAI's chat completion API for text generation.
    Supports GPT-4, GPT-4o, GPT-3.5-turbo, and other chat models.
    """

    def __init__(
        self,
        model_id: str = "gpt-4o-mini",
        api_key: Optional[str] = None,
        api_key_env: str = "OPENAI_API_KEY",
        base_url: Optional[str] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs,
    ):
        """Initialize the OpenAI backend.

        Args:
            model_id: OpenAI model ID (e.g., "gpt-4o-mini", "gpt-4o", "gpt-4")
            api_key: API key (if None, reads from environment)
            api_key_env: Environment variable for API key
            base_url: Optional custom base URL (for Azure or compatible APIs)
            max_retries: Maximum retry attempts
            retry_delay: Base delay between retries (exponential backoff)
        """
        self.model_id = model_id
        self.api_key = api_key or os.getenv(api_key_env)
        self.base_url = base_url
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._client = None

    def _get_client(self):
        """Get or create the OpenAI client.

        Returns:
            OpenAI client instance

        Raises:
            ImportError: If openai package is not installed
            ValueError: If API key is not set
        """
        if self._client is None:
            try:
                from openai import OpenAI
            except ImportError:
                raise ImportError(
                    "OpenAI package required. Install with: pip install openai"
                )

            if not self.api_key:
                raise ValueError(
                    "OpenAI API key not set. Set OPENAI_API_KEY environment variable "
                    "or pass api_key parameter."
                )

            client_kwargs = {"api_key": self.api_key}
            if self.base_url:
                client_kwargs["base_url"] = self.base_url

            self._client = OpenAI(**client_kwargs)

        return self._client

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
        config = config or GenerationConfig()
        client = self._get_client()

        last_error = None
        for attempt in range(self.max_retries):
            try:
                response = client.chat.completions.create(
                    model=self.model_id,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=config.max_new_tokens,
                    temperature=config.temperature,
                    top_p=config.top_p,
                    stop=config.stop_sequences or None,
                )

                return LLMResponse(
                    text=response.choices[0].message.content or "",
                    finish_reason=response.choices[0].finish_reason,
                    usage={
                        "prompt_tokens": response.usage.prompt_tokens,
                        "completion_tokens": response.usage.completion_tokens,
                    },
                    raw_response=response,
                )

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    time.sleep(self.retry_delay * (2**attempt))

        raise RuntimeError(
            f"OpenAI API failed after {self.max_retries} attempts: {last_error}"
        )

    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
    ) -> List[LLMResponse]:
        """Generate responses for multiple prompts.

        Note: Uses sequential calls. For true batch processing,
        consider using OpenAI's batch API.

        Args:
            prompts: List of input prompts
            config: Generation configuration

        Returns:
            List of LLMResponse objects
        """
        return [self.generate(p, config) for p in prompts]

    def is_available(self) -> bool:
        """Check if this backend is available."""
        if not self.api_key:
            return False
        try:
            self._get_client()
            return True
        except Exception:
            return False

    @property
    def backend_name(self) -> str:
        """Return the name of this backend."""
        return f"openai:{self.model_id}"
