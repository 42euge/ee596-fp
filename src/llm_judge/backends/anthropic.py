"""Anthropic API backend.

Supports Anthropic's Claude models for rubric generation and scoring.
"""

import os
import time
from typing import List, Optional

from .base import LLMBackend
from ..config import GenerationConfig, LLMResponse


class AnthropicBackend(LLMBackend):
    """Anthropic API backend.

    Uses Anthropic's messages API for text generation.
    Supports Claude 3.5, Claude 3, and other Claude models.
    """

    def __init__(
        self,
        model_id: str = "claude-3-5-sonnet-20241022",
        api_key: Optional[str] = None,
        api_key_env: str = "ANTHROPIC_API_KEY",
        max_retries: int = 3,
        retry_delay: float = 1.0,
        **kwargs,
    ):
        """Initialize the Anthropic backend.

        Args:
            model_id: Anthropic model ID (e.g., "claude-3-5-sonnet-20241022")
            api_key: API key (if None, reads from environment)
            api_key_env: Environment variable for API key
            max_retries: Maximum retry attempts
            retry_delay: Base delay between retries (exponential backoff)
        """
        self.model_id = model_id
        self.api_key = api_key or os.getenv(api_key_env)
        self.max_retries = max_retries
        self.retry_delay = retry_delay

        self._client = None

    def _get_client(self):
        """Get or create the Anthropic client.

        Returns:
            Anthropic client instance

        Raises:
            ImportError: If anthropic package is not installed
            ValueError: If API key is not set
        """
        if self._client is None:
            try:
                import anthropic
            except ImportError:
                raise ImportError(
                    "Anthropic package required. Install with: pip install anthropic"
                )

            if not self.api_key:
                raise ValueError(
                    "Anthropic API key not set. Set ANTHROPIC_API_KEY environment "
                    "variable or pass api_key parameter."
                )

            self._client = anthropic.Anthropic(api_key=self.api_key)

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
                # Build request kwargs
                request_kwargs = {
                    "model": self.model_id,
                    "max_tokens": config.max_new_tokens,
                    "messages": [{"role": "user", "content": prompt}],
                }

                # Add optional parameters
                if config.temperature > 0:
                    request_kwargs["temperature"] = config.temperature
                if config.top_p < 1.0:
                    request_kwargs["top_p"] = config.top_p
                if config.stop_sequences:
                    request_kwargs["stop_sequences"] = config.stop_sequences

                response = client.messages.create(**request_kwargs)

                # Extract text from response
                text = ""
                if response.content:
                    text = response.content[0].text

                return LLMResponse(
                    text=text,
                    finish_reason=response.stop_reason,
                    usage={
                        "prompt_tokens": response.usage.input_tokens,
                        "completion_tokens": response.usage.output_tokens,
                    },
                    raw_response=response,
                )

            except Exception as e:
                last_error = e
                if attempt < self.max_retries - 1:
                    # Exponential backoff
                    time.sleep(self.retry_delay * (2**attempt))

        raise RuntimeError(
            f"Anthropic API failed after {self.max_retries} attempts: {last_error}"
        )

    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
    ) -> List[LLMResponse]:
        """Generate responses for multiple prompts.

        Note: Uses sequential calls. For true batch processing,
        consider using Anthropic's batch API when available.

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
        return f"anthropic:{self.model_id}"
