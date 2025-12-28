"""Abstract base class for LLM backends.

Defines the interface that all LLM backends must implement.
Uses strategy pattern for backend-agnostic rubric generation and scoring.
"""

from abc import ABC, abstractmethod
from typing import List, Optional

from ..config import GenerationConfig, LLMResponse


class LLMBackend(ABC):
    """Abstract base class for LLM backends.

    Strategy pattern: Each backend implements the same interface,
    allowing the RubricGenerator and ResponseScorer to be backend-agnostic.
    """

    @abstractmethod
    def generate(
        self,
        prompt: str,
        config: Optional[GenerationConfig] = None,
    ) -> LLMResponse:
        """Generate a single response.

        Args:
            prompt: The input prompt
            config: Generation configuration

        Returns:
            LLMResponse with generated text
        """
        pass

    @abstractmethod
    def generate_batch(
        self,
        prompts: List[str],
        config: Optional[GenerationConfig] = None,
    ) -> List[LLMResponse]:
        """Generate responses for multiple prompts.

        Args:
            prompts: List of input prompts
            config: Generation configuration

        Returns:
            List of LLMResponse objects
        """
        pass

    @abstractmethod
    def is_available(self) -> bool:
        """Check if this backend is available (model loaded, API key set, etc.)."""
        pass

    @property
    @abstractmethod
    def backend_name(self) -> str:
        """Return the name of this backend for logging/identification."""
        pass

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(name={self.backend_name})"
