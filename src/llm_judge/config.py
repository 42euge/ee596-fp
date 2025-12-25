"""Configuration dataclasses for LLM-based rubric generation and scoring.

Provides configuration for:
- LLM backends (HuggingFace, OpenAI, Anthropic, vLLM)
- Rubric generation settings
- Response scoring settings
"""

from dataclasses import dataclass, field
from typing import Optional, Literal, Dict, Any, List, Tuple
from enum import Enum


class JudgeMode(Enum):
    """Mode for LLM-as-judge scoring."""
    RUBRIC_BASED = "rubric_based"       # Score against a provided rubric
    COMPARATIVE = "comparative"          # Compare two responses
    REFERENCE_BASED = "reference_based"  # Score against a reference answer


@dataclass
class GenerationConfig:
    """Configuration for LLM generation."""
    max_new_tokens: int = 1024
    temperature: float = 0.7
    top_p: float = 0.95
    top_k: int = 50
    stop_sequences: List[str] = field(default_factory=list)


@dataclass
class LLMResponse:
    """Standardized response from any LLM backend."""
    text: str
    finish_reason: Optional[str] = None
    usage: Optional[Dict[str, int]] = None  # {"prompt_tokens": N, "completion_tokens": M}
    raw_response: Optional[Any] = None  # Backend-specific raw response


@dataclass
class RubricGeneratorConfig:
    """Configuration for rubric generation."""
    backend: str = "huggingface"  # "huggingface", "vllm", "openai", "anthropic"
    model_id: str = "google/gemma-3-1b-it"

    # Generation settings
    max_new_tokens: int = 1024
    temperature: float = 0.3  # Lower for more deterministic rubrics

    # Caching
    cache_enabled: bool = True
    cache_dir: str = "./.rubric_cache"

    # Question type detection
    auto_detect_question_type: bool = True
    default_question_type: str = "default"

    # Rubric format
    num_criteria: int = 5
    score_range: Tuple[int, int] = (0, 10)
    include_examples: bool = True

    # Backend-specific configs
    backend_config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def for_openai(
        cls,
        model: str = "gpt-4o-mini",
        **kwargs,
    ) -> "RubricGeneratorConfig":
        """Create config for OpenAI backend."""
        return cls(
            backend="openai",
            model_id=model,
            backend_config={"api_key_env": "OPENAI_API_KEY"},
            **kwargs,
        )

    @classmethod
    def for_anthropic(
        cls,
        model: str = "claude-3-5-sonnet-20241022",
        **kwargs,
    ) -> "RubricGeneratorConfig":
        """Create config for Anthropic backend."""
        return cls(
            backend="anthropic",
            model_id=model,
            backend_config={"api_key_env": "ANTHROPIC_API_KEY"},
            **kwargs,
        )

    @classmethod
    def for_local(
        cls,
        model_id: str = "google/gemma-3-1b-it",
        device: str = "auto",
        **kwargs,
    ) -> "RubricGeneratorConfig":
        """Create config for local HuggingFace backend."""
        return cls(
            backend="huggingface",
            model_id=model_id,
            backend_config={"device": device},
            **kwargs,
        )

    @classmethod
    def for_vllm(
        cls,
        model_id: str = "google/gemma-3-1b-it",
        tensor_parallel_size: int = 1,
        **kwargs,
    ) -> "RubricGeneratorConfig":
        """Create config for vLLM backend."""
        return cls(
            backend="vllm",
            model_id=model_id,
            backend_config={"tensor_parallel_size": tensor_parallel_size},
            **kwargs,
        )


@dataclass
class ResponseScorerConfig:
    """Configuration for response scoring (LLM-as-judge)."""
    backend: str = "huggingface"
    model_id: str = "google/gemma-3-1b-it"

    # Scoring mode
    judge_mode: JudgeMode = JudgeMode.RUBRIC_BASED

    # Generation settings
    max_new_tokens: int = 512
    temperature: float = 0.1  # Very low for consistent scoring

    # Scoring format
    output_format: Literal["json", "xml", "plain"] = "json"
    include_reasoning: bool = True

    # Retry logic for API backends
    max_retries: int = 3
    retry_delay: float = 1.0

    # Backend-specific configs
    backend_config: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def for_openai(
        cls,
        model: str = "gpt-4o-mini",
        **kwargs,
    ) -> "ResponseScorerConfig":
        """Create config for OpenAI backend."""
        return cls(
            backend="openai",
            model_id=model,
            backend_config={"api_key_env": "OPENAI_API_KEY"},
            **kwargs,
        )

    @classmethod
    def for_anthropic(
        cls,
        model: str = "claude-3-5-sonnet-20241022",
        **kwargs,
    ) -> "ResponseScorerConfig":
        """Create config for Anthropic backend."""
        return cls(
            backend="anthropic",
            model_id=model,
            backend_config={"api_key_env": "ANTHROPIC_API_KEY"},
            **kwargs,
        )

    @classmethod
    def for_local(
        cls,
        model_id: str = "google/gemma-3-1b-it",
        device: str = "auto",
        **kwargs,
    ) -> "ResponseScorerConfig":
        """Create config for local HuggingFace backend."""
        return cls(
            backend="huggingface",
            model_id=model_id,
            backend_config={"device": device},
            **kwargs,
        )
