"""Backend registry and factory for LLM backends.

Provides a unified interface to create and manage different LLM backends.
"""

from typing import Dict, Type, List

from .base import LLMBackend
from ..config import GenerationConfig, LLMResponse

# Re-export commonly used types
__all__ = [
    "LLMBackend",
    "GenerationConfig",
    "LLMResponse",
    "register_backend",
    "get_backend",
    "list_backends",
]

# Backend registry
_BACKEND_REGISTRY: Dict[str, Type[LLMBackend]] = {}


def register_backend(name: str):
    """Decorator to register a backend class.

    Usage:
        @register_backend("my_backend")
        class MyBackend(LLMBackend):
            ...
    """

    def decorator(cls: Type[LLMBackend]):
        _BACKEND_REGISTRY[name] = cls
        return cls

    return decorator


def get_backend(
    name: str,
    **kwargs,
) -> LLMBackend:
    """Factory function to create a backend by name.

    Args:
        name: Backend name ("huggingface", "vllm", "openai", "anthropic")
        **kwargs: Backend-specific configuration

    Returns:
        Initialized LLMBackend instance

    Raises:
        ValueError: If backend name is not registered
    """
    if name not in _BACKEND_REGISTRY:
        available = list(_BACKEND_REGISTRY.keys())
        raise ValueError(f"Unknown backend: {name}. Available: {available}")

    return _BACKEND_REGISTRY[name](**kwargs)


def list_backends() -> List[str]:
    """List all registered backend names."""
    return list(_BACKEND_REGISTRY.keys())


# Auto-register built-in backends
# HuggingFace backend (always available)
from .huggingface import HuggingFaceBackend

register_backend("huggingface")(HuggingFaceBackend)

# OpenAI backend (available if openai package is installed)
try:
    from .openai import OpenAIBackend

    register_backend("openai")(OpenAIBackend)
except ImportError:
    pass

# Anthropic backend (available if anthropic package is installed)
try:
    from .anthropic import AnthropicBackend

    register_backend("anthropic")(AnthropicBackend)
except ImportError:
    pass

# vLLM backend (available if vllm package is installed)
try:
    from .vllm import VLLMBackend

    register_backend("vllm")(VLLMBackend)
except ImportError:
    pass
