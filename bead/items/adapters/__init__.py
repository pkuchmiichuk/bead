"""Model adapters for judgment prediction during item construction.

Integrates HuggingFace transformers, OpenAI, Anthropic, Google, and Together
AI models. Separate from template filling adapters (Stage 2).
"""

# API utilities - explicit re-exports for type checkers
from bead.items.adapters.api_utils import (
    RateLimiter,
    rate_limit,
    retry_with_backoff,
)
from bead.items.adapters.base import ModelAdapter, TextGenerator
from bead.items.adapters.huggingface import (
    HuggingFaceLanguageModel,
    HuggingFaceMaskedLanguageModel,
    HuggingFaceNLI,
)

# Registry - explicit re-exports for type checkers
from bead.items.adapters.registry import (
    ModelAdapterRegistry,
    default_registry,
)
from bead.items.adapters.sentence_transformers import (
    HuggingFaceSentenceTransformer,
)

# API adapters (optional, may not be available if dependencies not installed)
try:
    from bead.items.adapters.openai import OpenAIAdapter
except ImportError:
    pass

try:
    from bead.items.adapters.anthropic import AnthropicAdapter
except ImportError:
    pass

try:
    from bead.items.adapters.google import GoogleAdapter
except ImportError:
    pass

try:
    from bead.items.adapters.togetherai import TogetherAIAdapter
except ImportError:
    pass

__all__ = [
    # Base
    "ModelAdapter",
    "TextGenerator",
    # HuggingFace adapters
    "HuggingFaceLanguageModel",
    "HuggingFaceMaskedLanguageModel",
    "HuggingFaceNLI",
    "HuggingFaceSentenceTransformer",
    # API utilities
    "RateLimiter",
    "rate_limit",
    "retry_with_backoff",
    # Registry
    "ModelAdapterRegistry",
    "default_registry",
    # API adapters (conditionally exported based on available dependencies)
    "OpenAIAdapter",
    "AnthropicAdapter",
    "GoogleAdapter",
    "TogetherAIAdapter",
]
