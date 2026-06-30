"""Model configuration."""

from __future__ import annotations

from typing import Literal

import didactic.api as dx


class ModelConfig(dx.Model):
    """Configuration for language models.

    Attributes
    ----------
    provider : Literal["huggingface", "openai", "anthropic"]
        Model provider.
    model_name : str
        Model identifier.
    batch_size : int
        Inference batch size (must be > 0).
    device : Literal["cpu", "cuda", "mps"]
        Device to use for computation.
    max_length : int
        Maximum sequence length (must be > 0).
    temperature : float
        Sampling temperature (must be >= 0).
    cache_outputs : bool
        Whether to cache model outputs.
    """

    provider: Literal["huggingface", "openai", "anthropic"] = "huggingface"
    model_name: str = "gpt2"
    batch_size: int = 8
    device: Literal["cpu", "cuda", "mps"] = "cpu"
    max_length: int = 512
    temperature: float = 1.0
    cache_outputs: bool = True

    __axioms__ = (
        dx.axiom("batch_size > 0", message="batch_size must be positive"),
        dx.axiom("max_length > 0", message="max_length must be positive"),
        dx.axiom("temperature >= 0", message="temperature must be non-negative"),
    )
