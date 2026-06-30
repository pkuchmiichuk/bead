"""Item configuration."""

from __future__ import annotations

import didactic.api as dx

from bead.config.model import ModelConfig


def _default_model_config() -> ModelConfig:
    return ModelConfig()


class ItemConfig(dx.Model):
    """Configuration for item generation.

    Attributes
    ----------
    model : ModelConfig
        Model configuration.
    apply_constraints : bool
        Whether to apply model-based constraints.
    track_metadata : bool
        Whether to track item metadata.
    parallel_processing : bool
        Whether to use parallel processing.
    num_workers : int
        Number of workers for parallel processing (must be > 0).
    """

    model: dx.Embed[ModelConfig] = dx.field(default_factory=_default_model_config)
    apply_constraints: bool = True
    track_metadata: bool = True
    parallel_processing: bool = False
    num_workers: int = 4
