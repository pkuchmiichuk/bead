"""Template configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import didactic.api as dx


class SlotStrategyConfig(dx.Model):
    """Configuration for a single slot's filling strategy.

    Attributes
    ----------
    strategy : Literal["exhaustive", "random", "stratified", "mlm"]
        Filling strategy.
    sample_size : int | None
        Sample size for random or stratified strategies.
    stratify_by : str | None
        Feature name to stratify by (stratified strategy only).
    beam_size : int | None
        Beam size for MLM strategy.
    """

    strategy: Literal["exhaustive", "random", "stratified", "mlm"]
    sample_size: int | None = None
    stratify_by: str | None = None
    beam_size: int | None = None


class TemplateConfig(dx.Model):
    """Configuration for template filling.

    Attributes
    ----------
    filling_strategy : Literal["exhaustive", "random", "stratified", "mlm", "mixed"]
        Strategy for filling templates.
    batch_size : int
        Batch size for filling operations (must be > 0).
    max_combinations : int | None
        Maximum combinations to generate.
    random_seed : int | None
        Random seed for reproducibility.
    stream_mode : bool
        Use streaming for large templates.
    use_csp_solver : bool
        Use CSP solver for multi-slot constraints.
    mlm_model_name : str | None
        HuggingFace model name for MLM filling.
    mlm_beam_size : int
        Beam search width for MLM (> 0).
    mlm_fill_direction : Literal[...]
        Direction for filling slots.
    mlm_custom_order : tuple[int, ...] | None
        Custom slot fill order for MLM.
    mlm_top_k : int
        Top-k candidates per slot in MLM (> 0).
    mlm_device : str
        Device for MLM inference.
    mlm_cache_enabled : bool
        Enable MLM prediction caching.
    mlm_cache_dir : Path | None
        Directory for MLM prediction cache.
    slot_strategies : dict[str, SlotStrategyConfig] | None
        Per-slot strategy configuration for mixed filling.
    """

    filling_strategy: Literal["exhaustive", "random", "stratified", "mlm", "mixed"] = (
        "exhaustive"
    )
    batch_size: int = 1000
    max_combinations: int | None = None
    random_seed: int | None = None
    stream_mode: bool = False
    use_csp_solver: bool = False

    mlm_model_name: str | None = None
    mlm_beam_size: int = 5
    mlm_fill_direction: Literal[
        "left_to_right", "right_to_left", "inside_out", "outside_in", "custom"
    ] = "left_to_right"
    mlm_custom_order: tuple[int, ...] | None = None
    mlm_top_k: int = 20
    mlm_device: str = "cpu"
    mlm_cache_enabled: bool = True
    mlm_cache_dir: Path | None = None

    slot_strategies: dict[str, dx.Embed[SlotStrategyConfig]] | None = None

    @dx.validates("max_combinations")
    def _check_max_combinations(self, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError(f"max_combinations must be positive, got {value}")
        return value

    @dx.validates("batch_size")
    def _check_batch_size(self, value: int) -> int:
        if value <= 0:
            raise ValueError(f"batch_size must be positive, got {value}")
        return value


def validate_template_config(config: TemplateConfig) -> None:
    """Raise ``ValueError`` if MLM / mixed-strategy fields are inconsistent."""
    if config.filling_strategy == "mlm" and config.mlm_model_name is None:
        raise ValueError(
            "mlm_model_name must be specified when filling_strategy is 'mlm'"
        )
    if config.mlm_fill_direction == "custom" and config.mlm_custom_order is None:
        raise ValueError(
            "mlm_custom_order must be specified when mlm_fill_direction is 'custom'"
        )
    if config.filling_strategy == "mixed" and config.slot_strategies is None:
        raise ValueError(
            "slot_strategies must be specified when filling_strategy is 'mixed'"
        )
    if config.slot_strategies is not None:
        for slot_name, slot_config in config.slot_strategies.items():
            if slot_config.strategy == "mlm" and config.mlm_model_name is None:
                raise ValueError(
                    f"mlm_model_name must be specified when slot '{slot_name}' uses MLM"
                )
