"""Simulation configuration."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import didactic.api as dx


class NoiseModelConfig(dx.Model):
    """Configuration for the noise model in simulated judgments.

    Attributes
    ----------
    noise_type : Literal["temperature", "systematic", "random", "none"]
        Type of noise.
    temperature : float
        Temperature scaling (0.01-10.0).
    bias_strength : float
        Strength of systematic biases (0.0-1.0).
    bias_type : str | None
        Type of bias (e.g. ``"length"``, ``"frequency"``, ``"position"``).
    random_noise_stddev : float
        Standard deviation for random noise (>= 0).
    """

    noise_type: Literal["temperature", "systematic", "random", "none"] = "temperature"
    temperature: float = 1.0
    bias_strength: float = 0.0
    bias_type: str | None = None
    random_noise_stddev: float = 0.0

    __axioms__ = (
        dx.axiom(
            "temperature >= 0.01 and temperature <= 10.0",
            message="temperature must be between 0.01 and 10.0",
        ),
        dx.axiom(
            "bias_strength >= 0 and bias_strength <= 1",
            message="bias_strength must be between 0 and 1",
        ),
        dx.axiom(
            "random_noise_stddev >= 0",
            message="random_noise_stddev must be non-negative",
        ),
    )


def _default_noise_model() -> NoiseModelConfig:
    return NoiseModelConfig()


class SimulatedAnnotatorConfig(dx.Model):
    """Configuration for a simulated annotator.

    Attributes
    ----------
    strategy : Literal["lm_score", "distance", "random", "oracle", "dsl"]
        Base strategy for generating judgments.
    noise_model : NoiseModelConfig
        Noise model configuration.
    dsl_expression : str | None
        Custom DSL expression for simulation logic.
    random_state : int | None
        Random seed for reproducibility.
    model_output_key : str
        Key to extract from ``Item.model_outputs``.
    fallback_to_random : bool
        Whether to fall back to random when model outputs are missing.
    """

    strategy: Literal["lm_score", "distance", "random", "oracle", "dsl"] = "lm_score"
    noise_model: dx.Embed[NoiseModelConfig] = dx.field(
        default_factory=_default_noise_model
    )
    dsl_expression: str | None = None
    random_state: int | None = None
    model_output_key: str = "lm_score"
    fallback_to_random: bool = True


def _default_annotator_configs() -> tuple[SimulatedAnnotatorConfig, ...]:
    return (SimulatedAnnotatorConfig(),)


class SimulationRunnerConfig(dx.Model):
    """Configuration for the simulation runner.

    Attributes
    ----------
    annotator_configs : tuple[SimulatedAnnotatorConfig, ...]
        Annotator configurations.
    n_annotators : int
        Number of simulated annotators (1-100).
    inter_annotator_correlation : float | None
        Desired correlation between annotators (0.0-1.0).
    output_format : Literal["dict", "dataframe", "jsonl"]
        Output format for simulation results.
    save_path : Path | None
        Path to save simulation results.
    """

    annotator_configs: tuple[dx.Embed[SimulatedAnnotatorConfig], ...] = dx.field(
        default_factory=_default_annotator_configs
    )
    n_annotators: int = 1
    inter_annotator_correlation: float | None = None
    output_format: Literal["dict", "dataframe", "jsonl"] = "dict"
    save_path: Path | None = None

    __axioms__ = (
        dx.axiom(
            "n_annotators >= 1 and n_annotators <= 100",
            message="n_annotators must be between 1 and 100",
        ),
        dx.axiom(
            "inter_annotator_correlation == None or "
            "(inter_annotator_correlation >= 0 and inter_annotator_correlation <= 1)",
            message="inter_annotator_correlation must be between 0 and 1",
        ),
    )
