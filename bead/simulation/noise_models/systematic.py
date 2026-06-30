"""Systematic bias noise model."""

from __future__ import annotations

import numpy as np

from bead.simulation.noise_models.base import NoiseModel


class SystematicNoiseModel(NoiseModel):
    """Systematic bias noise model.

    Adds consistent biases to responses:
    - length: Prefer shorter/longer options
    - frequency: Prefer common/rare words
    - position: Prefer first/last option
    - endpoint: Prefer endpoints on ordinal scales
    - midpoint: Prefer midpoint on ordinal scales

    Parameters
    ----------
    bias_type
        Type of bias ("length", "frequency", "position", "endpoint", "midpoint").
        Default: "position".
    bias_strength
        Strength of bias (0.0-1.0). Default: 0.0.

    Examples
    --------
    >>> noise_model = SystematicNoiseModel(bias_type="position", bias_strength=0.3)
    >>> # Adds 30% bias toward first option in forced choice
    """

    def __init__(self, bias_type: str = "position", bias_strength: float = 0.0) -> None:
        self.bias_type = bias_type
        self.bias_strength = bias_strength

    def apply(
        self,
        value: str | int | float | bool | list[str],
        context: dict[str, str | int | float | bool | list[str]],
        rng: np.random.RandomState,
    ) -> str | int | float | bool | list[str]:
        """Apply systematic bias.

        Parameters
        ----------
        value
            Original value.
        context : dict
            Context with item, template, strategy.
        rng : np.random.RandomState
            Random number generator.

        Returns
        -------
        str | int | float | bool | list[str]
            Value with bias applied.
        """
        if self.bias_strength == 0.0:
            return value

        strategy = context.get("strategy")
        template = context.get("template")

        if not strategy or not template:
            return value

        task_type = strategy.supported_task_type

        # position bias for choice tasks
        is_choice_task = task_type in ["forced_choice", "categorical"]
        if is_choice_task and self.bias_type == "position":
            return self._apply_position_bias(value, template, rng)

        # endpoint/midpoint bias for ordinal scales
        elif task_type == "ordinal_scale":
            if self.bias_type == "endpoint":
                return self._apply_endpoint_bias(value, template, rng)
            elif self.bias_type == "midpoint":
                return self._apply_midpoint_bias(value, template, rng)

        # no bias for other combinations
        return value

    def _apply_position_bias(
        self, value: str, template: str, rng: np.random.RandomState
    ) -> str:
        """Apply position bias to choice tasks."""
        options = template.task_spec.options
        if not options or len(options) < 2:
            return value

        # bias toward first option
        if rng.random() < self.bias_strength:
            return options[0]

        return value

    def _apply_endpoint_bias(
        self, value: int, template: str, rng: np.random.RandomState
    ) -> int:
        """Apply endpoint bias to ordinal scales."""
        scale_bounds = template.task_spec.scale_bounds
        if scale_bounds is not None:
            min_val, max_val = scale_bounds.min, scale_bounds.max
        else:
            min_val, max_val = 1, 7

        # bias toward endpoints (min or max)
        if rng.random() < self.bias_strength:
            return min_val if rng.random() < 0.5 else max_val

        return value

    def _apply_midpoint_bias(
        self, value: int, template: str, rng: np.random.RandomState
    ) -> int:
        """Apply midpoint bias to ordinal scales."""
        scale_bounds = template.task_spec.scale_bounds
        if scale_bounds is not None:
            min_val, max_val = scale_bounds.min, scale_bounds.max
        else:
            min_val, max_val = 1, 7

        midpoint = (min_val + max_val) // 2

        # bias toward midpoint
        if rng.random() < self.bias_strength:
            return midpoint

        return value
