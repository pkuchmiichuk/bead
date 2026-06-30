"""Ordinal scale simulation strategy."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

from bead.simulation.strategies.base import SimulationStrategy

if TYPE_CHECKING:
    from bead.items.item import Item
    from bead.items.item_template import ItemTemplate


class OrdinalScaleStrategy(SimulationStrategy):
    """Strategy for ordinal_scale tasks (Likert scales).

    Handles discrete ordinal scales (e.g., 1-7, 1-5). Maps model outputs
    to scale positions, then samples with noise around that position.

    For ordinal scales with LM score:
        - Map score to continuous position on scale
        - Add noise
        - Round to nearest integer within bounds

    Examples
    --------
    >>> strategy = OrdinalScaleStrategy()
    >>> strategy.supported_task_type
    'ordinal_scale'
    """

    @property
    def supported_task_type(self) -> str:
        """Return 'ordinal_scale'.

        Returns
        -------
        str
            Task type identifier.
        """
        return "ordinal_scale"

    def validate_item(self, item: Item, item_template: ItemTemplate) -> None:
        """Validate item for ordinal scale.

        Checks:
        - task_type is 'ordinal_scale'
        - task_spec.scale_bounds is defined
        - scale_bounds has valid min/max

        Parameters
        ----------
        item : Item
            Item to validate.
        item_template : ItemTemplate
            Template defining task.

        Raises
        ------
        ValueError
            If validation fails.
        """
        if item_template.task_type != "ordinal_scale":
            msg = f"Expected task_type 'ordinal_scale', got '{item_template.task_type}'"
            raise ValueError(msg)

        if not item_template.task_spec.scale_bounds:
            msg = "task_spec.scale_bounds must be defined for ordinal_scale"
            raise ValueError(msg)

        bounds = item_template.task_spec.scale_bounds
        min_val, max_val = bounds.min, bounds.max
        if min_val >= max_val:
            msg = f"scale_bounds min ({min_val}) must be less than max ({max_val})"
            raise ValueError(msg)

    def simulate_response(
        self,
        item: Item,
        item_template: ItemTemplate,
        model_output_key: str,
        rng: np.random.RandomState,
    ) -> int:
        """Generate ordinal scale response.

        Parameters
        ----------
        item : Item
            Item to respond to.
        item_template : ItemTemplate
            Template defining task.
        model_output_key : str
            Key for model outputs (e.g., "lm_score").
        rng : np.random.RandomState
            Random number generator.

        Returns
        -------
        int
            Rating on ordinal scale.
        """
        scale_bounds = item_template.task_spec.scale_bounds
        if scale_bounds is None:
            msg = "task_spec.scale_bounds must be defined"
            raise ValueError(msg)

        min_val, max_val = scale_bounds.min, scale_bounds.max
        scale_range = max_val - min_val

        # extract model output (expecting single score)
        scores = self.extract_model_outputs(item, model_output_key, required_count=1)

        if scores is None:
            # fallback to uniform random across scale
            return int(rng.randint(min_val, max_val + 1))

        # map LM score to scale position; use sigmoid to map unbounded score to [0, 1]
        score = scores[0]
        sigmoid_score = 1.0 / (1.0 + np.exp(-score))

        # map [0, 1] to scale range
        continuous_rating = min_val + sigmoid_score * scale_range

        # round to nearest integer
        rating = int(np.round(continuous_rating))

        # clamp to scale bounds (in case of rounding issues)
        rating = max(min_val, min(max_val, rating))

        return rating
