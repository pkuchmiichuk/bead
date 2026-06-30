"""Oracle (perfect performance) annotator."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bead.simulation.annotators.base import SimulatedAnnotator
from bead.simulation.annotators.random import RandomAnnotator

if TYPE_CHECKING:
    from bead.config.simulation import SimulatedAnnotatorConfig
    from bead.items.item import Item
    from bead.items.item_template import ItemTemplate


class OracleAnnotator(SimulatedAnnotator):
    """Perfect performance annotator using ground truth.

    Returns ground truth labels from item.item_metadata['ground_truth'].
    Falls back to random when ground truth is not available.

    Useful for establishing upper bound on performance.

    Parameters
    ----------
    config
        Configuration for annotator.

    Examples
    --------
    >>> from bead.config.simulation import SimulatedAnnotatorConfig
    >>> config = SimulatedAnnotatorConfig(strategy="oracle", random_state=42)
    >>> annotator = OracleAnnotator(config)
    >>> # judgment = annotator.annotate(item, template)
    """

    def __init__(self, config: SimulatedAnnotatorConfig) -> None:
        super().__init__(config)

        # create random annotator for fallback
        self.random_annotator = RandomAnnotator(config)

    def annotate(
        self, item: Item, item_template: ItemTemplate
    ) -> str | int | float | bool | list[str]:
        """Generate oracle annotation using ground truth.

        Parameters
        ----------
        item : Item
            Item to annotate.
        item_template : ItemTemplate
            Template defining task.

        Returns
        -------
        str | int | float | bool | list[str]
            Ground truth annotation or random fallback.
        """
        # try to get ground truth from item metadata
        if hasattr(item, "item_metadata") and item.item_metadata:
            ground_truth = item.item_metadata.get("ground_truth")

            if ground_truth is not None:
                # validate and return ground truth
                return self._validate_ground_truth(ground_truth, item_template)

        # fallback to random if no ground truth
        return self.random_annotator.annotate(item, item_template)

    def _validate_ground_truth(
        self, ground_truth: str | int | float | bool | list[str], template: ItemTemplate
    ) -> str | int | float | bool | list[str]:
        """Validate ground truth against task spec.

        Parameters
        ----------
        ground_truth
            Ground truth value.
        template : ItemTemplate
            Template defining task constraints.

        Returns
        -------
        str | int | float | bool | list[str]
            Validated ground truth.

        Raises
        ------
        ValueError
            If ground truth is invalid for task type.
        """
        task_type = template.task_type

        if task_type == "forced_choice":
            if not isinstance(ground_truth, str):
                msg = (
                    f"forced_choice ground truth must be str, got {type(ground_truth)}"
                )
                raise ValueError(msg)
            options = template.task_spec.options or []
            if ground_truth not in options:
                msg = f"Ground truth '{ground_truth}' not in options {options}"
                raise ValueError(msg)
            return ground_truth

        elif task_type == "binary":
            if not isinstance(ground_truth, bool):
                msg = f"binary ground truth must be bool, got {type(ground_truth)}"
                raise ValueError(msg)
            return ground_truth

        elif task_type == "ordinal_scale":
            if not isinstance(ground_truth, int):
                msg = (
                    f"ordinal_scale ground truth must be int, got {type(ground_truth)}"
                )
                raise ValueError(msg)
            scale_bounds = template.task_spec.scale_bounds
            if scale_bounds is not None:
                min_val, max_val = scale_bounds.min, scale_bounds.max
            else:
                min_val, max_val = 1, 7
            if not (min_val <= ground_truth <= max_val):
                msg = f"Ground truth {ground_truth} not in range [{min_val}, {max_val}]"
                raise ValueError(msg)
            return ground_truth

        elif task_type == "categorical":
            if not isinstance(ground_truth, str):
                msg = f"categorical ground truth must be str, got {type(ground_truth)}"
                raise ValueError(msg)
            options = template.task_spec.options or []
            if ground_truth not in options:
                msg = f"Ground truth '{ground_truth}' not in options {options}"
                raise ValueError(msg)
            return ground_truth

        elif task_type == "magnitude":
            if not isinstance(ground_truth, int | float):
                msg = (
                    f"magnitude ground truth must be numeric, got {type(ground_truth)}"
                )
                raise ValueError(msg)
            return float(ground_truth)

        elif task_type == "multi_select":
            if not isinstance(ground_truth, list | tuple):
                msg = (
                    f"multi_select ground truth must be list or tuple, "
                    f"got {type(ground_truth)}"
                )
                raise ValueError(msg)
            options = template.task_spec.options or []
            for item_val in ground_truth:
                if item_val not in options:
                    msg = f"Ground truth item '{item_val}' not in options {options}"
                    raise ValueError(msg)
            return ground_truth

        elif task_type == "free_text":
            if not isinstance(ground_truth, str):
                msg = f"free_text ground truth must be str, got {type(ground_truth)}"
                raise ValueError(msg)
            return ground_truth

        elif task_type == "cloze":
            if not isinstance(ground_truth, dict):
                msg = f"cloze ground truth must be dict, got {type(ground_truth)}"
                raise ValueError(msg)
            # validate all required slots are present
            for slot in template.unfilled_slots:
                if slot.slot_name not in ground_truth:
                    msg = (
                        f"Ground truth missing slot '{slot.slot_name}' "
                        f"(expected slots: {[s.slot_name for s in template.unfilled_slots]})"  # noqa: E501
                    )
                    raise ValueError(msg)
            # return dict of slot_name -> value
            return {k: str(v) for k, v in ground_truth.items()}

        else:
            raise ValueError(f"Unsupported task type: {task_type}")
