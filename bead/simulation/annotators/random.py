"""Random baseline annotator."""

from __future__ import annotations

from typing import TYPE_CHECKING

from bead.simulation.annotators.base import SimulatedAnnotator

if TYPE_CHECKING:
    from bead.config.simulation import SimulatedAnnotatorConfig
    from bead.items.item import Item
    from bead.items.item_template import ItemTemplate


class RandomAnnotator(SimulatedAnnotator):
    """Pure random baseline annotator.

    Generates random responses that respect task spec constraints
    (options, scale ranges, etc.) but are otherwise uninformed.

    Useful for establishing baseline performance.

    Parameters
    ----------
    config
        Configuration for annotator.

    Examples
    --------
    >>> from bead.config.simulation import SimulatedAnnotatorConfig
    >>> config = SimulatedAnnotatorConfig(strategy="random", random_state=42)
    >>> annotator = RandomAnnotator(config)
    >>> # judgment = annotator.annotate(item, template)
    """

    def __init__(self, config: SimulatedAnnotatorConfig) -> None:
        super().__init__(config)

        # no strategies or noise models needed for random

    def annotate(
        self, item: Item, item_template: ItemTemplate
    ) -> str | int | float | bool | list[str]:
        """Generate random annotation.

        Parameters
        ----------
        item : Item
            Item to annotate (ignored).
        item_template : ItemTemplate
            Template defining task constraints.

        Returns
        -------
        str | int | float | bool | list[str]
            Random annotation (format depends on task type).

        Raises
        ------
        ValueError
            If task type is not supported.
        """
        task_type = item_template.task_type

        if task_type == "forced_choice":
            return self._random_forced_choice(item_template)
        elif task_type == "binary":
            return self._random_binary()
        elif task_type == "ordinal_scale":
            return self._random_ordinal(item_template)
        elif task_type == "categorical":
            return self._random_categorical(item_template)
        elif task_type == "magnitude":
            return self._random_magnitude()
        elif task_type == "multi_select":
            return self._random_multi_select(item_template)
        elif task_type == "free_text":
            return self._random_free_text()
        elif task_type == "cloze":
            return self._random_cloze(item)
        else:
            raise ValueError(f"Unsupported task type: {task_type}")

    def _random_forced_choice(self, template: ItemTemplate) -> str:
        """Generate random forced choice response."""
        options = template.task_spec.options or []
        if not options:
            raise ValueError("forced_choice requires options")
        return str(self.rng.choice(options))

    def _random_binary(self) -> bool:
        """Generate random binary response."""
        return bool(self.rng.choice([True, False]))

    def _random_ordinal(self, template: ItemTemplate) -> int:
        """Generate random ordinal scale response."""
        # get scale bounds from task_spec
        scale_bounds = template.task_spec.scale_bounds
        if scale_bounds is not None:
            min_val, max_val = scale_bounds.min, scale_bounds.max
        else:
            min_val, max_val = 1, 7
        return int(self.rng.randint(min_val, max_val + 1))

    def _random_categorical(self, template: ItemTemplate) -> str:
        """Generate random categorical response."""
        options = template.task_spec.options or []
        if not options:
            raise ValueError("categorical requires options")
        return str(self.rng.choice(options))

    def _random_magnitude(self) -> float:
        """Generate random magnitude response."""
        # log-normal distribution for positive magnitudes
        return float(self.rng.lognormal(mean=0, sigma=1))

    def _random_multi_select(self, template: ItemTemplate) -> list[str]:
        """Generate random multi-select response."""
        options = template.task_spec.options or []
        if not options:
            raise ValueError("multi_select requires options")

        # randomly select subset of options
        selected = []
        for option in options:
            if self.rng.random() < 0.5:
                selected.append(option)
        return selected

    def _random_free_text(self) -> str:
        """Generate random free text response."""
        # simple random responses
        responses = [
            "No response",
            "Unclear",
            "Cannot determine",
            "Not applicable",
            "Unknown",
        ]
        return str(self.rng.choice(responses))

    def _random_cloze(self, item: Item) -> dict[str, str]:
        """Generate random cloze response."""
        response = {}

        # common word bank for random selection
        word_bank = [
            "the",
            "a",
            "is",
            "was",
            "has",
            "can",
            "will",
            "thing",
            "person",
            "place",
            "time",
            "way",
            "good",
            "new",
            "old",
            "big",
            "small",
            "very",
            "well",
            "just",
            "now",
            "here",
            "in",
            "on",
            "at",
            "to",
            "for",
        ]

        for slot in item.unfilled_slots:
            # randomly select a word from the bank
            response[slot.slot_name] = str(self.rng.choice(word_bank))

        return response
