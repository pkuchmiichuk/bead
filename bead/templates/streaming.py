"""Streaming template filling for large combinatorial spaces."""

from __future__ import annotations

from collections.abc import Iterator

from bead.data.language_codes import LanguageCode, validate_iso639_code
from bead.resources.adapters.registry import AdapterRegistry
from bead.resources.lexical_item import LexicalItem
from bead.resources.lexicon import Lexicon
from bead.resources.template import Template
from bead.templates.combinatorics import cartesian_product
from bead.templates.filler import FilledTemplate
from bead.templates.resolver import ConstraintResolver


class StreamingFiller:
    """Fill templates with lazy evaluation.

    Generate filled templates one at a time without storing
    all combinations in memory. Use for very large combinatorial
    spaces where ExhaustiveStrategy would cause OOM.

    Parameters
    ----------
    lexicon : Lexicon
        Lexicon containing candidate items.
    adapter_registry : AdapterRegistry | None
        Adapter registry for resource-based constraints.
    max_combinations : int | None
        Maximum combinations to generate. Default: None (unlimited).

    Examples
    --------
    >>> filler = StreamingFiller(lexicon, max_combinations=1000)
    >>> for filled in filler.stream(template):
    ...     process(filled)  # process one at a time
    ...     if some_condition:
    ...         break  # can stop early
    """

    def __init__(
        self,
        lexicon: Lexicon,
        adapter_registry: AdapterRegistry | None = None,
        max_combinations: int | None = None,
    ) -> None:
        self.lexicon = lexicon
        self.adapter_registry = adapter_registry
        self.max_combinations = max_combinations

        self.resolver = ConstraintResolver()

    def stream(
        self,
        template: Template,
        language_code: LanguageCode | None = None,
    ) -> Iterator[FilledTemplate]:
        """Stream filled templates lazily.

        Generate filled templates one at a time using lazy evaluation.
        Memory-efficient for large combinatorial spaces.

        Parameters
        ----------
        template : Template
            Template to fill.
        language_code : LanguageCode | None
            Optional language filter.

        Yields
        ------
        FilledTemplate
            Filled template instances.

        Raises
        ------
        ValueError
            If any slot has no valid items.

        Examples
        --------
        >>> for i, filled in enumerate(filler.stream(template)):
        ...     if i >= 100:
        ...         break  # take first 100
        ...     print(filled.rendered_text)
        """
        # normalize language code to ISO 639-3 format if provided
        normalized_language_code = validate_iso639_code(language_code)

        # resolve slot constraints
        slot_items = self._resolve_slot_constraints(template, normalized_language_code)

        # check for empty slots
        empty_slots = [name for name, items in slot_items.items() if not items]
        if empty_slots:
            raise ValueError(f"No valid items for slots: {empty_slots}")

        # get ordered slot names and item lists
        slot_names = list(slot_items.keys())
        item_lists = [slot_items[name] for name in slot_names]

        # stream combinations
        count = 0
        for combo_tuple in cartesian_product(*item_lists):
            if self.max_combinations and count >= self.max_combinations:
                break

            # create slot_fillers dict
            slot_fillers = dict(zip(slot_names, combo_tuple, strict=True))

            # render template
            rendered = self._render_template(template, slot_fillers)

            # create FilledTemplate
            filled = FilledTemplate(
                template_id=str(template.id),
                template_name=template.name,
                slot_fillers=slot_fillers,
                rendered_text=rendered,
                strategy_name="streaming",
            )

            yield filled
            count += 1

    def _resolve_slot_constraints(
        self,
        template: Template,
        language_code: LanguageCode | None,
    ) -> dict[str, list[LexicalItem]]:
        """Resolve constraints for each slot.

        Parameters
        ----------
        template : Template
            Template with slots and constraints.
        language_code : LanguageCode | None
            Optional language filter.

        Returns
        -------
        dict[str, list[LexicalItem]]
            Mapping of slot names to valid items.
        """
        # normalize language code if provided
        normalized_lang = validate_iso639_code(language_code) if language_code else None

        slot_items: dict[str, list[LexicalItem]] = {}
        for slot_name, slot in template.slots.items():
            candidates: list[LexicalItem] = []
            for item in self.lexicon.items:
                # filter by language code if specified
                if normalized_lang:
                    # normalize item language code for comparison
                    item_lang = (
                        validate_iso639_code(item.language_code)
                        if item.language_code
                        else None
                    )
                    if item_lang != normalized_lang:
                        continue

                # check if item satisfies slot constraints
                if self.resolver.evaluate_slot_constraints(item, slot.constraints):
                    candidates.append(item)

            slot_items[slot_name] = candidates

        return slot_items

    def _render_template(
        self,
        template: Template,
        slot_fillers: dict[str, LexicalItem],
    ) -> str:
        """Render template string with slot fillers.

        Parameters
        ----------
        template : Template
            Template with template_string.
        slot_fillers : dict[str, LexicalItem]
            Items filling each slot.

        Returns
        -------
        str
            Rendered template string.
        """
        rendered = template.template_string
        for slot_name, item in slot_fillers.items():
            placeholder = f"{{{slot_name}}}"
            rendered = rendered.replace(placeholder, item.lemma)
        return rendered
