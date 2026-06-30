"""Utilities for creating cloze experimental items.

This module provides language-agnostic utilities for creating cloze
items where participants fill in missing words/phrases in partially-filled
templates.

**SPECIAL**: This is the ONLY task type that uses the Item.unfilled_slots field.

Cloze items are unique in that:
- They use partially-filled templates with specific slots left blank
- UI widgets are inferred from slot constraints at deployment time:
  - Extensional constraint (finite set) → dropdown
  - Intensional constraint (rules) → text input with validation
  - No constraint → free text input
- Multiple slots can be unfilled in a single item

Integration Points
------------------
- Active Learning: bead/active_learning/models/cloze.py
- Simulation: bead/simulation/strategies/cloze.py
- Deployment: bead/deployment/jspsych/ (dynamic widget generation)
- Resources: bead/resources/template.py (Template and Slot models)
"""

from __future__ import annotations

import random
import re
from collections import defaultdict
from collections.abc import Callable
from itertools import combinations
from typing import Any
from uuid import UUID, uuid4

from bead.items.item import Item, MetadataValue, UnfilledSlot


def create_cloze_item(
    template: Any,
    unfilled_slot_names: list[str],
    filled_slots: dict[str, str] | None = None,
    instructions: str | None = None,
    *,
    item_template_id: UUID | None = None,
    metadata: dict[str, MetadataValue] | None = None,
) -> Item:
    """Create a cloze item from a template with specific slots unfilled.

    Parameters
    ----------
    template : Template
        Source template with slots.
    unfilled_slot_names : list[str]
        Names of slots to leave unfilled (must exist in template.slots).
    filled_slots : dict[str, str] | None
        Pre-filled slots (keys must be valid slot names, disjoint from unfilled).
    instructions : str | None
        Optional instructions for filling (e.g., "Fill in the verb").
    item_template_id : UUID | None
        Template ID for the item. If None, generates new UUID.
    metadata : dict[str, MetadataValue] | None
        Additional metadata for item_metadata field.

    Returns
    -------
    Item
        Cloze item with unfilled_slots populated.

    Raises
    ------
    ValueError
        If unfilled_slot_names not in template, if filled_slots not in template,
        if unfilled and filled overlap, if no unfilled slots, or if validation fails.

    Examples
    --------
    >>> from bead.resources.template import Template, Slot
    >>> template = Template(
    ...     name="simple",
    ...     template_string="{det} {noun} {verb}.",
    ...     slots={
    ...         "det": Slot(name="det"),
    ...         "noun": Slot(name="noun"),
    ...         "verb": Slot(name="verb")
    ...     }
    ... )
    >>> item = create_cloze_item(
    ...     template,
    ...     unfilled_slot_names=["verb"],
    ...     filled_slots={"det": "The", "noun": "cat"}
    ... )
    >>> item.rendered_elements["text"]
    'The cat ___.'
    >>> len(item.unfilled_slots)
    1
    >>> item.unfilled_slots[0].slot_name
    'verb'
    >>> item.unfilled_slots[0].position
    2
    """
    if filled_slots is None:
        filled_slots = {}

    # Validate parameters
    _validate_cloze_parameters(template, unfilled_slot_names, filled_slots)

    # Render template with filled values and "___" for unfilled slots
    rendered_text = _render_template_for_cloze(
        template.template_string, filled_slots, unfilled_slot_names
    )

    # Calculate positions for unfilled slots
    positions = _calculate_positions(
        template.template_string, unfilled_slot_names, filled_slots
    )

    # Extract constraint IDs for each unfilled slot
    unfilled_slots_list: list[UnfilledSlot] = []
    for slot_name in unfilled_slot_names:
        position = positions[slot_name]
        constraint_ids = _extract_constraint_ids(template, slot_name)
        unfilled_slots_list.append(
            UnfilledSlot(
                slot_name=slot_name, position=position, constraint_ids=constraint_ids
            )
        )

    # Build rendered_elements
    rendered_elements: dict[str, str] = {"text": rendered_text}
    if instructions:
        rendered_elements["instructions"] = instructions

    # Build item_metadata
    # Convert filled_slots to MetadataValue format
    filled_slots_metadata: dict[str, MetadataValue] = dict(filled_slots)
    item_metadata: dict[str, MetadataValue] = {
        "template_id": str(template.id),
        "filled_slots": filled_slots_metadata,
        "n_unfilled_slots": len(unfilled_slot_names),
    }
    if metadata:
        item_metadata.update(metadata)

    if item_template_id is None:
        item_template_id = uuid4()

    return Item(
        item_template_id=item_template_id,
        rendered_elements=rendered_elements,
        unfilled_slots=tuple(unfilled_slots_list),
        item_metadata=item_metadata,
    )


def create_cloze_items_from_template(
    template: Any,
    n_unfilled: int = 1,
    strategy: str = "all_combinations",
    unfilled_combinations: list[list[str]] | None = None,
    instructions: str | None = None,
    *,
    item_template_id: UUID | None = None,
    metadata_fn: Callable[[list[str]], dict[str, MetadataValue]] | None = None,
) -> list[Item]:
    """Create multiple cloze items from a template, varying unfilled slots.

    Parameters
    ----------
    template : Template
        Source template.
    n_unfilled : int
        Number of slots to leave unfilled per item (default: 1).
    strategy : str
        How to choose unfilled slots:
        - 'random': Randomly sample combinations
        - 'all_combinations': Generate all C(n_slots, n_unfilled) combinations
        - 'specified': Use provided list
    unfilled_combinations : list[list[str]] | None
        For strategy='specified', list of slot name combinations to unfill.
    instructions : str | None
        Instructions for all items.
    item_template_id : UUID | None
        Template ID for all items.
    metadata_fn : Callable[[list[str]], dict[str, MetadataValue]] | None
        Generate metadata from unfilled slot names.

    Returns
    -------
    list[Item]
        Cloze items with varying unfilled slots.

    Raises
    ------
    ValueError
        If n_unfilled invalid, if strategy='specified' without unfilled_combinations,
        or if any combination contains invalid slots.

    Examples
    --------
    >>> # Generate all single-slot cloze items
    >>> items = create_cloze_items_from_template(
    ...     template, n_unfilled=1, strategy='all_combinations'
    ... )
    >>> len(items)  # One for each slot
    3
    """
    slot_names = list(template.slots.keys())

    # Validate n_unfilled
    if n_unfilled < 1:
        raise ValueError(
            f"n_unfilled must be at least 1, got {n_unfilled}. "
            f"Provide a positive number of slots to leave unfilled."
        )

    if n_unfilled >= len(slot_names):
        raise ValueError(
            f"n_unfilled ({n_unfilled}) must be less than total slots "
            f"({len(slot_names)}). Cannot unfill all slots in a cloze item."
        )

    # Generate combinations based on strategy
    if strategy == "all_combinations":
        combos = list(combinations(slot_names, n_unfilled))
    elif strategy == "specified":
        if unfilled_combinations is None:
            raise ValueError(
                "strategy='specified' requires unfilled_combinations parameter. "
                "Provide a list of slot name combinations to unfill."
            )
        combos = [tuple(c) for c in unfilled_combinations]
    elif strategy == "random":
        # Generate one random combination (can be extended to generate N random ones)
        combos = [tuple(random.sample(slot_names, n_unfilled))]
    else:
        raise ValueError(
            f"Invalid strategy '{strategy}'. "
            f"Must be one of ['random', 'all_combinations', 'specified']."
        )

    # Validate all combinations
    for combo in combos:
        if len(combo) != n_unfilled:
            raise ValueError(
                f"Each combination must have exactly {n_unfilled} slots, "
                f"but got {len(combo)}: {combo}"
            )
        for slot_name in combo:
            if slot_name not in template.slots:
                raise ValueError(
                    f"Slot '{slot_name}' in combination not found in template. "
                    f"Available slots: {list(template.slots.keys())}"
                )

    # Create items
    items: list[Item] = []
    for combo in combos:
        unfilled_list = list(combo)

        # Generate metadata if function provided
        item_metadata = metadata_fn(unfilled_list) if metadata_fn else None

        item = create_cloze_item(
            template=template,
            unfilled_slot_names=unfilled_list,
            filled_slots=None,  # Don't pre-fill any slots
            instructions=instructions,
            item_template_id=item_template_id,
            metadata=item_metadata,
        )
        items.append(item)

    return items


def create_simple_cloze_item(
    text: str,
    blank_positions: list[int],
    blank_labels: list[str] | None = None,
    instructions: str | None = None,
    *,
    item_template_id: UUID | None = None,
    metadata: dict[str, MetadataValue] | None = None,
) -> Item:
    """Create a cloze item from plain text (no template).

    Replaces words at specified positions with blanks. This is a simplified
    helper for creating cloze items without the template infrastructure.

    Parameters
    ----------
    text : str
        Full text with no blanks.
    blank_positions : list[int]
        Word positions to blank (0-indexed).
    blank_labels : list[str] | None
        Optional labels for blanks (for slot_name field). If None, uses
        generic labels like "blank_0", "blank_1".
    instructions : str | None
        Optional instructions.
    item_template_id : UUID | None
        Template ID for the item.
    metadata : dict[str, MetadataValue] | None
        Additional metadata.

    Returns
    -------
    Item
        Cloze item with text-based blanks.

    Raises
    ------
    ValueError
        If blank_positions out of range or if blank_labels length mismatch.

    Examples
    --------
    >>> item = create_simple_cloze_item(
    ...     text="The quick brown fox",
    ...     blank_positions=[1],  # "quick"
    ...     blank_labels=["adjective"]
    ... )
    >>> item.rendered_elements["text"]
    'The ___ brown fox'
    >>> item.unfilled_slots[0].slot_name
    'adjective'
    >>> item.unfilled_slots[0].position
    1
    """
    if not text or not text.strip():
        raise ValueError("text cannot be empty")

    if not blank_positions:
        raise ValueError(
            "blank_positions cannot be empty. "
            "Provide at least one position to blank out."
        )

    # Tokenize text by whitespace
    tokens = text.split()

    # Validate positions
    for pos in blank_positions:
        if pos < 0 or pos >= len(tokens):
            raise ValueError(
                f"blank_position {pos} is out of range. "
                f"Text has {len(tokens)} tokens (valid range: 0-{len(tokens) - 1})"
            )

    # Validate labels if provided
    if blank_labels is not None:
        if len(blank_labels) != len(blank_positions):
            raise ValueError(
                f"blank_labels length ({len(blank_labels)}) must match "
                f"blank_positions length ({len(blank_positions)})"
            )
    else:
        # Generate default labels
        blank_labels = [f"blank_{i}" for i in range(len(blank_positions))]

    # Create unfilled slots
    unfilled_slots_list: list[UnfilledSlot] = []
    for pos, label in zip(blank_positions, blank_labels, strict=True):
        unfilled_slots_list.append(
            UnfilledSlot(slot_name=label, position=pos, constraint_ids=())
        )

    # Replace tokens at blank positions with "___"
    blanked_tokens = tokens.copy()
    for pos in blank_positions:
        blanked_tokens[pos] = "___"
    rendered_text = " ".join(blanked_tokens)

    # Build rendered_elements
    rendered_elements: dict[str, str] = {"text": rendered_text}
    if instructions:
        rendered_elements["instructions"] = instructions

    # Build item_metadata
    item_metadata: dict[str, MetadataValue] = {
        "n_unfilled_slots": len(blank_positions),
        "original_text": text,
    }
    if metadata:
        item_metadata.update(metadata)

    if item_template_id is None:
        item_template_id = uuid4()

    return Item(
        item_template_id=item_template_id,
        rendered_elements=rendered_elements,
        unfilled_slots=tuple(unfilled_slots_list),
        item_metadata=item_metadata,
    )


def create_cloze_items_from_groups(
    items: list[Item],
    group_by: Callable[[Item], Any],
    n_slots_to_unfill: int = 1,
    *,
    extract_text: Callable[[Item], str] | None = None,
    include_group_metadata: bool = True,
    item_template_id: UUID | None = None,
) -> list[Item]:
    """Create cloze items from grouped source items.

    Groups items and creates cloze items from them. If source items have
    template metadata, uses template-based cloze. Otherwise, falls back to
    simple text-based cloze.

    Parameters
    ----------
    items : list[Item]
        Source items to group.
    group_by : Callable[[Item], Any]
        Grouping function.
    n_slots_to_unfill : int
        Number of slots/words to unfill.
    extract_text : Callable[[Item], str] | None
        Text extraction function. If None, tries common keys.
    include_group_metadata : bool
        Whether to include group_key in metadata.
    item_template_id : UUID | None
        Template ID for created items.

    Returns
    -------
    list[Item]
        Cloze items from grouped source items.

    Examples
    --------
    >>> cloze_items = create_cloze_items_from_groups(
    ...     items=source_items,
    ...     group_by=lambda i: i.item_metadata.get("category"),
    ...     n_slots_to_unfill=1
    ... )  # doctest: +SKIP
    """
    # Group items
    groups: dict[Any, list[Item]] = defaultdict(list)
    for item in items:
        group_key = group_by(item)
        groups[group_key].append(item)

    cloze_items: list[Item] = []

    for group_key, group_items in groups.items():
        for item in group_items:
            # Extract text
            if extract_text:
                text: str = extract_text(item)
            else:
                text = _extract_text_from_item(item)

            # Build metadata
            item_metadata: dict[str, MetadataValue] = {
                "source_item_id": str(item.id),
            }
            if include_group_metadata:
                item_metadata["group_key"] = str(group_key)

            # Create simple text-based cloze (fallback without template)
            # Blank out the first n_slots_to_unfill words
            tokens = text.split()
            if n_slots_to_unfill > len(tokens):
                # Skip items that are too short
                continue

            blank_positions = list(range(n_slots_to_unfill))

            cloze_item = create_simple_cloze_item(
                text=text,
                blank_positions=blank_positions,
                item_template_id=item_template_id,
                metadata=item_metadata,
            )
            cloze_items.append(cloze_item)

    return cloze_items


def create_filtered_cloze_items(
    templates: list[Any],
    n_slots_to_unfill: int = 1,
    *,
    template_filter: Callable[[Any], bool] | None = None,
    slot_filter: Callable[[str, Any], bool] | None = None,
    item_template_id: UUID | None = None,
) -> list[Item]:
    """Create cloze items with multi-level filtering.

    Filters templates and/or slots before creating cloze items.

    Parameters
    ----------
    templates : list[Template]
        Source templates.
    n_slots_to_unfill : int
        Number of slots to unfill.
    template_filter : Callable[[Template], bool] | None
        Filter templates.
    slot_filter : Callable[[str, Slot], bool] | None
        Filter which slots can be unfilled (receives slot_name and Slot object).
    item_template_id : UUID | None
        Template ID for created items.

    Returns
    -------
    list[Item]
        Filtered cloze items.

    Examples
    --------
    >>> # Only unfill slots with constraints
    >>> cloze_items = create_filtered_cloze_items(
    ...     templates=all_templates,
    ...     n_slots_to_unfill=1,
    ...     template_filter=lambda t: len(t.slots) >= 3,
    ...     slot_filter=lambda name, slot: len(slot.constraints) > 0
    ... )  # doctest: +SKIP
    """
    # Filter templates
    filtered_templates = templates
    if template_filter:
        filtered_templates = [t for t in templates if template_filter(t)]

    cloze_items: list[Item] = []

    for template in filtered_templates:
        # Filter slots if slot_filter provided
        available_slots = list(template.slots.keys())
        if slot_filter:
            available_slots = [
                name
                for name in available_slots
                if slot_filter(name, template.slots[name])
            ]

        # Skip if not enough slots
        if len(available_slots) < n_slots_to_unfill:
            continue

        # Create cloze items from this template
        items = create_cloze_items_from_template(
            template=template,
            n_unfilled=n_slots_to_unfill,
            strategy="all_combinations",
            item_template_id=item_template_id,
        )

        # Further filter items if slot_filter was used
        if slot_filter:
            # Only keep items where all unfilled slots pass the filter
            items = [
                item
                for item in items
                if all(
                    slot.slot_name in available_slots for slot in item.unfilled_slots
                )
            ]

        cloze_items.extend(items)

    return cloze_items


def _extract_text_from_item(item: Item) -> str:
    """Extract text from item's rendered_elements.

    Tries common keys: "text", "sentence", "content".
    Raises error if no suitable text found.

    Parameters
    ----------
    item : Item
        Item to extract text from.

    Returns
    -------
    str
        Extracted text.

    Raises
    ------
    ValueError
        If no suitable text key found in rendered_elements.
    """
    for key in ["text", "sentence", "content"]:
        if key in item.rendered_elements:
            return item.rendered_elements[key]

    raise ValueError(
        f"Cannot extract text from item {item.id}. "
        f"Expected one of ['text', 'sentence', 'content'] in rendered_elements, "
        f"but found keys: {list(item.rendered_elements.keys())}. "
        f"Use the extract_text parameter to provide a custom extraction function."
    )


# Helper functions


def _validate_cloze_parameters(
    template: Any, unfilled_slot_names: list[str], filled_slots: dict[str, str]
) -> None:
    """Validate cloze item parameters.

    Raises
    ------
    ValueError
        If validation fails with descriptive message.
    """
    # Check unfilled_slot_names not empty
    if not unfilled_slot_names:
        raise ValueError(
            "Must have at least 1 unfilled slot. "
            "Provide at least one slot name in unfilled_slot_names parameter."
        )

    # Check all unfilled slots exist in template
    for slot_name in unfilled_slot_names:
        if slot_name not in template.slots:
            raise ValueError(
                f"Unfilled slot '{slot_name}' not found in template. "
                f"Available slots: {list(template.slots.keys())}"
            )

    # Check filled_slots if provided
    if filled_slots:
        for slot_name in filled_slots.keys():
            if slot_name not in template.slots:
                raise ValueError(
                    f"Filled slot '{slot_name}' not found in template. "
                    f"Available slots: {list(template.slots.keys())}"
                )

        # Check no overlap
        overlap = set(unfilled_slot_names) & set(filled_slots.keys())
        if overlap:
            raise ValueError(
                f"Slots cannot be both filled and unfilled. "
                f"Overlapping slots: {overlap}"
            )


def _render_template_for_cloze(
    template_string: str, filled_slots: dict[str, str], unfilled_slot_names: list[str]
) -> str:
    """Render template with filled values and '___' for unfilled slots.

    Parameters
    ----------
    template_string : str
        Template string with {slot_name} placeholders.
    filled_slots : dict[str, str]
        Mapping of slot names to fill values.
    unfilled_slot_names : list[str]
        Names of slots to leave unfilled (replaced with "___").

    Returns
    -------
    str
        Rendered template string.
    """
    result = template_string

    # Replace unfilled slots with "___"
    for slot_name in unfilled_slot_names:
        result = result.replace(f"{{{slot_name}}}", "___")

    # Replace filled slots with their values
    for slot_name, value in filled_slots.items():
        result = result.replace(f"{{{slot_name}}}", value)

    return result


def _calculate_positions(
    template_string: str, unfilled_slot_names: list[str], filled_slots: dict[str, str]
) -> dict[str, int]:
    """Calculate token positions for unfilled slots.

    Parameters
    ----------
    template_string : str
        Template string with {slot_name} placeholders.
    unfilled_slot_names : list[str]
        Names of slots that are unfilled.
    filled_slots : dict[str, str]
        Mapping of slot names to fill values.

    Returns
    -------
    dict[str, int]
        Mapping from slot_name to position (token index, 0-indexed).
    """
    # Extract all slot placeholders in order
    slot_pattern = re.compile(r"\{(\w+)\}")
    slot_matches = slot_pattern.finditer(template_string)

    positions: dict[str, int] = {}
    token_index = 0

    # Track position in template string
    last_end = 0

    for match in slot_matches:
        slot_name = match.group(1)

        # Count tokens before this slot
        text_before = template_string[last_end : match.start()]
        # Split by whitespace and count non-empty tokens
        tokens_before = [t for t in text_before.split() if t]
        token_index += len(tokens_before)

        # This slot becomes one token (either filled value or "___")
        if slot_name in unfilled_slot_names:
            positions[slot_name] = token_index

        token_index += 1
        last_end = match.end()

    return positions


def _extract_constraint_ids(template: Any, slot_name: str) -> tuple[UUID, ...]:
    """Extract constraint UUIDs from a template slot.

    Parameters
    ----------
    template : Template
        Source template.
    slot_name : str
        Name of slot to extract constraints from.

    Returns
    -------
    list[UUID]
        Constraint UUIDs for this slot.
    """
    if slot_name not in template.slots:
        return ()

    slot = template.slots[slot_name]

    if not hasattr(slot, "constraints") or slot.constraints is None:
        return ()

    return tuple(c.id for c in slot.constraints if hasattr(c, "id"))
