"""Utilities for creating multi-select experimental items.

This module provides language-agnostic utilities for creating multi-select
items where participants select one or more options from a set (checkboxes).

Integration Points
------------------
- Active Learning: bead/active_learning/models/multi_select.py
- Simulation: bead/simulation/strategies/multi_select.py
- Deployment: bead/deployment/jspsych/ (checkbox plugin)
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from itertools import combinations, product
from typing import Any
from uuid import UUID, uuid4

from bead.items.item import Item, MetadataValue


def create_multi_select_item(
    *options: str,
    min_selections: int = 1,
    max_selections: int | None = None,
    item_template_id: UUID | None = None,
    metadata: dict[str, MetadataValue] | None = None,
) -> Item:
    """Create a multi-select item from N text options.

    Parameters
    ----------
    *options : str
        Text for each option (2 or more required).
    min_selections : int
        Minimum number of options that must be selected (default: 1).
    max_selections : int | None
        Maximum number of options that can be selected. If None, defaults to
        number of options (no upper limit).
    item_template_id : UUID | None
        Template ID for the item. If None, generates new UUID.
    metadata : dict[str, MetadataValue] | None
        Additional metadata for item_metadata field.

    Returns
    -------
    Item
        Multi-select item with options stored in the options field.

    Raises
    ------
    ValueError
        If fewer than 2 options provided, or if min_selections > max_selections,
        or if min_selections < 1, or if max_selections > number of options.

    Examples
    --------
    >>> item = create_multi_select_item(
    ...     "She walks.",
    ...     "She walk.",
    ...     "They walks.",
    ...     "They walk.",
    ...     min_selections=1,
    ...     max_selections=4,
    ...     metadata={"task": "select_grammatical"}
    ... )
    >>> item.options[0]
    'She walks.'
    >>> item.item_metadata["min_selections"]
    1
    >>> item.item_metadata["max_selections"]
    4

    >>> # Multi-select with default max (all options)
    >>> item = create_multi_select_item(
    ...     "Option A",
    ...     "Option B",
    ...     "Option C"
    ... )
    >>> item.item_metadata["max_selections"]
    3
    """
    if len(options) < 2:
        raise ValueError("At least 2 options required for multi-select item")

    if max_selections is None:
        max_selections = len(options)

    if min_selections < 1:
        raise ValueError("min_selections must be at least 1")

    if min_selections > max_selections:
        raise ValueError("min_selections cannot be greater than max_selections")

    if max_selections > len(options):
        raise ValueError(
            f"max_selections ({max_selections}) cannot exceed "
            f"number of options ({len(options)})"
        )

    if item_template_id is None:
        item_template_id = uuid4()

    # Build item metadata
    item_metadata: dict[str, MetadataValue] = {
        "min_selections": min_selections,
        "max_selections": max_selections,
    }
    if metadata:
        item_metadata.update(metadata)

    return Item(
        item_template_id=item_template_id,
        options=tuple(options),
        item_metadata=item_metadata,
    )


def create_multi_select_items_from_groups(
    items: list[Item],
    group_by: Callable[[Item], Any],
    n_options: int | None = None,
    min_selections: int = 1,
    max_selections: int | None = None,
    *,
    extract_text: Callable[[Item], str] | None = None,
    include_group_metadata: bool = True,
    item_template_id: UUID | None = None,
) -> list[Item]:
    """Create multi-select items by grouping source items.

    Groups items by a property, then creates multi-select items from each
    group's items as options.

    Parameters
    ----------
    items : list[Item]
        Source items to group and combine.
    group_by : Callable[[Item], Any]
        Function to extract grouping key from items.
    n_options : int | None
        Number of options per multi-select item. If None, uses all items in
        each group.
    min_selections : int
        Minimum number of selections required (default: 1).
    max_selections : int | None
        Maximum number of selections allowed. If None, defaults to n_options.
    extract_text : Callable[[Item], str] | None
        Function to extract text from item. If None, tries common keys
        ("text", "sentence", "content") from rendered_elements.
    include_group_metadata : bool
        Whether to include group key in item metadata.
    item_template_id : UUID | None
        Template ID for all created items. If None, generates one per item.

    Returns
    -------
    list[Item]
        Multi-select items created from groupings.

    Examples
    --------
    Create multi-select items grouped by verb (select all acceptable frames):
    >>> items = [
    ...     Item(
    ...         item_template_id=uuid4(),
    ...         rendered_elements={"text": "She walks."},
    ...         item_metadata={"verb": "walk", "frame": "intransitive"}
    ...     ),
    ...     Item(
    ...         item_template_id=uuid4(),
    ...         rendered_elements={"text": "She walks the dog."},
    ...         item_metadata={"verb": "walk", "frame": "transitive"}
    ...     ),
    ...     Item(
    ...         item_template_id=uuid4(),
    ...         rendered_elements={"text": "She walks to school."},
    ...         item_metadata={"verb": "walk", "frame": "intransitive_pp"}
    ...     )
    ... ]
    >>> ms_items = create_multi_select_items_from_groups(
    ...     items,
    ...     group_by=lambda item: item.item_metadata["verb"],
    ...     min_selections=1,
    ...     max_selections=3
    ... )
    >>> len(ms_items)
    1
    >>> len(ms_items[0].rendered_elements)
    3
    """
    # Group items
    groups: dict[Any, list[Item]] = defaultdict(list)
    for item in items:
        group_key = group_by(item)
        groups[group_key].append(item)

    # Create multi-select items from each group
    ms_items: list[Item] = []

    for group_key, group_items in groups.items():
        # Validate n_options
        if n_options is not None and n_options > len(group_items):
            raise ValueError(
                f"Group '{group_key}' has only {len(group_items)} item(s), "
                f"but n_options={n_options} was requested. "
                f"Cannot create {n_options}-option items from fewer items."
            )

        # If n_options specified, create combinations
        if n_options is not None and n_options < len(group_items):
            item_combos = combinations(group_items, n_options)
        else:
            # Use all items in group as single combination
            item_combos = [tuple(group_items)]

        for combo in item_combos:
            # Extract text from each item
            texts: list[str] = []
            for item in combo:
                if extract_text:
                    text: str = extract_text(item)
                else:
                    text = _extract_text_from_item(item)
                texts.append(text)

            # Build metadata
            metadata: dict[str, MetadataValue] = {}
            if include_group_metadata:
                metadata["group_key"] = str(group_key)

            # Include source item IDs
            for i, item in enumerate(combo):
                metadata[f"source_item_{i}_id"] = str(item.id)

            # Create multi-select item
            ms_item = create_multi_select_item(
                *texts,
                min_selections=min_selections,
                max_selections=max_selections,
                item_template_id=item_template_id,
                metadata=metadata,
            )
            ms_items.append(ms_item)

    return ms_items


def create_multi_select_items_with_foils(
    correct_items: list[Item],
    foil_items: list[Item],
    n_correct: int = 2,
    n_foils: int = 2,
    *,
    extract_text: Callable[[Item], str] | None = None,
    item_template_id: UUID | None = None,
    metadata_fn: (
        Callable[[list[Item], list[Item]], dict[str, MetadataValue]] | None
    ) = None,
) -> list[Item]:
    """Create multi-select items by combining correct items with foils.

    Useful for tasks like "Select all grammatical sentences" where some
    options are correct and others are foils (distractors).

    Parameters
    ----------
    correct_items : list[Item]
        Items that are correct (should be selected).
    foil_items : list[Item]
        Items that are foils/distractors (should not be selected).
    n_correct : int
        Number of correct items to include per multi-select item (default: 2).
    n_foils : int
        Number of foil items to include per multi-select item (default: 2).
    extract_text : Callable[[Item], str] | None
        Function to extract text from items.
    item_template_id : UUID | None
        Template ID for all created items.
    metadata_fn : Callable[[list[Item], list[Item]], dict[str, MetadataValue]] | None
        Function to generate metadata from (correct_items_used, foil_items_used).

    Returns
    -------
    list[Item]
        Multi-select items with correct items and foils.

    Examples
    --------
    >>> grammatical = [
    ...     Item(uuid4(), rendered_elements={"text": "She walks."},
    ...          item_metadata={"grammatical": True}),
    ...     Item(uuid4(), rendered_elements={"text": "They walk."},
    ...          item_metadata={"grammatical": True})
    ... ]
    >>> ungrammatical = [
    ...     Item(uuid4(), rendered_elements={"text": "She walk."},
    ...          item_metadata={"grammatical": False}),
    ...     Item(uuid4(), rendered_elements={"text": "They walks."},
    ...          item_metadata={"grammatical": False})
    ... ]
    >>> ms_items = create_multi_select_items_with_foils(
    ...     grammatical,
    ...     ungrammatical,
    ...     n_correct=2,
    ...     n_foils=2
    ... )
    >>> len(ms_items)
    1
    >>> ms_items[0].item_metadata["min_selections"]
    1
    >>> ms_items[0].item_metadata["max_selections"]
    4
    """
    # Generate combinations from each group
    correct_combos = list(combinations(correct_items, n_correct))
    foil_combos = list(combinations(foil_items, n_foils))

    ms_items: list[Item] = []

    # Cross-product of combinations
    for correct_combo, foil_combo in product(correct_combos, foil_combos):
        all_items = list(correct_combo) + list(foil_combo)

        # Extract texts
        texts: list[str] = []
        for item in all_items:
            if extract_text:
                text: str = extract_text(item)
            else:
                text = _extract_text_from_item(item)
            texts.append(text)

        # Build metadata
        metadata: dict[str, MetadataValue]
        if metadata_fn:
            metadata = metadata_fn(list(correct_combo), list(foil_combo))
        else:
            metadata = {
                "correct_item_ids": tuple(str(item.id) for item in correct_combo),
                "foil_item_ids": tuple(str(item.id) for item in foil_combo),
                "n_correct": n_correct,
                "n_foils": n_foils,
            }

        # Create multi-select item
        # min_selections=1 (at least one must be selected)
        # max_selections=total (all can be selected)
        ms_item = create_multi_select_item(
            *texts,
            min_selections=1,
            max_selections=len(texts),
            item_template_id=item_template_id,
            metadata=metadata,
        )
        ms_items.append(ms_item)

    return ms_items


def create_multi_select_items_cross_product(
    group1_items: list[Item],
    group2_items: list[Item],
    n_from_group1: int = 1,
    n_from_group2: int = 1,
    min_selections: int = 1,
    max_selections: int | None = None,
    *,
    extract_text: Callable[[Item], str] | None = None,
    item_template_id: UUID | None = None,
    metadata_fn: (
        Callable[[list[Item], list[Item]], dict[str, MetadataValue]] | None
    ) = None,
) -> list[Item]:
    """Create multi-select items from cross-product of two groups.

    Combines n items from group1 with n items from group2 to create
    multi-select items with (n_from_group1 + n_from_group2) options.

    Parameters
    ----------
    group1_items : list[Item]
        Items in first group.
    group2_items : list[Item]
        Items in second group.
    n_from_group1 : int
        Number of items to select from group1 per combination (default: 1).
    n_from_group2 : int
        Number of items to select from group2 per combination (default: 1).
    min_selections : int
        Minimum number of selections required (default: 1).
    max_selections : int | None
        Maximum number of selections allowed. If None, defaults to total options.
    extract_text : Callable[[Item], str] | None
        Function to extract text from items.
    item_template_id : UUID | None
        Template ID for all created items.
    metadata_fn : Callable[[list[Item], list[Item]], dict[str, MetadataValue]] | None
        Function to generate metadata from (group1_items_used, group2_items_used).

    Returns
    -------
    list[Item]
        Multi-select items from cross-product.

    Examples
    --------
    >>> active = [Item(uuid4(), rendered_elements={"text": "She walks."})]
    >>> passive = [Item(uuid4(), rendered_elements={"text": "She is walked."})]
    >>> ms_items = create_multi_select_items_cross_product(
    ...     active, passive,
    ...     n_from_group1=1,
    ...     n_from_group2=1,
    ...     min_selections=1,
    ...     max_selections=2
    ... )
    >>> len(ms_items)
    1
    """
    # Generate combinations from each group
    group1_combos = list(combinations(group1_items, n_from_group1))
    group2_combos = list(combinations(group2_items, n_from_group2))

    ms_items: list[Item] = []

    # Cross-product of combinations
    for combo1, combo2 in product(group1_combos, group2_combos):
        all_items = list(combo1) + list(combo2)

        # Extract texts
        texts: list[str] = []
        for item in all_items:
            if extract_text:
                text: str = extract_text(item)
            else:
                text = _extract_text_from_item(item)
            texts.append(text)

        # Build metadata
        metadata: dict[str, MetadataValue]
        if metadata_fn:
            metadata = metadata_fn(list(combo1), list(combo2))
        else:
            metadata = {
                "source_group1_ids": tuple(str(item.id) for item in combo1),
                "source_group2_ids": tuple(str(item.id) for item in combo2),
            }

        # Create multi-select item
        ms_item = create_multi_select_item(
            *texts,
            min_selections=min_selections,
            max_selections=max_selections,
            item_template_id=item_template_id,
            metadata=metadata,
        )
        ms_items.append(ms_item)

    return ms_items


def create_filtered_multi_select_items(
    items: list[Item],
    group_by: Callable[[Item], Any],
    n_options: int | None = None,
    min_selections: int = 1,
    max_selections: int | None = None,
    *,
    item_filter: Callable[[Item], bool] | None = None,
    group_filter: Callable[[Any, list[Item]], bool] | None = None,
    combination_filter: Callable[[tuple[Item, ...]], bool] | None = None,
    extract_text: Callable[[Item], str] | None = None,
    item_template_id: UUID | None = None,
) -> list[Item]:
    """Create multi-select items with multi-level filtering.

    Parameters
    ----------
    items : list[Item]
        Source items.
    group_by : Callable[[Item], Any]
        Grouping function.
    n_options : int | None
        Number of options per item. If None, uses all items in each group.
    min_selections : int
        Minimum number of selections required.
    max_selections : int | None
        Maximum number of selections allowed.
    item_filter : Callable[[Item], bool] | None
        Filter individual items before grouping.
    group_filter : Callable[[Any, list[Item]], bool] | None
        Filter groups (receives group_key and group_items).
    combination_filter : Callable[[tuple[Item, ...]], bool] | None
        Filter specific combinations.
    extract_text : Callable[[Item], str] | None
        Text extraction function.
    item_template_id : UUID | None
        Template ID for created items.

    Returns
    -------
    list[Item]
        Filtered multi-select items.

    Examples
    --------
    >>> ms_items = create_filtered_multi_select_items(
    ...     items,
    ...     group_by=lambda i: i.item_metadata["verb"],
    ...     n_options=3,
    ...     item_filter=lambda i: i.item_metadata.get("valid", True),
    ...     group_filter=lambda key, items: len(items) >= 3,
    ...     min_selections=1,
    ...     max_selections=3
    ... )  # doctest: +SKIP
    """
    # Filter items
    filtered_items = items
    if item_filter:
        filtered_items = [item for item in items if item_filter(item)]

    # Group items
    groups: dict[Any, list[Item]] = defaultdict(list)
    for item in filtered_items:
        group_key = group_by(item)
        groups[group_key].append(item)

    # Filter groups
    if group_filter:
        groups = {k: v for k, v in groups.items() if group_filter(k, v)}

    # Create combinations
    ms_items: list[Item] = []
    for group_key, group_items in groups.items():
        # Validate group size
        if len(group_items) < 2:
            raise ValueError(
                f"Group '{group_key}' has only {len(group_items)} item(s) "
                f"after filtering. Multi-select requires at least 2 items. "
                f"Use group_filter to exclude small groups."
            )

        # Validate n_options
        if n_options is not None and n_options > len(group_items):
            raise ValueError(
                f"Group '{group_key}' has only {len(group_items)} item(s), "
                f"but n_options={n_options} was requested. "
                f"Cannot create {n_options}-option items from fewer items."
            )

        # Determine combinations
        if n_options is not None and n_options < len(group_items):
            item_combos = combinations(group_items, n_options)
        else:
            item_combos = [tuple(group_items)]

        for combo in item_combos:
            # Filter combination
            if combination_filter and not combination_filter(combo):
                continue

            # Extract texts
            texts: list[str] = []
            for item in combo:
                if extract_text:
                    text: str = extract_text(item)
                else:
                    text = _extract_text_from_item(item)
                texts.append(text)

            # Create item
            metadata: dict[str, MetadataValue] = {
                "group_key": str(group_key),
                "source_item_ids": tuple(str(item.id) for item in combo),
            }

            ms_item = create_multi_select_item(
                *texts,
                min_selections=min_selections,
                max_selections=max_selections,
                item_template_id=item_template_id,
                metadata=metadata,
            )
            ms_items.append(ms_item)

    return ms_items


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
