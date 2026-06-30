"""Utilities for creating N-AFC (forced-choice) experimental items.

This module provides language-agnostic utilities for creating forced-choice
items where participants select from N alternatives (2AFC, 3AFC, 4AFC, etc.).
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable
from itertools import combinations, product
from typing import Any
from uuid import UUID, uuid4

from bead.items.item import Item, MetadataValue


def create_forced_choice_item(
    *options: str,
    item_template_id: UUID | None = None,
    metadata: dict[str, MetadataValue] | None = None,
) -> Item:
    """Create an N-AFC (forced-choice) item from N text options.

    Parameters
    ----------
    *options : str
        Text for each option (2 or more required).
    item_template_id : UUID | None
        Template ID for the item. If None, generates new UUID.
    metadata : dict[str, MetadataValue] | None
        Additional metadata for item_metadata field.

    Returns
    -------
    Item
        Forced-choice item with options stored in the options field.

    Raises
    ------
    ValueError
        If fewer than 2 options provided.

    Examples
    --------
    >>> item = create_forced_choice_item(
    ...     "The cat sat on the mat.",
    ...     "The cats sat on the mat.",
    ...     metadata={"contrast": "number"}
    ... )
    >>> item.options[0]
    'The cat sat on the mat.'
    >>> item.options[1]
    'The cats sat on the mat.'

    >>> # 4AFC item
    >>> item = create_forced_choice_item(
    ...     "Option A text",
    ...     "Option B text",
    ...     "Option C text",
    ...     "Option D text"
    ... )
    >>> len(item.options)
    4
    """
    if len(options) < 2:
        raise ValueError("At least 2 options required for forced-choice item")

    if item_template_id is None:
        item_template_id = uuid4()

    # Build item metadata with n_options (consistent with other task types)
    item_metadata: dict[str, MetadataValue] = {
        "n_options": len(options),
    }
    if metadata:
        item_metadata.update(metadata)

    return Item(
        item_template_id=item_template_id,
        options=tuple(options),
        item_metadata=item_metadata,
    )


def create_forced_choice_items_from_groups(
    items: list[Item],
    group_by: Callable[[Item], Any],
    n_alternatives: int = 2,
    *,
    extract_text: Callable[[Item], str] | None = None,
    include_group_metadata: bool = True,
    item_template_id: UUID | None = None,
) -> list[Item]:
    """Create forced-choice items by grouping source items.

    Groups items by a property, then creates all N-way combinations within
    each group as forced-choice items.

    Parameters
    ----------
    items : list[Item]
        Source items to group and combine.
    group_by : Callable[[Item], Any]
        Function to extract grouping key from items.
    n_alternatives : int
        Number of alternatives per forced-choice item (default: 2 for 2AFC).
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
        Forced-choice items created from groupings.

    Examples
    --------
    Create 2AFC items with same verb (same-verb minimal pairs):
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
    ...     )
    ... ]
    >>> fc_items = create_forced_choice_items_from_groups(
    ...     items,
    ...     group_by=lambda item: item.item_metadata["verb"],
    ...     n_alternatives=2
    ... )
    >>> len(fc_items)
    1
    >>> fc_items[0].rendered_elements["option_a"]
    'She walks.'

    Create 3AFC items grouped by template:
    >>> fc_items = create_forced_choice_items_from_groups(
    ...     items,
    ...     group_by=lambda item: item.item_template_id,
    ...     n_alternatives=3
    ... )  # doctest: +SKIP
    """
    # Group items
    groups: dict[Any, list[Item]] = defaultdict(list)
    for item in items:
        group_key = group_by(item)
        groups[group_key].append(item)

    # Create forced-choice items from each group
    fc_items: list[Item] = []

    for group_key, group_items in groups.items():
        # Generate all N-way combinations within group
        for combo in combinations(group_items, n_alternatives):
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

            # Create forced-choice item
            fc_item = create_forced_choice_item(
                *texts, item_template_id=item_template_id, metadata=metadata
            )
            fc_items.append(fc_item)

    return fc_items


def create_forced_choice_items_cross_product(
    group1_items: list[Item],
    group2_items: list[Item],
    n_from_group1: int = 1,
    n_from_group2: int = 1,
    *,
    extract_text: Callable[[Item], str] | None = None,
    item_template_id: UUID | None = None,
    metadata_fn: (
        Callable[[list[Item], list[Item]], dict[str, MetadataValue]] | None
    ) = None,
) -> list[Item]:
    """Create forced-choice items from cross-product of two groups.

    Combines n items from group1 with n items from group2 to create
    (n_from_group1 + n_from_group2)-AFC items.

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
    extract_text : Callable[[Item], str] | None
        Function to extract text from items.
    item_template_id : UUID | None
        Template ID for all created items.
    metadata_fn : Callable[[list[Item], list[Item]], dict[str, MetadataValue]] | None
        Function to generate metadata from (group1_items_used, group2_items_used).

    Returns
    -------
    list[Item]
        Forced-choice items from cross-product.

    Examples
    --------
    Create 2AFC items pairing grammatical with ungrammatical:
    >>> grammatical = [
    ...     Item(
    ...         uuid4(),
    ...         rendered_elements={"text": "She walks."},
    ...         item_metadata={"grammatical": True}
    ...     )
    ... ]
    >>> ungrammatical = [
    ...     Item(
    ...         uuid4(),
    ...         rendered_elements={"text": "She walk."},
    ...         item_metadata={"grammatical": False}
    ...     )
    ... ]
    >>> fc_items = create_forced_choice_items_cross_product(
    ...     grammatical,
    ...     ungrammatical,
    ...     n_from_group1=1,
    ...     n_from_group2=1
    ... )
    >>> len(fc_items)
    1
    """
    # Generate combinations from each group
    group1_combos = list(combinations(group1_items, n_from_group1))
    group2_combos = list(combinations(group2_items, n_from_group2))

    fc_items: list[Item] = []

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

        # Create forced-choice item
        fc_item = create_forced_choice_item(
            *texts, item_template_id=item_template_id, metadata=metadata
        )
        fc_items.append(fc_item)

    return fc_items


def create_filtered_forced_choice_items(
    items: list[Item],
    group_by: Callable[[Item], Any],
    n_alternatives: int = 2,
    *,
    item_filter: Callable[[Item], bool] | None = None,
    group_filter: Callable[[Any, list[Item]], bool] | None = None,
    combination_filter: Callable[[tuple[Item, ...]], bool] | None = None,
    extract_text: Callable[[Item], str] | None = None,
    item_template_id: UUID | None = None,
) -> list[Item]:
    """Create forced-choice items with multi-level filtering.

    Parameters
    ----------
    items : list[Item]
        Source items.
    group_by : Callable[[Item], Any]
        Grouping function.
    n_alternatives : int
        Number of alternatives per item.
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
        Filtered forced-choice items.

    Examples
    --------
    >>> fc_items = create_filtered_forced_choice_items(
    ...     items,
    ...     group_by=lambda i: i.item_metadata["verb"],
    ...     n_alternatives=2,
    ...     item_filter=lambda i: i.item_metadata.get("valid", True),
    ...     group_filter=lambda key, items: len(items) >= 2,
    ...     combination_filter=lambda combo: combo[0].id != combo[1].id
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
    fc_items: list[Item] = []
    for group_key, group_items in groups.items():
        for combo in combinations(group_items, n_alternatives):
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

            fc_item = create_forced_choice_item(
                *texts, item_template_id=item_template_id, metadata=metadata
            )
            fc_items.append(fc_item)

    return fc_items


def _extract_text_from_item(item: Item) -> str:
    """Extract text from item's rendered_elements.

    Tries common keys: "text", "sentence", "content".
    Falls back to string representation if not found.

    Parameters
    ----------
    item : Item
        Item to extract text from.

    Returns
    -------
    str
        Extracted text.
    """
    for key in ["text", "sentence", "content"]:
        if key in item.rendered_elements:
            return item.rendered_elements[key]

    # Fallback: use first value or string representation
    if item.rendered_elements:
        return next(iter(item.rendered_elements.values()))

    return str(item.rendered_elements)
