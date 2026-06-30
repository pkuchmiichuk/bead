"""Utilities for creating ordinal scale experimental items.

This module provides language-agnostic utilities for creating ordinal scale
items where participants rate a single stimulus on an ordered discrete scale
(e.g., 1-7 Likert scale, acceptability ratings).

Integration Points
------------------
- Active Learning: bead/active_learning/models/ordinal_scale.py
- Simulation: bead/simulation/strategies/ordinal_scale.py
- Deployment: bead/deployment/jspsych/ (slider or radio buttons)
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Hashable
from itertools import product
from uuid import UUID, uuid4

from bead.items.item import Item, MetadataValue
from bead.items.item_template import ScaleBounds, ScalePointLabel

_DEFAULT_SCALE_BOUNDS = ScaleBounds(min=1, max=7)


def create_ordinal_scale_item(
    text: str,
    scale_bounds: ScaleBounds = _DEFAULT_SCALE_BOUNDS,
    prompt: str | None = None,
    scale_labels: dict[int, str] | tuple[ScalePointLabel, ...] | None = None,
    item_template_id: UUID | None = None,
    metadata: dict[str, MetadataValue] | None = None,
) -> Item:
    """Create an ordinal scale rating item.

    Parameters
    ----------
    text : str
        The stimulus text to rate.
    scale_bounds : tuple[int, int]
        Tuple of (min, max) for the scale. Both must be integers with min < max.
        Default: (1, 7) for a 7-point scale.
    prompt : str | None
        Optional question/prompt for the rating.
        If None, uses "Rate this item:".
    scale_labels : dict[int, str] | None
        Optional labels for specific scale values (e.g., {1: "Bad", 7: "Good"}).
        All keys must be within [scale_min, scale_max].
    item_template_id : UUID | None
        Template ID for the item. If None, generates new UUID.
    metadata : dict[str, MetadataValue] | None
        Additional metadata for item_metadata field.

    Returns
    -------
    Item
        Ordinal scale item with text and prompt in rendered_elements.

    Raises
    ------
    ValueError
        If text is empty, if scale_bounds are invalid, or if scale_labels
        contain values outside scale bounds.

    Examples
    --------
    >>> item = create_ordinal_scale_item(
    ...     text="The cat sat on the mat.",
    ...     scale_bounds=ScaleBounds(min=1, max=7),
    ...     prompt="How natural is this sentence?",
    ...     metadata={"task": "acceptability"}
    ... )
    >>> item.rendered_elements["text"]
    'The cat sat on the mat.'
    >>> item.item_metadata["scale_min"]
    1
    >>> item.item_metadata["scale_max"]
    7

    >>> # 5-point Likert with labels
    >>> item = create_ordinal_scale_item(
    ...     text="I enjoy linguistics.",
    ...     scale_bounds=ScaleBounds(min=1, max=5),
    ...     scale_labels=(
    ...         ScalePointLabel(point=1, label="Strongly Disagree"),
    ...         ScalePointLabel(point=5, label="Strongly Agree"),
    ...     )
    ... )
    >>> item.item_metadata["scale_labels"][1]
    'Strongly Disagree'
    """
    if not text or not text.strip():
        raise ValueError("text cannot be empty")

    scale_min, scale_max = scale_bounds.min, scale_bounds.max

    if scale_min >= scale_max:
        raise ValueError(
            f"scale_min ({scale_min}) must be less than scale_max ({scale_max})"
        )

    # Normalize scale_labels: accept dict[int, str] or tuple[ScalePointLabel, ...]
    if isinstance(scale_labels, tuple):
        scale_labels = {sl.point: sl.label for sl in scale_labels}
    if scale_labels:
        for value in scale_labels.keys():
            if not (scale_min <= value <= scale_max):
                raise ValueError(
                    f"scale_labels key {value} is outside scale bounds "
                    f"[{scale_min}, {scale_max}]"
                )

    if item_template_id is None:
        item_template_id = uuid4()

    if prompt is None:
        prompt = "Rate this item:"

    rendered_elements: dict[str, str] = {
        "text": text,
        "prompt": prompt,
    }

    # Build item metadata
    item_metadata: dict[str, MetadataValue] = {
        "scale_min": scale_min,
        "scale_max": scale_max,
    }

    if scale_labels:
        item_metadata["scale_labels"] = {str(k): v for k, v in scale_labels.items()}

    if metadata:
        item_metadata.update(metadata)

    return Item(
        item_template_id=item_template_id,
        rendered_elements=rendered_elements,
        item_metadata=item_metadata,
    )


def create_ordinal_scale_items_from_texts(
    texts: list[str],
    scale_bounds: ScaleBounds = _DEFAULT_SCALE_BOUNDS,
    prompt: str | None = None,
    scale_labels: dict[int, str] | tuple[ScalePointLabel, ...] | None = None,
    *,
    item_template_id: UUID | None = None,
    metadata_fn: Callable[[str], dict[str, MetadataValue]] | None = None,
) -> list[Item]:
    """Create ordinal scale items from a list of texts.

    Parameters
    ----------
    texts : list[str]
        List of stimulus texts.
    scale_bounds : tuple[int, int]
        Scale bounds (min, max) for all items.
    prompt : str | None
        The question/prompt for all items.
    scale_labels : dict[int, str] | None
        Optional scale labels for all items.
    item_template_id : UUID | None
        Template ID for all created items. If None, generates one per item.
    metadata_fn : Callable[[str], dict[str, MetadataValue]] | None
        Function to generate metadata from each text.

    Returns
    -------
    list[Item]
        Ordinal scale items for each text.

    Examples
    --------
    >>> texts = ["She walks.", "She walk.", "They walk."]
    >>> items = create_ordinal_scale_items_from_texts(
    ...     texts,
    ...     scale_bounds=ScaleBounds(min=1, max=5),
    ...     prompt="How acceptable is this sentence?",
    ...     metadata_fn=lambda t: {"text_length": len(t)}
    ... )
    >>> len(items)
    3
    >>> items[0].item_metadata["scale_min"]
    1
    """
    ordinal_items: list[Item] = []

    for text in texts:
        item_metadata: dict[str, MetadataValue] = {}
        if metadata_fn:
            item_metadata = metadata_fn(text)

        item = create_ordinal_scale_item(
            text=text,
            scale_bounds=scale_bounds,
            prompt=prompt,
            scale_labels=scale_labels,
            item_template_id=item_template_id,
            metadata=item_metadata,
        )
        ordinal_items.append(item)

    return ordinal_items


def create_ordinal_scale_items_from_groups(
    items: list[Item],
    group_by: Callable[[Item], Hashable],
    scale_bounds: ScaleBounds = _DEFAULT_SCALE_BOUNDS,
    prompt: str | None = None,
    scale_labels: dict[int, str] | tuple[ScalePointLabel, ...] | None = None,
    *,
    extract_text: Callable[[Item], str] | None = None,
    include_group_metadata: bool = True,
    item_template_id: UUID | None = None,
) -> list[Item]:
    """Create ordinal scale items from grouped source items.

    Groups items and creates one ordinal scale item per source item,
    preserving group information in metadata.

    Parameters
    ----------
    items : list[Item]
        Source items to process.
    group_by : Callable[[Item], Hashable]
        Function to extract grouping key from items.
    scale_bounds : tuple[int, int]
        Scale bounds (min, max) for all items.
    prompt : str | None
        The question/prompt for all items.
    scale_labels : dict[int, str] | None
        Optional scale labels for all items.
    extract_text : Callable[[Item], str] | None
        Function to extract text from item. If None, tries common keys.
    include_group_metadata : bool
        Whether to include group key in item metadata.
    item_template_id : UUID | None
        Template ID for all created items. If None, generates one per item.

    Returns
    -------
    list[Item]
        Ordinal scale items from source items.

    Examples
    --------
    >>> source_items = [
    ...     Item(
    ...         uuid4(),
    ...         rendered_elements={"text": "She walks."},
    ...         item_metadata={"verb": "walk"}
    ...     )
    ... ]
    >>> ordinal_items = create_ordinal_scale_items_from_groups(
    ...     source_items,
    ...     group_by=lambda i: i.item_metadata["verb"],
    ...     scale_bounds=ScaleBounds(min=1, max=7),
    ...     prompt="Rate the acceptability:"
    ... )
    >>> len(ordinal_items)
    1
    """
    # Group items
    groups: dict[Hashable, list[Item]] = defaultdict(list)
    for item in items:
        group_key = group_by(item)
        groups[group_key].append(item)

    ordinal_items: list[Item] = []

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

            # Create ordinal scale item
            ordinal_item = create_ordinal_scale_item(
                text=text,
                scale_bounds=scale_bounds,
                prompt=prompt,
                scale_labels=scale_labels,
                item_template_id=item_template_id,
                metadata=item_metadata,
            )
            ordinal_items.append(ordinal_item)

    return ordinal_items


def create_ordinal_scale_items_cross_product(
    texts: list[str],
    prompts: list[str],
    scale_bounds: ScaleBounds = _DEFAULT_SCALE_BOUNDS,
    scale_labels: dict[int, str] | tuple[ScalePointLabel, ...] | None = None,
    *,
    item_template_id: UUID | None = None,
    metadata_fn: (Callable[[str, str], dict[str, MetadataValue]] | None) = None,
) -> list[Item]:
    """Create ordinal scale items from cross-product of texts and prompts.

    Useful when you want to apply multiple prompts to each text.

    Parameters
    ----------
    texts : list[str]
        List of stimulus texts.
    prompts : list[str]
        List of prompts to apply.
    scale_bounds : tuple[int, int]
        Scale bounds (min, max) for all items.
    scale_labels : dict[int, str] | None
        Optional scale labels for all items.
    item_template_id : UUID | None
        Template ID for all created items.
    metadata_fn : Callable[[str, str], dict[str, MetadataValue]] | None
        Function to generate metadata from (text, prompt).

    Returns
    -------
    list[Item]
        Ordinal scale items from cross-product.

    Examples
    --------
    >>> texts = ["The cat sat.", "The dog ran."]
    >>> prompts = ["How natural is this?", "How acceptable is this?"]
    >>> items = create_ordinal_scale_items_cross_product(
    ...     texts, prompts, scale_bounds=ScaleBounds(min=1, max=5)
    ... )
    >>> len(items)
    4
    """
    ordinal_items: list[Item] = []

    for text, prompt in product(texts, prompts):
        item_metadata: dict[str, MetadataValue] = {}
        if metadata_fn:
            item_metadata = metadata_fn(text, prompt)

        item = create_ordinal_scale_item(
            text=text,
            scale_bounds=scale_bounds,
            prompt=prompt,
            scale_labels=scale_labels,
            item_template_id=item_template_id,
            metadata=item_metadata,
        )
        ordinal_items.append(item)

    return ordinal_items


def create_filtered_ordinal_scale_items(
    items: list[Item],
    scale_bounds: ScaleBounds = _DEFAULT_SCALE_BOUNDS,
    prompt: str | None = None,
    scale_labels: dict[int, str] | tuple[ScalePointLabel, ...] | None = None,
    *,
    item_filter: Callable[[Item], bool] | None = None,
    extract_text: Callable[[Item], str] | None = None,
    item_template_id: UUID | None = None,
) -> list[Item]:
    """Create ordinal scale items with filtering.

    Parameters
    ----------
    items : list[Item]
        Source items.
    scale_bounds : tuple[int, int]
        Scale bounds (min, max) for all items.
    prompt : str | None
        The question/prompt for all items.
    scale_labels : dict[int, str] | None
        Optional scale labels for all items.
    item_filter : Callable[[Item], bool] | None
        Filter individual items.
    extract_text : Callable[[Item], str] | None
        Text extraction function.
    item_template_id : UUID | None
        Template ID for created items.

    Returns
    -------
    list[Item]
        Filtered ordinal scale items.

    Examples
    --------
    >>> ordinal_items = create_filtered_ordinal_scale_items(
    ...     items,
    ...     scale_bounds=ScaleBounds(min=1, max=7),
    ...     prompt="Rate the acceptability:",
    ...     item_filter=lambda i: i.item_metadata.get("valid", True)
    ... )  # doctest: +SKIP
    """
    # Filter items
    filtered_items = items
    if item_filter:
        filtered_items = [item for item in items if item_filter(item)]

    ordinal_items: list[Item] = []

    for item in filtered_items:
        # Extract text
        if extract_text:
            text: str = extract_text(item)
        else:
            text = _extract_text_from_item(item)

        # Create ordinal scale item
        item_metadata: dict[str, MetadataValue] = {
            "source_item_id": str(item.id),
        }

        ordinal_item = create_ordinal_scale_item(
            text=text,
            scale_bounds=scale_bounds,
            prompt=prompt,
            scale_labels=scale_labels,
            item_template_id=item_template_id,
            metadata=item_metadata,
        )
        ordinal_items.append(ordinal_item)

    return ordinal_items


def create_likert_5_item(
    text: str,
    prompt: str | None = None,
    item_template_id: UUID | None = None,
    metadata: dict[str, MetadataValue] | None = None,
) -> Item:
    """Create a 5-point Likert scale item.

    Convenience function for standard 5-point Likert scale with
    "Strongly Disagree" to "Strongly Agree" labels.

    Parameters
    ----------
    text : str
        The stimulus text (statement) to rate.
    prompt : str | None
        Optional prompt. If None, uses "Rate your agreement:".
    item_template_id : UUID | None
        Template ID for the item. If None, generates new UUID.
    metadata : dict[str, MetadataValue] | None
        Additional metadata for item_metadata field.

    Returns
    -------
    Item
        5-point Likert scale item.

    Examples
    --------
    >>> item = create_likert_5_item("I enjoy studying linguistics.")
    >>> item.item_metadata["scale_min"]
    1
    >>> item.item_metadata["scale_max"]
    5
    """
    if prompt is None:
        prompt = "Rate your agreement:"

    return create_ordinal_scale_item(
        text,
        scale_bounds=ScaleBounds(min=1, max=5),
        prompt=prompt,
        scale_labels=(
            ScalePointLabel(point=1, label="Strongly Disagree"),
            ScalePointLabel(point=2, label="Disagree"),
            ScalePointLabel(point=3, label="Neutral"),
            ScalePointLabel(point=4, label="Agree"),
            ScalePointLabel(point=5, label="Strongly Agree"),
        ),
        item_template_id=item_template_id,
        metadata=metadata,
    )


def create_likert_7_item(
    text: str,
    prompt: str | None = None,
    item_template_id: UUID | None = None,
    metadata: dict[str, MetadataValue] | None = None,
) -> Item:
    """Create a 7-point Likert scale item.

    Convenience function for standard 7-point Likert scale with
    "Strongly Disagree" to "Strongly Agree" labels.

    Parameters
    ----------
    text : str
        The stimulus text (statement) to rate.
    prompt : str | None
        Optional prompt. If None, uses "Rate your agreement:".
    item_template_id : UUID | None
        Template ID for the item. If None, generates new UUID.
    metadata : dict[str, MetadataValue] | None
        Additional metadata for item_metadata field.

    Returns
    -------
    Item
        7-point Likert scale item.

    Examples
    --------
    >>> item = create_likert_7_item("I enjoy studying linguistics.")
    >>> item.item_metadata["scale_min"]
    1
    >>> item.item_metadata["scale_max"]
    7
    """
    if prompt is None:
        prompt = "Rate your agreement:"

    return create_ordinal_scale_item(
        text,
        scale_bounds=ScaleBounds(min=1, max=7),
        prompt=prompt,
        scale_labels=(
            ScalePointLabel(point=1, label="Strongly Disagree"),
            ScalePointLabel(point=7, label="Strongly Agree"),
        ),
        item_template_id=item_template_id,
        metadata=metadata,
    )


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
