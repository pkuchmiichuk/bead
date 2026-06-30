"""Utilities for creating binary experimental items.

This module provides language-agnostic utilities for creating binary items
where participants make yes/no or true/false judgments about a single stimulus.

IMPORTANT: Binary tasks are semantically distinct from 2AFC tasks:
- Binary: Absolute judgment about single stimulus ("Is this grammatical?")
- 2AFC: Relative choice between two stimuli ("Which is more natural?")

Integration Points
------------------
- Active Learning: bead/active_learning/models/binary.py
- Simulation: bead/simulation/strategies/binary.py
- Deployment: bead/deployment/jspsych/ (binary button plugin)
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Hashable
from itertools import product
from uuid import UUID, uuid4

from bead.items.item import Item, MetadataValue


def create_binary_item(
    text: str,
    prompt: str = "Yes/No?",
    binary_options: tuple[str, str] = ("yes", "no"),
    item_template_id: UUID | None = None,
    metadata: dict[str, MetadataValue] | None = None,
) -> Item:
    """Create a binary judgment item for a single stimulus.

    Parameters
    ----------
    text : str
        The stimulus text to judge.
    prompt : str
        The question/prompt for the judgment (default: "Yes/No?").
    binary_options : tuple[str, str]
        The two response options (default: ("yes", "no")).
        Can also be ("true", "false"), ("acceptable", "unacceptable"), etc.
    item_template_id : UUID | None
        Template ID for the item. If None, generates new UUID.
    metadata : dict[str, MetadataValue] | None
        Additional metadata for item_metadata field.

    Returns
    -------
    Item
        Binary item with text and prompt in rendered_elements.

    Raises
    ------
    ValueError
        If text is empty or if binary_options doesn't have exactly 2 values.

    Examples
    --------
    >>> item = create_binary_item(
    ...     "The cat sat on the mat.",
    ...     prompt="Is this sentence grammatical?",
    ...     metadata={"judgment": "grammaticality"}
    ... )
    >>> item.rendered_elements["text"]
    'The cat sat on the mat.'
    >>> item.rendered_elements["prompt"]
    'Is this sentence grammatical?'
    >>> item.item_metadata["binary_options"]
    ['yes', 'no']

    >>> # Truth value judgment
    >>> item = create_binary_item(
    ...     "The sky is blue.",
    ...     prompt="Is this statement true?",
    ...     binary_options=("true", "false")
    ... )
    >>> item.item_metadata["binary_options"]
    ['true', 'false']
    """
    if not text or not text.strip():
        raise ValueError("text cannot be empty")

    if len(binary_options) != 2:
        raise ValueError("binary_options must contain exactly 2 values")

    if item_template_id is None:
        item_template_id = uuid4()

    rendered_elements: dict[str, str] = {
        "text": text,
        "prompt": prompt,
    }

    # Build item metadata
    item_metadata: dict[str, MetadataValue] = {
        "binary_options": tuple(binary_options),
    }
    if metadata:
        item_metadata.update(metadata)

    return Item(
        item_template_id=item_template_id,
        rendered_elements=rendered_elements,
        item_metadata=item_metadata,
    )


def create_binary_items_from_texts(
    texts: list[str],
    prompt: str,
    binary_options: tuple[str, str] = ("yes", "no"),
    *,
    item_template_id: UUID | None = None,
    metadata_fn: Callable[[str], dict[str, MetadataValue]] | None = None,
) -> list[Item]:
    """Create binary items from a list of texts with the same prompt.

    Parameters
    ----------
    texts : list[str]
        List of stimulus texts.
    prompt : str
        The question/prompt for all items.
    binary_options : tuple[str, str]
        The two response options (default: ("yes", "no")).
    item_template_id : UUID | None
        Template ID for all created items. If None, generates one per item.
    metadata_fn : Callable[[str], dict[str, MetadataValue]] | None
        Function to generate metadata from each text.

    Returns
    -------
    list[Item]
        Binary items for each text.

    Examples
    --------
    >>> texts = [
    ...     "She walks.",
    ...     "She walk.",
    ...     "They walk.",
    ...     "They walks."
    ... ]
    >>> items = create_binary_items_from_texts(
    ...     texts,
    ...     prompt="Is this sentence grammatical?",
    ...     binary_options=("yes", "no")
    ... )
    >>> len(items)
    4
    >>> items[0].rendered_elements["text"]
    'She walks.'
    """
    binary_items: list[Item] = []

    for text in texts:
        metadata: dict[str, MetadataValue] = {}
        if metadata_fn:
            metadata = metadata_fn(text)

        item = create_binary_item(
            text=text,
            prompt=prompt,
            binary_options=binary_options,
            item_template_id=item_template_id,
            metadata=metadata,
        )
        binary_items.append(item)

    return binary_items


def create_binary_items_with_context(
    contexts: list[str],
    targets: list[str],
    prompt: str,
    binary_options: tuple[str, str] = ("yes", "no"),
    *,
    context_label: str = "Context",
    target_label: str = "Statement",
    item_template_id: UUID | None = None,
    metadata_fn: (Callable[[str, str], dict[str, MetadataValue]] | None) = None,
) -> list[Item]:
    """Create binary items with context + target structure.

    Useful for judgments like "Given context X, is statement Y true?".

    Parameters
    ----------
    contexts : list[str]
        Context texts (same length as targets).
    targets : list[str]
        Target texts to judge given context.
    prompt : str
        The question/prompt for the judgment.
    binary_options : tuple[str, str]
        The two response options (default: ("yes", "no")).
    context_label : str
        Label for context in rendered text (default: "Context").
    target_label : str
        Label for target in rendered text (default: "Statement").
    item_template_id : UUID | None
        Template ID for all created items. If None, generates one per item.
    metadata_fn : Callable[[str, str], dict[str, MetadataValue]] | None
        Function to generate metadata from (context, target).

    Returns
    -------
    list[Item]
        Binary items with context + target structure.

    Raises
    ------
    ValueError
        If contexts and targets have different lengths.

    Examples
    --------
    >>> contexts = ["The dog barked loudly."]
    >>> targets = ["The dog made a sound."]
    >>> items = create_binary_items_with_context(
    ...     contexts,
    ...     targets,
    ...     prompt="Is the statement true given the context?",
    ...     binary_options=("true", "false")
    ... )
    >>> len(items)
    1
    >>> "Context:" in items[0].rendered_elements["text"]
    True
    """
    if len(contexts) != len(targets):
        raise ValueError("contexts and targets must have same length")

    binary_items: list[Item] = []

    for context, target in zip(contexts, targets, strict=True):
        # Combine context and target into single text
        combined_text = f"{context_label}: {context}\n{target_label}: {target}"

        metadata: dict[str, MetadataValue] = {
            "context": context,
            "target": target,
        }
        if metadata_fn:
            metadata.update(metadata_fn(context, target))

        item = create_binary_item(
            text=combined_text,
            prompt=prompt,
            binary_options=binary_options,
            item_template_id=item_template_id,
            metadata=metadata,
        )
        binary_items.append(item)

    return binary_items


def create_binary_items_from_groups(
    items: list[Item],
    group_by: Callable[[Item], Hashable],
    prompt: str,
    binary_options: tuple[str, str] = ("yes", "no"),
    *,
    extract_text: Callable[[Item], str] | None = None,
    include_group_metadata: bool = True,
    item_template_id: UUID | None = None,
) -> list[Item]:
    """Create binary items from grouped source items.

    Groups items and creates one binary item per source item, preserving
    group information in metadata.

    Parameters
    ----------
    items : list[Item]
        Source items to process.
    group_by : Callable[[Item], Hashable]
        Function to extract grouping key from items.
    prompt : str
        The question/prompt for all items.
    binary_options : tuple[str, str]
        The two response options (default: ("yes", "no")).
    extract_text : Callable[[Item], str] | None
        Function to extract text from item. If None, tries common keys.
    include_group_metadata : bool
        Whether to include group key in item metadata.
    item_template_id : UUID | None
        Template ID for all created items. If None, generates one per item.

    Returns
    -------
    list[Item]
        Binary items from source items.

    Examples
    --------
    >>> source_items = [
    ...     Item(
    ...         uuid4(),
    ...         rendered_elements={"text": "She walks."},
    ...         item_metadata={"verb": "walk"}
    ...     ),
    ...     Item(
    ...         uuid4(),
    ...         rendered_elements={"text": "She runs."},
    ...         item_metadata={"verb": "run"}
    ...     )
    ... ]
    >>> binary_items = create_binary_items_from_groups(
    ...     source_items,
    ...     group_by=lambda i: i.item_metadata["verb"],
    ...     prompt="Is this sentence grammatical?"
    ... )
    >>> len(binary_items)
    2
    """
    # Group items
    groups: dict[Hashable, list[Item]] = defaultdict(list)
    for item in items:
        group_key = group_by(item)
        groups[group_key].append(item)

    binary_items: list[Item] = []

    for group_key, group_items in groups.items():
        for item in group_items:
            # Extract text
            if extract_text:
                text: str = extract_text(item)
            else:
                text = _extract_text_from_item(item)

            # Build metadata
            metadata: dict[str, MetadataValue] = {
                "source_item_id": str(item.id),
            }
            if include_group_metadata:
                metadata["group_key"] = str(group_key)

            # Create binary item
            binary_item = create_binary_item(
                text=text,
                prompt=prompt,
                binary_options=binary_options,
                item_template_id=item_template_id,
                metadata=metadata,
            )
            binary_items.append(binary_item)

    return binary_items


def create_binary_items_cross_product(
    texts: list[str],
    prompts: list[str],
    binary_options: tuple[str, str] = ("yes", "no"),
    *,
    item_template_id: UUID | None = None,
    metadata_fn: (Callable[[str, str], dict[str, MetadataValue]] | None) = None,
) -> list[Item]:
    """Create binary items from cross-product of texts and prompts.

    Useful when you want to apply multiple prompts to each text.

    Parameters
    ----------
    texts : list[str]
        List of stimulus texts.
    prompts : list[str]
        List of prompts to apply.
    binary_options : tuple[str, str]
        The two response options (default: ("yes", "no")).
    item_template_id : UUID | None
        Template ID for all created items.
    metadata_fn : Callable[[str, str], dict[str, MetadataValue]] | None
        Function to generate metadata from (text, prompt).

    Returns
    -------
    list[Item]
        Binary items from cross-product.

    Examples
    --------
    >>> texts = ["The cat sat.", "The dog ran."]
    >>> prompts = ["Is this grammatical?", "Is this natural?"]
    >>> items = create_binary_items_cross_product(texts, prompts)
    >>> len(items)
    4
    """
    binary_items: list[Item] = []

    for text, prompt in product(texts, prompts):
        metadata: dict[str, MetadataValue] = {}
        if metadata_fn:
            metadata = metadata_fn(text, prompt)

        item = create_binary_item(
            text=text,
            prompt=prompt,
            binary_options=binary_options,
            item_template_id=item_template_id,
            metadata=metadata,
        )
        binary_items.append(item)

    return binary_items


def create_filtered_binary_items(
    items: list[Item],
    prompt: str,
    binary_options: tuple[str, str] = ("yes", "no"),
    *,
    item_filter: Callable[[Item], bool] | None = None,
    extract_text: Callable[[Item], str] | None = None,
    item_template_id: UUID | None = None,
) -> list[Item]:
    """Create binary items with filtering.

    Parameters
    ----------
    items : list[Item]
        Source items.
    prompt : str
        The question/prompt for all items.
    binary_options : tuple[str, str]
        The two response options (default: ("yes", "no")).
    item_filter : Callable[[Item], bool] | None
        Filter individual items.
    extract_text : Callable[[Item], str] | None
        Text extraction function.
    item_template_id : UUID | None
        Template ID for created items.

    Returns
    -------
    list[Item]
        Filtered binary items.

    Examples
    --------
    >>> binary_items = create_filtered_binary_items(
    ...     items,
    ...     prompt="Is this grammatical?",
    ...     item_filter=lambda i: i.item_metadata.get("valid", True)
    ... )  # doctest: +SKIP
    """
    # Filter items
    filtered_items = items
    if item_filter:
        filtered_items = [item for item in items if item_filter(item)]

    binary_items: list[Item] = []

    for item in filtered_items:
        # Extract text
        if extract_text:
            text: str = extract_text(item)
        else:
            text = _extract_text_from_item(item)

        # Create binary item
        metadata: dict[str, MetadataValue] = {
            "source_item_id": str(item.id),
        }

        binary_item = create_binary_item(
            text=text,
            prompt=prompt,
            binary_options=binary_options,
            item_template_id=item_template_id,
            metadata=metadata,
        )
        binary_items.append(binary_item)

    return binary_items


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
