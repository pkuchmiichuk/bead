"""Utilities for creating categorical experimental items.

This module provides language-agnostic utilities for creating categorical
items where participants select from N unordered categories (e.g., NLI labels,
POS tags, semantic relations).

Integration Points
------------------
- Active Learning: bead/active_learning/models/categorical.py
- Simulation: bead/simulation/strategies/categorical.py
- Deployment: bead/deployment/jspsych/ (dropdown or radio buttons)
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Hashable
from itertools import product
from uuid import UUID, uuid4

from bead.items.item import Item, MetadataValue


def create_categorical_item(
    text: str,
    categories: list[str],
    prompt: str | None = None,
    item_template_id: UUID | None = None,
    metadata: dict[str, MetadataValue] | None = None,
) -> Item:
    """Create a categorical classification item.

    Parameters
    ----------
    text : str
        The stimulus text to classify.
    categories : list[str]
        List of category labels (unordered). Must have at least 2 categories.
    prompt : str | None
        Optional question/prompt for the classification.
        If None, uses "Select a category:".
    item_template_id : UUID | None
        Template ID for the item. If None, generates new UUID.
    metadata : dict[str, MetadataValue] | None
        Additional metadata for item_metadata field.

    Returns
    -------
    Item
        Categorical item with text and prompt in rendered_elements.

    Raises
    ------
    ValueError
        If text is empty or if fewer than 2 categories provided.

    Examples
    --------
    >>> item = create_categorical_item(
    ...     text="Premise: All dogs bark. Hypothesis: Some dogs bark.",
    ...     categories=["entailment", "neutral", "contradiction"],
    ...     prompt="What is the relationship?",
    ...     metadata={"task": "nli"}
    ... )
    >>> item.rendered_elements["text"]
    'Premise: All dogs bark. Hypothesis: Some dogs bark.'
    >>> item.rendered_elements["prompt"]
    'What is the relationship?'
    >>> item.item_metadata["categories"]
    ['entailment', 'neutral', 'contradiction']

    >>> # POS tagging
    >>> item = create_categorical_item(
    ...     text="The cat sat on the mat.",
    ...     categories=["noun", "verb", "adjective", "determiner", "preposition"],
    ...     prompt="What is the part of speech of 'cat'?"
    ... )
    >>> len(item.item_metadata["categories"])
    5
    """
    if not text or not text.strip():
        raise ValueError("text cannot be empty")

    if len(categories) < 2:
        raise ValueError("At least 2 categories required for categorical item")

    if item_template_id is None:
        item_template_id = uuid4()

    if prompt is None:
        prompt = "Select a category:"

    rendered_elements: dict[str, str] = {
        "text": text,
        "prompt": prompt,
    }

    # Build item metadata
    item_metadata: dict[str, MetadataValue] = {
        "categories": tuple(categories),
    }
    if metadata:
        item_metadata.update(metadata)

    return Item(
        item_template_id=item_template_id,
        rendered_elements=rendered_elements,
        item_metadata=item_metadata,
    )


def create_nli_item(
    premise: str,
    hypothesis: str,
    categories: list[str] | None = None,
    prompt: str | None = None,
    item_template_id: UUID | None = None,
    metadata: dict[str, MetadataValue] | None = None,
) -> Item:
    """Create a Natural Language Inference (NLI) item.

    Specialized helper for NLI tasks with automatic formatting and default
    categories.

    Parameters
    ----------
    premise : str
        The premise text.
    hypothesis : str
        The hypothesis text.
    categories : list[str] | None
        Category labels. If None, uses ["entailment", "neutral", "contradiction"].
    prompt : str | None
        Question/prompt. If None, uses "What is the relationship?".
    item_template_id : UUID | None
        Template ID for the item. If None, generates new UUID.
    metadata : dict[str, MetadataValue] | None
        Additional metadata for item_metadata field.

    Returns
    -------
    Item
        NLI categorical item.

    Examples
    --------
    >>> item = create_nli_item(
    ...     premise="All dogs bark.",
    ...     hypothesis="Some dogs bark."
    ... )
    >>> "Premise:" in item.rendered_elements["text"]
    True
    >>> "Hypothesis:" in item.rendered_elements["text"]
    True
    >>> item.item_metadata["categories"]
    ['entailment', 'neutral', 'contradiction']
    >>> item.item_metadata["premise"]
    'All dogs bark.'

    >>> # Custom categories
    >>> item = create_nli_item(
    ...     premise="The cat is on the mat.",
    ...     hypothesis="There is an animal on the mat.",
    ...     categories=["entails", "contradicts", "neither"]
    ... )
    >>> item.item_metadata["categories"]
    ['entails', 'contradicts', 'neither']
    """
    if categories is None:
        categories = ["entailment", "neutral", "contradiction"]

    if prompt is None:
        prompt = "What is the relationship?"

    # Format as premise-hypothesis pair
    combined_text = f"Premise: {premise}\nHypothesis: {hypothesis}"

    # Build metadata with premise and hypothesis
    nli_metadata: dict[str, MetadataValue] = {
        "premise": premise,
        "hypothesis": hypothesis,
        "task": "nli",
    }
    if metadata:
        nli_metadata.update(metadata)

    return create_categorical_item(
        text=combined_text,
        categories=categories,
        prompt=prompt,
        item_template_id=item_template_id,
        metadata=nli_metadata,
    )


def create_categorical_items_from_texts(
    texts: list[str],
    categories: list[str],
    prompt: str | None = None,
    *,
    item_template_id: UUID | None = None,
    metadata_fn: Callable[[str], dict[str, MetadataValue]] | None = None,
) -> list[Item]:
    """Create categorical items from a list of texts with the same categories.

    Parameters
    ----------
    texts : list[str]
        List of stimulus texts.
    categories : list[str]
        Category labels for all items.
    prompt : str | None
        The question/prompt for all items.
    item_template_id : UUID | None
        Template ID for all created items. If None, generates one per item.
    metadata_fn : Callable[[str], dict[str, MetadataValue]] | None
        Function to generate metadata from each text.

    Returns
    -------
    list[Item]
        Categorical items for each text.

    Examples
    --------
    >>> texts = ["The cat sat.", "The dog ran.", "The bird flew."]
    >>> categories = ["past", "present", "future"]
    >>> items = create_categorical_items_from_texts(
    ...     texts,
    ...     categories=categories,
    ...     prompt="What is the tense?"
    ... )
    >>> len(items)
    3
    >>> items[0].item_metadata["categories"]
    ['past', 'present', 'future']
    """
    categorical_items: list[Item] = []

    for text in texts:
        metadata: dict[str, MetadataValue] = {}
        if metadata_fn:
            metadata = metadata_fn(text)

        item = create_categorical_item(
            text=text,
            categories=categories,
            prompt=prompt,
            item_template_id=item_template_id,
            metadata=metadata,
        )
        categorical_items.append(item)

    return categorical_items


def create_categorical_items_from_pairs(
    pairs: list[tuple[str, str]],
    categories: list[str],
    prompt: str | None = None,
    *,
    pair_label1: str = "Text 1",
    pair_label2: str = "Text 2",
    item_template_id: UUID | None = None,
    metadata_fn: (Callable[[str, str], dict[str, MetadataValue]] | None) = None,
) -> list[Item]:
    """Create categorical items from pairs of texts.

    Useful for NLI, paraphrase detection, semantic similarity, etc.

    Parameters
    ----------
    pairs : list[tuple[str, str]]
        List of (text1, text2) pairs.
    categories : list[str]
        Category labels for all items.
    prompt : str | None
        The question/prompt for all items.
    pair_label1 : str
        Label for first text in pair (default: "Text 1").
    pair_label2 : str
        Label for second text in pair (default: "Text 2").
    item_template_id : UUID | None
        Template ID for all created items. If None, generates one per item.
    metadata_fn : Callable[[str, str], dict[str, MetadataValue]] | None
        Function to generate metadata from (text1, text2).

    Returns
    -------
    list[Item]
        Categorical items from pairs.

    Examples
    --------
    >>> pairs = [
    ...     ("All dogs bark.", "Some dogs bark."),
    ...     ("The sky is blue.", "The sky is not blue.")
    ... ]
    >>> items = create_categorical_items_from_pairs(
    ...     pairs,
    ...     categories=["entailment", "neutral", "contradiction"],
    ...     prompt="What is the relationship?",
    ...     pair_label1="Premise",
    ...     pair_label2="Hypothesis"
    ... )
    >>> len(items)
    2
    >>> "Premise:" in items[0].rendered_elements["text"]
    True
    """
    categorical_items: list[Item] = []

    for text1, text2 in pairs:
        # Combine pairs into single text
        combined_text = f"{pair_label1}: {text1}\n{pair_label2}: {text2}"

        metadata: dict[str, MetadataValue] = {
            "text1": text1,
            "text2": text2,
        }
        if metadata_fn:
            metadata.update(metadata_fn(text1, text2))

        item = create_categorical_item(
            text=combined_text,
            categories=categories,
            prompt=prompt,
            item_template_id=item_template_id,
            metadata=metadata,
        )
        categorical_items.append(item)

    return categorical_items


def create_categorical_items_from_groups(
    items: list[Item],
    group_by: Callable[[Item], Hashable],
    categories: list[str],
    prompt: str | None = None,
    *,
    extract_text: Callable[[Item], str] | None = None,
    include_group_metadata: bool = True,
    item_template_id: UUID | None = None,
) -> list[Item]:
    """Create categorical items from grouped source items.

    Groups items and creates one categorical item per source item, preserving
    group information in metadata.

    Parameters
    ----------
    items : list[Item]
        Source items to process.
    group_by : Callable[[Item], Hashable]
        Function to extract grouping key from items.
    categories : list[str]
        Category labels for all items.
    prompt : str | None
        The question/prompt for all items.
    extract_text : Callable[[Item], str] | None
        Function to extract text from item. If None, tries common keys.
    include_group_metadata : bool
        Whether to include group key in item metadata.
    item_template_id : UUID | None
        Template ID for all created items. If None, generates one per item.

    Returns
    -------
    list[Item]
        Categorical items from source items.

    Examples
    --------
    >>> source_items = [
    ...     Item(
    ...         uuid4(),
    ...         rendered_elements={"text": "The cat sat."},
    ...         item_metadata={"tense": "past"}
    ...     ),
    ...     Item(
    ...         uuid4(),
    ...         rendered_elements={"text": "The dog runs."},
    ...         item_metadata={"tense": "present"}
    ...     )
    ... ]
    >>> categorical_items = create_categorical_items_from_groups(
    ...     source_items,
    ...     group_by=lambda i: i.item_metadata["tense"],
    ...     categories=["past", "present", "future"],
    ...     prompt="What is the tense?"
    ... )
    >>> len(categorical_items)
    2
    """
    # Group items
    groups: dict[Hashable, list[Item]] = defaultdict(list)
    for item in items:
        group_key = group_by(item)
        groups[group_key].append(item)

    categorical_items: list[Item] = []

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

            # Create categorical item
            categorical_item = create_categorical_item(
                text=text,
                categories=categories,
                prompt=prompt,
                item_template_id=item_template_id,
                metadata=metadata,
            )
            categorical_items.append(categorical_item)

    return categorical_items


def create_categorical_items_cross_product(
    texts: list[str],
    prompts: list[str],
    categories: list[str],
    *,
    item_template_id: UUID | None = None,
    metadata_fn: (Callable[[str, str], dict[str, MetadataValue]] | None) = None,
) -> list[Item]:
    """Create categorical items from cross-product of texts and prompts.

    Useful when you want to apply multiple prompts to each text.

    Parameters
    ----------
    texts : list[str]
        List of stimulus texts.
    prompts : list[str]
        List of prompts to apply.
    categories : list[str]
        Category labels for all items.
    item_template_id : UUID | None
        Template ID for all created items.
    metadata_fn : Callable[[str, str], dict[str, MetadataValue]] | None
        Function to generate metadata from (text, prompt).

    Returns
    -------
    list[Item]
        Categorical items from cross-product.

    Examples
    --------
    >>> texts = ["The cat sat.", "The dog ran."]
    >>> prompts = ["What is the tense?", "What is the aspect?"]
    >>> categories = ["past", "present", "future"]
    >>> items = create_categorical_items_cross_product(
    ...     texts, prompts, categories
    ... )
    >>> len(items)
    4
    """
    categorical_items: list[Item] = []

    for text, prompt in product(texts, prompts):
        metadata: dict[str, MetadataValue] = {}
        if metadata_fn:
            metadata = metadata_fn(text, prompt)

        item = create_categorical_item(
            text=text,
            categories=categories,
            prompt=prompt,
            item_template_id=item_template_id,
            metadata=metadata,
        )
        categorical_items.append(item)

    return categorical_items


def create_filtered_categorical_items(
    items: list[Item],
    categories: list[str],
    prompt: str | None = None,
    *,
    item_filter: Callable[[Item], bool] | None = None,
    extract_text: Callable[[Item], str] | None = None,
    item_template_id: UUID | None = None,
) -> list[Item]:
    """Create categorical items with filtering.

    Parameters
    ----------
    items : list[Item]
        Source items.
    categories : list[str]
        Category labels for all items.
    prompt : str | None
        The question/prompt for all items.
    item_filter : Callable[[Item], bool] | None
        Filter individual items.
    extract_text : Callable[[Item], str] | None
        Text extraction function.
    item_template_id : UUID | None
        Template ID for created items.

    Returns
    -------
    list[Item]
        Filtered categorical items.

    Examples
    --------
    >>> categorical_items = create_filtered_categorical_items(
    ...     items,
    ...     categories=["past", "present", "future"],
    ...     prompt="What is the tense?",
    ...     item_filter=lambda i: i.item_metadata.get("valid", True)
    ... )  # doctest: +SKIP
    """
    # Filter items
    filtered_items = items
    if item_filter:
        filtered_items = [item for item in items if item_filter(item)]

    categorical_items: list[Item] = []

    for item in filtered_items:
        # Extract text
        if extract_text:
            text: str = extract_text(item)
        else:
            text = _extract_text_from_item(item)

        # Create categorical item
        metadata: dict[str, MetadataValue] = {
            "source_item_id": str(item.id),
        }

        categorical_item = create_categorical_item(
            text=text,
            categories=categories,
            prompt=prompt,
            item_template_id=item_template_id,
            metadata=metadata,
        )
        categorical_items.append(categorical_item)

    return categorical_items


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
