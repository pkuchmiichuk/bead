"""Utilities for generating cross-product items from templates and lexicons.

This module provides language-agnostic utilities for generating items
by combining templates with lexical resources in various patterns.

RELATIONSHIP TO ItemConstructor:
- This module (generation.py): Generates cross-product combinations of
  templates × lexical items BEFORE template filling. Creates lightweight
  Item objects with just template_id, metadata, and unfilled information.
  Use when: You want to systematically explore all combinations of a lexical
  property (e.g., every verb in every frame).

- ItemConstructor (constructor.py): Builds Items FROM ItemTemplates +
  FilledTemplates with constraint evaluation and model scoring. Takes filled
  templates and combines them into experimental items with multi-slot
  constraints checked.
  Use when: You have filled templates and want to construct experimental
  items with model-based constraint checking.

These modules are COMPLEMENTARY, not redundant. Typical pipeline:
1. generation.py: Generate cross-product → unfilled item specifications
2. Template filling: Fill template slots → FilledTemplates
3. constructor.py: Construct items → Items with constraints checked
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Iterator
from pathlib import Path

from bead.items.item import Item, MetadataValue
from bead.resources.lexical_item import LexicalItem
from bead.resources.lexicon import Lexicon
from bead.resources.template import Template


def create_cross_product_items(
    templates: list[Template],
    lexicons: dict[str, Lexicon],
    *,
    cross_product_slot: str = "verb",
    metadata_extractor: (
        Callable[[Template, LexicalItem], dict[str, MetadataValue]] | None
    ) = None,
    filter_fn: Callable[[Template, LexicalItem], bool] | None = None,
) -> Iterator[Item]:
    """Generate cross-product items from templates and lexicons.

    Creates an item for each combination of template × lexical item from
    the specified slot's lexicon. This is useful for systematic exploration
    of a lexical property (e.g., every verb in every frame).

    Items are generated lazily via iterator for memory efficiency with
    large cross-products.

    Parameters
    ----------
    templates : list[Template]
        Templates to use for generation.
    lexicons : dict[str, Lexicon]
        Lexicons keyed by slot name.
    cross_product_slot : str
        Slot name to vary across items (default: "verb").
        This slot's lexicon will be crossed with all templates.
    metadata_extractor : Callable[[Template, LexicalItem], \
            dict[str, MetadataValue]] | None
        Optional function to extract metadata from template and lexical item.
        Receives (template, lexical_item) and returns dict for item_metadata.
    filter_fn : Callable[[Template, LexicalItem], bool] | None
        Optional filter function. Receives (template, lexical_item) and
        returns True to include, False to skip.

    Yields
    ------
    Item
        Items representing template × lexical item combinations.

    Examples
    --------
    Basic verb × template cross-product:
    >>> from uuid import uuid4
    >>> templates = [
    ...     Template(
    ...         name="transitive",
    ...         template_string="{subject} {verb} {object}.",
    ...         slots={}
    ...     )
    ... ]
    >>> verb_lex = Lexicon(name="verbs")
    >>> verb_lex.add(LexicalItem(lemma="walk"))
    >>> verb_lex.add(LexicalItem(lemma="eat"))
    >>> lexicons = {"verb": verb_lex}
    >>> items = list(create_cross_product_items(templates, lexicons))
    >>> len(items)
    2

    With metadata extraction:
    >>> def extract_metadata(template, item):
    ...     return {
    ...         "verb_lemma": item.lemma,
    ...         "template_name": template.name,
    ...         "verb_pos": item.pos
    ...     }
    >>> items = list(create_cross_product_items(
    ...     templates,
    ...     lexicons,
    ...     metadata_extractor=extract_metadata
    ... ))  # doctest: +SKIP

    With filtering:
    >>> def filter_transitive_only(template, item):
    ...     return "transitive" in template.name
    >>> items = list(create_cross_product_items(
    ...     templates,
    ...     lexicons,
    ...     filter_fn=filter_transitive_only
    ... ))  # doctest: +SKIP
    """
    # get the lexicon for the cross-product slot
    if cross_product_slot not in lexicons:
        raise ValueError(
            f"Lexicon for slot '{cross_product_slot}' not found. "
            f"Available: {list(lexicons.keys())}"
        )

    cross_product_lexicon = lexicons[cross_product_slot]

    # generate items
    for template in templates:
        for lexical_item in cross_product_lexicon:
            # apply filter if provided
            if filter_fn and not filter_fn(template, lexical_item):
                continue

            # extract metadata
            if metadata_extractor:
                item_metadata = metadata_extractor(template, lexical_item)
            else:
                item_metadata = _default_metadata_extractor(template, lexical_item)

            # create rendered elements
            rendered_elements = {
                "template_name": template.name,
                "template_string": template.template_string,
                f"{cross_product_slot}_lemma": lexical_item.lemma,
                f"{cross_product_slot}_form": lexical_item.form or lexical_item.lemma,
            }

            # create item
            item = Item(
                item_template_id=template.id,
                rendered_elements=rendered_elements,
                item_metadata=item_metadata,
            )

            yield item


def _default_metadata_extractor(
    template: Template, lexical_item: LexicalItem
) -> dict[str, MetadataValue]:
    """Extract default metadata for cross-product items.

    Parameters
    ----------
    template
        Template being used.
    lexical_item
        Lexical item being crossed.

    Returns
    -------
    dict[str, MetadataValue]
        Default metadata dictionary.
    """
    metadata: dict[str, MetadataValue] = {
        "template_id": str(template.id),
        "template_name": template.name,
        "template_structure": template.template_string,
        "lexical_item_id": str(lexical_item.id),
        "lexical_item_lemma": lexical_item.lemma,
        "combination_type": "cross_product",
    }

    if lexical_item.features:
        for key, value in lexical_item.features.items():
            metadata[f"lexical_feature_{key}"] = _coerce_to_metadata(value)
            metadata[f"lexical_attr_{key}"] = _coerce_to_metadata(value)

    return metadata


def _coerce_to_metadata(value: object) -> MetadataValue:
    """Coerce a JsonValue tree into a MetadataValue tree (lists become tuples)."""
    if isinstance(value, list):
        return tuple(_coerce_to_metadata(elem) for elem in value)
    if isinstance(value, dict):
        return {str(k): _coerce_to_metadata(v) for k, v in value.items()}
    if isinstance(value, str | int | float | bool | type(None)):
        return value
    raise TypeError(f"Unsupported metadata value type: {type(value).__name__}")


def create_filtered_cross_product_items(
    templates: list[Template],
    lexicons: dict[str, Lexicon],
    *,
    cross_product_slot: str = "verb",
    template_filter: Callable[[Template], bool] | None = None,
    item_filter: Callable[[LexicalItem], bool] | None = None,
    combination_filter: Callable[[Template, LexicalItem], bool] | None = None,
    metadata_extractor: (
        Callable[[Template, LexicalItem], dict[str, MetadataValue]] | None
    ) = None,
) -> Iterator[Item]:
    """Generate cross-product items with multiple filter levels.

    Provides separate filters for templates, lexical items, and their
    combinations, offering more control than the basic cross-product function.

    Parameters
    ----------
    templates : list[Template]
        Templates to use for generation.
    lexicons : dict[str, Lexicon]
        Lexicons keyed by slot name.
    cross_product_slot : str
        Slot name to vary across items.
    template_filter : Callable[[Template], bool] | None
        Filter for templates (applied before cross-product).
    item_filter : Callable[[LexicalItem], bool] | None
        Filter for lexical items (applied before cross-product).
    combination_filter : Callable[[Template, LexicalItem], bool] | None
        Filter for combinations (applied during generation).
    metadata_extractor : Callable[[Template, LexicalItem], \
            dict[str, MetadataValue]] | None
        Metadata extraction function.

    Yields
    ------
    Item
        Filtered cross-product items.

    Examples
    --------
    Filter at multiple levels:
    >>> def template_filter(t):
    ...     return "transitive" in t.name
    >>> def item_filter(i):
    ...     return i.pos == "VERB"
    >>> def combination_filter(t, i):
    ...     # Only combine if verb is compatible with template
    ...     return True
    >>> items = list(create_filtered_cross_product_items(
    ...     templates,
    ...     lexicons,
    ...     template_filter=template_filter,
    ...     item_filter=item_filter,
    ...     combination_filter=combination_filter
    ... ))  # doctest: +SKIP
    """
    # get lexicon
    if cross_product_slot not in lexicons:
        raise ValueError(
            f"Lexicon for slot '{cross_product_slot}' not found. "
            f"Available: {list(lexicons.keys())}"
        )

    cross_product_lexicon = lexicons[cross_product_slot]

    # filter templates
    filtered_templates = templates
    if template_filter:
        filtered_templates = [t for t in templates if template_filter(t)]

    # filter lexical items
    filtered_items = list(cross_product_lexicon)
    if item_filter:
        filtered_items = [item for item in filtered_items if item_filter(item)]

    # generate cross-product with combination filter
    yield from create_cross_product_items(
        filtered_templates,
        {cross_product_slot: _create_temp_lexicon(filtered_items)},
        cross_product_slot=cross_product_slot,
        metadata_extractor=metadata_extractor,
        filter_fn=combination_filter,
    )


def _create_temp_lexicon(items: list[LexicalItem]) -> Lexicon:
    """Create temporary lexicon from list of items.

    Parameters
    ----------
    items : list[LexicalItem]
        Lexical items to include.

    Returns
    -------
    Lexicon
        Temporary lexicon.
    """
    return Lexicon(name="temp", items=tuple(items))


def create_stratified_cross_product_items(
    templates: list[Template],
    lexicons: dict[str, Lexicon],
    *,
    cross_product_slot: str = "verb",
    stratify_by: Callable[[LexicalItem], str],
    items_per_stratum: int,
    metadata_extractor: (
        Callable[[Template, LexicalItem], dict[str, MetadataValue]] | None
    ) = None,
) -> Iterator[Item]:
    """Generate stratified sample of cross-product items.

    Instead of full cross-product, samples a fixed number of lexical items
    from each stratum (defined by stratify_by function) and crosses them
    with all templates.

    Parameters
    ----------
    templates : list[Template]
        Templates to use for generation.
    lexicons : dict[str, Lexicon]
        Lexicons keyed by slot name.
    cross_product_slot : str
        Slot name to vary across items.
    stratify_by : Callable[[LexicalItem], str]
        Function to extract stratum key from lexical items.
    items_per_stratum : int
        Number of items to sample from each stratum.
    metadata_extractor : Callable[[Template, LexicalItem], \
            dict[str, MetadataValue]] | None
        Metadata extraction function.

    Yields
    ------
    Item
        Stratified cross-product items.

    Examples
    --------
    Sample verbs stratified by frequency:
    >>> def stratify_by_frequency(item):
    ...     freq = item.attributes.get("frequency", 0)
    ...     if freq > 1000:
    ...         return "high"
    ...     elif freq > 100:
    ...         return "medium"
    ...     else:
    ...         return "low"
    >>> items = list(create_stratified_cross_product_items(
    ...     templates,
    ...     lexicons,
    ...     stratify_by=stratify_by_frequency,
    ...     items_per_stratum=10
    ... ))  # doctest: +SKIP
    """
    # get lexicon
    if cross_product_slot not in lexicons:
        raise ValueError(
            f"Lexicon for slot '{cross_product_slot}' not found. "
            f"Available: {list(lexicons.keys())}"
        )

    cross_product_lexicon = lexicons[cross_product_slot]

    # group items by stratum
    strata: dict[str, list[LexicalItem]] = defaultdict(list)
    for item in cross_product_lexicon:
        stratum = stratify_by(item)
        strata[stratum].append(item)

    # sample from each stratum
    sampled_items: list[LexicalItem] = []
    for _stratum, stratum_items in strata.items():
        # take first items_per_stratum items (or all if fewer available)
        n_to_take = min(items_per_stratum, len(stratum_items))
        sampled_items.extend(stratum_items[:n_to_take])

    # generate cross-product with sampled items
    for item in create_cross_product_items(
        templates,
        {cross_product_slot: _create_temp_lexicon(sampled_items)},
        cross_product_slot=cross_product_slot,
        metadata_extractor=metadata_extractor,
    ):
        yield item


def items_to_jsonl(
    items: Iterator[Item], output_path: str, progress_interval: int = 1000
) -> int:
    """Write iterator of items to JSONL file with progress tracking.

    Utility function for efficient streaming write of large item sets.

    Parameters
    ----------
    items : Iterator[Item]
        Items to write.
    output_path : str
        Path to output JSONL file.
    progress_interval : int
        Print progress every N items (default: 1000).

    Returns
    -------
    int
        Number of items written.

    Examples
    --------
    >>> items = create_cross_product_items(templates, lexicons)  # doctest: +SKIP
    >>> n = items_to_jsonl(items, "output.jsonl")  # doctest: +SKIP
    >>> print(f"Wrote {n} items")  # doctest: +SKIP
    """
    output_file = Path(output_path)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    count = 0
    with open(output_file, "w", encoding="utf-8") as f:
        for item in items:
            f.write(item.model_dump_json() + "\n")
            count += 1

            if count % progress_interval == 0:
                print(f"  Progress: {count:,} items written...")

    return count
