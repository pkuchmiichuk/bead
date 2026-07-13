"""Persist gallery artifacts through the ``layers`` schema and ``lairs`` codec.

Every item artifact in this gallery is written as a ``layers`` fragment (the
same document ``bead layers encode`` produces) and reloaded by decoding it back
through the ``bead`` lairs codec, so the canonical on-disk form is the layers
representation rather than an ad hoc JSON dump. A materialized Arrow/Parquet
corpus is emitted alongside for downstream consumers and model training.

The codec round-trip is law-verified (``encode(decode(x)) == x``), so item
identity and all construction metadata survive the conversion.
"""

from __future__ import annotations

import json
from pathlib import Path
from uuid import UUID, uuid4

from lairs.integrations.codecs import CorpusFragment
from lairs.records import resource

from bead.interop.layers import (
    EXPERIMENT_LIST_LAYERS,
    BeadCodec,
    ExperimentListLayers,
    items_to_corpus,
    materialize_corpus,
)
from bead.interop.layers.resource_lens import (
    FILLED_TEMPLATE_FILLING,
    LEXICON_COLLECTION,
    TEMPLATE_LAYERS,
    LexiconLayers,
)
from bead.items.item import Item, ItemCollection
from bead.lists.experiment_list import ExperimentList
from bead.resources.lexicon import Lexicon
from bead.resources.template import Template
from bead.templates.filler import FilledTemplate


def items_to_collection(
    items: list[Item],
    *,
    name: str,
    source_template_collection_id: UUID | None = None,
    source_filled_collection_id: UUID | None = None,
) -> ItemCollection:
    """Wrap items in an ``ItemCollection`` ready for layers conversion.

    Parameters
    ----------
    items : list[Item]
        Items to wrap.
    name : str
        Collection name.
    source_template_collection_id : UUID | None
        Provenance reference; a fresh id is used when omitted.
    source_filled_collection_id : UUID | None
        Provenance reference; a fresh id is used when omitted.

    Returns
    -------
    ItemCollection
        The wrapped collection.
    """
    return ItemCollection(
        name=name,
        source_template_collection_id=source_template_collection_id or uuid4(),
        source_filled_collection_id=source_filled_collection_id or uuid4(),
        items=tuple(items),
    )


def write_fragment(collection: ItemCollection, fragment_path: Path) -> None:
    """Write a collection as a ``layers`` fragment JSON document.

    Mirrors ``bead layers encode``: the bead ``ItemCollection`` is decoded into
    a ``lairs`` ``CorpusFragment`` and serialized.

    Parameters
    ----------
    collection : ItemCollection
        Collection to encode.
    fragment_path : Path
        Output path for the fragment JSON.
    """
    fragment = BeadCodec().decode(collection.model_dump_json())
    fragment_path.parent.mkdir(parents=True, exist_ok=True)
    fragment_path.write_text(fragment.model_dump_json(), encoding="utf-8")


def read_items(fragment_path: Path) -> list[Item]:
    """Load items from a ``layers`` fragment JSON document.

    Mirrors ``bead layers decode``: the fragment is decoded back into an
    ``ItemCollection`` through the ``bead`` codec.

    Parameters
    ----------
    fragment_path : Path
        Path to a fragment written by :func:`write_fragment`.

    Returns
    -------
    list[Item]
        The reconstructed items.
    """
    fragment = CorpusFragment.model_validate_json(
        fragment_path.read_text(encoding="utf-8")
    )
    collection = ItemCollection.model_validate_json(
        BeadCodec().encode(fragment.records)
    )
    return list(collection.items)


def materialize(collection: ItemCollection, out_dir: Path, *, name: str) -> list[Path]:
    """Materialize a collection as an Arrow/Parquet ``layers`` corpus.

    Parameters
    ----------
    collection : ItemCollection
        Collection to materialize.
    out_dir : Path
        Output directory for the Parquet views.
    name : str
        Corpus name.

    Returns
    -------
    list[Path]
        The written view files.
    """
    corpus = items_to_corpus(collection, corpus_name=name)
    out_dir.mkdir(parents=True, exist_ok=True)
    return materialize_corpus(corpus, out_dir)


def write_items(
    items: list[Item],
    *,
    name: str,
    fragment_path: Path,
    materialize_dir: Path | None = None,
) -> ItemCollection:
    """Write items as a layers fragment and an optional materialized corpus.

    Parameters
    ----------
    items : list[Item]
        Items to persist.
    name : str
        Collection / corpus name.
    fragment_path : Path
        Output path for the layers fragment.
    materialize_dir : Path | None
        When set, also materialize an Arrow/Parquet corpus into this directory.

    Returns
    -------
    ItemCollection
        The collection that was written.
    """
    collection = items_to_collection(items, name=name)
    write_fragment(collection, fragment_path)
    if materialize_dir is not None:
        materialize(collection, materialize_dir, name=name)
    return collection


def write_lexicon_layers(lexicon: Lexicon, path: Path) -> None:
    """Write a lexicon as a ``layers`` resource collection.

    Stores the layers view (a ``LexiconLayers`` of ``entry`` records) together
    with the bead-only complement so the lexicon round-trips.

    Parameters
    ----------
    lexicon : Lexicon
        Lexicon to encode.
    path : Path
        Output path for the layers JSON.
    """
    view, complement = LEXICON_COLLECTION.forward(lexicon)
    payload = {"view": view.model_dump_json(), "complement": complement}
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def read_lexicon_layers(path: Path) -> Lexicon:
    """Load a lexicon from a ``layers`` resource collection.

    Parameters
    ----------
    path : Path
        Path written by :func:`write_lexicon_layers`.

    Returns
    -------
    Lexicon
        The reconstructed lexicon.
    """
    payload = json.loads(path.read_text(encoding="utf-8"))
    view = LexiconLayers.model_validate_json(payload["view"])
    return LEXICON_COLLECTION.backward(view, payload["complement"])


def save_lexicon(lexicon: Lexicon, jsonl_path: Path) -> None:
    """Save a lexicon as bead-native JSONL and a ``layers`` resource collection.

    Writes ``jsonl_path`` plus a sibling ``<stem>.layers.json``.

    Parameters
    ----------
    lexicon : Lexicon
        Lexicon to save.
    jsonl_path : Path
        Output path for the bead-native JSONL.
    """
    jsonl_path.parent.mkdir(parents=True, exist_ok=True)
    lexicon.to_jsonl(str(jsonl_path))
    write_lexicon_layers(lexicon, jsonl_path.with_suffix(".layers.json"))


def write_templates_layers(templates: list[Template], path: Path) -> None:
    """Write templates as ``layers`` resource templates.

    Each template is mapped through the ``TEMPLATE_LAYERS`` lens to a layers
    ``template`` record; the bead-only complement travels alongside so the
    template round-trips.

    Parameters
    ----------
    templates : list[Template]
        Templates to encode.
    path : Path
        Output path for the layers JSON.
    """
    records: list[dict[str, object]] = []
    for template in templates:
        view, complement = TEMPLATE_LAYERS.forward(template)
        records.append({"view": view.model_dump_json(), "complement": complement})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records), encoding="utf-8")


def read_templates_layers(path: Path) -> list[Template]:
    """Load templates from a ``layers`` resource-template document.

    Parameters
    ----------
    path : Path
        Path written by :func:`write_templates_layers`.

    Returns
    -------
    list[Template]
        The reconstructed templates.
    """
    records = json.loads(path.read_text(encoding="utf-8"))
    templates: list[Template] = []
    for record in records:
        view = resource.Template.model_validate_json(record["view"])
        templates.append(TEMPLATE_LAYERS.backward(view, record["complement"]))
    return templates


def write_fillings_layers(filled: list[FilledTemplate], path: Path) -> None:
    """Write filled templates as ``layers`` filling records.

    Each filled template is mapped through ``FILLED_TEMPLATE_FILLING`` to a
    layers ``filling`` record; the bead-only complement travels alongside so the
    filled template round-trips.

    Parameters
    ----------
    filled : list[FilledTemplate]
        Filled templates to encode.
    path : Path
        Output path for the layers JSON.
    """
    records: list[dict[str, object]] = []
    for item in filled:
        view, complement = FILLED_TEMPLATE_FILLING.forward(item)
        records.append({"view": view.model_dump_json(), "complement": complement})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records), encoding="utf-8")


def read_fillings_layers(path: Path) -> list[FilledTemplate]:
    """Load filled templates from a ``layers`` filling document.

    Parameters
    ----------
    path : Path
        Path written by :func:`write_fillings_layers`.

    Returns
    -------
    list[FilledTemplate]
        The reconstructed filled templates.
    """
    records = json.loads(path.read_text(encoding="utf-8"))
    filled: list[FilledTemplate] = []
    for record in records:
        view = resource.Filling.model_validate_json(record["view"])
        filled.append(FILLED_TEMPLATE_FILLING.backward(view, record["complement"]))
    return filled


def write_experiment_lists_layers(lists: list[ExperimentList], path: Path) -> None:
    """Write experiment lists as ``layers`` collection aggregates.

    Each list is mapped through ``EXPERIMENT_LIST_LAYERS`` to a layers
    ``collection`` with one membership per item and its list constraints; the
    bead-only complement travels alongside so the list round-trips.

    Parameters
    ----------
    lists : list[ExperimentList]
        Experiment lists to encode.
    path : Path
        Output path for the layers JSON.
    """
    records: list[dict[str, object]] = []
    for experiment_list in lists:
        view, complement = EXPERIMENT_LIST_LAYERS.forward(experiment_list)
        records.append({"view": view.model_dump_json(), "complement": complement})
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(records), encoding="utf-8")


def read_experiment_lists_layers(path: Path) -> list[ExperimentList]:
    """Load experiment lists from a ``layers`` collection-aggregate document.

    Parameters
    ----------
    path : Path
        Path written by :func:`write_experiment_lists_layers`.

    Returns
    -------
    list[ExperimentList]
        The reconstructed experiment lists.
    """
    records = json.loads(path.read_text(encoding="utf-8"))
    lists: list[ExperimentList] = []
    for record in records:
        view = ExperimentListLayers.model_validate_json(record["view"])
        lists.append(EXPERIMENT_LIST_LAYERS.backward(view, record["complement"]))
    return lists
