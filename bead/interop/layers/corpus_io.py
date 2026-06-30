"""Ingest a layers ``Corpus`` into bead models and emit bead data as a corpus.

Ingest (layers to bead):

- :func:`expression_to_record` / :func:`corpus_to_records` stream a corpus's
  expressions as :class:`~bead.corpus.records.CorpusRecord` instances.
- :func:`corpus_to_graph` builds a :class:`~bead.corpus.graph.CorpusGraph` from a
  corpus, deriving a ``parent`` edge from each expression's ``parentRef``.
- :func:`corpus_to_items` reconstructs best-effort :class:`~bead.items.item.Item`
  instances from a corpus's span and relation annotation layers.
- :func:`load_layers_corpus` is a thin wrapper over :func:`lairs.load_corpus`.

Egress (bead to layers):

- :func:`items_to_corpus` and :func:`graph_to_corpus` build a
  :class:`lairs.data.Corpus` from bead data, reusing the canonical lenses.
- :func:`materialize_corpus`, :func:`save_corpus_repo`, and :func:`publish_corpus`
  delegate to the corresponding ``lairs`` store and publish entry points; the
  network-bound publish path is opt-in and defaults to a dry run.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import didactic.api as dx
from lairs.data import Corpus, load_corpus
from lairs.data.dataset import Dataset
from lairs.records import annotation, expression, segmentation
from lairs.records import corpus as corpus_records

from bead.corpus.graph import CorpusEdge, CorpusGraph, CorpusNode
from bead.corpus.records import CorpusRecord
from bead.interop.layers._convert import (
    ANNOTATION_LAYER_NSID,
    CONFIDENCE_SCALE,
    CORPUS_NSID,
    EXPRESSION_NSID,
    MEMBERSHIP_NSID,
    SEGMENTATION_NSID,
    feature_map,
    read_feature_map_scalar,
)
from bead.interop.layers.item_bridge import ITEM_LAYERS
from bead.items.item import Item, ItemCollection
from bead.items.spans import Span, SpanLabel, SpanRelation, SpanSegment

if TYPE_CHECKING:
    from collections.abc import Iterator
    from pathlib import Path
    from uuid import UUID

    import httpx
    from lairs.atproto.pds import PdsClient
    from lairs.store import Repository

_PARENT_EDGE = "parent"
_DEFAULT_ELEMENT = "text"


def _rkey(at_uri: str) -> str:
    """Return the trailing identifier of an AT-URI (or the string itself)."""
    return at_uri.rsplit("/", 1)[-1]


def _each[T: dx.Model](dataset: Dataset[T]) -> Iterator[T]:
    """Yield a dataset's records one at a time, flattening its batches."""
    for batch in dataset.iter():
        yield from batch


def _source(source_name: str | None, expr: expression.Expression) -> str:
    """Resolve a per-expression source name, defaulting to the expression id."""
    return source_name if source_name is not None else (expr.id or "layers")


# --- ingest: layers -> bead -------------------------------------------------


def expression_to_record(
    expr: expression.Expression, *, source_name: str, record_index: int = 0
) -> CorpusRecord:
    """Build a :class:`CorpusRecord` from a layers ``expression``."""
    return CorpusRecord(
        text=expr.text if expr.text is not None else "",
        source_name=source_name,
        record_index=record_index,
        provenance=read_feature_map_scalar(expr.features),
    )


def corpus_to_records(
    corpus: Corpus, *, source_name: str | None = None
) -> Iterator[CorpusRecord]:
    """Stream a corpus's member expressions as corpus records."""
    for index, expr in enumerate(_each(corpus.expressions)):
        yield expression_to_record(
            expr, source_name=_source(source_name, expr), record_index=index
        )


def corpus_to_graph(corpus: Corpus, *, source_name: str | None = None) -> CorpusGraph:
    """Build a corpus graph from a corpus, with a ``parent`` edge per ``parentRef``."""
    nodes: list[CorpusNode] = []
    edges: list[CorpusEdge] = []
    for index, expr in enumerate(_each(corpus.expressions)):
        record = expression_to_record(
            expr, source_name=_source(source_name, expr), record_index=index
        )
        nodes.append(CorpusNode(node_id=expr.id, node_type=expr.kind, record=record))
        if expr.parentRef is not None:
            edges.append(
                CorpusEdge(
                    source_id=expr.id,
                    target_id=_rkey(expr.parentRef),
                    edge_type=_PARENT_EDGE,
                )
            )
    return CorpusGraph(nodes=tuple(nodes), edges=tuple(edges))


def _span_from_annotation(
    item_annotation: annotation.Annotation, element: str
) -> Span | None:
    sequence = (
        item_annotation.anchor.tokenRefSequence
        if item_annotation.anchor is not None
        else None
    )
    if sequence is None:
        return None
    knowledge = item_annotation.knowledgeRefs or ()
    label_id = knowledge[0].identifier if knowledge else None
    confidence = (
        item_annotation.confidence / CONFIDENCE_SCALE
        if item_annotation.confidence is not None
        else None
    )
    label = (
        SpanLabel(label=item_annotation.label, label_id=label_id, confidence=confidence)
        if item_annotation.label is not None
        else None
    )
    return Span(
        span_id=item_annotation.uuid.value,
        segments=(SpanSegment(element_name=element, indices=sequence.tokenIndexes),),
        head_index=sequence.anchorTokenIndex,
        label=label,
    )


def _relation_from_annotation(
    item_annotation: annotation.Annotation,
) -> SpanRelation | None:
    arguments = {
        argument.role: argument.target.localId.value
        for argument in (item_annotation.arguments or ())
        if argument.target.localId is not None
    }
    if "source" not in arguments or "target" not in arguments:
        return None
    label = (
        SpanLabel(label=item_annotation.label)
        if item_annotation.label is not None
        else None
    )
    return SpanRelation(
        relation_id=item_annotation.uuid.value,
        source_span_id=arguments["source"],
        target_span_id=arguments["target"],
        label=label,
    )


def corpus_to_items(corpus: Corpus, *, item_template_id: UUID) -> Iterator[Item]:
    """Reconstruct best-effort items from a corpus's annotation layers.

    Third-party corpora carry no bead complement, so item-construction fields
    take their defaults; the rendered text, spans, and relations are recovered
    from the expression and its span and relation annotation layers.
    """
    for expr_with in _each(corpus.with_annotations()):
        element = _DEFAULT_ELEMENT
        spans: list[Span] = []
        relations: list[SpanRelation] = []
        for layer in expr_with.annotation_layers:
            if layer.kind == "span":
                spans.extend(
                    span
                    for span in (
                        _span_from_annotation(item_annotation, element)
                        for item_annotation in layer.annotations
                    )
                    if span is not None
                )
            elif layer.kind == "relation":
                relations.extend(
                    relation
                    for relation in (
                        _relation_from_annotation(item_annotation)
                        for item_annotation in layer.annotations
                    )
                    if relation is not None
                )
        text = expr_with.expression.text
        yield Item(
            item_template_id=item_template_id,
            rendered_elements={element: text if text is not None else ""},
            spans=tuple(spans),
            span_relations=tuple(relations),
        )


def load_layers_corpus(
    uri: str,
    *,
    source: str = "pds",
    pds_client: PdsClient | None = None,
    follow_refs: bool = True,
) -> Corpus:
    """Load a layers corpus by AT-URI (a thin wrapper over ``lairs.load_corpus``)."""
    return load_corpus(
        uri, source=source, pds_client=pds_client, follow_refs=follow_refs
    )


# --- egress: bead -> layers -------------------------------------------------


def _expression_uri(authority: str, name: str, key: str) -> str:
    return f"at://{authority}/{name}/{key}"


def items_to_corpus(
    collection: ItemCollection, *, corpus_name: str, authority: str = "local"
) -> Corpus:
    """Build a layers corpus from a bead item collection.

    Each item's rendered elements become expressions, its tokenizations become
    segmentations, and its spans and relations become annotation layers (reusing
    :data:`~bead.interop.layers.item_bridge.ITEM_LAYERS`); a membership links each
    expression to the corpus, with one corpus record describing the dataset.
    """
    corpus_uri = _expression_uri(authority, CORPUS_NSID, corpus_name)
    corpus = Corpus.new(corpus_uri)
    expression_index = 0
    for item_index, item in enumerate(collection.items):
        fragment, _complement = ITEM_LAYERS.forward(item)
        local_to_uri: dict[str, str] = {}
        for record in fragment.records:
            key = f"item{item_index}-{record.local_id.replace(':', '-')}"
            uri = _expression_uri(authority, record.nsid, key)
            if record.nsid == EXPRESSION_NSID:
                expr = expression.Expression.model_validate_json(record.value_json)
                local_to_uri[f"at://local/{expr.id}"] = uri
                corpus.add_expression(uri, expr)
                corpus.add_membership(
                    _expression_uri(authority, MEMBERSHIP_NSID, key),
                    corpus_records.Membership(
                        corpusRef=corpus_uri,
                        expressionRef=uri,
                        createdAt=item.created_at,
                        ordinal=expression_index,
                    ),
                )
                expression_index += 1
        for record in fragment.records:
            key = f"item{item_index}-{record.local_id.replace(':', '-')}"
            uri = _expression_uri(authority, record.nsid, key)
            if record.nsid == SEGMENTATION_NSID:
                seg = segmentation.Segmentation.model_validate_json(record.value_json)
                corpus.add_record(
                    uri,
                    seg.with_(
                        expression=local_to_uri.get(seg.expression, seg.expression)
                    ),
                )
            elif record.nsid == ANNOTATION_LAYER_NSID:
                layer = annotation.AnnotationLayer.model_validate_json(
                    record.value_json
                )
                corpus.add_annotation_layer(
                    uri,
                    layer.with_(
                        expression=local_to_uri.get(layer.expression, layer.expression)
                    ),
                )
    corpus.add_record(
        corpus_uri,
        corpus_records.Corpus(
            name=corpus_name,
            createdAt=collection.created_at,
            expressionCount=expression_index,
        ),
    )
    return corpus


def graph_to_corpus(
    graph: CorpusGraph, *, corpus_name: str, authority: str = "local"
) -> Corpus:
    """Build a layers corpus from a corpus graph's record-bearing nodes."""
    corpus_uri = _expression_uri(authority, CORPUS_NSID, corpus_name)
    corpus = Corpus.new(corpus_uri)
    ordinal = 0
    for node in graph.nodes:
        if node.record is None:
            continue
        uri = _expression_uri(authority, EXPRESSION_NSID, node.node_id)
        corpus.add_expression(
            uri,
            expression.Expression(
                id=node.node_id,
                kind=node.node_type,
                createdAt=node.record.created_at,
                text=node.record.text,
                features=feature_map(node.record.provenance),
            ),
        )
        corpus.add_membership(
            _expression_uri(authority, MEMBERSHIP_NSID, node.node_id),
            corpus_records.Membership(
                corpusRef=corpus_uri,
                expressionRef=uri,
                createdAt=node.record.created_at,
                ordinal=ordinal,
            ),
        )
        ordinal += 1
    corpus.add_record(
        corpus_uri,
        corpus_records.Corpus(
            name=corpus_name, createdAt=graph.created_at, expressionCount=ordinal
        ),
    )
    return corpus


def materialize_corpus(corpus: Corpus, out_dir: Path) -> list[Path]:
    """Materialize a corpus to Arrow/Parquet views (delegates to ``lairs``)."""
    return corpus.materialize(out_dir)


def save_corpus_repo(corpus: Corpus, path: Path) -> str:
    """Commit a corpus to a local lairs repository and return the revision."""
    return corpus.save_to_repo(path)


def publish_corpus(
    repo: Repository,
    revision: str,
    *,
    to: str,
    endpoint: str | None = None,
    client: httpx.Client | None = None,
    dry_run: bool = True,
) -> object:
    """Publish a committed corpus revision to a PDS (opt-in; default dry run).

    Returns the ``lairs`` ``PublishPlan``. ``endpoint`` is the PDS base URL and
    ``client`` an authorized ``httpx.Client``; both are required for an actual
    write (``dry_run=False``). The network-bound publish entry point is imported
    lazily so importing this module never pulls the publish stack.
    """
    from lairs.author import publish as publish_module  # noqa: PLC0415

    return publish_module.publish(
        repo, revision, to=to, endpoint=endpoint, client=client, dry_run=dry_run
    )
