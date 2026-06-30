"""Core span annotation models.

Provides data models for labeled spans, span segments, span labels, span
relations, and span specifications. Supports discontiguous spans,
overlapping spans (nested and intersecting), static and interactive modes,
and two label sources (fixed sets and Wikidata entity search).
"""

from __future__ import annotations

from typing import Literal

import didactic.api as dx

from bead.data.base import BeadBaseModel

type MetadataValue = (
    str
    | int
    | float
    | bool
    | None
    | tuple[MetadataValue, ...]
    | dict[str, MetadataValue]
)

# same recursive type as in item.py and item_template.py; duplicated here
# to avoid circular imports.

SpanIndexMode = Literal["token", "character"]
SpanInteractionMode = Literal["static", "interactive"]
LabelSourceType = Literal["fixed", "wikidata"]


class SpanSegment(BeadBaseModel):
    """Contiguous or discontiguous indices within a single element.

    Attributes
    ----------
    element_name : str
        Which rendered element this segment belongs to.
    indices : tuple[int, ...]
        Token or character indices within the element.
    """

    element_name: str
    indices: tuple[int, ...]

    @dx.validates("element_name")
    def _check_element_name(self, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("element_name cannot be empty")
        return value.strip()

    @dx.validates("indices")
    def _check_indices(self, value: tuple[int, ...]) -> tuple[int, ...]:
        if not value:
            raise ValueError("indices cannot be empty")
        if any(i < 0 for i in value):
            raise ValueError("indices must be non-negative")
        return value


class SpanLabel(BeadBaseModel):
    """Label applied to a span or relation.

    Attributes
    ----------
    label : str
        Human-readable label text.
    label_id : str | None
        External identifier (e.g. Wikidata QID "Q5").
    confidence : float | None
        Confidence score for model-assigned labels.
    """

    label: str
    label_id: str | None = None
    confidence: float | None = None

    @dx.validates("label")
    def _check_label(self, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("label cannot be empty")
        return value.strip()


class Span(BeadBaseModel):
    """Labeled span across one or more elements.

    Supports discontiguous, overlapping, and nested spans.

    Attributes
    ----------
    span_id : str
        Unique identifier within the item.
    segments : tuple[SpanSegment, ...]
        Index segments composing this span.
    head_index : int | None
        Syntactic head token index.
    label : SpanLabel | None
        Label applied to this span (None = to-be-labeled).
    span_type : str | None
        Semantic category (e.g. "entity", "event", "role").
    span_metadata : dict[str, MetadataValue]
        Additional span-specific metadata.
    """

    span_id: str
    segments: tuple[dx.Embed[SpanSegment], ...] = ()
    head_index: int | None = None
    label: dx.Embed[SpanLabel] | None = None
    span_type: str | None = None
    span_metadata: dict[str, MetadataValue] = dx.field(default_factory=dict)

    @dx.validates("span_id")
    def _check_span_id(self, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("span_id cannot be empty")
        return value.strip()


class SpanRelation(BeadBaseModel):
    """A typed, directed relation between two spans.

    Attributes
    ----------
    relation_id : str
        Unique identifier within the item.
    source_span_id : str
        ``span_id`` of the source span.
    target_span_id : str
        ``span_id`` of the target span.
    label : SpanLabel | None
        Relation label.
    directed : bool
        Whether the relation is directed (A -> B) or undirected (A -- B).
    relation_metadata : dict[str, MetadataValue]
        Additional relation-specific metadata.
    """

    relation_id: str
    source_span_id: str
    target_span_id: str
    label: dx.Embed[SpanLabel] | None = None
    directed: bool = True
    relation_metadata: dict[str, MetadataValue] = dx.field(default_factory=dict)

    @dx.validates("relation_id")
    def _check_relation_id(self, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("relation_id cannot be empty")
        return value.strip()

    @dx.validates("source_span_id", "target_span_id")
    def _check_span_id_field(self, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("span ID cannot be empty")
        return value.strip()


class SpanSpec(BeadBaseModel):
    """Specification for span labeling behavior.

    Attributes
    ----------
    index_mode : SpanIndexMode
        Whether spans index by token or character position.
    interaction_mode : SpanInteractionMode
        ``static`` for read-only highlights, ``interactive`` for
        participant annotation.
    label_source : LabelSourceType
        Source of span labels (``fixed`` or ``wikidata``).
    labels : tuple[str, ...] | None
        Fixed span label set when ``label_source == "fixed"``.
    label_colors : dict[str, str] | None
        CSS colors keyed by label name.
    allow_overlapping : bool
        Whether overlapping spans are permitted.
    min_spans : int | None
        Minimum number of spans required (interactive mode).
    max_spans : int | None
        Maximum number of spans allowed (interactive mode).
    enable_relations : bool
        Whether relation annotation is enabled.
    relation_label_source : LabelSourceType
        Source of relation labels.
    relation_labels : tuple[str, ...] | None
        Fixed relation label set.
    relation_label_colors : dict[str, str] | None
        CSS colors keyed by relation label name.
    relation_directed : bool
        Default directionality for new relations.
    min_relations : int | None
        Minimum number of relations required (interactive mode).
    max_relations : int | None
        Maximum number of relations allowed (interactive mode).
    wikidata_language : str
        Language for Wikidata entity search.
    wikidata_entity_types : tuple[str, ...] | None
        Restrict Wikidata search to these entity types.
    wikidata_result_limit : int
        Maximum number of Wikidata search results.
    """

    index_mode: SpanIndexMode = "token"
    interaction_mode: SpanInteractionMode = "static"
    label_source: LabelSourceType = "fixed"
    labels: tuple[str, ...] | None = None
    label_colors: dict[str, str] | None = None
    allow_overlapping: bool = True
    min_spans: int | None = None
    max_spans: int | None = None
    enable_relations: bool = False
    relation_label_source: LabelSourceType = "fixed"
    relation_labels: tuple[str, ...] | None = None
    relation_label_colors: dict[str, str] | None = None
    relation_directed: bool = True
    min_relations: int | None = None
    max_relations: int | None = None
    wikidata_language: str = "en"
    wikidata_entity_types: tuple[str, ...] | None = None
    wikidata_result_limit: int = 10
