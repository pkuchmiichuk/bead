"""Faithful didactic mirrors of the linguistic ``layers`` record types.

Mirrors the expression, segmentation, annotation, graph, media, and ontology
records field-for-field (reusing the shared-def mirrors in
:mod:`bead.interop.layers.models`). Like the shared defs, they serialize to and
from layers JSON losslessly through the generic snake<->camel conversion.
Binary ``blob`` fields are mirrored as their reference string.
"""

from __future__ import annotations

import didactic.api as dx

from bead.interop.layers.models import (
    Anchor,
    AnnotationMetadata,
    FeatureMap,
    KnowledgeRef,
    LayersConstraint,
    LayersSpan,
    LayersUuid,
    ObjectRef,
    SpatialExpression,
    TemporalExpression,
    TemporalSpan,
)


class Expression(dx.Model):
    """A layers ``expression`` (a text/linguistic unit, recursively nested)."""

    id: str
    kind: str
    created_at: str
    kind_uri: str | None = None
    text: str | None = None
    parent_ref: str | None = None
    anchor: dx.Embed[Anchor] | None = None
    media_ref: str | None = None
    media_blob: str | None = None
    metadata: dx.Embed[AnnotationMetadata] | None = None
    features: dx.Embed[FeatureMap] | None = None
    source_url: str | None = None
    source_ref: str | None = None
    eprint_ref: str | None = None
    knowledge_refs: tuple[dx.Embed[KnowledgeRef], ...] = ()
    languages: tuple[str, ...] = ()


class Token(dx.Model):
    """A layers ``token`` within a tokenization."""

    token_index: int
    text: str | None = None
    text_span: dx.Embed[LayersSpan] | None = None
    temporal_span: dx.Embed[TemporalSpan] | None = None


class Tokenization(dx.Model):
    """A layers ``tokenization`` (one segmentation of an expression)."""

    uuid: dx.Embed[LayersUuid]
    kind: str
    kind_uri: str | None = None
    expression_ref: str | None = None
    tokens: tuple[dx.Embed[Token], ...] = ()
    metadata: dx.Embed[AnnotationMetadata] | None = None


class ArgumentRef(dx.Model):
    """A layers ``argumentRef`` (a role-filling argument of a predicate)."""

    role: str
    target: dx.Embed[ObjectRef]
    features: dx.Embed[FeatureMap] | None = None


class Annotation(dx.Model):
    """A layers ``annotation`` (the polymorphic annotation object)."""

    uuid: dx.Embed[LayersUuid]
    anchor: dx.Embed[Anchor] | None = None
    token_index: int | None = None
    label: str | None = None
    value: str | None = None
    text: str | None = None
    parent_id: dx.Embed[LayersUuid] | None = None
    child_ids: tuple[dx.Embed[LayersUuid], ...] = ()
    head_index: int | None = None
    target_index: int | None = None
    arguments: tuple[dx.Embed[ArgumentRef], ...] = ()
    confidence: int | None = None
    ontology_type_ref: str | None = None
    knowledge_refs: tuple[dx.Embed[KnowledgeRef], ...] = ()
    temporal: dx.Embed[TemporalExpression] | None = None
    spatial: dx.Embed[SpatialExpression] | None = None
    features: dx.Embed[FeatureMap] | None = None


class Cluster(dx.Model):
    """A layers ``cluster`` (a coreference/equivalence set)."""

    uuid: dx.Embed[LayersUuid]
    canonical_label: str | None = None
    members: tuple[dx.Embed[ObjectRef], ...] = ()
    knowledge_refs: tuple[dx.Embed[KnowledgeRef], ...] = ()
    features: dx.Embed[FeatureMap] | None = None


class AnnotationLayer(dx.Model):
    """A layers ``annotationLayer`` (a typed layer of annotations)."""

    expression: str
    kind: str
    created_at: str
    kind_uri: str | None = None
    subkind_uri: str | None = None
    subkind: str | None = None
    formalism_uri: str | None = None
    formalism: str | None = None
    source_method_uri: str | None = None
    source_method: str | None = None
    label_set: str | None = None
    ontology_ref: str | None = None
    tokenization_id: dx.Embed[LayersUuid] | None = None
    rank: int | None = None
    alternatives_ref: str | None = None
    parent_layer_ref: str | None = None
    annotations: tuple[dx.Embed[Annotation], ...] = ()
    metadata: dx.Embed[AnnotationMetadata] | None = None
    languages: tuple[str, ...] = ()


class GraphNode(dx.Model):
    """A layers ``graphNode`` (a standalone property-graph node)."""

    node_type: str
    created_at: str
    node_type_uri: str | None = None
    label: str | None = None
    properties: dx.Embed[FeatureMap] | None = None
    knowledge_refs: tuple[dx.Embed[KnowledgeRef], ...] = ()
    metadata: dx.Embed[AnnotationMetadata] | None = None


class GraphEdge(dx.Model):
    """A layers ``graphEdge`` (a single typed, directed edge record)."""

    source: dx.Embed[ObjectRef]
    target: dx.Embed[ObjectRef]
    edge_type: str
    created_at: str
    edge_type_uri: str | None = None
    label: str | None = None
    ordinal: int | None = None
    confidence: int | None = None
    properties: dx.Embed[FeatureMap] | None = None
    metadata: dx.Embed[AnnotationMetadata] | None = None


class GraphEdgeEntry(dx.Model):
    """A layers ``graphEdgeEntry`` (one edge within a graphEdgeSet)."""

    uuid: dx.Embed[LayersUuid]
    edge_type: str
    source: dx.Embed[ObjectRef]
    target: dx.Embed[ObjectRef]
    edge_type_uri: str | None = None
    confidence: int | None = None
    features: dx.Embed[FeatureMap] | None = None


class GraphEdgeSet(dx.Model):
    """A layers ``graphEdgeSet`` (a batch of typed, directed edges)."""

    created_at: str
    edges: tuple[dx.Embed[GraphEdgeEntry], ...] = ()
    expression: str | None = None
    edge_type_uri: str | None = None
    edge_type: str | None = None
    metadata: dx.Embed[AnnotationMetadata] | None = None
    knowledge_refs: tuple[dx.Embed[KnowledgeRef], ...] = ()
    features: dx.Embed[FeatureMap] | None = None


class AudioInfo(dx.Model):
    """A layers ``audioInfo`` media descriptor."""

    sample_rate: int | None = None
    channels: int | None = None
    bit_depth: int | None = None
    codec: str | None = None
    bit_rate: int | None = None
    bit_rate_mode: str | None = None
    number_of_samples: int | None = None
    speaker_count: int | None = None
    transcript_ref: str | None = None
    segmentation_ref: str | None = None


class VideoInfo(dx.Model):
    """A layers ``videoInfo`` media descriptor."""

    width: int | None = None
    height: int | None = None
    frame_rate: int | None = None
    codec: str | None = None
    aspect_ratio: str | None = None
    color_space: str | None = None
    bit_rate: int | None = None
    scan_type: str | None = None


class DocumentInfo(dx.Model):
    """A layers ``documentInfo`` media descriptor."""

    dpi: int | None = None
    color_mode: str | None = None
    page_count: int | None = None
    script_system: str | None = None
    writing_direction: str | None = None
    ocr_engine: str | None = None


class RoleSlot(dx.Model):
    """A layers ``roleSlot`` (a role in a type definition)."""

    role_name: str
    role_description: str | None = None
    filler_type_refs: tuple[str, ...] = ()
    collection_ref: str | None = None
    required: bool | None = None
    default_value: str | None = None
    constraints: tuple[dx.Embed[LayersConstraint], ...] = ()
    knowledge_refs: tuple[dx.Embed[KnowledgeRef], ...] = ()
    features: dx.Embed[FeatureMap] | None = None


class TypeDef(dx.Model):
    """A layers ``typeDef`` (an ontology type definition)."""

    ontology_ref: str
    name: str
    created_at: str
    type_kind: str | None = None
    type_kind_uri: str | None = None
    gloss: str | None = None
    parent_type_ref: str | None = None
    allowed_roles: tuple[dx.Embed[RoleSlot], ...] = ()
    allowed_values: tuple[str, ...] = ()
    knowledge_refs: tuple[dx.Embed[KnowledgeRef], ...] = ()
    features: dx.Embed[FeatureMap] | None = None
