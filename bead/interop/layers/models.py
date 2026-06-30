"""Faithful didactic mirrors of the ``layers`` shared object definitions.

Each model mirrors a ``pub.layers.defs`` object field-for-field (snake_case
names corresponding to layers' camelCase, nested objects as embedded models,
feature maps as :class:`FeatureMap`, confidence as an integer 0-1000). The
structural fidelity lets :mod:`bead.interop.layers._mirror` serialize any of
them to and from layers JSON losslessly with a single generic conversion.

Names that would clash with bead's own models are prefixed ``Layers``.
"""

from __future__ import annotations

import didactic.api as dx


class LayersUuid(dx.Model):
    """A layers ``uuid`` value object."""

    value: str


class Feature(dx.Model):
    """A single key/value entry in a layers ``featureMap``."""

    key: str
    value: str


class FeatureMap(dx.Model):
    """A layers ``featureMap`` (ordered key/value entries)."""

    entries: tuple[dx.Embed[Feature], ...] = ()


class KnowledgeRef(dx.Model):
    """A layers ``knowledgeRef`` grounding to an external knowledge base."""

    source: str
    identifier: str
    source_uri: str | None = None
    uri: str | None = None
    label: str | None = None


class BoundingBox(dx.Model):
    """A layers ``boundingBox`` (pixel region)."""

    x: int
    y: int
    width: int
    height: int


class TemporalSpan(dx.Model):
    """A layers ``temporalSpan`` (millisecond interval)."""

    start: int
    ending: int


class AgentRef(dx.Model):
    """A layers ``agentRef`` (annotating agent)."""

    did: str | None = None
    id: str | None = None
    name: str | None = None
    knowledge_ref: dx.Embed[KnowledgeRef] | None = None


class ObjectRef(dx.Model):
    """A layers ``objectRef`` (local, record, or external reference)."""

    local_id: dx.Embed[LayersUuid] | None = None
    record_ref: str | None = None
    object_id: dx.Embed[LayersUuid] | None = None
    knowledge_ref: dx.Embed[KnowledgeRef] | None = None


class LayersSpan(dx.Model):
    """A layers ``span`` (UTF-8 byte offsets, optional char offsets)."""

    byte_start: int
    byte_end: int
    char_start: int | None = None
    char_end: int | None = None


class TokenRef(dx.Model):
    """A layers ``tokenRef`` (single token in a tokenization)."""

    tokenization_id: dx.Embed[LayersUuid]
    token_index: int


class TokenRefSequence(dx.Model):
    """A layers ``tokenRefSequence`` (ordered tokens, optional anchor)."""

    tokenization_id: dx.Embed[LayersUuid]
    token_indexes: tuple[int, ...] = ()
    anchor_token_index: int | None = None


class Keyframe(dx.Model):
    """A layers ``keyframe`` (a bounding box at a video time)."""

    time_ms: int
    bbox: dx.Embed[BoundingBox]
    features: dx.Embed[FeatureMap] | None = None


class SpatioTemporalAnchor(dx.Model):
    """A layers ``spatioTemporalAnchor`` (time span plus keyframes)."""

    temporal_span: dx.Embed[TemporalSpan]
    keyframes: tuple[dx.Embed[Keyframe], ...] = ()
    interpolation_uri: str | None = None
    interpolation: str | None = None


class TemporalEntity(dx.Model):
    """A layers ``temporalEntity`` (instant/interval/duration value)."""

    instant: str | None = None
    interval_start: str | None = None
    interval_end: str | None = None
    duration: str | None = None
    earliest: str | None = None
    latest: str | None = None
    granularity_uri: str | None = None
    granularity: str | None = None
    calendar_uri: str | None = None
    calendar: str | None = None
    recurrence: str | None = None
    features: dx.Embed[FeatureMap] | None = None


class TemporalModifier(dx.Model):
    """A layers ``temporalModifier``."""

    mod_uri: str | None = None
    mod: str | None = None
    features: dx.Embed[FeatureMap] | None = None


class TemporalExpression(dx.Model):
    """A layers ``temporalExpression``."""

    type_uri: str | None = None
    type: str | None = None
    value: dx.Embed[TemporalEntity] | None = None
    modifier: dx.Embed[TemporalModifier] | None = None
    anchor_ref: dx.Embed[ObjectRef] | None = None
    function_uri: str | None = None
    function: str | None = None
    features: dx.Embed[FeatureMap] | None = None


class SpatialEntity(dx.Model):
    """A layers ``spatialEntity`` (geometry/region value)."""

    bbox: dx.Embed[BoundingBox] | None = None
    geometry: str | None = None
    type_uri: str | None = None
    type: str | None = None
    geometry_format_uri: str | None = None
    geometry_format: str | None = None
    crs_uri: str | None = None
    crs: str | None = None
    dimensions: int | None = None
    uncertainty: str | None = None
    features: dx.Embed[FeatureMap] | None = None


class SpatialModifier(dx.Model):
    """A layers ``spatialModifier``."""

    mod_uri: str | None = None
    mod: str | None = None
    features: dx.Embed[FeatureMap] | None = None


class SpatialExpression(dx.Model):
    """A layers ``spatialExpression``."""

    type_uri: str | None = None
    type: str | None = None
    value: dx.Embed[SpatialEntity] | None = None
    modifier: dx.Embed[SpatialModifier] | None = None
    anchor_ref: dx.Embed[ObjectRef] | None = None
    function_uri: str | None = None
    function: str | None = None
    features: dx.Embed[FeatureMap] | None = None


class PageAnchor(dx.Model):
    """A layers ``pageAnchor`` (a region on a document page)."""

    page: int
    bounding_box: dx.Embed[BoundingBox] | None = None
    text_span: dx.Embed[LayersSpan] | None = None


class TextQuoteSelector(dx.Model):
    """A W3C-style ``textQuoteSelector``."""

    exact: str
    prefix: str | None = None
    suffix: str | None = None


class TextPositionSelector(dx.Model):
    """A W3C-style ``textPositionSelector``."""

    byte_start: int
    byte_end: int
    char_start: int | None = None
    char_end: int | None = None


class FragmentSelector(dx.Model):
    """A W3C-style ``fragmentSelector``."""

    value: str
    conforms_to: str | None = None


class Selector(dx.Model):
    """The selector union of a layers ``externalTarget``."""

    text_quote_selector: dx.Embed[TextQuoteSelector] | None = None
    text_position_selector: dx.Embed[TextPositionSelector] | None = None
    fragment_selector: dx.Embed[FragmentSelector] | None = None


class ExternalTarget(dx.Model):
    """A layers ``externalTarget`` (a web resource + selector)."""

    source: str
    source_hash: str | None = None
    title: str | None = None
    selector: dx.Embed[Selector] | None = None


class Anchor(dx.Model):
    """A layers ``anchor`` (the polymorphic attachment point)."""

    text_span: dx.Embed[LayersSpan] | None = None
    token_ref: dx.Embed[TokenRef] | None = None
    token_ref_sequence: dx.Embed[TokenRefSequence] | None = None
    temporal_span: dx.Embed[TemporalSpan] | None = None
    spatio_temporal_anchor: dx.Embed[SpatioTemporalAnchor] | None = None
    page_anchor: dx.Embed[PageAnchor] | None = None
    external_target: dx.Embed[ExternalTarget] | None = None


class AlignmentLink(dx.Model):
    """A layers ``alignmentLink`` (aligned token-index sets)."""

    source_indices: tuple[int, ...] = ()
    target_indices: tuple[int, ...] = ()
    confidence: int | None = None
    label: str | None = None
    knowledge_refs: tuple[dx.Embed[KnowledgeRef], ...] = ()
    features: dx.Embed[FeatureMap] | None = None


class AnnotationMetadata(dx.Model):
    """A layers ``annotationMetadata`` (provenance for an annotation)."""

    tool: str
    agent: dx.Embed[AgentRef] | None = None
    timestamp: str | None = None
    confidence: int | None = None
    persona_ref: str | None = None
    dependencies: tuple[dx.Embed[ObjectRef], ...] = ()
    digest: str | None = None


class LayersConstraint(dx.Model):
    """A layers ``constraint`` (an expression with scope)."""

    expression: str
    expression_format_uri: str | None = None
    expression_format: str | None = None
    scope_uri: str | None = None
    scope: str | None = None
    context: tuple[str, ...] = ()
    description: str | None = None
