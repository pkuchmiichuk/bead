"""Generic lossless isos between faithful mirror models and layers JSON.

A single :class:`MirrorIso` (parameterized by model type) serves every faithful
mirror model, since they all serialize through the structural snake<->camel
conversion in :mod:`bead.interop.layers._mirror`. ``SHARED_DEF_ISOS`` registers
one iso per shared-def mirror so a coverage test can assert every construct has
a law-passing mapping.
"""

from __future__ import annotations

import didactic.api as dx

from bead.data.base import JsonValue
from bead.interop.layers._mirror import mirror_from_layers, mirror_to_layers
from bead.interop.layers.models import (
    AgentRef,
    AlignmentLink,
    Anchor,
    AnnotationMetadata,
    BoundingBox,
    ExternalTarget,
    Feature,
    FeatureMap,
    FragmentSelector,
    Keyframe,
    KnowledgeRef,
    LayersConstraint,
    LayersSpan,
    LayersUuid,
    ObjectRef,
    PageAnchor,
    Selector,
    SpatialEntity,
    SpatialExpression,
    SpatialModifier,
    SpatioTemporalAnchor,
    TemporalEntity,
    TemporalExpression,
    TemporalModifier,
    TemporalSpan,
    TextPositionSelector,
    TextQuoteSelector,
    TokenRef,
    TokenRefSequence,
)
from bead.interop.layers.models_records import (
    Annotation,
    AnnotationLayer,
    ArgumentRef,
    AudioInfo,
    Cluster,
    DocumentInfo,
    Expression,
    GraphEdge,
    GraphEdgeEntry,
    GraphEdgeSet,
    GraphNode,
    RoleSlot,
    Token,
    Tokenization,
    TypeDef,
    VideoInfo,
)


class MirrorIso[T: dx.Model](dx.Iso[T, JsonValue]):
    """Lossless iso between a faithful mirror model and layers-shaped JSON."""

    def __init__(self, model_type: type[T]) -> None:
        self._model_type = model_type

    def forward(self, model: T) -> JsonValue:
        """Serialize the mirror model to layers JSON."""
        return mirror_to_layers(model)

    def backward(self, data: JsonValue) -> T:
        """Deserialize layers JSON back into the mirror model."""
        return mirror_from_layers(self._model_type, data)


def mirror_iso[T: dx.Model](model_type: type[T]) -> MirrorIso[T]:
    """Build a :class:`MirrorIso` for a mirror model type."""
    return MirrorIso(model_type)


#: Every shared-def mirror model, for coverage and registration.
SHARED_DEF_MODELS: tuple[type[dx.Model], ...] = (
    LayersUuid,
    Feature,
    FeatureMap,
    KnowledgeRef,
    BoundingBox,
    TemporalSpan,
    AgentRef,
    ObjectRef,
    LayersSpan,
    TokenRef,
    TokenRefSequence,
    Keyframe,
    SpatioTemporalAnchor,
    TemporalEntity,
    TemporalModifier,
    TemporalExpression,
    SpatialEntity,
    SpatialModifier,
    SpatialExpression,
    PageAnchor,
    TextQuoteSelector,
    TextPositionSelector,
    FragmentSelector,
    Selector,
    ExternalTarget,
    Anchor,
    AlignmentLink,
    AnnotationMetadata,
    LayersConstraint,
)

#: One lossless iso per shared-def mirror model.
SHARED_DEF_ISOS: dict[type[dx.Model], MirrorIso[dx.Model]] = {
    model_type: MirrorIso(model_type) for model_type in SHARED_DEF_MODELS
}

#: Every linguistic record mirror model.
RECORD_MODELS: tuple[type[dx.Model], ...] = (
    Expression,
    Token,
    Tokenization,
    ArgumentRef,
    Annotation,
    Cluster,
    AnnotationLayer,
    GraphNode,
    GraphEdge,
    GraphEdgeEntry,
    GraphEdgeSet,
    AudioInfo,
    VideoInfo,
    DocumentInfo,
    RoleSlot,
    TypeDef,
)

#: One lossless iso per record mirror model.
RECORD_ISOS: dict[type[dx.Model], MirrorIso[dx.Model]] = {
    model_type: MirrorIso(model_type) for model_type in RECORD_MODELS
}

#: All mirror isos (shared defs + records), keyed by model type.
ALL_MIRROR_ISOS: dict[type[dx.Model], MirrorIso[dx.Model]] = {
    **SHARED_DEF_ISOS,
    **RECORD_ISOS,
}
