"""Coverage guard: every targeted layers construct has a law-passing mapping.

If a new layers construct is mirrored, it must be registered (and round-trip
tested in the per-construct suites). This test fails loudly if a targeted
construct loses its registered iso.
"""

from __future__ import annotations

from bead.interop.layers.model_lenses import ALL_MIRROR_ISOS, MirrorIso

# layers construct slug -> bead mirror model class name.
_EXPECTED: dict[str, str] = {
    # pub.layers.defs shared objects
    "uuid": "LayersUuid",
    "feature": "Feature",
    "featureMap": "FeatureMap",
    "knowledgeRef": "KnowledgeRef",
    "boundingBox": "BoundingBox",
    "temporalSpan": "TemporalSpan",
    "agentRef": "AgentRef",
    "objectRef": "ObjectRef",
    "span": "LayersSpan",
    "tokenRef": "TokenRef",
    "tokenRefSequence": "TokenRefSequence",
    "keyframe": "Keyframe",
    "spatioTemporalAnchor": "SpatioTemporalAnchor",
    "temporalEntity": "TemporalEntity",
    "temporalModifier": "TemporalModifier",
    "temporalExpression": "TemporalExpression",
    "spatialEntity": "SpatialEntity",
    "spatialModifier": "SpatialModifier",
    "spatialExpression": "SpatialExpression",
    "pageAnchor": "PageAnchor",
    "textQuoteSelector": "TextQuoteSelector",
    "textPositionSelector": "TextPositionSelector",
    "fragmentSelector": "FragmentSelector",
    "externalTarget": "ExternalTarget",
    "anchor": "Anchor",
    "alignmentLink": "AlignmentLink",
    "annotationMetadata": "AnnotationMetadata",
    "constraint": "LayersConstraint",
    # linguistic record types
    "expression": "Expression",
    "token": "Token",
    "tokenization": "Tokenization",
    "argumentRef": "ArgumentRef",
    "annotation": "Annotation",
    "cluster": "Cluster",
    "annotationLayer": "AnnotationLayer",
    "graphNode": "GraphNode",
    "graphEdge": "GraphEdge",
    "graphEdgeEntry": "GraphEdgeEntry",
    "graphEdgeSet": "GraphEdgeSet",
    "audioInfo": "AudioInfo",
    "videoInfo": "VideoInfo",
    "documentInfo": "DocumentInfo",
    "roleSlot": "RoleSlot",
    "typeDef": "TypeDef",
}


def test_every_targeted_construct_is_mapped() -> None:
    mapped_model_names = {model_type.__name__ for model_type in ALL_MIRROR_ISOS}
    missing = {
        slug: name for slug, name in _EXPECTED.items() if name not in mapped_model_names
    }
    assert not missing, f"layers constructs without a mirror iso: {missing}"


def test_all_registrations_are_law_lenses() -> None:
    assert ALL_MIRROR_ISOS
    assert all(isinstance(iso, MirrorIso) for iso in ALL_MIRROR_ISOS.values())
