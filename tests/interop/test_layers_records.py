"""Round-trip law tests for the layers record mirror isos."""

from __future__ import annotations

import didactic.api as dx
import pytest

from bead.interop.layers import models as m
from bead.interop.layers import models_records as r
from bead.interop.layers.model_lenses import RECORD_ISOS

_UUID = m.LayersUuid(value="u1")
_KR = m.KnowledgeRef(source="wikidata", identifier="Q5")
_REF = m.ObjectRef(local_id=_UUID)
_META = m.AnnotationMetadata(tool="spacy", timestamp="2026-05-29T00:00:00+00:00")
_NOW = "2026-05-29T00:00:00+00:00"

# One representative instance per record mirror model.
_EXAMPLES: tuple[dx.Model, ...] = (
    r.Expression(
        id="doc1",
        kind="document",
        created_at=_NOW,
        text="Hello world.",
        anchor=m.Anchor(text_span=m.LayersSpan(byte_start=0, byte_end=12)),
        metadata=_META,
        features=m.FeatureMap(entries=(m.Feature(key="lang", value="en"),)),
        knowledge_refs=(_KR,),
        languages=("en",),
    ),
    r.Token(
        token_index=0, text="Hello", text_span=m.LayersSpan(byte_start=0, byte_end=5)
    ),
    r.Tokenization(
        uuid=_UUID,
        kind="penn-treebank",
        tokens=(r.Token(token_index=0, text="Hi"),),
        metadata=_META,
    ),
    r.ArgumentRef(role="ARG0", target=_REF),
    r.Annotation(
        uuid=_UUID,
        token_index=2,
        label="nsubj",
        head_index=3,
        arguments=(r.ArgumentRef(role="ARG0", target=_REF),),
        confidence=900,
        knowledge_refs=(_KR,),
        temporal=m.TemporalExpression(type="date"),
    ),
    r.Cluster(uuid=_UUID, canonical_label="Alice", members=(_REF,)),
    r.AnnotationLayer(
        expression="at://x",
        kind="relation",
        subkind="dependency",
        formalism="universal-dependencies",
        created_at=_NOW,
        tokenization_id=_UUID,
        annotations=(r.Annotation(uuid=_UUID, token_index=0, label="root"),),
        metadata=_META,
    ),
    r.GraphNode(
        node_type="entity",
        created_at=_NOW,
        label="Alice",
        properties=m.FeatureMap(entries=(m.Feature(key="k", value="v"),)),
        knowledge_refs=(_KR,),
    ),
    r.GraphEdge(
        source=_REF,
        target=_REF,
        edge_type="coreference",
        created_at=_NOW,
        ordinal=1,
        confidence=800,
    ),
    r.GraphEdgeEntry(uuid=_UUID, edge_type="reply-to", source=_REF, target=_REF),
    r.GraphEdgeSet(
        created_at=_NOW,
        edges=(
            r.GraphEdgeEntry(uuid=_UUID, edge_type="reply-to", source=_REF, target=_REF),
        ),
        expression="at://x",
    ),
    r.AudioInfo(sample_rate=44100, channels=2, codec="pcm"),
    r.VideoInfo(width=1920, height=1080, frame_rate=30, codec="h264"),
    r.DocumentInfo(dpi=300, page_count=12, writing_direction="ltr"),
    r.RoleSlot(
        role_name="Agent",
        filler_type_refs=("at://t",),
        required=True,
        constraints=(m.LayersConstraint(expression="x>0"),),
    ),
    r.TypeDef(
        ontology_ref="at://o",
        name="give",
        created_at=_NOW,
        type_kind="frame",
        allowed_roles=(r.RoleSlot(role_name="Agent"),),
        allowed_values=("a", "b"),
    ),
)


@pytest.mark.parametrize("example", _EXAMPLES, ids=lambda e: type(e).__name__)
def test_record_roundtrip(example: dx.Model) -> None:
    iso = RECORD_ISOS[type(example)]
    view = iso.forward(example)
    assert iso.backward(view) == example
    assert iso.forward(iso.backward(view)) == view


def test_every_record_has_a_law_passing_iso() -> None:
    example_types = {type(example) for example in _EXAMPLES}
    assert example_types == set(RECORD_ISOS)


def test_annotation_layer_is_camelcased() -> None:
    iso = RECORD_ISOS[r.AnnotationLayer]
    view = iso.forward(
        r.AnnotationLayer(
            expression="at://x", kind="relation", subkind="dependency", created_at=_NOW
        )
    )
    assert isinstance(view, dict)
    assert view["expression"] == "at://x"
    assert view["subkind"] == "dependency"
    assert "createdAt" in view
