"""Round-trip law tests for the layers shared-def mirror isos."""

from __future__ import annotations

import didactic.api as dx
import pytest
from didactic.lenses._testing import verify_iso
from hypothesis import strategies as st

from bead.interop.layers import models as m
from bead.interop.layers.model_lenses import SHARED_DEF_ISOS, MirrorIso, mirror_iso

_KR = m.KnowledgeRef(source="wikidata", identifier="Q5")
_UUID = m.LayersUuid(value="u1")
_BBOX = m.BoundingBox(x=1, y=2, width=3, height=4)
_FEATURES = m.FeatureMap(entries=(m.Feature(key="k", value="v"),))

# One representative instance per shared-def mirror model.
_EXAMPLES: tuple[dx.Model, ...] = (
    _UUID,
    m.Feature(key="k", value="v"),
    _FEATURES,
    m.KnowledgeRef(source="wikidata", identifier="Q5", label="human"),
    _BBOX,
    m.TemporalSpan(start=0, ending=100),
    m.AgentRef(did="did:plc:x", name="A", knowledge_ref=_KR),
    m.ObjectRef(local_id=_UUID, knowledge_ref=_KR),
    m.LayersSpan(byte_start=0, byte_end=5, char_start=0, char_end=5),
    m.TokenRef(tokenization_id=_UUID, token_index=2),
    m.TokenRefSequence(
        tokenization_id=_UUID, token_indexes=(1, 2, 3), anchor_token_index=2
    ),
    m.Keyframe(time_ms=10, bbox=_BBOX, features=_FEATURES),
    m.SpatioTemporalAnchor(
        temporal_span=m.TemporalSpan(start=0, ending=10),
        keyframes=(m.Keyframe(time_ms=1, bbox=_BBOX),),
        interpolation="linear",
    ),
    m.TemporalEntity(instant="2026-05-29", granularity="day", features=_FEATURES),
    m.TemporalModifier(mod="approx"),
    m.TemporalExpression(
        type="date",
        value=m.TemporalEntity(instant="2026-05-29"),
        modifier=m.TemporalModifier(mod="approx"),
        anchor_ref=m.ObjectRef(local_id=_UUID),
    ),
    m.SpatialEntity(geometry="POINT(0 0)", type="point", dimensions=2),
    m.SpatialModifier(mod="near"),
    m.SpatialExpression(type="loc", value=m.SpatialEntity(geometry="g")),
    m.PageAnchor(
        page=1, bounding_box=_BBOX, text_span=m.LayersSpan(byte_start=0, byte_end=2)
    ),
    m.TextQuoteSelector(exact="quote", prefix="a", suffix="b"),
    m.TextPositionSelector(byte_start=0, byte_end=5),
    m.FragmentSelector(value="#frag", conforms_to="https://example/spec"),
    m.Selector(text_quote_selector=m.TextQuoteSelector(exact="q")),
    m.ExternalTarget(
        source="http://x",
        title="t",
        selector=m.Selector(fragment_selector=m.FragmentSelector(value="#f")),
    ),
    m.Anchor(token_ref=m.TokenRef(tokenization_id=_UUID, token_index=0)),
    m.AlignmentLink(
        source_indices=(0, 1),
        target_indices=(2,),
        confidence=900,
        label="align",
        knowledge_refs=(_KR,),
    ),
    m.AnnotationMetadata(
        tool="spacy",
        agent=m.AgentRef(name="A"),
        timestamp="2026-05-29T00:00:00+00:00",
        confidence=950,
        dependencies=(m.ObjectRef(local_id=_UUID),),
    ),
    m.LayersConstraint(
        expression="x>0", scope="token", context=("a", "b"), description="d"
    ),
)


@pytest.mark.parametrize("example", _EXAMPLES, ids=lambda e: type(e).__name__)
def test_shared_def_roundtrip(example: dx.Model) -> None:
    iso = SHARED_DEF_ISOS[type(example)]
    view = iso.forward(example)
    # GetPut: reconstruct exactly from the layers JSON.
    assert iso.backward(view) == example
    # PutGet: re-projection is stable.
    assert iso.forward(iso.backward(view)) == view


def test_every_shared_def_has_a_law_passing_iso() -> None:
    # Coverage guard: each example's type has a registered iso, and every
    # registered iso is exercised by an example (no silent omission).
    example_types = {type(example) for example in _EXAMPLES}
    assert example_types == set(SHARED_DEF_ISOS)


def test_camelcase_projection() -> None:
    view = mirror_iso(m.LayersSpan).forward(
        m.LayersSpan(byte_start=1, byte_end=9, char_start=1, char_end=9)
    )
    assert view == {"byteStart": 1, "byteEnd": 9, "charStart": 1, "charEnd": 9}


# --- didactic law verification on flat models -------------------------------


def test_verify_iso_uuid() -> None:
    iso: MirrorIso[m.LayersUuid] = mirror_iso(m.LayersUuid)
    verify_iso(iso, st.builds(m.LayersUuid, value=st.text(max_size=8)), max_examples=30)


def test_verify_iso_bounding_box() -> None:
    iso: MirrorIso[m.BoundingBox] = mirror_iso(m.BoundingBox)
    ints = st.integers(0, 1000)
    verify_iso(
        iso,
        st.builds(m.BoundingBox, x=ints, y=ints, width=ints, height=ints),
        max_examples=30,
    )


def test_verify_iso_knowledge_ref() -> None:
    iso: MirrorIso[m.KnowledgeRef] = mirror_iso(m.KnowledgeRef)
    text = st.text(max_size=6)
    opt = st.one_of(st.none(), text)
    verify_iso(
        iso,
        st.builds(
            m.KnowledgeRef,
            source=text,
            identifier=text,
            source_uri=opt,
            uri=opt,
            label=opt,
        ),
        max_examples=30,
    )
