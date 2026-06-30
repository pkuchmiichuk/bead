"""Round-trip law tests for the Item <-> layers annotation fragment lens."""

from __future__ import annotations

from uuid import uuid4

from lairs.records import annotation, expression, segmentation

from bead.interop.layers.item_bridge import ITEM_LAYERS, item_to_layers
from bead.items.item import Item
from bead.items.spans import Span, SpanLabel, SpanRelation, SpanSegment

LENS = ITEM_LAYERS


def _assert_roundtrip(item: Item) -> None:
    view, complement = LENS.forward(item)
    assert LENS.backward(view, complement) == item
    view2, complement2 = LENS.forward(LENS.backward(view, complement))
    assert (view2, complement2) == (view, complement)


def _span_layer(item: Item, element: str) -> annotation.AnnotationLayer:
    view = item_to_layers(item)
    record = next(
        record for record in view.records if record.local_id == f"spans:{element}"
    )
    return annotation.AnnotationLayer.model_validate_json(record.value_json)


def _relation_layer(item: Item) -> annotation.AnnotationLayer:
    view = item_to_layers(item)
    record = next(record for record in view.records if record.local_id == "relations")
    return annotation.AnnotationLayer.model_validate_json(record.value_json)


class TestExampleRoundTrips:
    """Deterministic round-trips over representative items."""

    def test_minimal(self) -> None:
        _assert_roundtrip(
            Item(item_template_id=uuid4(), rendered_elements={"text": "hi"})
        )

    def test_single_token_span(self) -> None:
        _assert_roundtrip(
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "The cat sat"},
                tokenized_elements={"text": ("The", "cat", "sat")},
                token_space_after={"text": (True, True, False)},
                spans=(
                    Span(
                        span_id="s1",
                        segments=(SpanSegment(element_name="text", indices=(1,)),),
                        label=SpanLabel(label="ANIMAL"),
                    ),
                ),
            )
        )

    def test_multi_element_item_with_metadata(self) -> None:
        _assert_roundtrip(
            Item(
                item_template_id=uuid4(),
                rendered_elements={"premise": "It rained", "hypothesis": "It was wet"},
                tokenized_elements={
                    "premise": ("It", "rained"),
                    "hypothesis": ("It", "was", "wet"),
                },
                token_space_after={"premise": (True, False)},
                options=("entailment", "neutral"),
                item_metadata={"pair_id": 7, "tags": ("nli", "weather")},
            )
        )


class TestViewProjection:
    """The faithful layers view carries the canonical anchors and arguments."""

    def test_discontiguous_span_uses_token_ref_sequence(self) -> None:
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"text": "The big red cat"},
            tokenized_elements={"text": ("The", "big", "red", "cat")},
            spans=(
                Span(
                    span_id="s1",
                    segments=(SpanSegment(element_name="text", indices=(0, 3)),),
                    head_index=3,
                    label=SpanLabel(label="NP"),
                ),
            ),
        )
        layer = _span_layer(item, "text")
        anchor = layer.annotations[0].anchor
        assert anchor is not None
        assert anchor.tokenRefSequence is not None
        assert anchor.tokenRefSequence.tokenIndexes == (0, 3)
        assert anchor.tokenRefSequence.anchorTokenIndex == 3

    def test_wikidata_label_id_becomes_knowledge_ref(self) -> None:
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"text": "Einstein"},
            tokenized_elements={"text": ("Einstein",)},
            spans=(
                Span(
                    span_id="s1",
                    segments=(SpanSegment(element_name="text", indices=(0,)),),
                    label=SpanLabel(label="PERSON", label_id="Q937", confidence=0.5),
                ),
            ),
        )
        layer = _span_layer(item, "text")
        refs = layer.annotations[0].knowledgeRefs or ()
        assert refs[0].source == "wikidata"
        assert refs[0].identifier == "Q937"
        assert layer.annotations[0].confidence == 500

    def test_relation_uses_argument_refs(self) -> None:
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"text": "Alice knows Bob"},
            tokenized_elements={"text": ("Alice", "knows", "Bob")},
            spans=(
                Span(
                    span_id="a",
                    segments=(SpanSegment(element_name="text", indices=(0,)),),
                ),
                Span(
                    span_id="b",
                    segments=(SpanSegment(element_name="text", indices=(2,)),),
                ),
            ),
            span_relations=(
                SpanRelation(
                    relation_id="r1",
                    source_span_id="a",
                    target_span_id="b",
                    label=SpanLabel(label="knows"),
                ),
            ),
        )
        layer = _relation_layer(item)
        arguments = layer.annotations[0].arguments or ()
        roles = {
            argument.role: argument.target.localId.value
            for argument in arguments
            if argument.target.localId is not None
        }
        assert roles == {"source": "a", "target": "b"}

    def test_records_validate_as_lairs_models(self) -> None:
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"text": "hi there"},
            tokenized_elements={"text": ("hi", "there")},
            token_space_after={"text": (True, False)},
        )
        view = item_to_layers(item)
        for record in view.records:
            if record.local_id.startswith("expression:"):
                expression.Expression.model_validate_json(record.value_json)
            elif record.local_id.startswith("segmentation:"):
                segmentation.Segmentation.model_validate_json(record.value_json)
