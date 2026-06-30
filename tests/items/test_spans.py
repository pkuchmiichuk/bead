"""Tests for span annotation models."""

from __future__ import annotations

from uuid import uuid4

import didactic.api as dx
import pytest

from bead.items.item import Item
from bead.items.spans import (
    Span,
    SpanLabel,
    SpanRelation,
    SpanSegment,
    SpanSpec,
)


class TestSpanSegment:
    """Tests for SpanSegment."""

    def test_create(self) -> None:
        segment = SpanSegment(element_name="text", indices=(0, 1, 2))
        assert segment.element_name == "text"
        assert segment.indices == (0, 1, 2)

    def test_empty_element_name_raises(self) -> None:
        with pytest.raises(dx.ValidationError, match="element_name cannot be empty"):
            SpanSegment(element_name="", indices=(0,))

    def test_empty_indices_raises(self) -> None:
        with pytest.raises(dx.ValidationError, match="indices cannot be empty"):
            SpanSegment(element_name="text", indices=())

    def test_negative_indices_raises(self) -> None:
        with pytest.raises(dx.ValidationError, match="indices must be non-negative"):
            SpanSegment(element_name="text", indices=(-1, 0))

    def test_discontiguous_indices(self) -> None:
        segment = SpanSegment(element_name="text", indices=(0, 2, 5))
        assert segment.indices == (0, 2, 5)


class TestSpanLabel:
    """Tests for SpanLabel."""

    def test_create_basic(self) -> None:
        label = SpanLabel(label="Person")
        assert label.label == "Person"
        assert label.label_id is None
        assert label.confidence is None

    def test_create_with_id(self) -> None:
        label = SpanLabel(label="human", label_id="Q5")
        assert label.label == "human"
        assert label.label_id == "Q5"

    def test_create_with_confidence(self) -> None:
        label = SpanLabel(label="Person", confidence=0.95)
        assert label.confidence == 0.95

    def test_empty_label_raises(self) -> None:
        with pytest.raises(dx.ValidationError, match="label cannot be empty"):
            SpanLabel(label="")


class TestSpan:
    """Tests for Span."""

    def test_create_basic(self) -> None:
        span = Span(
            span_id="span_0",
            segments=(SpanSegment(element_name="text", indices=(0, 1)),),
        )
        assert span.span_id == "span_0"
        assert len(span.segments) == 1
        assert span.label is None
        assert span.head_index is None

    def test_create_with_label(self) -> None:
        span = Span(
            span_id="span_0",
            segments=(SpanSegment(element_name="text", indices=(0, 1)),),
            label=SpanLabel(label="Person"),
        )
        assert span.label is not None
        assert span.label.label == "Person"

    def test_discontiguous_segments(self) -> None:
        span = Span(
            span_id="span_0",
            segments=(
                SpanSegment(element_name="text", indices=(0, 1)),
                SpanSegment(element_name="text", indices=(5, 6)),
            ),
        )
        assert len(span.segments) == 2

    def test_cross_element_segments(self) -> None:
        span = Span(
            span_id="span_0",
            segments=(
                SpanSegment(element_name="context", indices=(0, 1)),
                SpanSegment(element_name="target", indices=(2, 3)),
            ),
        )
        assert span.segments[0].element_name == "context"
        assert span.segments[1].element_name == "target"

    def test_with_metadata(self) -> None:
        span = Span(
            span_id="span_0",
            segments=(SpanSegment(element_name="text", indices=(0,)),),
            span_metadata={"source": "manual"},
        )
        assert span.span_metadata["source"] == "manual"

    def test_empty_span_id_raises(self) -> None:
        with pytest.raises(dx.ValidationError, match="span_id cannot be empty"):
            Span(span_id="")


class TestSpanRelation:
    """Tests for SpanRelation."""

    def test_create_directed(self) -> None:
        rel = SpanRelation(
            relation_id="rel_0",
            source_span_id="span_0",
            target_span_id="span_1",
            label=SpanLabel(label="agent-of"),
        )
        assert rel.relation_id == "rel_0"
        assert rel.directed is True
        assert rel.label is not None
        assert rel.label.label == "agent-of"

    def test_create_undirected(self) -> None:
        rel = SpanRelation(
            relation_id="rel_0",
            source_span_id="span_0",
            target_span_id="span_1",
            directed=False,
        )
        assert rel.directed is False

    def test_with_wikidata_label(self) -> None:
        rel = SpanRelation(
            relation_id="rel_0",
            source_span_id="span_0",
            target_span_id="span_1",
            label=SpanLabel(label="instance of", label_id="P31"),
        )
        assert rel.label is not None
        assert rel.label.label_id == "P31"

    def test_empty_relation_id_raises(self) -> None:
        with pytest.raises(dx.ValidationError, match="relation_id cannot be empty"):
            SpanRelation(
                relation_id="",
                source_span_id="span_0",
                target_span_id="span_1",
            )

    def test_empty_span_id_raises(self) -> None:
        with pytest.raises(dx.ValidationError, match="span ID cannot be empty"):
            SpanRelation(
                relation_id="rel_0",
                source_span_id="",
                target_span_id="span_1",
            )


class TestSpanOnItem:
    """Tests for the span fields on Item."""

    def test_item_with_no_spans(self) -> None:
        item = Item(item_template_id=uuid4())
        assert item.spans == ()
        assert item.span_relations == ()
        assert item.tokenized_elements == {}
        assert item.token_space_after == {}

    def test_item_with_spans(self) -> None:
        span = Span(
            span_id="span_0",
            segments=(SpanSegment(element_name="text", indices=(0, 1)),),
            label=SpanLabel(label="Person"),
        )
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"text": "The cat"},
            spans=(span,),
            tokenized_elements={"text": ("The", "cat")},
            token_space_after={"text": (True, False)},
        )
        assert len(item.spans) == 1
        assert item.spans[0].span_id == "span_0"
        assert item.tokenized_elements["text"] == ("The", "cat")

    def test_item_with_relations(self) -> None:
        spans = (
            Span(
                span_id="span_0",
                segments=(SpanSegment(element_name="text", indices=(0,)),),
            ),
            Span(
                span_id="span_1",
                segments=(SpanSegment(element_name="text", indices=(2,)),),
            ),
        )
        rel = SpanRelation(
            relation_id="rel_0",
            source_span_id="span_0",
            target_span_id="span_1",
            label=SpanLabel(label="agent-of"),
        )
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"text": "The cat chased the mouse"},
            spans=spans,
            span_relations=(rel,),
            tokenized_elements={
                "text": ("The", "cat", "chased", "the", "mouse"),
            },
        )
        assert len(item.span_relations) == 1
        assert item.span_relations[0].source_span_id == "span_0"

    def test_serialization_round_trip(self) -> None:
        span = Span(
            span_id="span_0",
            segments=(SpanSegment(element_name="text", indices=(0, 1)),),
            label=SpanLabel(label="Person", label_id="Q5"),
        )
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"text": "John Smith"},
            spans=(span,),
            tokenized_elements={"text": ("John", "Smith")},
            token_space_after={"text": (True, False)},
        )

        restored = Item.model_validate_json(item.model_dump_json())
        assert len(restored.spans) == 1
        assert restored.spans[0].span_id == "span_0"
        assert restored.spans[0].label is not None
        assert restored.spans[0].label.label == "Person"
        assert restored.spans[0].label.label_id == "Q5"
        assert restored.tokenized_elements == {"text": ("John", "Smith")}
        assert restored.token_space_after == {"text": (True, False)}


class TestSpanSpec:
    """Tests for SpanSpec."""

    def test_default_values(self) -> None:
        spec = SpanSpec()
        assert spec.index_mode == "token"
        assert spec.interaction_mode == "static"
        assert spec.label_source == "fixed"
        assert spec.labels is None
        assert spec.allow_overlapping is True
        assert spec.enable_relations is False
        assert spec.wikidata_language == "en"
        assert spec.wikidata_result_limit == 10

    def test_interactive_with_labels(self) -> None:
        spec = SpanSpec(
            interaction_mode="interactive",
            label_source="fixed",
            labels=("Person", "Organization", "Location"),
            min_spans=1,
            max_spans=10,
        )
        assert spec.interaction_mode == "interactive"
        assert spec.labels == ("Person", "Organization", "Location")
        assert spec.min_spans == 1
        assert spec.max_spans == 10

    def test_wikidata_config(self) -> None:
        spec = SpanSpec(
            label_source="wikidata",
            wikidata_language="de",
            wikidata_entity_types=("item",),
            wikidata_result_limit=20,
        )
        assert spec.label_source == "wikidata"
        assert spec.wikidata_language == "de"
        assert spec.wikidata_entity_types == ("item",)

    def test_relation_config(self) -> None:
        spec = SpanSpec(
            enable_relations=True,
            relation_label_source="fixed",
            relation_labels=("agent-of", "patient-of"),
            relation_directed=True,
            min_relations=0,
            max_relations=5,
        )
        assert spec.enable_relations is True
        assert spec.relation_labels == ("agent-of", "patient-of")
        assert spec.relation_directed is True

    def test_label_colors(self) -> None:
        spec = SpanSpec(
            labels=("PER", "ORG"),
            label_colors={"PER": "#FF0000", "ORG": "#00FF00"},
        )
        assert spec.label_colors == {"PER": "#FF0000", "ORG": "#00FF00"}
