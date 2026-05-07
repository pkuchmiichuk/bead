"""Tests for transform integration with prompt reference resolution."""

from __future__ import annotations

from uuid import uuid4

import pytest

from bead.deployment.jspsych.config import SpanDisplayConfig
from bead.deployment.jspsych.trials import (
    SpanColorMap,
    _assign_span_colors,
    _build_transform_context,
    _resolve_prompt_references,
)
from bead.items.item import Item
from bead.items.spans import Span, SpanLabel, SpanSegment
from bead.labels import parse_label_refs
from bead.transforms.base import TransformRegistry


class TestParsePromptReferencesWithTransforms:
    """Tests for parse_label_refs() transform syntax."""

    def test_no_transforms(self) -> None:
        """Plain label has empty transforms list."""
        refs = parse_label_refs("[[agent]]")

        assert refs[0].transforms == ()

    def test_single_transform(self) -> None:
        """Single transform after pipe is captured."""
        refs = parse_label_refs("[[situation|gerund]]")

        assert refs[0].label == "situation"
        assert refs[0].display_text is None
        assert refs[0].transforms == ("gerund",)

    def test_multiple_transforms(self) -> None:
        """Chained transforms are split on pipe."""
        refs = parse_label_refs("[[situation|gerund|lower]]")

        assert refs[0].label == "situation"
        assert refs[0].transforms == ("gerund", "lower")

    def test_explicit_text_with_transform(self) -> None:
        """Display text and transforms can coexist."""
        refs = parse_label_refs("[[event:the running|upper]]")

        assert refs[0].label == "event"
        assert refs[0].display_text == "the running"
        assert refs[0].transforms == ("upper",)

    def test_backward_compatible_colon_syntax(self) -> None:
        """Existing [[label:text]] syntax still works."""
        refs = parse_label_refs("[[event:the breaking]]")

        assert refs[0].label == "event"
        assert refs[0].display_text == "the breaking"
        assert refs[0].transforms == ()

    def test_backward_compatible_plain_label(self) -> None:
        """Existing [[label]] syntax still works."""
        refs = parse_label_refs("[[agent]]")

        assert refs[0].label == "agent"
        assert refs[0].display_text is None
        assert refs[0].transforms == ()

    def test_mixed_references(self) -> None:
        """Various syntax forms in one prompt are parsed correctly."""
        prompt = "Did [[agent]] do [[event|gerund]] to [[patient:the vase|upper]]?"
        refs = parse_label_refs(prompt)

        assert len(refs) == 3

        assert refs[0].label == "agent"
        assert refs[0].transforms == ()

        assert refs[1].label == "event"
        assert refs[1].transforms == ("gerund",)

        assert refs[2].label == "patient"
        assert refs[2].display_text == "the vase"
        assert refs[2].transforms == ("upper",)


class TestResolvePromptReferencesWithTransforms:
    """Tests for _resolve_prompt_references() with transforms."""

    @pytest.fixture
    def span_item(self) -> Item:
        """Item with spans and tokenized elements."""
        return Item(
            item_template_id=uuid4(),
            rendered_elements={"text": "The boy broke the vase."},
            tokenized_elements={
                "text": ["The", "boy", "broke", "the", "vase", "."],
            },
            token_space_after={"text": [True, True, True, True, False, False]},
            spans=[
                Span(
                    span_id="span_0",
                    segments=[
                        SpanSegment(element_name="text", indices=[0, 1]),
                    ],
                    label=SpanLabel(label="breaker"),
                    head_index=1,
                ),
                Span(
                    span_id="span_1",
                    segments=[
                        SpanSegment(element_name="text", indices=[2]),
                    ],
                    label=SpanLabel(label="event"),
                    head_index=0,
                    span_metadata={"lemma": "break", "pos": "VERB"},
                ),
            ],
        )

    @pytest.fixture
    def color_map(self, span_item: Item) -> SpanColorMap:
        span_display = SpanDisplayConfig()
        return _assign_span_colors(span_item.spans, span_display)

    @pytest.fixture
    def registry(self) -> TransformRegistry:
        """Registry with text transforms only (no unimorph dependency)."""
        reg = TransformRegistry()
        reg.register("upper", lambda t, c: t.upper())
        reg.register("lower", lambda t, c: t.lower())
        reg.register("exclaim", lambda t, c: t + "!")
        return reg

    def test_no_transforms_no_registry(
        self,
        span_item: Item,
        color_map: SpanColorMap,
    ) -> None:
        """Without transforms, works exactly as before."""
        result = _resolve_prompt_references(
            "Did [[breaker]] do it?", span_item, color_map
        )

        assert "The boy" in result
        assert "bead-q-highlight" in result

    def test_transform_applied(
        self,
        span_item: Item,
        color_map: SpanColorMap,
        registry: TransformRegistry,
    ) -> None:
        """Transform modifies the display text."""
        result = _resolve_prompt_references(
            "Did [[breaker|upper]] do it?",
            span_item,
            color_map,
            transform_registry=registry,
        )

        assert "THE BOY" in result
        assert "bead-q-highlight" in result

    def test_chained_transforms(
        self,
        span_item: Item,
        color_map: SpanColorMap,
        registry: TransformRegistry,
    ) -> None:
        """Multiple transforms are applied in order."""
        result = _resolve_prompt_references(
            "Did [[breaker|upper|exclaim]] do it?",
            span_item,
            color_map,
            transform_registry=registry,
        )

        assert "THE BOY!" in result

    def test_explicit_text_with_transform(
        self,
        span_item: Item,
        color_map: SpanColorMap,
        registry: TransformRegistry,
    ) -> None:
        """Explicit text is transformed."""
        result = _resolve_prompt_references(
            "Did [[event:the breaking|upper]] happen?",
            span_item,
            color_map,
            transform_registry=registry,
        )

        assert "THE BREAKING" in result

    def test_transforms_ignored_without_registry(
        self,
        span_item: Item,
        color_map: SpanColorMap,
    ) -> None:
        """When no registry is provided, transforms are silently ignored."""
        result = _resolve_prompt_references(
            "Did [[breaker|upper]] do it?",
            span_item,
            color_map,
            transform_registry=None,
        )

        # should use the un-transformed text
        assert "The boy" in result

    def test_unknown_transform_raises(
        self,
        span_item: Item,
        color_map: SpanColorMap,
        registry: TransformRegistry,
    ) -> None:
        """Unknown transform name raises KeyError."""
        with pytest.raises(KeyError, match="nonexistent"):
            _resolve_prompt_references(
                "Did [[breaker|nonexistent]] do it?",
                span_item,
                color_map,
                transform_registry=registry,
            )


class TestBuildTransformContext:
    """Tests for _build_transform_context."""

    @pytest.fixture
    def item_with_metadata(self) -> Item:
        """Item with span metadata including lemma and pos."""
        return Item(
            item_template_id=uuid4(),
            rendered_elements={"text": "The boy ran quickly."},
            tokenized_elements={
                "text": ["The", "boy", "ran", "quickly", "."],
            },
            token_space_after={"text": [True, True, True, False, False]},
            spans=[
                Span(
                    span_id="s0",
                    segments=[SpanSegment(element_name="text", indices=[2])],
                    label=SpanLabel(label="event"),
                    head_index=0,
                    span_metadata={"lemma": "run", "pos": "VERB"},
                ),
                Span(
                    span_id="s1",
                    segments=[SpanSegment(element_name="text", indices=[0, 1])],
                    label=SpanLabel(label="agent"),
                ),
            ],
        )

    def test_extracts_lemma(self, item_with_metadata: Item) -> None:
        ctx = _build_transform_context("event", item_with_metadata)

        assert ctx.lemma == "run"

    def test_extracts_pos(self, item_with_metadata: Item) -> None:
        ctx = _build_transform_context("event", item_with_metadata)

        assert ctx.pos == "VERB"

    def test_extracts_tokens(self, item_with_metadata: Item) -> None:
        ctx = _build_transform_context("event", item_with_metadata)

        assert ctx.tokens == ("ran",)

    def test_extracts_head_index(self, item_with_metadata: Item) -> None:
        ctx = _build_transform_context("event", item_with_metadata)

        assert ctx.head_index == 0

    def test_no_metadata_span(self, item_with_metadata: Item) -> None:
        """Span without metadata still produces a context."""
        ctx = _build_transform_context("agent", item_with_metadata)

        assert ctx.lemma is None
        assert ctx.pos is None
        assert ctx.tokens == ("The", "boy")

    def test_missing_label(self, item_with_metadata: Item) -> None:
        """Missing label returns empty context."""
        ctx = _build_transform_context("nonexistent", item_with_metadata)

        assert ctx.lemma is None
        assert ctx.tokens == ()
