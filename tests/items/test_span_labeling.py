"""Tests for span labeling item creation utilities."""

from __future__ import annotations

from uuid import uuid4

import didactic.api as dx
import pytest

from bead.items.item import Item
from bead.items.span_labeling import (
    add_spans_to_item,
    create_interactive_span_item,
    create_span_item,
    create_span_items_from_texts,
    tokenize_item,
)
from bead.items.spans import (
    Span,
    SpanLabel,
    SpanSegment,
)
from bead.tokenization.config import TokenizerConfig


class TestCreateSpanItem:
    """Test create_span_item() function."""

    def test_create_basic(self) -> None:
        """Test creating a basic span item."""
        spans = [
            Span(
                span_id="span_0",
                segments=[SpanSegment(element_name="text", indices=[0, 1])],
                label=SpanLabel(label="Person"),
            ),
        ]

        item = create_span_item(
            text="John Smith is here.",
            spans=spans,
            prompt="Identify the entities.",
            tokenizer_config=TokenizerConfig(backend="whitespace"),
        )

        assert isinstance(item, Item)
        assert item.rendered_elements["text"] == "John Smith is here."
        assert item.rendered_elements["prompt"] == "Identify the entities."
        assert len(item.spans) == 1
        assert item.tokenized_elements["text"] == (
            "John",
            "Smith",
            "is",
            "here.",
        )

    def test_with_pre_tokenized(self) -> None:
        """Test creating span item with pre-tokenized text."""
        tokens = ["John", "Smith", "is", "here", "."]
        spans = [
            Span(
                span_id="span_0",
                segments=[SpanSegment(element_name="text", indices=[0, 1])],
                label=SpanLabel(label="Person"),
            ),
        ]

        item = create_span_item(
            text="John Smith is here.",
            spans=spans,
            prompt="Identify the entities.",
            tokens=tokens,
        )

        assert item.tokenized_elements["text"] == tuple(tokens)

    def test_empty_text_raises(self) -> None:
        """Test that empty text raises error."""
        with pytest.raises(
            (ValueError, dx.ValidationError), match="text cannot be empty"
        ):
            create_span_item(text="", spans=[], prompt="Test")

    def test_invalid_span_index_raises(self) -> None:
        """Test that out-of-bounds span index raises error."""
        spans = [
            Span(
                span_id="span_0",
                segments=[SpanSegment(element_name="text", indices=[99])],
            ),
        ]

        with pytest.raises((ValueError, dx.ValidationError), match="index 99"):
            create_span_item(
                text="Short text.",
                spans=spans,
                prompt="Test",
                tokenizer_config=TokenizerConfig(backend="whitespace"),
            )

    def test_with_labels(self) -> None:
        """Test creating span item with label set."""
        item = create_span_item(
            text="The cat sat.",
            spans=[],
            prompt="Label spans.",
            labels=["Person", "Location"],
            tokenizer_config=TokenizerConfig(backend="whitespace"),
        )

        assert isinstance(item, Item)

    def test_with_metadata(self) -> None:
        """Test creating span item with metadata."""
        item = create_span_item(
            text="Hello world.",
            spans=[],
            prompt="Test",
            metadata={"source": "test"},
            tokenizer_config=TokenizerConfig(backend="whitespace"),
        )

        assert item.item_metadata["source"] == "test"


class TestCreateInteractiveSpanItem:
    """Test create_interactive_span_item() function."""

    def test_create_basic(self) -> None:
        """Test creating interactive span item."""
        item = create_interactive_span_item(
            text="The cat sat on the mat.",
            prompt="Select all entities.",
            tokenizer_config=TokenizerConfig(backend="whitespace"),
        )

        assert isinstance(item, Item)
        assert item.spans == ()  # No pre-defined spans
        assert "text" in item.tokenized_elements

    def test_with_label_set(self) -> None:
        """Test interactive item with fixed label set."""
        item = create_interactive_span_item(
            text="Hello world.",
            prompt="Select spans.",
            label_set=["PER", "ORG", "LOC"],
            tokenizer_config=TokenizerConfig(backend="whitespace"),
        )

        assert isinstance(item, Item)

    def test_empty_text_raises(self) -> None:
        """Test that empty text raises error."""
        with pytest.raises(
            (ValueError, dx.ValidationError), match="text cannot be empty"
        ):
            create_interactive_span_item(text="", prompt="Test")


class TestAddSpansToItem:
    """Test add_spans_to_item() function."""

    def test_add_to_ordinal_item(self) -> None:
        """Test adding spans to an ordinal scale item."""
        # Create base ordinal item
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"text": "The cat sat.", "prompt": "Rate this."},
            item_metadata={"scale_min": 1, "scale_max": 7},
        )

        spans = [
            Span(
                span_id="span_0",
                segments=[SpanSegment(element_name="text", indices=[1])],
                label=SpanLabel(label="Entity"),
            ),
        ]

        result = add_spans_to_item(
            item,
            spans,
            tokenizer_config=TokenizerConfig(backend="whitespace"),
        )

        assert len(result.spans) == 1
        assert result.item_metadata["scale_min"] == 1  # preserved
        assert result.rendered_elements["text"] == "The cat sat."  # preserved

    def test_add_to_already_tokenized(self) -> None:
        """Test adding spans to already tokenized item."""
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"text": "Hello world"},
            tokenized_elements={"text": ["Hello", "world"]},
            token_space_after={"text": [True, False]},
        )

        spans = [
            Span(
                span_id="span_0",
                segments=[SpanSegment(element_name="text", indices=[0])],
            ),
        ]

        result = add_spans_to_item(item, spans)

        assert len(result.spans) == 1
        # Token data preserved
        assert result.tokenized_elements["text"] == (
            "Hello",
            "world",
        )

    def test_preserves_existing_fields(self) -> None:
        """Test that adding spans preserves all existing fields."""
        template_id = uuid4()
        item = Item(
            item_template_id=template_id,
            rendered_elements={"text": "Test text"},
            options=["A", "B"],
            item_metadata={"key": "value"},
        )

        result = add_spans_to_item(
            item,
            spans=[],
            tokenizer_config=TokenizerConfig(backend="whitespace"),
        )

        assert result.item_template_id == template_id
        assert result.options == ("A", "B")
        assert result.item_metadata["key"] == "value"

    def test_invalid_span_raises(self) -> None:
        """Test that invalid span index raises error."""
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"text": "Hi"},
            tokenized_elements={"text": ["Hi"]},
        )

        spans = [
            Span(
                span_id="span_0",
                segments=[SpanSegment(element_name="text", indices=[99])],
            ),
        ]

        with pytest.raises((ValueError, dx.ValidationError), match="index 99"):
            add_spans_to_item(item, spans)


class TestTokenizeItem:
    """Test tokenize_item() function."""

    def test_whitespace_tokenizer(self) -> None:
        """Test tokenizing with whitespace backend."""
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"text": "Hello world"},
        )

        result = tokenize_item(item, TokenizerConfig(backend="whitespace"))

        assert result.tokenized_elements["text"] == (
            "Hello",
            "world",
        )
        assert result.token_space_after["text"] == (
            True,
            False,
        )

    def test_multiple_elements(self) -> None:
        """Test tokenizing item with multiple rendered elements."""
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={
                "context": "The cat sat.",
                "target": "The dog ran.",
            },
        )

        result = tokenize_item(item, TokenizerConfig(backend="whitespace"))

        assert "context" in result.tokenized_elements
        assert "target" in result.tokenized_elements
        assert result.tokenized_elements["context"] == (
            "The",
            "cat",
            "sat.",
        )
        assert result.tokenized_elements["target"] == (
            "The",
            "dog",
            "ran.",
        )

    def test_default_config(self) -> None:
        """Test tokenizing with default config."""
        pytest.importorskip("spacy")
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"text": "Hello"},
        )
        result = tokenize_item(item)
        assert "text" in result.tokenized_elements


class TestCreateSpanItemsFromTexts:
    """Test create_span_items_from_texts() function."""

    def test_batch_create(self) -> None:
        """Test batch creating span items."""

        def extractor(text: str, tokens: list[str]) -> list[Span]:
            return [
                Span(
                    span_id="span_0",
                    segments=[SpanSegment(element_name="text", indices=[0])],
                    label=SpanLabel(label="First"),
                ),
            ]

        items = create_span_items_from_texts(
            texts=["Hello world.", "Goodbye world."],
            span_extractor=extractor,
            prompt="Label first word.",
            tokenizer_config=TokenizerConfig(backend="whitespace"),
        )

        assert len(items) == 2
        assert all(len(item.spans) == 1 for item in items)
        assert items[0].rendered_elements["text"] == "Hello world."
        assert items[1].rendered_elements["text"] == "Goodbye world."
