"""Tests for binary item creation utilities."""

from __future__ import annotations

from uuid import uuid4

import didactic.api as dx
import pytest

from bead.items.binary import (
    create_binary_item,
    create_binary_items_cross_product,
    create_binary_items_from_groups,
    create_binary_items_from_texts,
    create_binary_items_with_context,
    create_filtered_binary_items,
)
from bead.items.item import Item


class TestCreateBinaryItem:
    """Test create_binary_item() function."""

    def test_create_basic_binary_item(self) -> None:
        """Test creating a basic binary item."""
        item = create_binary_item(
            "The cat sat on the mat.", prompt="Is this grammatical?"
        )

        assert isinstance(item, Item)
        assert item.rendered_elements["text"] == "The cat sat on the mat."
        assert item.rendered_elements["prompt"] == "Is this grammatical?"
        assert item.item_metadata["binary_options"] == (
            "yes",
            "no",
        )

    def test_default_prompt(self) -> None:
        """Test default prompt."""
        item = create_binary_item("The cat sat.")

        assert item.rendered_elements["prompt"] == "Yes/No?"

    def test_custom_binary_options(self) -> None:
        """Test custom binary options."""
        item = create_binary_item(
            "The sky is blue.",
            prompt="Is this true?",
            binary_options=("true", "false"),
        )

        assert item.item_metadata["binary_options"] == (
            "true",
            "false",
        )

    def test_empty_text_raises_error(self) -> None:
        """Test that empty text raises error."""
        with pytest.raises(
            (ValueError, dx.ValidationError), match="text cannot be empty"
        ):
            create_binary_item("")

        with pytest.raises(
            (ValueError, dx.ValidationError), match="text cannot be empty"
        ):
            create_binary_item("   ")

    def test_invalid_binary_options_raises_error(self) -> None:
        """Test that invalid binary_options raises error."""
        with pytest.raises(
            (ValueError, dx.ValidationError),
            match="binary_options must contain exactly 2",
        ):
            create_binary_item("Text", binary_options=("yes",))

        with pytest.raises(
            (ValueError, dx.ValidationError),
            match="binary_options must contain exactly 2",
        ):
            create_binary_item("Text", binary_options=("yes", "no", "maybe"))

    def test_with_custom_template_id(self) -> None:
        """Test creating item with custom template ID."""
        template_id = uuid4()
        item = create_binary_item("Text", item_template_id=template_id)

        assert item.item_template_id == template_id

    def test_with_metadata(self) -> None:
        """Test creating item with metadata."""
        item = create_binary_item(
            "Text", metadata={"judgment": "grammaticality", "verb": "walk"}
        )

        assert item.item_metadata["judgment"] == "grammaticality"
        assert item.item_metadata["verb"] == "walk"
        assert item.item_metadata["binary_options"] == (
            "yes",
            "no",
        )


class TestCreateBinaryItemsFromTexts:
    """Test create_binary_items_from_texts() function."""

    def test_basic_batch_creation(self) -> None:
        """Test basic batch creation from texts."""
        texts = ["She walks.", "She walk.", "They walk.", "They walks."]

        items = create_binary_items_from_texts(
            texts, prompt="Is this grammatical?", binary_options=("yes", "no")
        )

        assert len(items) == 4
        assert all(isinstance(item, Item) for item in items)
        assert items[0].rendered_elements["text"] == "She walks."
        assert items[0].rendered_elements["prompt"] == "Is this grammatical?"

    def test_with_metadata_function(self) -> None:
        """Test with metadata function."""
        texts = ["Sentence 1", "Sentence 2"]

        items = create_binary_items_from_texts(
            texts,
            prompt="Is this valid?",
            metadata_fn=lambda text: {"text_length": len(text)},
        )

        assert items[0].item_metadata["text_length"] == len("Sentence 1")
        assert items[1].item_metadata["text_length"] == len("Sentence 2")


class TestCreateBinaryItemsWithContext:
    """Test create_binary_items_with_context() function."""

    def test_basic_context_target(self) -> None:
        """Test creating items with context and target."""
        contexts = ["The dog barked loudly."]
        targets = ["The dog made a sound."]

        items = create_binary_items_with_context(
            contexts,
            targets,
            prompt="Is the statement true given the context?",
            binary_options=("true", "false"),
        )

        assert len(items) == 1
        assert "Context:" in items[0].rendered_elements["text"]
        assert "Statement:" in items[0].rendered_elements["text"]
        assert items[0].item_metadata["context"] == contexts[0]
        assert items[0].item_metadata["target"] == targets[0]

    def test_custom_labels(self) -> None:
        """Test custom context and target labels."""
        contexts = ["Premise text"]
        targets = ["Hypothesis text"]

        items = create_binary_items_with_context(
            contexts,
            targets,
            prompt="Does the premise support the hypothesis?",
            context_label="Premise",
            target_label="Hypothesis",
        )

        assert "Premise:" in items[0].rendered_elements["text"]
        assert "Hypothesis:" in items[0].rendered_elements["text"]

    def test_mismatched_lengths_raises_error(self) -> None:
        """Test that mismatched lengths raise error."""
        contexts = ["Context 1", "Context 2"]
        targets = ["Target 1"]

        with pytest.raises(
            ValueError, match="contexts and targets must have same length"
        ):
            create_binary_items_with_context(contexts, targets, prompt="Question?")


class TestCreateBinaryItemsFromGroups:
    """Test create_binary_items_from_groups() function."""

    def test_basic_grouping(self) -> None:
        """Test basic grouping with preservation of group info."""
        source_items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "She walks."},
                item_metadata={"verb": "walk"},
            ),
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "She runs."},
                item_metadata={"verb": "run"},
            ),
        ]

        binary_items = create_binary_items_from_groups(
            source_items,
            group_by=lambda i: i.item_metadata["verb"],
            prompt="Is this grammatical?",
        )

        assert len(binary_items) == 2
        assert all("group_key" in item.item_metadata for item in binary_items)
        assert all("source_item_id" in item.item_metadata for item in binary_items)


class TestCreateBinaryItemsCrossProduct:
    """Test create_binary_items_cross_product() function."""

    def test_basic_cross_product(self) -> None:
        """Test basic cross-product of texts and prompts."""
        texts = ["The cat sat.", "The dog ran."]
        prompts = ["Is this grammatical?", "Is this natural?"]

        items = create_binary_items_cross_product(texts, prompts)

        # 2 texts × 2 prompts = 4 items
        assert len(items) == 4
        assert items[0].rendered_elements["text"] == "The cat sat."
        assert items[0].rendered_elements["prompt"] == "Is this grammatical?"


class TestCreateFilteredBinaryItems:
    """Test create_filtered_binary_items() function."""

    def test_basic_filtering(self) -> None:
        """Test basic item filtering."""
        items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "Valid"},
                item_metadata={"valid": True},
            ),
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "Invalid"},
                item_metadata={"valid": False},
            ),
        ]

        binary_items = create_filtered_binary_items(
            items,
            prompt="Is this valid?",
            item_filter=lambda i: i.item_metadata.get("valid", True),
        )

        assert len(binary_items) == 1
        assert binary_items[0].rendered_elements["text"] == "Valid"

    def test_no_filter_includes_all(self) -> None:
        """Test that no filter includes all items."""
        items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": f"Text {i}"},
                item_metadata={},
            )
            for i in range(3)
        ]

        binary_items = create_filtered_binary_items(items, prompt="Question?")

        assert len(binary_items) == 3
