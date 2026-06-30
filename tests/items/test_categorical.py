"""Tests for categorical item creation utilities."""

from __future__ import annotations

from uuid import uuid4

import didactic.api as dx
import pytest

from bead.items.categorical import (
    create_categorical_item,
    create_categorical_items_cross_product,
    create_categorical_items_from_groups,
    create_categorical_items_from_pairs,
    create_categorical_items_from_texts,
    create_filtered_categorical_items,
    create_nli_item,
)
from bead.items.item import Item


class TestCreateCategoricalItem:
    """Test create_categorical_item() function."""

    def test_create_basic_categorical_item(self) -> None:
        """Test creating a basic categorical item."""
        item = create_categorical_item(
            text="The cat sat on the mat.",
            categories=["past", "present", "future"],
            prompt="What is the tense?",
        )

        assert isinstance(item, Item)
        assert item.rendered_elements["text"] == "The cat sat on the mat."
        assert item.rendered_elements["prompt"] == "What is the tense?"
        assert item.item_metadata["categories"] == (
            "past",
            "present",
            "future",
        )

    def test_default_prompt(self) -> None:
        """Test default prompt."""
        item = create_categorical_item("Text", categories=["A", "B", "C"])

        assert item.rendered_elements["prompt"] == "Select a category:"

    def test_empty_text_raises_error(self) -> None:
        """Test that empty text raises error."""
        with pytest.raises(
            (ValueError, dx.ValidationError), match="text cannot be empty"
        ):
            create_categorical_item("", categories=["A", "B"])

        with pytest.raises(
            (ValueError, dx.ValidationError), match="text cannot be empty"
        ):
            create_categorical_item("   ", categories=["A", "B"])

    def test_too_few_categories_raises_error(self) -> None:
        """Test that fewer than 2 categories raises error."""
        with pytest.raises(
            ValueError, match="At least 2 categories required for categorical item"
        ):
            create_categorical_item("Text", categories=["Only one"])

    def test_with_custom_template_id(self) -> None:
        """Test creating item with custom template ID."""
        template_id = uuid4()
        item = create_categorical_item(
            "Text", categories=["A", "B"], item_template_id=template_id
        )

        assert item.item_template_id == template_id

    def test_with_metadata(self) -> None:
        """Test creating item with metadata."""
        item = create_categorical_item(
            "Text",
            categories=["A", "B", "C"],
            metadata={"task": "classification", "language": "en"},
        )

        assert item.item_metadata["task"] == "classification"
        assert item.item_metadata["language"] == "en"
        assert item.item_metadata["categories"] == (
            "A",
            "B",
            "C",
        )

    def test_nli_example(self) -> None:
        """Test NLI classification example."""
        item = create_categorical_item(
            text="Premise: All dogs bark. Hypothesis: Some dogs bark.",
            categories=["entailment", "neutral", "contradiction"],
            prompt="What is the relationship?",
            metadata={"task": "nli"},
        )

        assert len(item.item_metadata["categories"]) == 3
        assert item.item_metadata["task"] == "nli"


class TestCreateNliItem:
    """Test create_nli_item() specialized function."""

    def test_basic_nli_item(self) -> None:
        """Test creating basic NLI item."""
        item = create_nli_item(premise="All dogs bark.", hypothesis="Some dogs bark.")

        assert "Premise:" in item.rendered_elements["text"]
        assert "Hypothesis:" in item.rendered_elements["text"]
        assert item.item_metadata["premise"] == "All dogs bark."
        assert item.item_metadata["hypothesis"] == "Some dogs bark."
        assert item.item_metadata["categories"] == (
            "entailment",
            "neutral",
            "contradiction",
        )
        assert item.item_metadata["task"] == "nli"

    def test_default_prompt(self) -> None:
        """Test default NLI prompt."""
        item = create_nli_item(premise="P", hypothesis="H")

        assert item.rendered_elements["prompt"] == "What is the relationship?"

    def test_custom_categories(self) -> None:
        """Test NLI with custom categories."""
        item = create_nli_item(
            premise="P",
            hypothesis="H",
            categories=["entails", "contradicts", "neither"],
        )

        assert item.item_metadata["categories"] == (
            "entails",
            "contradicts",
            "neither",
        )

    def test_custom_prompt(self) -> None:
        """Test NLI with custom prompt."""
        item = create_nli_item(
            premise="P",
            hypothesis="H",
            prompt="Does the premise entail the hypothesis?",
        )

        assert (
            item.rendered_elements["prompt"]
            == "Does the premise entail the hypothesis?"
        )


class TestCreateCategoricalItemsFromTexts:
    """Test create_categorical_items_from_texts() function."""

    def test_basic_batch_creation(self) -> None:
        """Test basic batch creation from texts."""
        texts = ["The cat sat.", "The dog ran.", "The bird flew."]
        categories = ["past", "present", "future"]

        items = create_categorical_items_from_texts(
            texts, categories=categories, prompt="What is the tense?"
        )

        assert len(items) == 3
        assert all(isinstance(item, Item) for item in items)
        assert items[0].rendered_elements["text"] == "The cat sat."
        assert all(
            item.item_metadata["categories"] == tuple(categories) for item in items
        )

    def test_with_metadata_function(self) -> None:
        """Test with metadata function."""
        texts = ["Sentence 1", "Sentence 2"]

        items = create_categorical_items_from_texts(
            texts,
            categories=["A", "B"],
            prompt="Classify:",
            metadata_fn=lambda text: {"text_length": len(text)},
        )

        assert items[0].item_metadata["text_length"] == len("Sentence 1")
        assert items[1].item_metadata["text_length"] == len("Sentence 2")


class TestCreateCategoricalItemsFromPairs:
    """Test create_categorical_items_from_pairs() function."""

    def test_basic_pair_creation(self) -> None:
        """Test creating items from pairs."""
        pairs = [
            ("All dogs bark.", "Some dogs bark."),
            ("The sky is blue.", "The sky is not blue."),
        ]

        items = create_categorical_items_from_pairs(
            pairs,
            categories=["entailment", "neutral", "contradiction"],
            prompt="What is the relationship?",
            pair_label1="Premise",
            pair_label2="Hypothesis",
        )

        assert len(items) == 2
        assert "Premise:" in items[0].rendered_elements["text"]
        assert "Hypothesis:" in items[0].rendered_elements["text"]
        assert items[0].item_metadata["text1"] == pairs[0][0]
        assert items[0].item_metadata["text2"] == pairs[0][1]

    def test_default_labels(self) -> None:
        """Test default pair labels."""
        pairs = [("Text 1", "Text 2")]

        items = create_categorical_items_from_pairs(
            pairs, categories=["A", "B"], prompt="Question?"
        )

        assert "Text 1:" in items[0].rendered_elements["text"]
        assert "Text 2:" in items[0].rendered_elements["text"]


class TestCreateCategoricalItemsFromGroups:
    """Test create_categorical_items_from_groups() function."""

    def test_basic_grouping(self) -> None:
        """Test basic grouping with preservation of group info."""
        source_items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "The cat sat."},
                item_metadata={"tense": "past"},
            ),
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "The dog runs."},
                item_metadata={"tense": "present"},
            ),
        ]

        categorical_items = create_categorical_items_from_groups(
            source_items,
            group_by=lambda i: i.item_metadata["tense"],
            categories=["past", "present", "future"],
            prompt="What is the tense?",
        )

        assert len(categorical_items) == 2
        assert all("group_key" in item.item_metadata for item in categorical_items)
        assert all("source_item_id" in item.item_metadata for item in categorical_items)


class TestCreateCategoricalItemsCrossProduct:
    """Test create_categorical_items_cross_product() function."""

    def test_basic_cross_product(self) -> None:
        """Test basic cross-product of texts and prompts."""
        texts = ["The cat sat.", "The dog ran."]
        prompts = ["What is the tense?", "What is the aspect?"]
        categories = ["past", "present", "future"]

        items = create_categorical_items_cross_product(texts, prompts, categories)

        # 2 texts × 2 prompts = 4 items
        assert len(items) == 4
        assert items[0].rendered_elements["text"] == "The cat sat."
        assert items[0].rendered_elements["prompt"] == "What is the tense?"
        assert all(
            item.item_metadata["categories"] == tuple(categories) for item in items
        )


class TestCreateFilteredCategoricalItems:
    """Test create_filtered_categorical_items() function."""

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

        categorical_items = create_filtered_categorical_items(
            items,
            categories=["A", "B"],
            prompt="Classify:",
            item_filter=lambda i: i.item_metadata.get("valid", True),
        )

        assert len(categorical_items) == 1
        assert categorical_items[0].rendered_elements["text"] == "Valid"

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

        categorical_items = create_filtered_categorical_items(
            items, categories=["A", "B", "C"], prompt="Question?"
        )

        assert len(categorical_items) == 3
