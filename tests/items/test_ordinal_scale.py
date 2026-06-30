"""Tests for ordinal scale item creation utilities."""

from __future__ import annotations

from uuid import uuid4

import didactic.api as dx
import pytest

from bead.items.item import Item
from bead.items.item_template import ScaleBounds, ScalePointLabel
from bead.items.ordinal_scale import (
    create_filtered_ordinal_scale_items,
    create_likert_5_item,
    create_likert_7_item,
    create_ordinal_scale_item,
    create_ordinal_scale_items_cross_product,
    create_ordinal_scale_items_from_groups,
    create_ordinal_scale_items_from_texts,
)


class TestCreateOrdinalScaleItem:
    """Test create_ordinal_scale_item() function."""

    def test_create_basic_ordinal_item(self) -> None:
        """Test creating a basic ordinal scale item."""
        item = create_ordinal_scale_item(
            "The cat sat on the mat.", scale_bounds=ScaleBounds(min=1, max=7)
        )

        assert isinstance(item, Item)
        assert item.rendered_elements["text"] == "The cat sat on the mat."
        assert item.rendered_elements["prompt"] == "Rate this item:"
        assert item.item_metadata["scale_min"] == 1
        assert item.item_metadata["scale_max"] == 7

    def test_default_prompt(self) -> None:
        """Test default prompt."""
        item = create_ordinal_scale_item(
            "The cat sat.", scale_bounds=ScaleBounds(min=1, max=5)
        )

        assert item.rendered_elements["prompt"] == "Rate this item:"

    def test_custom_prompt(self) -> None:
        """Test custom prompt."""
        item = create_ordinal_scale_item(
            "The cat sat.",
            scale_bounds=ScaleBounds(min=1, max=7),
            prompt="How natural is this sentence?",
        )

        assert item.rendered_elements["prompt"] == "How natural is this sentence?"

    def test_scale_labels(self) -> None:
        """Test scale labels."""
        item = create_ordinal_scale_item(
            "The sky is blue.",
            scale_bounds=ScaleBounds(min=1, max=5),
            scale_labels=(
                ScalePointLabel(point=1, label="Very Bad"),
                ScalePointLabel(point=5, label="Very Good"),
            ),
        )

        assert item.item_metadata["scale_labels"]["1"] == "Very Bad"
        assert item.item_metadata["scale_labels"]["5"] == "Very Good"

    def test_empty_text_raises_error(self) -> None:
        """Test that empty text raises error."""
        with pytest.raises(
            (ValueError, dx.ValidationError), match="text cannot be empty"
        ):
            create_ordinal_scale_item("", scale_bounds=ScaleBounds(min=1, max=7))

        with pytest.raises(
            (ValueError, dx.ValidationError), match="text cannot be empty"
        ):
            create_ordinal_scale_item("   ", scale_bounds=ScaleBounds(min=1, max=7))

    def test_invalid_scale_bounds_raises_error(self) -> None:
        """Test that invalid scale bounds raise error."""
        # min >= max
        with pytest.raises(
            (ValueError, dx.ValidationError), match="scale_min.*must be less than"
        ):
            create_ordinal_scale_item("Text", scale_bounds=ScaleBounds(min=5, max=5))

        with pytest.raises(
            (ValueError, dx.ValidationError), match="scale_min.*must be less than"
        ):
            create_ordinal_scale_item("Text", scale_bounds=ScaleBounds(min=7, max=3))

    def test_scale_labels_outside_bounds_raises_error(self) -> None:
        """Test that scale labels outside bounds raise error."""
        with pytest.raises(
            (ValueError, dx.ValidationError),
            match="scale_labels key.*outside scale bounds",
        ):
            create_ordinal_scale_item(
                "Text",
                scale_bounds=ScaleBounds(min=1, max=5),
                scale_labels=(
                    ScalePointLabel(point=0, label="Too Low"),
                    ScalePointLabel(point=5, label="Good"),
                ),
            )

        with pytest.raises(
            (ValueError, dx.ValidationError),
            match="scale_labels key.*outside scale bounds",
        ):
            create_ordinal_scale_item(
                "Text",
                scale_bounds=ScaleBounds(min=1, max=5),
                scale_labels=(
                    ScalePointLabel(point=1, label="Low"),
                    ScalePointLabel(point=6, label="Too High"),
                ),
            )

    def test_with_custom_template_id(self) -> None:
        """Test creating item with custom template ID."""
        template_id = uuid4()
        item = create_ordinal_scale_item(
            "Text", scale_bounds=ScaleBounds(min=1, max=7), item_template_id=template_id
        )

        assert item.item_template_id == template_id

    def test_with_metadata(self) -> None:
        """Test creating item with metadata."""
        item = create_ordinal_scale_item(
            "Text",
            scale_bounds=ScaleBounds(min=1, max=7),
            metadata={"task": "acceptability", "verb": "walk"},
        )

        assert item.item_metadata["task"] == "acceptability"
        assert item.item_metadata["verb"] == "walk"
        assert item.item_metadata["scale_min"] == 1


class TestCreateOrdinalScaleItemsFromTexts:
    """Test create_ordinal_scale_items_from_texts() function."""

    def test_basic_batch_creation(self) -> None:
        """Test basic batch creation from texts."""
        texts = ["She walks.", "She walk.", "They walk.", "They walks."]

        items = create_ordinal_scale_items_from_texts(
            texts,
            scale_bounds=ScaleBounds(min=1, max=5),
            prompt="Rate the acceptability:",
        )

        assert len(items) == 4
        assert all(isinstance(item, Item) for item in items)
        assert items[0].rendered_elements["text"] == "She walks."
        assert items[0].rendered_elements["prompt"] == "Rate the acceptability:"
        assert items[0].item_metadata["scale_min"] == 1
        assert items[0].item_metadata["scale_max"] == 5

    def test_with_metadata_function(self) -> None:
        """Test with metadata function."""
        texts = ["Sentence 1", "Sentence 2"]

        items = create_ordinal_scale_items_from_texts(
            texts,
            scale_bounds=ScaleBounds(min=1, max=7),
            prompt="Rate this:",
            metadata_fn=lambda text: {"text_length": len(text)},
        )

        assert items[0].item_metadata["text_length"] == len("Sentence 1")
        assert items[1].item_metadata["text_length"] == len("Sentence 2")

    def test_with_scale_labels(self) -> None:
        """Test batch creation with scale labels."""
        texts = ["Text 1", "Text 2"]
        labels = {1: "Bad", 7: "Good"}

        items = create_ordinal_scale_items_from_texts(
            texts, scale_bounds=ScaleBounds(min=1, max=7), scale_labels=labels
        )

        assert all("scale_labels" in item.item_metadata for item in items)
        assert items[0].item_metadata["scale_labels"]["1"] == "Bad"


class TestCreateOrdinalScaleItemsFromGroups:
    """Test create_ordinal_scale_items_from_groups() function."""

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

        ordinal_items = create_ordinal_scale_items_from_groups(
            source_items,
            group_by=lambda i: i.item_metadata["verb"],
            scale_bounds=ScaleBounds(min=1, max=7),
            prompt="Rate the acceptability:",
        )

        assert len(ordinal_items) == 2
        assert all("group_key" in item.item_metadata for item in ordinal_items)
        assert all("source_item_id" in item.item_metadata for item in ordinal_items)
        assert all(item.item_metadata["scale_min"] == 1 for item in ordinal_items)

    def test_without_group_metadata(self) -> None:
        """Test grouping without including group metadata."""
        source_items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "Text"},
                item_metadata={"group": "A"},
            )
        ]

        ordinal_items = create_ordinal_scale_items_from_groups(
            source_items,
            group_by=lambda i: i.item_metadata["group"],
            scale_bounds=ScaleBounds(min=1, max=5),
            include_group_metadata=False,
        )

        assert "group_key" not in ordinal_items[0].item_metadata
        assert "source_item_id" in ordinal_items[0].item_metadata


class TestCreateOrdinalScaleItemsCrossProduct:
    """Test create_ordinal_scale_items_cross_product() function."""

    def test_basic_cross_product(self) -> None:
        """Test basic cross-product of texts and prompts."""
        texts = ["The cat sat.", "The dog ran."]
        prompts = ["How natural is this?", "How acceptable is this?"]

        items = create_ordinal_scale_items_cross_product(
            texts, prompts, scale_bounds=ScaleBounds(min=1, max=7)
        )

        # 2 texts × 2 prompts = 4 items
        assert len(items) == 4
        assert items[0].rendered_elements["text"] == "The cat sat."
        assert items[0].rendered_elements["prompt"] == "How natural is this?"
        assert items[0].item_metadata["scale_min"] == 1

    def test_with_metadata_function(self) -> None:
        """Test cross-product with metadata function."""
        texts = ["Text 1"]
        prompts = ["Prompt 1"]

        items = create_ordinal_scale_items_cross_product(
            texts,
            prompts,
            scale_bounds=ScaleBounds(min=1, max=5),
            metadata_fn=lambda t, p: {"combined_length": len(t) + len(p)},
        )

        assert items[0].item_metadata["combined_length"] == len("Text 1") + len(
            "Prompt 1"
        )


class TestCreateFilteredOrdinalScaleItems:
    """Test create_filtered_ordinal_scale_items() function."""

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

        ordinal_items = create_filtered_ordinal_scale_items(
            items,
            scale_bounds=ScaleBounds(min=1, max=7),
            prompt="Rate this:",
            item_filter=lambda i: i.item_metadata.get("valid", True),
        )

        assert len(ordinal_items) == 1
        assert ordinal_items[0].rendered_elements["text"] == "Valid"

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

        ordinal_items = create_filtered_ordinal_scale_items(
            items, scale_bounds=ScaleBounds(min=1, max=5), prompt="Rate this:"
        )

        assert len(ordinal_items) == 3


class TestCreateLikert5Item:
    """Test create_likert_5_item() function."""

    def test_basic_likert_5(self) -> None:
        """Test creating a 5-point Likert item."""
        item = create_likert_5_item("I enjoy linguistics.")

        assert item.rendered_elements["text"] == "I enjoy linguistics."
        assert item.item_metadata["scale_min"] == 1
        assert item.item_metadata["scale_max"] == 5
        assert item.item_metadata["scale_labels"]["1"] == "Strongly Disagree"
        assert item.item_metadata["scale_labels"]["5"] == "Strongly Agree"

    def test_default_prompt(self) -> None:
        """Test default prompt for Likert 5."""
        item = create_likert_5_item("Statement")

        assert item.rendered_elements["prompt"] == "Rate your agreement:"

    def test_custom_prompt(self) -> None:
        """Test custom prompt for Likert 5."""
        item = create_likert_5_item("Statement", prompt="Do you agree?")

        assert item.rendered_elements["prompt"] == "Do you agree?"


class TestCreateLikert7Item:
    """Test create_likert_7_item() function."""

    def test_basic_likert_7(self) -> None:
        """Test creating a 7-point Likert item."""
        item = create_likert_7_item("I enjoy linguistics.")

        assert item.rendered_elements["text"] == "I enjoy linguistics."
        assert item.item_metadata["scale_min"] == 1
        assert item.item_metadata["scale_max"] == 7
        assert item.item_metadata["scale_labels"]["1"] == "Strongly Disagree"
        assert item.item_metadata["scale_labels"]["7"] == "Strongly Agree"

    def test_default_prompt(self) -> None:
        """Test default prompt for Likert 7."""
        item = create_likert_7_item("Statement")

        assert item.rendered_elements["prompt"] == "Rate your agreement:"
