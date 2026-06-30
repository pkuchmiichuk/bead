"""Tests for magnitude item creation utilities."""

from __future__ import annotations

from uuid import uuid4

import didactic.api as dx
import pytest

from bead.items.item import Item
from bead.items.magnitude import (
    create_confidence_item,
    create_filtered_magnitude_items,
    create_magnitude_item,
    create_magnitude_items_cross_product,
    create_magnitude_items_from_groups,
    create_magnitude_items_from_texts,
    create_reading_time_item,
)


class TestCreateMagnitudeItem:
    """Test create_magnitude_item() function."""

    def test_create_basic_unbounded_item(self) -> None:
        """Test creating a basic unbounded magnitude item."""
        item = create_magnitude_item(
            "How many times did this occur?", bounds=(None, None)
        )

        assert isinstance(item, Item)
        assert item.rendered_elements["text"] == "How many times did this occur?"
        assert item.rendered_elements["prompt"] == "Enter a value:"
        assert item.item_metadata["min_value"] is None
        assert item.item_metadata["max_value"] is None

    def test_default_prompt(self) -> None:
        """Test default prompt."""
        item = create_magnitude_item("Enter count")

        assert item.rendered_elements["prompt"] == "Enter a value:"

    def test_custom_prompt(self) -> None:
        """Test custom prompt."""
        item = create_magnitude_item("How long?", prompt="Enter reading time in ms:")

        assert item.rendered_elements["prompt"] == "Enter reading time in ms:"

    def test_with_unit(self) -> None:
        """Test item with unit."""
        item = create_magnitude_item("Reading time?", unit="ms", bounds=(0, None))

        assert item.rendered_elements["unit"] == "ms"
        assert item.item_metadata["unit"] == "ms"

    def test_with_lower_bound_only(self) -> None:
        """Test item with lower bound only."""
        item = create_magnitude_item("Enter positive value", bounds=(0, None))

        assert item.item_metadata["min_value"] == 0
        assert item.item_metadata["max_value"] is None

    def test_with_upper_bound_only(self) -> None:
        """Test item with upper bound only."""
        item = create_magnitude_item("Enter value below 100", bounds=(None, 100))

        assert item.item_metadata["min_value"] is None
        assert item.item_metadata["max_value"] == 100

    def test_with_both_bounds(self) -> None:
        """Test item with both bounds."""
        item = create_magnitude_item("Confidence?", unit="%", bounds=(0, 100))

        assert item.item_metadata["min_value"] == 0
        assert item.item_metadata["max_value"] == 100

    def test_with_float_bounds(self) -> None:
        """Test item with float bounds."""
        item = create_magnitude_item("Rate", bounds=(0.0, 1.0))

        assert item.item_metadata["min_value"] == 0.0
        assert item.item_metadata["max_value"] == 1.0

    def test_with_step(self) -> None:
        """Test item with step parameter."""
        item = create_magnitude_item("Value", bounds=(0, 100), step=0.1)

        assert item.item_metadata["step"] == 0.1

    def test_empty_text_raises_error(self) -> None:
        """Test that empty text raises error."""
        with pytest.raises(
            (ValueError, dx.ValidationError), match="text cannot be empty"
        ):
            create_magnitude_item("")

        with pytest.raises(
            (ValueError, dx.ValidationError), match="text cannot be empty"
        ):
            create_magnitude_item("   ")

    def test_invalid_bounds_raises_error(self) -> None:
        """Test that invalid bounds raise error."""
        with pytest.raises(
            (ValueError, dx.ValidationError), match="min_value.*must be less than"
        ):
            create_magnitude_item("Text", bounds=(100, 100))

        with pytest.raises(
            (ValueError, dx.ValidationError), match="min_value.*must be less than"
        ):
            create_magnitude_item("Text", bounds=(100, 50))

    def test_with_custom_template_id(self) -> None:
        """Test creating item with custom template ID."""
        template_id = uuid4()
        item = create_magnitude_item("Text", item_template_id=template_id)

        assert item.item_template_id == template_id

    def test_with_metadata(self) -> None:
        """Test creating item with metadata."""
        item = create_magnitude_item(
            "Text", metadata={"task": "reading_time", "condition": "easy"}
        )

        assert item.item_metadata["task"] == "reading_time"
        assert item.item_metadata["condition"] == "easy"


class TestCreateMagnitudeItemsFromTexts:
    """Test create_magnitude_items_from_texts() function."""

    def test_basic_batch_creation(self) -> None:
        """Test basic batch creation from texts."""
        texts = ["Sentence 1", "Sentence 2", "Sentence 3"]

        items = create_magnitude_items_from_texts(
            texts, unit="ms", bounds=(0, None), prompt="Reading time?"
        )

        assert len(items) == 3
        assert all(isinstance(item, Item) for item in items)
        assert items[0].rendered_elements["text"] == "Sentence 1"
        assert items[0].rendered_elements["prompt"] == "Reading time?"
        assert items[0].item_metadata["unit"] == "ms"
        assert items[0].item_metadata["min_value"] == 0

    def test_with_metadata_function(self) -> None:
        """Test with metadata function."""
        texts = ["Short", "Medium length"]

        items = create_magnitude_items_from_texts(
            texts,
            unit="ms",
            bounds=(0, None),
            metadata_fn=lambda text: {"text_length": len(text)},
        )

        assert items[0].item_metadata["text_length"] == len("Short")
        assert items[1].item_metadata["text_length"] == len("Medium length")


class TestCreateMagnitudeItemsFromGroups:
    """Test create_magnitude_items_from_groups() function."""

    def test_basic_grouping(self) -> None:
        """Test basic grouping with preservation of group info."""
        source_items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "Simple sentence."},
                item_metadata={"complexity": "simple"},
            ),
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "Complex sentence."},
                item_metadata={"complexity": "complex"},
            ),
        ]

        magnitude_items = create_magnitude_items_from_groups(
            source_items,
            group_by=lambda i: i.item_metadata["complexity"],
            unit="ms",
            bounds=(0, None),
            prompt="Reading time?",
        )

        assert len(magnitude_items) == 2
        assert all("group_key" in item.item_metadata for item in magnitude_items)
        assert all("source_item_id" in item.item_metadata for item in magnitude_items)

    def test_without_group_metadata(self) -> None:
        """Test grouping without including group metadata."""
        source_items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "Text"},
                item_metadata={"group": "A"},
            )
        ]

        magnitude_items = create_magnitude_items_from_groups(
            source_items,
            group_by=lambda i: i.item_metadata["group"],
            unit="ms",
            bounds=(0, None),
            include_group_metadata=False,
        )

        assert "group_key" not in magnitude_items[0].item_metadata
        assert "source_item_id" in magnitude_items[0].item_metadata


class TestCreateMagnitudeItemsCrossProduct:
    """Test create_magnitude_items_cross_product() function."""

    def test_basic_cross_product(self) -> None:
        """Test basic cross-product of texts and prompts."""
        texts = ["Sentence 1", "Sentence 2"]
        prompts = ["Reading time?", "Processing time?"]

        items = create_magnitude_items_cross_product(
            texts, prompts, unit="ms", bounds=(0, None)
        )

        # 2 texts × 2 prompts = 4 items
        assert len(items) == 4
        assert items[0].rendered_elements["text"] == "Sentence 1"
        assert items[0].rendered_elements["prompt"] == "Reading time?"
        assert items[0].item_metadata["unit"] == "ms"

    def test_with_metadata_function(self) -> None:
        """Test cross-product with metadata function."""
        texts = ["Text 1"]
        prompts = ["Prompt 1"]

        items = create_magnitude_items_cross_product(
            texts,
            prompts,
            unit="ms",
            bounds=(0, None),
            metadata_fn=lambda t, p: {"combined_length": len(t) + len(p)},
        )

        assert items[0].item_metadata["combined_length"] == len("Text 1") + len(
            "Prompt 1"
        )


class TestCreateFilteredMagnitudeItems:
    """Test create_filtered_magnitude_items() function."""

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

        magnitude_items = create_filtered_magnitude_items(
            items,
            unit="ms",
            bounds=(0, None),
            prompt="Time?",
            item_filter=lambda i: i.item_metadata.get("valid", True),
        )

        assert len(magnitude_items) == 1
        assert magnitude_items[0].rendered_elements["text"] == "Valid"

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

        magnitude_items = create_filtered_magnitude_items(
            items, unit="ms", bounds=(0, None), prompt="Time?"
        )

        assert len(magnitude_items) == 3


class TestCreateReadingTimeItem:
    """Test create_reading_time_item() function."""

    def test_basic_reading_time(self) -> None:
        """Test creating a reading time item."""
        item = create_reading_time_item("The cat sat on the mat.")

        assert item.rendered_elements["text"] == "The cat sat on the mat."
        assert item.rendered_elements["prompt"] == "How long did it take to read?"
        assert item.item_metadata["unit"] == "ms"
        assert item.item_metadata["min_value"] == 0
        assert item.item_metadata["max_value"] is None
        assert item.item_metadata["step"] == 1

    def test_with_metadata(self) -> None:
        """Test reading time item with custom metadata."""
        item = create_reading_time_item("Sentence", metadata={"condition": "baseline"})

        assert item.item_metadata["condition"] == "baseline"
        assert item.item_metadata["unit"] == "ms"


class TestCreateConfidenceItem:
    """Test create_confidence_item() function."""

    def test_basic_confidence(self) -> None:
        """Test creating a confidence item."""
        item = create_confidence_item("Is this sentence grammatical?")

        assert item.rendered_elements["text"] == "Is this sentence grammatical?"
        assert item.rendered_elements["prompt"] == "How confident are you?"
        assert item.item_metadata["unit"] == "%"
        assert item.item_metadata["min_value"] == 0
        assert item.item_metadata["max_value"] == 100
        assert item.item_metadata["step"] == 1

    def test_with_metadata(self) -> None:
        """Test confidence item with custom metadata."""
        item = create_confidence_item("Question", metadata={"task": "nli"})

        assert item.item_metadata["task"] == "nli"
        assert item.item_metadata["unit"] == "%"
