"""Tests for forced-choice item creation utilities."""

from __future__ import annotations

from uuid import uuid4

import didactic.api as dx
import pytest

from bead.items.forced_choice import (
    create_filtered_forced_choice_items,
    create_forced_choice_item,
    create_forced_choice_items_cross_product,
    create_forced_choice_items_from_groups,
)
from bead.items.item import Item


class TestCreateForcedChoiceItem:
    """Test create_forced_choice_item() function."""

    def test_create_2afc_item(self) -> None:
        """Test creating a 2AFC item."""
        item = create_forced_choice_item("Option A", "Option B")

        assert isinstance(item, Item)
        assert item.options[0] == "Option A"
        assert item.options[1] == "Option B"
        assert len(item.options) == 2

    def test_create_3afc_item(self) -> None:
        """Test creating a 3AFC item."""
        item = create_forced_choice_item("A", "B", "C")

        assert item.options[0] == "A"
        assert item.options[1] == "B"
        assert item.options[2] == "C"
        assert len(item.options) == 3

    def test_create_4afc_item(self) -> None:
        """Test creating a 4AFC item."""
        item = create_forced_choice_item("A", "B", "C", "D")

        assert len(item.options) == 4
        assert item.options[3] == "D"

    def test_requires_at_least_two_options(self) -> None:
        """Test that at least 2 options are required."""
        with pytest.raises(
            (ValueError, dx.ValidationError), match="At least 2 options required"
        ):
            create_forced_choice_item("Only one")

    def test_with_custom_template_id(self) -> None:
        """Test creating item with custom template ID."""
        template_id = uuid4()
        item = create_forced_choice_item("A", "B", item_template_id=template_id)

        assert item.item_template_id == template_id

    def test_with_metadata(self) -> None:
        """Test creating item with metadata."""
        item = create_forced_choice_item(
            "A", "B", metadata={"contrast": "number", "verb": "walk"}
        )

        assert item.item_metadata["contrast"] == "number"
        assert item.item_metadata["verb"] == "walk"

    def test_n_options_metadata(self) -> None:
        """Test that n_options is included in metadata."""
        item = create_forced_choice_item("A", "B", "C")

        assert item.item_metadata["n_options"] == 3

    def test_default_generates_uuid(self) -> None:
        """Test that default behavior generates UUID."""
        item = create_forced_choice_item("A", "B")

        assert item.item_template_id is not None

    def test_rendered_elements_empty(self) -> None:
        """Test that rendered_elements is empty for forced choice items."""
        item = create_forced_choice_item("A", "B")

        assert len(item.rendered_elements) == 0


class TestCreateForcedChoiceItemsFromGroups:
    """Test create_forced_choice_items_from_groups() function."""

    def test_basic_grouping(self) -> None:
        """Test basic grouping by property."""
        items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "She walks."},
                item_metadata={"verb": "walk", "frame": "intransitive"},
            ),
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "She walks the dog."},
                item_metadata={"verb": "walk", "frame": "transitive"},
            ),
        ]

        fc_items = create_forced_choice_items_from_groups(
            items, group_by=lambda item: item.item_metadata["verb"], n_alternatives=2
        )

        assert len(fc_items) == 1
        assert fc_items[0].options[0] == "She walks."
        assert fc_items[0].options[1] == "She walks the dog."

    def test_multiple_groups(self) -> None:
        """Test grouping with multiple groups."""
        items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "walks"},
                item_metadata={"verb": "walk"},
            ),
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "walked"},
                item_metadata={"verb": "walk"},
            ),
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "runs"},
                item_metadata={"verb": "run"},
            ),
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "ran"},
                item_metadata={"verb": "run"},
            ),
        ]

        fc_items = create_forced_choice_items_from_groups(
            items, group_by=lambda item: item.item_metadata["verb"], n_alternatives=2
        )

        # 2 groups × C(2,2) = 2 items
        assert len(fc_items) == 2

        # Check group metadata is included
        assert all("group_key" in item.item_metadata for item in fc_items)

    def test_3afc_combinations(self) -> None:
        """Test creating 3AFC items from groups."""
        items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": f"Option {i}"},
                item_metadata={"group": "A"},
            )
            for i in range(4)
        ]

        fc_items = create_forced_choice_items_from_groups(
            items, group_by=lambda item: item.item_metadata["group"], n_alternatives=3
        )

        # C(4,3) = 4 combinations
        assert len(fc_items) == 4

        # Each should have 3 options
        for fc_item in fc_items:
            assert len(fc_item.options) == 3

    def test_custom_extract_text(self) -> None:
        """Test with custom text extraction function."""
        items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"sentence": "First sentence"},
                item_metadata={"group": "A"},
            ),
            Item(
                item_template_id=uuid4(),
                rendered_elements={"sentence": "Second sentence"},
                item_metadata={"group": "A"},
            ),
        ]

        fc_items = create_forced_choice_items_from_groups(
            items,
            group_by=lambda item: item.item_metadata["group"],
            n_alternatives=2,
            extract_text=lambda item: item.rendered_elements["sentence"],
        )

        assert len(fc_items) == 1
        assert fc_items[0].options[0] == "First sentence"

    def test_without_group_metadata(self) -> None:
        """Test without including group metadata."""
        items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "A"},
                item_metadata={"g": "1"},
            ),
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "B"},
                item_metadata={"g": "1"},
            ),
        ]

        fc_items = create_forced_choice_items_from_groups(
            items,
            group_by=lambda item: item.item_metadata["g"],
            n_alternatives=2,
            include_group_metadata=False,
        )

        assert "group_key" not in fc_items[0].item_metadata

    def test_source_item_ids_included(self) -> None:
        """Test that source item IDs are included in metadata."""
        items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "A"},
                item_metadata={"g": "1"},
            ),
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "B"},
                item_metadata={"g": "1"},
            ),
        ]

        fc_items = create_forced_choice_items_from_groups(
            items, group_by=lambda item: item.item_metadata["g"], n_alternatives=2
        )

        metadata = fc_items[0].item_metadata
        assert "source_item_0_id" in metadata
        assert "source_item_1_id" in metadata

    def test_fallback_text_extraction(self) -> None:
        """Test fallback text extraction from common keys."""
        items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"content": "Content 1"},
                item_metadata={"g": "1"},
            ),
            Item(
                item_template_id=uuid4(),
                rendered_elements={"sentence": "Sentence 2"},
                item_metadata={"g": "1"},
            ),
        ]

        fc_items = create_forced_choice_items_from_groups(
            items, group_by=lambda item: item.item_metadata["g"], n_alternatives=2
        )

        # Should extract using fallback logic
        assert len(fc_items) == 1


class TestCreateForcedChoiceItemsCrossProduct:
    """Test create_forced_choice_items_cross_product() function."""

    def test_basic_cross_product(self) -> None:
        """Test basic cross-product of two groups."""
        group1 = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "Grammatical"},
                item_metadata={"g": True},
            )
        ]
        group2 = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "Ungrammatical"},
                item_metadata={"g": False},
            )
        ]

        fc_items = create_forced_choice_items_cross_product(
            group1, group2, n_from_group1=1, n_from_group2=1
        )

        assert len(fc_items) == 1
        assert fc_items[0].options[0] == "Grammatical"
        assert fc_items[0].options[1] == "Ungrammatical"

    def test_multiple_from_each_group(self) -> None:
        """Test selecting multiple items from each group."""
        group1 = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "G1"},
                item_metadata={},
            ),
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "G2"},
                item_metadata={},
            ),
        ]
        group2 = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "U1"},
                item_metadata={},
            ),
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "U2"},
                item_metadata={},
            ),
        ]

        fc_items = create_forced_choice_items_cross_product(
            group1, group2, n_from_group1=1, n_from_group2=1
        )

        # C(2,1) × C(2,1) = 2 × 2 = 4 items
        assert len(fc_items) == 4

    def test_with_metadata_fn(self) -> None:
        """Test with custom metadata function."""
        group1 = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "A"},
                item_metadata={},
            )
        ]
        group2 = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "B"},
                item_metadata={},
            )
        ]

        def custom_metadata(combo1, combo2):
            return {
                "custom_key": "custom_value",
                "n_group1": len(combo1),
                "n_group2": len(combo2),
            }

        fc_items = create_forced_choice_items_cross_product(
            group1,
            group2,
            n_from_group1=1,
            n_from_group2=1,
            metadata_fn=custom_metadata,
        )

        assert fc_items[0].item_metadata["custom_key"] == "custom_value"
        assert fc_items[0].item_metadata["n_group1"] == 1
        assert fc_items[0].item_metadata["n_group2"] == 1

    def test_default_metadata(self) -> None:
        """Test default metadata includes source IDs."""
        group1 = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "A"},
                item_metadata={},
            )
        ]
        group2 = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "B"},
                item_metadata={},
            )
        ]

        fc_items = create_forced_choice_items_cross_product(
            group1, group2, n_from_group1=1, n_from_group2=1
        )

        metadata = fc_items[0].item_metadata
        assert "source_group1_ids" in metadata
        assert "source_group2_ids" in metadata

    def test_custom_extract_text(self) -> None:
        """Test with custom text extraction."""
        group1 = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"sentence": "Sent 1"},
                item_metadata={},
            )
        ]
        group2 = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"sentence": "Sent 2"},
                item_metadata={},
            )
        ]

        fc_items = create_forced_choice_items_cross_product(
            group1,
            group2,
            n_from_group1=1,
            n_from_group2=1,
            extract_text=lambda item: item.rendered_elements["sentence"],
        )

        assert fc_items[0].options[0] == "Sent 1"
        assert fc_items[0].options[1] == "Sent 2"


class TestCreateFilteredForcedChoiceItems:
    """Test create_filtered_forced_choice_items() function."""

    def test_item_filter(self) -> None:
        """Test filtering individual items."""
        items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "Valid"},
                item_metadata={"valid": True, "g": "A"},
            ),
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "Invalid"},
                item_metadata={"valid": False, "g": "A"},
            ),
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "Valid2"},
                item_metadata={"valid": True, "g": "A"},
            ),
        ]

        fc_items = create_filtered_forced_choice_items(
            items,
            group_by=lambda i: i.item_metadata["g"],
            n_alternatives=2,
            item_filter=lambda i: i.item_metadata.get("valid", True),
        )

        # Only 2 valid items, so C(2,2) = 1 combination
        assert len(fc_items) == 1

    def test_group_filter(self) -> None:
        """Test filtering groups."""
        items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "A1"},
                item_metadata={"g": "A"},
            ),
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "A2"},
                item_metadata={"g": "A"},
            ),
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "B1"},
                item_metadata={"g": "B"},
            ),
        ]

        fc_items = create_filtered_forced_choice_items(
            items,
            group_by=lambda i: i.item_metadata["g"],
            n_alternatives=2,
            group_filter=lambda key, items: len(items) >= 2,
        )

        # Only group A has ≥2 items
        # C(2,2) = 1 combination
        assert len(fc_items) == 1
        assert "A1" in fc_items[0].options[0] or "A2" in fc_items[0].options[0]

    def test_combination_filter(self) -> None:
        """Test filtering specific combinations."""
        items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "Short"},
                item_metadata={"g": "A"},
            ),
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "This is much longer text"},
                item_metadata={"g": "A"},
            ),
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "Medium text"},
                item_metadata={"g": "A"},
            ),
        ]

        def filter_by_length_difference(combo):
            # Only allow combinations where length difference < 10
            texts = [item.rendered_elements["text"] for item in combo]
            lengths = [len(text) for text in texts]
            return max(lengths) - min(lengths) < 10

        fc_items = create_filtered_forced_choice_items(
            items,
            group_by=lambda i: i.item_metadata["g"],
            n_alternatives=2,
            combination_filter=filter_by_length_difference,
        )

        # "Short" (5) and "This is much longer text" (24) differ by 19 → filtered out
        # Other combinations should pass
        assert len(fc_items) > 0

    def test_all_filters_combined(self) -> None:
        """Test using all filters together."""
        items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "A1"},
                item_metadata={"valid": True, "g": "A"},
            ),
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "A2"},
                item_metadata={"valid": False, "g": "A"},
            ),
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "A3"},
                item_metadata={"valid": True, "g": "A"},
            ),
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "B1"},
                item_metadata={"valid": True, "g": "B"},
            ),
        ]

        fc_items = create_filtered_forced_choice_items(
            items,
            group_by=lambda i: i.item_metadata["g"],
            n_alternatives=2,
            item_filter=lambda i: i.item_metadata["valid"],
            group_filter=lambda key, items: len(items) >= 2,
            combination_filter=lambda combo: combo[0].id != combo[1].id,
        )

        # After item_filter: A1, A3, B1
        # After group_filter: only group A (has 2 items)
        # C(2,2) = 1 combination
        assert len(fc_items) == 1

    def test_no_filters(self) -> None:
        """Test that function works with no filters."""
        items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "A1"},
                item_metadata={"g": "A"},
            ),
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "A2"},
                item_metadata={"g": "A"},
            ),
        ]

        fc_items = create_filtered_forced_choice_items(
            items, group_by=lambda i: i.item_metadata["g"], n_alternatives=2
        )

        assert len(fc_items) == 1

    def test_source_item_ids_included(self) -> None:
        """Test that source_item_ids are included in metadata."""
        items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "A1"},
                item_metadata={"g": "A"},
            ),
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "A2"},
                item_metadata={"g": "A"},
            ),
        ]

        fc_items = create_filtered_forced_choice_items(
            items, group_by=lambda i: i.item_metadata["g"], n_alternatives=2
        )

        assert "source_item_ids" in fc_items[0].item_metadata
        assert len(fc_items[0].item_metadata["source_item_ids"]) == 2
