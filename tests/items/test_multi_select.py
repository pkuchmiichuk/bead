"""Tests for multi-select item creation utilities."""

from __future__ import annotations

from uuid import uuid4

import didactic.api as dx
import pytest

from bead.items.item import Item
from bead.items.multi_select import (
    create_filtered_multi_select_items,
    create_multi_select_item,
    create_multi_select_items_cross_product,
    create_multi_select_items_from_groups,
    create_multi_select_items_with_foils,
)


class TestCreateMultiSelectItem:
    """Test create_multi_select_item() function."""

    def test_create_basic_multi_select(self) -> None:
        """Test creating a basic multi-select item."""
        item = create_multi_select_item("Option A", "Option B", "Option C")

        assert isinstance(item, Item)
        assert item.options[0] == "Option A"
        assert item.options[1] == "Option B"
        assert item.options[2] == "Option C"
        assert len(item.options) == 3

    def test_default_min_max_selections(self) -> None:
        """Test default min/max selections."""
        item = create_multi_select_item("A", "B", "C")

        assert item.item_metadata["min_selections"] == 1
        assert item.item_metadata["max_selections"] == 3

    def test_custom_min_max_selections(self) -> None:
        """Test custom min/max selections."""
        item = create_multi_select_item(
            "A", "B", "C", "D", min_selections=2, max_selections=3
        )

        assert item.item_metadata["min_selections"] == 2
        assert item.item_metadata["max_selections"] == 3

    def test_requires_at_least_two_options(self) -> None:
        """Test that at least 2 options are required."""
        with pytest.raises(
            (ValueError, dx.ValidationError), match="At least 2 options required"
        ):
            create_multi_select_item("Only one")

    def test_min_selections_validation(self) -> None:
        """Test that min_selections must be at least 1."""
        with pytest.raises(
            (ValueError, dx.ValidationError), match="min_selections must be at least 1"
        ):
            create_multi_select_item("A", "B", min_selections=0)

    def test_min_max_validation(self) -> None:
        """Test that min_selections cannot exceed max_selections."""
        with pytest.raises(
            ValueError, match="min_selections cannot be greater than max_selections"
        ):
            create_multi_select_item("A", "B", "C", min_selections=3, max_selections=2)

    def test_max_selections_validation(self) -> None:
        """Test that max_selections cannot exceed number of options."""
        with pytest.raises(
            ValueError, match="max_selections .* cannot exceed number of options"
        ):
            create_multi_select_item("A", "B", "C", max_selections=5)

    def test_many_options_allowed(self) -> None:
        """Test that more than 26 options is allowed with list-based storage."""
        options = [f"Option {i}" for i in range(30)]

        item = create_multi_select_item(*options)
        assert len(item.options) == 30
        assert item.item_metadata["max_selections"] == 30

    def test_with_custom_template_id(self) -> None:
        """Test creating item with custom template ID."""
        template_id = uuid4()
        item = create_multi_select_item("A", "B", item_template_id=template_id)

        assert item.item_template_id == template_id

    def test_with_metadata(self) -> None:
        """Test creating item with metadata."""
        item = create_multi_select_item(
            "A", "B", "C", metadata={"task": "select_grammatical"}
        )

        assert item.item_metadata["task"] == "select_grammatical"
        assert item.item_metadata["min_selections"] == 1
        assert item.item_metadata["max_selections"] == 3

    def test_options_stored_in_list(self) -> None:
        """Test that options are stored in the options list field."""
        item = create_multi_select_item("A", "B", "C")

        # Options are stored in item.options list, not rendered_elements
        assert len(item.options) == 3
        assert item.options[0] == "A"
        assert item.options[1] == "B"
        assert item.options[2] == "C"
        assert len(item.rendered_elements) == 0


class TestCreateMultiSelectItemsFromGroups:
    """Test create_multi_select_items_from_groups() function."""

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
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "She walks to school."},
                item_metadata={"verb": "walk", "frame": "intransitive_pp"},
            ),
        ]

        ms_items = create_multi_select_items_from_groups(
            items, group_by=lambda item: item.item_metadata["verb"]
        )

        assert len(ms_items) == 1
        assert len(ms_items[0].options) == 3
        assert ms_items[0].item_metadata["group_key"] == "walk"

    def test_n_options_parameter(self) -> None:
        """Test n_options parameter to limit options per item."""
        items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": f"Sentence {i}"},
                item_metadata={},
            )
            for i in range(5)
        ]

        ms_items = create_multi_select_items_from_groups(
            items, group_by=lambda i: "group", n_options=3
        )

        # Should create combinations of 3 from 5 = C(5,3) = 10
        assert len(ms_items) == 10
        for item in ms_items:
            assert len(item.options) == 3

    def test_custom_min_max(self) -> None:
        """Test custom min/max selections."""
        items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": f"Sentence {i}"},
                item_metadata={},
            )
            for i in range(3)
        ]

        ms_items = create_multi_select_items_from_groups(
            items,
            group_by=lambda i: "group",
            min_selections=1,
            max_selections=2,
        )

        assert ms_items[0].item_metadata["min_selections"] == 1
        assert ms_items[0].item_metadata["max_selections"] == 2


class TestCreateMultiSelectItemsWithFoils:
    """Test create_multi_select_items_with_foils() function."""

    def test_basic_foils(self) -> None:
        """Test combining correct items with foils."""
        correct_items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "She walks."},
                item_metadata={"grammatical": True},
            ),
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "They walk."},
                item_metadata={"grammatical": True},
            ),
        ]

        foil_items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "She walk."},
                item_metadata={"grammatical": False},
            ),
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "They walks."},
                item_metadata={"grammatical": False},
            ),
        ]

        ms_items = create_multi_select_items_with_foils(
            correct_items, foil_items, n_correct=2, n_foils=2
        )

        assert len(ms_items) == 1
        assert len(ms_items[0].options) == 4
        assert ms_items[0].item_metadata["n_correct"] == 2
        assert ms_items[0].item_metadata["n_foils"] == 2

    def test_multiple_combinations(self) -> None:
        """Test creating multiple combinations from foils."""
        correct_items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": f"Correct {i}"},
                item_metadata={},
            )
            for i in range(3)
        ]

        foil_items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": f"Foil {i}"},
                item_metadata={},
            )
            for i in range(3)
        ]

        ms_items = create_multi_select_items_with_foils(
            correct_items, foil_items, n_correct=2, n_foils=2
        )

        # C(3,2) * C(3,2) = 3 * 3 = 9
        assert len(ms_items) == 9


class TestCreateMultiSelectItemsCrossProduct:
    """Test create_multi_select_items_cross_product() function."""

    def test_basic_cross_product(self) -> None:
        """Test basic cross-product."""
        group1 = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "A1"},
                item_metadata={},
            )
        ]
        group2 = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "B1"},
                item_metadata={},
            )
        ]

        ms_items = create_multi_select_items_cross_product(
            group1, group2, n_from_group1=1, n_from_group2=1
        )

        assert len(ms_items) == 1
        assert len(ms_items[0].options) == 2


class TestCreateFilteredMultiSelectItems:
    """Test create_filtered_multi_select_items() function."""

    def test_item_filter(self) -> None:
        """Test filtering individual items."""
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

        # Should raise error when filtering leaves only 1 item
        with pytest.raises(
            (ValueError, dx.ValidationError),
            match="has only 1 item\\(s\\) after filtering",
        ):
            create_filtered_multi_select_items(
                items,
                group_by=lambda i: "group",
                item_filter=lambda i: i.item_metadata.get("valid", True),
            )

        # But should work with group_filter to exclude small groups
        ms_items = create_filtered_multi_select_items(
            items,
            group_by=lambda i: "group",
            item_filter=lambda i: i.item_metadata.get("valid", True),
            group_filter=lambda key, items: len(items) >= 2,
        )
        assert len(ms_items) == 0

    def test_group_filter(self) -> None:
        """Test filtering groups."""
        items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": f"A{i}"},
                item_metadata={"g": "a"},
            )
            for i in range(2)
        ] + [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": f"B{i}"},
                item_metadata={"g": "b"},
            )
            for i in range(4)
        ]

        ms_items = create_filtered_multi_select_items(
            items,
            group_by=lambda i: i.item_metadata["g"],
            group_filter=lambda key, items: len(items) >= 3,
        )

        # Only group "b" has >= 3 items
        assert len(ms_items) == 1
        assert len(ms_items[0].options) == 4
