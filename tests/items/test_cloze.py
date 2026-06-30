"""Tests for cloze item creation utilities."""

from __future__ import annotations

from uuid import uuid4

import didactic.api as dx
import pytest

from bead.items.cloze import (
    create_cloze_item,
    create_cloze_items_from_groups,
    create_cloze_items_from_template,
    create_filtered_cloze_items,
    create_simple_cloze_item,
)
from bead.items.item import Item
from bead.resources.constraints import Constraint
from bead.resources.template import Slot, Template


class TestCreateClozeItem:
    """Test create_cloze_item() function."""

    def test_create_basic_single_slot_cloze(self) -> None:
        """Test creating cloze item with one unfilled slot."""
        template = Template(
            name="simple",
            template_string="{det} {noun} {verb}.",
            slots={
                "det": Slot(name="det"),
                "noun": Slot(name="noun"),
                "verb": Slot(name="verb"),
            },
        )

        item = create_cloze_item(
            template,
            unfilled_slot_names=["verb"],
            filled_slots={"det": "The", "noun": "cat"},
        )

        assert isinstance(item, Item)
        assert item.rendered_elements["text"] == "The cat ___."
        assert len(item.unfilled_slots) == 1
        assert item.unfilled_slots[0].slot_name == "verb"
        assert item.unfilled_slots[0].position == 2
        assert item.item_metadata["template_id"] == str(template.id)
        assert item.item_metadata["n_unfilled_slots"] == 1
        assert item.item_metadata["filled_slots"] == {"det": "The", "noun": "cat"}

    def test_create_multi_slot_cloze(self) -> None:
        """Test creating cloze item with multiple unfilled slots."""
        template = Template(
            name="multi",
            template_string="{det} {adj} {noun} {verb} {obj}.",
            slots={
                "det": Slot(name="det"),
                "adj": Slot(name="adj"),
                "noun": Slot(name="noun"),
                "verb": Slot(name="verb"),
                "obj": Slot(name="obj"),
            },
        )

        item = create_cloze_item(
            template,
            unfilled_slot_names=["adj", "verb"],
            filled_slots={"det": "The", "noun": "cat", "obj": "mouse"},
        )

        assert item.rendered_elements["text"] == "The ___ cat ___ mouse."
        assert len(item.unfilled_slots) == 2
        assert item.unfilled_slots[0].slot_name == "adj"
        assert item.unfilled_slots[1].slot_name == "verb"
        assert item.item_metadata["n_unfilled_slots"] == 2

    def test_positions_are_correct(self) -> None:
        """Test that UnfilledSlot positions match token indices."""
        template = Template(
            name="positions",
            template_string="{word1} {word2} {word3} {word4}",
            slots={
                "word1": Slot(name="word1"),
                "word2": Slot(name="word2"),
                "word3": Slot(name="word3"),
                "word4": Slot(name="word4"),
            },
        )

        # Unfill word2 (should be at position 1)
        item = create_cloze_item(
            template,
            unfilled_slot_names=["word2"],
            filled_slots={"word1": "One", "word3": "Three", "word4": "Four"},
        )

        assert item.unfilled_slots[0].position == 1
        assert item.rendered_elements["text"] == "One ___ Three Four"

        # Unfill word4 (should be at position 3)
        item2 = create_cloze_item(
            template,
            unfilled_slot_names=["word4"],
            filled_slots={"word1": "One", "word2": "Two", "word3": "Three"},
        )

        assert item2.unfilled_slots[0].position == 3

    def test_constraint_ids_extracted(self) -> None:
        """Test that constraint_ids are extracted from template slots."""
        constraint1 = Constraint(expression="self.pos == 'VERB'")
        constraint2 = Constraint(expression="self.features.tense == 'past'")

        template = Template(
            name="constrained",
            template_string="{subj} {verb} {obj}.",
            slots={
                "subj": Slot(name="subj"),
                "verb": Slot(name="verb", constraints=[constraint1, constraint2]),
                "obj": Slot(name="obj"),
            },
        )

        item = create_cloze_item(
            template,
            unfilled_slot_names=["verb"],
            filled_slots={"subj": "She", "obj": "it"},
        )

        assert len(item.unfilled_slots) == 1
        assert len(item.unfilled_slots[0].constraint_ids) == 2
        assert constraint1.id in item.unfilled_slots[0].constraint_ids
        assert constraint2.id in item.unfilled_slots[0].constraint_ids

    def test_unfilled_slot_not_in_template_raises_error(self) -> None:
        """Test that invalid unfilled_slot_names raise error."""
        template = Template(
            name="test",
            template_string="{a} {b}",
            slots={"a": Slot(name="a"), "b": Slot(name="b")},
        )

        with pytest.raises(
            (ValueError, dx.ValidationError), match="not found in template"
        ):
            create_cloze_item(template, unfilled_slot_names=["invalid_slot"])

    def test_filled_slot_not_in_template_raises_error(self) -> None:
        """Test that invalid filled_slots keys raise error."""
        template = Template(
            name="test",
            template_string="{a} {b}",
            slots={"a": Slot(name="a"), "b": Slot(name="b")},
        )

        with pytest.raises(
            (ValueError, dx.ValidationError), match="not found in template"
        ):
            create_cloze_item(
                template, unfilled_slot_names=["a"], filled_slots={"invalid": "value"}
            )

    def test_unfilled_and_filled_overlap_raises_error(self) -> None:
        """Test that overlapping unfilled and filled raises error."""
        template = Template(
            name="test",
            template_string="{a} {b}",
            slots={"a": Slot(name="a"), "b": Slot(name="b")},
        )

        with pytest.raises((ValueError, dx.ValidationError), match="Overlapping slots"):
            create_cloze_item(
                template, unfilled_slot_names=["a"], filled_slots={"a": "value"}
            )

    def test_no_unfilled_slots_raises_error(self) -> None:
        """Test that empty unfilled_slot_names raises error."""
        template = Template(
            name="test",
            template_string="{a} {b}",
            slots={"a": Slot(name="a"), "b": Slot(name="b")},
        )

        with pytest.raises(
            (ValueError, dx.ValidationError), match="at least 1 unfilled slot"
        ):
            create_cloze_item(template, unfilled_slot_names=[])

    def test_with_instructions(self) -> None:
        """Test cloze item with custom instructions."""
        template = Template(
            name="test",
            template_string="{subj} {verb}",
            slots={"subj": Slot(name="subj"), "verb": Slot(name="verb")},
        )

        item = create_cloze_item(
            template,
            unfilled_slot_names=["verb"],
            filled_slots={"subj": "She"},
            instructions="Fill in the verb",
        )

        assert item.rendered_elements["instructions"] == "Fill in the verb"

    def test_with_metadata(self) -> None:
        """Test cloze item with custom metadata."""
        template = Template(
            name="test",
            template_string="{a} {b}",
            slots={"a": Slot(name="a"), "b": Slot(name="b")},
        )

        item = create_cloze_item(
            template,
            unfilled_slot_names=["b"],
            filled_slots={"a": "word"},
            metadata={"task": "cloze", "difficulty": "easy"},
        )

        assert item.item_metadata["task"] == "cloze"
        assert item.item_metadata["difficulty"] == "easy"

    def test_with_custom_template_id(self) -> None:
        """Test with custom template ID."""
        template = Template(
            name="test",
            template_string="{a} {b}",
            slots={"a": Slot(name="a"), "b": Slot(name="b")},
        )

        custom_id = uuid4()
        item = create_cloze_item(
            template,
            unfilled_slot_names=["b"],
            filled_slots={"a": "word"},
            item_template_id=custom_id,
        )

        assert item.item_template_id == custom_id


class TestCreateClozeItemsFromTemplate:
    """Test create_cloze_items_from_template() function."""

    def test_strategy_all_combinations(self) -> None:
        """Test all_combinations strategy."""
        template = Template(
            name="test",
            template_string="{a} {b} {c} {d}",
            slots={
                "a": Slot(name="a"),
                "b": Slot(name="b"),
                "c": Slot(name="c"),
                "d": Slot(name="d"),
            },
        )

        items = create_cloze_items_from_template(
            template, n_unfilled=2, strategy="all_combinations"
        )

        # C(4,2) = 6 combinations
        assert len(items) == 6
        assert all(len(item.unfilled_slots) == 2 for item in items)

    def test_strategy_specified(self) -> None:
        """Test specified strategy with custom combinations."""
        template = Template(
            name="test",
            template_string="{a} {b} {c}",
            slots={
                "a": Slot(name="a"),
                "b": Slot(name="b"),
                "c": Slot(name="c"),
            },
        )

        items = create_cloze_items_from_template(
            template,
            n_unfilled=2,
            strategy="specified",
            unfilled_combinations=[["a", "b"], ["b", "c"]],
        )

        assert len(items) == 2
        assert items[0].unfilled_slots[0].slot_name == "a"
        assert items[0].unfilled_slots[1].slot_name == "b"
        assert items[1].unfilled_slots[0].slot_name == "b"
        assert items[1].unfilled_slots[1].slot_name == "c"

    def test_strategy_random(self) -> None:
        """Test random strategy."""
        template = Template(
            name="test",
            template_string="{a} {b} {c}",
            slots={
                "a": Slot(name="a"),
                "b": Slot(name="b"),
                "c": Slot(name="c"),
            },
        )

        items = create_cloze_items_from_template(
            template, n_unfilled=1, strategy="random"
        )

        # Random should generate at least 1 item
        assert len(items) >= 1
        assert len(items[0].unfilled_slots) == 1

    def test_n_unfilled_too_large_raises_error(self) -> None:
        """Test that n_unfilled >= n_slots raises error."""
        template = Template(
            name="test",
            template_string="{a} {b}",
            slots={"a": Slot(name="a"), "b": Slot(name="b")},
        )

        with pytest.raises(
            (ValueError, dx.ValidationError), match="must be less than total slots"
        ):
            create_cloze_items_from_template(template, n_unfilled=2)

    def test_n_unfilled_zero_raises_error(self) -> None:
        """Test that n_unfilled = 0 raises error."""
        template = Template(
            name="test",
            template_string="{a} {b}",
            slots={"a": Slot(name="a"), "b": Slot(name="b")},
        )

        with pytest.raises(
            (ValueError, dx.ValidationError), match="must be at least 1"
        ):
            create_cloze_items_from_template(template, n_unfilled=0)

    def test_with_metadata_function(self) -> None:
        """Test with metadata function."""
        template = Template(
            name="test",
            template_string="{a} {b}",
            slots={"a": Slot(name="a"), "b": Slot(name="b")},
        )

        items = create_cloze_items_from_template(
            template,
            n_unfilled=1,
            strategy="all_combinations",
            metadata_fn=lambda slots: {"unfilled_slots": ",".join(slots)},
        )

        assert len(items) == 2
        assert "unfilled_slots" in items[0].item_metadata


class TestCreateSimpleClozeItem:
    """Test create_simple_cloze_item() helper."""

    def test_basic_simple_cloze(self) -> None:
        """Test creating simple cloze from plain text."""
        item = create_simple_cloze_item(
            text="The quick brown fox",
            blank_positions=[1],  # "quick"
        )

        assert item.rendered_elements["text"] == "The ___ brown fox"
        assert len(item.unfilled_slots) == 1
        assert item.unfilled_slots[0].position == 1
        assert item.unfilled_slots[0].slot_name == "blank_0"
        assert item.unfilled_slots[0].constraint_ids == ()
        assert item.item_metadata["n_unfilled_slots"] == 1

    def test_multiple_blanks(self) -> None:
        """Test multiple blank positions."""
        item = create_simple_cloze_item(
            text="The quick brown fox jumps",
            blank_positions=[1, 4],  # "quick", "jumps"
        )

        assert item.rendered_elements["text"] == "The ___ brown fox ___"
        assert len(item.unfilled_slots) == 2
        assert item.unfilled_slots[0].position == 1
        assert item.unfilled_slots[1].position == 4

    def test_blank_positions_out_of_range_raises_error(self) -> None:
        """Test that invalid positions raise error."""
        with pytest.raises((ValueError, dx.ValidationError), match="out of range"):
            create_simple_cloze_item(text="Short text", blank_positions=[100])

    def test_negative_position_raises_error(self) -> None:
        """Test that negative positions raise error."""
        with pytest.raises((ValueError, dx.ValidationError), match="out of range"):
            create_simple_cloze_item(text="Some text", blank_positions=[-1])

    def test_with_blank_labels(self) -> None:
        """Test custom labels for blanks."""
        item = create_simple_cloze_item(
            text="The cat runs",
            blank_positions=[2],
            blank_labels=["verb"],
        )

        assert item.unfilled_slots[0].slot_name == "verb"

    def test_blank_labels_length_mismatch_raises_error(self) -> None:
        """Test that mismatched labels raise error."""
        with pytest.raises(
            (ValueError, dx.ValidationError), match="blank_labels length"
        ):
            create_simple_cloze_item(
                text="The cat runs",
                blank_positions=[1, 2],
                blank_labels=["verb"],  # Only 1 label for 2 positions
            )

    def test_empty_text_raises_error(self) -> None:
        """Test that empty text raises error."""
        with pytest.raises(
            (ValueError, dx.ValidationError), match="text cannot be empty"
        ):
            create_simple_cloze_item(text="", blank_positions=[0])

    def test_empty_blank_positions_raises_error(self) -> None:
        """Test that empty blank_positions raises error."""
        with pytest.raises(
            (ValueError, dx.ValidationError), match="blank_positions cannot be empty"
        ):
            create_simple_cloze_item(text="Some text", blank_positions=[])

    def test_with_instructions(self) -> None:
        """Test simple cloze with instructions."""
        item = create_simple_cloze_item(
            text="The cat runs",
            blank_positions=[2],
            instructions="Fill in the missing word",
        )

        assert item.rendered_elements["instructions"] == "Fill in the missing word"

    def test_with_metadata(self) -> None:
        """Test simple cloze with custom metadata."""
        item = create_simple_cloze_item(
            text="The cat runs",
            blank_positions=[2],
            metadata={"difficulty": "easy"},
        )

        assert item.item_metadata["difficulty"] == "easy"


class TestCreateClozeItemsFromGroups:
    """Test create_cloze_items_from_groups() function."""

    def test_basic_grouping(self) -> None:
        """Test basic grouping with text-based cloze."""
        source_items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "The quick brown fox"},
                item_metadata={"type": "simple"},
            ),
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "A fast gray wolf"},
                item_metadata={"type": "simple"},
            ),
        ]

        cloze_items = create_cloze_items_from_groups(
            source_items,
            group_by=lambda i: i.item_metadata["type"],
            n_slots_to_unfill=1,
        )

        assert len(cloze_items) == 2
        assert all("source_item_id" in item.item_metadata for item in cloze_items)
        assert all("group_key" in item.item_metadata for item in cloze_items)

    def test_without_group_metadata(self) -> None:
        """Test grouping without including group metadata."""
        source_items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "Text"},
                item_metadata={"group": "A"},
            )
        ]

        cloze_items = create_cloze_items_from_groups(
            source_items,
            group_by=lambda i: i.item_metadata["group"],
            n_slots_to_unfill=1,
            include_group_metadata=False,
        )

        assert "group_key" not in cloze_items[0].item_metadata
        assert "source_item_id" in cloze_items[0].item_metadata


class TestCreateFilteredClozeItems:
    """Test create_filtered_cloze_items() function."""

    def test_template_filter(self) -> None:
        """Test filtering templates."""
        template1 = Template(
            name="short",
            template_string="{a} {b}",
            slots={"a": Slot(name="a"), "b": Slot(name="b")},
        )
        template2 = Template(
            name="long",
            template_string="{a} {b} {c} {d}",
            slots={
                "a": Slot(name="a"),
                "b": Slot(name="b"),
                "c": Slot(name="c"),
                "d": Slot(name="d"),
            },
        )

        cloze_items = create_filtered_cloze_items(
            templates=[template1, template2],
            n_slots_to_unfill=1,
            template_filter=lambda t: len(t.slots) >= 3,  # Only template2
        )

        # Only template2 has >=3 slots, so 4 items (one per slot)
        assert len(cloze_items) == 4

    def test_slot_filter(self) -> None:
        """Test filtering which slots can be unfilled."""
        constraint = Constraint(expression="self.pos == 'VERB'")

        template = Template(
            name="test",
            template_string="{noun} {verb} {adj}",
            slots={
                "noun": Slot(name="noun"),
                "verb": Slot(name="verb", constraints=[constraint]),
                "adj": Slot(name="adj"),
            },
        )

        cloze_items = create_filtered_cloze_items(
            templates=[template],
            n_slots_to_unfill=1,
            slot_filter=lambda name, slot: len(slot.constraints) > 0,  # Only verb
        )

        # Only verb slot has constraints, so 1 item
        assert len(cloze_items) == 1
        assert cloze_items[0].unfilled_slots[0].slot_name == "verb"
