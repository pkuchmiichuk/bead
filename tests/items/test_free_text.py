"""Tests for free text item creation utilities."""

from __future__ import annotations

from uuid import uuid4

import didactic.api as dx
import pytest

from bead.items.free_text import (
    create_filtered_free_text_items,
    create_free_text_item,
    create_free_text_items_cross_product,
    create_free_text_items_from_groups,
    create_free_text_items_from_texts,
    create_free_text_items_with_context,
    create_paraphrase_item,
    create_wh_question_item,
)
from bead.items.item import Item


class TestCreateFreeTextItem:
    """Test create_free_text_item() function."""

    def test_create_basic_free_text_item(self) -> None:
        """Test creating a basic free text item."""
        item = create_free_text_item(
            "The cat sat on the mat.", prompt="Who sat on the mat?"
        )

        assert isinstance(item, Item)
        assert item.rendered_elements["text"] == "The cat sat on the mat."
        assert item.rendered_elements["prompt"] == "Who sat on the mat?"
        assert item.item_metadata["multiline"] is False

    def test_with_max_length(self) -> None:
        """Test item with max length."""
        item = create_free_text_item("Text", prompt="Question?", max_length=100)

        assert item.item_metadata["max_length"] == 100

    def test_with_min_length(self) -> None:
        """Test item with min length."""
        item = create_free_text_item("Text", prompt="Question?", min_length=10)

        assert item.item_metadata["min_length"] == 10

    def test_with_validation_pattern(self) -> None:
        """Test item with validation pattern."""
        item = create_free_text_item(
            "Text", prompt="Question?", validation_pattern=r"^.+$"
        )

        assert item.item_metadata["validation_pattern"] == r"^.+$"

    def test_multiline_textarea(self) -> None:
        """Test multiline (textarea) item."""
        item = create_free_text_item("Text", prompt="Question?", multiline=True)

        assert item.item_metadata["multiline"] is True

    def test_empty_text_raises_error(self) -> None:
        """Test that empty text raises error."""
        with pytest.raises(
            (ValueError, dx.ValidationError), match="text cannot be empty"
        ):
            create_free_text_item("", prompt="Question?")

        with pytest.raises(
            (ValueError, dx.ValidationError), match="text cannot be empty"
        ):
            create_free_text_item("   ", prompt="Question?")

    def test_empty_prompt_raises_error(self) -> None:
        """Test that empty prompt raises error."""
        with pytest.raises(
            (ValueError, dx.ValidationError), match="prompt is required"
        ):
            create_free_text_item("Text", prompt="")

        with pytest.raises(
            (ValueError, dx.ValidationError), match="prompt is required"
        ):
            create_free_text_item("Text", prompt="   ")

    def test_invalid_length_constraints_raise_error(self) -> None:
        """Test that invalid length constraints raise error."""
        with pytest.raises(
            (ValueError, dx.ValidationError), match="min_length.*cannot be greater than"
        ):
            create_free_text_item(
                "Text", prompt="Question?", min_length=100, max_length=50
            )

    def test_with_custom_template_id(self) -> None:
        """Test creating item with custom template ID."""
        template_id = uuid4()
        item = create_free_text_item(
            "Text", prompt="Question?", item_template_id=template_id
        )

        assert item.item_template_id == template_id

    def test_with_metadata(self) -> None:
        """Test creating item with metadata."""
        item = create_free_text_item(
            "Text",
            prompt="Question?",
            metadata={"task": "qa", "difficulty": "easy"},
        )

        assert item.item_metadata["task"] == "qa"
        assert item.item_metadata["difficulty"] == "easy"


class TestCreateFreeTextItemsFromTexts:
    """Test create_free_text_items_from_texts() function."""

    def test_basic_batch_creation(self) -> None:
        """Test basic batch creation from texts."""
        texts = ["Text 1", "Text 2", "Text 3"]

        items = create_free_text_items_from_texts(
            texts, prompt="Paraphrase this:", multiline=True, max_length=200
        )

        assert len(items) == 3
        assert all(isinstance(item, Item) for item in items)
        assert items[0].rendered_elements["text"] == "Text 1"
        assert items[0].rendered_elements["prompt"] == "Paraphrase this:"
        assert items[0].item_metadata["multiline"] is True
        assert items[0].item_metadata["max_length"] == 200

    def test_with_metadata_function(self) -> None:
        """Test with metadata function."""
        texts = ["Short", "Medium length"]

        items = create_free_text_items_from_texts(
            texts,
            prompt="Question?",
            metadata_fn=lambda text: {"original_length": len(text)},
        )

        assert items[0].item_metadata["original_length"] == len("Short")
        assert items[1].item_metadata["original_length"] == len("Medium length")


class TestCreateFreeTextItemsWithContext:
    """Test create_free_text_items_with_context() function."""

    def test_basic_context_pairs(self) -> None:
        """Test creating items with context and prompt pairs."""
        contexts = ["The cat sat on the mat."]
        prompts = ["What sat on the mat?"]

        items = create_free_text_items_with_context(contexts, prompts, max_length=50)

        assert len(items) == 1
        assert items[0].rendered_elements["text"] == "The cat sat on the mat."
        assert items[0].rendered_elements["prompt"] == "What sat on the mat?"
        assert items[0].item_metadata["context"] == "The cat sat on the mat."
        assert items[0].item_metadata["max_length"] == 50

    def test_multiple_pairs(self) -> None:
        """Test multiple context-prompt pairs."""
        contexts = ["Context 1", "Context 2"]
        prompts = ["Question 1?", "Question 2?"]

        items = create_free_text_items_with_context(contexts, prompts)

        assert len(items) == 2
        assert items[0].rendered_elements["text"] == "Context 1"
        assert items[0].rendered_elements["prompt"] == "Question 1?"
        assert items[1].rendered_elements["text"] == "Context 2"
        assert items[1].rendered_elements["prompt"] == "Question 2?"

    def test_mismatched_lengths_raises_error(self) -> None:
        """Test that mismatched lengths raise error."""
        contexts = ["Context 1", "Context 2"]
        prompts = ["Question 1?"]

        with pytest.raises(
            ValueError, match="contexts and prompts must have same length"
        ):
            create_free_text_items_with_context(contexts, prompts)

    def test_with_metadata_function(self) -> None:
        """Test with metadata function."""
        contexts = ["Context"]
        prompts = ["Prompt"]

        items = create_free_text_items_with_context(
            contexts,
            prompts,
            metadata_fn=lambda c, p: {"combined_length": len(c) + len(p)},
        )

        assert items[0].item_metadata["combined_length"] == len("Context") + len(
            "Prompt"
        )


class TestCreateFreeTextItemsFromGroups:
    """Test create_free_text_items_from_groups() function."""

    def test_basic_grouping(self) -> None:
        """Test basic grouping with preservation of group info."""
        source_items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "Simple sentence."},
                item_metadata={"type": "simple"},
            ),
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "Complex sentence."},
                item_metadata={"type": "complex"},
            ),
        ]

        free_text_items = create_free_text_items_from_groups(
            source_items,
            group_by=lambda i: i.item_metadata["type"],
            prompt="Paraphrase this:",
            multiline=True,
        )

        assert len(free_text_items) == 2
        assert all("group_key" in item.item_metadata for item in free_text_items)
        assert all("source_item_id" in item.item_metadata for item in free_text_items)

    def test_without_group_metadata(self) -> None:
        """Test grouping without including group metadata."""
        source_items = [
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "Text"},
                item_metadata={"group": "A"},
            )
        ]

        free_text_items = create_free_text_items_from_groups(
            source_items,
            group_by=lambda i: i.item_metadata["group"],
            prompt="Question?",
            include_group_metadata=False,
        )

        assert "group_key" not in free_text_items[0].item_metadata
        assert "source_item_id" in free_text_items[0].item_metadata


class TestCreateFreeTextItemsCrossProduct:
    """Test create_free_text_items_cross_product() function."""

    def test_basic_cross_product(self) -> None:
        """Test basic cross-product of texts and prompts."""
        texts = ["Text 1", "Text 2"]
        prompts = ["Paraphrase this:", "Summarize this:"]

        items = create_free_text_items_cross_product(
            texts, prompts, multiline=True, max_length=200
        )

        # 2 texts × 2 prompts = 4 items
        assert len(items) == 4
        assert items[0].rendered_elements["text"] == "Text 1"
        assert items[0].rendered_elements["prompt"] == "Paraphrase this:"
        assert items[0].item_metadata["multiline"] is True

    def test_with_metadata_function(self) -> None:
        """Test cross-product with metadata function."""
        texts = ["Text 1"]
        prompts = ["Prompt 1"]

        items = create_free_text_items_cross_product(
            texts,
            prompts,
            metadata_fn=lambda t, p: {"combined_length": len(t) + len(p)},
        )

        assert items[0].item_metadata["combined_length"] == len("Text 1") + len(
            "Prompt 1"
        )


class TestCreateFilteredFreeTextItems:
    """Test create_filtered_free_text_items() function."""

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

        free_text_items = create_filtered_free_text_items(
            items,
            prompt="Question?",
            item_filter=lambda i: i.item_metadata.get("valid", True),
        )

        assert len(free_text_items) == 1
        assert free_text_items[0].rendered_elements["text"] == "Valid"

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

        free_text_items = create_filtered_free_text_items(items, prompt="Question?")

        assert len(free_text_items) == 3


class TestCreateParaphraseItem:
    """Test create_paraphrase_item() function."""

    def test_basic_paraphrase(self) -> None:
        """Test creating a paraphrase item."""
        item = create_paraphrase_item("The quick brown fox jumps over the lazy dog.")

        assert (
            item.rendered_elements["text"]
            == "The quick brown fox jumps over the lazy dog."
        )
        assert item.rendered_elements["prompt"] == "Rewrite in your own words:"
        assert item.item_metadata["multiline"] is True
        assert item.item_metadata["max_length"] == 500

    def test_custom_instruction(self) -> None:
        """Test paraphrase with custom instruction."""
        item = create_paraphrase_item("Sentence", instruction="Rephrase this sentence:")

        assert item.rendered_elements["prompt"] == "Rephrase this sentence:"

    def test_with_metadata(self) -> None:
        """Test paraphrase item with custom metadata."""
        item = create_paraphrase_item("Sentence", metadata={"difficulty": "easy"})

        assert item.item_metadata["difficulty"] == "easy"
        assert item.item_metadata["multiline"] is True


class TestCreateWhQuestionItem:
    """Test create_wh_question_item() function."""

    def test_basic_wh_question(self) -> None:
        """Test creating a WH-question item."""
        item = create_wh_question_item("The dog chased the cat.", question_word="Who")

        assert item.rendered_elements["text"] == "The dog chased the cat."
        assert "Who" in item.rendered_elements["prompt"]
        assert item.item_metadata["multiline"] is False
        assert item.item_metadata["max_length"] == 100

    def test_different_question_words(self) -> None:
        """Test different question words."""
        for word in ["What", "When", "Where", "Why", "How"]:
            item = create_wh_question_item("Context", question_word=word)
            assert word in item.rendered_elements["prompt"]

    def test_with_metadata(self) -> None:
        """Test WH-question item with custom metadata."""
        item = create_wh_question_item(
            "Context", question_word="What", metadata={"task": "qa"}
        )

        assert item.item_metadata["task"] == "qa"
        assert item.item_metadata["multiline"] is False
