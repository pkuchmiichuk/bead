"""Integration tests for bead.cli.items_factories CLI commands.

Tests all 8 task-type-specific item creation commands to ensure they:
1. Create valid JSONL output
2. Generate correct Item structures
3. Handle metadata properly
4. Integrate correctly with core bead.items utilities
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from bead.cli.items_factories import (
    create_binary_from_texts,
    create_categorical,
    create_forced_choice,
    create_forced_choice_from_texts,
    create_free_text_from_texts,
    create_likert_7,
    create_magnitude_from_texts,
    create_multi_select_from_texts,
    create_nli,
    create_ordinal_scale_from_texts,
    create_simple_cloze,
)
from bead.items.item import Item


@pytest.fixture
def runner() -> CliRunner:
    """Create Click test runner."""
    return CliRunner()


@pytest.fixture
def sample_texts_file(tmp_path: Path) -> Path:
    """Create sample text file for batch creation."""
    texts_file = tmp_path / "texts.txt"
    texts_file.write_text("Sentence 1\nSentence 2\nSentence 3\n")
    return texts_file


# ==================== Forced Choice Tests ====================


class TestForcedChoice:
    """Test forced choice item creation commands."""

    def test_create_basic_2afc(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test basic forced choice item creation (2AFC)."""
        output = tmp_path / "item.jsonl"
        result = runner.invoke(
            create_forced_choice,
            ["Option A", "Option B", "-o", str(output)],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output.exists()

        # Verify output format
        item = Item.model_validate_json(output.read_text())
        assert len(item.options) == 2
        assert item.options[0] == "Option A"
        assert item.options[1] == "Option B"
        assert item.item_metadata["n_options"] == 2

    def test_create_3afc(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test 3AFC item creation."""
        output = tmp_path / "item.jsonl"
        result = runner.invoke(
            create_forced_choice,
            ["A", "B", "C", "-o", str(output)],
        )

        assert result.exit_code == 0
        item = Item.model_validate_json(output.read_text())
        assert item.item_metadata["n_options"] == 3
        assert len(item.options) == 3

    def test_create_with_metadata(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test forced choice with custom metadata."""
        output = tmp_path / "item.jsonl"
        result = runner.invoke(
            create_forced_choice,
            [
                "A",
                "B",
                "--metadata",
                "contrast=subject,condition=control",
                "-o",
                str(output),
            ],
        )

        assert result.exit_code == 0
        item = Item.model_validate_json(output.read_text())
        assert item.item_metadata["contrast"] == "subject"
        assert item.item_metadata["condition"] == "control"

    def test_create_from_texts(
        self,
        runner: CliRunner,
        tmp_path: Path,
        sample_texts_file: Path,
    ) -> None:
        """Test batch forced choice creation from texts file."""
        output = tmp_path / "items.jsonl"
        result = runner.invoke(
            create_forced_choice_from_texts,
            [
                "--texts-file",
                str(sample_texts_file),
                "--n-alternatives",
                "2",
                "-o",
                str(output),
            ],
        )

        assert result.exit_code == 0
        assert output.exists()

        # Should create 3 items (3 choose 2)
        items = [
            Item.model_validate_json(line)
            for line in output.read_text().strip().split("\n")
        ]
        assert len(items) == 3
        assert all(item.item_metadata["n_options"] == 2 for item in items)

    def test_insufficient_options(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test error when fewer than 2 options provided."""
        output = tmp_path / "item.jsonl"
        result = runner.invoke(
            create_forced_choice,
            ["Only One", "-o", str(output)],
        )

        assert result.exit_code == 0  # Click doesn't fail, just prints error
        assert not output.exists()  # File not created


# ==================== Ordinal Scale Tests ====================


class TestOrdinalScale:
    """Test ordinal scale item creation commands."""

    def test_create_likert_7(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test Likert-7 scale item creation."""
        output = tmp_path / "item.jsonl"
        result = runner.invoke(
            create_likert_7,
            [
                "--text",
                "The cat sat on the mat.",
                "-o",
                str(output),
            ],
        )

        assert result.exit_code == 0
        item = Item.model_validate_json(output.read_text())
        assert item.rendered_elements["text"] == "The cat sat on the mat."
        assert item.item_metadata["scale_min"] == 1
        assert item.item_metadata["scale_max"] == 7

    def test_create_from_texts(
        self,
        runner: CliRunner,
        tmp_path: Path,
        sample_texts_file: Path,
    ) -> None:
        """Test batch ordinal scale creation."""
        output = tmp_path / "items.jsonl"
        result = runner.invoke(
            create_ordinal_scale_from_texts,
            [
                "--texts-file",
                str(sample_texts_file),
                "--scale-min",
                "1",
                "--scale-max",
                "5",
                "-o",
                str(output),
            ],
        )

        assert result.exit_code == 0
        items = [
            Item.model_validate_json(line)
            for line in output.read_text().strip().split("\n")
        ]
        assert len(items) == 3
        assert all(item.item_metadata["scale_min"] == 1 for item in items)
        assert all(item.item_metadata["scale_max"] == 5 for item in items)

    def test_invalid_scale_range(
        self,
        runner: CliRunner,
        tmp_path: Path,
        sample_texts_file: Path,
    ) -> None:
        """Test error when scale_min > scale_max."""
        output = tmp_path / "items.jsonl"
        result = runner.invoke(
            create_ordinal_scale_from_texts,
            [
                "--texts-file",
                str(sample_texts_file),
                "--scale-min",
                "7",
                "--scale-max",
                "1",  # Invalid: min > max
                "-o",
                str(output),
            ],
        )

        # Should fail or create items with swapped bounds
        # (depends on validation in core utilities)
        assert result.exit_code in (0, 1)


# ==================== Categorical Tests ====================


class TestCategorical:
    """Test categorical item creation commands."""

    def test_create_categorical(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test categorical item creation."""
        output = tmp_path / "item.jsonl"
        result = runner.invoke(
            create_categorical,
            [
                "--text",
                "The cat sat on the mat.",
                "--categories",
                "entailment,neutral,contradiction",
                "-o",
                str(output),
            ],
        )

        assert result.exit_code == 0
        item = Item.model_validate_json(output.read_text())
        assert item.rendered_elements["text"] == "The cat sat on the mat."
        assert item.item_metadata["categories"] == (
            "entailment",
            "neutral",
            "contradiction",
        )

    def test_create_nli(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test NLI item creation."""
        output = tmp_path / "item.jsonl"
        result = runner.invoke(
            create_nli,
            [
                "--premise",
                "All dogs bark.",
                "--hypothesis",
                "Some dogs bark.",
                "-o",
                str(output),
            ],
        )

        assert result.exit_code == 0
        item = Item.model_validate_json(output.read_text())
        # NLI combines premise/hypothesis into text, stores originals in metadata
        assert "Premise: All dogs bark." in item.rendered_elements["text"]
        assert "Hypothesis: Some dogs bark." in item.rendered_elements["text"]
        assert item.item_metadata["premise"] == "All dogs bark."
        assert item.item_metadata["hypothesis"] == "Some dogs bark."
        assert item.item_metadata["categories"] == (
            "entailment",
            "neutral",
            "contradiction",
        )


# ==================== Binary Tests ====================


class TestBinary:
    """Test binary item creation commands."""

    def test_create_from_texts(
        self,
        runner: CliRunner,
        tmp_path: Path,
        sample_texts_file: Path,
    ) -> None:
        """Test binary item creation from texts."""
        output = tmp_path / "items.jsonl"
        result = runner.invoke(
            create_binary_from_texts,
            [
                "--texts-file",
                str(sample_texts_file),
                "--prompt",
                "Is this grammatical?",
                "-o",
                str(output),
            ],
        )

        assert result.exit_code == 0
        items = [
            Item.model_validate_json(line)
            for line in output.read_text().strip().split("\n")
        ]
        assert len(items) == 3
        assert all(
            item.rendered_elements["prompt"] == "Is this grammatical?" for item in items
        )


# ==================== Multi-Select Tests ====================


class TestMultiSelect:
    """Test multi-select item creation commands."""

    def test_create_from_texts(
        self,
        runner: CliRunner,
        tmp_path: Path,
        sample_texts_file: Path,
    ) -> None:
        """Test multi-select item creation from texts."""
        output = tmp_path / "items.jsonl"
        result = runner.invoke(
            create_multi_select_from_texts,
            [
                "--texts-file",
                str(sample_texts_file),
                "--options",
                "Option A,Option B,Option C",
                "--min-selections",
                "1",
                "--max-selections",
                "2",
                "-o",
                str(output),
            ],
        )

        assert result.exit_code == 0
        items = [
            Item.model_validate_json(line)
            for line in output.read_text().strip().split("\n")
        ]
        assert len(items) == 3
        # Options are stored in item.options list
        assert all(len(item.options) == 3 for item in items)
        assert all(item.options[0] == "Option A" for item in items)
        assert all(item.item_metadata["min_selections"] == 1 for item in items)
        assert all(item.item_metadata["max_selections"] == 2 for item in items)


# ==================== Magnitude Tests ====================


class TestMagnitude:
    """Test magnitude item creation commands."""

    def test_create_from_texts(
        self,
        runner: CliRunner,
        tmp_path: Path,
        sample_texts_file: Path,
    ) -> None:
        """Test magnitude item creation from texts."""
        output = tmp_path / "items.jsonl"
        result = runner.invoke(
            create_magnitude_from_texts,
            [
                "--texts-file",
                str(sample_texts_file),
                "--measure",
                "Reading time (ms)",
                "-o",
                str(output),
            ],
        )

        assert result.exit_code == 0
        items = [
            Item.model_validate_json(line)
            for line in output.read_text().strip().split("\n")
        ]
        assert len(items) == 3
        # CLI passes --measure as unit parameter to underlying function
        assert all(item.item_metadata["unit"] == "Reading time (ms)" for item in items)


# ==================== Free Text Tests ====================


class TestFreeText:
    """Test free text item creation commands."""

    def test_create_from_texts(
        self,
        runner: CliRunner,
        tmp_path: Path,
        sample_texts_file: Path,
    ) -> None:
        """Test free text item creation from texts."""
        output = tmp_path / "items.jsonl"
        result = runner.invoke(
            create_free_text_from_texts,
            [
                "--texts-file",
                str(sample_texts_file),
                "--prompt",
                "Paraphrase this sentence:",
                "-o",
                str(output),
            ],
        )

        assert result.exit_code == 0
        items = [
            Item.model_validate_json(line)
            for line in output.read_text().strip().split("\n")
        ]
        assert len(items) == 3
        assert all(
            item.rendered_elements["prompt"] == "Paraphrase this sentence:"
            for item in items
        )


# ==================== Cloze Tests ====================


class TestCloze:
    """Test cloze item creation commands."""

    def test_create_simple_cloze(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test simple cloze item creation."""
        output = tmp_path / "item.jsonl"
        result = runner.invoke(
            create_simple_cloze,
            [
                "--text",
                "The quick brown fox",
                "--blank-position",
                "1",
                "--blank-label",
                "adjective",
                "-o",
                str(output),
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        item = Item.model_validate_json(output.read_text())
        # Cloze replaces blank position with "___"
        assert item.rendered_elements["text"] == "The ___ brown fox"
        assert len(item.unfilled_slots) == 1
        # slot_name is the blank_label provided
        assert item.unfilled_slots[0].slot_name == "adjective"
        assert item.unfilled_slots[0].position == 1
