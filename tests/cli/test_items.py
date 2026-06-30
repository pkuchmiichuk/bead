"""Tests for item CLI commands."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from click.testing import CliRunner

from bead.cli.items import items
from bead.items.item_template import (
    ItemElement,
    ItemTemplate,
    PresentationSpec,
    TaskSpec,
)
from bead.resources.lexical_item import LexicalItem
from bead.templates.filler import FilledTemplate


def test_list_empty_directory(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test listing items in empty directory."""
    result = cli_runner.invoke(
        items,
        ["list", "--directory", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "No files found" in result.output


def test_list_items(
    cli_runner: CliRunner, tmp_path: Path, mock_items_file: Path
) -> None:
    """Test listing items."""
    # mock_items_file is already in tmp_path
    result = cli_runner.invoke(
        items,
        ["list", "--directory", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "items.jsonl" in result.output


def test_validate_valid(cli_runner: CliRunner, mock_items_file: Path) -> None:
    """Test validating valid items file."""
    result = cli_runner.invoke(
        items,
        ["validate", str(mock_items_file)],
    )

    assert result.exit_code == 0
    assert "is valid" in result.output


def test_validate_invalid_json(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test validating items with invalid JSON."""
    items_file = tmp_path / "invalid.jsonl"
    items_file.write_text("not valid json\n")

    result = cli_runner.invoke(
        items,
        ["validate", str(items_file)],
    )

    assert result.exit_code == 1
    assert "Validation failed" in result.output


def test_show_stats(cli_runner: CliRunner, mock_items_file: Path) -> None:
    """Test showing statistics for items."""
    result = cli_runner.invoke(
        items,
        ["show-stats", str(mock_items_file)],
    )

    assert result.exit_code == 0
    assert "Total Items" in result.output


def test_items_help(cli_runner: CliRunner) -> None:
    """Test items command help."""
    result = cli_runner.invoke(items, ["--help"])

    assert result.exit_code == 0
    assert "Item construction commands" in result.output


def test_construct_simple(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test basic item construction without constraints."""
    # Create ItemTemplate file
    template = ItemTemplate(
        name="test_template",
        judgment_type="acceptability",
        task_type="binary",
        task_spec=TaskSpec(prompt="Is this natural?"),
        presentation_spec=PresentationSpec(mode="static"),
        elements=[
            ItemElement(
                element_type="text",
                element_name="sentence",
                content="Test sentence",
            )
        ],
        constraints=[],
    )

    template_file = tmp_path / "templates.jsonl"
    with open(template_file, "w") as f:
        f.write(template.model_dump_json() + "\n")

    # Create empty filled templates file (not used by text-only template)
    filled_file = tmp_path / "filled.jsonl"
    filled_file.write_text("")

    # Output file
    output_file = tmp_path / "items.jsonl"

    result = cli_runner.invoke(
        items,
        [
            "construct",
            "--item-template",
            str(template_file),
            "--filled-templates",
            str(filled_file),
            "--output",
            str(output_file),
            "--no-cache",
        ],
    )

    assert result.exit_code == 0
    assert "Created 1 item(s)" in result.output
    assert output_file.exists()


def test_construct_with_filled_templates(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test construction with filled template references."""
    # Create a filled template
    filled_id = uuid4()
    filled = FilledTemplate(
        template_id="t1",
        template_name="transitive",
        slot_fillers={
            "subject": LexicalItem(
                lemma="cat", language_code="eng", features={"pos": "NOUN"}
            ),
            "verb": LexicalItem(
                lemma="ran", language_code="eng", features={"pos": "VERB"}
            ),
        },
        rendered_text="The cat ran",
        strategy_name="exhaustive",
    )
    filled = filled.with_(id=filled_id)

    filled_file = tmp_path / "filled.jsonl"
    with open(filled_file, "w") as f:
        f.write(filled.model_dump_json() + "\n")

    # Create ItemTemplate that references the filled template
    template = ItemTemplate(
        name="test_template",
        judgment_type="acceptability",
        task_type="binary",
        task_spec=TaskSpec(prompt="Is this natural?"),
        presentation_spec=PresentationSpec(mode="static"),
        elements=[
            ItemElement(
                element_type="filled_template_ref",
                element_name="sentence",
                filled_template_ref_id=filled_id,
            )
        ],
        constraints=[],
    )

    template_file = tmp_path / "templates.jsonl"
    with open(template_file, "w") as f:
        f.write(template.model_dump_json() + "\n")

    output_file = tmp_path / "items.jsonl"

    result = cli_runner.invoke(
        items,
        [
            "construct",
            "--item-template",
            str(template_file),
            "--filled-templates",
            str(filled_file),
            "--output",
            str(output_file),
            "--no-cache",
        ],
    )

    assert result.exit_code == 0
    assert output_file.exists()


def test_construct_dry_run(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test construction dry run mode."""
    template = ItemTemplate(
        name="test_template",
        judgment_type="acceptability",
        task_type="binary",
        task_spec=TaskSpec(prompt="Natural?"),
        presentation_spec=PresentationSpec(mode="static"),
        elements=[
            ItemElement(
                element_type="text",
                element_name="sentence",
                content="Test",
            )
        ],
    )

    template_file = tmp_path / "templates.jsonl"
    with open(template_file, "w") as f:
        f.write(template.model_dump_json() + "\n")

    filled_file = tmp_path / "filled.jsonl"
    filled_file.write_text("")

    output_file = tmp_path / "items.jsonl"

    result = cli_runner.invoke(
        items,
        [
            "construct",
            "--item-template",
            str(template_file),
            "--filled-templates",
            str(filled_file),
            "--output",
            str(output_file),
            "--dry-run",
        ],
    )

    assert result.exit_code == 0
    assert "[DRY RUN]" in result.output
    assert not output_file.exists()  # No output in dry run


def test_construct_missing_template_file(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test error when template file is missing."""
    filled_file = tmp_path / "filled.jsonl"
    filled_file.write_text("")

    output_file = tmp_path / "items.jsonl"

    result = cli_runner.invoke(
        items,
        [
            "construct",
            "--item-template",
            str(tmp_path / "nonexistent.jsonl"),
            "--filled-templates",
            str(filled_file),
            "--output",
            str(output_file),
        ],
    )

    assert result.exit_code == 2  # Click file not found error


def test_construct_help(cli_runner: CliRunner) -> None:
    """Test construct command help."""
    result = cli_runner.invoke(items, ["construct", "--help"])

    assert result.exit_code == 0
    assert "Construct experimental items" in result.output
