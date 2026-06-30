"""Integration tests for template utility CLI commands.

Tests template utility commands:
1. filter-filled - Filter filled templates by criteria
2. merge-filled - Merge multiple filled template files
3. export-csv - Export to CSV format
4. export-json - Export to JSON array
5. sample-combinations - Stratified sampling
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from bead.cli.templates import (
    export_csv,
    export_json,
    filter_filled,
    merge_filled,
    sample_combinations,
)
from bead.resources.lexicon import Lexicon
from bead.resources.template import Template
from bead.resources.template_collection import TemplateCollection
from bead.templates.filler import FilledTemplate


@pytest.fixture
def runner() -> CliRunner:
    """Create Click test runner."""
    return CliRunner()


@pytest.fixture
def sample_filled_file(tmp_path: Path) -> Path:
    """Create sample filled templates file."""
    from bead.resources.lexical_item import LexicalItem  # noqa: PLC0415

    filled_templates = [
        FilledTemplate(
            template_id="00000000-0000-0000-0000-000000000001",
            template_name="template1",
            rendered_text="Short text",
            slot_fillers={"verb": LexicalItem(lemma="walk", language_code="eng")},
            template_slots={"verb": True},
            strategy_name="exhaustive",
        ),
        FilledTemplate(
            template_id="00000000-0000-0000-0000-000000000001",
            template_name="template1",
            rendered_text="This is a longer piece of text",
            slot_fillers={"verb": LexicalItem(lemma="run", language_code="eng")},
            template_slots={"verb": True},
            strategy_name="exhaustive",
        ),
        FilledTemplate(
            template_id="00000000-0000-0000-0000-000000000002",
            template_name="template2",
            rendered_text="Medium length",
            slot_fillers={"verb": LexicalItem(lemma="jump", language_code="eng")},
            template_slots={"verb": True},
            strategy_name="random",
        ),
    ]

    file_path = tmp_path / "filled.jsonl"
    with open(file_path, "w", encoding="utf-8") as f:
        for filled in filled_templates:
            f.write(filled.model_dump_json() + "\n")

    return file_path


@pytest.fixture
def sample_lexicon_file(tmp_path: Path) -> Path:
    """Create sample lexicon file."""
    lexicon = Lexicon(name="test_lexicon")
    from bead.resources.lexical_item import LexicalItem  # noqa: PLC0415

    items = [
        LexicalItem(lemma="walk", language_code="eng"),
        LexicalItem(lemma="run", language_code="eng"),
        LexicalItem(lemma="jump", language_code="eng"),
    ]

    for item in items:
        lexicon = lexicon.with_item(item)

    file_path = tmp_path / "lexicon.jsonl"
    lexicon.to_jsonl(str(file_path))
    return file_path


@pytest.fixture
def sample_template_file(tmp_path: Path) -> Path:
    """Create sample template file."""
    template = Template(
        name="test_template",
        template_string="{verb}",
        slots={"verb": {"name": "verb", "required": True}},
    )

    collection = TemplateCollection(name="test_templates")
    collection = collection.with_template(template)

    file_path = tmp_path / "templates.jsonl"
    collection.to_jsonl(str(file_path))
    return file_path


# ==================== Filter Tests ====================


class TestFilterFilled:
    """Test filtering filled templates."""

    def test_filter_by_min_length(
        self,
        runner: CliRunner,
        tmp_path: Path,
        sample_filled_file: Path,
    ) -> None:
        """Test filtering by minimum text length."""
        output = tmp_path / "filtered.jsonl"
        result = runner.invoke(
            filter_filled,
            [
                str(sample_filled_file),
                str(output),
                "--min-length",
                "15",
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output.exists()

        # Verify filtering
        # "Short text" = 10 chars (filtered out)
        # "This is a longer piece of text" = 30 chars (kept)
        # "Medium length" = 13 chars (filtered out)
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 1  # Only 1 item >= 15 chars

        for line in lines:
            filled = FilledTemplate.model_validate_json(line)
            assert len(filled.rendered_text) >= 15

    def test_filter_by_max_length(
        self,
        runner: CliRunner,
        tmp_path: Path,
        sample_filled_file: Path,
    ) -> None:
        """Test filtering by maximum text length."""
        output = tmp_path / "filtered.jsonl"
        result = runner.invoke(
            filter_filled,
            [
                str(sample_filled_file),
                str(output),
                "--max-length",
                "15",
            ],
        )

        assert result.exit_code == 0
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 2  # 2 items <= 15 chars

    def test_filter_by_template_name(
        self,
        runner: CliRunner,
        tmp_path: Path,
        sample_filled_file: Path,
    ) -> None:
        """Test filtering by template name."""
        output = tmp_path / "filtered.jsonl"
        result = runner.invoke(
            filter_filled,
            [
                str(sample_filled_file),
                str(output),
                "--template-name",
                "template1",
            ],
        )

        assert result.exit_code == 0
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 2

        for line in lines:
            filled = FilledTemplate.model_validate_json(line)
            assert filled.template_name == "template1"

    def test_filter_by_strategy(
        self,
        runner: CliRunner,
        tmp_path: Path,
        sample_filled_file: Path,
    ) -> None:
        """Test filtering by strategy name."""
        output = tmp_path / "filtered.jsonl"
        result = runner.invoke(
            filter_filled,
            [
                str(sample_filled_file),
                str(output),
                "--strategy",
                "random",
            ],
        )

        assert result.exit_code == 0
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 1

        filled = FilledTemplate.model_validate_json(lines[0])
        assert filled.strategy_name == "random"

    def test_filter_combined(
        self,
        runner: CliRunner,
        tmp_path: Path,
        sample_filled_file: Path,
    ) -> None:
        """Test filtering with multiple criteria."""
        output = tmp_path / "filtered.jsonl"
        result = runner.invoke(
            filter_filled,
            [
                str(sample_filled_file),
                str(output),
                "--min-length",
                "10",
                "--template-name",
                "template1",
            ],
        )

        assert result.exit_code == 0
        lines = output.read_text().strip().split("\n")
        # "Short text" (10 chars) and "This is a longer piece of text" (30 chars)
        # both match template1 and both >= 10 chars
        assert len(lines) == 2  # Both template1 items match


# ==================== Merge Tests ====================


class TestMergeFilled:
    """Test merging filled template files."""

    def test_merge_basic(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test basic merge of multiple files."""
        # Create two input files
        file1 = tmp_path / "file1.jsonl"
        file2 = tmp_path / "file2.jsonl"

        filled1 = FilledTemplate(
            template_id="00000000-0000-0000-0000-000000000001",
            template_name="template1",
            rendered_text="Text 1",
            slot_fillers={},
            template_slots={},
            strategy_name="exhaustive",
        )
        filled2 = FilledTemplate(
            template_id="00000000-0000-0000-0000-000000000002",
            template_name="template2",
            rendered_text="Text 2",
            slot_fillers={},
            template_slots={},
            strategy_name="random",
        )

        file1.write_text(filled1.model_dump_json() + "\n")
        file2.write_text(filled2.model_dump_json() + "\n")

        output = tmp_path / "merged.jsonl"
        result = runner.invoke(
            merge_filled,
            [str(file1), str(file2), str(output)],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output.exists()

        lines = output.read_text().strip().split("\n")
        assert len(lines) == 2

    def test_merge_with_deduplication(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test merge with duplicate removal."""
        # Create files with duplicate IDs
        file1 = tmp_path / "file1.jsonl"
        file2 = tmp_path / "file2.jsonl"

        filled = FilledTemplate(
            template_id="00000000-0000-0000-0000-000000000001",
            template_name="template1",
            rendered_text="Text 1",
            slot_fillers={},
            template_slots={},
            strategy_name="exhaustive",
        )

        # Same ID in both files
        file1.write_text(filled.model_dump_json() + "\n")
        file2.write_text(filled.model_dump_json() + "\n")

        output = tmp_path / "merged.jsonl"
        result = runner.invoke(
            merge_filled,
            [str(file1), str(file2), str(output), "--deduplicate"],
        )

        assert result.exit_code == 0
        assert "Removed 1 duplicate" in result.output

        lines = output.read_text().strip().split("\n")
        assert len(lines) == 1  # Duplicate removed

    def test_merge_error_no_files(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test error when no input files provided."""
        output = tmp_path / "merged.jsonl"
        result = runner.invoke(merge_filled, [str(output)])

        assert result.exit_code == 1
        assert "No input files" in result.output


# ==================== Export CSV Tests ====================


class TestExportCSV:
    """Test exporting to CSV format."""

    def test_export_basic(
        self,
        runner: CliRunner,
        tmp_path: Path,
        sample_filled_file: Path,
    ) -> None:
        """Test basic CSV export."""
        output = tmp_path / "export.csv"
        result = runner.invoke(
            export_csv,
            [str(sample_filled_file), str(output)],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output.exists()

        # Verify CSV structure
        with open(output, encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

            assert len(rows) == 3
            assert set(rows[0].keys()) == {
                "id",
                "template_id",
                "template_name",
                "rendered_text",
                "strategy_name",
                "slot_count",
            }

            # Verify data
            assert rows[0]["rendered_text"] == "Short text"
            assert rows[0]["strategy_name"] == "exhaustive"


# ==================== Export JSON Tests ====================


class TestExportJSON:
    """Test exporting to JSON array format."""

    def test_export_basic(
        self,
        runner: CliRunner,
        tmp_path: Path,
        sample_filled_file: Path,
    ) -> None:
        """Test basic JSON export."""
        output = tmp_path / "export.json"
        result = runner.invoke(
            export_json,
            [str(sample_filled_file), str(output)],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output.exists()

        # Verify JSON structure
        data = json.loads(output.read_text())
        assert isinstance(data, list)
        assert len(data) == 3

        # Verify each item is valid
        for item in data:
            FilledTemplate.model_validate_json(json.dumps(item))

    def test_export_pretty(
        self,
        runner: CliRunner,
        tmp_path: Path,
        sample_filled_file: Path,
    ) -> None:
        """Test pretty-printed JSON export."""
        output = tmp_path / "export.json"
        result = runner.invoke(
            export_json,
            [str(sample_filled_file), str(output), "--pretty"],
        )

        assert result.exit_code == 0
        content = output.read_text()

        # Verify pretty printing (should have indentation)
        assert "  " in content  # Has indentation
        assert "\n" in content  # Has newlines

    def test_export_error_empty_file(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test error with empty input file."""
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("")

        output = tmp_path / "export.json"
        result = runner.invoke(
            export_json,
            [str(empty_file), str(output)],
        )

        assert result.exit_code == 1
        assert "No valid filled templates" in result.output


# ==================== Sample Combinations Tests ====================


class TestSampleCombinations:
    """Test stratified sampling of combinations."""

    def test_sample_basic(
        self,
        runner: CliRunner,
        tmp_path: Path,
        sample_template_file: Path,
        sample_lexicon_file: Path,
    ) -> None:
        """Test basic combination sampling."""
        output = tmp_path / "samples.jsonl"
        result = runner.invoke(
            sample_combinations,
            [
                str(sample_template_file),
                str(sample_lexicon_file),
                str(output),
                "--n-samples",
                "2",
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output.exists()

        # Verify samples created
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 2

        for line in lines:
            FilledTemplate.model_validate_json(line)

    def test_sample_with_seed(
        self,
        runner: CliRunner,
        tmp_path: Path,
        sample_template_file: Path,
        sample_lexicon_file: Path,
    ) -> None:
        """Test sampling with reproducible seed."""
        output1 = tmp_path / "samples1.jsonl"
        output2 = tmp_path / "samples2.jsonl"

        # Generate two samples with same seed
        result1 = runner.invoke(
            sample_combinations,
            [
                str(sample_template_file),
                str(sample_lexicon_file),
                str(output1),
                "--n-samples",
                "3",
                "--seed",
                "42",
            ],
        )

        result2 = runner.invoke(
            sample_combinations,
            [
                str(sample_template_file),
                str(sample_lexicon_file),
                str(output2),
                "--n-samples",
                "3",
                "--seed",
                "42",
            ],
        )

        assert result1.exit_code == 0
        assert result2.exit_code == 0

        # Results should have same rendered text (UUIDs/timestamps will differ)
        lines1 = [json.loads(line) for line in output1.read_text().strip().split("\n")]
        lines2 = [json.loads(line) for line in output2.read_text().strip().split("\n")]

        assert len(lines1) == len(lines2)
        rendered_texts1 = [line["rendered_text"] for line in lines1]
        rendered_texts2 = [line["rendered_text"] for line in lines2]
        assert rendered_texts1 == rendered_texts2  # Same content with same seed

    def test_sample_with_language_filter(
        self,
        runner: CliRunner,
        tmp_path: Path,
        sample_template_file: Path,
        sample_lexicon_file: Path,
    ) -> None:
        """Test sampling with language code filter."""
        output = tmp_path / "samples.jsonl"
        result = runner.invoke(
            sample_combinations,
            [
                str(sample_template_file),
                str(sample_lexicon_file),
                str(output),
                "--n-samples",
                "2",
                "--language-code",
                "eng",
            ],
        )

        assert result.exit_code == 0
        assert output.exists()
