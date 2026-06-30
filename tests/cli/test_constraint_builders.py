"""Integration tests for bead.cli.constraint_builders CLI commands.

Tests constraint creation commands to ensure they:
1. Create valid JSONL constraint output
2. Generate correct Constraint structures for 3 types
3. Handle context variables properly (extensional)
4. Validate DSL expressions (intensional, relational)
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from bead.cli.constraint_builders import create_constraint
from bead.resources.constraints import Constraint


@pytest.fixture
def runner() -> CliRunner:
    """Create Click test runner."""
    return CliRunner()


@pytest.fixture
def values_file(tmp_path: Path) -> Path:
    """Create sample values file for extensional constraints."""
    file_path = tmp_path / "values.txt"
    file_path.write_text("walk\nrun\njump\nswim\n")
    return file_path


# ==================== Extensional Constraint Tests ====================


class TestExtensionalConstraints:
    """Test extensional constraint creation."""

    def test_create_from_file(
        self, runner: CliRunner, tmp_path: Path, values_file: Path
    ) -> None:
        """Test extensional constraint creation from values file."""
        output = tmp_path / "constraint.jsonl"
        result = runner.invoke(
            create_constraint,
            [
                str(output),
                "--type",
                "extensional",
                "--slot",
                "verb",
                "--values-file",
                str(values_file),
                "--description",
                "Motion verbs",
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output.exists()

        # Verify constraint structure
        constraint = Constraint.model_validate_json(output.read_text())

        assert "self.lemma in allowed_values" in constraint.expression
        # Sets are serialized as lists in JSON
        assert set(constraint.context["allowed_values"]) == {
            "walk",
            "run",
            "jump",
            "swim",
        }
        assert constraint.description == "Motion verbs"

    def test_create_from_comma_separated(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test extensional constraint from comma-separated values."""
        output = tmp_path / "constraint.jsonl"
        result = runner.invoke(
            create_constraint,
            [
                str(output),
                "--type",
                "extensional",
                "--slot",
                "noun",
                "--values",
                "cat,dog,bird",
            ],
        )

        assert result.exit_code == 0
        constraint = Constraint.model_validate_json(output.read_text())

        # Sets are serialized as lists in JSON
        assert set(constraint.context["allowed_values"]) == {"cat", "dog", "bird"}

    def test_create_custom_property(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test extensional constraint with custom property."""
        output = tmp_path / "constraint.jsonl"
        result = runner.invoke(
            create_constraint,
            [
                str(output),
                "--type",
                "extensional",
                "--slot",
                "verb",
                "--values",
                "V,N,ADJ",
                "--prop-name",
                "pos",
            ],
        )

        assert result.exit_code == 0
        constraint = Constraint.model_validate_json(output.read_text())

        assert "self.pos in allowed_values" in constraint.expression

    def test_error_without_values(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test error when no values provided."""
        output = tmp_path / "constraint.jsonl"
        result = runner.invoke(
            create_constraint,
            [
                str(output),
                "--type",
                "extensional",
                "--slot",
                "verb",
            ],
        )

        assert result.exit_code == 1
        assert "require --values-file or --values" in result.output

    def test_error_without_slot(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test error when no slot specified."""
        output = tmp_path / "constraint.jsonl"
        result = runner.invoke(
            create_constraint,
            [
                str(output),
                "--type",
                "extensional",
                "--values",
                "cat,dog",
            ],
        )

        assert result.exit_code == 1
        assert "require --slot" in result.output


# ==================== Intensional Constraint Tests ====================


class TestIntensionalConstraints:
    """Test intensional constraint creation."""

    def test_create_basic(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test basic intensional constraint creation."""
        output = tmp_path / "constraint.jsonl"
        result = runner.invoke(
            create_constraint,
            [
                str(output),
                "--type",
                "intensional",
                "--slot",
                "verb",
                "--expression",
                "self.pos == 'VERB' and self.features.tense == 'past'",
                "--description",
                "Past tense verbs",
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output.exists()

        constraint = Constraint.model_validate_json(output.read_text())

        assert (
            constraint.expression
            == "self.pos == 'VERB' and self.features.tense == 'past'"
        )
        assert constraint.description == "Past tense verbs"
        assert constraint.context == {}

    def test_create_without_slot(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test intensional constraint without slot (template-level)."""
        output = tmp_path / "constraint.jsonl"
        result = runner.invoke(
            create_constraint,
            [
                str(output),
                "--type",
                "intensional",
                "--expression",
                "self.pos == 'NOUN'",
            ],
        )

        assert result.exit_code == 0
        assert "No --slot specified" in result.output

    def test_error_without_expression(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test error when no expression provided."""
        output = tmp_path / "constraint.jsonl"
        result = runner.invoke(
            create_constraint,
            [
                str(output),
                "--type",
                "intensional",
                "--slot",
                "verb",
            ],
        )

        assert result.exit_code == 1
        assert "require --expression" in result.output

    def test_error_invalid_expression(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test error when expression doesn't start with 'self.' for slot."""
        output = tmp_path / "constraint.jsonl"
        result = runner.invoke(
            create_constraint,
            [
                str(output),
                "--type",
                "intensional",
                "--slot",
                "verb",
                "--expression",
                "pos == 'VERB'",  # Missing 'self.'
            ],
        )

        assert result.exit_code == 1
        assert "must start with 'self.'" in result.output


# ==================== Relational Constraint Tests ====================


class TestRelationalConstraints:
    """Test relational constraint creation."""

    def test_create_basic(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test basic relational constraint creation."""
        output = tmp_path / "constraint.jsonl"
        result = runner.invoke(
            create_constraint,
            [
                str(output),
                "--type",
                "relational",
                "--relation",
                "subject.features.number == verb.features.number",
                "--description",
                "Subject-verb agreement",
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output.exists()

        constraint = Constraint.model_validate_json(output.read_text())

        assert (
            constraint.expression == "subject.features.number == verb.features.number"
        )
        assert constraint.description == "Subject-verb agreement"

    def test_create_if_then(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test relational constraint with if-then logic."""
        output = tmp_path / "constraint.jsonl"
        result = runner.invoke(
            create_constraint,
            [
                str(output),
                "--type",
                "relational",
                "--relation",
                "det.lemma != 'a' or noun.features.number == 'singular'",
            ],
        )

        assert result.exit_code == 0

    def test_error_without_relation(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test error when no relation provided."""
        output = tmp_path / "constraint.jsonl"
        result = runner.invoke(
            create_constraint,
            [
                str(output),
                "--type",
                "relational",
            ],
        )

        assert result.exit_code == 1
        assert "require --relation" in result.output

    def test_error_with_self(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test error when relation uses 'self.' (should be multi-slot)."""
        output = tmp_path / "constraint.jsonl"
        result = runner.invoke(
            create_constraint,
            [
                str(output),
                "--type",
                "relational",
                "--relation",
                "self.pos == 'VERB'",
            ],
        )

        assert result.exit_code == 1
        assert "do not use 'self.'" in result.output


# ==================== Output File Tests ====================


class TestOutputFile:
    """Test output file handling."""

    def test_append_to_existing(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test appending to existing constraint file."""
        output = tmp_path / "constraints.jsonl"

        # Create first constraint
        result1 = runner.invoke(
            create_constraint,
            [
                str(output),
                "--type",
                "extensional",
                "--slot",
                "verb",
                "--values",
                "walk,run",
            ],
        )
        assert result1.exit_code == 0

        # Append second constraint
        result2 = runner.invoke(
            create_constraint,
            [
                str(output),
                "--type",
                "intensional",
                "--slot",
                "noun",
                "--expression",
                "self.pos == 'NOUN'",
            ],
        )
        assert result2.exit_code == 0

        # Verify both constraints exist
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 2

        constraint1 = Constraint.model_validate_json(lines[0])
        constraint2 = Constraint.model_validate_json(lines[1])

        assert "allowed_values" in constraint1.context
        assert "self.pos == 'NOUN'" in constraint2.expression

    def test_create_directory(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test creating output directory if it doesn't exist."""
        output = tmp_path / "nested" / "dir" / "constraint.jsonl"
        result = runner.invoke(
            create_constraint,
            [
                str(output),
                "--type",
                "extensional",
                "--slot",
                "verb",
                "--values",
                "walk",
            ],
        )

        assert result.exit_code == 0
        assert output.exists()
        assert output.parent.exists()
