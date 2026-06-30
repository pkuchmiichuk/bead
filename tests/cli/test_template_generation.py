"""Integration tests for template generation CLI commands.

Tests template generation commands to ensure they:
1. Generate valid templates from patterns
2. Auto-detect slots from pattern
3. Support explicit slot specifications
4. Create template variants properly
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from bead.cli.resources import generate_template_variants, generate_templates
from bead.resources.template import Template


@pytest.fixture
def runner() -> CliRunner:
    """Create Click test runner."""
    return CliRunner()


@pytest.fixture
def base_template_file(tmp_path: Path) -> Path:
    """Create a base template file for variant testing."""
    template = Template(
        name="simple_transitive",
        template_string="{subject} {verb} {object}",
        slots={
            "subject": {"name": "subject", "required": True},
            "verb": {"name": "verb", "required": True},
            "object": {"name": "object", "required": True},
        },
    )

    file_path = tmp_path / "base_template.jsonl"
    file_path.write_text(template.model_dump_json() + "\n")
    return file_path


# ==================== Generate Templates Tests ====================


class TestGenerateTemplates:
    """Test template generation from patterns."""

    def test_generate_basic(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test basic template generation with auto-detected slots."""
        output = tmp_path / "template.jsonl"
        result = runner.invoke(
            generate_templates,
            [
                str(output),
                "--pattern",
                "{subject} {verb} {object}",
                "--name",
                "simple_transitive",
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output.exists()

        # Verify template structure
        template = Template.model_validate_json(output.read_text())

        assert template.name == "simple_transitive"
        assert template.template_string == "{subject} {verb} {object}"
        assert len(template.slots) == 3
        assert all(slot.required for slot in template.slots.values())

    def test_generate_with_explicit_slots(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test template generation with explicit slot specifications."""
        output = tmp_path / "template.jsonl"
        result = runner.invoke(
            generate_templates,
            [
                str(output),
                "--pattern",
                "{subject} {verb} {object}",
                "--name",
                "transitive",
                "--slot",
                "subject:true",
                "--slot",
                "verb:true",
                "--slot",
                "object:false",  # Optional object
            ],
        )

        assert result.exit_code == 0
        template = Template.model_validate_json(output.read_text())

        assert template.slots["subject"].required is True
        assert template.slots["verb"].required is True
        assert template.slots["object"].required is False

    def test_generate_with_description(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test template generation with description."""
        output = tmp_path / "template.jsonl"
        result = runner.invoke(
            generate_templates,
            [
                str(output),
                "--pattern",
                "{subject} {verb}",
                "--name",
                "intransitive",
                "--description",
                "Intransitive sentence template",
            ],
        )

        assert result.exit_code == 0
        template = Template.model_validate_json(output.read_text())

        assert template.description == "Intransitive sentence template"

    def test_generate_with_language_and_tags(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test template generation with language code and tags."""
        output = tmp_path / "template.jsonl"
        result = runner.invoke(
            generate_templates,
            [
                str(output),
                "--pattern",
                "{subject} {verb} {object}",
                "--name",
                "transitive",
                "--language-code",
                "eng",
                "--tags",
                "transitive,simple,declarative",
            ],
        )

        assert result.exit_code == 0
        template = Template.model_validate_json(output.read_text())

        assert template.language_code == "eng"
        assert set(template.tags) == {"transitive", "simple", "declarative"}

    def test_error_no_slots_in_pattern(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test error when pattern has no slot placeholders."""
        output = tmp_path / "template.jsonl"
        result = runner.invoke(
            generate_templates,
            [
                str(output),
                "--pattern",
                "This is a plain string",
                "--name",
                "bad_template",
            ],
        )

        assert result.exit_code == 1
        assert "No slot placeholders found" in result.output

    def test_error_slot_not_in_pattern(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test error when explicit slot not found in pattern."""
        output = tmp_path / "template.jsonl"
        result = runner.invoke(
            generate_templates,
            [
                str(output),
                "--pattern",
                "{subject} {verb}",
                "--name",
                "test",
                "--slot",
                "object:true",  # Not in pattern
            ],
        )

        assert result.exit_code == 1
        assert "not found in pattern" in result.output

    def test_error_invalid_slot_spec(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test error with invalid slot specification format."""
        output = tmp_path / "template.jsonl"
        result = runner.invoke(
            generate_templates,
            [
                str(output),
                "--pattern",
                "{subject} {verb}",
                "--name",
                "test",
                "--slot",
                "subject",  # Missing :true/:false
            ],
        )

        assert result.exit_code == 1
        assert "Invalid slot specification" in result.output

    def test_append_to_existing(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test appending multiple templates to same file."""
        output = tmp_path / "templates.jsonl"

        # Create first template
        result1 = runner.invoke(
            generate_templates,
            [
                str(output),
                "--pattern",
                "{subject} {verb}",
                "--name",
                "intransitive",
            ],
        )
        assert result1.exit_code == 0

        # Append second template
        result2 = runner.invoke(
            generate_templates,
            [
                str(output),
                "--pattern",
                "{subject} {verb} {object}",
                "--name",
                "transitive",
            ],
        )
        assert result2.exit_code == 0

        # Verify both templates exist
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 2

        template1 = Template.model_validate_json(lines[0])
        template2 = Template.model_validate_json(lines[1])

        assert template1.name == "intransitive"
        assert template2.name == "transitive"


# ==================== Generate Template Variants Tests ====================


class TestGenerateTemplateVariants:
    """Test template variant generation."""

    def test_generate_basic_variants(
        self,
        runner: CliRunner,
        tmp_path: Path,
        base_template_file: Path,
    ) -> None:
        """Test basic variant generation."""
        output = tmp_path / "variants.jsonl"
        result = runner.invoke(
            generate_template_variants,
            [
                str(base_template_file),
                str(output),
                "--max-variants",
                "5",
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output.exists()

        # Verify variants created
        lines = output.read_text().strip().split("\n")
        assert len(lines) == 5

        # Check variant structure
        for i, line in enumerate(lines):
            variant = Template.model_validate_json(line)
            assert f"variant_{i}" in variant.name
            assert variant.metadata["variant_index"] == i
            assert variant.metadata["base_template"] == "simple_transitive"

    def test_custom_name_pattern(
        self,
        runner: CliRunner,
        tmp_path: Path,
        base_template_file: Path,
    ) -> None:
        """Test variant generation with custom name pattern."""
        output = tmp_path / "variants.jsonl"
        result = runner.invoke(
            generate_template_variants,
            [
                str(base_template_file),
                str(output),
                "--name-pattern",
                "{base_name}_v{index}",
                "--max-variants",
                "3",
            ],
        )

        assert result.exit_code == 0
        lines = output.read_text().strip().split("\n")

        for i, line in enumerate(lines):
            variant = Template.model_validate_json(line)
            assert variant.name == f"simple_transitive_v{i}"

    def test_error_empty_base_file(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test error with empty base template file."""
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("")

        output = tmp_path / "variants.jsonl"
        result = runner.invoke(
            generate_template_variants,
            [
                str(empty_file),
                str(output),
            ],
        )

        assert result.exit_code == 1
        assert "empty" in result.output

    def test_slot_variants_not_implemented(
        self,
        runner: CliRunner,
        tmp_path: Path,
        base_template_file: Path,
    ) -> None:
        """Test that slot variant generation shows not implemented message."""
        slot_variants_file = tmp_path / "slot_variants.json"
        slot_variants_file.write_text('{"subject": ["{subject}", "{object}"]}')

        output = tmp_path / "variants.jsonl"
        result = runner.invoke(
            generate_template_variants,
            [
                str(base_template_file),
                str(output),
                "--slot-variants",
                str(slot_variants_file),
            ],
        )

        assert result.exit_code == 0
        assert output.exists()
