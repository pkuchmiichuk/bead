"""Tests for resource management CLI commands."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

from click.testing import CliRunner

from bead.cli.resources import resources
from bead.resources.lexical_item import LexicalItem
from bead.resources.lexicon import Lexicon
from bead.resources.template import Template


def test_create_lexicon_from_csv(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test creating lexicon from CSV file."""
    # Create CSV file
    csv_file = tmp_path / "verbs.csv"
    csv_file.write_text("lemma,pos\nrun,VERB\nwalk,VERB\n")

    output_file = tmp_path / "lexicon.jsonl"

    result = cli_runner.invoke(
        resources,
        [
            "create-lexicon",
            str(output_file),
            "--name",
            "verbs",
            "--from-csv",
            str(csv_file),
        ],
    )

    assert result.exit_code == 0
    assert output_file.exists()
    assert "Created lexicon 'verbs' with 2 items" in result.output

    # Verify content
    lexicon = Lexicon.from_jsonl(str(output_file), "verbs")
    assert len(lexicon) == 2


def test_create_lexicon_from_csv_with_features(
    cli_runner: CliRunner, tmp_path: Path
) -> None:
    """Test creating lexicon from CSV with features and attributes."""
    csv_file = tmp_path / "verbs.csv"
    csv_file.write_text(
        "lemma,pos,feature_tense,attr_frequency\n"
        "run,VERB,present,1000\n"
        "walked,VERB,past,500\n"
    )

    output_file = tmp_path / "lexicon.jsonl"

    result = cli_runner.invoke(
        resources,
        [
            "create-lexicon",
            str(output_file),
            "--name",
            "verbs",
            "--from-csv",
            str(csv_file),
            "--language-code",
            "eng",
        ],
    )

    assert result.exit_code == 0
    assert output_file.exists()

    # Verify features and attributes were extracted
    lexicon = Lexicon.from_jsonl(str(output_file), "verbs")
    items = list(lexicon)
    assert items[0].features.get("tense") == "present"
    assert items[0].features.get("frequency") == "1000"


def test_create_lexicon_from_json(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test creating lexicon from JSON file."""
    json_file = tmp_path / "verbs.json"
    json_data = [
        {"lemma": "run", "pos": "VERB"},
        {"lemma": "walk", "pos": "VERB"},
    ]
    json_file.write_text(json.dumps(json_data))

    output_file = tmp_path / "lexicon.jsonl"

    result = cli_runner.invoke(
        resources,
        [
            "create-lexicon",
            str(output_file),
            "--name",
            "verbs",
            "--from-json",
            str(json_file),
        ],
    )

    assert result.exit_code == 0
    assert output_file.exists()

    lexicon = Lexicon.from_jsonl(str(output_file), "verbs")
    assert len(lexicon) == 2


def test_create_lexicon_no_source(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test error when no source provided."""
    output_file = tmp_path / "lexicon.jsonl"

    result = cli_runner.invoke(
        resources,
        ["create-lexicon", str(output_file), "--name", "verbs"],
    )

    assert result.exit_code == 1
    assert "Must provide one source" in result.output


def test_create_lexicon_multiple_sources(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test error when multiple sources provided."""
    csv_file = tmp_path / "verbs.csv"
    csv_file.write_text("lemma\nrun\n")

    json_file = tmp_path / "verbs.json"
    json_file.write_text('[{"lemma": "run"}]')

    output_file = tmp_path / "lexicon.jsonl"

    result = cli_runner.invoke(
        resources,
        [
            "create-lexicon",
            str(output_file),
            "--name",
            "verbs",
            "--from-csv",
            str(csv_file),
            "--from-json",
            str(json_file),
        ],
    )

    assert result.exit_code == 1
    assert "Only one source allowed" in result.output


def test_create_lexicon_csv_missing_lemma(
    cli_runner: CliRunner, tmp_path: Path
) -> None:
    """Test error when CSV missing lemma column."""
    csv_file = tmp_path / "verbs.csv"
    csv_file.write_text("pos\nVERB\n")

    output_file = tmp_path / "lexicon.jsonl"

    result = cli_runner.invoke(
        resources,
        [
            "create-lexicon",
            str(output_file),
            "--name",
            "verbs",
            "--from-csv",
            str(csv_file),
        ],
    )

    assert result.exit_code == 1
    assert "must have 'lemma' column" in result.output


def test_create_lexicon_json_not_array(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test error when JSON file is not an array."""
    json_file = tmp_path / "verbs.json"
    json_file.write_text('{"lemma": "run"}')

    output_file = tmp_path / "lexicon.jsonl"

    result = cli_runner.invoke(
        resources,
        [
            "create-lexicon",
            str(output_file),
            "--name",
            "verbs",
            "--from-json",
            str(json_file),
        ],
    )

    assert result.exit_code == 1
    assert "must contain an array" in result.output


def test_create_template_simple(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test creating simple template."""
    output_file = tmp_path / "template.jsonl"

    result = cli_runner.invoke(
        resources,
        [
            "create-template",
            str(output_file),
            "--name",
            "transitive",
            "--template-string",
            "{subject} {verb} {object}",
        ],
    )

    assert result.exit_code == 0
    assert output_file.exists()
    assert "Created template 'transitive' with 3 slots" in result.output

    # Verify template
    with open(output_file) as f:
        template = Template.model_validate_json(f.readline())
        assert len(template.slots) == 3
        assert "subject" in template.slots
        assert "verb" in template.slots
        assert "object" in template.slots


def test_create_template_with_slot_specs(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test creating template with slot specifications."""
    output_file = tmp_path / "template.jsonl"

    result = cli_runner.invoke(
        resources,
        [
            "create-template",
            str(output_file),
            "--name",
            "transitive",
            "--template-string",
            "{subject} {verb} {object}",
            "--slot",
            "subject:true",
            "--slot",
            "verb:true",
            "--slot",
            "object:false",
        ],
    )

    assert result.exit_code == 0

    # Verify slot required flags
    with open(output_file) as f:
        template = Template.model_validate_json(f.readline())
        assert template.slots["subject"].required is True
        assert template.slots["verb"].required is True
        assert template.slots["object"].required is False


def test_create_template_with_language(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test creating template with language code."""
    output_file = tmp_path / "template.jsonl"

    result = cli_runner.invoke(
        resources,
        [
            "create-template",
            str(output_file),
            "--name",
            "test",
            "--template-string",
            "{x} {y}",
            "--language-code",
            "eng",
            "--description",
            "Test template",
        ],
    )

    assert result.exit_code == 0

    with open(output_file) as f:
        template = Template.model_validate_json(f.readline())
        assert template.language_code == "eng"
        assert template.description == "Test template"


def test_create_template_no_slots(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test error when template has no slots."""
    output_file = tmp_path / "template.jsonl"

    result = cli_runner.invoke(
        resources,
        [
            "create-template",
            str(output_file),
            "--name",
            "bad",
            "--template-string",
            "no slots here",
        ],
    )

    assert result.exit_code == 1
    assert "must contain at least one" in result.output


def test_create_template_invalid_slot_spec(
    cli_runner: CliRunner, tmp_path: Path
) -> None:
    """Test error with invalid slot specification."""
    output_file = tmp_path / "template.jsonl"

    result = cli_runner.invoke(
        resources,
        [
            "create-template",
            str(output_file),
            "--name",
            "test",
            "--template-string",
            "{x}",
            "--slot",
            "invalid_format",
        ],
    )

    assert result.exit_code == 1
    assert "Invalid slot definition" in result.output


def test_list_lexicons_empty_directory(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test listing lexicons in empty directory."""
    result = cli_runner.invoke(
        resources,
        ["list-lexicons", "--directory", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "No lexicon files found" in result.output


def test_list_lexicons(
    cli_runner: CliRunner, tmp_path: Path, mock_lexicon_file: Path
) -> None:
    """Test listing lexicons."""
    # Copy mock lexicon to test directory
    dest = tmp_path / "verbs.jsonl"
    shutil.copy(mock_lexicon_file, dest)

    result = cli_runner.invoke(
        resources,
        ["list-lexicons", "--directory", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "verbs.jsonl" in result.output


def test_list_lexicons_with_pattern(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test listing lexicons with pattern."""
    # Create multiple lexicons
    for name in ["verbs", "nouns", "adjectives"]:
        lexicon = Lexicon(name=name, language_code="eng")
        lexicon = lexicon.with_item(
            LexicalItem(lemma="test", language_code="eng", features={"pos": "VERB"})
        )
        output = tmp_path / f"{name}.jsonl"
        lexicon.to_jsonl(str(output))

    result = cli_runner.invoke(
        resources,
        ["list-lexicons", "--directory", str(tmp_path), "--pattern", "verb*.jsonl"],
    )

    assert result.exit_code == 0
    assert "verbs.jsonl" in result.output
    assert "nouns.jsonl" not in result.output


def test_list_templates_empty_directory(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test listing templates in empty directory."""
    result = cli_runner.invoke(
        resources,
        ["list-templates", "--directory", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "No template files found" in result.output


def test_list_templates(
    cli_runner: CliRunner, tmp_path: Path, mock_template_file: Path
) -> None:
    """Test listing templates."""
    dest = tmp_path / "templates.jsonl"
    shutil.copy(mock_template_file, dest)

    result = cli_runner.invoke(
        resources,
        ["list-templates", "--directory", str(tmp_path)],
    )

    assert result.exit_code == 0
    assert "templates.jsonl" in result.output
    assert "test_template" in result.output


def test_validate_lexicon_valid(cli_runner: CliRunner, mock_lexicon_file: Path) -> None:
    """Test validating valid lexicon."""
    result = cli_runner.invoke(
        resources,
        ["validate-lexicon", str(mock_lexicon_file)],
    )

    assert result.exit_code == 0
    assert "Lexicon is valid" in result.output


def test_validate_lexicon_invalid_json(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test validating lexicon with invalid JSON."""
    lexicon_file = tmp_path / "invalid.jsonl"
    lexicon_file.write_text("not valid json\n")

    result = cli_runner.invoke(
        resources,
        ["validate-lexicon", str(lexicon_file)],
    )

    assert result.exit_code == 1
    assert "Validation failed" in result.output


def test_validate_lexicon_invalid_item(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test validating lexicon with invalid item."""
    lexicon_file = tmp_path / "invalid.jsonl"
    lexicon_file.write_text('{"pos": "VERB"}\n')  # Missing required lemma

    result = cli_runner.invoke(
        resources,
        ["validate-lexicon", str(lexicon_file)],
    )

    assert result.exit_code == 1
    assert "Validation failed" in result.output


def test_validate_template_valid(
    cli_runner: CliRunner, mock_template_file: Path
) -> None:
    """Test validating valid template."""
    result = cli_runner.invoke(
        resources,
        ["validate-template", str(mock_template_file)],
    )

    assert result.exit_code == 0
    assert "Template file is valid" in result.output


def test_validate_template_invalid_json(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test validating template with invalid JSON."""
    template_file = tmp_path / "invalid.jsonl"
    template_file.write_text("not valid json\n")

    result = cli_runner.invoke(
        resources,
        ["validate-template", str(template_file)],
    )

    assert result.exit_code == 1
    assert "Validation failed" in result.output


def test_validate_template_invalid_template(
    cli_runner: CliRunner, tmp_path: Path
) -> None:
    """Test validating template with missing required fields."""
    template_file = tmp_path / "invalid.jsonl"
    template_file.write_text('{"name": "test"}\n')  # Missing template_string

    result = cli_runner.invoke(
        resources,
        ["validate-template", str(template_file)],
    )

    assert result.exit_code == 1
    assert "Validation failed" in result.output


def test_resources_help(cli_runner: CliRunner) -> None:
    """Test resources command help."""
    result = cli_runner.invoke(resources, ["--help"])

    assert result.exit_code == 0
    assert "Resource management commands" in result.output


def test_create_lexicon_help(cli_runner: CliRunner) -> None:
    """Test create-lexicon command help."""
    result = cli_runner.invoke(resources, ["create-lexicon", "--help"])

    assert result.exit_code == 0
    assert "Create a lexicon from various sources" in result.output


def test_create_template_help(cli_runner: CliRunner) -> None:
    """Test create-template command help."""
    result = cli_runner.invoke(resources, ["create-template", "--help"])

    assert result.exit_code == 0
    assert "Create a template with slots" in result.output
