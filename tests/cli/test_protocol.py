"""Integration tests for the ``bead protocol`` CLI."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
import yaml
from click.testing import CliRunner

from bead.cli.protocol import protocol


def _project(tmp_path: Path, *, with_families: bool = True) -> Path:
    """Write a minimal bead.toml-equivalent YAML config and return its path."""
    cfg: dict[str, object] = {
        "profile": "default",
        "paths": {
            "data_dir": str(tmp_path),
            "output_dir": str(tmp_path / "out"),
            "cache_dir": str(tmp_path / ".cache"),
        },
        "protocol": {
            "name": "test-protocol",
            "drift": {
                "min_length": 5,
                "require_question_mark": True,
            },
            "families": (
                [
                    {
                        "anchor": {
                            "name": "completion",
                            "target_property": "telicity",
                            "canonical_prompt": (
                                "Does [[situation]] reach an endpoint?"
                            ),
                            "options": ["no", "yes"],
                            "is_ordered": False,
                            "required_span_labels": ["situation"],
                        },
                        "realization_kind": "template",
                    }
                ]
                if with_families
                else []
            ),
        },
    }
    config_path = tmp_path / "bead.yaml"
    config_path.write_text(yaml.safe_dump(cfg))
    (tmp_path / "out").mkdir(exist_ok=True)
    return config_path


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


def test_validate_reports_families(runner: CliRunner, tmp_path: Path) -> None:
    config_path = _project(tmp_path)
    result = runner.invoke(protocol, ["validate", "--config-file", str(config_path)])
    assert result.exit_code == 0, result.output
    assert "test-protocol" in result.output
    assert "completion" in result.output


def test_validate_empty_protocol_still_passes(
    runner: CliRunner, tmp_path: Path
) -> None:
    config_path = _project(tmp_path, with_families=False)
    result = runner.invoke(protocol, ["validate", "--config-file", str(config_path)])
    assert result.exit_code == 0
    assert "0 families" in result.output


def test_realize_writes_realizations(runner: CliRunner, tmp_path: Path) -> None:
    config_path = _project(tmp_path)
    contexts_file = tmp_path / "contexts.jsonl"
    contexts = [
        {
            "sentence": f"Mary built sandcastle {i}.",
            "target_lemma": "build",
            "target_form": "built",
            "target_upos": "VERB",
            "target_position": 2,
            "target_span_text": f"built sandcastle {i}",
            "target_span_positions": [2, 3, 4],
            "target_id": f"item-{i}",
        }
        for i in range(3)
    ]
    contexts_file.write_text("\n".join(json.dumps(c) for c in contexts) + "\n")
    output_file = tmp_path / "realizations.jsonl"
    result = runner.invoke(
        protocol,
        [
            "realize",
            str(contexts_file),
            str(output_file),
            "--config-file",
            str(config_path),
        ],
    )
    assert result.exit_code == 0, result.output
    lines = output_file.read_text().strip().splitlines()
    assert len(lines) == 3
    parsed = [json.loads(line) for line in lines]
    for record in parsed:
        assert "prompt" in record
        assert "[[situation]]" in record["prompt"]


def test_realize_emit_items(runner: CliRunner, tmp_path: Path) -> None:
    config_path = _project(tmp_path)
    contexts_file = tmp_path / "contexts.jsonl"
    contexts_file.write_text(
        json.dumps(
            {
                "sentence": "Mary built a sandcastle.",
                "target_lemma": "build",
                "target_form": "built",
                "target_upos": "VERB",
                "target_position": 2,
                "target_span_text": "built a sandcastle",
                "target_span_positions": [2, 3, 4],
                "target_id": "item-0",
            }
        )
        + "\n"
    )
    output_file = tmp_path / "items.jsonl"
    result = runner.invoke(
        protocol,
        [
            "realize",
            str(contexts_file),
            str(output_file),
            "--config-file",
            str(config_path),
            "--emit-items",
        ],
    )
    assert result.exit_code == 0, result.output
    lines = output_file.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert "item_template_id" in record
    assert "spans" in record


def test_items_writes_templates(runner: CliRunner, tmp_path: Path) -> None:
    config_path = _project(tmp_path)
    output_file = tmp_path / "templates.jsonl"
    result = runner.invoke(
        protocol,
        ["items", str(output_file), "--config-file", str(config_path)],
    )
    assert result.exit_code == 0, result.output
    lines = output_file.read_text().strip().splitlines()
    assert len(lines) == 1
    record = json.loads(lines[0])
    assert record["name"] == "completion"
    assert record["task_type"] == "binary"


def test_realize_empty_protocol_errors(runner: CliRunner, tmp_path: Path) -> None:
    config_path = _project(tmp_path, with_families=False)
    contexts_file = tmp_path / "contexts.jsonl"
    contexts_file.write_text("")
    output_file = tmp_path / "out.jsonl"
    result = runner.invoke(
        protocol,
        [
            "realize",
            str(contexts_file),
            str(output_file),
            "--config-file",
            str(config_path),
        ],
    )
    assert result.exit_code == 1
    assert "empty" in result.output.lower()
