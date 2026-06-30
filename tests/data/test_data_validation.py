"""Tests for validation utilities."""

from __future__ import annotations

from pathlib import Path
from uuid import UUID, uuid4

from bead.data.base import BeadBaseModel
from bead.data.metadata import MetadataTracker
from bead.data.serialization import write_jsonlines
from bead.data.validation import (
    ValidationReport,
    validate_jsonlines_file,
    validate_provenance_chain,
    validate_uuid_references,
)


class SimpleItem(BeadBaseModel):
    """Simple model with optional single UUID reference."""

    name: str
    parent_id: UUID | None = None


class ItemWithTuple(BeadBaseModel):
    """Model with a tuple of UUID references."""

    name: str
    parent_ids: tuple[UUID, ...] = ()


class Template(BeadBaseModel):
    """Template model."""

    name: str


# ValidationReport tests
def test_validation_report_creation() -> None:
    report = ValidationReport(valid=True)
    assert report.valid is True
    assert report.errors == ()
    assert report.warnings == ()
    assert report.object_count == 0


def test_validation_report_add_error_returns_invalid_copy() -> None:
    report = ValidationReport(valid=True).add_error("Something went wrong")
    assert report.valid is False
    assert report.errors == ("Something went wrong",)


def test_validation_report_add_warning_keeps_valid() -> None:
    report = ValidationReport(valid=True).add_warning("This might be an issue")
    assert report.valid is True
    assert report.warnings == ("This might be an issue",)


def test_validation_report_has_errors() -> None:
    report = ValidationReport(valid=True)
    assert not report.has_errors()
    assert report.add_error("error").has_errors()


def test_validation_report_has_warnings() -> None:
    report = ValidationReport(valid=True)
    assert not report.has_warnings()
    assert report.add_warning("warning").has_warnings()


# File validation tests
def test_validate_jsonlines_file_valid(tmp_path: Path) -> None:
    file_path = tmp_path / "valid.jsonl"
    objects = [SimpleItem(name="test1"), SimpleItem(name="test2")]
    write_jsonlines(objects, file_path)

    report = validate_jsonlines_file(file_path, SimpleItem)
    assert report.valid is True
    assert report.errors == ()
    assert report.object_count == 2


def test_validate_jsonlines_file_strict_mode_stops_at_first(tmp_path: Path) -> None:
    file_path = tmp_path / "invalid.jsonl"
    file_path.write_text('{"invalid1": "error"}\n{"invalid2": "error"}\n')

    report = validate_jsonlines_file(file_path, SimpleItem, strict=True)
    assert report.valid is False
    assert len(report.errors) == 1


def test_validate_jsonlines_file_nonstrict_collects_all(tmp_path: Path) -> None:
    file_path = tmp_path / "invalid.jsonl"
    file_path.write_text('{"invalid1": "error"}\n{"invalid2": "error"}\n')

    report = validate_jsonlines_file(file_path, SimpleItem, strict=False)
    assert report.valid is False
    assert len(report.errors) == 2


def test_validate_jsonlines_file_missing_file(tmp_path: Path) -> None:
    file_path = tmp_path / "nonexistent.jsonl"
    report = validate_jsonlines_file(file_path, SimpleItem)
    assert report.valid is False
    assert len(report.errors) == 1
    assert "File not found" in report.errors[0]


def test_validate_jsonlines_file_empty_lines_skipped(tmp_path: Path) -> None:
    file_path = tmp_path / "with_empty_lines.jsonl"
    obj = SimpleItem(name="test")
    file_path.write_text(
        "\n" + obj.model_dump_json() + "\n\n" + obj.model_dump_json() + "\n\n"
    )

    report = validate_jsonlines_file(file_path, SimpleItem)
    assert report.valid is True
    assert report.object_count == 2


def test_validate_jsonlines_file_counts_objects(tmp_path: Path) -> None:
    file_path = tmp_path / "count.jsonl"
    objects = [SimpleItem(name=f"test{i}") for i in range(10)]
    write_jsonlines(objects, file_path)

    report = validate_jsonlines_file(file_path, SimpleItem)
    assert report.valid is True
    assert report.object_count == 10


# Reference validation tests
def test_validate_uuid_references_valid() -> None:
    parent = SimpleItem(name="parent")
    child = SimpleItem(name="child", parent_id=parent.id)
    report = validate_uuid_references([child], {parent.id: parent})
    assert report.valid is True
    assert report.errors == ()


def test_validate_uuid_references_missing() -> None:
    child = SimpleItem(name="child", parent_id=uuid4())
    report = validate_uuid_references([child], {})
    assert report.valid is False
    assert len(report.errors) == 1
    assert "missing UUID" in report.errors[0]


def test_validate_uuid_references_tuple_of_uuids() -> None:
    parent1 = SimpleItem(name="parent1")
    parent2 = SimpleItem(name="parent2")
    child = ItemWithTuple(name="child", parent_ids=(parent1.id, parent2.id))
    pool = {parent1.id: parent1, parent2.id: parent2}
    report = validate_uuid_references([child], pool)
    assert report.valid is True
    assert report.errors == ()


def test_validate_uuid_references_no_uuid_fields() -> None:
    class SimpleModel(BeadBaseModel):
        name: str
        value: int

    obj = SimpleModel(name="test", value=42)
    report = validate_uuid_references([obj], {})
    assert report.valid is True
    assert report.errors == ()


# Provenance validation tests
def test_validate_provenance_chain_valid() -> None:
    template = Template(name="template")
    metadata = MetadataTracker().with_provenance(template.id, "Template", "filled_from")
    report = validate_provenance_chain(metadata, {template.id: template})
    assert report.valid is True


def test_validate_provenance_chain_missing_parent() -> None:
    metadata = MetadataTracker().with_provenance(uuid4(), "Template", "filled_from")
    report = validate_provenance_chain(metadata, {})
    assert report.valid is False
    assert "missing parent" in report.errors[0]


def test_validate_provenance_chain_type_mismatch() -> None:
    template = Template(name="template")
    metadata = MetadataTracker().with_provenance(
        template.id, "WrongType", "filled_from"
    )
    report = validate_provenance_chain(metadata, {template.id: template})
    assert report.valid is False
    assert "expected type" in report.errors[0]


def test_validate_provenance_chain_empty() -> None:
    report = validate_provenance_chain(MetadataTracker(), {})
    assert report.valid is True
    assert report.object_count == 0


def test_validate_provenance_chain_multiple_parents() -> None:
    template1 = Template(name="template1")
    template2 = Template(name="template2")
    metadata = (
        MetadataTracker()
        .with_provenance(template1.id, "Template", "filled_from")
        .with_provenance(template2.id, "Template", "derived_from")
    )
    repository = {template1.id: template1, template2.id: template2}
    report = validate_provenance_chain(metadata, repository)
    assert report.valid is True
    assert report.object_count == 2
