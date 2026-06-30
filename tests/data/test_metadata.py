"""Tests for metadata tracking models."""

from __future__ import annotations

import time
from uuid import uuid4

from bead.data.metadata import MetadataTracker, ProcessingRecord, ProvenanceRecord


def test_provenance_record_creation() -> None:
    parent_id = uuid4()
    record = ProvenanceRecord(
        parent_id=parent_id, parent_type="Template", relationship="filled_from"
    )
    assert record.parent_id == parent_id
    assert record.parent_type == "Template"
    assert record.relationship == "filled_from"
    assert record.id is not None
    assert record.created_at is not None


def test_provenance_record_has_timestamp() -> None:
    record = ProvenanceRecord(
        parent_id=uuid4(), parent_type="LexicalItem", relationship="derived_from"
    )
    assert record.timestamp is not None
    assert record.timestamp.tzinfo is not None


def test_provenance_record_serialization() -> None:
    parent_id = uuid4()
    record = ProvenanceRecord(
        parent_id=parent_id, parent_type="Template", relationship="filled_from"
    )
    data = record.model_dump()
    assert data["parent_type"] == "Template"
    assert data["relationship"] == "filled_from"

    restored = ProvenanceRecord.model_validate_json(record.model_dump_json())
    assert restored.parent_id == parent_id
    assert restored.parent_type == record.parent_type
    assert restored.relationship == record.relationship


def test_processing_record_creation() -> None:
    record = ProcessingRecord(
        operation="fill_template",
        parameters={"strategy": "exhaustive", "max_items": 100},
        operator="TemplateFiller-v1.0",
    )
    assert record.operation == "fill_template"
    assert record.parameters["strategy"] == "exhaustive"
    assert record.parameters["max_items"] == 100
    assert record.operator == "TemplateFiller-v1.0"
    assert record.timestamp is not None


def test_processing_record_default_parameters() -> None:
    record = ProcessingRecord(operation="test_operation")
    assert record.parameters == {}
    assert isinstance(record.parameters, dict)


def test_processing_record_optional_operator() -> None:
    record = ProcessingRecord(operation="op", parameters={"key": "value"})
    assert record.operator is None


def test_metadata_tracker_creation() -> None:
    tracker = MetadataTracker()
    assert tracker.provenance == ()
    assert tracker.processing_history == ()
    assert tracker.custom_metadata == {}
    assert tracker.id is not None


def test_with_provenance_returns_new_tracker_with_record() -> None:
    parent_id = uuid4()
    tracker = MetadataTracker().with_provenance(parent_id, "Template", "filled_from")
    assert len(tracker.provenance) == 1
    assert tracker.provenance[0].parent_id == parent_id
    assert tracker.provenance[0].parent_type == "Template"
    assert tracker.provenance[0].relationship == "filled_from"


def test_with_provenance_sets_timestamp() -> None:
    tracker = MetadataTracker().with_provenance(uuid4(), "Template", "filled_from")
    record = tracker.provenance[0]
    assert record.timestamp is not None
    assert record.timestamp.tzinfo is not None


def test_with_provenance_chain_preserves_order() -> None:
    parent1 = uuid4()
    parent2 = uuid4()
    parent3 = uuid4()
    tracker = (
        MetadataTracker()
        .with_provenance(parent1, "Template", "filled_from")
        .with_provenance(parent2, "LexicalItem", "derived_from")
        .with_provenance(parent3, "Constraint", "filtered_by")
    )
    assert len(tracker.provenance) == 3
    assert tracker.provenance[0].parent_id == parent1
    assert tracker.provenance[1].parent_id == parent2
    assert tracker.provenance[2].parent_id == parent3


def test_with_processing_returns_new_tracker_with_record() -> None:
    tracker = MetadataTracker().with_processing(
        "fill_template", {"strategy": "exhaustive"}
    )
    assert len(tracker.processing_history) == 1
    assert tracker.processing_history[0].operation == "fill_template"
    assert tracker.processing_history[0].parameters["strategy"] == "exhaustive"


def test_with_processing_with_parameters() -> None:
    params = {"strategy": "exhaustive", "max_items": 100, "timeout": 30}
    tracker = MetadataTracker().with_processing(
        "fill_template", params, "TemplateFiller-v1.0"
    )
    record = tracker.processing_history[0]
    assert record.parameters == params
    assert record.operator == "TemplateFiller-v1.0"


def test_with_processing_without_operator() -> None:
    tracker = MetadataTracker().with_processing("op", {"key": "value"})
    assert tracker.processing_history[0].operator is None


def test_get_provenance_chain() -> None:
    parent1 = uuid4()
    parent2 = uuid4()
    parent3 = uuid4()
    tracker = (
        MetadataTracker()
        .with_provenance(parent1, "Template", "filled_from")
        .with_provenance(parent2, "LexicalItem", "derived_from")
        .with_provenance(parent3, "Constraint", "filtered_by")
    )
    chain = tracker.get_provenance_chain()
    assert chain == (parent1, parent2, parent3)


def test_get_recent_processing() -> None:
    tracker = MetadataTracker()
    for i in range(5):
        tracker = tracker.with_processing(f"operation{i}", {"index": i})
        time.sleep(0.001)
    recent = tracker.get_recent_processing(n=3)
    assert len(recent) == 3
    assert recent[0].operation == "operation4"
    assert recent[1].operation == "operation3"
    assert recent[2].operation == "operation2"


def test_get_recent_processing_fewer_than_n() -> None:
    tracker = (
        MetadataTracker().with_processing("operation1").with_processing("operation2")
    )
    recent = tracker.get_recent_processing(n=5)
    assert len(recent) == 2
    assert recent[0].operation == "operation2"
    assert recent[1].operation == "operation1"


def test_custom_metadata_via_with_() -> None:
    tracker = MetadataTracker().with_(
        custom_metadata={"author": "Alice", "project": "bead", "version": "1.0"}
    )
    assert tracker.custom_metadata["author"] == "Alice"
    assert tracker.custom_metadata["project"] == "bead"
    assert tracker.custom_metadata["version"] == "1.0"


def test_metadata_serialization_roundtrip() -> None:
    parent_id = uuid4()
    tracker = (
        MetadataTracker()
        .with_provenance(parent_id, "Template", "filled_from")
        .with_processing("fill_template", {"strategy": "exhaustive"})
        .with_(custom_metadata={"test": "value"})
    )
    payload = tracker.model_dump_json()

    restored = MetadataTracker.model_validate_json(payload)
    assert len(restored.provenance) == 1
    assert restored.provenance[0].parent_id == parent_id
    assert len(restored.processing_history) == 1
    assert restored.processing_history[0].operation == "fill_template"
    assert restored.custom_metadata["test"] == "value"
