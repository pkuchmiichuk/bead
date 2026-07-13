"""Round-trip law tests for the judgment overlap lenses."""

from __future__ import annotations

import pytest
from lairs.records import judgment

from bead.evaluation.reliability import AnnotationRecord
from bead.interop.layers._convert import from_object_ref
from bead.interop.layers.judgment_lens import (
    ANNOTATION_RECORD_JUDGMENT,
    judgment_set_to_records,
    records_to_judgment_set,
)


def _record(
    *,
    annotator_id: str = "ann_1",
    item_id: str = "item_1",
    question_name: str = "completion",
    response_label: str = "yes",
) -> AnnotationRecord:
    return AnnotationRecord(
        annotator_id=annotator_id,
        item_id=item_id,
        question_name=question_name,
        response_label=response_label,
    )


class TestAnnotationRecordJudgment:
    """AnnotationRecord <-> layers judgment."""

    def test_view_shape(self) -> None:
        record = _record()
        view, _ = ANNOTATION_RECORD_JUDGMENT.forward(record)
        assert from_object_ref(view.item) == record.item_id
        assert view.categoricalValue == record.response_label

    def test_roundtrip_exact(self) -> None:
        record = _record()
        view, complement = ANNOTATION_RECORD_JUDGMENT.forward(record)
        assert ANNOTATION_RECORD_JUDGMENT.backward(view, complement) == record

    def test_roundtrip_through_serialization(self) -> None:
        record = _record()
        view, complement = ANNOTATION_RECORD_JUDGMENT.forward(record)
        view2 = judgment.Judgment.model_validate_json(view.model_dump_json())
        assert ANNOTATION_RECORD_JUDGMENT.backward(view2, complement) == record


class TestAnnotationSetJudgment:
    """tuple[AnnotationRecord, ...] <-> layers judgment set."""

    def test_roundtrip_exact(self) -> None:
        records = (
            _record(item_id="item_1", response_label="yes"),
            _record(item_id="item_2", response_label="no"),
            _record(item_id="item_3", response_label="yes"),
        )
        view, complement = records_to_judgment_set(records)
        assert view.agent is not None
        assert view.agent.id == "ann_1"
        assert len(view.judgments) == 3
        assert judgment_set_to_records(view, complement) == records

    def test_roundtrip_through_serialization(self) -> None:
        records = (
            _record(item_id="item_1", response_label="yes"),
            _record(item_id="item_2", response_label="no"),
        )
        view, complement = records_to_judgment_set(records)
        view2 = judgment.JudgmentSet.model_validate_json(view.model_dump_json())
        assert judgment_set_to_records(view2, complement) == records

    def test_mixed_annotator_raises(self) -> None:
        records = (
            _record(annotator_id="ann_1"),
            _record(annotator_id="ann_2"),
        )
        with pytest.raises(ValueError, match="share one"):
            records_to_judgment_set(records)

    def test_empty_raises(self) -> None:
        with pytest.raises(ValueError, match="at least one record"):
            records_to_judgment_set(())
