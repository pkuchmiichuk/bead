"""Tests for :mod:`bead.protocol.diagnostics`."""

from __future__ import annotations

from bead.protocol.anchor import ResponseSpace, SemanticAnchor
from bead.protocol.diagnostics import (
    ConditionalObservationValidator,
    DatasetReport,
    DiagnosticLevel,
    DiagnosticRecord,
    RecordLike,
)
from bead.protocol.family import AnnotationProtocol, QuestionFamily


def _anchor(name: str) -> SemanticAnchor:
    return SemanticAnchor(
        name=name,
        target_property=name,
        canonical_prompt=f"Question for [[situation]] ({name})?",
        response_space=ResponseSpace(options=("no", "yes"), is_ordered=False),
        required_span_labels=frozenset({"situation"}),
    )


class _Record:
    """Concrete RecordLike for tests."""

    def __init__(self, item_id: str, response_label: str, question_name: str) -> None:
        self.item_id = item_id
        self.response_label = response_label
        self.question_name = question_name


class TestDiagnosticLevel:
    """Tests for :class:`DiagnosticLevel`."""

    def test_str_values(self) -> None:
        assert DiagnosticLevel.INFO.value == "info"
        assert DiagnosticLevel.WARNING.value == "warning"
        assert DiagnosticLevel.ERROR.value == "error"


class TestDatasetReport:
    """Tests for :class:`DatasetReport`."""

    def test_immutable_add(self) -> None:
        r0 = DatasetReport(n_records_input=10)
        r1 = r0.add(DiagnosticLevel.WARNING, "cat", "msg")
        assert len(r0.findings) == 0
        assert len(r1.findings) == 1
        assert r1.findings[0].level == DiagnosticLevel.WARNING
        assert r1.has_warnings is True
        assert r1.has_errors is False

    def test_extend(self) -> None:
        rec1 = DiagnosticRecord(level=DiagnosticLevel.ERROR, category="c", message="m1")
        rec2 = DiagnosticRecord(level=DiagnosticLevel.INFO, category="c", message="m2")
        report = DatasetReport().extend([rec1, rec2])
        assert len(report.findings) == 2

    def test_with_coverage(self) -> None:
        report = DatasetReport().with_coverage("q1", 0.95)
        report = report.with_coverage("q2", 0.5)
        assert report.coverage == {"q1": 0.95, "q2": 0.5}

    def test_with_missing_embedding_dedups(self) -> None:
        report = DatasetReport()
        report = report.with_missing_embedding("i1")
        report = report.with_missing_embedding("i2")
        report = report.with_missing_embedding("i1")  # duplicate
        assert report.items_missing_embeddings == ("i1", "i2")

    def test_filters(self) -> None:
        report = (
            DatasetReport()
            .add(DiagnosticLevel.WARNING, "missing", "m1")
            .add(DiagnosticLevel.ERROR, "schema", "m2")
            .add(DiagnosticLevel.WARNING, "schema", "m3")
        )
        assert len(report.warnings) == 2
        assert len(report.errors) == 1
        assert len(report.by_category("schema")) == 2

    def test_summary(self) -> None:
        report = (
            DatasetReport(
                n_records_input=10,
                n_items=5,
                n_records_encoded=8,
                n_records_dropped=2,
            )
            .with_coverage("completion", 0.8)
            .add(DiagnosticLevel.WARNING, "missing", "msg")
        )
        text = report.summary()
        assert "5 items" in text
        assert "completion: 80.0%" in text
        assert "warnings: 1" in text


class TestRecordLike:
    """Tests for :class:`RecordLike` Protocol."""

    def test_record_conforms(self) -> None:
        rec = _Record(item_id="i1", response_label="yes", question_name="q1")
        assert isinstance(rec, RecordLike)


class TestConditionalObservationValidator:
    """Tests for :class:`ConditionalObservationValidator`."""

    def _protocol(self) -> AnnotationProtocol:
        change = QuestionFamily(anchor=_anchor("change"))
        uniformity = QuestionFamily(
            anchor=_anchor("uniformity"),
            depends_on=("change",),
        )
        return AnnotationProtocol(families=[change, uniformity])

    def test_passes_when_dependency_present(self) -> None:
        proto = self._protocol()
        records = {
            "change": [_Record("i1", "yes", "change")],
            "uniformity": [_Record("i1", "yes", "uniformity")],
        }
        validator = ConditionalObservationValidator()
        findings = validator.validate(records, proto)
        assert findings == ()

    def test_warns_on_missing_dependency(self) -> None:
        proto = self._protocol()
        records = {
            "uniformity": [_Record("i1", "yes", "uniformity")],
        }
        validator = ConditionalObservationValidator()
        findings = validator.validate(records, proto)
        assert len(findings) == 1
        assert findings[0].category == "conditional_missing_dependency"
        assert findings[0].item_id == "i1"

    def test_warns_on_inapplicable_value(self) -> None:
        proto = self._protocol()
        records = {
            "change": [_Record("i1", "no", "change")],
            "uniformity": [_Record("i1", "yes", "uniformity")],
        }
        validator = ConditionalObservationValidator(
            conditioning_values={"uniformity": {"yes"}},
        )
        findings = validator.validate(records, proto)
        assert len(findings) == 1
        assert findings[0].category == "conditional_inapplicable"

    def test_skips_unconditional_families(self) -> None:
        proto = AnnotationProtocol(families=[QuestionFamily(anchor=_anchor("solo"))])
        records = {
            "solo": [_Record("i1", "yes", "solo")],
        }
        validator = ConditionalObservationValidator()
        assert validator.validate(records, proto) == ()


def test_dataset_report_round_trip_through_with() -> None:
    """``with_`` preserves identity (UUID) but allows attribute updates."""
    r0 = DatasetReport(n_records_input=5)
    r1 = r0.with_(n_records_input=10)
    assert r0.n_records_input == 5
    assert r1.n_records_input == 10
    assert r0.id == r1.id
