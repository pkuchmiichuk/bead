"""Tests for :mod:`bead.data_collection.records`."""

from __future__ import annotations

from bead.data_collection.records import jatos_results_to_annotation_records
from bead.evaluation import annotator_reliability


def _result(
    *,
    annotator: str = "P001",
    trials: list[dict[str, object]],
    annotator_key: str = "PROLIFIC_PID",
) -> dict[str, object]:
    return {
        "urlQueryParameters": {annotator_key: annotator},
        "worker_id": "worker-9",
        "data": trials,
    }


def test_basic_conversion() -> None:
    results = [
        _result(
            annotator="P001",
            trials=[
                {
                    "item_id": "i1",
                    "template_name": "completion",
                    "response": "yes",
                },
                {
                    "item_id": "i2",
                    "template_name": "completion",
                    "response": "no",
                },
            ],
        ),
        _result(
            annotator="P002",
            trials=[
                {
                    "item_id": "i1",
                    "template_name": "completion",
                    "response": "yes",
                },
            ],
        ),
    ]
    records = jatos_results_to_annotation_records(results)
    assert len(records) == 3
    assert records[0].annotator_id == "P001"
    assert records[0].item_id == "i1"
    assert records[0].question_name == "completion"
    assert records[0].response_label == "yes"
    assert records[2].annotator_id == "P002"


def test_falls_back_to_worker_id() -> None:
    results = [
        {
            "urlQueryParameters": {},
            "worker_id": "worker-42",
            "data": [
                {
                    "item_id": "i1",
                    "template_name": "q",
                    "response": "yes",
                },
            ],
        },
    ]
    records = jatos_results_to_annotation_records(results)
    assert len(records) == 1
    assert records[0].annotator_id == "worker-42"


def test_skips_non_question_trials() -> None:
    results = [
        _result(
            trials=[
                {"trial_type": "instructions", "response": None},
                {
                    "item_id": "i1",
                    "template_name": "q",
                    "response": "yes",
                },
            ],
        ),
    ]
    records = jatos_results_to_annotation_records(results)
    assert len(records) == 1


def test_numeric_response_coerced_to_str() -> None:
    results = [
        _result(
            trials=[
                {
                    "item_id": "i1",
                    "template_name": "rating",
                    "response": 5,
                },
                {
                    "item_id": "i2",
                    "template_name": "rating",
                    "response": 4.5,
                },
            ],
        ),
    ]
    records = jatos_results_to_annotation_records(results)
    assert records[0].response_label == "5"
    assert records[1].response_label == "4.5"


def test_response_object_with_response_key() -> None:
    results = [
        _result(
            trials=[
                {
                    "item_id": "i1",
                    "template_name": "q",
                    "response": {"response": "yes", "rt": 100},
                },
            ],
        ),
    ]
    records = jatos_results_to_annotation_records(results)
    assert records[0].response_label == "yes"


def test_missing_annotator_skipped() -> None:
    results = [
        {
            "urlQueryParameters": {},
            "data": [
                {"item_id": "i1", "template_name": "q", "response": "yes"},
            ],
        },
    ]
    records = jatos_results_to_annotation_records(results)
    assert records == ()


def test_custom_annotator_key() -> None:
    results = [
        _result(
            annotator="C42",
            annotator_key="custom_id",
            trials=[
                {"item_id": "i1", "template_name": "q", "response": "yes"},
            ],
        ),
    ]
    records = jatos_results_to_annotation_records(results, annotator_id_key="custom_id")
    assert records[0].annotator_id == "C42"


def test_pipes_into_annotator_reliability() -> None:
    """The bridge output composes with annotator_reliability end-to-end."""
    results = [
        _result(
            annotator="A",
            trials=[
                {"item_id": "i1", "template_name": "q", "response": "yes"},
                {"item_id": "i2", "template_name": "q", "response": "no"},
            ],
        ),
        _result(
            annotator="B",
            trials=[
                {"item_id": "i1", "template_name": "q", "response": "yes"},
                {"item_id": "i2", "template_name": "q", "response": "yes"},
            ],
        ),
    ]
    records = jatos_results_to_annotation_records(results)
    profiles = annotator_reliability(records)
    by_id = {p.annotator_id: p for p in profiles}
    # A used both labels → entropy = 1.0; B used one label → entropy = 0.0
    assert by_id["A"].entropy("q") == 1.0
    assert by_id["B"].entropy("q") == 0.0
