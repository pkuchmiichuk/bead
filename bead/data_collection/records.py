"""Bridge from JATOS results to bead annotation records.

JATOS returns experimental results as nested JSON: each study run
contains a ``data`` array of trial objects, each carrying the
metadata serialized by
:func:`bead.deployment.jspsych.trials._serialize_item_metadata` and a
jsPsych ``response`` field.

This module is the single canonical conversion from that
representation into :class:`~bead.evaluation.AnnotationRecord`
instances, the input shape consumed by every reliability,
inter-annotator-agreement, and conditional-observation check in bead.
There is no other path from raw JATOS output to bead records.
"""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

from bead.data.base import JsonValue
from bead.evaluation.reliability import AnnotationRecord


def _coerce_response_label(response: JsonValue) -> str:
    """Normalize a jsPsych ``response`` value to a string label.

    Parameters
    ----------
    response : JsonValue
        The raw value emitted by jsPsych. For binary, categorical, and
        forced-choice tasks this is already a string label; for
        ordinal, magnitude, and similar numeric tasks it is an int or
        float; jsPsych may also wrap the response in an object with a
        ``"response"`` key for some plugin variants.

    Returns
    -------
    str
        String form suitable for :class:`AnnotationRecord.response_label`.
    """
    if isinstance(response, str):
        return response
    if isinstance(response, bool):
        return "true" if response else "false"
    if isinstance(response, int | float):
        return str(response)
    if isinstance(response, dict) and "response" in response:
        inner = response["response"]
        if isinstance(inner, str):
            return inner
        return str(inner)
    return str(response)


def _annotator_id(
    result: Mapping[str, Any],
    *,
    annotator_id_key: str,
) -> str | None:
    """Extract the annotator id from a JATOS result envelope.

    Looks first in the URL query parameters (``urlQueryParameters``)
    for the configured key (typically ``"PROLIFIC_PID"``), then falls
    back to the JATOS-assigned ``worker_id``.
    """
    url_params = result.get("urlQueryParameters")
    if isinstance(url_params, Mapping) and annotator_id_key in url_params:
        candidate = url_params[annotator_id_key]
        if isinstance(candidate, str) and candidate:
            return candidate
    worker_id = result.get("worker_id")
    if isinstance(worker_id, str) and worker_id:
        return worker_id
    if isinstance(worker_id, int):
        return str(worker_id)
    return None


def jatos_results_to_annotation_records(
    results: Iterable[Mapping[str, Any]],
    *,
    annotator_id_key: str = "PROLIFIC_PID",
) -> tuple[AnnotationRecord, ...]:
    """Convert a sequence of JATOS results to :class:`AnnotationRecord`s.

    Each JATOS result is expected to be the dict shape returned by
    :class:`~bead.data_collection.JATOSDataCollector` (a study run
    with a ``data`` field carrying jsPsych trial dicts). Trials that
    lack ``item_id`` or ``template_name`` are silently skipped, since
    they correspond to non-question trials such as instructions or
    consent.

    Parameters
    ----------
    results : Iterable[Mapping[str, Any]]
        JATOS result envelopes.
    annotator_id_key : str, optional
        Query-parameter key carrying the annotator identifier.
        Defaults to ``"PROLIFIC_PID"``.

    Returns
    -------
    tuple[AnnotationRecord, ...]
        One record per (item, template_name) trial. Records appear in
        result-then-trial order. Trials missing required fields are
        skipped.
    """
    records: list[AnnotationRecord] = []
    for result in results:
        annotator_id = _annotator_id(result, annotator_id_key=annotator_id_key)
        if annotator_id is None:
            continue
        trials = result.get("data")
        if not isinstance(trials, list):
            continue
        for trial in trials:
            if not isinstance(trial, Mapping):
                continue
            item_id = trial.get("item_id")
            question_name = trial.get("template_name")
            response = trial.get("response")
            if (
                not isinstance(item_id, str)
                or not isinstance(question_name, str)
                or response is None
            ):
                continue
            records.append(
                AnnotationRecord(
                    annotator_id=annotator_id,
                    item_id=item_id,
                    question_name=question_name,
                    response_label=_coerce_response_label(response),
                )
            )
    return tuple(records)
