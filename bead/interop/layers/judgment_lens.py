"""Lenses between bead annotation records and layers judgment records.

Maps bead's reliability inputs to their canonical
:mod:`lairs.records.judgment` counterparts:

- ``AnnotationRecord`` <-> a layers ``judgment``
- a tuple of ``AnnotationRecord`` sharing one annotator <-> a layers
  ``judgmentSet``

A single layers ``judgment`` has no slot for the annotator or the question name;
those belong to the parent ``judgmentSet`` and experiment. The lens therefore
keeps the bead framework identity, the annotator id, the question name, and the
raw item id / response label in the lens complement so reconstruction is exact.
The aggregate is modelled as two plain functions (a tuple is not a single
``dx.Model``), each one delegating to the per-record lens so every record
round-trips exactly.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import didactic.api as dx
from lairs.records import defs, judgment

from bead.data.base import JsonValue
from bead.evaluation.reliability import AnnotationRecord
from bead.interop.layers._convert import (
    apply_identity,
    identity_of,
    j_list,
    j_obj,
    j_str,
    object_ref,
)
from bead.interop.layers.participant_lens import agent_ref_of, participant_features

if TYPE_CHECKING:
    from bead.participants.models import Participant

# layers requires an experiment reference on every judgment set; bead annotation
# records carry none, so the view uses an empty at-uri and nothing round-trips.
_NO_EXPERIMENT_REF = ""


class AnnotationRecordJudgmentLens(
    dx.Lens[AnnotationRecord, judgment.Judgment, JsonValue]
):
    """Lossless lens ``AnnotationRecord <-> (layers judgment, complement)``.

    The layers ``judgment`` record (:class:`lairs.records.judgment.Judgment`) is
    the canonical representation of a single annotator response: it carries the
    item reference and the categorical value. The bead-only remainder (identity,
    annotator id, question name, and the raw item id / response label) travels in
    the lens complement, since a single ``judgment`` has no slot for an annotator
    or a question name.
    """

    def forward(self, record: AnnotationRecord) -> tuple[judgment.Judgment, JsonValue]:
        """Project an annotation record to a layers judgment and complement."""
        view = judgment.Judgment(
            item=object_ref(record.item_id),
            categoricalValue=record.response_label,
        )
        complement: JsonValue = {
            "identity": identity_of(record),
            "annotator_id": record.annotator_id,
            "question_name": record.question_name,
            "item_id": record.item_id,
            "response_label": record.response_label,
        }
        return view, complement

    def backward(
        self, view: judgment.Judgment, complement: JsonValue
    ) -> AnnotationRecord:
        """Reconstruct an annotation record from its layers judgment + complement."""
        comp = j_obj(complement)
        record = AnnotationRecord(
            annotator_id=j_str(comp["annotator_id"]),
            item_id=j_str(comp["item_id"]),
            question_name=j_str(comp["question_name"]),
            response_label=j_str(comp["response_label"]),
        )
        return apply_identity(record, comp["identity"])


ANNOTATION_RECORD_JUDGMENT = AnnotationRecordJudgmentLens()


def records_to_judgment_set(
    records: tuple[AnnotationRecord, ...],
    participant: Participant | None = None,
) -> tuple[judgment.JudgmentSet, JsonValue]:
    """Project records sharing one annotator to a layers judgment set.

    Groups a homogeneous tuple of annotation records (all from one annotator)
    into a layers ``judgmentSet``, projecting each record through
    :data:`ANNOTATION_RECORD_JUDGMENT`. The set's ``agent`` carries the shared
    annotator id and its ``createdAt`` is the earliest record creation time. The
    complement carries the shared annotator id and the per-record complements, so
    :func:`judgment_set_to_records` inverts it exactly.

    When the annotator's :class:`~bead.participants.models.Participant` is
    supplied, the set's ``agent`` is that participant's ``agentRef`` and its
    ``features`` carries the participant's study fields (demographics, sessions,
    consent), which the ``judgmentSet`` schema documents as the home for
    "annotator demographics, session metadata, completion time, payment info".
    The participant is supplementary provenance on the view: the records
    reconstruct from the complement alone, so ``judgment_set_to_records`` is
    unaffected.

    Parameters
    ----------
    records : tuple[AnnotationRecord, ...]
        Annotation records that all share one ``annotator_id``. Must be
        non-empty.
    participant : Participant | None, optional
        The annotator behind ``records``. When given, its identity and study
        fields enrich the judgment set's ``agent`` and ``features``.

    Returns
    -------
    tuple[judgment.JudgmentSet, JsonValue]
        The layers judgment set view and the lens complement.

    Raises
    ------
    ValueError
        If ``records`` is empty, or if the records do not all share the same
        ``annotator_id``.

    See Also
    --------
    judgment_set_to_records : Invert this projection.
    """
    if not records:
        raise ValueError(
            "records_to_judgment_set requires at least one record; "
            "an empty tuple has no annotator to attach to the judgment set."
        )
    annotator_ids = {record.annotator_id for record in records}
    if len(annotator_ids) != 1:
        raise ValueError(
            "records_to_judgment_set requires every record to share one "
            f"annotator_id, got {len(annotator_ids)} distinct ids: "
            f"{sorted(annotator_ids)}."
        )
    annotator_id = records[0].annotator_id

    judgments: list[judgment.Judgment] = []
    record_complements: list[JsonValue] = []
    for record in records:
        judgment_view, record_complement = ANNOTATION_RECORD_JUDGMENT.forward(record)
        judgments.append(judgment_view)
        record_complements.append(record_complement)

    agent = agent_ref_of(participant) if participant else defs.AgentRef(id=annotator_id)
    features = participant_features(participant) if participant else None
    view = judgment.JudgmentSet(
        agent=agent,
        createdAt=min(record.created_at for record in records),
        experimentRef=_NO_EXPERIMENT_REF,
        features=features,
        judgments=tuple(judgments),
    )
    complement: JsonValue = {
        "annotator_id": annotator_id,
        "record_complements": tuple(record_complements),
    }
    return view, complement


def judgment_set_to_records(
    view: judgment.JudgmentSet, complement: JsonValue
) -> tuple[AnnotationRecord, ...]:
    """Reconstruct annotation records from a layers judgment set + complement.

    Inverts :func:`records_to_judgment_set` by zipping each layers ``judgment``
    in the set with its per-record complement and delegating to
    :data:`ANNOTATION_RECORD_JUDGMENT`, so every record round-trips exactly.

    Parameters
    ----------
    view : judgment.JudgmentSet
        The layers judgment set produced by :func:`records_to_judgment_set`.
    complement : JsonValue
        The matching lens complement (per-record complements plus the shared
        annotator id).

    Returns
    -------
    tuple[AnnotationRecord, ...]
        The reconstructed annotation records, in their original order.

    See Also
    --------
    records_to_judgment_set : The forward projection.
    """
    comp = j_obj(complement)
    record_complements = j_list(comp["record_complements"])
    return tuple(
        ANNOTATION_RECORD_JUDGMENT.backward(judgment_view, record_complement)
        for judgment_view, record_complement in zip(
            view.judgments, record_complements, strict=True
        )
    )
