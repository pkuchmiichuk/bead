"""Per-annotator reliability summaries.

Sits next to :class:`bead.evaluation.InterAnnotatorMetrics`. Where the
inter-annotator metrics quantify *agreement* across raters, this
module quantifies *response diversity* of each individual rater. Low
within-annotator entropy is a flag that the annotator is collapsing
the response space (always picking ``"yes"``, always picking the
midpoint, and so on), which biases agreement metrics in misleading
directions.

The canonical input is a sequence of :class:`AnnotationRecord`
instances, each carrying an ``annotator_id``, ``item_id``,
``response_label``, and ``question_name``. The Shannon entropy of
each annotator's per-question response distribution is computed in
bits.
"""

from __future__ import annotations

import math
from collections.abc import Mapping, Sequence

import didactic.api as dx

from bead.data.base import BeadBaseModel
from bead.protocol.encoding import ResponseEncoding


class AnnotationRecord(BeadBaseModel):
    """A single annotator response.

    Canonical record shape consumed by reliability and inter-annotator
    metrics. Conforms structurally to
    :class:`bead.protocol.diagnostics.RecordLike`.

    Attributes
    ----------
    annotator_id : str
        Identifier of the annotator who produced the response.
    item_id : str
        Identifier of the annotation item.
    question_name : str
        Anchor name of the question that was answered.
    response_label : str
        The annotator's response label (must be one of the labels of
        the corresponding :class:`ResponseEncoding`).
    """

    annotator_id: str
    item_id: str
    question_name: str
    response_label: str


class AnnotatorReliability(BeadBaseModel):
    """Per-annotator reliability summary.

    Captures how diverse a single annotator's responses are within
    each question. Low entropy means the annotator collapses the
    response space.

    Attributes
    ----------
    annotator_id : str
        The annotator's identifier.
    n_responses : int
        Total responses from this annotator across all questions.
    response_distribution : dict[str, dict[str, int]]
        Per-question distribution of responses, keyed by anchor name
        and then by response label, with counts as values.
    entropy_per_question : dict[str, float]
        Per-question Shannon entropy in bits. ``0.0`` when the
        annotator only used one label for that question.

    Examples
    --------
    >>> rel = AnnotatorReliability(
    ...     annotator_id="ann_1",
    ...     n_responses=4,
    ...     response_distribution={
    ...         "completion": {"yes": 2, "no": 2},
    ...     },
    ...     entropy_per_question={"completion": 1.0},
    ... )
    >>> rel.entropy("completion")
    1.0
    >>> rel.entropy("missing") is None
    True
    """

    annotator_id: str
    n_responses: int = 0
    response_distribution: dict[str, dict[str, int]] = dx.field(default_factory=dict)
    entropy_per_question: dict[str, float] = dx.field(default_factory=dict)

    def entropy(self, question_name: str) -> float | None:
        """Return the Shannon entropy for one question, or ``None``.

        Parameters
        ----------
        question_name : str
            Anchor name to look up.

        Returns
        -------
        float | None
            Entropy in bits, or ``None`` if no responses were recorded
            for this question.
        """
        return self.entropy_per_question.get(question_name)


def _shannon_entropy(counts: Mapping[str, int]) -> float:
    """Return the Shannon entropy in bits of a count distribution.

    Parameters
    ----------
    counts : Mapping[str, int]
        Per-label counts. Zero-count labels are treated as absent.

    Returns
    -------
    float
        Entropy in bits. ``0.0`` for an empty or singleton
        distribution.
    """
    total = sum(counts.values())
    if total == 0:
        return 0.0
    entropy = 0.0
    for count in counts.values():
        if count <= 0:
            continue
        p = count / total
        entropy -= p * math.log2(p)
    return entropy


def annotator_reliability(
    records: Sequence[AnnotationRecord],
    encodings: Mapping[str, ResponseEncoding] | None = None,
) -> tuple[AnnotatorReliability, ...]:
    """Compute per-annotator reliability summaries.

    Groups records by annotator, then by question, and computes
    Shannon entropy in bits on each annotator-question label
    distribution. When ``encodings`` is supplied, response labels not
    present in the encoding for a question are silently skipped (a
    common case after schema evolution).

    Parameters
    ----------
    records : Sequence[AnnotationRecord]
        All records across questions and annotators.
    encodings : Mapping[str, ResponseEncoding] | None, optional
        Per-question encodings used to filter unrecognized labels.
        When ``None`` (the default), every label is counted.

    Returns
    -------
    tuple[AnnotatorReliability, ...]
        One summary per annotator, sorted by annotator id.

    Examples
    --------
    >>> records = [
    ...     AnnotationRecord(annotator_id="a1", item_id="i1",
    ...                      question_name="q", response_label="yes"),
    ...     AnnotationRecord(annotator_id="a1", item_id="i2",
    ...                      question_name="q", response_label="no"),
    ...     AnnotationRecord(annotator_id="a2", item_id="i1",
    ...                      question_name="q", response_label="yes"),
    ...     AnnotationRecord(annotator_id="a2", item_id="i2",
    ...                      question_name="q", response_label="yes"),
    ... ]
    >>> profiles = annotator_reliability(records)
    >>> [(p.annotator_id, p.entropy("q")) for p in profiles]
    [('a1', 1.0), ('a2', 0.0)]
    """
    by_annotator: dict[str, list[AnnotationRecord]] = {}
    for rec in records:
        by_annotator.setdefault(rec.annotator_id, []).append(rec)

    summaries: list[AnnotatorReliability] = []
    for ann_id in sorted(by_annotator):
        ann_records = by_annotator[ann_id]
        distribution: dict[str, dict[str, int]] = {}
        entropy_per_question: dict[str, float] = {}

        by_question: dict[str, list[str]] = {}
        for rec in ann_records:
            if encodings is not None:
                encoding = encodings.get(rec.question_name)
                if encoding is not None and rec.response_label not in encoding.labels:
                    continue
            by_question.setdefault(rec.question_name, []).append(rec.response_label)

        for q_name, labels in by_question.items():
            counts: dict[str, int] = {}
            for label in labels:
                counts[label] = counts.get(label, 0) + 1
            distribution[q_name] = counts
            entropy_per_question[q_name] = _shannon_entropy(counts)

        summaries.append(
            AnnotatorReliability(
                annotator_id=ann_id,
                n_responses=sum(len(v) for v in by_question.values()),
                response_distribution=distribution,
                entropy_per_question=entropy_per_question,
            )
        )

    return tuple(summaries)


def low_entropy_annotators(
    profiles: Sequence[AnnotatorReliability],
    *,
    threshold: float,
    question_name: str | None = None,
    require_min_responses: int = 1,
) -> tuple[str, ...]:
    """Return annotator ids whose entropy falls at or below a threshold.

    Useful for flagging annotators who collapse the response space.
    When ``question_name`` is supplied, the threshold is checked
    against that one question's entropy; otherwise it is checked
    against the *minimum* per-question entropy across every question
    the annotator answered.

    Parameters
    ----------
    profiles : Sequence[AnnotatorReliability]
        Reliability summaries to scan.
    threshold : float
        Entropy ceiling in bits. Annotators with entropy at or below
        this value are returned.
    question_name : str | None, optional
        Restrict the check to one question. Defaults to ``None`` (all
        questions, returning the minimum).
    require_min_responses : int, optional
        Skip annotators whose response count is below this value.
        Defaults to ``1``.

    Returns
    -------
    tuple[str, ...]
        Annotator ids meeting the criterion, sorted.

    Examples
    --------
    >>> profiles = (
    ...     AnnotatorReliability(annotator_id="a1", n_responses=10,
    ...                          entropy_per_question={"q": 0.0}),
    ...     AnnotatorReliability(annotator_id="a2", n_responses=10,
    ...                          entropy_per_question={"q": 0.95}),
    ... )
    >>> low_entropy_annotators(profiles, threshold=0.5)
    ('a1',)
    """
    flagged: list[str] = []
    for profile in profiles:
        if profile.n_responses < require_min_responses:
            continue
        if question_name is not None:
            entropy = profile.entropy(question_name)
            if entropy is None:
                continue
            if entropy <= threshold:
                flagged.append(profile.annotator_id)
        else:
            entropies = tuple(profile.entropy_per_question.values())
            if not entropies:
                continue
            if min(entropies) <= threshold:
                flagged.append(profile.annotator_id)

    return tuple(sorted(flagged))
