"""Tests for :mod:`bead.evaluation.reliability`."""

from __future__ import annotations

import math

import pytest

from bead.evaluation.reliability import (
    AnnotationRecord,
    AnnotatorReliability,
    annotator_reliability,
    low_entropy_annotators,
)
from bead.protocol.encoding import ResponseEncoding, ScaleType


def _record(
    annotator: str, item: str, label: str, question: str = "q"
) -> AnnotationRecord:
    return AnnotationRecord(
        annotator_id=annotator,
        item_id=item,
        question_name=question,
        response_label=label,
    )


class TestAnnotatorReliability:
    """Tests for :class:`AnnotatorReliability`."""

    def test_entropy_lookup(self) -> None:
        rel = AnnotatorReliability(
            annotator_id="a1",
            n_responses=4,
            response_distribution={"q": {"yes": 2, "no": 2}},
            entropy_per_question={"q": 1.0},
        )
        assert rel.entropy("q") == pytest.approx(1.0)
        assert rel.entropy("missing") is None


class TestAnnotatorReliabilityFunction:
    """Tests for :func:`annotator_reliability`."""

    def test_uniform_distribution_max_entropy(self) -> None:
        records = [
            _record("a1", "i1", "yes"),
            _record("a1", "i2", "no"),
        ]
        profiles = annotator_reliability(records)
        assert len(profiles) == 1
        assert profiles[0].entropy("q") == pytest.approx(1.0)

    def test_constant_response_zero_entropy(self) -> None:
        records = [
            _record("a1", "i1", "yes"),
            _record("a1", "i2", "yes"),
            _record("a1", "i3", "yes"),
        ]
        profiles = annotator_reliability(records)
        assert profiles[0].entropy("q") == pytest.approx(0.0)
        assert profiles[0].n_responses == 3

    def test_three_way_uniform(self) -> None:
        records = [
            _record("a", "i1", "a"),
            _record("a", "i2", "b"),
            _record("a", "i3", "c"),
        ]
        profiles = annotator_reliability(records)
        # Shannon entropy of uniform 3-way = log2(3)
        assert profiles[0].entropy("q") == pytest.approx(math.log2(3))

    def test_grouped_by_question(self) -> None:
        records = [
            _record("a1", "i1", "yes", question="q1"),
            _record("a1", "i2", "no", question="q1"),
            _record("a1", "i1", "always", question="q2"),
            _record("a1", "i2", "always", question="q2"),
        ]
        profiles = annotator_reliability(records)
        assert profiles[0].entropy("q1") == pytest.approx(1.0)
        assert profiles[0].entropy("q2") == pytest.approx(0.0)

    def test_filters_unknown_labels_with_encoding(self) -> None:
        encoding = ResponseEncoding(
            name="q",
            n_levels=2,
            scale_type=ScaleType.BINARY,
            labels=("no", "yes"),
        )
        records = [
            _record("a1", "i1", "yes"),
            _record("a1", "i2", "maybe"),  # unknown label
            _record("a1", "i3", "no"),
        ]
        profiles = annotator_reliability(records, {"q": encoding})
        assert profiles[0].n_responses == 2
        assert profiles[0].response_distribution["q"] == {"yes": 1, "no": 1}

    def test_sorted_by_annotator_id(self) -> None:
        records = [
            _record("c", "i1", "yes"),
            _record("a", "i1", "yes"),
            _record("b", "i1", "yes"),
        ]
        profiles = annotator_reliability(records)
        assert [p.annotator_id for p in profiles] == ["a", "b", "c"]


class TestLowEntropyAnnotators:
    """Tests for :func:`low_entropy_annotators`."""

    def _profiles(self) -> tuple[AnnotatorReliability, ...]:
        return (
            AnnotatorReliability(
                annotator_id="lazy",
                n_responses=10,
                entropy_per_question={"q1": 0.0, "q2": 0.5},
            ),
            AnnotatorReliability(
                annotator_id="diligent",
                n_responses=10,
                entropy_per_question={"q1": 0.95, "q2": 1.5},
            ),
            AnnotatorReliability(
                annotator_id="newcomer",
                n_responses=1,
                entropy_per_question={"q1": 0.0},
            ),
        )

    def test_global_min_threshold(self) -> None:
        flagged = low_entropy_annotators(self._profiles(), threshold=0.5)
        # 'lazy' has min 0.0 (<= 0.5); 'newcomer' min 0.0 but min responses
        # is 1 by default so it's still flagged
        assert flagged == ("lazy", "newcomer")

    def test_per_question_threshold(self) -> None:
        flagged = low_entropy_annotators(
            self._profiles(),
            threshold=0.5,
            question_name="q1",
        )
        assert flagged == ("lazy", "newcomer")

    def test_min_responses_filter(self) -> None:
        flagged = low_entropy_annotators(
            self._profiles(),
            threshold=0.5,
            require_min_responses=5,
        )
        assert flagged == ("lazy",)  # newcomer dropped

    def test_no_matches(self) -> None:
        flagged = low_entropy_annotators(self._profiles(), threshold=-0.1)
        assert flagged == ()

    def test_unknown_question_returns_empty(self) -> None:
        flagged = low_entropy_annotators(
            self._profiles(),
            threshold=0.5,
            question_name="missing",
        )
        assert flagged == ()
