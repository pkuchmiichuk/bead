"""Tests for :mod:`bead.protocol.family`."""

from __future__ import annotations

import pytest

from bead.protocol.anchor import ResponseSpace, SemanticAnchor
from bead.protocol.context import ProtocolContext
from bead.protocol.drift import DriftGuard, StructuralDriftValidator
from bead.protocol.family import (
    AnnotationProtocol,
    QuestionFamily,
    QuestionRealization,
)
from bead.protocol.realization import TemplateRealization


def _anchor(
    name: str,
    *,
    canonical: str = "Does [[situation]] reach an endpoint?",
) -> SemanticAnchor:
    return SemanticAnchor(
        name=name,
        target_property=name,
        canonical_prompt=canonical,
        response_space=ResponseSpace(options=("no", "yes"), is_ordered=False),
        required_span_labels=frozenset({"situation"}),
    )


class TestQuestionFamily:
    """Tests for :class:`QuestionFamily`."""

    def test_default_realization_uses_canonical(self) -> None:
        family = QuestionFamily(anchor=_anchor("completion"))
        ctx = ProtocolContext()
        result = family.realize(ctx)
        assert isinstance(result, QuestionRealization)
        assert result.prompt == family.anchor.canonical_prompt
        assert result.passed_drift_check is True
        assert result.strategy_name == "TemplateRealization"

    def test_drift_failure_falls_back(self) -> None:
        anchor = _anchor("completion")
        family = QuestionFamily(
            anchor=anchor,
            realization=TemplateRealization(template="Bad realization."),
            drift_guard=DriftGuard(validators=[StructuralDriftValidator()]),
            fallback_on_drift=True,
        )
        result = family.realize(ProtocolContext())
        # Fell back to canonical, which has the [[situation]] tag
        assert result.prompt == anchor.canonical_prompt
        assert "fallback" in result.strategy_name

    def test_drift_failure_raises_when_no_fallback(self) -> None:
        anchor = _anchor("completion")
        family = QuestionFamily(
            anchor=anchor,
            realization=TemplateRealization(template="Bad realization."),
            drift_guard=DriftGuard(validators=[StructuralDriftValidator()]),
            fallback_on_drift=False,
        )
        with pytest.raises(ValueError, match="Drift validation failed"):
            family.realize(ProtocolContext())

    def test_is_always_applicable_default(self) -> None:
        family = QuestionFamily(anchor=_anchor("a"))
        assert family.is_always_applicable is True

    def test_explicit_condition_marks_conditional(self) -> None:
        family = QuestionFamily(
            anchor=_anchor("a"),
            condition=lambda ctx: ctx.target_upos == "VERB",
        )
        assert family.is_always_applicable is False
        assert family.is_applicable(ProtocolContext(target_upos="VERB")) is True
        assert family.is_applicable(ProtocolContext(target_upos="NOUN")) is False

    def test_depends_on_recorded(self) -> None:
        family = QuestionFamily(
            anchor=_anchor("uniformity"),
            depends_on=("change",),
        )
        assert family.depends_on == ("change",)


class TestAnnotationProtocol:
    """Tests for :class:`AnnotationProtocol`."""

    def test_construction_records_families(self) -> None:
        a = QuestionFamily(anchor=_anchor("a"))
        b = QuestionFamily(anchor=_anchor("b"))
        proto = AnnotationProtocol(families=[a, b], name="demo")
        assert len(proto) == 2
        assert proto.name == "demo"

    def test_duplicate_anchor_names_rejected(self) -> None:
        a = QuestionFamily(anchor=_anchor("dup"))
        b = QuestionFamily(anchor=_anchor("dup"))
        with pytest.raises(ValueError, match="Duplicate"):
            AnnotationProtocol(families=[a, b])

    def test_append_rejects_duplicate(self) -> None:
        a = QuestionFamily(anchor=_anchor("a"))
        proto = AnnotationProtocol(families=[a])
        with pytest.raises(ValueError, match="Duplicate"):
            proto.append(QuestionFamily(anchor=_anchor("a")))

    def test_family_by_name_lookup(self) -> None:
        a = QuestionFamily(anchor=_anchor("a"))
        b = QuestionFamily(anchor=_anchor("b"))
        proto = AnnotationProtocol(families=[a, b])
        assert proto.family_by_name("b") is b
        with pytest.raises(KeyError):
            proto.family_by_name("missing")

    def test_realize_all_threads_responses(self) -> None:
        """Second family conditioned on the first's response."""
        first = QuestionFamily(anchor=_anchor("change"))

        def is_dynamic(ctx: ProtocolContext) -> bool:
            return ctx.previous_responses.get("change") == "yes"

        second = QuestionFamily(
            anchor=_anchor("uniformity"),
            condition=is_dynamic,
            depends_on=("change",),
        )
        proto = AnnotationProtocol(families=[first, second])

        # With responses={'change': 'yes'} both questions fire
        results = proto.realize_all(
            ProtocolContext(),
            responses={"change": "yes"},
        )
        assert [r.anchor.name for r in results] == ["change", "uniformity"]

        # Without an explicit response, the placeholder is the first
        # option (`"no"`), so the second family's condition is false.
        results2 = proto.realize_all(ProtocolContext())
        assert [r.anchor.name for r in results2] == ["change"]

    def test_realize_all_rejects_unknown_response(self) -> None:
        proto = AnnotationProtocol(families=[QuestionFamily(anchor=_anchor("a"))])
        with pytest.raises(ValueError, match="unknown anchors"):
            proto.realize_all(ProtocolContext(), responses={"missing": "yes"})

    def test_self_dependency_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="depends on itself"):
            AnnotationProtocol(
                families=[
                    QuestionFamily(
                        anchor=_anchor("a"),
                        depends_on=("a",),
                    ),
                ],
            )

    def test_forward_dependency_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="not earlier"):
            AnnotationProtocol(
                families=[
                    QuestionFamily(
                        anchor=_anchor("a"),
                        depends_on=("b",),
                    ),
                    QuestionFamily(anchor=_anchor("b")),
                ],
            )

    def test_unknown_dependency_rejected_at_construction(self) -> None:
        with pytest.raises(ValueError, match="not earlier"):
            AnnotationProtocol(
                families=[
                    QuestionFamily(
                        anchor=_anchor("a"),
                        depends_on=("ghost",),
                    ),
                ],
            )

    def test_append_rejects_unknown_dependency(self) -> None:
        proto = AnnotationProtocol(
            families=[QuestionFamily(anchor=_anchor("first"))],
        )
        with pytest.raises(ValueError, match="not in the protocol"):
            proto.append(
                QuestionFamily(
                    anchor=_anchor("second"),
                    depends_on=("ghost",),
                ),
            )

    def test_append_rejects_self_dependency(self) -> None:
        proto = AnnotationProtocol(
            families=[QuestionFamily(anchor=_anchor("first"))],
        )
        with pytest.raises(ValueError, match="depends on itself"):
            proto.append(
                QuestionFamily(
                    anchor=_anchor("second"),
                    depends_on=("second",),
                ),
            )
