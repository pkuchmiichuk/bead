"""Tests for :mod:`bead.protocol.anchor`."""

from __future__ import annotations

import pytest

from bead.protocol.anchor import ResponseSpace, SemanticAnchor, SemanticPoles


class TestResponseSpace:
    """Tests for :class:`ResponseSpace`."""

    def test_construction_defaults(self) -> None:
        rs = ResponseSpace(options=("no", "yes"))
        assert rs.options == ("no", "yes")
        assert rs.is_ordered is True
        assert rs.semantic_poles is None

    def test_membership_and_length(self) -> None:
        rs = ResponseSpace(
            options=("definitely no", "unsure", "definitely yes"),
            is_ordered=True,
            semantic_poles=SemanticPoles(low="definitely no", high="definitely yes"),
        )
        assert len(rs) == 3
        assert "unsure" in rs
        assert "absent" not in rs

    def test_frozen_with_round_trip(self) -> None:
        rs = ResponseSpace(options=("a", "b"))
        rs2 = rs.with_(options=("a", "b", "c"))
        assert rs.options == ("a", "b")
        assert rs2.options == ("a", "b", "c")
        assert rs.id == rs2.id  # with_ preserves identity


class TestSemanticPoles:
    """Tests for :class:`SemanticPoles`."""

    def test_as_tuple(self) -> None:
        poles = SemanticPoles(low="never", high="always")
        assert poles.as_tuple() == ("never", "always")


class TestSemanticAnchor:
    """Tests for :class:`SemanticAnchor`."""

    def _build(self) -> SemanticAnchor:
        rs = ResponseSpace(
            options=("no", "yes"),
            is_ordered=False,
        )
        return SemanticAnchor(
            name="completion",
            target_property="telicity",
            canonical_prompt="Does [[situation]] reach an endpoint?",
            response_space=rs,
            required_span_labels=frozenset({"situation"}),
            required_keywords=frozenset({"endpoint"}),
            description="Whether the event culminates.",
        )

    def test_construction(self) -> None:
        anchor = self._build()
        assert anchor.name == "completion"
        assert anchor.target_property == "telicity"
        assert anchor.required_span_labels == frozenset({"situation"})
        assert anchor.required_keywords == frozenset({"endpoint"})
        assert anchor.max_drift == pytest.approx(0.3)

    def test_from_response_options(self) -> None:
        anchor = SemanticAnchor.from_response_options(
            name="freq",
            target_property="frequency",
            canonical_prompt="How often does [[situation]] happen?",
            options=("never", "sometimes", "always"),
            is_ordered=True,
            semantic_poles=SemanticPoles(low="never", high="always"),
            required_span_labels=frozenset({"situation"}),
        )
        assert anchor.response_space.is_ordered is True
        poles = anchor.response_space.semantic_poles
        assert poles is not None
        assert poles.as_tuple() == ("never", "always")
        assert anchor.required_span_labels == frozenset({"situation"})

    def test_with_round_trip(self) -> None:
        anchor = self._build()
        anchor2 = anchor.with_(max_drift=0.5)
        assert anchor.max_drift == pytest.approx(0.3)
        assert anchor2.max_drift == pytest.approx(0.5)
