"""Tests for :mod:`bead.protocol.drift`."""

from __future__ import annotations

from collections.abc import Sequence

import pytest

from bead.protocol.anchor import ResponseSpace, SemanticAnchor
from bead.protocol.context import ProtocolContext
from bead.protocol.drift import (
    DriftGuard,
    DriftScore,
    DriftValidator,
    EmbeddingAdapter,
    EmbeddingDriftValidator,
    PerplexityAdapter,
    PerplexityDriftValidator,
    StructuralDriftValidator,
)


def _anchor(
    *,
    required_span_labels: frozenset[str] = frozenset({"situation"}),
    required_keywords: frozenset[str] = frozenset(),
    embedding_center: tuple[float, ...] | None = None,
    max_drift: float = 0.3,
) -> SemanticAnchor:
    return SemanticAnchor(
        name="completion",
        target_property="telicity",
        canonical_prompt="Does [[situation]] reach an endpoint?",
        response_space=ResponseSpace(options=("no", "yes"), is_ordered=False),
        required_span_labels=required_span_labels,
        required_keywords=required_keywords,
        embedding_center=embedding_center,
        max_drift=max_drift,
    )


class TestStructuralDriftValidator:
    """Tests for :class:`StructuralDriftValidator`."""

    def test_passes_well_formed(self) -> None:
        validator = StructuralDriftValidator()
        score = validator.validate(
            "Does [[situation]] end with an endpoint?",
            _anchor(required_keywords=frozenset({"endpoint"})),
            ProtocolContext(),
        )
        assert score.passed is True
        assert score.findings == ()

    def test_missing_span_label(self) -> None:
        validator = StructuralDriftValidator()
        score = validator.validate(
            "Does it reach an endpoint?",
            _anchor(),
            ProtocolContext(),
        )
        assert score.passed is False
        assert any("[[situation]]" in f for f in score.findings)

    def test_missing_keyword_case_insensitive(self) -> None:
        validator = StructuralDriftValidator(keyword_case_sensitive=False)
        score = validator.validate(
            "Does [[situation]] reach a stopping point?",
            _anchor(required_keywords=frozenset({"Endpoint"})),
            ProtocolContext(),
        )
        assert score.passed is False
        assert any("endpoint" in f.lower() for f in score.findings)

    def test_missing_question_mark(self) -> None:
        validator = StructuralDriftValidator()
        score = validator.validate(
            "Does [[situation]] reach an endpoint",
            _anchor(),
            ProtocolContext(),
        )
        assert score.passed is False
        assert any("'?'" in f for f in score.findings)

    def test_too_short(self) -> None:
        validator = StructuralDriftValidator(min_length=30)
        score = validator.validate(
            "Short [[situation]]?",
            _anchor(),
            ProtocolContext(),
        )
        assert score.passed is False
        assert any("too short" in f for f in score.findings)

    def test_label_with_transform_recognized(self) -> None:
        validator = StructuralDriftValidator()
        score = validator.validate(
            "Did [[situation|gerund]] reach completion?",
            _anchor(),
            ProtocolContext(),
        )
        assert score.passed is True


class _StubAdapter:
    """Stub adapter exposing get_embedding and compute_perplexity.

    Conforms to :class:`EmbeddingAdapter` and :class:`PerplexityAdapter`.
    """

    def __init__(
        self,
        *,
        embed_map: dict[str, tuple[float, ...]] | None = None,
        default_embedding: tuple[float, ...] = (1.0, 0.0, 0.0),
        perplexity: float = 30.0,
    ) -> None:
        self._embed_map = embed_map or {}
        self._default = default_embedding
        self._perplexity = perplexity

    def get_embedding(self, text: str) -> Sequence[float]:
        return self._embed_map.get(text, self._default)

    def compute_perplexity(self, text: str) -> float:
        del text  # unused: stub returns a fixed value
        return self._perplexity


class TestEmbeddingDriftValidator:
    """Tests for :class:`EmbeddingDriftValidator`."""

    def test_stub_adapter_conforms_to_protocol(self) -> None:
        adapter = _StubAdapter()
        assert isinstance(adapter, EmbeddingAdapter)
        assert isinstance(adapter, PerplexityAdapter)

    def test_passes_under_max_drift(self) -> None:
        adapter = _StubAdapter(
            embed_map={
                "Does [[situation]] reach an endpoint?": (1.0, 0.0, 0.0),
                "Did [[situation]] finish?": (0.99, 0.05, 0.0),
            }
        )
        validator = EmbeddingDriftValidator(adapter)
        anchor = _anchor()
        score = validator.validate(
            "Did [[situation]] finish?", anchor, ProtocolContext()
        )
        assert score.passed is True
        assert score.embedding_distance is not None
        assert score.embedding_distance < 0.3

    def test_fails_over_max_drift(self) -> None:
        adapter = _StubAdapter(
            embed_map={
                "Does [[situation]] reach an endpoint?": (1.0, 0.0, 0.0),
                "Banana cake": (0.0, 1.0, 0.0),
            }
        )
        validator = EmbeddingDriftValidator(adapter, max_distance=0.1)
        anchor = _anchor()
        score = validator.validate("Banana cake", anchor, ProtocolContext())
        assert score.passed is False
        assert any("Embedding distance" in f for f in score.findings)

    def test_uses_anchor_embedding_center_when_present(self) -> None:
        # When the anchor carries a center, the adapter is not asked
        # to embed the canonical prompt.
        seen: list[str] = []

        class TrackingAdapter(_StubAdapter):
            def get_embedding(self, text: str) -> Sequence[float]:
                seen.append(text)
                return super().get_embedding(text)

        adapter = TrackingAdapter(
            embed_map={"realization": (1.0, 0.0, 0.0)},
        )
        anchor = _anchor(embedding_center=(1.0, 0.0, 0.0))
        validator = EmbeddingDriftValidator(adapter)
        score = validator.validate("realization", anchor, ProtocolContext())
        assert score.passed is True
        assert seen == ["realization"]


class TestPerplexityDriftValidator:
    """Tests for :class:`PerplexityDriftValidator`."""

    def test_passes_within_ceiling(self) -> None:
        adapter = _StubAdapter(perplexity=30.0)
        validator = PerplexityDriftValidator(adapter, max_perplexity=50.0)
        score = validator.validate(
            "Did [[situation]] finish?", _anchor(), ProtocolContext()
        )
        assert score.passed is True
        assert score.perplexity == pytest.approx(30.0)

    def test_fails_when_exceeds_ceiling(self) -> None:
        adapter = _StubAdapter(perplexity=200.0)
        validator = PerplexityDriftValidator(adapter, max_perplexity=50.0)
        score = validator.validate("Garbled output", _anchor(), ProtocolContext())
        assert score.passed is False
        assert any("Perplexity" in f for f in score.findings)

    def test_invalid_max_perplexity(self) -> None:
        adapter = _StubAdapter()
        with pytest.raises(ValueError, match="positive"):
            PerplexityDriftValidator(adapter, max_perplexity=0.0)


class TestDriftGuard:
    """Tests for :class:`DriftGuard`."""

    def test_empty_guard_always_passes(self) -> None:
        guard = DriftGuard()
        score = guard.check(
            "any string?",
            _anchor(required_span_labels=frozenset()),
            ProtocolContext(),
        )
        assert score.passed is True

    def test_aggregates_findings(self) -> None:
        guard = DriftGuard()
        guard.add(StructuralDriftValidator())
        adapter = _StubAdapter(
            embed_map={
                "Does [[situation]] reach an endpoint?": (1.0, 0.0, 0.0),
                "Bad [[situation]]?": (0.0, 1.0, 0.0),
            }
        )
        guard.add(EmbeddingDriftValidator(adapter, max_distance=0.1))
        score = guard.check(
            "Bad [[situation]]?",
            _anchor(),
            ProtocolContext(),
        )
        # Embedding fails, structural passes
        assert score.passed is False
        assert score.structural_ok is True
        assert score.embedding_distance is not None

    def test_drift_validator_protocol(self) -> None:
        validator: DriftValidator = StructuralDriftValidator()
        score = validator.validate(
            "Does [[situation]] reach an endpoint?",
            _anchor(),
            ProtocolContext(),
        )
        assert isinstance(score, DriftScore)

    def test_len(self) -> None:
        guard = DriftGuard()
        guard.add(StructuralDriftValidator())
        guard.add(StructuralDriftValidator())
        assert len(guard) == 2
