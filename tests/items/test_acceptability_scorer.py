"""Tests for AcceptabilityScorer using a stub forced-choice model."""

from __future__ import annotations

from uuid import uuid4

import numpy as np

from bead.items.item import Item
from bead.items.scoring import AcceptabilityScorer


class _StubMixedEffects:
    mode = "fixed"


class _StubConfig:
    mixed_effects = _StubMixedEffects()


class _StubForcedChoiceModel:
    """Minimal stand-in for ForcedChoiceModel exposing predict_proba."""

    def __init__(self, proba: np.ndarray) -> None:
        self._proba = proba
        self.config = _StubConfig()

    def predict_proba(
        self, items: list[Item], participant_ids: list[str] | None = None
    ) -> np.ndarray:
        # fixed-effects model is asked with participant_ids=None
        assert participant_ids is None
        return self._proba[: len(items)]


def _item(text: str = "a vs b") -> Item:
    return Item(
        item_template_id=uuid4(),
        rendered_elements={"option_a": "A", "option_b": "B"},
        options=("first", "second"),
        item_metadata={"text": text},
    )


def test_score_returns_preference_margin() -> None:
    # p_first=0.7 -> margin = 2*0.7-1 = 0.4
    model = _StubForcedChoiceModel(np.array([[0.7, 0.3]]))
    scorer = AcceptabilityScorer(model)  # type: ignore[arg-type]
    assert abs(scorer.score(_item()) - 0.4) < 1e-9


def test_margin_is_column_agnostic() -> None:
    # whichever option dominates, margin reflects |p_a - p_b|
    model = _StubForcedChoiceModel(np.array([[0.2, 0.8]]))
    scorer = AcceptabilityScorer(model)  # type: ignore[arg-type]
    assert abs(scorer.score(_item()) - 0.6) < 1e-9


def test_tie_gives_zero_margin() -> None:
    model = _StubForcedChoiceModel(np.array([[0.5, 0.5]]))
    scorer = AcceptabilityScorer(model)  # type: ignore[arg-type]
    assert abs(scorer.score(_item())) < 1e-9


def test_score_batch() -> None:
    proba = np.array([[0.9, 0.1], [0.5, 0.5], [0.3, 0.7]])
    scorer = AcceptabilityScorer(_StubForcedChoiceModel(proba))  # type: ignore[arg-type]
    margins = scorer.score_batch([_item(), _item(), _item()])
    assert len(margins) == 3
    assert abs(margins[0] - 0.8) < 1e-9
    assert abs(margins[1]) < 1e-9
    assert abs(margins[2] - 0.4) < 1e-9


def test_score_batch_empty() -> None:
    scorer = AcceptabilityScorer(_StubForcedChoiceModel(np.zeros((0, 2))))  # type: ignore[arg-type]
    assert scorer.score_batch([]) == []


def test_score_with_metadata() -> None:
    proba = np.array([[0.7, 0.3]])
    item = _item()
    scorer = AcceptabilityScorer(_StubForcedChoiceModel(proba))  # type: ignore[arg-type]
    out = scorer.score_with_metadata([item])
    entry = out[item.id]
    assert abs(entry["score"] - 0.4) < 1e-9  # type: ignore[operator]
    assert abs(entry["acceptability_margin"] - 0.4) < 1e-9  # type: ignore[operator]
    assert abs(entry["p_first"] - 0.7) < 1e-9  # type: ignore[operator]
    assert entry["predicted_option"] == 0
