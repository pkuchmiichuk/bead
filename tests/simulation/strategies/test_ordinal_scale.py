"""Tests for ordinal scale simulation strategy."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from bead.items.item import Item, ModelOutput
from bead.items.item_template import (
    ItemTemplate,
    PresentationSpec,
    ScaleBounds,
    ScalePointLabel,
    TaskSpec,
)
from bead.simulation.strategies.ordinal_scale import OrdinalScaleStrategy


def _create_model_output(score: float) -> ModelOutput:
    """Create ModelOutput with required fields."""
    return ModelOutput(
        model_name="test_model",
        model_version="1.0",
        operation="lm_score",
        inputs={},
        output=score,
        cache_key=f"key_{score}",
    )


def test_strategy_instantiation() -> None:
    """Test that strategy can be instantiated."""
    strategy = OrdinalScaleStrategy()
    assert strategy.supported_task_type == "ordinal_scale"


def test_validate_item_correct_task_type() -> None:
    """Test validation passes with correct task type."""
    strategy = OrdinalScaleStrategy()

    template = ItemTemplate(
        name="test_ordinal",
        judgment_type="acceptability",
        task_type="ordinal_scale",
        task_spec=TaskSpec(
            prompt="Rate naturalness", scale_bounds=ScaleBounds(min=1, max=7)
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        rendered_elements={"sentence": "The cat sat on the mat."},
    )

    # Should not raise
    strategy.validate_item(item, template)


def test_validate_item_wrong_task_type() -> None:
    """Test validation fails with wrong task type."""
    strategy = OrdinalScaleStrategy()

    template = ItemTemplate(
        name="test_binary",
        judgment_type="acceptability",
        task_type="binary",
        task_spec=TaskSpec(prompt="Is this grammatical?"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(item_template_id=uuid4())

    with pytest.raises(ValueError, match="Expected task_type 'ordinal_scale'"):
        strategy.validate_item(item, template)


def test_validate_item_no_scale_bounds() -> None:
    """Test validation fails when scale_bounds not defined."""
    strategy = OrdinalScaleStrategy()

    template = ItemTemplate(
        name="test_no_bounds",
        judgment_type="acceptability",
        task_type="ordinal_scale",
        task_spec=TaskSpec(prompt="Rate this"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(item_template_id=uuid4())

    with pytest.raises(ValueError, match="task_spec.scale_bounds must be defined"):
        strategy.validate_item(item, template)


def test_validate_item_invalid_scale_bounds() -> None:
    """Test validation fails when min >= max."""
    strategy = OrdinalScaleStrategy()

    template = ItemTemplate(
        name="test_invalid_bounds",
        judgment_type="acceptability",
        task_type="ordinal_scale",
        task_spec=TaskSpec(prompt="Rate this", scale_bounds=ScaleBounds(min=7, max=1)),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(item_template_id=uuid4())

    with pytest.raises(ValueError, match="scale_bounds min .* must be less than max"):
        strategy.validate_item(item, template)


def test_simulate_response_with_high_score() -> None:
    """Test response with high score tends toward upper bound."""
    strategy = OrdinalScaleStrategy()

    template = ItemTemplate(
        name="test_ordinal",
        judgment_type="acceptability",
        task_type="ordinal_scale",
        task_spec=TaskSpec(
            prompt="Rate naturalness", scale_bounds=ScaleBounds(min=1, max=7)
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # High positive score should map to upper end of scale
    item = Item(
        item_template_id=uuid4(),
        model_outputs=[_create_model_output(5.0)],
    )

    rng = np.random.RandomState(42)

    # Generate multiple responses
    responses = [
        strategy.simulate_response(item, template, "lm_score", rng) for _ in range(100)
    ]

    # Should be mostly 7
    mean_rating = np.mean(responses)
    assert mean_rating > 6.5


def test_simulate_response_with_low_score() -> None:
    """Test response with low score tends toward lower bound."""
    strategy = OrdinalScaleStrategy()

    template = ItemTemplate(
        name="test_ordinal",
        judgment_type="acceptability",
        task_type="ordinal_scale",
        task_spec=TaskSpec(
            prompt="Rate naturalness", scale_bounds=ScaleBounds(min=1, max=7)
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Low negative score should map to lower end of scale
    item = Item(
        item_template_id=uuid4(),
        model_outputs=[_create_model_output(-5.0)],
    )

    rng = np.random.RandomState(42)

    # Generate multiple responses
    responses = [
        strategy.simulate_response(item, template, "lm_score", rng) for _ in range(100)
    ]

    # Should be mostly 1
    mean_rating = np.mean(responses)
    assert mean_rating < 1.5


def test_simulate_response_with_zero_score() -> None:
    """Test response with zero score gives middle of scale."""
    strategy = OrdinalScaleStrategy()

    template = ItemTemplate(
        name="test_ordinal",
        judgment_type="acceptability",
        task_type="ordinal_scale",
        task_spec=TaskSpec(
            prompt="Rate naturalness", scale_bounds=ScaleBounds(min=1, max=7)
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Zero score should map to middle of scale (around 4)
    item = Item(
        item_template_id=uuid4(),
        model_outputs=[_create_model_output(0.0)],
    )

    rng = np.random.RandomState(42)

    # Generate multiple responses
    responses = [
        strategy.simulate_response(item, template, "lm_score", rng) for _ in range(100)
    ]

    # Should be around 4
    mean_rating = np.mean(responses)
    assert 3.5 < mean_rating < 4.5


def test_simulate_response_without_model_outputs() -> None:
    """Test fallback to random when model outputs missing."""
    strategy = OrdinalScaleStrategy()

    template = ItemTemplate(
        name="test_ordinal",
        judgment_type="acceptability",
        task_type="ordinal_scale",
        task_spec=TaskSpec(
            prompt="Rate naturalness", scale_bounds=ScaleBounds(min=1, max=7)
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Item without model outputs
    item = Item(item_template_id=uuid4())

    rng = np.random.RandomState(42)

    # Generate multiple responses
    responses = [
        strategy.simulate_response(item, template, "lm_score", rng) for _ in range(1000)
    ]

    # Should cover full range uniformly
    unique_responses = set(responses)
    assert unique_responses == {1, 2, 3, 4, 5, 6, 7}

    # Mean should be around middle
    mean_rating = np.mean(responses)
    assert 3.5 < mean_rating < 4.5


def test_simulate_response_respects_bounds() -> None:
    """Test that responses always stay within scale bounds."""
    strategy = OrdinalScaleStrategy()

    template = ItemTemplate(
        name="test_ordinal",
        judgment_type="acceptability",
        task_type="ordinal_scale",
        task_spec=TaskSpec(
            prompt="Rate naturalness", scale_bounds=ScaleBounds(min=1, max=5)
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Test with extreme scores
    for score in [-100.0, -10.0, 0.0, 10.0, 100.0]:
        item = Item(
            item_template_id=uuid4(),
            model_outputs=[_create_model_output(score)],
        )

        rng = np.random.RandomState(42)
        responses = [
            strategy.simulate_response(item, template, "lm_score", rng)
            for _ in range(100)
        ]

        # All responses should be within bounds
        assert all(1 <= r <= 5 for r in responses)


def test_simulate_response_with_different_scale() -> None:
    """Test with different scale bounds (1-10)."""
    strategy = OrdinalScaleStrategy()

    template = ItemTemplate(
        name="test_ordinal",
        judgment_type="acceptability",
        task_type="ordinal_scale",
        task_spec=TaskSpec(
            prompt="Rate quality", scale_bounds=ScaleBounds(min=1, max=10)
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Zero score should map to middle (around 5.5)
    item = Item(
        item_template_id=uuid4(),
        model_outputs=[_create_model_output(0.0)],
    )

    rng = np.random.RandomState(42)
    responses = [
        strategy.simulate_response(item, template, "lm_score", rng) for _ in range(100)
    ]

    # Should be around 5-6
    mean_rating = np.mean(responses)
    assert 5.0 < mean_rating < 6.5


def test_simulate_response_with_item_metadata() -> None:
    """Test extraction from item_metadata."""
    strategy = OrdinalScaleStrategy()

    template = ItemTemplate(
        name="test_ordinal",
        judgment_type="acceptability",
        task_type="ordinal_scale",
        task_spec=TaskSpec(
            prompt="Rate naturalness", scale_bounds=ScaleBounds(min=1, max=7)
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Score in item_metadata
    item = Item(
        item_template_id=uuid4(),
        item_metadata={"lm_score1": 3.0},
    )

    rng = np.random.RandomState(42)
    responses = [
        strategy.simulate_response(item, template, "lm_score", rng) for _ in range(100)
    ]

    # Should be toward upper end with positive score
    mean_rating = np.mean(responses)
    assert mean_rating > 5.0


def test_simulate_response_deterministic_with_seed() -> None:
    """Test that responses are deterministic with fixed seed."""
    strategy = OrdinalScaleStrategy()

    template = ItemTemplate(
        name="test_ordinal",
        judgment_type="acceptability",
        task_type="ordinal_scale",
        task_spec=TaskSpec(
            prompt="Rate naturalness", scale_bounds=ScaleBounds(min=1, max=7)
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        model_outputs=[_create_model_output(0.5)],
    )

    # Generate responses with same seed
    rng1 = np.random.RandomState(123)
    response1 = strategy.simulate_response(item, template, "lm_score", rng1)

    rng2 = np.random.RandomState(123)
    response2 = strategy.simulate_response(item, template, "lm_score", rng2)

    # Should be identical
    assert response1 == response2


def test_simulate_response_returns_integer() -> None:
    """Test that response is always an integer."""
    strategy = OrdinalScaleStrategy()

    template = ItemTemplate(
        name="test_ordinal",
        judgment_type="acceptability",
        task_type="ordinal_scale",
        task_spec=TaskSpec(
            prompt="Rate naturalness", scale_bounds=ScaleBounds(min=1, max=7)
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        model_outputs=[_create_model_output(1.7)],
    )

    rng = np.random.RandomState(42)
    response = strategy.simulate_response(item, template, "lm_score", rng)

    assert isinstance(response, int | np.integer)


def test_simulate_response_with_scale_labels() -> None:
    """Test that scale_labels don't affect numeric response."""
    strategy = OrdinalScaleStrategy()

    template = ItemTemplate(
        name="test_ordinal",
        judgment_type="acceptability",
        task_type="ordinal_scale",
        task_spec=TaskSpec(
            prompt="Rate naturalness",
            scale_bounds=ScaleBounds(min=1, max=7),
            scale_labels=(
                ScalePointLabel(point=1, label="Very unnatural"),
                ScalePointLabel(point=7, label="Very natural"),
            ),
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        model_outputs=[_create_model_output(0.0)],
    )

    rng = np.random.RandomState(42)
    response = strategy.simulate_response(item, template, "lm_score", rng)

    # Should still be integer in bounds
    assert isinstance(response, int | np.integer)
    assert 1 <= response <= 7


def test_simulate_response_with_negative_bounds() -> None:
    """Test with scale that includes negative numbers."""
    strategy = OrdinalScaleStrategy()

    template = ItemTemplate(
        name="test_ordinal",
        judgment_type="similarity",
        task_type="ordinal_scale",
        task_spec=TaskSpec(
            prompt="Rate difference", scale_bounds=ScaleBounds(min=-3, max=3)
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Zero score should map to middle (0)
    item = Item(
        item_template_id=uuid4(),
        model_outputs=[_create_model_output(0.0)],
    )

    rng = np.random.RandomState(42)
    responses = [
        strategy.simulate_response(item, template, "lm_score", rng) for _ in range(100)
    ]

    # Should be around 0
    mean_rating = np.mean(responses)
    assert -0.5 < mean_rating < 0.5

    # All responses should be in bounds
    assert all(-3 <= r <= 3 for r in responses)


def test_extract_model_outputs_wrong_count() -> None:
    """Test that wrong count returns None."""
    strategy = OrdinalScaleStrategy()

    # Item has 2 scores but we need 1
    item = Item(
        item_template_id=uuid4(),
        model_outputs=[
            _create_model_output(2.5),
            _create_model_output(3.5),
        ],
    )

    scores = strategy.extract_model_outputs(item, "lm_score", required_count=1)

    # Should return None because count mismatch
    assert scores is None
