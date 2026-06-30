"""Tests for forced choice simulation strategy."""

from __future__ import annotations

from uuid import uuid4

import numpy as np
import pytest

from bead.items.item import Item, ModelOutput
from bead.items.item_template import (
    ItemTemplate,
    PresentationSpec,
    ScaleBounds,
    TaskSpec,
)
from bead.simulation.strategies.forced_choice import ForcedChoiceStrategy


def test_strategy_instantiation() -> None:
    """Test that strategy can be instantiated."""
    strategy = ForcedChoiceStrategy()
    assert strategy.supported_task_type == "forced_choice"


def test_validate_item_correct_task_type() -> None:
    """Test validation passes with correct task type."""
    strategy = ForcedChoiceStrategy()

    template = ItemTemplate(
        name="test_2afc",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(prompt="Which is better?", options=["a", "b"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        rendered_elements={"a": "Option A", "b": "Option B"},
    )

    # Should not raise
    strategy.validate_item(item, template)


def test_validate_item_wrong_task_type() -> None:
    """Test validation fails with wrong task type."""
    strategy = ForcedChoiceStrategy()

    template = ItemTemplate(
        name="test_ordinal",
        judgment_type="acceptability",
        task_type="ordinal_scale",
        task_spec=TaskSpec(prompt="Rate this", scale_bounds=ScaleBounds(min=1, max=7)),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(item_template_id=uuid4())

    with pytest.raises(ValueError, match="Expected task_type 'forced_choice'"):
        strategy.validate_item(item, template)


def test_validate_item_no_options() -> None:
    """Test validation fails when options not defined."""
    strategy = ForcedChoiceStrategy()

    template = ItemTemplate(
        name="test_no_options",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(prompt="Which is better?"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(item_template_id=uuid4())

    with pytest.raises(ValueError, match="task_spec.options must be defined"):
        strategy.validate_item(item, template)


def test_validate_item_too_few_options() -> None:
    """Test validation fails with fewer than 2 options."""
    strategy = ForcedChoiceStrategy()

    template = ItemTemplate(
        name="test_one_option",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(prompt="Which is better?", options=["a"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(item_template_id=uuid4())

    with pytest.raises(ValueError, match="requires at least 2 options"):
        strategy.validate_item(item, template)


def test_simulate_response_2afc_with_model_outputs() -> None:
    """Test 2AFC response with model outputs."""
    strategy = ForcedChoiceStrategy()
    rng = np.random.RandomState(42)

    template = ItemTemplate(
        name="test_2afc",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(prompt="Which is better?", options=["option_a", "option_b"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Create item with model outputs favoring option_a
    item = Item(
        item_template_id=uuid4(),
        rendered_elements={"option_a": "Text A", "option_b": "Text B"},
        model_outputs=[
            ModelOutput(
                model_name="test_model",
                model_version="1.0",
                operation="lm_score",
                inputs={"text": "Text A"},
                output=-2.0,  # Higher score (less negative)
                cache_key="key1",
            ),
            ModelOutput(
                model_name="test_model",
                model_version="1.0",
                operation="lm_score",
                inputs={"text": "Text B"},
                output=-5.0,  # Lower score (more negative)
                cache_key="key2",
            ),
        ],
    )

    # Run simulation multiple times to check probability
    choices = []
    for _ in range(100):
        choice = strategy.simulate_response(item, template, "lm_score", rng)
        choices.append(choice)

    # Should heavily favor option_a due to better score
    option_a_count = choices.count("option_a")
    assert option_a_count > 70  # Should be >90% but allow some randomness


def test_simulate_response_3afc_with_model_outputs() -> None:
    """Test 3AFC response with model outputs."""
    strategy = ForcedChoiceStrategy()
    rng = np.random.RandomState(42)

    template = ItemTemplate(
        name="test_3afc",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(
            prompt="Which is best?", options=["option_a", "option_b", "option_c"]
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Create item with model outputs favoring option_b
    item = Item(
        item_template_id=uuid4(),
        rendered_elements={
            "option_a": "Text A",
            "option_b": "Text B",
            "option_c": "Text C",
        },
        model_outputs=[
            ModelOutput(
                model_name="test_model",
                model_version="1.0",
                operation="lm_score",
                inputs={"text": "Text A"},
                output=-5.0,
                cache_key="key1",
            ),
            ModelOutput(
                model_name="test_model",
                model_version="1.0",
                operation="lm_score",
                inputs={"text": "Text B"},
                output=-1.0,  # Best score
                cache_key="key2",
            ),
            ModelOutput(
                model_name="test_model",
                model_version="1.0",
                operation="lm_score",
                inputs={"text": "Text C"},
                output=-4.0,
                cache_key="key3",
            ),
        ],
    )

    # Run simulation multiple times
    choices = []
    for _ in range(100):
        choice = strategy.simulate_response(item, template, "lm_score", rng)
        choices.append(choice)

    # Should heavily favor option_b
    option_b_count = choices.count("option_b")
    assert option_b_count > 70


def test_simulate_response_fallback_to_random() -> None:
    """Test fallback to random when model outputs missing."""
    strategy = ForcedChoiceStrategy()
    rng = np.random.RandomState(42)

    template = ItemTemplate(
        name="test_2afc",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(prompt="Which is better?", options=["option_a", "option_b"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Create item without model outputs
    item = Item(
        item_template_id=uuid4(),
        rendered_elements={"option_a": "Text A", "option_b": "Text B"},
    )

    # Run simulation multiple times
    choices = []
    for _ in range(100):
        choice = strategy.simulate_response(item, template, "lm_score", rng)
        choices.append(choice)

    # Should be roughly uniform
    option_a_count = choices.count("option_a")
    assert 30 < option_a_count < 70  # Should be ~50% with some variance


def test_simulate_response_wrong_model_output_key() -> None:
    """Test fallback when wrong model output key specified."""
    strategy = ForcedChoiceStrategy()
    rng = np.random.RandomState(42)

    template = ItemTemplate(
        name="test_2afc",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(prompt="Which is better?", options=["option_a", "option_b"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Create item with lm_score but request embedding
    item = Item(
        item_template_id=uuid4(),
        rendered_elements={"option_a": "Text A", "option_b": "Text B"},
        model_outputs=[
            ModelOutput(
                model_name="test_model",
                model_version="1.0",
                operation="lm_score",
                inputs={"text": "Text A"},
                output=-2.0,
                cache_key="key1",
            ),
            ModelOutput(
                model_name="test_model",
                model_version="1.0",
                operation="lm_score",
                inputs={"text": "Text B"},
                output=-5.0,
                cache_key="key2",
            ),
        ],
    )

    # Should fall back to random
    choices = []
    for _ in range(100):
        choice = strategy.simulate_response(item, template, "embedding", rng)
        choices.append(choice)

    option_a_count = choices.count("option_a")
    assert 30 < option_a_count < 70


def test_simulate_response_from_item_metadata() -> None:
    """Test extraction from item_metadata."""
    strategy = ForcedChoiceStrategy()
    rng = np.random.RandomState(42)

    template = ItemTemplate(
        name="test_2afc",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(prompt="Which is better?", options=["option_a", "option_b"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Create item with scores in metadata
    item = Item(
        item_template_id=uuid4(),
        rendered_elements={"option_a": "Text A", "option_b": "Text B"},
        item_metadata={"lm_score1": -2.0, "lm_score2": -5.0},
    )

    # Run simulation multiple times
    choices = []
    for _ in range(100):
        choice = strategy.simulate_response(item, template, "lm_score", rng)
        choices.append(choice)

    # Should favor option_a
    option_a_count = choices.count("option_a")
    assert option_a_count > 70


def test_simulate_response_with_equal_scores() -> None:
    """Test response with equal scores (uniform distribution)."""
    strategy = ForcedChoiceStrategy()
    rng = np.random.RandomState(42)

    template = ItemTemplate(
        name="test_2afc",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(prompt="Which is better?", options=["option_a", "option_b"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Create item with equal scores
    item = Item(
        item_template_id=uuid4(),
        rendered_elements={"option_a": "Text A", "option_b": "Text B"},
        model_outputs=[
            ModelOutput(
                model_name="test_model",
                model_version="1.0",
                operation="lm_score",
                inputs={"text": "Text A"},
                output=-3.0,
                cache_key="key1",
            ),
            ModelOutput(
                model_name="test_model",
                model_version="1.0",
                operation="lm_score",
                inputs={"text": "Text B"},
                output=-3.0,
                cache_key="key2",
            ),
        ],
    )

    # Run simulation multiple times
    choices = []
    for _ in range(100):
        choice = strategy.simulate_response(item, template, "lm_score", rng)
        choices.append(choice)

    # Should be roughly uniform
    option_a_count = choices.count("option_a")
    assert 30 < option_a_count < 70


def test_simulate_response_4afc() -> None:
    """Test 4AFC with model outputs."""
    strategy = ForcedChoiceStrategy()
    rng = np.random.RandomState(42)

    template = ItemTemplate(
        name="test_4afc",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(
            prompt="Which is best?",
            options=["option_a", "option_b", "option_c", "option_d"],
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Create item with model outputs
    item = Item(
        item_template_id=uuid4(),
        rendered_elements={
            "option_a": "A",
            "option_b": "B",
            "option_c": "C",
            "option_d": "D",
        },
        model_outputs=[
            ModelOutput(
                model_name="test_model",
                model_version="1.0",
                operation="lm_score",
                inputs={"text": "A"},
                output=-5.0,
                cache_key="key1",
            ),
            ModelOutput(
                model_name="test_model",
                model_version="1.0",
                operation="lm_score",
                inputs={"text": "B"},
                output=-3.0,
                cache_key="key2",
            ),
            ModelOutput(
                model_name="test_model",
                model_version="1.0",
                operation="lm_score",
                inputs={"text": "C"},
                output=-1.0,  # Best
                cache_key="key3",
            ),
            ModelOutput(
                model_name="test_model",
                model_version="1.0",
                operation="lm_score",
                inputs={"text": "D"},
                output=-4.0,
                cache_key="key4",
            ),
        ],
    )

    # Run simulation
    choices = []
    for _ in range(100):
        choice = strategy.simulate_response(item, template, "lm_score", rng)
        choices.append(choice)

    # Should favor option_c
    option_c_count = choices.count("option_c")
    assert option_c_count > 60


def test_simulate_response_no_options_raises() -> None:
    """Test that simulate_response raises if options not defined."""
    strategy = ForcedChoiceStrategy()
    rng = np.random.RandomState(42)

    template = ItemTemplate(
        name="test_no_options",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(prompt="Which is better?"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(item_template_id=uuid4())

    with pytest.raises(ValueError, match="task_spec.options must be defined"):
        strategy.simulate_response(item, template, "lm_score", rng)


def test_extract_model_outputs_helper() -> None:
    """Test the extract_model_outputs helper method."""
    strategy = ForcedChoiceStrategy()

    # Test with model outputs
    item = Item(
        item_template_id=uuid4(),
        model_outputs=[
            ModelOutput(
                model_name="test",
                model_version="1.0",
                operation="lm_score",
                inputs={},
                output=-2.0,
                cache_key="k1",
            ),
            ModelOutput(
                model_name="test",
                model_version="1.0",
                operation="lm_score",
                inputs={},
                output=-3.0,
                cache_key="k2",
            ),
        ],
    )

    scores = strategy.extract_model_outputs(item, "lm_score", required_count=2)
    assert scores == [-2.0, -3.0]


def test_extract_model_outputs_wrong_count() -> None:
    """Test extract_model_outputs returns None with wrong count."""
    strategy = ForcedChoiceStrategy()

    item = Item(
        item_template_id=uuid4(),
        model_outputs=[
            ModelOutput(
                model_name="test",
                model_version="1.0",
                operation="lm_score",
                inputs={},
                output=-2.0,
                cache_key="k1",
            ),
        ],
    )

    scores = strategy.extract_model_outputs(item, "lm_score", required_count=2)
    assert scores is None
