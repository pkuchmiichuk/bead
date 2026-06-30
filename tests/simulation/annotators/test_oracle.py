"""Tests for oracle annotator."""

from __future__ import annotations

from uuid import uuid4

import pytest

from bead.config.simulation import SimulatedAnnotatorConfig
from bead.items.item import Item
from bead.items.item_template import (
    ItemTemplate,
    PresentationSpec,
    ScaleBounds,
    TaskSpec,
)
from bead.simulation.annotators.oracle import OracleAnnotator


def test_annotator_instantiation() -> None:
    """Test that annotator can be instantiated."""
    config = SimulatedAnnotatorConfig(strategy="oracle", random_state=42)
    annotator = OracleAnnotator(config)
    assert annotator.config.strategy == "oracle"
    assert annotator.random_state == 42


def test_annotate_forced_choice_with_ground_truth() -> None:
    """Test oracle forced choice with ground truth."""
    config = SimulatedAnnotatorConfig(strategy="oracle", random_state=42)
    annotator = OracleAnnotator(config)

    template = ItemTemplate(
        name="test_fc",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(prompt="Which?", options=["option_a", "option_b"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        item_metadata={"ground_truth": "option_a"},
    )

    annotation = annotator.annotate(item, template)
    assert annotation == "option_a"


def test_annotate_binary_with_ground_truth() -> None:
    """Test oracle binary with ground truth."""
    config = SimulatedAnnotatorConfig(strategy="oracle", random_state=42)
    annotator = OracleAnnotator(config)

    template = ItemTemplate(
        name="test_binary",
        judgment_type="acceptability",
        task_type="binary",
        task_spec=TaskSpec(prompt="Is this good?"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        item_metadata={"ground_truth": True},
    )

    annotation = annotator.annotate(item, template)
    assert annotation is True


def test_annotate_ordinal_with_ground_truth() -> None:
    """Test oracle ordinal scale with ground truth."""
    config = SimulatedAnnotatorConfig(strategy="oracle", random_state=42)
    annotator = OracleAnnotator(config)

    template = ItemTemplate(
        name="test_ordinal",
        judgment_type="plausibility",
        task_type="ordinal_scale",
        task_spec=TaskSpec(prompt="Rate 1-7:", scale_bounds=ScaleBounds(min=1, max=7)),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        item_metadata={"ground_truth": 5},
    )

    annotation = annotator.annotate(item, template)
    assert annotation == 5


def test_annotate_categorical_with_ground_truth() -> None:
    """Test oracle categorical with ground truth."""
    config = SimulatedAnnotatorConfig(strategy="oracle", random_state=42)
    annotator = OracleAnnotator(config)

    template = ItemTemplate(
        name="test_categorical",
        judgment_type="inference",
        task_type="categorical",
        task_spec=TaskSpec(prompt="Classify:", options=["cat_a", "cat_b", "cat_c"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        item_metadata={"ground_truth": "cat_b"},
    )

    annotation = annotator.annotate(item, template)
    assert annotation == "cat_b"


def test_annotate_magnitude_with_ground_truth() -> None:
    """Test oracle magnitude with ground truth."""
    config = SimulatedAnnotatorConfig(strategy="oracle", random_state=42)
    annotator = OracleAnnotator(config)

    template = ItemTemplate(
        name="test_magnitude",
        judgment_type="plausibility",
        task_type="magnitude",
        task_spec=TaskSpec(prompt="Estimate:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        item_metadata={"ground_truth": 42.5},
    )

    annotation = annotator.annotate(item, template)
    assert annotation == 42.5


def test_annotate_multi_select_with_ground_truth() -> None:
    """Test oracle multi-select with ground truth."""
    config = SimulatedAnnotatorConfig(strategy="oracle", random_state=42)
    annotator = OracleAnnotator(config)

    template = ItemTemplate(
        name="test_multi",
        judgment_type="preference",
        task_type="multi_select",
        task_spec=TaskSpec(prompt="Select:", options=["opt_a", "opt_b", "opt_c"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        item_metadata={"ground_truth": ["opt_a", "opt_c"]},
    )

    annotation = annotator.annotate(item, template)
    assert tuple(annotation) == ("opt_a", "opt_c")


def test_annotate_free_text_with_ground_truth() -> None:
    """Test oracle free text with ground truth."""
    config = SimulatedAnnotatorConfig(strategy="oracle", random_state=42)
    annotator = OracleAnnotator(config)

    template = ItemTemplate(
        name="test_free",
        judgment_type="comprehension",
        task_type="free_text",
        task_spec=TaskSpec(prompt="Describe:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        item_metadata={"ground_truth": "Perfect answer"},
    )

    annotation = annotator.annotate(item, template)
    assert annotation == "Perfect answer"


def test_annotate_forced_choice_without_ground_truth() -> None:
    """Test fallback to random when no ground truth."""
    config = SimulatedAnnotatorConfig(strategy="oracle", random_state=42)
    annotator = OracleAnnotator(config)

    template = ItemTemplate(
        name="test_fc",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(prompt="Which?", options=["option_a", "option_b"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(item_template_id=uuid4())

    # Should fallback to random
    annotations = [annotator.annotate(item, template) for _ in range(100)]

    # Should have variety (random fallback)
    assert "option_a" in annotations
    assert "option_b" in annotations


def test_annotate_invalid_ground_truth_forced_choice() -> None:
    """Test error for invalid ground truth in forced choice."""
    config = SimulatedAnnotatorConfig(strategy="oracle", random_state=42)
    annotator = OracleAnnotator(config)

    template = ItemTemplate(
        name="test_fc",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(prompt="Which?", options=["option_a", "option_b"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        item_metadata={"ground_truth": "invalid_option"},
    )

    with pytest.raises(ValueError, match="not in options"):
        annotator.annotate(item, template)


def test_annotate_invalid_ground_truth_binary() -> None:
    """Test error for invalid ground truth in binary."""
    config = SimulatedAnnotatorConfig(strategy="oracle", random_state=42)
    annotator = OracleAnnotator(config)

    template = ItemTemplate(
        name="test_binary",
        judgment_type="acceptability",
        task_type="binary",
        task_spec=TaskSpec(prompt="Is this good?"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        item_metadata={"ground_truth": "not_a_bool"},
    )

    with pytest.raises(ValueError, match="binary ground truth must be bool"):
        annotator.annotate(item, template)


def test_annotate_invalid_ground_truth_ordinal() -> None:
    """Test error for invalid ground truth in ordinal."""
    config = SimulatedAnnotatorConfig(strategy="oracle", random_state=42)
    annotator = OracleAnnotator(config)

    template = ItemTemplate(
        name="test_ordinal",
        judgment_type="plausibility",
        task_type="ordinal_scale",
        task_spec=TaskSpec(prompt="Rate 1-7:", scale_bounds=ScaleBounds(min=1, max=7)),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Out of range
    item = Item(
        item_template_id=uuid4(),
        item_metadata={"ground_truth": 10},
    )

    with pytest.raises(ValueError, match="not in range"):
        annotator.annotate(item, template)


def test_annotate_invalid_ground_truth_multi_select() -> None:
    """Test error for invalid ground truth in multi_select."""
    config = SimulatedAnnotatorConfig(strategy="oracle", random_state=42)
    annotator = OracleAnnotator(config)

    template = ItemTemplate(
        name="test_multi",
        judgment_type="preference",
        task_type="multi_select",
        task_spec=TaskSpec(prompt="Select:", options=["opt_a", "opt_b", "opt_c"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        item_metadata={"ground_truth": ["opt_a", "invalid"]},
    )

    with pytest.raises(ValueError, match="not in options"):
        annotator.annotate(item, template)


def test_annotate_with_none_ground_truth() -> None:
    """Test that None ground truth falls back to random."""
    config = SimulatedAnnotatorConfig(strategy="oracle", random_state=42)
    annotator = OracleAnnotator(config)

    template = ItemTemplate(
        name="test_fc",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(prompt="Which?", options=["option_a", "option_b"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        item_metadata={"ground_truth": None},
    )

    # Should fallback to random
    annotations = [annotator.annotate(item, template) for _ in range(100)]

    # Should have variety
    assert "option_a" in annotations
    assert "option_b" in annotations


def test_annotate_with_empty_metadata() -> None:
    """Test that empty metadata falls back to random."""
    config = SimulatedAnnotatorConfig(strategy="oracle", random_state=42)
    annotator = OracleAnnotator(config)

    template = ItemTemplate(
        name="test_fc",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(prompt="Which?", options=["option_a", "option_b"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        item_metadata={},
    )

    # Should fallback to random
    annotations = [annotator.annotate(item, template) for _ in range(100)]

    # Should have variety
    assert "option_a" in annotations
    assert "option_b" in annotations


def test_annotate_batch_with_ground_truth() -> None:
    """Test batch annotation with ground truth."""
    config = SimulatedAnnotatorConfig(strategy="oracle", random_state=42)
    annotator = OracleAnnotator(config)

    template = ItemTemplate(
        name="test_fc",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(prompt="Which?", options=["option_a", "option_b"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    items = [
        Item(item_template_id=uuid4(), item_metadata={"ground_truth": "option_a"}),
        Item(item_template_id=uuid4(), item_metadata={"ground_truth": "option_b"}),
        Item(item_template_id=uuid4(), item_metadata={"ground_truth": "option_a"}),
    ]

    annotations = annotator.annotate_batch(items, template)

    # Should return ground truth for each
    assert annotations[str(items[0].id)] == "option_a"
    assert annotations[str(items[1].id)] == "option_b"
    assert annotations[str(items[2].id)] == "option_a"


def test_annotate_magnitude_int_ground_truth() -> None:
    """Test that int ground truth is converted to float for magnitude."""
    config = SimulatedAnnotatorConfig(strategy="oracle", random_state=42)
    annotator = OracleAnnotator(config)

    template = ItemTemplate(
        name="test_magnitude",
        judgment_type="plausibility",
        task_type="magnitude",
        task_spec=TaskSpec(prompt="Estimate:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        item_metadata={"ground_truth": 42},  # int
    )

    annotation = annotator.annotate(item, template)
    assert annotation == 42.0
    assert isinstance(annotation, float)
