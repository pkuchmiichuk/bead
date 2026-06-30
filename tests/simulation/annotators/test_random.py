"""Tests for random annotator."""

from __future__ import annotations

from uuid import uuid4

import pytest

from bead.config.simulation import SimulatedAnnotatorConfig
from bead.items.item import Item, UnfilledSlot
from bead.items.item_template import (
    ItemTemplate,
    PresentationSpec,
    ScaleBounds,
    TaskSpec,
)
from bead.simulation.annotators.random import RandomAnnotator


def test_annotator_instantiation() -> None:
    """Test that annotator can be instantiated."""
    config = SimulatedAnnotatorConfig(strategy="random", random_state=42)
    annotator = RandomAnnotator(config)
    assert annotator.config.strategy == "random"
    assert annotator.random_state == 42


def test_annotate_forced_choice() -> None:
    """Test random forced choice annotation."""
    config = SimulatedAnnotatorConfig(strategy="random", random_state=42)
    annotator = RandomAnnotator(config)

    template = ItemTemplate(
        name="test_fc",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(prompt="Which?", options=["option_a", "option_b"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(item_template_id=uuid4())

    # Generate multiple annotations
    annotations = [annotator.annotate(item, template) for _ in range(100)]

    # Should have both options
    assert "option_a" in annotations
    assert "option_b" in annotations
    # All should be valid options
    assert all(a in ["option_a", "option_b"] for a in annotations)


def test_annotate_binary() -> None:
    """Test random binary annotation."""
    config = SimulatedAnnotatorConfig(strategy="random", random_state=42)
    annotator = RandomAnnotator(config)

    template = ItemTemplate(
        name="test_binary",
        judgment_type="acceptability",
        task_type="binary",
        task_spec=TaskSpec(prompt="Is this good?"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(item_template_id=uuid4())

    # Generate multiple annotations
    annotations = [annotator.annotate(item, template) for _ in range(100)]

    # Should have both True and False
    assert True in annotations
    assert False in annotations
    # All should be boolean
    assert all(isinstance(a, bool) for a in annotations)


def test_annotate_ordinal_scale() -> None:
    """Test random ordinal scale annotation."""
    config = SimulatedAnnotatorConfig(strategy="random", random_state=42)
    annotator = RandomAnnotator(config)

    template = ItemTemplate(
        name="test_ordinal",
        judgment_type="plausibility",
        task_type="ordinal_scale",
        task_spec=TaskSpec(prompt="Rate 1-7:", scale_bounds=ScaleBounds(min=1, max=7)),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(item_template_id=uuid4())

    # Generate multiple annotations
    annotations = [annotator.annotate(item, template) for _ in range(100)]

    # Should be in range
    assert all(1 <= a <= 7 for a in annotations)
    # Should have variety
    unique_values = set(annotations)
    assert len(unique_values) > 3


def test_annotate_ordinal_scale_custom_range() -> None:
    """Test random ordinal scale with custom range."""
    config = SimulatedAnnotatorConfig(strategy="random", random_state=42)
    annotator = RandomAnnotator(config)

    template = ItemTemplate(
        name="test_ordinal",
        judgment_type="plausibility",
        task_type="ordinal_scale",
        task_spec=TaskSpec(
            prompt="Rate 0-10:", scale_bounds=ScaleBounds(min=0, max=10)
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(item_template_id=uuid4())

    # Generate multiple annotations
    annotations = [annotator.annotate(item, template) for _ in range(100)]

    # Should be in range
    assert all(0 <= a <= 10 for a in annotations)


def test_annotate_categorical() -> None:
    """Test random categorical annotation."""
    config = SimulatedAnnotatorConfig(strategy="random", random_state=42)
    annotator = RandomAnnotator(config)

    template = ItemTemplate(
        name="test_categorical",
        judgment_type="inference",
        task_type="categorical",
        task_spec=TaskSpec(prompt="Classify:", options=["cat_a", "cat_b", "cat_c"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(item_template_id=uuid4())

    # Generate multiple annotations
    annotations = [annotator.annotate(item, template) for _ in range(100)]

    # Should have all categories
    assert "cat_a" in annotations
    assert "cat_b" in annotations
    assert "cat_c" in annotations
    # All should be valid
    assert all(a in ["cat_a", "cat_b", "cat_c"] for a in annotations)


def test_annotate_magnitude() -> None:
    """Test random magnitude annotation."""
    config = SimulatedAnnotatorConfig(strategy="random", random_state=42)
    annotator = RandomAnnotator(config)

    template = ItemTemplate(
        name="test_magnitude",
        judgment_type="plausibility",
        task_type="magnitude",
        task_spec=TaskSpec(prompt="Estimate:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(item_template_id=uuid4())

    # Generate multiple annotations
    annotations = [annotator.annotate(item, template) for _ in range(100)]

    # All should be positive floats
    assert all(isinstance(a, float) for a in annotations)
    assert all(a > 0 for a in annotations)
    # Should have variety
    assert max(annotations) > min(annotations) * 2


def test_annotate_multi_select() -> None:
    """Test random multi-select annotation."""
    config = SimulatedAnnotatorConfig(strategy="random", random_state=42)
    annotator = RandomAnnotator(config)

    template = ItemTemplate(
        name="test_multi",
        judgment_type="preference",
        task_type="multi_select",
        task_spec=TaskSpec(prompt="Select:", options=["opt_a", "opt_b", "opt_c"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(item_template_id=uuid4())

    # Generate multiple annotations
    annotations = [annotator.annotate(item, template) for _ in range(100)]

    # All should be lists
    assert all(isinstance(a, list) for a in annotations)
    # Should have variety in selections
    all_selected = []
    for ann in annotations:
        all_selected.extend(ann)

    # Each option should be selected at least once
    assert "opt_a" in all_selected
    assert "opt_b" in all_selected
    assert "opt_c" in all_selected


def test_annotate_free_text() -> None:
    """Test random free text annotation."""
    config = SimulatedAnnotatorConfig(strategy="random", random_state=42)
    annotator = RandomAnnotator(config)

    template = ItemTemplate(
        name="test_free",
        judgment_type="comprehension",
        task_type="free_text",
        task_spec=TaskSpec(prompt="Describe:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(item_template_id=uuid4())

    # Generate annotations
    annotation = annotator.annotate(item, template)

    # Should be a string
    assert isinstance(annotation, str)
    assert len(annotation) > 0


def test_annotate_cloze() -> None:
    """Test random cloze annotation."""
    config = SimulatedAnnotatorConfig(strategy="random", random_state=42)
    annotator = RandomAnnotator(config)

    template = ItemTemplate(
        name="test_cloze",
        judgment_type="comprehension",
        task_type="cloze",
        task_spec=TaskSpec(prompt="Fill in the blanks:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        unfilled_slots=[
            UnfilledSlot(slot_name="determiner", position=0, constraint_ids=[]),
            UnfilledSlot(slot_name="noun", position=2, constraint_ids=[]),
        ],
    )

    annotation = annotator.annotate(item, template)

    # Should be a dict with filled slots
    assert isinstance(annotation, dict)
    assert "determiner" in annotation
    assert "noun" in annotation
    assert isinstance(annotation["determiner"], str)
    assert isinstance(annotation["noun"], str)


def test_annotate_forced_choice_missing_options() -> None:
    """Test error when forced choice has no options."""
    config = SimulatedAnnotatorConfig(strategy="random", random_state=42)
    annotator = RandomAnnotator(config)

    template = ItemTemplate(
        name="test_fc",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(prompt="Which?"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(item_template_id=uuid4())

    with pytest.raises(ValueError, match="forced_choice requires options"):
        annotator.annotate(item, template)


def test_annotate_categorical_missing_options() -> None:
    """Test error when categorical has no options."""
    config = SimulatedAnnotatorConfig(strategy="random", random_state=42)
    annotator = RandomAnnotator(config)

    template = ItemTemplate(
        name="test_cat",
        judgment_type="inference",
        task_type="categorical",
        task_spec=TaskSpec(prompt="Classify:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(item_template_id=uuid4())

    with pytest.raises(ValueError, match="categorical requires options"):
        annotator.annotate(item, template)


def test_annotate_multi_select_missing_options() -> None:
    """Test error when multi_select has no options."""
    config = SimulatedAnnotatorConfig(strategy="random", random_state=42)
    annotator = RandomAnnotator(config)

    template = ItemTemplate(
        name="test_multi",
        judgment_type="preference",
        task_type="multi_select",
        task_spec=TaskSpec(prompt="Select:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(item_template_id=uuid4())

    with pytest.raises(ValueError, match="multi_select requires options"):
        annotator.annotate(item, template)


def test_annotate_deterministic_with_seed() -> None:
    """Test that annotations are deterministic with seed."""
    template = ItemTemplate(
        name="test_fc",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(prompt="Which?", options=["option_a", "option_b"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(item_template_id=uuid4())

    # Two annotators with same seed
    config1 = SimulatedAnnotatorConfig(strategy="random", random_state=123)
    annotator1 = RandomAnnotator(config1)
    annotations1 = [annotator1.annotate(item, template) for _ in range(10)]

    config2 = SimulatedAnnotatorConfig(strategy="random", random_state=123)
    annotator2 = RandomAnnotator(config2)
    annotations2 = [annotator2.annotate(item, template) for _ in range(10)]

    # Should be identical
    assert annotations1 == annotations2


def test_annotate_different_with_different_seed() -> None:
    """Test that different seeds give different results."""
    template = ItemTemplate(
        name="test_fc",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(prompt="Which?", options=["option_a", "option_b"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(item_template_id=uuid4())

    # Two annotators with different seeds
    config1 = SimulatedAnnotatorConfig(strategy="random", random_state=123)
    annotator1 = RandomAnnotator(config1)
    annotations1 = [annotator1.annotate(item, template) for _ in range(100)]

    config2 = SimulatedAnnotatorConfig(strategy="random", random_state=456)
    annotator2 = RandomAnnotator(config2)
    annotations2 = [annotator2.annotate(item, template) for _ in range(100)]

    # Should be different
    assert annotations1 != annotations2


def test_annotate_batch() -> None:
    """Test batch annotation."""
    config = SimulatedAnnotatorConfig(strategy="random", random_state=42)
    annotator = RandomAnnotator(config)

    template = ItemTemplate(
        name="test_fc",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(prompt="Which?", options=["option_a", "option_b"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    items = [Item(item_template_id=uuid4()) for _ in range(10)]

    # Annotate batch
    annotations = annotator.annotate_batch(items, template)

    # Should have all item IDs
    assert len(annotations) == 10
    assert all(str(item.id) in annotations for item in items)
    # All should be valid options
    assert all(v in ["option_a", "option_b"] for v in annotations.values())
