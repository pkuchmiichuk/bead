"""Tests for distance-based annotator."""

from __future__ import annotations

from uuid import uuid4

from bead.config.simulation import NoiseModelConfig, SimulatedAnnotatorConfig
from bead.items.item import Item, ModelOutput
from bead.items.item_template import (
    ItemTemplate,
    PresentationSpec,
    ScaleBounds,
    TaskSpec,
)
from bead.simulation.annotators.distance_based import DistanceBasedAnnotator


def _create_model_output(score: float, idx: int = 0) -> ModelOutput:
    """Create ModelOutput with required fields."""
    return ModelOutput(
        model_name="test_model",
        model_version="1.0",
        operation="embedding",
        inputs={},
        output=score,
        cache_key=f"key_{score}_{idx}",
    )


def test_annotator_instantiation() -> None:
    """Test that annotator can be instantiated."""
    config = SimulatedAnnotatorConfig(
        strategy="distance", model_output_key="embedding", random_state=42
    )
    annotator = DistanceBasedAnnotator(config)
    assert annotator.config.strategy == "distance"
    assert annotator.config.model_output_key == "embedding"
    assert annotator.random_state == 42


def test_annotator_has_all_strategies() -> None:
    """Test that annotator has strategies for all task types."""
    config = SimulatedAnnotatorConfig(strategy="distance", model_output_key="embedding")
    annotator = DistanceBasedAnnotator(config)

    # Should have all 7 task type strategies
    assert "forced_choice" in annotator.strategies
    assert "binary" in annotator.strategies
    assert "ordinal_scale" in annotator.strategies
    assert "categorical" in annotator.strategies
    assert "magnitude" in annotator.strategies
    assert "multi_select" in annotator.strategies
    assert "free_text" in annotator.strategies


def test_annotate_forced_choice_with_embeddings() -> None:
    """Test forced choice annotation with embeddings."""
    config = SimulatedAnnotatorConfig(
        strategy="distance",
        model_output_key="embedding",
        noise_model=NoiseModelConfig(noise_type="none"),
        random_state=42,
    )
    annotator = DistanceBasedAnnotator(config)

    template = ItemTemplate(
        name="test_fc",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(prompt="Which?", options=["option_a", "option_b"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Embeddings as scores (higher = more similar = better)
    item = Item(
        item_template_id=uuid4(),
        model_outputs=[
            _create_model_output(5.0, 0),  # option_a
            _create_model_output(2.0, 1),  # option_b
        ],
    )

    # Generate multiple annotations to test distribution
    annotations = [annotator.annotate(item, template) for _ in range(100)]

    # Should mostly choose option_a (higher score)
    assert annotations.count("option_a") > 80


def test_annotate_binary_with_embeddings() -> None:
    """Test binary annotation with embeddings."""
    config = SimulatedAnnotatorConfig(
        strategy="distance",
        model_output_key="embedding",
        noise_model=NoiseModelConfig(noise_type="none"),
        random_state=42,
    )
    annotator = DistanceBasedAnnotator(config)

    template = ItemTemplate(
        name="test_binary",
        judgment_type="acceptability",
        task_type="binary",
        task_spec=TaskSpec(prompt="Is this good?"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # High similarity score
    item = Item(
        item_template_id=uuid4(),
        model_outputs=[_create_model_output(5.0)],
    )

    annotations = [annotator.annotate(item, template) for _ in range(100)]

    # Should be mostly True with high score
    assert sum(annotations) > 90


def test_annotate_with_temperature_noise() -> None:
    """Test that temperature noise is applied."""
    config = SimulatedAnnotatorConfig(
        strategy="distance",
        model_output_key="embedding",
        noise_model=NoiseModelConfig(noise_type="temperature", temperature=2.0),
        random_state=42,
    )
    annotator = DistanceBasedAnnotator(config)

    template = ItemTemplate(
        name="test_fc",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(prompt="Which?", options=["option_a", "option_b"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        model_outputs=[
            _create_model_output(3.0, 0),
            _create_model_output(2.0, 1),
        ],
    )

    # With temperature noise, should have more variability
    annotations = [annotator.annotate(item, template) for _ in range(100)]

    # Should still prefer option_a but less strongly
    option_a_count = annotations.count("option_a")
    assert 60 < option_a_count < 90


def test_annotate_ordinal_scale() -> None:
    """Test ordinal scale annotation."""
    config = SimulatedAnnotatorConfig(
        strategy="distance",
        model_output_key="embedding",
        noise_model=NoiseModelConfig(noise_type="none"),
        random_state=42,
    )
    annotator = DistanceBasedAnnotator(config)

    template = ItemTemplate(
        name="test_ordinal",
        judgment_type="plausibility",
        task_type="ordinal_scale",
        task_spec=TaskSpec(prompt="Rate:", scale_bounds=ScaleBounds(min=1, max=7)),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        model_outputs=[_create_model_output(2.0)],
    )

    annotation = annotator.annotate(item, template)

    # Should be an integer in range
    assert isinstance(annotation, int)
    assert 1 <= annotation <= 7


def test_annotate_categorical() -> None:
    """Test categorical annotation."""
    config = SimulatedAnnotatorConfig(
        strategy="distance",
        model_output_key="embedding",
        random_state=42,
    )
    annotator = DistanceBasedAnnotator(config)

    template = ItemTemplate(
        name="test_categorical",
        judgment_type="inference",
        task_type="categorical",
        task_spec=TaskSpec(prompt="Classify:", options=["cat_a", "cat_b", "cat_c"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        model_outputs=[
            _create_model_output(5.0, 0),
            _create_model_output(2.0, 1),
            _create_model_output(1.0, 2),
        ],
    )

    annotations = [annotator.annotate(item, template) for _ in range(100)]

    # Should mostly choose cat_a (highest score)
    assert annotations.count("cat_a") > 80


def test_annotate_magnitude() -> None:
    """Test magnitude annotation."""
    config = SimulatedAnnotatorConfig(
        strategy="distance", model_output_key="embedding", random_state=42
    )
    annotator = DistanceBasedAnnotator(config)

    template = ItemTemplate(
        name="test_magnitude",
        judgment_type="plausibility",
        task_type="magnitude",
        task_spec=TaskSpec(prompt="Estimate:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        model_outputs=[_create_model_output(-10.0)],
    )

    annotation = annotator.annotate(item, template)

    # Should be a positive float
    assert isinstance(annotation, float)
    assert annotation > 0


def test_annotate_multi_select() -> None:
    """Test multi-select annotation."""
    config = SimulatedAnnotatorConfig(
        strategy="distance", model_output_key="embedding", random_state=42
    )
    annotator = DistanceBasedAnnotator(config)

    template = ItemTemplate(
        name="test_multi",
        judgment_type="preference",
        task_type="multi_select",
        task_spec=TaskSpec(prompt="Select:", options=["opt_a", "opt_b", "opt_c"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        model_outputs=[
            _create_model_output(5.0, 0),  # High - likely selected
            _create_model_output(-5.0, 1),  # Low - likely not selected
            _create_model_output(0.0, 2),  # Medium - 50/50
        ],
    )

    annotations = [annotator.annotate(item, template) for _ in range(100)]
    all_selected = []
    for ann in annotations:
        all_selected.extend(ann)

    # opt_a should be selected often
    assert all_selected.count("opt_a") > 80
    # opt_b should be selected rarely
    assert all_selected.count("opt_b") < 20


def test_annotate_free_text() -> None:
    """Test free text annotation."""
    config = SimulatedAnnotatorConfig(
        strategy="distance", model_output_key="embedding", random_state=42
    )
    annotator = DistanceBasedAnnotator(config)

    template = ItemTemplate(
        name="test_free",
        judgment_type="comprehension",
        task_type="free_text",
        task_spec=TaskSpec(prompt="Describe:"),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        rendered_elements={"text": "Sample text"},
    )

    annotation = annotator.annotate(item, template)

    # Should be a string
    assert isinstance(annotation, str)
    assert len(annotation) > 0


def test_annotate_without_embeddings_falls_back() -> None:
    """Test fallback when embeddings are missing."""
    config = SimulatedAnnotatorConfig(
        strategy="distance", model_output_key="embedding", random_state=42
    )
    annotator = DistanceBasedAnnotator(config)

    template = ItemTemplate(
        name="test_fc",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(prompt="Which?", options=["option_a", "option_b"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    # Item without embeddings
    item = Item(item_template_id=uuid4())

    annotations = [annotator.annotate(item, template) for _ in range(100)]

    # Should fall back to random - both options should appear
    assert "option_a" in annotations
    assert "option_b" in annotations


def test_annotate_batch() -> None:
    """Test batch annotation."""
    config = SimulatedAnnotatorConfig(
        strategy="distance", model_output_key="embedding", random_state=42
    )
    annotator = DistanceBasedAnnotator(config)

    template = ItemTemplate(
        name="test_fc",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(prompt="Which?", options=["option_a", "option_b"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    items = [
        Item(
            item_template_id=uuid4(),
            model_outputs=[
                _create_model_output(5.0, 0),
                _create_model_output(2.0, 1),
            ],
        )
        for _ in range(10)
    ]

    annotations = annotator.annotate_batch(items, template)

    # Should have all item IDs
    assert len(annotations) == 10
    assert all(str(item.id) in annotations for item in items)


def test_annotate_reproducible_with_seed() -> None:
    """Test that annotations are reproducible with seed."""
    template = ItemTemplate(
        name="test_fc",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(prompt="Which?", options=["option_a", "option_b"]),
        presentation_spec=PresentationSpec(mode="static"),
    )

    item = Item(
        item_template_id=uuid4(),
        model_outputs=[
            _create_model_output(3.0, 0),
            _create_model_output(2.0, 1),
        ],
    )

    # Two annotators with same seed
    config1 = SimulatedAnnotatorConfig(
        strategy="distance", model_output_key="embedding", random_state=123
    )
    annotator1 = DistanceBasedAnnotator(config1)
    annotations1 = [annotator1.annotate(item, template) for _ in range(10)]

    config2 = SimulatedAnnotatorConfig(
        strategy="distance", model_output_key="embedding", random_state=123
    )
    annotator2 = DistanceBasedAnnotator(config2)
    annotations2 = [annotator2.annotate(item, template) for _ in range(10)]

    # Should be identical
    assert annotations1 == annotations2


def test_from_config() -> None:
    """Test creating annotator from config via from_config."""
    from bead.simulation.annotators.base import (  # noqa: PLC0415
        SimulatedAnnotator,
    )

    config = SimulatedAnnotatorConfig(
        strategy="distance", model_output_key="embedding", random_state=42
    )

    annotator = SimulatedAnnotator.from_config(config)

    assert isinstance(annotator, DistanceBasedAnnotator)
    assert annotator.config.model_output_key == "embedding"
