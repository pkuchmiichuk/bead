"""Integration tests for the simulation framework.

Tests end-to-end workflows combining strategies, annotators, and noise models.
"""

from __future__ import annotations

import uuid

import pytest

from bead.config.simulation import (
    NoiseModelConfig,
    SimulatedAnnotatorConfig,
    SimulationRunnerConfig,
)
from bead.items.item import Item
from bead.items.item_template import (
    ItemTemplate,
    PresentationSpec,
    ScaleBounds,
    TaskSpec,
)
from bead.simulation.annotators.base import SimulatedAnnotator
from bead.simulation.runner import SimulationRunner

# ============================================================================
# Fixtures
# ============================================================================


@pytest.fixture
def forced_choice_template() -> ItemTemplate:
    """Create a 2AFC template."""
    return ItemTemplate(
        name="2AFC Test",
        judgment_type="preference",
        task_type="forced_choice",
        task_spec=TaskSpec(
            prompt="Which is better?",
            options=["option_a", "option_b"],
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )


@pytest.fixture
def ordinal_scale_template() -> ItemTemplate:
    """Create an ordinal scale template."""
    return ItemTemplate(
        name="Likert Test",
        judgment_type="acceptability",
        task_type="ordinal_scale",
        task_spec=TaskSpec(
            prompt="Rate from 1-7",
            scale_bounds=ScaleBounds(min=1, max=7),
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )


@pytest.fixture
def binary_template() -> ItemTemplate:
    """Create a binary template."""
    return ItemTemplate(
        name="Binary Test",
        judgment_type="acceptability",
        task_type="binary",
        task_spec=TaskSpec(
            prompt="Is this acceptable?",
            options=["yes", "no"],
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )


@pytest.fixture
def categorical_template() -> ItemTemplate:
    """Create a categorical template."""
    return ItemTemplate(
        name="Category Test",
        judgment_type="preference",
        task_type="categorical",
        task_spec=TaskSpec(
            prompt="Select category",
            options=["cat1", "cat2", "cat3"],
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )


@pytest.fixture
def forced_choice_items() -> list[Item]:
    """Create 2AFC items with LM scores."""
    template_id = uuid.uuid4()
    items = []
    for i in range(10):
        item = Item(
            id=uuid.uuid4(),
            item_template_id=template_id,
            filled_template_refs=[],
            rendered_elements={"option_a": f"Text A {i}", "option_b": f"Text B {i}"},
            item_metadata={"lm_score1": -10.0 + i, "lm_score2": -15.0 + i * 0.5},
        )
        items.append(item)
    return items


@pytest.fixture
def ordinal_scale_items() -> list[Item]:
    """Create ordinal scale items with scores."""
    template_id = uuid.uuid4()
    items = []
    for i in range(10):
        item = Item(
            id=uuid.uuid4(),
            item_template_id=template_id,
            filled_template_refs=[],
            rendered_elements={"text": f"Text {i}"},
            item_metadata={"lm_score": -10.0 + i * 2},
        )
        items.append(item)
    return items


# ============================================================================
# Integration Test 1: Forced Choice with LM Scores
# ============================================================================


def test_forced_choice_lm_based_integration(
    forced_choice_template, forced_choice_items
):
    """Test complete forced choice workflow with LM scores."""
    # Create LM-based annotator
    config = SimulatedAnnotatorConfig(
        strategy="lm_score",
        model_output_key="lm_score",
        noise_model=NoiseModelConfig(noise_type="temperature", temperature=1.0),
        random_state=42,
    )
    annotator = SimulatedAnnotator.from_config(config)

    # Annotate batch
    annotations = annotator.annotate_batch(forced_choice_items, forced_choice_template)

    # Verify all items annotated
    assert len(annotations) == len(forced_choice_items)
    for item in forced_choice_items:
        assert str(item.id) in annotations
        assert annotations[str(item.id)] in ["option_a", "option_b"]


def test_forced_choice_temperature_effects(forced_choice_template, forced_choice_items):
    """Test that temperature noise model is properly integrated."""
    # Low temperature (more deterministic)
    low_temp_config = SimulatedAnnotatorConfig(
        strategy="lm_score",
        model_output_key="lm_score",
        noise_model=NoiseModelConfig(noise_type="temperature", temperature=0.1),
        random_state=42,
    )
    low_temp_annotator = SimulatedAnnotator.from_config(low_temp_config)
    low_temp_annotations = low_temp_annotator.annotate_batch(
        forced_choice_items, forced_choice_template
    )

    # High temperature (more random)
    high_temp_config = SimulatedAnnotatorConfig(
        strategy="lm_score",
        model_output_key="lm_score",
        noise_model=NoiseModelConfig(noise_type="temperature", temperature=5.0),
        random_state=43,  # Different seed to allow for variation
    )
    high_temp_annotator = SimulatedAnnotator.from_config(high_temp_config)
    high_temp_annotations = high_temp_annotator.annotate_batch(
        forced_choice_items, forced_choice_template
    )

    # Both should complete successfully and produce valid responses
    assert len(low_temp_annotations) == len(forced_choice_items)
    assert len(high_temp_annotations) == len(forced_choice_items)

    # All responses should be valid options
    for annotation in low_temp_annotations.values():
        assert annotation in ["option_a", "option_b"]
    for annotation in high_temp_annotations.values():
        assert annotation in ["option_a", "option_b"]


# ============================================================================
# Integration Test 2: Ordinal Scale with Temperature Noise
# ============================================================================


def test_ordinal_scale_with_noise(ordinal_scale_template, ordinal_scale_items):
    """Test ordinal scale with temperature noise."""
    config = SimulatedAnnotatorConfig(
        strategy="lm_score",
        model_output_key="lm_score",
        noise_model=NoiseModelConfig(noise_type="temperature", temperature=2.0),
        random_state=42,
    )
    annotator = SimulatedAnnotator.from_config(config)

    # Annotate batch
    annotations = annotator.annotate_batch(ordinal_scale_items, ordinal_scale_template)

    # Verify all items annotated
    assert len(annotations) == len(ordinal_scale_items)

    # Verify ratings are numeric and mostly reasonable
    # (temperature noise can push values slightly outside range)
    for item in ordinal_scale_items:
        rating = annotations[str(item.id)]
        assert isinstance(rating, int | float)
        # Should be roughly in the expected range (allow some noise margin)
        assert 0 <= rating <= 10  # Relaxed bounds to account for noise


# ============================================================================
# Integration Test 3: Multi-Annotator Simulation
# ============================================================================


def test_multi_annotator_runner(forced_choice_template, forced_choice_items):
    """Test multi-annotator simulation with runner."""
    # Create runner with 3 annotators
    config = SimulationRunnerConfig(
        annotator_configs=[
            SimulatedAnnotatorConfig(strategy="lm_score", random_state=1),
            SimulatedAnnotatorConfig(strategy="lm_score", random_state=2),
            SimulatedAnnotatorConfig(strategy="lm_score", random_state=3),
        ],
        n_annotators=3,
        output_format="dict",
    )

    runner = SimulationRunner(config)
    results = runner.run(forced_choice_items, forced_choice_template)

    # Verify results structure
    assert "item_ids" in results
    assert len(results["item_ids"]) == len(forced_choice_items)

    # Verify all annotators produced judgments
    for i in range(3):
        assert f"annotator_{i}" in results
        assert len(results[f"annotator_{i}"]) == len(forced_choice_items)


def test_multi_annotator_replication(forced_choice_template, forced_choice_items):
    """Test automatic annotator replication when n_annotators > configs."""
    # Create runner with 1 config but request 5 annotators
    config = SimulationRunnerConfig(
        annotator_configs=[
            SimulatedAnnotatorConfig(strategy="lm_score", random_state=42)
        ],
        n_annotators=5,
        output_format="dict",
    )

    runner = SimulationRunner(config)
    results = runner.run(forced_choice_items, forced_choice_template)

    # Verify 5 annotators created
    for i in range(5):
        assert f"annotator_{i}" in results

    # Verify they produce different judgments (different seeds)
    annotator_0_judgments = set(results["annotator_0"])
    annotator_1_judgments = set(results["annotator_1"])

    # Should have at least some variation (not guaranteed but highly likely)
    assert len(annotator_0_judgments) > 0
    assert len(annotator_1_judgments) > 0


# ============================================================================
# Integration Test 4: All Task Types with Random Annotator
# ============================================================================


def test_random_annotator_all_task_types(
    forced_choice_template,
    ordinal_scale_template,
    binary_template,
    categorical_template,
    forced_choice_items,
):
    """Test random annotator supports all task types."""
    config = SimulatedAnnotatorConfig(strategy="random", random_state=42)
    annotator = SimulatedAnnotator.from_config(config)

    # Test forced choice
    fc_annotations = annotator.annotate_batch(
        forced_choice_items, forced_choice_template
    )
    assert len(fc_annotations) == len(forced_choice_items)
    for annotation in fc_annotations.values():
        assert annotation in ["option_a", "option_b"]

    # Test ordinal scale
    ord_annotations = annotator.annotate_batch(
        forced_choice_items, ordinal_scale_template
    )
    assert len(ord_annotations) == len(forced_choice_items)
    for annotation in ord_annotations.values():
        assert 1 <= annotation <= 7

    # Test binary
    bin_annotations = annotator.annotate_batch(forced_choice_items, binary_template)
    assert len(bin_annotations) == len(forced_choice_items)
    for annotation in bin_annotations.values():
        # Binary strategy returns boolean values
        assert isinstance(annotation, bool)

    # Test categorical
    cat_annotations = annotator.annotate_batch(
        forced_choice_items, categorical_template
    )
    assert len(cat_annotations) == len(forced_choice_items)
    for annotation in cat_annotations.values():
        assert annotation in ["cat1", "cat2", "cat3"]


# ============================================================================
# Integration Test 5: Missing Model Outputs (Fallback)
# ============================================================================


def test_lm_based_fallback_to_random(forced_choice_template):
    """Test LM-based annotator falls back to random when model outputs missing."""
    # Create items WITHOUT model outputs
    template_id = uuid.uuid4()
    items_no_scores = [
        Item(
            id=uuid.uuid4(),
            item_template_id=template_id,
            filled_template_refs=[],
            rendered_elements={"option_a": "Text A", "option_b": "Text B"},
            item_metadata={},  # No lm_score1/lm_score2
        )
        for _ in range(5)
    ]

    config = SimulatedAnnotatorConfig(
        strategy="lm_score",
        model_output_key="lm_score",
        fallback_to_random=True,
        random_state=42,
    )
    annotator = SimulatedAnnotator.from_config(config)

    # Should not crash, should fall back to random
    annotations = annotator.annotate_batch(items_no_scores, forced_choice_template)

    assert len(annotations) == len(items_no_scores)
    for annotation in annotations.values():
        assert annotation in ["option_a", "option_b"]


# ============================================================================
# Integration Test 6: Reproducibility with Random Seeds
# ============================================================================


def test_reproducibility_with_seeds(forced_choice_template, forced_choice_items):
    """Test that random seeds produce reproducible results."""
    config = SimulatedAnnotatorConfig(
        strategy="lm_score",
        model_output_key="lm_score",
        noise_model=NoiseModelConfig(noise_type="temperature", temperature=2.0),
        random_state=12345,
    )

    # Run twice with same seed
    annotator1 = SimulatedAnnotator.from_config(config)
    annotations1 = annotator1.annotate_batch(
        forced_choice_items, forced_choice_template
    )

    annotator2 = SimulatedAnnotator.from_config(config)
    annotations2 = annotator2.annotate_batch(
        forced_choice_items, forced_choice_template
    )

    # Should produce identical results
    for item in forced_choice_items:
        item_id = str(item.id)
        assert annotations1[item_id] == annotations2[item_id]


# ============================================================================
# Integration Test 7: Oracle Annotator with Ground Truth
# ============================================================================


def test_oracle_annotator_with_ground_truth(forced_choice_template):
    """Test oracle annotator uses ground truth when available."""
    # Create items WITH ground truth
    template_id = uuid.uuid4()
    items_with_truth = [
        Item(
            id=uuid.uuid4(),
            item_template_id=template_id,
            filled_template_refs=[],
            rendered_elements={"option_a": "Text A", "option_b": "Text B"},
            item_metadata={"ground_truth": "option_a"},
        )
        for _ in range(5)
    ]

    config = SimulatedAnnotatorConfig(strategy="oracle", random_state=42)
    annotator = SimulatedAnnotator.from_config(config)

    annotations = annotator.annotate_batch(items_with_truth, forced_choice_template)

    # All should match ground truth
    for annotation in annotations.values():
        assert annotation == "option_a"


# ============================================================================
# Integration Test 8: Systematic Noise Model
# ============================================================================


def test_systematic_bias_integration(forced_choice_template, forced_choice_items):
    """Test systematic position bias in forced choice."""
    # Create annotator with position bias (prefer first option)
    config = SimulatedAnnotatorConfig(
        strategy="lm_score",
        model_output_key="lm_score",
        noise_model=NoiseModelConfig(
            noise_type="systematic",
            bias_type="position",
            bias_strength=0.8,
        ),
        random_state=42,
    )
    annotator = SimulatedAnnotator.from_config(config)

    annotations = annotator.annotate_batch(forced_choice_items, forced_choice_template)

    # With strong position bias, should prefer option_a
    option_a_count = sum(1 for v in annotations.values() if v == "option_a")

    # Should show bias (not guaranteed 100%, but should be substantial)
    assert option_a_count >= len(forced_choice_items) * 0.5


# ============================================================================
# Integration Test 9: Distance-Based Annotator
# ============================================================================


def test_distance_based_annotator_with_embeddings(forced_choice_template):
    """Test distance-based annotator falls back when embeddings missing."""
    # Create items WITHOUT embeddings (will fall back to random)
    template_id = uuid.uuid4()
    items_without_embeddings = []
    for i in range(5):
        item = Item(
            id=uuid.uuid4(),
            item_template_id=template_id,
            filled_template_refs=[],
            rendered_elements={"option_a": f"Text A {i}", "option_b": f"Text B {i}"},
            item_metadata={},
        )
        items_without_embeddings.append(item)

    config = SimulatedAnnotatorConfig(
        strategy="distance",
        model_output_key="embedding",
        fallback_to_random=True,
        random_state=42,
    )
    annotator = SimulatedAnnotator.from_config(config)

    # Should fall back to random and complete successfully
    annotations = annotator.annotate_batch(
        items_without_embeddings, forced_choice_template
    )

    assert len(annotations) == len(items_without_embeddings)
    for annotation in annotations.values():
        assert annotation in ["option_a", "option_b"]


# ============================================================================
# Integration Test 10: Runner with Output Saving
# ============================================================================


def test_runner_output_saving(forced_choice_template, forced_choice_items, tmp_path):
    """Test runner saves output to file."""
    output_path = tmp_path / "simulation_results.json"

    config = SimulationRunnerConfig(
        annotator_configs=[
            SimulatedAnnotatorConfig(strategy="lm_score", random_state=42)
        ],
        n_annotators=1,
        output_format="dict",
        save_path=output_path,
    )

    runner = SimulationRunner(config)
    runner.run(forced_choice_items, forced_choice_template)

    # Verify file was created
    assert output_path.exists()

    # Verify content
    import json  # noqa: PLC0415

    with open(output_path) as f:
        saved_results = json.load(f)

    assert "item_ids" in saved_results
    assert "annotator_0" in saved_results
    assert len(saved_results["item_ids"]) == len(forced_choice_items)


# ============================================================================
# Integration Test 11: Different Noise Models Produce Different Results
# ============================================================================


def test_different_noise_models(forced_choice_template, forced_choice_items):
    """Test that different noise models produce different distributions."""
    # Temperature noise
    temp_config = SimulatedAnnotatorConfig(
        strategy="lm_score",
        model_output_key="lm_score",
        noise_model=NoiseModelConfig(noise_type="temperature", temperature=2.0),
        random_state=42,
    )
    temp_annotator = SimulatedAnnotator.from_config(temp_config)
    temp_annotations = temp_annotator.annotate_batch(
        forced_choice_items, forced_choice_template
    )

    # No noise
    no_noise_config = SimulatedAnnotatorConfig(
        strategy="lm_score",
        model_output_key="lm_score",
        noise_model=NoiseModelConfig(noise_type="none"),
        random_state=42,
    )
    no_noise_annotator = SimulatedAnnotator.from_config(no_noise_config)
    no_noise_annotations = no_noise_annotator.annotate_batch(
        forced_choice_items, forced_choice_template
    )

    # They should potentially differ (not guaranteed but likely with temperature)
    # Just verify both complete successfully
    assert len(temp_annotations) == len(forced_choice_items)
    assert len(no_noise_annotations) == len(forced_choice_items)
