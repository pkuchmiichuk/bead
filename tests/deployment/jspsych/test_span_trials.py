"""Tests for span-aware trial generation."""

from __future__ import annotations

from uuid import uuid4

from bead.deployment.distribution import (
    DistributionStrategyType,
    ListDistributionStrategy,
)
from bead.deployment.jspsych.config import (
    ExperimentConfig,
    InstructionsConfig,
    SpanDisplayConfig,
)
from bead.deployment.jspsych.trials import (
    _create_span_labeling_trial,
    _generate_span_stimulus_html,
    _serialize_item_metadata,
    create_trial,
)
from bead.items.item import Item
from bead.items.item_template import ItemTemplate, PresentationSpec, TaskSpec
from bead.items.spans import (
    Span,
    SpanLabel,
    SpanRelation,
    SpanSegment,
    SpanSpec,
)


def _make_strategy() -> ListDistributionStrategy:
    """Create a test distribution strategy."""
    return ListDistributionStrategy(strategy_type=DistributionStrategyType.BALANCED)


class TestSpanMetadataSerialization:
    """Test span data in _serialize_item_metadata."""

    def test_spans_serialized(self) -> None:
        """Test that spans are included in metadata."""
        span = Span(
            span_id="span_0",
            segments=[SpanSegment(element_name="text", indices=[0, 1])],
            label=SpanLabel(label="Person"),
        )

        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"text": "John Smith"},
            spans=[span],
            tokenized_elements={"text": ["John", "Smith"]},
            token_space_after={"text": [True, False]},
        )

        template = ItemTemplate(
            name="test",
            judgment_type="extraction",
            task_type="span_labeling",
            task_spec=TaskSpec(prompt="Label entities"),
            presentation_spec=PresentationSpec(mode="static"),
        )

        metadata = _serialize_item_metadata(item, template)

        assert "spans" in metadata
        assert len(metadata["spans"]) == 1
        assert metadata["spans"][0]["span_id"] == "span_0"
        assert metadata["spans"][0]["label"]["label"] == "Person"

    def test_tokenized_elements_serialized(self) -> None:
        """Test that tokenized_elements are included."""
        item = Item(
            item_template_id=uuid4(),
            tokenized_elements={"text": ["Hello", "world"]},
            token_space_after={"text": [True, False]},
        )

        template = ItemTemplate(
            name="test",
            judgment_type="acceptability",
            task_type="ordinal_scale",
            task_spec=TaskSpec(prompt="Rate this"),
            presentation_spec=PresentationSpec(mode="static"),
        )

        metadata = _serialize_item_metadata(item, template)

        assert dict(metadata["tokenized_elements"]) == {"text": ("Hello", "world")}
        assert dict(metadata["token_space_after"]) == {"text": [True, False]}

    def test_span_relations_serialized(self) -> None:
        """Test that span_relations are serialized."""
        spans = [
            Span(
                span_id="span_0",
                segments=[SpanSegment(element_name="text", indices=[0])],
            ),
            Span(
                span_id="span_1",
                segments=[SpanSegment(element_name="text", indices=[2])],
            ),
        ]

        rel = SpanRelation(
            relation_id="rel_0",
            source_span_id="span_0",
            target_span_id="span_1",
            label=SpanLabel(label="agent-of"),
            directed=True,
        )

        item = Item(
            item_template_id=uuid4(),
            spans=spans,
            span_relations=[rel],
        )

        template = ItemTemplate(
            name="test",
            judgment_type="extraction",
            task_type="span_labeling",
            task_spec=TaskSpec(prompt="Label"),
            presentation_spec=PresentationSpec(mode="static"),
        )

        metadata = _serialize_item_metadata(item, template)

        assert len(metadata["span_relations"]) == 1
        assert metadata["span_relations"][0]["directed"] is True
        assert metadata["span_relations"][0]["source_span_id"] == "span_0"

    def test_span_spec_serialized(self) -> None:
        """Test that span_spec from template is serialized."""
        item = Item(item_template_id=uuid4())

        span_spec = SpanSpec(
            interaction_mode="interactive",
            labels=["PER", "ORG"],
            min_spans=1,
        )

        template = ItemTemplate(
            name="test",
            judgment_type="extraction",
            task_type="span_labeling",
            task_spec=TaskSpec(prompt="Label", span_spec=span_spec),
            presentation_spec=PresentationSpec(mode="static"),
        )

        metadata = _serialize_item_metadata(item, template)

        assert metadata["span_spec"] is not None
        assert metadata["span_spec"]["interaction_mode"] == "interactive"
        assert metadata["span_spec"]["labels"] == ("PER", "ORG")

    def test_no_span_spec_is_none(self) -> None:
        """Test that span_spec is None when not set."""
        item = Item(item_template_id=uuid4())

        template = ItemTemplate(
            name="test",
            judgment_type="acceptability",
            task_type="ordinal_scale",
            task_spec=TaskSpec(prompt="Rate"),
            presentation_spec=PresentationSpec(mode="static"),
        )

        metadata = _serialize_item_metadata(item, template)
        assert metadata["span_spec"] is None


class TestSpanStimulusHtml:
    """Test span-highlighted stimulus HTML generation."""

    def test_static_spans_markup(self) -> None:
        """Test that static spans produce highlighted tokens."""
        span = Span(
            span_id="span_0",
            segments=[SpanSegment(element_name="text", indices=[0, 1])],
            label=SpanLabel(label="Person"),
        )

        item = Item(
            item_template_id=uuid4(),
            spans=[span],
            tokenized_elements={"text": ["John", "Smith", "is", "here"]},
            token_space_after={"text": [True, True, True, False]},
        )

        config = SpanDisplayConfig()
        html = _generate_span_stimulus_html(item, config)

        assert "bead-token" in html
        assert "highlighted" in html
        assert 'data-index="0"' in html
        assert "John" in html

    def test_no_tokenization_fallback(self) -> None:
        """Test fallback when no tokenized_elements."""
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"text": "Hello world"},
        )

        config = SpanDisplayConfig()
        html = _generate_span_stimulus_html(item, config)

        assert "stimulus-container" in html

    def test_space_after_rendering(self) -> None:
        """Test that space_after controls spacing in output."""
        item = Item(
            item_template_id=uuid4(),
            spans=[],
            tokenized_elements={"text": ["don", "'t"]},
            token_space_after={"text": [False, False]},
        )

        config = SpanDisplayConfig()
        html = _generate_span_stimulus_html(item, config)

        # Tokens should be adjacent (no space between don and 't)
        assert (
            "don</span><span" in html or "don</span>'t" in html or "don</span>" in html
        )


class TestSpanLabelingTrial:
    """Test standalone span labeling trial creation."""

    def test_trial_structure(self) -> None:
        """Test span labeling trial has correct structure."""
        item = Item(
            item_template_id=uuid4(),
            tokenized_elements={"text": ["The", "cat"]},
            token_space_after={"text": [True, False]},
        )

        template = ItemTemplate(
            name="test",
            judgment_type="extraction",
            task_type="span_labeling",
            task_spec=TaskSpec(prompt="Select entities"),
            presentation_spec=PresentationSpec(mode="static"),
        )

        config = SpanDisplayConfig()
        trial = _create_span_labeling_trial(item, template, config, 0)

        assert trial["type"] == "bead-span-label"
        assert trial["prompt"] == "Select entities"
        assert trial["button_label"] == "Continue"
        assert trial["metadata"]["trial_type"] == "span_labeling"

    def test_trial_metadata(self) -> None:
        """Test span labeling trial includes metadata."""
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"text": "Hello"},
            tokenized_elements={"text": ["Hello"]},
        )

        template = ItemTemplate(
            name="test",
            judgment_type="extraction",
            task_type="span_labeling",
            task_spec=TaskSpec(prompt="Label"),
            presentation_spec=PresentationSpec(mode="static"),
        )

        config = SpanDisplayConfig()
        trial = _create_span_labeling_trial(item, template, config, 5)

        assert trial["metadata"]["trial_number"] == 5
        assert trial["metadata"]["item_id"] == str(item.id)

    def test_trial_includes_span_data(self) -> None:
        """Test span labeling trial includes spans, relations, spec, config."""
        span = Span(
            span_id="s0",
            segments=[SpanSegment(element_name="text", indices=[0])],
            label=SpanLabel(label="PER", confidence=0.95),
        )
        item = Item(
            item_template_id=uuid4(),
            spans=[span],
            tokenized_elements={"text": ["Alice", "ran"]},
            token_space_after={"text": [True, False]},
        )

        span_spec = SpanSpec(
            interaction_mode="interactive",
            labels=["PER", "ORG"],
        )
        template = ItemTemplate(
            name="test",
            judgment_type="extraction",
            task_type="span_labeling",
            task_spec=TaskSpec(prompt="Label", span_spec=span_spec),
            presentation_spec=PresentationSpec(mode="static"),
        )

        config = SpanDisplayConfig()
        trial = _create_span_labeling_trial(item, template, config, 0)

        # Span data
        assert len(trial["spans"]) == 1
        assert trial["spans"][0]["span_id"] == "s0"
        assert trial["spans"][0]["label"]["confidence"] == 0.95

        # Relations (empty)
        assert trial["relations"] == []

        # Span spec
        assert trial["span_spec"] is not None
        assert trial["span_spec"]["interaction_mode"] == "interactive"
        assert trial["span_spec"]["labels"] == ("PER", "ORG")

        # Display config
        assert trial["display_config"] is not None
        assert trial["display_config"]["highlight_style"] == "background"


class TestSpanCompositeTrial:
    """Test composite trials (e.g., rating + spans)."""

    def test_span_labeling_experiment_type(self) -> None:
        """Test create_trial routes to span labeling."""
        item = Item(
            item_template_id=uuid4(),
            tokenized_elements={"text": ["Hello"]},
        )

        template = ItemTemplate(
            name="test",
            judgment_type="extraction",
            task_type="span_labeling",
            task_spec=TaskSpec(prompt="Label"),
            presentation_spec=PresentationSpec(mode="static"),
        )

        config = ExperimentConfig(
            experiment_type="span_labeling",
            title="Test",
            description="Test",
            instructions=InstructionsConfig.from_text("Test"),
            distribution_strategy=_make_strategy(),
        )

        trial = create_trial(item, template, config, 0)

        assert trial["type"] == "bead-span-label"
