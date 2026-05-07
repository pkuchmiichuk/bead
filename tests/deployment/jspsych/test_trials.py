"""Tests for trial generation."""

from __future__ import annotations

from uuid import uuid4

import pytest
from didactic.api import ValidationError

from bead.data.range import Range
from bead.deployment.distribution import (
    DistributionStrategyType,
    ListDistributionStrategy,
)
from bead.deployment.jspsych.config import (
    ChoiceConfig,
    DemographicsConfig,
    DemographicsFieldConfig,
    ExperimentConfig,
    InstructionPage,
    InstructionsConfig,
    RatingScaleConfig,
    SpanDisplayConfig,
)
from bead.deployment.jspsych.trials import (
    SpanColorMap,
    _assign_span_colors,
    _generate_stimulus_html,
    _resolve_prompt_references,
    create_completion_trial,
    create_consent_trial,
    create_demographics_trial,
    create_instructions_trial,
    create_trial,
)
from bead.items.item import Item
from bead.items.item_template import (
    ItemTemplate,
    PresentationSpec,
    ScaleBounds,
    TaskSpec,
)
from bead.items.spans import Span, SpanLabel, SpanSegment
from bead.labels import parse_label_refs


class TestCreateTrial:
    """Tests for create_trial() with different experiment types."""

    def test_likert_rating(
        self,
        sample_item: Item,
        sample_item_template: ItemTemplate,
        sample_experiment_config: ExperimentConfig,
        sample_rating_config: RatingScaleConfig,
    ) -> None:
        """Test Likert rating trial creation."""
        trial = create_trial(
            item=sample_item,
            template=sample_item_template,
            experiment_config=sample_experiment_config,
            trial_number=0,
            rating_config=sample_rating_config,
        )

        assert trial["type"] == "bead-rating"
        assert trial["scale_min"] == 1
        assert trial["scale_max"] == 7
        assert trial["metadata"]["item_id"] == str(sample_item.id)
        assert trial["metadata"]["trial_type"] == "likert_rating"

    def test_slider_rating(
        self, sample_item: Item, sample_item_template: ItemTemplate
    ) -> None:
        """Test slider rating trial creation."""
        config = ExperimentConfig(
            experiment_type="slider_rating",
            title="Test",
            description="Test",
            instructions=InstructionsConfig.from_text("Test instructions"),
            distribution_strategy=ListDistributionStrategy(
                strategy_type=DistributionStrategyType.BALANCED
            ),
        )
        rating_config = RatingScaleConfig(scale=Range[int](min=1, max=7))

        trial = create_trial(
            item=sample_item,
            template=sample_item_template,
            experiment_config=config,
            trial_number=0,
            rating_config=rating_config,
        )

        assert trial["type"] == "bead-slider-rating"
        assert trial["slider_min"] == 1
        assert trial["slider_max"] == 7
        assert trial["metadata"]["trial_type"] == "slider_rating"

    def test_binary_choice(
        self, sample_item: Item, sample_item_template: ItemTemplate
    ) -> None:
        """Test binary choice trial creation."""
        config = ExperimentConfig(
            experiment_type="binary_choice",
            title="Test",
            description="Test",
            instructions=InstructionsConfig.from_text("Test instructions"),
            distribution_strategy=ListDistributionStrategy(
                strategy_type=DistributionStrategyType.BALANCED
            ),
        )
        choice_config = ChoiceConfig()

        trial = create_trial(
            item=sample_item,
            template=sample_item_template,
            experiment_config=config,
            trial_number=0,
            choice_config=choice_config,
        )

        assert trial["type"] == "bead-binary-choice"
        assert trial["choices"] == ["Yes", "No"]
        assert trial["metadata"]["trial_type"] == "binary_choice"

    def test_forced_choice(self) -> None:
        """Test forced choice trial creation."""
        template = ItemTemplate(
            name="test_template",
            description="Test item template",
            judgment_type="preference",
            task_type="forced_choice",
            task_spec=TaskSpec(
                prompt="Which is more natural?",
            ),
            presentation_spec=PresentationSpec(mode="static"),
        )

        item = Item(
            item_template_id=template.id,
            options=[
                "The cat broke the vase.",
                "The vase was broken by the cat.",
            ],
        )

        config = ExperimentConfig(
            experiment_type="forced_choice",
            title="Test",
            description="Test",
            instructions=InstructionsConfig.from_text("Test instructions"),
            distribution_strategy=ListDistributionStrategy(
                strategy_type=DistributionStrategyType.BALANCED
            ),
        )
        choice_config = ChoiceConfig()

        trial = create_trial(
            item=item,
            template=template,
            experiment_config=config,
            trial_number=0,
            choice_config=choice_config,
        )

        assert trial["type"] == "bead-forced-choice"
        assert len(trial["alternatives"]) == 2
        assert trial["metadata"]["trial_type"] == "forced_choice"

    def test_missing_config_raises_error(self) -> None:
        """Test trial creation with missing required config."""
        template = ItemTemplate(
            name="test_template",
            description="Test item template",
            judgment_type="acceptability",
            task_type="ordinal_scale",
            task_spec=TaskSpec(
                prompt="How natural is this sentence?",
                scale_bounds=ScaleBounds(min=1, max=7),
            ),
            presentation_spec=PresentationSpec(mode="static"),
        )

        item = Item(
            item_template_id=template.id,
            rendered_elements={"sentence": "Test sentence."},
        )

        config = ExperimentConfig(
            experiment_type="likert_rating",
            title="Test",
            description="Test",
            instructions=InstructionsConfig.from_text("Test instructions"),
            distribution_strategy=ListDistributionStrategy(
                strategy_type=DistributionStrategyType.BALANCED
            ),
        )

        with pytest.raises(ValueError, match="rating_config required"):
            create_trial(
                item=item,
                template=template,
                experiment_config=config,
                trial_number=0,
            )

    def test_unknown_type_raises_error(self) -> None:
        """Test that Pydantic validates experiment type."""
        Item(
            item_template_id=uuid4(),
            rendered_elements={"sentence": "Test sentence."},
        )

        # Test that Pydantic validation prevents invalid experiment types
        with pytest.raises(ValidationError):
            ExperimentConfig(
                experiment_type="invalid_type",  # type: ignore
                title="Test",
                description="Test",
                instructions=InstructionsConfig.from_text("Test instructions"),
                distribution_strategy=ListDistributionStrategy(
                    strategy_type=DistributionStrategyType.BALANCED
                ),
            )

    def test_metadata_inclusion(
        self, sample_item: Item, sample_item_template: ItemTemplate
    ) -> None:
        """Test that item metadata is included in trial data."""
        config = ExperimentConfig(
            experiment_type="likert_rating",
            title="Test",
            description="Test",
            instructions=InstructionsConfig.from_text("Test instructions"),
            distribution_strategy=ListDistributionStrategy(
                strategy_type=DistributionStrategyType.BALANCED
            ),
        )
        rating_config = RatingScaleConfig()

        trial = create_trial(
            item=sample_item,
            template=sample_item_template,
            experiment_config=config,
            trial_number=5,
            rating_config=rating_config,
        )

        assert trial["metadata"]["trial_number"] == 5
        assert trial["metadata"]["item_metadata"] == sample_item.item_metadata


class TestLikertConfiguration:
    """Tests for Likert trial configuration."""

    def test_custom_labels(self) -> None:
        """Test Likert trial with custom labels."""
        template = ItemTemplate(
            name="test_template",
            description="Test item template",
            judgment_type="acceptability",
            task_type="ordinal_scale",
            task_spec=TaskSpec(
                prompt="How natural is this sentence?",
                scale_bounds=ScaleBounds(min=1, max=5),
            ),
            presentation_spec=PresentationSpec(mode="static"),
        )

        item = Item(
            item_template_id=template.id,
            rendered_elements={"sentence": "Test sentence."},
        )

        config = ExperimentConfig(
            experiment_type="likert_rating",
            title="Test",
            description="Test",
            instructions=InstructionsConfig.from_text("Test instructions"),
            distribution_strategy=ListDistributionStrategy(
                strategy_type=DistributionStrategyType.BALANCED
            ),
        )

        rating_config = RatingScaleConfig(
            scale=Range[int](min=1, max=5),
            min_label="Strongly disagree",
            max_label="Strongly agree",
        )

        trial = create_trial(
            item=item,
            template=template,
            experiment_config=config,
            trial_number=0,
            rating_config=rating_config,
        )

        assert trial["scale_labels"]["1"] == "Strongly disagree"
        assert trial["scale_labels"]["5"] == "Strongly agree"
        assert trial["scale_min"] == 1
        assert trial["scale_max"] == 5


class TestSliderConfiguration:
    """Tests for slider trial configuration."""

    def test_require_movement(self) -> None:
        """Test slider trial with require_movement setting."""
        template = ItemTemplate(
            name="test_template",
            description="Test item template",
            judgment_type="acceptability",
            task_type="ordinal_scale",
            task_spec=TaskSpec(
                prompt="How natural is this sentence?",
                scale_bounds=ScaleBounds(min=1, max=7),
            ),
            presentation_spec=PresentationSpec(mode="static"),
        )

        item = Item(
            item_template_id=template.id,
            rendered_elements={"sentence": "Test sentence."},
        )

        config = ExperimentConfig(
            experiment_type="slider_rating",
            title="Test",
            description="Test",
            instructions=InstructionsConfig.from_text("Test instructions"),
            distribution_strategy=ListDistributionStrategy(
                strategy_type=DistributionStrategyType.BALANCED
            ),
        )

        rating_config = RatingScaleConfig(required=True)

        trial = create_trial(
            item=item,
            template=template,
            experiment_config=config,
            trial_number=0,
            rating_config=rating_config,
        )

        assert trial["require_movement"] is True


class TestStimulusGeneration:
    """Tests for stimulus HTML generation."""

    def test_single_element(self) -> None:
        """Test stimulus HTML generation with single element."""
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={"sentence": "The cat broke the vase."},
        )

        html = _generate_stimulus_html(item)

        assert "The cat broke the vase." in html
        assert "stimulus-container" in html

    def test_multiple_elements(self) -> None:
        """Test stimulus HTML generation with multiple elements."""
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={
                "sentence1": "First sentence.",
                "sentence2": "Second sentence.",
            },
        )

        html = _generate_stimulus_html(item, include_all=True)

        assert "First sentence." in html
        assert "Second sentence." in html

    def test_first_element_only(self) -> None:
        """Test stimulus HTML generation with only first element."""
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={
                "sentence": "Main sentence.",
                "choice_0": "Choice A",
                "choice_1": "Choice B",
            },
        )

        html = _generate_stimulus_html(item, include_all=False)

        assert html.count("<p>") == 1

    def test_empty_elements(self) -> None:
        """Test stimulus HTML generation with no elements."""
        item = Item(
            item_template_id=uuid4(),
            rendered_elements={},
        )

        html = _generate_stimulus_html(item)

        assert "No stimulus available" in html


class TestSpecialTrials:
    """Tests for instruction, consent, and completion trials."""

    def test_instruction_trial_simple_string(self) -> None:
        """Test instruction trial creation with simple string."""
        trial = create_instructions_trial("Please follow these instructions carefully.")

        assert trial["type"] == "html-keyboard-response"
        assert "Please follow these instructions carefully." in trial["stimulus"]
        assert trial["data"]["trial_type"] == "instructions"
        assert "Press any key" in trial["stimulus"]

    def test_instruction_trial_multi_page(self) -> None:
        """Test instruction trial creation with multi-page config."""
        config = InstructionsConfig(
            pages=[
                InstructionPage(
                    title="Welcome", content="<p>Welcome to the study!</p>"
                ),
                InstructionPage(
                    title="Task", content="<p>Your task is to rate sentences.</p>"
                ),
            ],
            allow_backwards=True,
            button_label_next="Continue",
            button_label_finish="Start Experiment",
        )
        trial = create_instructions_trial(config)

        assert trial["type"] == "instructions"
        assert len(trial["pages"]) == 2
        assert trial["allow_backward"] is True
        assert trial["button_label_next"] == "Continue"
        assert trial["button_label_finish"] == "Start Experiment"
        assert trial["data"]["trial_type"] == "instructions"

    def test_demographics_trial(self) -> None:
        """Test demographics trial creation."""
        config = DemographicsConfig(
            enabled=True,
            title="About You",
            fields=[
                DemographicsFieldConfig(
                    name="age",
                    field_type="number",
                    label="Your Age",
                    required=True,
                ),
                DemographicsFieldConfig(
                    name="education",
                    field_type="dropdown",
                    label="Education Level",
                    options=["High School", "Bachelors", "Masters", "PhD"],
                ),
            ],
            submit_button_text="Next",
        )
        trial = create_demographics_trial(config)

        assert trial["type"] == "survey"
        assert trial["title"] == "About You"
        assert trial["button_label_finish"] == "Next"
        assert trial["data"]["trial_type"] == "demographics"

    def test_consent_trial(self) -> None:
        """Test consent trial creation."""
        consent_text = "This study involves rating sentences."

        trial = create_consent_trial(consent_text)

        assert trial["type"] == "html-button-response"
        assert consent_text in trial["stimulus"]
        assert trial["choices"] == ["I agree", "I do not agree"]
        assert trial["data"]["trial_type"] == "consent"

    def test_completion_trial_default(self) -> None:
        """Test completion trial creation with default message."""
        trial = create_completion_trial()

        assert trial["type"] == "html-keyboard-response"
        assert "Thank you for participating!" in trial["stimulus"]
        assert trial["choices"] == "NO_KEYS"
        assert trial["data"]["trial_type"] == "completion"

    def test_completion_trial_custom_message(self) -> None:
        """Test completion trial with custom message."""
        custom_message = "Great job! Your responses have been recorded."

        trial = create_completion_trial(completion_message=custom_message)

        assert custom_message in trial["stimulus"]


class TestParsePromptReferences:
    """Tests for parse_label_refs()."""

    def test_no_references(self) -> None:
        """Plain text without references returns an empty tuple."""
        refs = parse_label_refs("How natural is this sentence?")

        assert refs == ()

    def test_auto_fill_reference(self) -> None:
        """Single auto-fill reference is parsed with label and no display_text."""
        refs = parse_label_refs("How natural is [[agent]]?")

        assert len(refs) == 1
        assert refs[0].label == "agent"
        assert refs[0].display_text is None

    def test_explicit_text_reference(self) -> None:
        """Explicit text reference is parsed with both label and display_text."""
        refs = parse_label_refs("Did [[event:the breaking]] happen?")

        assert len(refs) == 1
        assert refs[0].label == "event"
        assert refs[0].display_text == "the breaking"

    def test_multiple_references(self) -> None:
        """Multiple references are parsed in order of appearance."""
        refs = parse_label_refs("Did [[agent]] cause [[event:the breaking]]?")

        assert len(refs) == 2
        assert refs[0].label == "agent"
        assert refs[0].display_text is None
        assert refs[1].label == "event"
        assert refs[1].display_text == "the breaking"


class TestAssignSpanColors:
    """Tests for _assign_span_colors() and SpanColorMap."""

    def test_same_label_same_color(self) -> None:
        """Two spans with the same label receive identical colors."""
        spans = [
            Span(
                span_id="s0",
                segments=[SpanSegment(element_name="text", indices=[0])],
                label=SpanLabel(label="agent"),
            ),
            Span(
                span_id="s1",
                segments=[SpanSegment(element_name="text", indices=[1])],
                label=SpanLabel(label="agent"),
            ),
        ]
        span_display = SpanDisplayConfig()

        color_map = _assign_span_colors(spans, span_display)

        assert color_map.light_by_span_id["s0"] == color_map.light_by_span_id["s1"]
        assert color_map.dark_by_span_id["s0"] == color_map.dark_by_span_id["s1"]

    def test_different_labels_different_colors(self) -> None:
        """Two spans with different labels receive different light colors."""
        spans = [
            Span(
                span_id="s0",
                segments=[SpanSegment(element_name="text", indices=[0])],
                label=SpanLabel(label="agent"),
            ),
            Span(
                span_id="s1",
                segments=[SpanSegment(element_name="text", indices=[1])],
                label=SpanLabel(label="patient"),
            ),
        ]
        span_display = SpanDisplayConfig()

        color_map = _assign_span_colors(spans, span_display)

        assert color_map.light_by_span_id["s0"] != color_map.light_by_span_id["s1"]

    def test_unlabeled_span_gets_own_color(self) -> None:
        """An unlabeled span receives its own unique color."""
        spans = [
            Span(
                span_id="s0",
                segments=[SpanSegment(element_name="text", indices=[0])],
                label=SpanLabel(label="agent"),
            ),
            Span(
                span_id="s1",
                segments=[SpanSegment(element_name="text", indices=[1])],
                label=None,
            ),
        ]
        span_display = SpanDisplayConfig()

        color_map = _assign_span_colors(spans, span_display)

        assert "s1" in color_map.light_by_span_id
        assert color_map.light_by_span_id["s1"] != color_map.light_by_span_id["s0"]


class TestResolvePromptReferences:
    """Tests for _resolve_prompt_references()."""

    @pytest.fixture
    def span_item(self) -> Item:
        """Create an item with tokenized elements and spans."""
        return Item(
            item_template_id=uuid4(),
            rendered_elements={"text": "The boy broke the vase."},
            tokenized_elements={
                "text": ["The", "boy", "broke", "the", "vase", "."],
            },
            token_space_after={"text": [True, True, True, True, False, False]},
            spans=[
                Span(
                    span_id="span_0",
                    segments=[
                        SpanSegment(element_name="text", indices=[0, 1]),
                    ],
                    label=SpanLabel(label="breaker"),
                ),
                Span(
                    span_id="span_1",
                    segments=[
                        SpanSegment(element_name="text", indices=[2]),
                    ],
                    label=SpanLabel(label="event"),
                ),
            ],
        )

    @pytest.fixture
    def color_map(self, span_item: Item) -> SpanColorMap:
        """Assign colors to the span_item's spans."""
        span_display = SpanDisplayConfig()
        return _assign_span_colors(span_item.spans, span_display)

    def test_no_refs_backward_compat(
        self, span_item: Item, color_map: SpanColorMap
    ) -> None:
        """Prompt without references is returned unchanged."""
        result = _resolve_prompt_references("How natural?", span_item, color_map)

        assert result == "How natural?"

    def test_auto_fill_produces_html(
        self, span_item: Item, color_map: SpanColorMap
    ) -> None:
        """Auto-fill reference produces highlighted HTML with span text."""
        result = _resolve_prompt_references(
            "Did [[breaker]] do it?", span_item, color_map
        )

        assert "bead-q-highlight" in result
        assert "bead-q-chip" in result
        assert "breaker" in result
        assert "The boy" in result

    def test_explicit_text_produces_html(
        self, span_item: Item, color_map: SpanColorMap
    ) -> None:
        """Explicit text reference renders the specified text with label."""
        result = _resolve_prompt_references(
            "Did [[event:the breaking]] happen?", span_item, color_map
        )

        assert "the breaking" in result
        assert "event" in result
        assert "bead-q-highlight" in result

    def test_nonexistent_label_raises_value_error(
        self, span_item: Item, color_map: SpanColorMap
    ) -> None:
        """Reference to a nonexistent label raises ValueError."""
        with pytest.raises(ValueError, match="nonexistent"):
            _resolve_prompt_references(
                "Did [[nonexistent]] do it?", span_item, color_map
            )

    def test_color_consistency(self, span_item: Item, color_map: SpanColorMap) -> None:
        """Resolved HTML uses the same colors as the color map."""
        result = _resolve_prompt_references(
            "Did [[breaker]] do it?", span_item, color_map
        )

        expected_light = color_map.light_by_label["breaker"]
        expected_dark = color_map.dark_by_label["breaker"]

        assert expected_light in result
        assert expected_dark in result

    def test_same_label_twice(self, span_item: Item, color_map: SpanColorMap) -> None:
        """Two references to the same label use the same background color."""
        result = _resolve_prompt_references(
            "Did [[breaker]] meet [[breaker:him]]?", span_item, color_map
        )

        expected_light = color_map.light_by_label["breaker"]
        assert result.count(expected_light) == 2
