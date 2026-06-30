"""Configuration models for jsPsych experiment generation."""

from __future__ import annotations

from typing import Literal

import didactic.api as dx

from bead.config.deployment import SlopitIntegrationConfig
from bead.data.range import Range
from bead.deployment.distribution import ListDistributionStrategy

type ExperimentType = Literal[
    "likert_rating",
    "slider_rating",
    "binary_choice",
    "forced_choice",
    "span_labeling",
]

type UITheme = Literal["light", "dark", "auto"]


def _default_span_color_palette() -> tuple[str, ...]:
    return (
        "#BBDEFB",
        "#C8E6C9",
        "#FFE0B2",
        "#F8BBD0",
        "#D1C4E9",
        "#B2EBF2",
        "#DCEDC8",
        "#FFD54F",
    )


def _default_span_dark_palette() -> tuple[str, ...]:
    return (
        "#1565C0",
        "#2E7D32",
        "#E65100",
        "#AD1457",
        "#4527A0",
        "#00838F",
        "#558B2F",
        "#F9A825",
    )


class SpanDisplayConfig(dx.Model):
    """Visual configuration for span rendering.

    Attributes
    ----------
    highlight_style : Literal["background", "underline", "border"]
        How to visually indicate spans.
    color_palette : tuple[str, ...]
        CSS colors for span highlighting (light backgrounds).
    dark_color_palette : tuple[str, ...]
        CSS colors for subscript label badges (dark, index-aligned with
        ``color_palette``).
    show_labels : bool
        Show span labels inline.
    show_tooltips : bool
        Show tooltips on hover.
    token_delimiter : str
        Delimiter between tokens in the display.
    label_position : Literal["inline", "below", "tooltip"]
        Where span labels are placed.
    """

    highlight_style: Literal["background", "underline", "border"] = "background"
    color_palette: tuple[str, ...] = dx.field(
        default_factory=_default_span_color_palette
    )
    dark_color_palette: tuple[str, ...] = dx.field(
        default_factory=_default_span_dark_palette
    )
    show_labels: bool = True
    show_tooltips: bool = True
    token_delimiter: str = " "
    label_position: Literal["inline", "below", "tooltip"] = "inline"


class DemographicsFieldConfig(dx.Model):
    """Configuration for a single demographics form field.

    Attributes
    ----------
    name : str
        Field name (used as key in collected data).
    field_type : Literal["text", "number", "dropdown", "radio", "checkbox"]
        Type of form input.
    label : str
        Display label.
    required : bool
        Whether the field is required.
    options : tuple[str, ...] | None
        Options for dropdown / radio fields.
    range : Range[float] | None
        Numeric range constraint for number fields.
    placeholder : str | None
        Placeholder text.
    help_text : str | None
        Help text displayed below the field.
    """

    name: str
    field_type: Literal["text", "number", "dropdown", "radio", "checkbox"]
    label: str
    required: bool = False
    options: tuple[str, ...] | None = None
    range: dx.Embed[Range[float]] | None = None
    placeholder: str | None = None
    help_text: str | None = None


class DemographicsConfig(dx.Model):
    """Configuration for the participant demographics form.

    Attributes
    ----------
    enabled : bool
        Show the demographics form.
    title : str
        Form title.
    fields : tuple[DemographicsFieldConfig, ...]
        Fields to include in the form.
    submit_button_text : str
        Submit button label.
    """

    enabled: bool = False
    title: str = "Participant Information"
    fields: tuple[dx.Embed[DemographicsFieldConfig], ...] = ()
    submit_button_text: str = "Continue"


class InstructionPage(dx.Model):
    """A single instruction page.

    Attributes
    ----------
    content : str
        HTML content.
    title : str | None
        Optional page title.
    """

    content: str
    title: str | None = None


class InstructionsConfig(dx.Model):
    """Configuration for multi-page experiment instructions.

    Attributes
    ----------
    pages : tuple[InstructionPage, ...]
        Instruction pages.
    show_page_numbers : bool
        Show page numbers.
    allow_backwards : bool
        Allow navigation to previous pages.
    button_label_next : str
        Label for the next button.
    button_label_finish : str
        Label for the final button.
    """

    pages: tuple[dx.Embed[InstructionPage], ...] = ()
    show_page_numbers: bool = True
    allow_backwards: bool = True
    button_label_next: str = "Next"
    button_label_finish: str = "Begin Experiment"

    @classmethod
    def from_text(cls, text: str) -> InstructionsConfig:
        """Build a single-page instructions config from plain text or HTML."""
        return cls(pages=(InstructionPage(content=text),))


def _default_slopit_integration() -> SlopitIntegrationConfig:
    return SlopitIntegrationConfig()


class ExperimentConfig(dx.Model):
    """Configuration for jsPsych experiment generation.

    Attributes
    ----------
    experiment_type : ExperimentType
        Type of experiment.
    title : str
        Experiment title.
    description : str
        Brief description.
    instructions : InstructionsConfig
        Instructions shown to participants. Use
        ``InstructionsConfig.from_text("...")`` for a single-page
        instruction set built from a plain string.
    distribution_strategy : ListDistributionStrategy
        List distribution strategy for batch mode.
    demographics : DemographicsConfig | None
        Demographics form shown before instructions.
    randomize_trial_order : bool
        Randomize trial order.
    show_progress_bar : bool
        Show a progress bar.
    ui_theme : UITheme
        UI theme.
    on_finish_url : str | None
        URL to redirect to after completion.
    allow_backwards : bool
        Allow navigation to previous trials.
    show_click_target : bool
        Show click target for accuracy tracking.
    minimum_duration_ms : int
        Minimum trial duration in milliseconds (>= 0).
    use_jatos : bool
        Enable JATOS integration.
    prolific_completion_code : str | None
        Prolific completion code; when set, ``on_finish_url`` is
        auto-generated.
    slopit : SlopitIntegrationConfig
        Slopit behavioral capture integration.
    span_display : SpanDisplayConfig | None
        Span display configuration; auto-enabled when items contain span
        annotations.
    """

    experiment_type: ExperimentType
    title: str
    description: str
    instructions: dx.Embed[InstructionsConfig]
    distribution_strategy: dx.Embed[ListDistributionStrategy]
    demographics: dx.Embed[DemographicsConfig] | None = None
    randomize_trial_order: bool = True
    show_progress_bar: bool = True
    ui_theme: UITheme = "light"
    on_finish_url: str | None = None
    allow_backwards: bool = False
    show_click_target: bool = False
    minimum_duration_ms: int = 0
    use_jatos: bool = True
    prolific_completion_code: str | None = None
    slopit: dx.Embed[SlopitIntegrationConfig] = dx.field(
        default_factory=_default_slopit_integration
    )
    span_display: dx.Embed[SpanDisplayConfig] | None = None


def _default_likert_scale() -> Range[int]:
    return Range[int](min=1, max=7)


class RatingScaleConfig(dx.Model):
    """Configuration for rating-scale trials.

    Attributes
    ----------
    scale : Range[int]
        Numeric range for the rating scale.
    min_label : str
        Label for the minimum value.
    max_label : str
        Label for the maximum value.
    step : int
        Step size between values (>= 1).
    show_numeric_labels : bool
        Show numeric labels on the scale.
    required : bool
        Whether a response is required.
    """

    scale: dx.Embed[Range[int]] = dx.field(default_factory=_default_likert_scale)
    min_label: str = "Not at all"
    max_label: str = "Very much"
    step: int = 1
    show_numeric_labels: bool = True
    required: bool = True


class ChoiceConfig(dx.Model):
    """Configuration for choice trials.

    Attributes
    ----------
    button_html : str | None
        Custom HTML for choice buttons.
    required : bool
        Whether a response is required.
    randomize_choice_order : bool
        Randomize the order of choices.
    layout : Literal["horizontal", "vertical"]
        Button layout.
    """

    button_html: str | None = None
    required: bool = True
    randomize_choice_order: bool = False
    layout: Literal["horizontal", "vertical"] = "horizontal"
