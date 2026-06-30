"""Data models for experimental item templates."""

from __future__ import annotations

from typing import Literal, Self
from uuid import UUID

import didactic.api as dx

from bead.data.base import BeadBaseModel
from bead.items.spans import SpanSpec
from bead.tokenization.config import TokenizerConfig

type MetadataValue = (
    str
    | int
    | float
    | bool
    | None
    | tuple[MetadataValue, ...]
    | dict[str, MetadataValue]
)

JudgmentType = Literal[
    "acceptability",
    "inference",
    "similarity",
    "plausibility",
    "comprehension",
    "preference",
    "extraction",
]

TaskType = Literal[
    "forced_choice",
    "multi_select",
    "ordinal_scale",
    "magnitude",
    "binary",
    "categorical",
    "free_text",
    "cloze",
    "span_labeling",
]

ElementRefType = Literal["text", "filled_template_ref"]
PresentationMode = Literal["static", "self_paced", "timed_sequence"]
ChunkingUnit = Literal["character", "word", "sentence", "constituent", "custom"]
ParseType = Literal["constituency", "dependency"]


class ChunkingSpec(BeadBaseModel):
    """Specification for text segmentation in incremental presentation.

    Attributes
    ----------
    unit : ChunkingUnit
        Segmentation unit type. Defaults to ``"word"``.
    parse_type : ParseType | None
        Type of parsing for constituent chunking.
    constituent_labels : tuple[str, ...] | None
        Labels for constituent chunking.
    parser : Literal["stanza", "spacy"] | None
        Parser library for constituent chunking.
    parse_language : str | None
        ISO 639 language code.
    custom_boundaries : tuple[int, ...] | None
        Token indices for custom chunking boundaries.
    """

    unit: ChunkingUnit = "word"
    parse_type: ParseType | None = None
    constituent_labels: tuple[str, ...] | None = None
    parser: Literal["stanza", "spacy"] | None = None
    parse_language: str | None = None
    custom_boundaries: tuple[int, ...] | None = None


class TimingParams(BeadBaseModel):
    """Timing parameters for stimulus presentation.

    Attributes
    ----------
    duration_ms : int | None
        Per-chunk display duration (ms).
    isi_ms : int | None
        Inter-stimulus interval (ms).
    timeout_ms : int | None
        Response timeout (ms).
    mask_char : str | None
        Character used to mask non-current chunks.
    cumulative : bool
        Show all previous chunks (``True``) or only the current chunk.
    """

    duration_ms: int | None = None
    isi_ms: int | None = None
    timeout_ms: int | None = None
    mask_char: str | None = None
    cumulative: bool = True


class ScaleBounds(BeadBaseModel):
    """Inclusive integer bounds for an ordinal scale.

    Attributes
    ----------
    min : int
        Minimum scale value (inclusive).
    max : int
        Maximum scale value (inclusive).
    """

    min: int
    max: int


class ScalePointLabel(BeadBaseModel):
    """A label attached to a specific point on an ordinal scale.

    Attributes
    ----------
    point : int
        Scale point value.
    label : str
        Human-readable label for that point.
    """

    point: int
    label: str


class TaskSpec(BeadBaseModel):
    """Parameters for the response collection task.

    Attributes
    ----------
    prompt : str
        Question or instruction shown to participants.
    scale_bounds : ScaleBounds | None
        Min and max values for ordinal_scale task.
    scale_labels : tuple[ScalePointLabel, ...]
        Labels for individual scale points.
    options : tuple[str, ...] | None
        Available options.
    min_selections : int | None
        Minimum number of selections (multi_select).
    max_selections : int | None
        Maximum number of selections (multi_select).
    text_validation_pattern : str | None
        Regex for free-text validation.
    max_length : int | None
        Maximum length for free-text responses.
    span_spec : SpanSpec | None
        Span labeling specification.
    """

    prompt: str
    scale_bounds: dx.Embed[ScaleBounds] | None = None
    scale_labels: tuple[dx.Embed[ScalePointLabel], ...] = ()
    options: tuple[str, ...] | None = None
    min_selections: int | None = None
    max_selections: int | None = None
    text_validation_pattern: str | None = None
    max_length: int | None = None
    span_spec: dx.Embed[SpanSpec] | None = None

    @dx.validates("prompt")
    def _check_prompt(self, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Prompt cannot be empty")
        return value.strip()


def _default_chunking() -> ChunkingSpec:
    return ChunkingSpec()


def _default_timing() -> TimingParams:
    return TimingParams()


class PresentationSpec(BeadBaseModel):
    """Specification of stimulus presentation method.

    Attributes
    ----------
    mode : PresentationMode
        Presentation mode.
    chunking : ChunkingSpec
        Chunking specification.
    timing : TimingParams
        Timing parameters.
    display_format : dict[str, str | int | float | bool]
        Additional display formatting options.
    tokenizer_config : TokenizerConfig | None
        Display tokenizer configuration for span annotation.
    """

    mode: PresentationMode = "static"
    chunking: dx.Embed[ChunkingSpec] = dx.field(default_factory=_default_chunking)
    timing: dx.Embed[TimingParams] = dx.field(default_factory=_default_timing)
    display_format: dict[str, str | int | float | bool] = dx.field(default_factory=dict)
    tokenizer_config: dx.Embed[TokenizerConfig] | None = None


class ItemElement(BeadBaseModel):
    """A structured element within an item template.

    Attributes
    ----------
    element_type : ElementRefType
        Type of element.
    element_name : str
        Unique name within the item.
    content : str | None
        Static text content (for text elements).
    filled_template_ref_id : UUID | None
        UUID of filled template (for reference elements).
    element_metadata : dict[str, MetadataValue]
        Additional metadata.
    order : int | None
        Display order.
    """

    element_type: ElementRefType
    element_name: str
    content: str | None = None
    filled_template_ref_id: UUID | None = None
    element_metadata: dict[str, MetadataValue] = dx.field(default_factory=dict)
    order: int | None = None

    @dx.validates("element_name")
    def _check_element_name(self, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Element name cannot be empty")
        return value.strip()

    @property
    def is_text(self) -> bool:
        """Return whether this is a text element."""
        return self.element_type == "text"

    @property
    def is_template_ref(self) -> bool:
        """Return whether this references a filled template."""
        return self.element_type == "filled_template_ref"


class ItemTemplate(BeadBaseModel):
    """Template specification for constructing experimental items.

    Attributes
    ----------
    name : str
        Template name.
    description : str | None
        Human-readable description.
    judgment_type : JudgmentType
        Semantic property being measured.
    task_type : TaskType
        Response collection method.
    elements : tuple[ItemElement, ...]
        Elements that compose this item.
    constraints : tuple[UUID, ...]
        UUIDs of constraints on items.
    task_spec : TaskSpec
        Task-specific parameters.
    presentation_spec : PresentationSpec
        Presentation specification.
    presentation_order : tuple[str, ...] | None
        Order to present elements (by element_name).
    template_metadata : dict[str, MetadataValue]
        Additional metadata.
    """

    name: str
    judgment_type: JudgmentType
    task_type: TaskType
    task_spec: dx.Embed[TaskSpec]
    presentation_spec: dx.Embed[PresentationSpec]
    description: str | None = None
    elements: tuple[dx.Embed[ItemElement], ...] = ()
    constraints: tuple[UUID, ...] = ()
    presentation_order: tuple[str, ...] | None = None
    template_metadata: dict[str, MetadataValue] = dx.field(default_factory=dict)

    @dx.validates("name")
    def _check_name(self, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Template name cannot be empty")
        return value.strip()

    @dx.validates("elements")
    def _check_unique_element_names(
        self, value: tuple[ItemElement, ...]
    ) -> tuple[ItemElement, ...]:
        if not value:
            return value
        names = [elem.element_name for elem in value]
        if len(names) != len(set(names)):
            duplicates = [name for name in names if names.count(name) > 1]
            raise ValueError(f"Duplicate element names: {set(duplicates)}")
        return value

    def get_element_by_name(self, name: str) -> ItemElement | None:
        """Return the element with the given *name*, or ``None``."""
        for elem in self.elements:
            if elem.element_name == name:
                return elem
        return None

    def get_template_ref_elements(self) -> tuple[ItemElement, ...]:
        """Return every element that references a filled template."""
        return tuple(elem for elem in self.elements if elem.is_template_ref)


class ItemTemplateCollection(BeadBaseModel):
    """A collection of item templates.

    Attributes
    ----------
    name : str
        Name of this collection.
    description : str | None
        Description.
    templates : tuple[ItemTemplate, ...]
        Item templates in this collection.
    """

    name: str
    description: str | None = None
    templates: tuple[dx.Embed[ItemTemplate], ...] = ()

    @dx.validates("name")
    def _check_name(self, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Collection name cannot be empty")
        return value.strip()

    def with_template(self, template: ItemTemplate) -> Self:
        """Return a new collection with *template* appended."""
        return self.with_(templates=(*self.templates, template)).touched()
