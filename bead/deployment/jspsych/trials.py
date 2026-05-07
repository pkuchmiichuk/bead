"""Trial generators for jsPsych experiments.

This module provides functions to generate jsPsych trial objects from
Item models. It supports various trial types including rating scales,
forced choice, binary choice, and span labeling trials. Composite tasks
(e.g., rating with span highlights) are also supported.
"""

from __future__ import annotations

from dataclasses import dataclass

from bead.data.base import JsonValue
from bead.deployment.jspsych.config import (
    ChoiceConfig,
    DemographicsConfig,
    DemographicsFieldConfig,
    ExperimentConfig,
    InstructionsConfig,
    RatingScaleConfig,
    SpanDisplayConfig,
)
from bead.items.item import Item
from bead.items.item_template import ItemTemplate
from bead.items.spans import Span
from bead.labels import parse_label_refs
from bead.transforms.base import TransformContext, TransformRegistry


def _serialize_item_metadata(
    item: Item, template: ItemTemplate
) -> dict[str, JsonValue]:
    """Serialize complete item and template metadata for trial data.

    Parameters
    ----------
    item : Item
        The item to serialize metadata from.
    template : ItemTemplate
        The item template to serialize metadata from.

    Returns
    -------
    dict[str, JsonValue]
        Metadata dictionary containing all item and template fields.
    """
    return {
        # item identification
        "item_id": str(item.id),
        "item_created": item.created_at.isoformat(),
        "item_modified": item.modified_at.isoformat(),
        # item template reference
        "item_template_id": str(item.item_template_id),
        # filled template references
        "filled_template_refs": [str(ref) for ref in item.filled_template_refs],
        # options (for forced_choice/multi_select)
        "options": list(item.options),
        # rendered elements
        "rendered_elements": dict(item.rendered_elements),
        # unfilled slots (for cloze tasks)
        "unfilled_slots": [
            {
                "slot_name": slot.slot_name,
                "position": slot.position,
                "constraint_ids": [str(cid) for cid in slot.constraint_ids],
            }
            for slot in item.unfilled_slots
        ],
        # model outputs
        "model_outputs": [
            {
                "model_name": output.model_name,
                "model_version": output.model_version,
                "operation": output.operation,
                "inputs": output.inputs,
                "output": output.output,
                "cache_key": output.cache_key,
                "computation_metadata": output.computation_metadata,
            }
            for output in item.model_outputs
        ],
        # constraint satisfaction
        "constraint_satisfaction": {
            str(cs.constraint_id): cs.satisfied for cs in item.constraint_satisfaction
        },
        # item-specific metadata
        "item_metadata": dict(item.item_metadata),
        # template information
        "template_name": template.name,
        "template_description": template.description,
        "judgment_type": template.judgment_type,
        "task_type": template.task_type,
        # template elements
        "template_elements": [
            {
                "element_type": elem.element_type,
                "element_name": elem.element_name,
                "content": elem.content,
                "filled_template_ref_id": (
                    str(elem.filled_template_ref_id)
                    if elem.filled_template_ref_id
                    else None
                ),
                "element_metadata": elem.element_metadata,
                "order": elem.order,
            }
            for elem in template.elements
        ],
        # template constraints
        "template_constraints": [str(c) for c in template.constraints],
        # task specification
        "task_spec": {
            "prompt": template.task_spec.prompt,
            "scale_bounds": (
                [
                    template.task_spec.scale_bounds.min,
                    template.task_spec.scale_bounds.max,
                ]
                if template.task_spec.scale_bounds is not None
                else None
            ),
            "scale_labels": {
                str(label.point): label.label
                for label in template.task_spec.scale_labels
            },
            "options": list(template.task_spec.options or ()),
            "min_selections": template.task_spec.min_selections,
            "max_selections": template.task_spec.max_selections,
            "text_validation_pattern": template.task_spec.text_validation_pattern,
            "max_length": template.task_spec.max_length,
        },
        # presentation specification
        "presentation_spec": {
            "mode": template.presentation_spec.mode,
            "chunking": (
                {
                    "unit": template.presentation_spec.chunking.unit,
                    "parse_type": (template.presentation_spec.chunking.parse_type),
                    "constituent_labels": (
                        template.presentation_spec.chunking.constituent_labels
                    ),
                    "parser": template.presentation_spec.chunking.parser,
                    "parse_language": (
                        template.presentation_spec.chunking.parse_language
                    ),
                    "custom_boundaries": (
                        template.presentation_spec.chunking.custom_boundaries
                    ),
                }
                if template.presentation_spec.chunking
                else None
            ),
            "timing": (
                {
                    "duration_ms": template.presentation_spec.timing.duration_ms,
                    "isi_ms": template.presentation_spec.timing.isi_ms,
                    "timeout_ms": template.presentation_spec.timing.timeout_ms,
                    "mask_char": template.presentation_spec.timing.mask_char,
                    "cumulative": template.presentation_spec.timing.cumulative,
                }
                if template.presentation_spec.timing
                else None
            ),
            "display_format": template.presentation_spec.display_format,
        },
        # presentation order
        "presentation_order": template.presentation_order,
        # template metadata
        "template_metadata": dict(template.template_metadata),
        # span annotation data
        "spans": [
            {
                "span_id": span.span_id,
                "segments": [
                    {
                        "element_name": seg.element_name,
                        "indices": seg.indices,
                    }
                    for seg in span.segments
                ],
                "head_index": span.head_index,
                "label": (
                    {
                        "label": span.label.label,
                        "label_id": span.label.label_id,
                        "confidence": span.label.confidence,
                    }
                    if span.label
                    else None
                ),
                "span_type": span.span_type,
                "span_metadata": dict(span.span_metadata),
            }
            for span in item.spans
        ],
        "span_relations": [
            {
                "relation_id": rel.relation_id,
                "source_span_id": rel.source_span_id,
                "target_span_id": rel.target_span_id,
                "label": (
                    {
                        "label": rel.label.label,
                        "label_id": rel.label.label_id,
                        "confidence": rel.label.confidence,
                    }
                    if rel.label
                    else None
                ),
                "directed": rel.directed,
                "relation_metadata": dict(rel.relation_metadata),
            }
            for rel in item.span_relations
        ],
        "tokenized_elements": dict(item.tokenized_elements),
        "token_space_after": {k: list(v) for k, v in item.token_space_after.items()},
        "span_spec": (
            {
                "index_mode": template.task_spec.span_spec.index_mode,
                "interaction_mode": template.task_spec.span_spec.interaction_mode,
                "label_source": template.task_spec.span_spec.label_source,
                "labels": template.task_spec.span_spec.labels,
                "label_colors": template.task_spec.span_spec.label_colors,
                "allow_overlapping": template.task_spec.span_spec.allow_overlapping,
                "min_spans": template.task_spec.span_spec.min_spans,
                "max_spans": template.task_spec.span_spec.max_spans,
                "enable_relations": template.task_spec.span_spec.enable_relations,
                "relation_label_source": (
                    template.task_spec.span_spec.relation_label_source
                ),
                "relation_labels": template.task_spec.span_spec.relation_labels,
                "relation_directed": template.task_spec.span_spec.relation_directed,
                "min_relations": template.task_spec.span_spec.min_relations,
                "max_relations": template.task_spec.span_spec.max_relations,
                "wikidata_language": template.task_spec.span_spec.wikidata_language,
                "wikidata_entity_types": (
                    template.task_spec.span_spec.wikidata_entity_types
                ),
                "wikidata_result_limit": (
                    template.task_spec.span_spec.wikidata_result_limit
                ),
            }
            if template.task_spec.span_spec
            else None
        ),
    }


def create_trial(
    item: Item,
    template: ItemTemplate,
    experiment_config: ExperimentConfig,
    trial_number: int,
    rating_config: RatingScaleConfig | None = None,
    choice_config: ChoiceConfig | None = None,
) -> dict[str, JsonValue]:
    """Create a jsPsych trial object from an Item.

    Parameters
    ----------
    item : Item
        The item to create a trial from.
    template : ItemTemplate
        The item template for this item.
    experiment_config : ExperimentConfig
        The experiment configuration.
    trial_number : int
        The trial number (for tracking).
    rating_config : RatingScaleConfig | None
        Configuration for rating scale trials (required for rating types).
    choice_config : ChoiceConfig | None
        Configuration for choice trials (required for choice types).

    Returns
    -------
    dict[str, JsonValue]
        A jsPsych trial object with item and template metadata.

    Raises
    ------
    ValueError
        If required configuration is missing for the experiment type.

    Examples
    --------
    >>> from uuid import UUID
    >>> from bead.items.item_template import TaskSpec, PresentationSpec
    >>> item = Item(
    ...     item_template_id=UUID("12345678-1234-5678-1234-567812345678"),
    ...     rendered_elements={"sentence": "The cat broke the vase"}
    ... )
    >>> template = ItemTemplate(
    ...     name="test",
    ...     judgment_type="acceptability",
    ...     task_type="ordinal_scale",
    ...     task_spec=TaskSpec(prompt="Rate this"),
    ...     presentation_spec=PresentationSpec(mode="static")
    ... )
    >>> config = ExperimentConfig(
    ...     experiment_type="likert_rating",
    ...     title="Test",
    ...     description="Test",
    ...     instructions="Test"
    ... )
    >>> rating_config = RatingScaleConfig()
    >>> trial = create_trial(item, template, config, 0, rating_config=rating_config)
    >>> trial["type"]
    'bead-slider-rating'
    """
    # standalone span_labeling experiment type
    if experiment_config.experiment_type == "span_labeling":
        span_display = experiment_config.span_display or SpanDisplayConfig()
        return _create_span_labeling_trial(item, template, span_display, trial_number)

    # for composite tasks: detect spans and use span-enhanced stimulus HTML
    has_spans = bool(item.spans) and bool(
        template.task_spec.span_spec if template.task_spec else False
    )

    # resolve span display config for composite tasks with spans
    span_display = experiment_config.span_display or SpanDisplayConfig()

    if experiment_config.experiment_type == "likert_rating":
        if rating_config is None:
            raise ValueError("rating_config required for likert_rating experiments")
        return _create_likert_trial(
            item,
            template,
            rating_config,
            trial_number,
            has_spans=has_spans,
            span_display=span_display,
        )
    elif experiment_config.experiment_type == "slider_rating":
        if rating_config is None:
            raise ValueError("rating_config required for slider_rating experiments")
        return _create_slider_trial(
            item,
            template,
            rating_config,
            trial_number,
            has_spans=has_spans,
            span_display=span_display,
        )
    elif experiment_config.experiment_type == "binary_choice":
        if choice_config is None:
            raise ValueError("choice_config required for binary_choice experiments")
        return _create_binary_choice_trial(
            item,
            template,
            choice_config,
            trial_number,
            has_spans=has_spans,
            span_display=span_display,
        )
    elif experiment_config.experiment_type == "forced_choice":
        if choice_config is None:
            raise ValueError("choice_config required for forced_choice experiments")
        return _create_forced_choice_trial(
            item,
            template,
            choice_config,
            trial_number,
            has_spans=has_spans,
            span_display=span_display,
        )
    else:
        raise ValueError(
            f"Unknown experiment type: {experiment_config.experiment_type}"
        )


def _create_likert_trial(
    item: Item,
    template: ItemTemplate,
    config: RatingScaleConfig,
    trial_number: int,
    has_spans: bool = False,
    span_display: SpanDisplayConfig | None = None,
) -> dict[str, JsonValue]:
    """Create a Likert rating trial.

    Parameters
    ----------
    item : Item
        The item to create a trial from.
    template : ItemTemplate
        The item template.
    config : RatingScaleConfig
        Rating scale configuration.
    trial_number : int
        The trial number.
    has_spans : bool
        Whether the item has span annotations.
    span_display : SpanDisplayConfig | None
        Span display configuration.

    Returns
    -------
    dict[str, JsonValue]
        A jsPsych bead-rating trial object.
    """
    # generate stimulus HTML from rendered elements
    if has_spans and span_display:
        stimulus_html = _generate_span_stimulus_html(item, span_display)
    else:
        stimulus_html = _generate_stimulus_html(item)

    # build scale labels dict for endpoint labels
    # keys are stringified ints (JSON object keys are always strings)
    scale_labels: dict[str, JsonValue] = {}
    if config.min_label:
        scale_labels[str(config.scale.min)] = config.min_label
    if config.max_label:
        scale_labels[str(config.scale.max)] = config.max_label

    # build prompt: stimulus HTML + task prompt if available
    prompt = stimulus_html
    if template.task_spec and template.task_spec.prompt:
        task_prompt = template.task_spec.prompt
        if has_spans and span_display:
            color_map = _assign_span_colors(item.spans, span_display)
            task_prompt = _resolve_prompt_references(task_prompt, item, color_map)
        prompt += f'<p class="bead-task-prompt">{task_prompt}</p>'

    # serialize complete metadata
    metadata = _serialize_item_metadata(item, template)
    metadata["trial_number"] = trial_number
    metadata["trial_type"] = "likert_rating"

    return {
        "type": "bead-rating",
        "prompt": prompt,
        "scale_min": config.scale.min,
        "scale_max": config.scale.max,
        "scale_labels": scale_labels,
        "require_response": config.required,
        "button_label": "Continue",
        "metadata": metadata,
    }


def _create_slider_trial(
    item: Item,
    template: ItemTemplate,
    config: RatingScaleConfig,
    trial_number: int,
    has_spans: bool = False,
    span_display: SpanDisplayConfig | None = None,
) -> dict[str, JsonValue]:
    """Create a slider rating trial.

    Parameters
    ----------
    item : Item
        The item to create a trial from.
    template : ItemTemplate
        The item template.
    config : RatingScaleConfig
        Rating scale configuration.
    trial_number : int
        The trial number.
    has_spans : bool
        Whether the item has span annotations.
    span_display : SpanDisplayConfig | None
        Span display configuration.

    Returns
    -------
    dict[str, JsonValue]
        A jsPsych bead-slider-rating trial object.
    """
    if has_spans and span_display:
        stimulus_html = _generate_span_stimulus_html(item, span_display)
    else:
        stimulus_html = _generate_stimulus_html(item)

    # build prompt: stimulus HTML + resolved task prompt
    prompt_html = stimulus_html
    if template.task_spec and template.task_spec.prompt:
        task_prompt = template.task_spec.prompt
        if has_spans and span_display:
            color_map = _assign_span_colors(item.spans, span_display)
            task_prompt = _resolve_prompt_references(task_prompt, item, color_map)
        prompt_html += f'<p class="bead-task-prompt">{task_prompt}</p>'

    # serialize complete metadata
    metadata = _serialize_item_metadata(item, template)
    metadata["trial_number"] = trial_number
    metadata["trial_type"] = "slider_rating"

    return {
        "type": "bead-slider-rating",
        "prompt": prompt_html,
        "labels": [config.min_label, config.max_label],
        "slider_min": config.scale.min,
        "slider_max": config.scale.max,
        "step": config.step,
        "slider_start": (config.scale.min + config.scale.max) // 2,
        "require_movement": config.required,
        "button_label": "Continue",
        "metadata": metadata,
    }


def _create_binary_choice_trial(
    item: Item,
    template: ItemTemplate,
    config: ChoiceConfig,
    trial_number: int,
    has_spans: bool = False,
    span_display: SpanDisplayConfig | None = None,
) -> dict[str, JsonValue]:
    """Create a binary choice trial.

    Parameters
    ----------
    item : Item
        The item to create a trial from.
    template : ItemTemplate
        The item template.
    config : ChoiceConfig
        Choice configuration.
    trial_number : int
        The trial number.
    has_spans : bool
        Whether the item has span annotations.
    span_display : SpanDisplayConfig | None
        Span display configuration.

    Returns
    -------
    dict[str, JsonValue]
        A jsPsych bead-binary-choice trial object.
    """
    if has_spans and span_display:
        stimulus_html = _generate_span_stimulus_html(item, span_display)
    else:
        stimulus_html = _generate_stimulus_html(item)

    # serialize complete metadata
    metadata = _serialize_item_metadata(item, template)
    metadata["trial_number"] = trial_number
    metadata["trial_type"] = "binary_choice"

    prompt = (
        template.task_spec.prompt
        if template.task_spec
        else "Is this sentence acceptable?"
    )

    if has_spans and span_display:
        color_map = _assign_span_colors(item.spans, span_display)
        prompt = _resolve_prompt_references(prompt, item, color_map)

    return {
        "type": "bead-binary-choice",
        "prompt": prompt,
        "stimulus": stimulus_html,
        "choices": ["Yes", "No"],
        "require_response": config.required,
        "metadata": metadata,
    }


def _create_forced_choice_trial(
    item: Item,
    template: ItemTemplate,
    config: ChoiceConfig,
    trial_number: int,
    has_spans: bool = False,
    span_display: SpanDisplayConfig | None = None,
) -> dict[str, JsonValue]:
    """Create a forced choice trial.

    Parameters
    ----------
    item : Item
        The item to create a trial from. Must have at least 2 options in
        the item.options list.
    template : ItemTemplate
        The item template.
    config : ChoiceConfig
        Choice configuration.
    trial_number : int
        The trial number.
    has_spans : bool
        Whether the item has span annotations.
    span_display : SpanDisplayConfig | None
        Span display configuration.

    Returns
    -------
    dict[str, JsonValue]
        A jsPsych bead-forced-choice trial object.

    Raises
    ------
    ValueError
        If item.options is empty or has fewer than 2 options.
    """
    prompt = (
        template.task_spec.prompt
        if template.task_spec
        else "Which option do you choose?"
    )

    # extract alternatives from item.options (single source of truth)
    if not item.options:
        raise ValueError(
            f"Item {item.id} has no options. "
            f"Forced choice items must have at least 2 options in item.options. "
            f"Use create_forced_choice_item() to create items with options."
        )
    if len(item.options) < 2:
        raise ValueError(
            f"Item {item.id} has only {len(item.options)} option(s). "
            f"Forced choice items require at least 2 options."
        )

    # for composite span tasks, render span-highlighted HTML into each alternative
    alternatives: list[str] = list(item.options)
    if has_spans and span_display:
        color_map = _assign_span_colors(item.spans, span_display)
        prompt = _resolve_prompt_references(prompt, item, color_map)
        stimulus_html = _generate_span_stimulus_html(item, span_display)
        prompt = stimulus_html + f"<p>{prompt}</p>"

    # serialize complete metadata
    metadata = _serialize_item_metadata(item, template)
    metadata["trial_number"] = trial_number
    metadata["trial_type"] = "forced_choice"

    return {
        "type": "bead-forced-choice",
        "prompt": prompt,
        "alternatives": alternatives,
        "layout": config.layout,
        "randomize_position": config.randomize_choice_order,
        "enable_keyboard": True,
        "require_response": config.required,
        "button_label": "Select",
        "metadata": metadata,
    }


def _generate_stimulus_html(item: Item, include_all: bool = True) -> str:
    """Generate HTML for stimulus presentation.

    Parameters
    ----------
    item : Item
        The item to generate HTML for.
    include_all : bool
        Whether to include all rendered elements (True) or just the first one (False).

    Returns
    -------
    str
        HTML string for the stimulus.
    """
    if not item.rendered_elements:
        return "<p>No stimulus available</p>"

    # get rendered elements in a consistent order
    sorted_keys = sorted(item.rendered_elements.keys())

    if include_all:
        # include all rendered elements
        elements = [
            f'<div class="stimulus-element"><p>{item.rendered_elements[k]}</p></div>'
            for k in sorted_keys
        ]
        return '<div class="stimulus-container">' + "".join(elements) + "</div>"
    else:
        # include only the first element (for forced choice where others are options)
        first_key = sorted_keys[0]
        element_html = item.rendered_elements[first_key]
        return f'<div class="stimulus-container"><p>{element_html}</p></div>'


def create_consent_trial(consent_text: str) -> dict[str, JsonValue]:
    """Create a consent trial.

    Parameters
    ----------
    consent_text : str
        The consent text to display.

    Returns
    -------
    dict[str, JsonValue]
        A jsPsych html-button-response trial object.
    """
    stimulus_html = (
        f'<div class="consent"><h2>Consent</h2><div>{consent_text}</div></div>'
    )

    return {
        "type": "html-button-response",
        "stimulus": stimulus_html,
        "choices": ["I agree", "I do not agree"],
        "data": {
            "trial_type": "consent",
        },
    }


def create_completion_trial(
    completion_message: str = "Thank you for participating!",
) -> dict[str, JsonValue]:
    """Create a completion trial.

    Parameters
    ----------
    completion_message : str
        The completion message to display.

    Returns
    -------
    dict[str, JsonValue]
        A jsPsych html-keyboard-response trial object.
    """
    stimulus_html = (
        f'<div class="completion"><h2>Complete</h2><p>{completion_message}</p></div>'
    )

    return {
        "type": "html-keyboard-response",
        "stimulus": stimulus_html,
        "choices": "NO_KEYS",
        "data": {
            "trial_type": "completion",
        },
    }


def _create_survey_question(field: DemographicsFieldConfig) -> dict[str, JsonValue]:
    """Create a jsPsych survey question from a demographics field config.

    Parameters
    ----------
    field : DemographicsFieldConfig
        The field configuration.

    Returns
    -------
    dict[str, JsonValue]
        A jsPsych survey question object.
    """
    question: dict[str, JsonValue] = {
        "name": field.name,
        "prompt": field.label,
        "required": field.required,
    }

    if field.field_type == "text":
        question["type"] = "text"
        if field.placeholder:
            question["placeholder"] = field.placeholder

    elif field.field_type == "number":
        question["type"] = "text"
        question["input_type"] = "number"
        if field.placeholder:
            question["placeholder"] = field.placeholder
        if field.range is not None:
            question["min"] = field.range.min
            question["max"] = field.range.max

    elif field.field_type == "dropdown":
        question["type"] = "drop-down"
        if field.options:
            question["options"] = field.options

    elif field.field_type == "radio":
        question["type"] = "multi-choice"
        if field.options:
            question["options"] = field.options

    elif field.field_type == "checkbox":
        question["type"] = "multi-select"
        if field.options:
            question["options"] = field.options

    return question


def create_demographics_trial(config: DemographicsConfig) -> dict[str, JsonValue]:
    """Create a demographics survey trial.

    Parameters
    ----------
    config : DemographicsConfig
        The demographics form configuration.

    Returns
    -------
    dict[str, JsonValue]
        A jsPsych survey trial object.

    Examples
    --------
    >>> from bead.deployment.jspsych.config import (
    ...     DemographicsConfig, DemographicsFieldConfig
    ... )
    >>> config = DemographicsConfig(
    ...     enabled=True,
    ...     title="About You",
    ...     fields=[
    ...         DemographicsFieldConfig(
    ...             name="age",
    ...             field_type="number",
    ...             label="Your Age",
    ...             required=True,
    ...         ),
    ...     ],
    ... )
    >>> trial = create_demographics_trial(config)
    >>> trial["type"]
    'survey'
    """
    questions = [_create_survey_question(field) for field in config.fields]

    return {
        "type": "survey",
        "title": config.title,
        "pages": [questions],
        "button_label_finish": config.submit_button_text,
        "data": {
            "trial_type": "demographics",
        },
    }


def create_instructions_trial(
    instructions: str | InstructionsConfig,
) -> dict[str, JsonValue]:
    """Create an instruction trial supporting both simple strings and rich config.

    Parameters
    ----------
    instructions : str | InstructionsConfig
        Either a simple instruction string (single page, keyboard response)
        or an InstructionsConfig for multi-page instructions.

    Returns
    -------
    dict[str, JsonValue]
        A jsPsych trial object. For simple strings, returns html-keyboard-response.
        For InstructionsConfig, returns an instructions plugin trial.

    Examples
    --------
    >>> # Simple string instructions
    >>> trial = create_instructions_trial("Rate each sentence from 1-7.")
    >>> trial["type"]
    'html-keyboard-response'

    >>> # Multi-page instructions
    >>> from bead.deployment.jspsych.config import InstructionsConfig, InstructionPage
    >>> config = InstructionsConfig(
    ...     pages=[
    ...         InstructionPage(title="Welcome", content="<p>Welcome!</p>"),
    ...         InstructionPage(title="Task", content="<p>Rate sentences.</p>"),
    ...     ],
    ... )
    >>> trial = create_instructions_trial(config)
    >>> trial["type"]
    'instructions'
    >>> len(trial["pages"])
    2
    """
    if isinstance(instructions, str):
        # simple string: use html-keyboard-response (backward compatible)
        stimulus_html = (
            f'<div class="instructions">'
            f"<h2>Instructions</h2>"
            f"<p>{instructions}</p>"
            f"<p><em>Press any key to continue</em></p>"
            f"</div>"
        )
        return {
            "type": "html-keyboard-response",
            "stimulus": stimulus_html,
            "data": {
                "trial_type": "instructions",
            },
        }

    # use jsPsych instructions plugin for InstructionsConfig (multi-page)
    pages: list[str] = []
    for i, page in enumerate(instructions.pages):
        page_html = '<div class="instructions-page">'
        if page.title:
            page_html += f"<h2>{page.title}</h2>"
        page_html += f"<div>{page.content}</div>"

        # add page numbers if enabled
        if instructions.show_page_numbers and len(instructions.pages) > 1:
            page_html += (
                f'<p class="page-number">Page {i + 1} of {len(instructions.pages)}</p>'
            )

        page_html += "</div>"
        pages.append(page_html)

    return {
        "type": "instructions",
        "pages": pages,
        "show_clickable_nav": True,
        "allow_backward": instructions.allow_backwards,
        "button_label_next": instructions.button_label_next,
        "button_label_previous": "Previous",
        "button_label_finish": instructions.button_label_finish,
        "data": {
            "trial_type": "instructions",
        },
    }


@dataclass(frozen=True)
class SpanColorMap:
    """Light and dark color assignments for spans.

    Attributes
    ----------
    light_by_span_id : dict[str, str]
        Light (background) colors keyed by span_id.
    dark_by_span_id : dict[str, str]
        Dark (badge) colors keyed by span_id.
    light_by_label : dict[str, str]
        Light (background) colors keyed by label name.
    dark_by_label : dict[str, str]
        Dark (badge) colors keyed by label name.
    """

    light_by_span_id: dict[str, str]
    dark_by_span_id: dict[str, str]
    light_by_label: dict[str, str]
    dark_by_label: dict[str, str]


def _assign_span_colors(
    spans: list[Span],
    span_display: SpanDisplayConfig,
) -> SpanColorMap:
    """Assign light and dark colors to spans.

    Same label gets the same color pair. Unlabeled spans each get
    their own color. Index-aligned light/dark palettes produce
    matching background and badge colors.

    Parameters
    ----------
    spans : list[Span]
        Spans to assign colors to.
    span_display : SpanDisplayConfig
        Display configuration with light and dark palettes.

    Returns
    -------
    SpanColorMap
        Color assignments keyed by span_id and by label.
    """
    light_palette = span_display.color_palette
    dark_palette = span_display.dark_color_palette

    light_by_label: dict[str, str] = {}
    dark_by_label: dict[str, str] = {}
    light_by_span_id: dict[str, str] = {}
    dark_by_span_id: dict[str, str] = {}
    color_idx = 0

    for span in spans:
        if span.label and span.label.label:
            label_name = span.label.label
            if label_name not in light_by_label:
                light_by_label[label_name] = light_palette[
                    color_idx % len(light_palette)
                ]
                dark_by_label[label_name] = dark_palette[color_idx % len(dark_palette)]
                color_idx += 1
            light_by_span_id[span.span_id] = light_by_label[label_name]
            dark_by_span_id[span.span_id] = dark_by_label[label_name]
        else:
            light_by_span_id[span.span_id] = light_palette[
                color_idx % len(light_palette)
            ]
            dark_by_span_id[span.span_id] = dark_palette[color_idx % len(dark_palette)]
            color_idx += 1

    return SpanColorMap(
        light_by_span_id=light_by_span_id,
        dark_by_span_id=dark_by_span_id,
        light_by_label=light_by_label,
        dark_by_label=dark_by_label,
    )


def _generate_span_stimulus_html(
    item: Item,
    span_display: SpanDisplayConfig,
) -> str:
    """Generate HTML with span-highlighted tokens for composite tasks.

    Renders tokens as individually wrapped ``<span>`` elements with
    highlight classes and data attributes for span identification.

    Parameters
    ----------
    item : Item
        Item with spans and tokenized_elements.
    span_display : SpanDisplayConfig
        Visual configuration.

    Returns
    -------
    str
        HTML string with span-highlighted token elements.
    """
    if not item.tokenized_elements:
        return _generate_stimulus_html(item)

    html_parts: list[str] = ['<div class="stimulus-container">']

    sorted_keys = sorted(item.tokenized_elements.keys())
    for element_name in sorted_keys:
        tokens = item.tokenized_elements[element_name]
        space_flags = item.token_space_after.get(element_name, [])

        # build token-to-span mapping
        token_spans: dict[int, list[str]] = {}
        for span in item.spans:
            for segment in span.segments:
                if segment.element_name == element_name:
                    for idx in segment.indices:
                        if idx not in token_spans:
                            token_spans[idx] = []
                        token_spans[idx].append(span.span_id)

        # assign colors (shared with prompt reference resolution)
        color_map = _assign_span_colors(item.spans, span_display)
        span_colors = color_map.light_by_span_id

        html_parts.append(
            f'<div class="stimulus-element bead-span-container" '
            f'data-element="{element_name}">'
        )

        for i, token_text in enumerate(tokens):
            span_ids = token_spans.get(i, [])
            n_spans = len(span_ids)

            classes = ["bead-token"]
            if n_spans > 0:
                classes.append("highlighted")

            fallback = span_display.color_palette[0]
            style_parts: list[str] = []
            if n_spans == 1:
                color = span_colors.get(span_ids[0], fallback)
                style_parts.append(f"background-color: {color}")
            elif n_spans > 1:
                # layer multiple spans
                colors = [span_colors.get(sid, fallback) for sid in span_ids]
                gradient = ", ".join(colors)
                style_parts.append(f"background: linear-gradient({gradient})")

            style_attr = f' style="{"; ".join(style_parts)}"' if style_parts else ""
            span_id_attr = f' data-span-ids="{",".join(span_ids)}"' if span_ids else ""
            count_attr = f' data-span-count="{n_spans}"' if n_spans > 0 else ""

            html_parts.append(
                f'<span class="{" ".join(classes)}" '
                f'data-index="{i}" data-element="{element_name}"'
                f"{count_attr}{span_id_attr}{style_attr}>"
                f"{token_text}</span>"
            )

            # add spacing
            if i < len(space_flags) and space_flags[i]:
                html_parts.append(" ")

        html_parts.append("</div>")

    html_parts.append("</div>")
    return "".join(html_parts)


# prompt span reference resolution


def _auto_fill_span_text(label: str, item: Item) -> str:
    """Reconstruct display text from a span's tokens.

    Finds the first span whose label matches, collects its token
    indices from the first segment's element, and joins them
    respecting ``token_space_after``.

    Parameters
    ----------
    label : str
        Span label to look up.
    item : Item
        Item with spans, tokenized_elements, and token_space_after.

    Returns
    -------
    str
        Reconstructed text from the span's tokens.

    Raises
    ------
    ValueError
        If no span with the given label exists or tokens are unavailable.
    """
    target_span: Span | None = None
    for span in item.spans:
        if span.label and span.label.label == label:
            target_span = span
            break

    if target_span is None:
        available = [s.label.label for s in item.spans if s.label and s.label.label]
        raise ValueError(
            f"Prompt references span label '{label}' but no span with "
            f"that label exists. Available labels: {available}"
        )

    parts: list[str] = []
    for segment in target_span.segments:
        element_name = segment.element_name
        tokens = item.tokenized_elements.get(element_name, [])
        space_flags = item.token_space_after.get(element_name, [])
        sorted_indices = sorted(segment.indices)
        for i, idx in enumerate(sorted_indices):
            if idx < len(tokens):
                parts.append(tokens[idx])
                if (
                    i < len(sorted_indices) - 1
                    and idx < len(space_flags)
                    and space_flags[idx]
                ):
                    parts.append(" ")

    return "".join(parts)


def _resolve_prompt_references(
    prompt: str,
    item: Item,
    color_map: SpanColorMap,
    transform_registry: TransformRegistry | None = None,
) -> str:
    """Replace ``[[label]]`` references in a prompt with highlighted HTML.

    When a reference includes transform names (e.g. ``[[label|gerund]]``),
    the display text is passed through the corresponding transforms
    looked up in *transform_registry*.

    Parameters
    ----------
    prompt : str
        Prompt template with ``[[label]]``, ``[[label:text]]``, or
        ``[[label|transform]]`` references.
    item : Item
        Item with spans and tokenized_elements.
    color_map : SpanColorMap
        Pre-computed color assignments from ``_assign_span_colors()``.
    transform_registry : TransformRegistry | None
        Optional registry for resolving ``|transform`` names.  When
        ``None``, any transforms in references are silently ignored.

    Returns
    -------
    str
        Prompt with references replaced by highlighted HTML.

    Raises
    ------
    ValueError
        If a reference points to a nonexistent label.
    KeyError
        If a transform name is not found in the registry.
    """
    refs = parse_label_refs(prompt)
    if not refs:
        return prompt

    available = {s.label.label for s in item.spans if s.label and s.label.label}
    for ref in refs:
        if ref.label not in available:
            raise ValueError(
                f"Prompt references span label '{ref.label}' but no span "
                f"with that label exists. Available labels: "
                f"{sorted(available)}"
            )

    result = prompt
    for ref in reversed(refs):
        display = (
            ref.display_text
            if ref.display_text is not None
            else _auto_fill_span_text(ref.label, item)
        )

        # apply transforms if requested and a registry is available
        if ref.transforms and transform_registry is not None:
            context = _build_transform_context(ref.label, item)
            pipeline = transform_registry.resolve_pipeline(list(ref.transforms))
            display = pipeline(display, context)

        light = color_map.light_by_label.get(ref.label, "#BBDEFB")
        dark = color_map.dark_by_label.get(ref.label, "#1565C0")
        html = (
            f'<span class="bead-q-highlight" style="background:{light}">'
            f"{display}"
            f'<span class="bead-q-chip" style="background:{dark}">'
            f"{ref.label}</span></span>"
        )
        result = result[: ref.start_offset] + html + result[ref.end_offset :]

    return result


def _build_transform_context(label: str, item: Item) -> TransformContext:
    """Build a TransformContext from an item's span metadata.

    Extracts head index, tokens, lemma, and POS from the first span
    matching *label* so that morphological transforms have the
    information they need.

    Parameters
    ----------
    label : str
        Span label to look up.
    item : Item
        Item with spans and tokenized_elements.

    Returns
    -------
    TransformContext
        Context populated with available span metadata.
    """
    target_span: Span | None = None
    for span in item.spans:
        if span.label and span.label.label == label:
            target_span = span
            break

    if target_span is None:
        return TransformContext()

    # extract tokens from the span
    tokens: list[str] = []
    for segment in target_span.segments:
        element_tokens = item.tokenized_elements.get(segment.element_name, [])
        for idx in sorted(segment.indices):
            if idx < len(element_tokens):
                tokens.append(element_tokens[idx])

    # extract metadata from span_metadata if available
    metadata = dict(target_span.span_metadata) if target_span.span_metadata else {}
    lemma = metadata.get("lemma") if isinstance(metadata.get("lemma"), str) else None
    pos = metadata.get("pos") if isinstance(metadata.get("pos"), str) else None

    return TransformContext(
        lemma=lemma,
        pos=pos,
        head_index=target_span.head_index,
        tokens=tokens,
        metadata=metadata,
    )


def _create_span_labeling_trial(
    item: Item,
    template: ItemTemplate,
    span_display: SpanDisplayConfig,
    trial_number: int,
) -> dict[str, JsonValue]:
    """Create a standalone span labeling trial.

    Uses the ``bead-span-label`` plugin for interactive or static span
    annotation.

    Parameters
    ----------
    item : Item
        Item with span data.
    template : ItemTemplate
        Item template with span_spec.
    span_display : SpanDisplayConfig
        Visual configuration.
    trial_number : int
        Trial number.

    Returns
    -------
    dict[str, JsonValue]
        A jsPsych trial object using the bead-span-label plugin.
    """
    metadata = _serialize_item_metadata(item, template)
    metadata["trial_number"] = trial_number
    metadata["trial_type"] = "span_labeling"

    prompt = (
        template.task_spec.prompt if template.task_spec else "Select and label spans"
    )

    if item.spans:
        color_map = _assign_span_colors(item.spans, span_display)
        prompt = _resolve_prompt_references(prompt, item, color_map)

    # serialize span data for the plugin
    spans_data = [
        {
            "span_id": span.span_id,
            "segments": [
                {"element_name": seg.element_name, "indices": seg.indices}
                for seg in span.segments
            ],
            "head_index": span.head_index,
            "label": (
                {
                    "label": span.label.label,
                    "label_id": span.label.label_id,
                    "confidence": span.label.confidence,
                }
                if span.label
                else None
            ),
            "span_type": span.span_type,
        }
        for span in item.spans
    ]

    relations_data = [
        {
            "relation_id": rel.relation_id,
            "source_span_id": rel.source_span_id,
            "target_span_id": rel.target_span_id,
            "label": (
                {
                    "label": rel.label.label,
                    "label_id": rel.label.label_id,
                    "confidence": rel.label.confidence,
                }
                if rel.label
                else None
            ),
            "directed": rel.directed,
        }
        for rel in item.span_relations
    ]

    # serialize span_spec
    span_spec_data = None
    if template.task_spec.span_spec:
        ss = template.task_spec.span_spec
        span_spec_data = {
            "index_mode": ss.index_mode,
            "interaction_mode": ss.interaction_mode,
            "label_source": ss.label_source,
            "labels": ss.labels,
            "label_colors": ss.label_colors,
            "allow_overlapping": ss.allow_overlapping,
            "min_spans": ss.min_spans,
            "max_spans": ss.max_spans,
            "enable_relations": ss.enable_relations,
            "relation_label_source": ss.relation_label_source,
            "relation_labels": ss.relation_labels,
            "relation_directed": ss.relation_directed,
            "min_relations": ss.min_relations,
            "max_relations": ss.max_relations,
            "wikidata_language": ss.wikidata_language,
            "wikidata_entity_types": ss.wikidata_entity_types,
            "wikidata_result_limit": ss.wikidata_result_limit,
        }

    # serialize display config
    display_config_data = {
        "highlight_style": span_display.highlight_style,
        "color_palette": span_display.color_palette,
        "show_labels": span_display.show_labels,
        "show_tooltips": span_display.show_tooltips,
        "token_delimiter": span_display.token_delimiter,
        "label_position": span_display.label_position,
    }

    return {
        "type": "bead-span-label",
        "tokens": dict(item.tokenized_elements),
        "space_after": {k: list(v) for k, v in item.token_space_after.items()},
        "spans": spans_data,
        "relations": relations_data,
        "span_spec": span_spec_data,
        "display_config": display_config_data,
        "prompt": prompt,
        "button_label": "Continue",
        "require_response": True,
        "metadata": metadata,
    }
