"""Bridge from the protocol layer to bead's item-construction layer.

A :class:`~bead.protocol.QuestionFamily` declares the type-level shape
of a question; a :class:`~bead.protocol.QuestionRealization` is one
realization of that question for a particular
:class:`~bead.protocol.ProtocolContext`. To deploy realizations through
bead's experimental pipeline they must be packaged as
:class:`~bead.items.item_template.ItemTemplate` and
:class:`~bead.items.item.Item` instances.

This module is the canonical bridge. It defines two mappings:

- :func:`scale_type_to_task_type` — the single canonical translation
  from :class:`~bead.protocol.ScaleType` to the
  :class:`~bead.items.item_template.TaskType` literal used by item
  templates and active-learning model selection.
- :func:`family_to_item_template` — build the per-family
  :class:`ItemTemplate` (one template per anchor; the same template is
  reused for every realization of that family).
- :func:`realization_to_item` — package a single
  :class:`QuestionRealization` as an :class:`Item` bound to the
  family's template, with sentence text and span metadata derived
  from the realization's :class:`ProtocolContext`.
- :func:`protocol_to_item_templates` — return a name-keyed dict of
  templates for an entire protocol.

The mapping is total: every supported :class:`ScaleType` corresponds
to exactly one :class:`TaskType`, and every protocol family produces
exactly one :class:`ItemTemplate`. There is no per-task-type factory
in the protocol layer; the family + realization pair is the single
canonical way to build items for a protocol.
"""

from __future__ import annotations

from typing import Final

from bead.items.item import Item
from bead.items.item_template import (
    ItemElement,
    ItemTemplate,
    JudgmentType,
    PresentationSpec,
    ScaleBounds,
    ScalePointLabel,
    TaskSpec,
    TaskType,
)
from bead.items.spans import Span, SpanLabel, SpanSegment
from bead.protocol.context import ContextItem, ProtocolContext
from bead.protocol.encoding import ScaleType, encode_response_space
from bead.protocol.family import (
    AnnotationProtocol,
    QuestionFamily,
    QuestionRealization,
)

_SCALE_TO_TASK: Final[dict[ScaleType, TaskType]] = {
    ScaleType.BINARY: "binary",
    ScaleType.ORDINAL: "ordinal_scale",
    ScaleType.NOMINAL: "categorical",
    ScaleType.FORCED_CHOICE: "forced_choice",
}
"""The canonical :class:`ScaleType` → :class:`TaskType` mapping."""


def scale_type_to_task_type(scale_type: ScaleType) -> TaskType:
    """Translate a :class:`ScaleType` to its :class:`TaskType`.

    This is the single canonical mapping used by every part of bead
    that bridges between protocol-layer encodings and item-layer task
    types (item construction, active-learning model selection, jsPsych
    deployment).

    Parameters
    ----------
    scale_type : ScaleType
        Protocol-layer scale type.

    Returns
    -------
    TaskType
        The matching :class:`TaskType` literal.

    Examples
    --------
    >>> from bead.protocol.encoding import ScaleType
    >>> scale_type_to_task_type(ScaleType.ORDINAL)
    'ordinal_scale'
    """
    return _SCALE_TO_TASK[scale_type]


def family_to_item_template(
    family: QuestionFamily,
    *,
    judgment_type: JudgmentType,
    presentation_spec: PresentationSpec | None = None,
) -> ItemTemplate:
    """Build the :class:`ItemTemplate` for a :class:`QuestionFamily`.

    The template's ``task_type`` is derived from the anchor's response
    space via :func:`scale_type_to_task_type`. Ordinal scales
    populate :attr:`TaskSpec.scale_bounds` (``0`` to ``n_levels - 1``)
    and :attr:`TaskSpec.scale_labels` (one
    :class:`ScalePointLabel` per option). Binary and nominal scales
    populate :attr:`TaskSpec.options` with the anchor's labels.
    Forced-choice scales leave :attr:`TaskSpec.options` unset (the
    per-item alternatives live on each :class:`Item` rather than on
    the template); the anchor's labels remain accessible via
    ``family.anchor.response_space.options``.

    The ``prompt`` field of the template's :class:`TaskSpec` is the
    anchor's canonical prompt (with ``[[label]]`` references intact);
    individual realizations override the prompt at item-construction
    time via the ``prompt`` rendered-element on the resulting
    :class:`Item`.

    Parameters
    ----------
    family : QuestionFamily
        The family to bridge.
    judgment_type : JudgmentType
        Semantic property being measured (caller-supplied because
        bead's :class:`JudgmentType` taxonomy is broader than
        :class:`~bead.protocol.encoding.ScaleType`).
    presentation_spec : PresentationSpec | None, optional
        Custom presentation spec. Defaults to a fresh
        :class:`PresentationSpec` with mode ``"static"``.

    Returns
    -------
    ItemTemplate
        Template with ``name`` set to the anchor name, ``task_type``
        derived from the scale, and ``elements`` covering ``"text"``
        (the sentence) and ``"prompt"`` (the realized question).
    """
    encoding = encode_response_space(family.anchor.name, family.anchor.response_space)
    task_type = scale_type_to_task_type(encoding.scale_type)

    if encoding.is_ordinal:
        scale_bounds: ScaleBounds | None = ScaleBounds(min=0, max=encoding.n_levels - 1)
        scale_labels = tuple(
            ScalePointLabel(point=i, label=label)
            for i, label in enumerate(encoding.labels)
        )
        options: tuple[str, ...] | None = None
    elif encoding.is_forced_choice:
        # forced-choice options live on each Item (the pair-specific
        # text); the template carries no per-template options.
        scale_bounds = None
        scale_labels = ()
        options = None
    else:
        scale_bounds = None
        scale_labels = ()
        options = encoding.labels

    task_spec = TaskSpec(
        prompt=family.anchor.canonical_prompt,
        scale_bounds=scale_bounds,
        scale_labels=scale_labels,
        options=options,
    )
    elements = (
        ItemElement(
            element_type="text",
            element_name="text",
            content="",
            order=0,
        ),
        ItemElement(
            element_type="text",
            element_name="prompt",
            content=family.anchor.canonical_prompt,
            order=1,
        ),
    )
    return ItemTemplate(
        name=family.anchor.name,
        description=family.anchor.description or None,
        judgment_type=judgment_type,
        task_type=task_type,
        task_spec=task_spec,
        presentation_spec=presentation_spec or PresentationSpec(),
        elements=elements,
    )


def _spans_from_context(
    realization: QuestionRealization,
) -> tuple[Span, ...]:
    """Extract :class:`Span` objects from a realization's context.

    Builds one span per ``required_span_label`` on the anchor, taking
    its token positions from the matching field on
    :class:`ProtocolContext` (target span for the anchor's primary
    label, dependent span for any other required label that matches a
    dependent's ``head_lemma``).

    Parameters
    ----------
    realization : QuestionRealization
        The realized question carrying the context.

    Returns
    -------
    tuple[Span, ...]
        Spans keyed to required label names. Empty when the anchor
        has no required span labels.
    """
    anchor = realization.anchor
    context = realization.context
    if not anchor.required_span_labels:
        return ()

    spans: list[Span] = []
    label_to_dependent: dict[str, ContextItem] = {
        d.head_lemma: d for d in context.dependents if d.head_lemma
    }

    for label in sorted(anchor.required_span_labels):
        if label_to_dependent and label in label_to_dependent:
            dep = label_to_dependent[label]
            if dep.span_positions:
                spans.append(
                    Span(
                        span_id=f"{anchor.name}-{label}",
                        segments=(
                            SpanSegment(
                                element_name="text",
                                indices=tuple(i - 1 for i in dep.span_positions),
                            ),
                        ),
                        label=SpanLabel(label=label),
                        head_index=(
                            dep.head_position - 1 if dep.head_position > 0 else None
                        ),
                    )
                )
                continue

        if context.target_span_positions:
            spans.append(
                Span(
                    span_id=f"{anchor.name}-{label}",
                    segments=(
                        SpanSegment(
                            element_name="text",
                            indices=tuple(i - 1 for i in context.target_span_positions),
                        ),
                    ),
                    label=SpanLabel(label=label),
                    head_index=(
                        context.target_position - 1
                        if context.target_position > 0
                        else None
                    ),
                )
            )

    return tuple(spans)


def realization_to_item(
    realization: QuestionRealization,
    *,
    item_template: ItemTemplate,
) -> Item:
    """Package a :class:`QuestionRealization` as an :class:`Item`.

    The resulting :class:`Item` references ``item_template`` by id,
    rendering the realization's prompt as the ``"prompt"`` element and
    the context's sentence as the ``"text"`` element. Tokenized
    elements are populated from
    :attr:`ProtocolContext.tokens` when present; spans for the
    anchor's ``required_span_labels`` are derived via
    :func:`_spans_from_context`.

    Parameters
    ----------
    realization : QuestionRealization
        A realized question produced by
        :meth:`QuestionFamily.realize`.
    item_template : ItemTemplate
        The template returned by :func:`family_to_item_template` for
        the originating family. The bridge does not validate that the
        template was produced from the same family — the caller is
        responsible for matching them.

    Returns
    -------
    Item
        Item bound to the template, with the realization's prompt and
        context materialized.
    """
    context = realization.context
    rendered_elements = {
        "text": context.sentence,
        "prompt": realization.prompt,
    }
    tokenized_elements: dict[str, tuple[str, ...]] = {}
    if context.tokens:
        tokenized_elements["text"] = context.tokens

    spans = _spans_from_context(realization)

    return Item(
        item_template_id=item_template.id,
        rendered_elements=rendered_elements,
        tokenized_elements=tokenized_elements,
        spans=spans,
    )


def protocol_to_item_templates(
    protocol: AnnotationProtocol,
    *,
    judgment_type: JudgmentType,
    presentation_spec: PresentationSpec | None = None,
) -> dict[str, ItemTemplate]:
    """Build one :class:`ItemTemplate` per family in the protocol.

    Parameters
    ----------
    protocol : AnnotationProtocol
        The protocol whose families to translate.
    judgment_type : JudgmentType
        Common judgment type to assign to every template.
    presentation_spec : PresentationSpec | None, optional
        Common presentation spec; defaults to a fresh one per call.

    Returns
    -------
    dict[str, ItemTemplate]
        Mapping from family / anchor name to its :class:`ItemTemplate`.
    """
    return {
        family.name: family_to_item_template(
            family,
            judgment_type=judgment_type,
            presentation_spec=presentation_spec,
        )
        for family in protocol.families
    }


def realize_protocol_to_items(
    protocol: AnnotationProtocol,
    context: ProtocolContext,
    *,
    judgment_type: JudgmentType,
    item_templates: dict[str, ItemTemplate] | None = None,
    responses: dict[str, str] | None = None,
    presentation_spec: PresentationSpec | None = None,
) -> tuple[tuple[QuestionRealization, Item], ...]:
    """Realize a protocol against one context, packaging items.

    Each applicable family is realized in protocol order; each
    :class:`QuestionRealization` is paired with the :class:`Item`
    produced by :func:`realization_to_item`.

    Parameters
    ----------
    protocol : AnnotationProtocol
        Protocol to realize.
    context : ProtocolContext
        Base context for realization.
    judgment_type : JudgmentType
        Judgment type assigned to every template.
    item_templates : dict[str, ItemTemplate] | None, optional
        Pre-built templates; built via
        :func:`protocol_to_item_templates` when ``None``.
    responses : dict[str, str] | None, optional
        Pre-supplied responses threaded into the context. Defaults to
        ``None``.
    presentation_spec : PresentationSpec | None, optional
        Common presentation spec when templates are built fresh.

    Returns
    -------
    tuple[tuple[QuestionRealization, Item], ...]
        For each applicable family, the ``(realization, item)`` pair
        in protocol order.
    """
    templates = item_templates or protocol_to_item_templates(
        protocol,
        judgment_type=judgment_type,
        presentation_spec=presentation_spec,
    )
    realizations = protocol.realize_all(context, responses=responses)
    return tuple(
        (r, realization_to_item(r, item_template=templates[r.anchor.name]))
        for r in realizations
    )
