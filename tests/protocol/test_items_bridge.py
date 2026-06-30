"""Tests for :mod:`bead.protocol.items`."""

from __future__ import annotations

import pytest

from bead.active_learning.models import ForcedChoiceModel, model_class_for_encoding
from bead.items.item_template import ItemTemplate, PresentationSpec
from bead.protocol import (
    AnnotationProtocol,
    ContextItem,
    ProtocolContext,
    QuestionFamily,
    ResponseSpace,
    ScaleType,
    SemanticAnchor,
    encode_response_space,
    family_to_item_template,
    protocol_to_item_templates,
    realization_to_item,
    realize_protocol_to_items,
    scale_type_to_task_type,
)


class TestScaleTypeToTaskType:
    """Tests for :func:`scale_type_to_task_type`."""

    def test_binary_maps(self) -> None:
        assert scale_type_to_task_type(ScaleType.BINARY) == "binary"

    def test_ordinal_maps(self) -> None:
        assert scale_type_to_task_type(ScaleType.ORDINAL) == "ordinal_scale"

    def test_nominal_maps(self) -> None:
        assert scale_type_to_task_type(ScaleType.NOMINAL) == "categorical"

    def test_forced_choice_maps(self) -> None:
        assert scale_type_to_task_type(ScaleType.FORCED_CHOICE) == "forced_choice"


def _build_forced_choice_anchor() -> SemanticAnchor:
    return SemanticAnchor(
        name="acceptability",
        target_property="acceptability",
        canonical_prompt="Which sentence sounds more natural?",
        response_space=ResponseSpace(
            options=("first", "second"),
            is_ordered=False,
            scale_type=ScaleType.FORCED_CHOICE,
        ),
        required_keywords=frozenset({"natural"}),
    )


class TestForcedChoiceFamilyTemplate:
    """family_to_item_template handles forced_choice anchors."""

    def test_forced_choice_template(self) -> None:
        family = QuestionFamily(anchor=_build_forced_choice_anchor())
        template = family_to_item_template(family, judgment_type="acceptability")
        assert template.task_type == "forced_choice"
        # forced-choice templates carry no per-template options;
        # the per-item alternatives live on each Item.
        assert template.task_spec.options is None
        assert template.task_spec.scale_bounds is None
        assert template.task_spec.scale_labels == ()

    def test_model_class_for_forced_choice_encoding(self) -> None:
        anchor = _build_forced_choice_anchor()
        encoding = encode_response_space(anchor.name, anchor.response_space)
        assert model_class_for_encoding(encoding) is ForcedChoiceModel


def _build_binary_anchor() -> SemanticAnchor:
    return SemanticAnchor(
        name="completion",
        target_property="telicity",
        canonical_prompt="Does [[situation]] reach an endpoint?",
        response_space=ResponseSpace(options=("no", "yes"), is_ordered=False),
        required_span_labels=frozenset({"situation"}),
        required_keywords=frozenset({"endpoint"}),
        description="Telicity probe.",
    )


def _build_ordinal_anchor() -> SemanticAnchor:
    return SemanticAnchor(
        name="confidence",
        target_property="confidence",
        canonical_prompt="How confident is [[situation]]?",
        response_space=ResponseSpace(
            options=("low", "medium", "high"),
            is_ordered=True,
        ),
        required_span_labels=frozenset({"situation"}),
    )


class TestFamilyToItemTemplate:
    """Tests for :func:`family_to_item_template`."""

    def test_binary_family(self) -> None:
        family = QuestionFamily(anchor=_build_binary_anchor())
        template = family_to_item_template(family, judgment_type="acceptability")
        assert isinstance(template, ItemTemplate)
        assert template.task_type == "binary"
        assert template.task_spec.options == ("no", "yes")
        assert template.task_spec.scale_bounds is None
        assert template.task_spec.scale_labels == ()
        # Two elements: text + prompt
        assert {e.element_name for e in template.elements} == {"text", "prompt"}

    def test_ordinal_family(self) -> None:
        family = QuestionFamily(anchor=_build_ordinal_anchor())
        template = family_to_item_template(family, judgment_type="acceptability")
        assert template.task_type == "ordinal_scale"
        assert template.task_spec.scale_bounds is not None
        assert template.task_spec.scale_bounds.min == 0
        assert template.task_spec.scale_bounds.max == 2
        assert len(template.task_spec.scale_labels) == 3
        labels = {(p.point, p.label) for p in template.task_spec.scale_labels}
        assert labels == {(0, "low"), (1, "medium"), (2, "high")}

    def test_custom_presentation_spec(self) -> None:
        family = QuestionFamily(anchor=_build_binary_anchor())
        spec = PresentationSpec(mode="self_paced")
        template = family_to_item_template(
            family,
            judgment_type="comprehension",
            presentation_spec=spec,
        )
        assert template.presentation_spec.mode == "self_paced"

    def test_judgment_type_propagates(self) -> None:
        family = QuestionFamily(anchor=_build_binary_anchor())
        template = family_to_item_template(family, judgment_type="inference")
        assert template.judgment_type == "inference"


class TestRealizationToItem:
    """Tests for :func:`realization_to_item`."""

    def test_basic_round_trip(self) -> None:
        family = QuestionFamily(anchor=_build_binary_anchor())
        template = family_to_item_template(family, judgment_type="acceptability")
        ctx = ProtocolContext(
            sentence="Mary built a sandcastle.",
            tokens=("Mary", "built", "a", "sandcastle", "."),
            target_position=2,
            target_span_text="built a sandcastle",
            target_span_positions=(2, 3, 4),
        )
        realization = family.realize(ctx)
        item = realization_to_item(realization, item_template=template)
        assert item.item_template_id == template.id
        assert item.rendered_elements["text"] == "Mary built a sandcastle."
        assert item.rendered_elements["prompt"] == realization.prompt
        assert item.tokenized_elements["text"] == ctx.tokens
        # The required_span_label "situation" yields a span anchored
        # to the target's positions (translated to 0-based indexing).
        assert len(item.spans) == 1
        span = item.spans[0]
        assert span.label is not None
        assert span.label.label == "situation"
        assert span.segments[0].element_name == "text"
        assert span.segments[0].indices == (1, 2, 3)
        assert span.head_index == 1

    def test_no_required_labels_no_spans(self) -> None:
        anchor = SemanticAnchor(
            name="dummy",
            target_property="dummy",
            canonical_prompt="Question?",
            response_space=ResponseSpace(options=("no", "yes"), is_ordered=False),
        )
        family = QuestionFamily(anchor=anchor)
        template = family_to_item_template(family, judgment_type="acceptability")
        ctx = ProtocolContext(sentence="Plain text.")
        realization = family.realize(ctx)
        item = realization_to_item(realization, item_template=template)
        assert item.spans == ()

    def test_dependent_span_picked_when_label_matches_lemma(self) -> None:
        anchor = SemanticAnchor(
            name="distrib",
            target_property="distributivity",
            canonical_prompt=(
                "Did [[situation]] involve [[participant]] one at a time?"
            ),
            response_space=ResponseSpace(options=("no", "yes"), is_ordered=False),
            required_span_labels=frozenset({"situation", "participant"}),
        )
        family = QuestionFamily(anchor=anchor)
        template = family_to_item_template(family, judgment_type="acceptability")
        ctx = ProtocolContext(
            sentence="The kids ran.",
            tokens=("The", "kids", "ran", "."),
            target_position=3,
            target_span_text="ran",
            target_span_positions=(3,),
            dependents=(
                ContextItem(
                    head_lemma="participant",  # matches required label
                    head_position=2,
                    span_text="The kids",
                    span_positions=(1, 2),
                ),
            ),
        )
        realization = family.realize(ctx)
        item = realization_to_item(realization, item_template=template)
        # Two spans; one for each required label
        labels_to_indices = {
            s.label.label: s.segments[0].indices
            for s in item.spans
            if s.label is not None
        }
        # situation → target span (3) → 0-based index 2
        assert labels_to_indices["situation"] == (2,)
        # participant → dependent span (1, 2) → 0-based (0, 1)
        assert labels_to_indices["participant"] == (0, 1)


class TestProtocolToItemTemplates:
    """Tests for :func:`protocol_to_item_templates` and :func:`realize_protocol_to_items`."""

    def test_protocol_to_templates(self) -> None:
        proto = AnnotationProtocol(
            families=[
                QuestionFamily(anchor=_build_binary_anchor()),
                QuestionFamily(anchor=_build_ordinal_anchor()),
            ]
        )
        templates = protocol_to_item_templates(proto, judgment_type="acceptability")
        assert set(templates) == {"completion", "confidence"}
        assert templates["completion"].task_type == "binary"
        assert templates["confidence"].task_type == "ordinal_scale"

    def test_realize_protocol_to_items(self) -> None:
        proto = AnnotationProtocol(
            families=[QuestionFamily(anchor=_build_binary_anchor())]
        )
        ctx = ProtocolContext(
            sentence="Mary built a sandcastle.",
            target_span_text="built a sandcastle",
            target_span_positions=(2, 3, 4),
        )
        pairs = realize_protocol_to_items(proto, ctx, judgment_type="acceptability")
        assert len(pairs) == 1
        realization, item = pairs[0]
        assert realization.anchor.name == "completion"
        assert item.rendered_elements["prompt"] == realization.prompt


def test_unknown_template_in_protocol_realize(
    _pytest_request: object | None = None,
) -> None:
    """``realize_protocol_to_items`` raises if a family has no template."""
    proto = AnnotationProtocol(families=[QuestionFamily(anchor=_build_binary_anchor())])
    other = QuestionFamily(anchor=_build_ordinal_anchor())
    other_templates = {
        "confidence": family_to_item_template(other, judgment_type="acceptability")
    }
    with pytest.raises(KeyError):
        realize_protocol_to_items(
            proto,
            ProtocolContext(),
            judgment_type="acceptability",
            item_templates=other_templates,
        )
