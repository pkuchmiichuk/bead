"""Lens between a bead ``Item`` and canonical layers annotation records.

A bead :class:`~bead.items.item.Item` carries standoff span and relation
annotations over one or more rendered elements. This lens projects an item to a
:class:`lairs.integrations.codecs.CorpusFragment` of canonical ``lairs`` records:

- one :class:`lairs.records.expression.Expression` per rendered element,
- one :class:`lairs.records.segmentation.Segmentation` per tokenized element,
- one span :class:`lairs.records.annotation.AnnotationLayer` per element whose
  spans anchor by ``tokenRefSequence`` (``head_index`` maps to
  ``anchorTokenIndex``; a Wikidata ``label_id`` maps to a ``knowledgeRef``),
- one relation ``AnnotationLayer`` whose annotations carry ``ArgumentRef``
  source/target ``objectRef`` arguments.

The layers ``token`` has no space-after slot and an ``Item`` carries many fields
``layers`` cannot express, so the round-trip is a ``dx.Lens``: the view captures
the text, tokenization, and a faithful annotation projection; the complement
carries the bead-only remainder (framework identity, the item-construction
fields, the space-after flags, and the spans and relations verbatim), so the
GetPut/PutGet laws hold for every item.
"""

from __future__ import annotations

import json
import re
from uuid import UUID

import didactic.api as dx
from lairs.integrations.codecs import CorpusFragment, FragmentRecord
from lairs.records import annotation, defs, expression, segmentation

from bead.data.base import JsonValue
from bead.interop.layers._convert import (
    ANNOTATION_LAYER_NSID,
    CONFIDENCE_SCALE,
    EXPRESSION_NSID,
    SEGMENTATION_NSID,
    apply_identity,
    dumps_meta,
    identity_of,
    j_list,
    j_obj,
    j_str,
    loads_meta,
)
from bead.items.item import (
    ConstraintSatisfaction,
    Item,
    ModelOutput,
    UnfilledSlot,
)
from bead.items.spans import Span, SpanRelation

_WIKIDATA_QID = re.compile(r"Q\d+")


def _expression_uri(element: str) -> str:
    return f"at://local/{element}"


def _tokenization_uuid(element: str) -> str:
    return f"{element}:tok"


def _token_spans(
    tokens: tuple[str, ...], space_after: tuple[bool, ...]
) -> list[defs.Span]:
    """Compute UTF-8 byte spans for tokens joined by their space-after flags."""
    spans: list[defs.Span] = []
    cursor = 0
    char_cursor = 0
    for index, token in enumerate(tokens):
        byte_start = cursor
        char_start = char_cursor
        cursor += len(token.encode("utf-8"))
        char_cursor += len(token)
        spans.append(
            defs.Span(
                byteStart=byte_start,
                byteEnd=cursor,
                charStart=char_start,
                charEnd=char_cursor,
            )
        )
        gap = index < len(space_after) and space_after[index]
        if gap:
            cursor += 1
            char_cursor += 1
    return spans


def _knowledge_refs(
    label_id: str | None, label: str | None
) -> tuple[defs.KnowledgeRef, ...]:
    if label_id is None:
        return ()
    source = "wikidata" if _WIKIDATA_QID.fullmatch(label_id) else "custom"
    return (defs.KnowledgeRef(identifier=label_id, source=source, label=label),)


def _span_annotation(span: Span, element: str) -> annotation.Annotation:
    indices = tuple(
        index
        for segment in span.segments
        if segment.element_name == element
        for index in segment.indices
    )
    label = span.label.label if span.label is not None else None
    label_id = span.label.label_id if span.label is not None else None
    confidence = (
        round(span.label.confidence * CONFIDENCE_SCALE)
        if span.label is not None and span.label.confidence is not None
        else None
    )
    return annotation.Annotation(
        uuid=defs.Uuid(value=span.span_id),
        anchor=defs.Anchor(
            tokenRefSequence=defs.TokenRefSequence(
                tokenIndexes=indices,
                anchorTokenIndex=span.head_index,
                tokenizationId=defs.Uuid(value=_tokenization_uuid(element)),
            )
        ),
        label=label,
        knowledgeRefs=_knowledge_refs(label_id, label),
        confidence=confidence,
    )


def _relation_annotation(relation: SpanRelation) -> annotation.Annotation:
    label = relation.label.label if relation.label is not None else None
    return annotation.Annotation(
        uuid=defs.Uuid(value=relation.relation_id),
        label=label,
        arguments=(
            annotation.ArgumentRef(
                role="source",
                target=defs.ObjectRef(localId=defs.Uuid(value=relation.source_span_id)),
            ),
            annotation.ArgumentRef(
                role="target",
                target=defs.ObjectRef(localId=defs.Uuid(value=relation.target_span_id)),
            ),
        ),
    )


class ItemLayersLens(dx.Lens[Item, CorpusFragment, JsonValue]):
    """Lossless lens ``Item <-> (layers corpus fragment, bead complement)``."""

    def forward(self, item: Item) -> tuple[CorpusFragment, JsonValue]:
        """Project an item to a layers fragment and bead complement."""
        records: list[FragmentRecord] = []
        spans_by_element: dict[str, list[Span]] = {}
        for span in item.spans:
            if span.segments:
                element = span.segments[0].element_name
                spans_by_element.setdefault(element, []).append(span)

        for element, text in item.rendered_elements.items():
            records.append(
                _record(
                    EXPRESSION_NSID,
                    f"expression:{element}",
                    expression.Expression(
                        id=element,
                        kind="sentence",
                        createdAt=item.created_at,
                        text=text,
                    ),
                )
            )
            tokens = item.tokenized_elements.get(element)
            if tokens:
                spaces = item.token_space_after.get(element, ())
                token_spans = _token_spans(tokens, spaces)
                tokenization = segmentation.Tokenization(
                    uuid=defs.Uuid(value=_tokenization_uuid(element)),
                    kind="custom",
                    tokens=tuple(
                        segmentation.Token(
                            tokenIndex=index, text=token, textSpan=token_spans[index]
                        )
                        for index, token in enumerate(tokens)
                    ),
                )
                records.append(
                    _record(
                        SEGMENTATION_NSID,
                        f"segmentation:{element}",
                        segmentation.Segmentation(
                            createdAt=item.created_at,
                            expression=_expression_uri(element),
                            tokenizations=(tokenization,),
                        ),
                    )
                )
            element_spans = spans_by_element.get(element)
            if element_spans:
                records.append(
                    _record(
                        ANNOTATION_LAYER_NSID,
                        f"spans:{element}",
                        annotation.AnnotationLayer(
                            annotations=tuple(
                                _span_annotation(span, element)
                                for span in element_spans
                            ),
                            createdAt=item.created_at,
                            expression=_expression_uri(element),
                            kind="span",
                            tokenizationId=defs.Uuid(value=_tokenization_uuid(element)),
                        ),
                    )
                )

        if item.span_relations:
            first_element = next(iter(item.rendered_elements), "text")
            records.append(
                _record(
                    ANNOTATION_LAYER_NSID,
                    "relations",
                    annotation.AnnotationLayer(
                        annotations=tuple(
                            _relation_annotation(relation)
                            for relation in item.span_relations
                        ),
                        createdAt=item.created_at,
                        expression=_expression_uri(first_element),
                        kind="relation",
                    ),
                )
            )

        view = CorpusFragment(records=tuple(records), source="bead")
        complement: JsonValue = {
            "identity": identity_of(item),
            "item_template_id": str(item.item_template_id),
            "filled_template_refs": tuple(
                str(ref) for ref in item.filled_template_refs
            ),
            "options": item.options,
            "unfilled_slots": tuple(
                slot.model_dump_json() for slot in item.unfilled_slots
            ),
            "model_outputs": tuple(
                output.model_dump_json() for output in item.model_outputs
            ),
            "constraint_satisfaction": tuple(
                record.model_dump_json() for record in item.constraint_satisfaction
            ),
            "item_metadata": dumps_meta(item.item_metadata),
            "token_space_after": json.dumps(
                {key: list(value) for key, value in item.token_space_after.items()}
            ),
            "spans": tuple(span.model_dump_json() for span in item.spans),
            "span_relations": tuple(
                relation.model_dump_json() for relation in item.span_relations
            ),
        }
        return view, complement

    def backward(self, view: CorpusFragment, complement: JsonValue) -> Item:
        """Reconstruct an item from its layers fragment and bead complement."""
        comp = j_obj(complement)
        rendered: dict[str, str] = {}
        tokenized: dict[str, tuple[str, ...]] = {}
        for record in view.records:
            if record.nsid == EXPRESSION_NSID:
                expr = expression.Expression.model_validate_json(record.value_json)
                rendered[expr.id] = expr.text if expr.text is not None else ""
            elif record.nsid == SEGMENTATION_NSID:
                seg = segmentation.Segmentation.model_validate_json(record.value_json)
                element = seg.expression.removeprefix("at://local/")
                tokenization = seg.tokenizations[0]
                tokenized[element] = tuple(
                    token.text if token.text is not None else ""
                    for token in tokenization.tokens
                )

        token_space_after = {
            key: tuple(bool(flag) for flag in value)
            for key, value in json.loads(j_str(comp["token_space_after"])).items()
        }
        item = Item(
            item_template_id=UUID(j_str(comp["item_template_id"])),
            filled_template_refs=tuple(
                UUID(j_str(ref)) for ref in j_list(comp["filled_template_refs"])
            ),
            rendered_elements=rendered,
            options=tuple(j_str(option) for option in j_list(comp["options"])),
            unfilled_slots=tuple(
                UnfilledSlot.model_validate_json(j_str(slot))
                for slot in j_list(comp["unfilled_slots"])
            ),
            model_outputs=tuple(
                ModelOutput.model_validate_json(j_str(output))
                for output in j_list(comp["model_outputs"])
            ),
            constraint_satisfaction=tuple(
                ConstraintSatisfaction.model_validate_json(j_str(record))
                for record in j_list(comp["constraint_satisfaction"])
            ),
            item_metadata=loads_meta(comp["item_metadata"]),
            spans=tuple(
                Span.model_validate_json(j_str(span)) for span in j_list(comp["spans"])
            ),
            span_relations=tuple(
                SpanRelation.model_validate_json(j_str(relation))
                for relation in j_list(comp["span_relations"])
            ),
            tokenized_elements=tokenized,
            token_space_after=token_space_after,
        )
        return apply_identity(item, comp["identity"])


def _record(nsid: str, local_id: str, model: dx.Model) -> FragmentRecord:
    return FragmentRecord(
        local_id=local_id, nsid=nsid, value_json=model.model_dump_json()
    )


ITEM_LAYERS = ItemLayersLens()


def item_to_layers(item: Item) -> CorpusFragment:
    """Return the standalone layers fragment view of an item."""
    view, _complement = ITEM_LAYERS.forward(item)
    return view
