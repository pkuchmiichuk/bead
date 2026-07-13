"""Lenses between bead resource models and layers resource records.

Maps bead's lexical and template resources to their canonical
:mod:`lairs.records.resource` counterparts:

- ``LexicalItem`` <-> a layers ``entry``
- ``Lexicon`` <-> a layers ``collection`` with its ``entry`` records
- ``Template`` <-> a layers ``template`` (with its slots and constraints)
- ``FilledTemplate`` <-> a layers ``filling`` (with its per-slot fillings)

Each lens produces a layers-shaped view from the generated models and keeps the
fields that have no layers equivalent (the bead framework identity, tags, the
``LexicalItem`` original ``form`` / free-text ``source``, the bead DSL
constraint context, and the filled-template slot requirement map) in the lens
complement, so reconstruction is exact.
"""

from __future__ import annotations

from typing import cast

import didactic.api as dx
from lairs.records import defs, resource

from bead.data.base import JsonValue
from bead.interop.layers._convert import (
    apply_identity,
    dumps_meta,
    feature_map,
    identity_of,
    j_bool,
    j_list,
    j_obj,
    j_str,
    j_str_or_none,
    loads_meta,
    read_feature_map,
)
from bead.resources.constraints import Constraint
from bead.resources.lexical_item import LexicalItem
from bead.resources.lexicon import Lexicon
from bead.resources.template import Slot, Template
from bead.templates.filler import FilledTemplate

_LEXICON_KIND = "lexicon"


def _languages(code: str | None) -> tuple[str, ...]:
    return (code,) if code is not None else ()


def _first_language(languages: tuple[str, ...] | None) -> str | None:
    return languages[0] if languages else None


class LexicalItemEntryLens(dx.Lens[LexicalItem, resource.Entry, JsonValue]):
    """Lossless lens ``LexicalItem <-> (layers entry, complement)``."""

    def forward(self, item: LexicalItem) -> tuple[resource.Entry, JsonValue]:
        """Project a lexical item to a layers entry and complement."""
        view = resource.Entry(
            form=item.form if item.form is not None else item.lemma,
            lemma=item.lemma,
            features=feature_map(item.features),
            languages=_languages(item.language_code),
            createdAt=item.created_at,
        )
        complement: JsonValue = {
            "identity": identity_of(item),
            "form": item.form,
            "source": item.source,
        }
        return view, complement

    def backward(self, view: resource.Entry, complement: JsonValue) -> LexicalItem:
        """Reconstruct a lexical item from its layers entry and complement."""
        comp = j_obj(complement)
        item = LexicalItem(
            lemma=view.lemma if view.lemma is not None else "",
            language_code=_first_language(view.languages),
            form=j_str_or_none(comp["form"]),
            features=read_feature_map(view.features),
            source=j_str_or_none(comp["source"]),
        )
        return apply_identity(item, comp["identity"])


LEXICAL_ITEM_ENTRY = LexicalItemEntryLens()


class LexiconLayers(dx.Model):
    """A layers view of a lexicon: a collection plus its entry records."""

    collection: dx.Embed[resource.Collection] = dx.field()
    entries: tuple[dx.Embed[resource.Entry], ...] = dx.field(default=())


class LexiconCollectionLens(dx.Lens[Lexicon, LexiconLayers, JsonValue]):
    """Lossless lens ``Lexicon <-> (layers collection + entries, complement)``."""

    def forward(self, lexicon: Lexicon) -> tuple[LexiconLayers, JsonValue]:
        """Project a lexicon to a layers collection + entry views."""
        entries: list[resource.Entry] = []
        item_complements: list[JsonValue] = []
        for item in lexicon.items:
            entry_view, entry_complement = LEXICAL_ITEM_ENTRY.forward(item)
            entries.append(entry_view)
            item_complements.append(entry_complement)
        view = LexiconLayers(
            collection=resource.Collection(
                name=lexicon.name,
                kind=_LEXICON_KIND,
                description=lexicon.description,
                languages=_languages(lexicon.language_code),
                createdAt=lexicon.created_at,
            ),
            entries=tuple(entries),
        )
        complement: JsonValue = {
            "identity": identity_of(lexicon),
            "tags": lexicon.tags,
            "item_complements": tuple(item_complements),
        }
        return view, complement

    def backward(self, view: LexiconLayers, complement: JsonValue) -> Lexicon:
        """Reconstruct a lexicon from its layers collection + complement."""
        comp = j_obj(complement)
        item_complements = j_list(comp["item_complements"])
        items = tuple(
            LEXICAL_ITEM_ENTRY.backward(entry, item_complement)
            for entry, item_complement in zip(
                view.entries, item_complements, strict=True
            )
        )
        lexicon = Lexicon(
            name=view.collection.name,
            description=view.collection.description,
            language_code=_first_language(view.collection.languages),
            items=items,
            tags=tuple(j_str(tag) for tag in j_list(comp["tags"])),
        )
        return apply_identity(lexicon, comp["identity"])


LEXICON_COLLECTION = LexiconCollectionLens()


def _constraint_forward(constraint: Constraint) -> tuple[defs.Constraint, JsonValue]:
    view = defs.Constraint(
        expression=constraint.expression, description=constraint.description
    )
    complement: JsonValue = {
        "identity": identity_of(constraint),
        "context": dumps_meta(constraint.context),
    }
    return view, complement


def _constraint_backward(view: defs.Constraint, complement: JsonValue) -> Constraint:
    comp = j_obj(complement)
    constraint = Constraint(
        expression=view.expression,
        context=loads_meta(comp["context"]),
        description=view.description,
    )
    return apply_identity(constraint, comp["identity"])


def _slot_forward(slot: Slot) -> tuple[resource.Slot, JsonValue]:
    constraint_views: list[defs.Constraint] = []
    constraint_complements: list[JsonValue] = []
    for constraint in slot.constraints:
        constraint_view, constraint_complement = _constraint_forward(constraint)
        constraint_views.append(constraint_view)
        constraint_complements.append(constraint_complement)
    view = resource.Slot(
        name=slot.name,
        description=slot.description,
        constraints=tuple(constraint_views),
        defaultValue=slot.default_value,
        required=slot.required,
    )
    complement: JsonValue = {
        "identity": identity_of(slot),
        "constraint_complements": tuple(constraint_complements),
    }
    return view, complement


def _slot_backward(view: resource.Slot, complement: JsonValue) -> Slot:
    comp = j_obj(complement)
    constraint_complements = j_list(comp["constraint_complements"])
    constraints = tuple(
        _constraint_backward(constraint_view, constraint_complement)
        for constraint_view, constraint_complement in zip(
            view.constraints or (), constraint_complements, strict=True
        )
    )
    slot = Slot(
        name=view.name,
        description=view.description,
        constraints=constraints,
        required=view.required if view.required is not None else True,
        default_value=view.defaultValue,
    )
    return apply_identity(slot, comp["identity"])


class TemplateLayersLens(dx.Lens[Template, resource.Template, JsonValue]):
    """Lossless lens ``Template <-> (layers template, complement)``."""

    def forward(self, template: Template) -> tuple[resource.Template, JsonValue]:
        """Project a template to a layers template and complement."""
        slot_views: list[resource.Slot] = []
        slot_complements: dict[str, JsonValue] = {}
        for slot_key, slot in template.slots.items():
            slot_view, slot_complement = _slot_forward(slot)
            slot_views.append(slot_view)
            slot_complements[slot_key] = slot_complement
        constraint_views: list[defs.Constraint] = []
        constraint_complements: list[JsonValue] = []
        for constraint in template.constraints:
            constraint_view, constraint_complement = _constraint_forward(constraint)
            constraint_views.append(constraint_view)
            constraint_complements.append(constraint_complement)
        view = resource.Template(
            name=template.name,
            text=template.template_string,
            slots=tuple(slot_views),
            constraints=tuple(constraint_views),
            languages=_languages(template.language_code),
            createdAt=template.created_at,
        )
        complement: JsonValue = {
            "identity": identity_of(template),
            "description": template.description,
            "tags": template.tags,
            "metadata": dumps_meta(template.metadata),
            "slot_order": tuple(template.slots),
            "slot_complements": slot_complements,
            "constraint_complements": tuple(constraint_complements),
        }
        return view, complement

    def backward(self, view: resource.Template, complement: JsonValue) -> Template:
        """Reconstruct a template from its layers template and complement."""
        comp = j_obj(complement)
        slot_complements = j_obj(comp["slot_complements"])
        slot_order = j_list(comp["slot_order"])
        slots: dict[str, Slot] = {}
        for slot_key_value, slot_view in zip(slot_order, view.slots, strict=True):
            slot_key = j_str(slot_key_value)
            slots[slot_key] = _slot_backward(slot_view, slot_complements[slot_key])
        constraint_complements = j_list(comp["constraint_complements"])
        constraints = tuple(
            _constraint_backward(constraint_view, constraint_complement)
            for constraint_view, constraint_complement in zip(
                view.constraints or (), constraint_complements, strict=True
            )
        )
        template = Template(
            name=view.name if view.name is not None else "",
            template_string=view.text,
            slots=slots,
            constraints=constraints,
            description=j_str_or_none(comp["description"]),
            language_code=_first_language(view.languages),
            tags=tuple(j_str(tag) for tag in j_list(comp["tags"])),
            metadata=loads_meta(comp["metadata"]),
        )
        return apply_identity(template, comp["identity"])


TEMPLATE_LAYERS = TemplateLayersLens()


class FilledTemplateFillingLens(dx.Lens[FilledTemplate, resource.Filling, JsonValue]):
    """Lossless lens ``FilledTemplate <-> (layers filling, complement)``.

    The layers ``filling`` record (``resource.Filling``) is the canonical
    representation of a filled template: it carries the template reference, the
    per-slot fillings, the rendered text, and the filling strategy. The bead-only
    remainder (identity, the source template name, the slot requirement map, and
    the exact lexical-item fillers) travels in the lens complement.
    """

    def forward(self, filled: FilledTemplate) -> tuple[resource.Filling, JsonValue]:
        """Project a filled template to a layers filling and complement."""
        slot_fillings: list[resource.SlotFilling] = []
        filler_complements: dict[str, JsonValue] = {}
        for slot_name, item in filled.slot_fillers.items():
            slot_fillings.append(
                resource.SlotFilling(
                    slotName=slot_name,
                    literalValue=item.lemma,
                    renderedForm=item.form if item.form is not None else item.lemma,
                    features=feature_map(item.features),
                )
            )
            filler_complements[slot_name] = item.model_dump_json()
        view = resource.Filling(
            templateRef=filled.template_id,
            slotFillings=tuple(slot_fillings),
            renderedText=filled.rendered_text,
            strategy=filled.strategy_name,
            createdAt=filled.created_at,
        )
        complement: JsonValue = {
            "identity": identity_of(filled),
            "template_id": filled.template_id,
            "template_name": filled.template_name,
            "template_slots": cast("dict[str, JsonValue]", dict(filled.template_slots)),
            "slot_order": tuple(filled.slot_fillers),
            "filler_complements": filler_complements,
        }
        return view, complement

    def backward(self, view: resource.Filling, complement: JsonValue) -> FilledTemplate:
        """Reconstruct a filled template from its layers filling and complement."""
        comp = j_obj(complement)
        filler_complements = j_obj(comp["filler_complements"])
        slot_fillers: dict[str, LexicalItem] = {}
        for slot_name_value in j_list(comp["slot_order"]):
            slot_name = j_str(slot_name_value)
            slot_fillers[slot_name] = LexicalItem.model_validate_json(
                j_str(filler_complements[slot_name])
            )
        template_slots = {
            slot_name: j_bool(required)
            for slot_name, required in j_obj(comp["template_slots"]).items()
        }
        filled = FilledTemplate(
            template_id=j_str(comp["template_id"]),
            template_name=j_str(comp["template_name"]),
            slot_fillers=slot_fillers,
            rendered_text=view.renderedText if view.renderedText is not None else "",
            strategy_name=view.strategy if view.strategy is not None else "",
            template_slots=template_slots,
        )
        return apply_identity(filled, comp["identity"])


FILLED_TEMPLATE_FILLING = FilledTemplateFillingLens()
