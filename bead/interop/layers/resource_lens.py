"""Bridge lenses between bead resource models and layers resource records.

The resource overlap is the cleanest experiment-domain mapping: bead's
``LexicalItem``/``Lexicon``/``Template`` correspond closely to layers'
``entry``/``collection``/``template``. Each lens projects a faithful layers view
and keeps the bead-only remainder (framework identity, single language code,
tags, DSL constraint context, the bead ``form``/``source`` fields layers slots
differently) in the complement, so the round-trip is exact (GetPut/PutGet).

The other experiment overlaps (judgment, corpus, persona, changelog) were
assessed as schema-divergent and are intentionally not mapped; see the module
docstring of :mod:`bead.interop.layers` and the project notes.
"""

from __future__ import annotations

import didactic.api as dx

from bead.data.base import JsonValue
from bead.interop.layers._convert import (
    apply_identity,
    from_feature_map,
    identity_of,
    j_bool,
    j_list,
    j_obj,
    j_str,
    j_str_or_none,
    to_feature_map,
)
from bead.resources.constraints import Constraint
from bead.resources.lexical_item import LexicalItem
from bead.resources.lexicon import Lexicon
from bead.resources.template import Slot, Template

_LEXICON_KIND = "lexicon"


class LexicalItemEntryLens(dx.Lens[LexicalItem, JsonValue, JsonValue]):
    """Lossless lens ``LexicalItem <-> (layers entry view, complement)``."""

    def forward(self, item: LexicalItem) -> tuple[JsonValue, JsonValue]:
        """Project a lexical item to a layers entry view and complement."""
        view: dict[str, JsonValue] = {
            "form": item.form if item.form is not None else item.lemma,
            "lemma": item.lemma,
            "features": to_feature_map(item.features),
        }
        if item.language_code is not None:
            view["languages"] = (item.language_code,)
        complement: JsonValue = {
            "identity": identity_of(item),
            "form": item.form,
            "language_code": item.language_code,
            "source": item.source,
        }
        return view, complement

    def backward(self, view: JsonValue, complement: JsonValue) -> LexicalItem:
        """Reconstruct a lexical item from its layers entry view and complement."""
        view_obj = j_obj(view)
        comp = j_obj(complement)
        item = LexicalItem(
            lemma=j_str(view_obj["lemma"]),
            language_code=j_str_or_none(comp["language_code"]),
            form=j_str_or_none(comp["form"]),
            features=from_feature_map(view_obj["features"]),
            source=j_str_or_none(comp["source"]),
        )
        return apply_identity(item, comp["identity"])


LEXICAL_ITEM_ENTRY = LexicalItemEntryLens()


class LexiconCollectionLens(dx.Lens[Lexicon, JsonValue, JsonValue]):
    """Lossless lens ``Lexicon <-> (layers collection + entries, complement)``."""

    def forward(self, lexicon: Lexicon) -> tuple[JsonValue, JsonValue]:
        """Project a lexicon to a layers collection + entry views."""
        collection: dict[str, JsonValue] = {"name": lexicon.name, "kind": _LEXICON_KIND}
        if lexicon.description is not None:
            collection["description"] = lexicon.description
        if lexicon.language_code is not None:
            collection["languages"] = (lexicon.language_code,)
        entries: list[JsonValue] = []
        item_complements: list[JsonValue] = []
        for item in lexicon.items:
            entry_view, entry_complement = LEXICAL_ITEM_ENTRY.forward(item)
            entries.append(entry_view)
            item_complements.append(entry_complement)
        view: JsonValue = {"collection": collection, "entries": tuple(entries)}
        complement: JsonValue = {
            "identity": identity_of(lexicon),
            "description": lexicon.description,
            "language_code": lexicon.language_code,
            "tags": lexicon.tags,
            "item_complements": tuple(item_complements),
        }
        return view, complement

    def backward(self, view: JsonValue, complement: JsonValue) -> Lexicon:
        """Reconstruct a lexicon from its layers collection + complement."""
        view_obj = j_obj(view)
        comp = j_obj(complement)
        collection = j_obj(view_obj["collection"])
        entries = j_list(view_obj["entries"])
        item_complements = j_list(comp["item_complements"])
        items = tuple(
            LEXICAL_ITEM_ENTRY.backward(entry, item_complement)
            for entry, item_complement in zip(entries, item_complements, strict=True)
        )
        lexicon = Lexicon(
            name=j_str(collection["name"]),
            description=j_str_or_none(comp["description"]),
            language_code=j_str_or_none(comp["language_code"]),
            items=items,
            tags=tuple(j_str(tag) for tag in j_list(comp["tags"])),
        )
        return apply_identity(lexicon, comp["identity"])


LEXICON_COLLECTION = LexiconCollectionLens()


def _constraint_forward(constraint: Constraint) -> tuple[JsonValue, JsonValue]:
    view: dict[str, JsonValue] = {"expression": constraint.expression}
    if constraint.description is not None:
        view["description"] = constraint.description
    complement: JsonValue = {
        "identity": identity_of(constraint),
        "context": to_feature_map(constraint.context),
    }
    return view, complement


def _constraint_backward(view: JsonValue, complement: JsonValue) -> Constraint:
    view_obj = j_obj(view)
    comp = j_obj(complement)
    constraint = Constraint(
        expression=j_str(view_obj["expression"]),
        context=from_feature_map(comp["context"]),
        description=j_str_or_none(view_obj.get("description")),
    )
    return apply_identity(constraint, comp["identity"])


def _slot_forward(slot: Slot) -> tuple[JsonValue, JsonValue]:
    view: dict[str, JsonValue] = {"name": slot.name, "required": slot.required}
    if slot.description is not None:
        view["description"] = slot.description
    if slot.default_value is not None:
        view["defaultValue"] = slot.default_value
    constraint_views: list[JsonValue] = []
    constraint_complements: list[JsonValue] = []
    for constraint in slot.constraints:
        constraint_view, constraint_complement = _constraint_forward(constraint)
        constraint_views.append(constraint_view)
        constraint_complements.append(constraint_complement)
    view["constraints"] = tuple(constraint_views)
    complement: JsonValue = {
        "identity": identity_of(slot),
        "constraint_complements": tuple(constraint_complements),
    }
    return view, complement


def _slot_backward(view: JsonValue, complement: JsonValue) -> Slot:
    view_obj = j_obj(view)
    comp = j_obj(complement)
    constraint_views = j_list(view_obj["constraints"])
    constraint_complements = j_list(comp["constraint_complements"])
    constraints = tuple(
        _constraint_backward(constraint_view, constraint_complement)
        for constraint_view, constraint_complement in zip(
            constraint_views, constraint_complements, strict=True
        )
    )
    slot = Slot(
        name=j_str(view_obj["name"]),
        description=j_str_or_none(view_obj.get("description")),
        constraints=constraints,
        required=j_bool(view_obj["required"]),
        default_value=j_str_or_none(view_obj.get("defaultValue")),
    )
    return apply_identity(slot, comp["identity"])


class TemplateLayersLens(dx.Lens[Template, JsonValue, JsonValue]):
    """Lossless lens ``Template <-> (layers template view, complement)``."""

    def forward(self, template: Template) -> tuple[JsonValue, JsonValue]:
        """Project a template to a layers template view and complement."""
        slot_views: dict[str, JsonValue] = {}
        slot_complements: dict[str, JsonValue] = {}
        for slot_key, slot in template.slots.items():
            slot_view, slot_complement = _slot_forward(slot)
            slot_views[slot_key] = slot_view
            slot_complements[slot_key] = slot_complement
        constraint_views: list[JsonValue] = []
        constraint_complements: list[JsonValue] = []
        for constraint in template.constraints:
            constraint_view, constraint_complement = _constraint_forward(constraint)
            constraint_views.append(constraint_view)
            constraint_complements.append(constraint_complement)
        view: dict[str, JsonValue] = {
            "name": template.name,
            "text": template.template_string,
            "slots": slot_views,
            "constraints": tuple(constraint_views),
        }
        if template.language_code is not None:
            view["languages"] = (template.language_code,)
        complement: JsonValue = {
            "identity": identity_of(template),
            "description": template.description,
            "language_code": template.language_code,
            "tags": template.tags,
            "metadata": to_feature_map(template.metadata),
            "slot_order": tuple(template.slots),
            "slot_complements": slot_complements,
            "constraint_complements": tuple(constraint_complements),
        }
        return view, complement

    def backward(self, view: JsonValue, complement: JsonValue) -> Template:
        """Reconstruct a template from its layers template view and complement."""
        view_obj = j_obj(view)
        comp = j_obj(complement)
        slot_views = j_obj(view_obj["slots"])
        slot_complements = j_obj(comp["slot_complements"])
        slots: dict[str, Slot] = {}
        for slot_key_value in j_list(comp["slot_order"]):
            slot_key = j_str(slot_key_value)
            slots[slot_key] = _slot_backward(
                slot_views[slot_key], slot_complements[slot_key]
            )
        constraint_views = j_list(view_obj["constraints"])
        constraint_complements = j_list(comp["constraint_complements"])
        constraints = tuple(
            _constraint_backward(constraint_view, constraint_complement)
            for constraint_view, constraint_complement in zip(
                constraint_views, constraint_complements, strict=True
            )
        )
        template = Template(
            name=j_str(view_obj["name"]),
            template_string=j_str(view_obj["text"]),
            slots=slots,
            constraints=constraints,
            description=j_str_or_none(comp["description"]),
            language_code=j_str_or_none(comp["language_code"]),
            tags=tuple(j_str(tag) for tag in j_list(comp["tags"])),
            metadata=from_feature_map(comp["metadata"]),
        )
        return apply_identity(template, comp["identity"])


TEMPLATE_LAYERS = TemplateLayersLens()
