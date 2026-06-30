"""Template and structure models for sentence generation.

Templates contain slots that are filled with lexical items during sentence
generation. Templates may be combined into sequences or hierarchical
trees.
"""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import didactic.api as dx

from bead.data.base import BeadBaseModel, JsonValue
from bead.data.language_codes import LanguageCode
from bead.resources.constraints import Constraint

if TYPE_CHECKING:
    from bead.templates.filler import FilledTemplate


class Slot(BeadBaseModel):
    """A slot in a template that can be filled with a lexical item.

    Attributes
    ----------
    name : str
        Unique name for the slot within the template.
    description : str | None
        Human-readable description.
    constraints : tuple[Constraint, ...]
        Constraints that determine valid fillers.
    required : bool
        Whether the slot must be filled.
    default_value : str | None
        Default string used if the slot is not filled.
    """

    name: str
    description: str | None = None
    constraints: tuple[dx.Embed[Constraint], ...] = ()
    required: bool = True
    default_value: str | None = None

    @dx.validates("name")
    def _check_name(self, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("name must be non-empty")
        if not value.isidentifier():
            raise ValueError(f"name '{value}' must be a valid Python identifier")
        return value


class Template(BeadBaseModel):
    """A sentence template with named slots.

    Attributes
    ----------
    name : str
        Unique template name.
    template_string : str
        Template body with ``{slot_name}`` placeholders.
    slots : dict[str, Slot]
        Slot definitions keyed by slot name.
    constraints : tuple[Constraint, ...]
        Multi-slot constraints (slot names appear as DSL variables).
    description : str | None
        Human-readable description.
    language_code : LanguageCode | None
        ISO 639-1 or 639-3 language code.
    tags : tuple[str, ...]
        Categorization tags.
    metadata : dict[str, JsonValue]
        Additional metadata.
    """

    name: str
    template_string: str
    slots: dict[str, dx.Embed[Slot]] = dx.field(default_factory=dict)
    constraints: tuple[dx.Embed[Constraint], ...] = ()
    description: str | None = None
    language_code: LanguageCode | None = None
    tags: tuple[str, ...] = ()
    metadata: dict[str, JsonValue] = dx.field(default_factory=dict)

    @dx.validates("name")
    def _check_name(self, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("name must be non-empty")
        return value

    @dx.validates("template_string")
    def _check_template_string(self, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("template_string must be non-empty")
        return value

    @dx.validates("language_code")
    def _check_language_code(self, value: LanguageCode | None) -> LanguageCode | None:
        from bead.data.language_codes import validate_iso639_code  # noqa: PLC0415

        return validate_iso639_code(value)

    @property
    def required_slot_names(self) -> frozenset[str]:
        """Names of all slots flagged as required."""
        return frozenset(name for name, slot in self.slots.items() if slot.required)

    def fill_with_values(
        self, slot_values: dict[str, str], strategy_name: str = "manual"
    ) -> FilledTemplate:
        """Build a ``FilledTemplate`` from a mapping of slot names to strings.

        Each slot value becomes a minimal ``LexicalItem`` whose lemma is
        the supplied string.
        """
        from bead.resources.lexical_item import LexicalItem  # noqa: PLC0415
        from bead.templates.filler import FilledTemplate  # noqa: PLC0415

        slot_fillers: dict[str, LexicalItem] = {}
        for slot_name, value in slot_values.items():
            if slot_name in self.slots:
                slot_fillers[slot_name] = LexicalItem(
                    lemma=value,
                    language_code=self.language_code or "eng",
                    features={"pos": "UNKNOWN"},
                )

        rendered_text = self.template_string
        for slot_name, value in slot_values.items():
            rendered_text = rendered_text.replace(f"{{{slot_name}}}", value)

        template_slots = {name: slot.required for name, slot in self.slots.items()}

        return FilledTemplate(
            template_id=str(self.id),
            template_name=self.name,
            slot_fillers=slot_fillers,
            rendered_text=rendered_text,
            strategy_name=strategy_name,
            template_slots=template_slots,
        )


def slots_match_template(template: Template) -> None:
    """Raise ``ValueError`` if *template*'s slot dict and string disagree.

    Validates that every ``{slot_name}`` placeholder has a matching entry
    in ``slots``, no extraneous slots are defined, and each slot's name
    matches its dict key.
    """
    template_slots = set(re.findall(r"\{(\w+)\}", template.template_string))
    dict_slots = set(template.slots.keys())

    missing_in_dict = template_slots - dict_slots
    if missing_in_dict:
        raise ValueError(
            f"Template references slots not in slots dict: {missing_in_dict}"
        )

    missing_in_template = dict_slots - template_slots
    if missing_in_template:
        raise ValueError(
            f"Slots dict contains slots not referenced in template: "
            f"{missing_in_template}"
        )

    for key, slot in template.slots.items():
        if slot.name != key:
            raise ValueError(f"Slot key '{key}' does not match slot name '{slot.name}'")


class TemplateSequence(BeadBaseModel):
    """A sequence of templates to be filled together.

    Attributes
    ----------
    name : str
        Unique name for the sequence.
    templates : tuple[Template, ...]
        Ordered list of templates.
    constraints : tuple[Constraint, ...]
        Cross-template constraints.
    """

    name: str
    templates: tuple[dx.Embed[Template], ...] = ()
    constraints: tuple[dx.Embed[Constraint], ...] = ()

    @dx.validates("name")
    def _check_name(self, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("name must be non-empty")
        return value


class TemplateTree(BeadBaseModel):
    """A tree of templates, used to model discourse structure.

    Attributes
    ----------
    name : str
        Unique tree name.
    root : Template
        Root template.
    children : tuple[TemplateTree, ...]
        Child subtrees.
    """

    name: str
    root: dx.Embed[Template]
    children: tuple[dx.Embed[TemplateTree], ...] = ()

    @dx.validates("name")
    def _check_name(self, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("name must be non-empty")
        return value
