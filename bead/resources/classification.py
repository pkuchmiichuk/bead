"""Linguistic classification models for lexical items and templates.

Models for grouping lexical items and templates by linguistic properties.
``LexicalItemClass`` and ``TemplateClass`` support cross-linguistic analysis
and alignment.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Self
from uuid import UUID

import didactic.api as dx

from bead.data.base import BeadBaseModel, JsonValue
from bead.data.language_codes import validate_iso639_code
from bead.resources.lexical_item import LexicalItem
from bead.resources.template import Template


class LexicalItemClass(BeadBaseModel):
    """A group of lexical items sharing a linguistic property.

    Items are stored as a tuple in insertion order; mutating methods
    (``with_item``, ``without_item``) return new instances.

    Attributes
    ----------
    name : str
        Class name.
    description : str | None
        Description of the classification.
    property_name : str
        The linguistic property defining the class.
    property_value : JsonValue
        Specific value of the property.
    items : tuple[LexicalItem, ...]
        Member items in insertion order.
    tags : tuple[str, ...]
        Categorization tags.
    class_metadata : dict[str, JsonValue]
        Additional metadata.
    """

    name: str
    property_name: str
    description: str | None = None
    property_value: JsonValue = None
    items: tuple[dx.Embed[LexicalItem], ...] = ()
    tags: tuple[str, ...] = ()
    class_metadata: dict[str, JsonValue] = dx.field(default_factory=dict)

    @dx.validates("name")
    def _check_name(self, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("name must be non-empty")
        return value

    @dx.validates("property_name")
    def _check_property_name(self, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("property_name must be non-empty")
        return value

    def __len__(self) -> int:
        """Return the number of items in the class."""
        return len(self.items)

    def __contains__(self, item_id: UUID) -> bool:
        """Return whether an item with *item_id* is present."""
        return any(item.id == item_id for item in self.items)

    def __iter__(self) -> Iterator[LexicalItem]:
        """Iterate over class members."""
        return iter(self.items)

    def by_id(self, item_id: UUID) -> LexicalItem | None:
        """Return the item with the matching id, or ``None``."""
        for item in self.items:
            if item.id == item_id:
                return item
        return None

    def with_item(self, item: LexicalItem) -> Self:
        """Return a new class with *item* appended."""
        if any(existing.id == item.id for existing in self.items):
            raise ValueError(f"Item with ID {item.id} already exists in class")
        return self.with_(items=(*self.items, item)).touched()

    def without_item(self, item_id: UUID) -> tuple[Self, LexicalItem]:
        """Return ``(new_class, removed_item)``."""
        for index, item in enumerate(self.items):
            if item.id == item_id:
                remaining = self.items[:index] + self.items[index + 1 :]
                return self.with_(items=remaining).touched(), item
        raise KeyError(f"Item with ID {item_id} not found in class")

    def languages(self) -> frozenset[str]:
        """Return the set of language codes (lowercased) present in the class."""
        return frozenset(
            item.language_code.lower()
            for item in self.items
            if item.language_code is not None
        )

    def get_items_by_language(self, language_code: str) -> tuple[LexicalItem, ...]:
        """Return items whose language code matches *language_code*.

        Codes are normalized via ``validate_iso639_code`` before comparison.
        """
        try:
            normalized = validate_iso639_code(language_code)
        except ValueError:
            return ()
        if normalized is None:
            return ()
        target = normalized.lower()
        return tuple(
            item
            for item in self.items
            if item.language_code is not None and item.language_code.lower() == target
        )

    def is_monolingual(self) -> bool:
        """Return whether the class spans at most one language."""
        return len(self.languages()) <= 1

    def is_multilingual(self) -> bool:
        """Return whether the class spans more than one language."""
        return len(self.languages()) > 1


class TemplateClass(BeadBaseModel):
    """A group of templates sharing a linguistic property.

    Attributes
    ----------
    name : str
        Class name.
    description : str | None
        Description.
    property_name : str
        Defining linguistic property.
    property_value : JsonValue
        Specific property value.
    templates : tuple[Template, ...]
        Member templates in insertion order.
    tags : tuple[str, ...]
        Categorization tags.
    class_metadata : dict[str, JsonValue]
        Additional metadata.
    """

    name: str
    property_name: str
    description: str | None = None
    property_value: JsonValue = None
    templates: tuple[dx.Embed[Template], ...] = ()
    tags: tuple[str, ...] = ()
    class_metadata: dict[str, JsonValue] = dx.field(default_factory=dict)

    @dx.validates("name")
    def _check_name(self, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("name must be non-empty")
        return value

    @dx.validates("property_name")
    def _check_property_name(self, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("property_name must be non-empty")
        return value

    def __len__(self) -> int:
        """Return the number of templates in the class."""
        return len(self.templates)

    def __contains__(self, template_id: UUID) -> bool:
        """Return whether a template with *template_id* is present."""
        return any(template.id == template_id for template in self.templates)

    def __iter__(self) -> Iterator[Template]:
        """Iterate over the templates in the class."""
        return iter(self.templates)

    def by_id(self, template_id: UUID) -> Template | None:
        """Return the template with the matching id, or ``None``."""
        for template in self.templates:
            if template.id == template_id:
                return template
        return None

    def with_template(self, template: Template) -> Self:
        """Return a new class with *template* appended."""
        if any(existing.id == template.id for existing in self.templates):
            raise ValueError(f"Template with ID {template.id} already exists in class")
        return self.with_(templates=(*self.templates, template)).touched()

    def without_template(self, template_id: UUID) -> tuple[Self, Template]:
        """Return ``(new_class, removed_template)``."""
        for index, template in enumerate(self.templates):
            if template.id == template_id:
                remaining = self.templates[:index] + self.templates[index + 1 :]
                return self.with_(templates=remaining).touched(), template
        raise KeyError(f"Template with ID {template_id} not found in class")

    def languages(self) -> frozenset[str]:
        """Return the set of language codes (lowercased) in the class."""
        return frozenset(
            template.language_code.lower()
            for template in self.templates
            if template.language_code is not None
        )

    def get_templates_by_language(self, language_code: str) -> tuple[Template, ...]:
        """Return templates whose language code matches *language_code*."""
        try:
            normalized = validate_iso639_code(language_code)
        except ValueError:
            return ()
        if normalized is None:
            return ()
        target = normalized.lower()
        return tuple(
            template
            for template in self.templates
            if template.language_code is not None
            and template.language_code.lower() == target
        )

    def is_monolingual(self) -> bool:
        """Return whether the class spans at most one language."""
        return len(self.languages()) <= 1

    def is_multilingual(self) -> bool:
        """Return whether the class spans more than one language."""
        return len(self.languages()) > 1
