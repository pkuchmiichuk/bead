"""Template collection management.

The ``TemplateCollection`` class manages collections of sentence templates.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator
from pathlib import Path
from typing import Literal, Self
from uuid import UUID

import didactic.api as dx
import pandas as pd
import polars as pl

from bead.data.base import BeadBaseModel, JsonValue
from bead.resources.template import Template

DataFrame = pd.DataFrame | pl.DataFrame


class TemplateCollection(BeadBaseModel):
    """A collection of templates supporting filtering, search, and merging.

    Templates are stored as a tuple in insertion order; mutating methods
    (``with_template``, ``without_template``, ``with_templates``) return new
    instances.

    Attributes
    ----------
    name : str
        Collection name.
    description : str | None
        Optional description.
    language_code : str | None
        ISO 639-1 or 639-3 language code.
    templates : tuple[Template, ...]
        Templates in insertion order.
    tags : tuple[str, ...]
        Categorization tags.
    """

    name: str
    description: str | None = None
    language_code: str | None = None
    templates: tuple[dx.Embed[Template], ...] = ()
    tags: tuple[str, ...] = ()

    def __len__(self) -> int:
        """Return the number of templates in the collection."""
        return len(self.templates)

    def __iter__(self) -> Iterator[Template]:
        """Iterate over the templates."""
        return iter(self.templates)

    def __contains__(self, template_id: UUID) -> bool:
        """Return whether a template with *template_id* is present."""
        return any(template.id == template_id for template in self.templates)

    def by_id(self, template_id: UUID) -> Template | None:
        """Return the template with the matching id, or ``None``."""
        for template in self.templates:
            if template.id == template_id:
                return template
        return None

    def with_template(self, template: Template) -> Self:
        """Return a new collection with *template* appended."""
        if any(existing.id == template.id for existing in self.templates):
            raise ValueError(
                f"Template with ID {template.id} already exists in collection"
            )
        return self.with_(templates=(*self.templates, template)).touched()

    def with_templates(self, templates: tuple[Template, ...] | list[Template]) -> Self:
        """Return a new collection with each template appended."""
        existing_ids = {template.id for template in self.templates}
        for template in templates:
            if template.id in existing_ids:
                raise ValueError(
                    f"Template with ID {template.id} already exists in collection"
                )
            existing_ids.add(template.id)
        return self.with_(templates=(*self.templates, *templates)).touched()

    def without_template(self, template_id: UUID) -> tuple[Self, Template]:
        """Return ``(new_collection, removed_template)``."""
        for index, template in enumerate(self.templates):
            if template.id == template_id:
                remaining = self.templates[:index] + self.templates[index + 1 :]
                return self.with_(templates=remaining).touched(), template
        raise KeyError(f"Template with ID {template_id} not found in collection")

    def filter(self, predicate: Callable[[Template], bool]) -> Self:
        """Return a new collection containing only templates matching *predicate*."""
        return self.with_(
            templates=tuple(t for t in self.templates if predicate(t)),
        )

    def filter_by_tag(self, tag: str) -> Self:
        """Return a new collection of templates carrying *tag*."""
        return self.filter(lambda template: tag in template.tags)

    def filter_by_slot_count(self, count: int) -> Self:
        """Return a new collection of templates with exactly *count* slots."""
        return self.filter(lambda template: len(template.slots) == count)

    def search(self, query: str, field: str = "name") -> Self:
        """Return a new collection of templates matching *query* in *field*.

        Parameters
        ----------
        query
            Substring to search for (case-insensitive).
        field
            One of ``"name"`` or ``"template_string"``.

        Raises
        ------
        ValueError
            If *field* is not a supported name.
        """
        q = query.lower()
        if field == "name":
            return self.filter(lambda template: q in template.name.lower())
        if field == "template_string":
            return self.filter(lambda template: q in template.template_string.lower())
        raise ValueError(
            f"Invalid field '{field}'. Must be 'name' or 'template_string'."
        )

    def merge(
        self,
        other: TemplateCollection,
        strategy: Literal["keep_first", "keep_second", "error"] = "keep_first",
    ) -> TemplateCollection:
        """Combine *self* and *other* into a new collection.

        Parameters
        ----------
        other
            Collection to merge with.
        strategy
            Conflict policy when templates share an id.

        Raises
        ------
        ValueError
            If ``strategy="error"`` and any duplicate ids are present.
        """
        self_ids = {template.id for template in self.templates}
        other_ids = {template.id for template in other.templates}
        duplicates = self_ids & other_ids
        if strategy == "error" and duplicates:
            raise ValueError(
                f"Duplicate template IDs found: {duplicates}. "
                "Use strategy='keep_first' or 'keep_second' to resolve."
            )

        if strategy == "keep_first":
            kept_self = self.templates
            kept_other = tuple(t for t in other.templates if t.id not in self_ids)
        else:
            kept_self = tuple(t for t in self.templates if t.id not in other_ids)
            kept_other = other.templates

        return TemplateCollection(
            name=f"{self.name}_merged",
            description=self.description,
            language_code=self.language_code or other.language_code,
            templates=(*kept_self, *kept_other),
            tags=tuple(sorted(set(self.tags) | set(other.tags))),
        )

    def to_dataframe(
        self, backend: Literal["pandas", "polars"] = "pandas"
    ) -> DataFrame:
        """Render the collection as a pandas or polars DataFrame.

        Columns are ``id``, ``name``, ``template_string``, ``description``,
        ``slot_count``, ``slot_names`` (comma-joined), ``tags``,
        ``created_at``, ``modified_at``.
        """
        if not self.templates:
            columns = [
                "id",
                "name",
                "template_string",
                "description",
                "slot_count",
                "slot_names",
                "tags",
                "created_at",
                "modified_at",
            ]
            if backend == "pandas":
                return pd.DataFrame(columns=columns)
            schema: dict[str, type[pl.Utf8]] = dict.fromkeys(columns, pl.Utf8)
            return pl.DataFrame(schema=schema)

        rows: list[dict[str, JsonValue]] = []
        for template in self.templates:
            rows.append(
                {
                    "id": str(template.id),
                    "name": template.name,
                    "template_string": template.template_string,
                    "description": template.description,
                    "slot_count": len(template.slots),
                    "slot_names": ",".join(sorted(template.slots.keys())),
                    "tags": ",".join(template.tags),
                    "created_at": template.created_at.isoformat(),
                    "modified_at": template.modified_at.isoformat(),
                }
            )

        if backend == "pandas":
            return pd.DataFrame(rows)
        return pl.DataFrame(rows)

    @classmethod
    def from_dataframe(cls, df: DataFrame, name: str) -> TemplateCollection:
        """Build an empty collection bound to *name*.

        DataFrame ingestion of ``Template`` objects requires their slot
        definitions, which are not present in tabular form. Use
        :meth:`from_jsonl` for full deserialization.
        """
        is_polars = isinstance(df, pl.DataFrame)
        if is_polars:
            assert isinstance(df, pl.DataFrame)
            columns_list: list[str] = df.columns
        else:
            assert isinstance(df, pd.DataFrame)
            columns_list = list(df.columns)

        if "name" not in columns_list or "template_string" not in columns_list:
            raise ValueError("DataFrame must have 'name' and 'template_string' columns")
        return cls(name=name)

    def to_jsonl(self, path: str) -> None:
        """Write the collection as JSONLines, one ``Template`` per line."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as f:
            for template in self.templates:
                f.write(template.model_dump_json() + "\n")

    @classmethod
    def from_jsonl(cls, path: str, name: str) -> TemplateCollection:
        """Read a JSONLines file and return a collection."""
        templates: list[Template] = []
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    templates.append(Template.model_validate_json(line))
        return cls(name=name, templates=tuple(templates))
