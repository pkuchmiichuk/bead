"""Lexicon management for collections of lexical items.

Provides the ``Lexicon`` class for managing, querying, and manipulating
collections of lexical items. Supports filtering, searching, merging, and
conversion to and from pandas / polars DataFrames.
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
from bead.data.language_codes import LanguageCode
from bead.resources.lexical_item import LexicalItem

DataFrame = pd.DataFrame | pl.DataFrame


class Lexicon(BeadBaseModel):
    """A collection of lexical items keyed by their UUIDs.

    Items are stored as a tuple; ``by_id`` provides O(n) lookup. Mutating
    methods (``with_item``, ``without_item``, ``with_items``) return new
    instances.

    Attributes
    ----------
    name : str
        Name of the lexicon.
    description : str | None
        Optional description.
    language_code : LanguageCode | None
        ISO 639-1 or ISO 639-3 language code.
    items : tuple[LexicalItem, ...]
        Items in insertion order.
    tags : tuple[str, ...]
        Categorization tags.
    """

    name: str
    description: str | None = None
    language_code: LanguageCode | None = None
    items: tuple[dx.Embed[LexicalItem], ...] = ()
    tags: tuple[str, ...] = ()

    @dx.validates("language_code")
    def _check_language_code(self, value: LanguageCode | None) -> LanguageCode | None:
        from bead.data.language_codes import validate_iso639_code  # noqa: PLC0415

        return validate_iso639_code(value)

    def __len__(self) -> int:
        """Return the number of items in the lexicon."""
        return len(self.items)

    def __iter__(self) -> Iterator[LexicalItem]:
        """Iterate over the lexicon's items."""
        return iter(self.items)

    def __contains__(self, item_id: UUID) -> bool:
        """Return whether *item_id* is present."""
        return any(item.id == item_id for item in self.items)

    def by_id(self, item_id: UUID) -> LexicalItem | None:
        """Return the item with the matching UUID, or ``None``."""
        for item in self.items:
            if item.id == item_id:
                return item
        return None

    def with_item(self, item: LexicalItem) -> Self:
        """Return a new lexicon with *item* appended.

        Raises
        ------
        ValueError
            If an item with the same id already exists.
        """
        if any(existing.id == item.id for existing in self.items):
            raise ValueError(f"Item with ID {item.id} already exists in lexicon")
        return self.with_(items=(*self.items, item)).touched()

    def with_items(self, items: tuple[LexicalItem, ...] | list[LexicalItem]) -> Self:
        """Return a new lexicon with each of *items* appended."""
        existing_ids = {item.id for item in self.items}
        for item in items:
            if item.id in existing_ids:
                raise ValueError(f"Item with ID {item.id} already exists in lexicon")
            existing_ids.add(item.id)
        return self.with_(items=(*self.items, *items)).touched()

    def without_item(self, item_id: UUID) -> tuple[Self, LexicalItem]:
        """Return ``(new_lexicon, removed_item)`` with *item_id* removed.

        Raises
        ------
        KeyError
            If *item_id* is not present.
        """
        for index, item in enumerate(self.items):
            if item.id == item_id:
                remaining = self.items[:index] + self.items[index + 1 :]
                return self.with_(items=remaining).touched(), item
        raise KeyError(f"Item with ID {item_id} not found in lexicon")

    def filter(self, predicate: Callable[[LexicalItem], bool]) -> Self:
        """Return a new lexicon containing only items satisfying *predicate*."""
        return self.with_(
            items=tuple(item for item in self.items if predicate(item)),
        )

    def filter_by_pos(self, pos: str) -> Self:
        """Return items whose ``features['pos']`` equals *pos*."""
        return self.filter(lambda item: item.features.get("pos") == pos)

    def filter_by_lemma(self, lemma: str) -> Self:
        """Return items whose lemma equals *lemma*."""
        return self.filter(lambda item: item.lemma == lemma)

    def filter_by_feature(self, feature_name: str, feature_value: JsonValue) -> Self:
        """Return items whose feature equals *feature_value*."""
        return self.filter(
            lambda item: item.features.get(feature_name) == feature_value,
        )

    def filter_by_attribute(self, attr_name: str, attr_value: JsonValue) -> Self:
        """Alias for :meth:`filter_by_feature`."""
        return self.filter_by_feature(attr_name, attr_value)

    def search(self, query: str, field: str = "lemma") -> Self:
        """Return a new lexicon with case-insensitive substring matches on *field*.

        Parameters
        ----------
        query
            Substring to look for.
        field
            One of ``"lemma"``, ``"pos"``, or ``"form"``.

        Raises
        ------
        ValueError
            If *field* is not one of the supported names.
        """
        q = query.lower()
        if field == "lemma":
            return self.filter(lambda item: q in item.lemma.lower())
        if field == "pos":
            return self.filter(
                lambda item: q in str(item.features.get("pos", "")).lower(),
            )
        if field == "form":
            return self.filter(
                lambda item: item.form is not None and q in item.form.lower(),
            )
        raise ValueError(f"Invalid field '{field}'. Must be 'lemma', 'pos', or 'form'.")

    def merge(
        self,
        other: Lexicon,
        strategy: Literal["keep_first", "keep_second", "error"] = "keep_first",
    ) -> Lexicon:
        """Combine *self* and *other* into a new lexicon.

        Parameters
        ----------
        other
            Lexicon to merge into *self*.
        strategy
            Conflict policy when items share an id.

        Raises
        ------
        ValueError
            If ``strategy="error"`` and any duplicate ids are present.
        """
        self_ids = {item.id for item in self.items}
        other_ids = {item.id for item in other.items}
        duplicates = self_ids & other_ids

        if strategy == "error" and duplicates:
            raise ValueError(
                f"Duplicate item IDs found: {duplicates}. "
                "Use strategy='keep_first' or 'keep_second' to resolve."
            )

        if strategy == "keep_first":
            kept_self = self.items
            kept_other = tuple(item for item in other.items if item.id not in self_ids)
        else:
            kept_self = tuple(item for item in self.items if item.id not in other_ids)
            kept_other = other.items

        return Lexicon(
            name=f"{self.name}_merged",
            description=self.description,
            language_code=self.language_code or other.language_code,
            items=(*kept_self, *kept_other),
            tags=tuple(sorted(set(self.tags) | set(other.tags))),
        )

    def to_dataframe(
        self, backend: Literal["pandas", "polars"] = "pandas"
    ) -> DataFrame:
        """Render the lexicon as a pandas or polars DataFrame.

        Columns include ``id``, ``lemma``, ``form``, ``language_code``,
        ``source``, ``created_at``, ``modified_at``, plus a
        ``feature_<name>`` column for every feature key seen across all
        items.
        """
        if not self.items:
            columns = [
                "id",
                "lemma",
                "form",
                "language_code",
                "source",
                "created_at",
                "modified_at",
            ]
            if backend == "pandas":
                return pd.DataFrame(columns=columns)
            schema: dict[str, type[pl.Utf8]] = dict.fromkeys(columns, pl.Utf8)
            return pl.DataFrame(schema=schema)

        rows: list[dict[str, JsonValue]] = []
        for item in self.items:
            row: dict[str, JsonValue] = {
                "id": str(item.id),
                "lemma": item.lemma,
                "form": item.form,
                "language_code": item.language_code,
                "source": item.source,
                "created_at": item.created_at.isoformat(),
                "modified_at": item.modified_at.isoformat(),
            }
            for key, value in item.features.items():
                row[f"feature_{key}"] = value
            rows.append(row)

        if backend == "pandas":
            return pd.DataFrame(rows)
        return pl.DataFrame(rows)

    @classmethod
    def from_dataframe(cls, df: DataFrame, name: str) -> Lexicon:
        """Build a lexicon from a pandas or polars DataFrame.

        The DataFrame must have a ``lemma`` column. Columns named ``pos``,
        ``feature_<name>``, or ``attr_<name>`` populate each item's
        ``features`` dict; ``language_code``, ``form``, and ``source``
        populate the corresponding fields.
        """
        is_polars = isinstance(df, pl.DataFrame)
        if is_polars:
            assert isinstance(df, pl.DataFrame)
            columns_list: list[str] = df.columns
            polars_rows = df.to_dicts()
            rows: list[dict[str, JsonValue]] = [dict(r) for r in polars_rows]
        else:
            assert isinstance(df, pd.DataFrame)
            columns_list = list(df.columns)
            pandas_rows = df.to_dict(orient="records")
            rows = [{str(k): v for k, v in r.items()} for r in pandas_rows]

        if "lemma" not in columns_list:
            raise ValueError("DataFrame must have a 'lemma' column")

        def is_not_null(value: object) -> bool:
            if value is None:
                return False
            if is_polars:
                return True
            if isinstance(value, float):
                return value == value  # NaN is the only float != itself
            return True

        items: list[LexicalItem] = []
        for row in rows:
            item_data: dict[str, JsonValue] = {"lemma": row["lemma"]}
            item_data["language_code"] = (
                row["language_code"]
                if "language_code" in row and is_not_null(row["language_code"])
                else "eng"
            )
            if "form" in row and is_not_null(row["form"]):
                item_data["form"] = row["form"]
            if "source" in row and is_not_null(row["source"]):
                item_data["source"] = row["source"]

            features: dict[str, JsonValue] = {}
            if "pos" in row and is_not_null(row["pos"]):
                features["pos"] = row["pos"]
            for col in columns_list:
                if col.startswith("feature_") and is_not_null(row[col]):
                    features[col[len("feature_") :]] = row[col]
                elif col.startswith("attr_") and is_not_null(row[col]):
                    features[col[len("attr_") :]] = row[col]
            if features:
                item_data["features"] = features

            items.append(LexicalItem(**item_data))

        return cls(name=name, items=tuple(items))

    def to_jsonl(self, path: str) -> None:
        """Write the lexicon as JSONLines, one ``LexicalItem`` per line."""
        file_path = Path(path)
        file_path.parent.mkdir(parents=True, exist_ok=True)
        with file_path.open("w", encoding="utf-8") as f:
            for item in self.items:
                f.write(item.model_dump_json() + "\n")

    @classmethod
    def from_jsonl(cls, path: str, name: str) -> Lexicon:
        """Read a JSONLines file and return a new lexicon."""
        items: list[LexicalItem] = []
        with Path(path).open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    items.append(LexicalItem.model_validate_json(line))
        return cls(name=name, items=tuple(items))
