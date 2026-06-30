"""List collection data model.

The ``ListCollection`` model groups multiple ``ExperimentList`` instances
together with metadata describing the partitioning process that produced
them.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Self, TypedDict
from uuid import UUID

import didactic.api as dx

from bead.data.base import BeadBaseModel
from bead.lists.experiment_list import ExperimentList

type MetadataValue = (
    str
    | int
    | float
    | bool
    | None
    | tuple[MetadataValue, ...]
    | dict[str, MetadataValue]
)


class CoverageValidationResult(TypedDict):
    """Outcome of a coverage check across a ``ListCollection``."""

    valid: bool
    missing_items: list[UUID]
    duplicate_items: list[UUID]
    total_assigned: int


class ListCollection(BeadBaseModel):
    """A collection of experimental lists with partitioning metadata.

    Attributes
    ----------
    name : str
        Collection name.
    source_items_id : UUID
        UUID of the source ``ItemCollection``.
    partitioning_strategy : str
        Strategy name (e.g. ``"balanced"``, ``"random"``).
    lists : tuple[ExperimentList, ...]
        Member experiment lists.
    partitioning_config : dict[str, MetadataValue]
        Configuration for the partitioning process.
    partitioning_stats : dict[str, MetadataValue]
        Statistics from the partitioning process.
    """

    name: str
    source_items_id: UUID
    partitioning_strategy: str
    lists: tuple[dx.Embed[ExperimentList], ...] = ()
    partitioning_config: dict[str, MetadataValue] = dx.field(default_factory=dict)
    partitioning_stats: dict[str, MetadataValue] = dx.field(default_factory=dict)

    @dx.validates("name", "partitioning_strategy")
    def _check_non_empty(self, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Field must be non-empty")
        return value.strip()

    @dx.validates("lists")
    def _check_unique_list_numbers(
        self, value: tuple[ExperimentList, ...]
    ) -> tuple[ExperimentList, ...]:
        if not value:
            return value
        list_numbers = [exp_list.list_number for exp_list in value]
        if len(list_numbers) != len(set(list_numbers)):
            duplicates = {num for num in list_numbers if list_numbers.count(num) > 1}
            raise ValueError(f"Duplicate list_numbers found: {duplicates}")
        return value

    def with_list(self, exp_list: ExperimentList) -> Self:
        """Return a new collection with *exp_list* appended."""
        return self.with_(lists=(*self.lists, exp_list)).touched()

    def get_list_by_number(self, list_number: int) -> ExperimentList | None:
        """Return the list with the matching ``list_number``, or ``None``."""
        for exp_list in self.lists:
            if exp_list.list_number == list_number:
                return exp_list
        return None

    def get_all_item_refs(self) -> tuple[UUID, ...]:
        """Return every distinct item UUID referenced across all lists."""
        all_refs: set[UUID] = set()
        for exp_list in self.lists:
            all_refs.update(exp_list.item_refs)
        return tuple(all_refs)

    def validate_coverage(self, all_item_ids: set[UUID]) -> CoverageValidationResult:
        """Check that every item in *all_item_ids* is assigned exactly once.

        Returns a report with keys ``valid``, ``missing_items``,
        ``duplicate_items``, and ``total_assigned``.
        """
        item_counts: dict[UUID, int] = {}
        for exp_list in self.lists:
            for item_id in exp_list.item_refs:
                item_counts[item_id] = item_counts.get(item_id, 0) + 1

        assigned = set(item_counts.keys())
        missing = list(all_item_ids - assigned)
        duplicates = [item_id for item_id, count in item_counts.items() if count > 1]

        return {
            "valid": not missing and not duplicates,
            "missing_items": missing,
            "duplicate_items": duplicates,
            "total_assigned": sum(item_counts.values()),
        }

    def to_jsonl(self, path: Path | str) -> None:
        """Write each contained list as a JSONL line at *path*."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            for exp_list in self.lists:
                f.write(exp_list.model_dump_json() + "\n")

    @classmethod
    def from_jsonl(
        cls,
        path: Path | str,
        name: str = "loaded_lists",
        source_items_id: UUID | None = None,
        partitioning_strategy: str = "unknown",
    ) -> ListCollection:
        """Build a collection from a JSONL file of experiment lists."""
        path = Path(path)
        lists: list[ExperimentList] = []
        with path.open(encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                lists.append(ExperimentList(**json.loads(line)))

        return cls(
            name=name,
            source_items_id=source_items_id or UUID(int=0),
            lists=tuple(lists),
            partitioning_strategy=partitioning_strategy,
        )
