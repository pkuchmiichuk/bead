"""Experiment list data model for organizing experimental items.

The ``ExperimentList`` model uses stand-off annotation: it stores only
item UUIDs, not full ``Item`` objects. Items are looked up by UUID
against an ``ItemCollection`` or ``Repository``.
"""

from __future__ import annotations

import random
from typing import Self
from uuid import UUID

import didactic.api as dx

from bead.data.base import BeadBaseModel
from bead.lists.constraints import ListConstraint

type MetadataValue = (
    str
    | int
    | float
    | bool
    | None
    | tuple[MetadataValue, ...]
    | dict[str, MetadataValue]
)


class ConstraintSatisfaction(BeadBaseModel):
    """Whether a single constraint is satisfied for the list.

    Attributes
    ----------
    constraint_id : UUID
        UUID of the constraint.
    satisfied : bool
        Whether the constraint holds.
    """

    constraint_id: UUID
    satisfied: bool


class ExperimentList(BeadBaseModel):
    """A list of experimental items selected for participant presentation.

    Attributes
    ----------
    name : str
        List name (e.g. ``"list_0"``, ``"practice_list"``).
    list_number : int
        Numeric identifier (>= 0).
    item_refs : tuple[UUID, ...]
        UUIDs of the items in this list, in insertion order.
    list_constraints : tuple[ListConstraint, ...]
        Constraints the list must satisfy.
    constraint_satisfaction : tuple[ConstraintSatisfaction, ...]
        Per-constraint satisfaction records.
    presentation_order : tuple[UUID, ...] | None
        Explicit presentation order; ``None`` falls back to ``item_refs``.
    list_metadata : dict[str, MetadataValue]
        Metadata for this list.
    balance_metrics : dict[str, MetadataValue]
        Metrics about list balance.
    """

    name: str
    list_number: int
    item_refs: tuple[UUID, ...] = ()
    list_constraints: tuple[dx.Embed[ListConstraint], ...] = ()
    constraint_satisfaction: tuple[dx.Embed[ConstraintSatisfaction], ...] = ()
    presentation_order: tuple[UUID, ...] | None = None
    list_metadata: dict[str, MetadataValue] = dx.field(default_factory=dict)
    balance_metrics: dict[str, MetadataValue] = dx.field(default_factory=dict)

    __axioms__ = (
        dx.axiom(
            "list_number >= 0",
            message="list_number must be non-negative",
        ),
    )

    @dx.validates("name")
    def _check_name(self, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("name must be non-empty")
        return value.strip()

    def with_item(self, item_id: UUID) -> Self:
        """Return a new list with *item_id* appended."""
        return self.with_(item_refs=(*self.item_refs, item_id)).touched()

    def without_item(self, item_id: UUID) -> Self:
        """Return a new list with *item_id* removed.

        Raises
        ------
        ValueError
            If *item_id* is not in the list.
        """
        if item_id not in self.item_refs:
            raise ValueError(f"Item {item_id} not found in list")
        new_refs = tuple(ref for ref in self.item_refs if ref != item_id)
        new_order: tuple[UUID, ...] | None = None
        if self.presentation_order is not None:
            new_order = tuple(ref for ref in self.presentation_order if ref != item_id)
        return self.with_(item_refs=new_refs, presentation_order=new_order).touched()

    def with_shuffled_order(self, seed: int | None = None) -> Self:
        """Return a new list whose ``presentation_order`` is a shuffle of items."""
        rng = random.Random(seed)
        order = list(self.item_refs)
        rng.shuffle(order)
        return self.with_(presentation_order=tuple(order)).touched()

    def get_presentation_order(self) -> tuple[UUID, ...]:
        """Return ``presentation_order`` if set, else ``item_refs``."""
        return (
            self.presentation_order
            if self.presentation_order is not None
            else self.item_refs
        )


def validate_presentation_order(experiment_list: ExperimentList) -> None:
    """Raise ``ValueError`` if ``presentation_order`` and ``item_refs`` disagree.

    The order must be a permutation of the item refs (no missing, extra,
    or duplicated UUIDs).
    """
    order = experiment_list.presentation_order
    if order is None:
        return

    if len(order) != len(set(order)):
        raise ValueError("presentation_order contains duplicate UUIDs")

    item_set = set(experiment_list.item_refs)
    order_set = set(order)
    if order_set != item_set:
        extra = order_set - item_set
        missing = item_set - order_set
        parts: list[str] = []
        if extra:
            parts.append(f"extra UUIDs: {extra}")
        if missing:
            parts.append(f"missing UUIDs: {missing}")
        raise ValueError(
            "presentation_order must contain exactly the same UUIDs as "
            f"item_refs ({', '.join(parts)})"
        )
