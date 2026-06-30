"""Data models for constructed experimental items."""

from __future__ import annotations

from typing import Self
from uuid import UUID

import didactic.api as dx

from bead.data.base import BeadBaseModel
from bead.items.spans import Span, SpanRelation

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
    """Whether a single constraint is satisfied for an item.

    Attributes
    ----------
    constraint_id : UUID
        UUID of the constraint.
    satisfied : bool
        Whether the constraint is satisfied.
    """

    constraint_id: UUID
    satisfied: bool


class UnfilledSlot(BeadBaseModel):
    """An unfilled slot in a cloze task item.

    Attributes
    ----------
    slot_name : str
        Name of the unfilled template slot.
    position : int
        Token index position in the rendered text.
    constraint_ids : tuple[UUID, ...]
        UUIDs of constraints that apply to this slot.
    """

    slot_name: str
    position: int
    constraint_ids: tuple[UUID, ...] = ()

    @dx.validates("slot_name")
    def _check_slot_name(self, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Slot name cannot be empty")
        return value.strip()


class ModelOutput(BeadBaseModel):
    """Output from a model computation.

    Attributes
    ----------
    model_name : str
        Name/identifier of the model.
    model_version : str
        Version of the model.
    operation : str
        Operation performed (e.g. "log_probability", "nli", "embedding").
    inputs : dict[str, MetadataValue]
        Inputs to the model.
    output : MetadataValue
        Model output.
    cache_key : str
        Cache key for this computation.
    computation_metadata : dict[str, MetadataValue]
        Metadata about the computation (timestamp, device, etc.).
    """

    model_name: str
    model_version: str
    operation: str
    inputs: dict[str, MetadataValue]
    output: MetadataValue
    cache_key: str
    computation_metadata: dict[str, MetadataValue] = dx.field(default_factory=dict)

    @dx.validates("model_name", "model_version", "operation", "cache_key")
    def _check_non_empty(self, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Field cannot be empty")
        return value.strip()


class Item(BeadBaseModel):
    """A constructed experimental item.

    Attributes
    ----------
    item_template_id : UUID
        UUID of the item template this was constructed from.
    filled_template_refs : tuple[UUID, ...]
        UUIDs of filled templates used in this item.
    rendered_elements : dict[str, str]
        Rendered text for each element.
    options : tuple[str, ...]
        Choice options for forced_choice/multi_select tasks.
    unfilled_slots : tuple[UnfilledSlot, ...]
        Unfilled slots for cloze tasks.
    model_outputs : tuple[ModelOutput, ...]
        Model computations for this item.
    constraint_satisfaction : tuple[ConstraintSatisfaction, ...]
        Per-constraint satisfaction records.
    item_metadata : dict[str, MetadataValue]
        Additional metadata.
    spans : tuple[Span, ...]
        Span annotations.
    span_relations : tuple[SpanRelation, ...]
        Relations between spans.
    tokenized_elements : dict[str, tuple[str, ...]]
        Tokenized text for span indexing.
    token_space_after : dict[str, tuple[bool, ...]]
        Per-token space_after flags for artifact-free rendering.
    """

    item_template_id: UUID
    filled_template_refs: tuple[UUID, ...] = ()
    rendered_elements: dict[str, str] = dx.field(default_factory=dict)
    options: tuple[str, ...] = ()
    unfilled_slots: tuple[dx.Embed[UnfilledSlot], ...] = ()
    model_outputs: tuple[dx.Embed[ModelOutput], ...] = ()
    constraint_satisfaction: tuple[dx.Embed[ConstraintSatisfaction], ...] = ()
    item_metadata: dict[str, MetadataValue] = dx.field(default_factory=dict)
    spans: tuple[dx.Embed[Span], ...] = ()
    span_relations: tuple[dx.Embed[SpanRelation], ...] = ()
    tokenized_elements: dict[str, tuple[str, ...]] = dx.field(default_factory=dict)
    token_space_after: dict[str, tuple[bool, ...]] = dx.field(default_factory=dict)

    def get_model_output(
        self,
        model_name: str,
        operation: str,
        inputs: dict[str, MetadataValue] | None = None,
    ) -> ModelOutput | None:
        """Return the matching ModelOutput, or ``None`` if absent."""
        for output in self.model_outputs:
            if output.model_name == model_name and output.operation == operation:
                if inputs is None or output.inputs == inputs:
                    return output
        return None

    def with_model_output(self, output: ModelOutput) -> Self:
        """Return a new Item with *output* appended to ``model_outputs``."""
        return self.with_(model_outputs=(*self.model_outputs, output)).touched()


class ItemCollection(BeadBaseModel):
    """A collection of constructed items.

    Attributes
    ----------
    name : str
        Name of this collection.
    source_template_collection_id : UUID
        UUID of the source item template collection.
    source_filled_collection_id : UUID
        UUID of the source filled template collection.
    items : tuple[Item, ...]
        The constructed items.
    construction_stats : dict[str, int]
        Statistics about item construction.
    """

    name: str
    source_template_collection_id: UUID
    source_filled_collection_id: UUID
    items: tuple[dx.Embed[Item], ...] = ()
    construction_stats: dict[str, int] = dx.field(default_factory=dict)

    @dx.validates("name")
    def _check_name(self, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Collection name cannot be empty")
        return value.strip()

    def with_item(self, item: Item) -> Self:
        """Return a new collection with *item* appended."""
        return self.with_(items=(*self.items, item)).touched()
