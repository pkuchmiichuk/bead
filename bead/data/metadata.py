"""Metadata tracking models for provenance and processing history.

Tracks provenance chains and processing history for full data lineage.
Models are frozen; updates return new instances through pure ``with_*``
methods.
"""

from __future__ import annotations

from datetime import datetime
from typing import Self
from uuid import UUID

import didactic.api as dx

from bead.data.base import BeadBaseModel, JsonValue
from bead.data.timestamps import now_iso8601


class ProvenanceRecord(BeadBaseModel):
    """A single parent-child relationship in a provenance chain.

    Attributes
    ----------
    parent_id : UUID
        UUID of the parent object.
    parent_type : str
        Type name of the parent object (e.g. "LexicalItem").
    relationship : str
        Nature of the relationship (e.g. "derived_from").
    timestamp : datetime
        When this relationship was established.
    """

    parent_id: UUID
    parent_type: str
    relationship: str
    timestamp: datetime = dx.field(default_factory=now_iso8601)


class ProcessingRecord(BeadBaseModel):
    """A single processing operation in an object's history.

    Attributes
    ----------
    operation : str
        Name of the operation.
    parameters : dict[str, JsonValue]
        Parameters passed to the operation.
    timestamp : datetime
        When the operation was performed.
    operator : str | None
        Identity of the agent that performed the operation.
    """

    operation: str
    parameters: dict[str, JsonValue] = dx.field(default_factory=dict)
    timestamp: datetime = dx.field(default_factory=now_iso8601)
    operator: str | None = None


class MetadataTracker(BeadBaseModel):
    """Frozen tracker for provenance and processing history.

    Attributes
    ----------
    provenance : tuple[ProvenanceRecord, ...]
        Provenance relationships in insertion order.
    processing_history : tuple[ProcessingRecord, ...]
        Processing operations in chronological order.
    custom_metadata : dict[str, JsonValue]
        Custom annotations.

    Examples
    --------
    >>> from uuid import uuid4
    >>> tracker = MetadataTracker()
    >>> parent_id = uuid4()
    >>> tracker = tracker.with_provenance(parent_id, "Template", "filled_from")
    >>> tracker = tracker.with_processing("fill_template", {"strategy": "exhaustive"})
    >>> len(tracker.provenance)
    1
    >>> len(tracker.processing_history)
    1
    """

    provenance: tuple[dx.Embed[ProvenanceRecord], ...] = ()
    processing_history: tuple[dx.Embed[ProcessingRecord], ...] = ()
    custom_metadata: dict[str, JsonValue] = dx.field(default_factory=dict)

    def with_provenance(
        self, parent_id: UUID, parent_type: str, relationship: str
    ) -> Self:
        """Return a new tracker with one additional provenance record."""
        record = ProvenanceRecord(
            parent_id=parent_id,
            parent_type=parent_type,
            relationship=relationship,
        )
        return self.with_(provenance=(*self.provenance, record))

    def with_processing(
        self,
        operation: str,
        parameters: dict[str, JsonValue] | None = None,
        operator: str | None = None,
    ) -> Self:
        """Return a new tracker with one additional processing record."""
        record = ProcessingRecord(
            operation=operation,
            parameters=parameters or {},
            operator=operator,
        )
        return self.with_(processing_history=(*self.processing_history, record))

    def get_provenance_chain(self) -> tuple[UUID, ...]:
        """Return the parent UUIDs of every provenance record in order."""
        return tuple(record.parent_id for record in self.provenance)

    def get_recent_processing(self, n: int = 5) -> tuple[ProcessingRecord, ...]:
        """Return the *n* most recent processing records, newest first."""
        return tuple(reversed(self.processing_history[-n:]))
