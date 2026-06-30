"""Root didactic model for all bead objects.

Provides ``BeadBaseModel``, the root ``didactic.api.Model`` that every bead
data model inherits from. Supplies UUIDv7 identity, UTC creation and
modification timestamps, schema versioning, and a metadata dictionary.

Models are frozen; updates produce new instances through ``with_`` or the
convenience method ``touched`` (which refreshes ``modified_at``).
"""

from __future__ import annotations

from datetime import datetime
from typing import Self
from uuid import UUID

import didactic.api as dx

from bead.data.identifiers import generate_uuid
from bead.data.timestamps import now_iso8601

type JsonValue = (
    str | int | float | bool | None | tuple["JsonValue", ...] | dict[str, "JsonValue"]
)


class BeadBaseModel(dx.Model):
    """Root didactic model for all bead objects.

    Attributes
    ----------
    id : UUID
        UUIDv7 generated at construction time.
    created_at : datetime
        UTC timestamp of construction.
    modified_at : datetime
        UTC timestamp of the last ``touched`` call (defaults to
        ``created_at``).
    version : str
        Schema version string.
    metadata : dict[str, JsonValue]
        Free-form key-value annotations.
    """

    id: UUID = dx.field(default_factory=generate_uuid)
    created_at: datetime = dx.field(default_factory=now_iso8601)
    modified_at: datetime = dx.field(default_factory=now_iso8601)
    version: str = "1.0.0"
    metadata: dict[str, JsonValue] = dx.field(default_factory=dict)

    def touched(self) -> Self:
        """Return a copy with ``modified_at`` set to the current UTC time."""
        return self.with_(modified_at=now_iso8601())
