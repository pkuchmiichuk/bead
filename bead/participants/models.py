"""Participant data models.

``Participant`` stores demographic and session metadata.
``ParticipantIDMapping`` records the link between an external participant
identifier (e.g. Prolific PID) and an internal UUID. The mapping is stored
separately so the external id can be deleted for privacy compliance while
the internal UUID is retained for analysis.
"""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Self
from uuid import UUID

import didactic.api as dx

from bead.data.base import BeadBaseModel, JsonValue
from bead.data.timestamps import now_iso8601

if TYPE_CHECKING:
    from bead.participants.metadata_spec import ParticipantMetadataSpec


class Participant(BeadBaseModel):
    """A study participant.

    Attributes
    ----------
    participant_metadata : dict[str, JsonValue]
        Demographic and other participant attributes.
    study_id : str | None
        Study identifier.
    session_ids : tuple[str, ...]
        Session identifiers (for longitudinal studies).
    consent_timestamp : datetime | None
        When the participant provided consent.
    notes : str | None
        Free-text notes.
    """

    participant_metadata: dict[str, JsonValue] = dx.field(default_factory=dict)
    study_id: str | None = None
    session_ids: tuple[str, ...] = ()
    consent_timestamp: datetime | None = None
    notes: str | None = None

    def validate_against_spec(
        self, spec: ParticipantMetadataSpec
    ) -> tuple[bool, list[str]]:
        """Validate ``participant_metadata`` against *spec*.

        Returns ``(is_valid, error_messages)``.
        """
        metadata: dict[str, str | int | float | bool | None] = {}
        for key, value in self.participant_metadata.items():
            if isinstance(value, str | int | float | bool) or value is None:
                metadata[key] = value
        return spec.validate_metadata(metadata)

    def get_attribute(self, key: str, default: JsonValue = None) -> JsonValue:
        """Return ``participant_metadata[key]`` if present, else *default*."""
        return self.participant_metadata.get(key, default)

    def with_attribute(self, key: str, value: JsonValue) -> Self:
        """Return a new participant with ``participant_metadata[key] = value``."""
        new_metadata = {**self.participant_metadata, key: value}
        return self.with_(participant_metadata=new_metadata).touched()

    def with_session(self, session_id: str) -> Self:
        """Return a new participant with *session_id* appended."""
        return self.with_(session_ids=(*self.session_ids, session_id)).touched()


class ParticipantIDMapping(BeadBaseModel):
    """Mapping between an external participant ID and an internal UUID.

    Attributes
    ----------
    external_id : str
        External participant identifier (e.g. Prolific PID).
    external_source : str
        Source of the external id (``"prolific"``, ``"mturk"``, etc.).
    participant_id : UUID
        Internal participant UUID.
    mapping_timestamp : datetime
        When the mapping was created.
    is_active : bool
        Whether the mapping is active (used for soft-delete).
    """

    external_id: str
    external_source: str
    participant_id: UUID
    mapping_timestamp: datetime = dx.field(default_factory=now_iso8601)
    is_active: bool = True

    @dx.validates("external_id", "external_source")
    def _check_non_empty(self, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Field cannot be empty")
        return value.strip()

    def deactivated(self) -> Self:
        """Return a new mapping with ``is_active=False``."""
        return self.with_(is_active=False).touched()
