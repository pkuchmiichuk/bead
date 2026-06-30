"""Participant collection with JSONL I/O and DataFrame support.

``ParticipantCollection`` and ``IDMappingCollection`` group multiple
``Participant`` and ``ParticipantIDMapping`` instances respectively, with
JSONL serialization and pandas / polars DataFrame conversion.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self
from uuid import UUID

import didactic.api as dx
import pandas as pd
import polars as pl

from bead.data.base import BeadBaseModel, JsonValue
from bead.data.serialization import read_jsonlines, write_jsonlines
from bead.participants.models import Participant, ParticipantIDMapping

if TYPE_CHECKING:
    from bead.participants.metadata_spec import ParticipantMetadataSpec

DataFrame = pd.DataFrame | pl.DataFrame


class ParticipantCollection(BeadBaseModel):
    """Collection of participants.

    Attributes
    ----------
    name : str
        Collection name.
    participants : tuple[Participant, ...]
        Member participants.
    metadata_spec_name : str | None
        Name of the metadata spec applied (for documentation).
    """

    name: str
    participants: tuple[dx.Embed[Participant], ...] = ()
    metadata_spec_name: str | None = None

    @dx.validates("name")
    def _check_name(self, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Collection name cannot be empty")
        return value.strip()

    def __len__(self) -> int:
        """Return the number of participants."""
        return len(self.participants)

    def with_participant(self, participant: Participant) -> Self:
        """Return a new collection with *participant* appended."""
        return self.with_(participants=(*self.participants, participant)).touched()

    def with_participants(
        self, participants: tuple[Participant, ...] | list[Participant]
    ) -> Self:
        """Return a new collection with each participant appended."""
        return self.with_(participants=(*self.participants, *participants)).touched()

    def get_by_id(self, participant_id: UUID) -> Participant | None:
        """Return the participant whose id matches, or ``None``."""
        for p in self.participants:
            if p.id == participant_id:
                return p
        return None

    def get_by_attribute(self, key: str, value: JsonValue) -> tuple[Participant, ...]:
        """Return participants whose ``participant_metadata[key] == value``."""
        return tuple(
            p for p in self.participants if p.participant_metadata.get(key) == value
        )

    def validate_all(self, spec: ParticipantMetadataSpec) -> dict[UUID, list[str]]:
        """Validate every participant against *spec*.

        Returns a mapping from offending participant id to error messages.
        """
        errors: dict[UUID, list[str]] = {}
        for p in self.participants:
            is_valid, error_list = p.validate_against_spec(spec)
            if not is_valid:
                errors[p.id] = error_list
        return errors

    def to_jsonl(self, path: Path | str) -> None:
        """Write each participant to *path* as a JSONL line."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonlines(self.participants, path)

    @classmethod
    def from_jsonl(
        cls,
        path: Path | str,
        name: str = "loaded_participants",
    ) -> ParticipantCollection:
        """Load participants from *path* as JSONL."""
        participants = read_jsonlines(Path(path), Participant)
        return cls(name=name, participants=tuple(participants))

    def to_dataframe(
        self,
        backend: Literal["pandas", "polars"] = "pandas",
        include_fields: tuple[str, ...] | None = None,
        exclude_fields: tuple[str, ...] | None = None,
        flatten_metadata: bool = True,
    ) -> DataFrame:
        """Render the collection as a DataFrame.

        Always emits ``participant_id``, ``created_at``, and ``study_id``
        columns; ``participant_metadata`` is flattened by default.
        """
        if not self.participants:
            columns = ["participant_id", "created_at", "study_id"]
            if backend == "pandas":
                return pd.DataFrame(columns=columns)
            schema: dict[str, type[pl.Utf8]] = dict.fromkeys(columns, pl.Utf8)
            return pl.DataFrame(schema=schema)

        records: list[dict[str, JsonValue]] = []
        for p in self.participants:
            record: dict[str, JsonValue] = {
                "participant_id": str(p.id),
                "created_at": p.created_at.isoformat(),
                "study_id": p.study_id,
            }
            if flatten_metadata:
                for key, value in p.participant_metadata.items():
                    if include_fields is not None and key not in include_fields:
                        continue
                    if exclude_fields is not None and key in exclude_fields:
                        continue
                    record[key] = value
            else:
                record["participant_metadata"] = p.participant_metadata
            records.append(record)

        if backend == "pandas":
            return pd.DataFrame(records)
        return pl.DataFrame(records)

    @classmethod
    def from_dataframe(
        cls,
        df: DataFrame,
        name: str,
        id_column: str = "participant_id",
        metadata_columns: tuple[str, ...] | None = None,
    ) -> ParticipantCollection:
        """Build a collection from a DataFrame of participant rows.

        Each row becomes a ``Participant``. The ``id_column`` is consumed
        as the participant UUID when present and parseable; otherwise a
        new UUID is generated. ``metadata_columns`` (if given) restricts
        which columns flow into ``participant_metadata``.
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

        participants: list[Participant] = []
        meta_cols = (
            list(metadata_columns)
            if metadata_columns is not None
            else [c for c in columns_list if c != id_column]
        )

        for row in rows:
            pid: UUID | None = None
            if id_column in columns_list:
                try:
                    pid = UUID(str(row[id_column]))
                except ValueError, TypeError:
                    pid = None

            metadata: dict[str, JsonValue] = {}
            for col in meta_cols:
                if col not in row or row[col] is None:
                    continue
                value = row[col]
                if not is_polars:
                    if isinstance(value, float) and value != value:  # NaN check
                        continue
                metadata[col] = value

            participants.append(
                Participant(id=pid, participant_metadata=metadata)
                if pid is not None
                else Participant(participant_metadata=metadata)
            )

        return cls(name=name, participants=tuple(participants))


class IDMappingCollection(BeadBaseModel):
    """Collection of external-to-internal participant ID mappings.

    Stored separately from participant data for IRB / privacy compliance.

    Attributes
    ----------
    name : str
        Collection name.
    source : str
        Primary external ID source (e.g. ``"prolific"``).
    mappings : tuple[ParticipantIDMapping, ...]
        Member mappings.
    """

    name: str
    source: str
    mappings: tuple[dx.Embed[ParticipantIDMapping], ...] = ()

    @dx.validates("name", "source")
    def _check_non_empty(self, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Field cannot be empty")
        return value.strip()

    def __len__(self) -> int:
        """Return the number of mappings."""
        return len(self.mappings)

    def with_mapping(
        self,
        external_id: str,
        participant_id: UUID,
        external_source: str | None = None,
    ) -> tuple[Self, ParticipantIDMapping]:
        """Return ``(new_collection, mapping)`` with one new mapping appended."""
        mapping = ParticipantIDMapping(
            external_id=external_id,
            external_source=external_source or self.source,
            participant_id=participant_id,
        )
        new_self = self.with_(mappings=(*self.mappings, mapping)).touched()
        return new_self, mapping

    def get_participant_id(self, external_id: str) -> UUID | None:
        """Return the internal UUID for *external_id* if a live mapping exists."""
        for m in self.mappings:
            if m.external_id == external_id and m.is_active:
                return m.participant_id
        return None

    def get_external_id(self, participant_id: UUID) -> str | None:
        """Return the external id for *participant_id* if a live mapping exists."""
        for m in self.mappings:
            if m.participant_id == participant_id and m.is_active:
                return m.external_id
        return None

    def deactivated_all(self) -> tuple[Self, int]:
        """Return ``(new_collection, count_deactivated)`` with all live mappings off."""
        new_mappings = tuple(
            m.deactivated() if m.is_active else m for m in self.mappings
        )
        count = sum(1 for m in self.mappings if m.is_active)
        return self.with_(mappings=new_mappings).touched(), count

    def to_jsonl(self, path: Path | str) -> None:
        """Write each mapping to *path* as a JSONL line."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonlines(self.mappings, path)

    @classmethod
    def from_jsonl(
        cls,
        path: Path | str,
        name: str = "loaded_mappings",
        source: str = "unknown",
    ) -> IDMappingCollection:
        """Load mappings from *path* as JSONL."""
        mappings = read_jsonlines(Path(path), ParticipantIDMapping)
        return cls(name=name, mappings=tuple(mappings), source=source)
