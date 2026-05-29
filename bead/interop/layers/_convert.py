"""Shared, reversible conversions between bead values and layers JSON shapes.

These helpers centralize the mechanical, lossless conversions every layers lens
relies on: feature maps, object references, confidence scaling, and capture /
restore of a bead model's framework identity (the ``BeadBaseModel`` id and
timestamps, which ``layers`` represents through its own identity scheme and so
travel in a lens complement rather than the layers view).
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from bead.corpus.records import ProvenanceValue
from bead.data.base import BeadBaseModel, JsonValue

if TYPE_CHECKING:
    from bead.items.item import MetadataValue


def to_feature_map(features: Mapping[str, MetadataValue]) -> JsonValue:
    """Encode a feature dict as a layers ``featureMap`` (values JSON-encoded).

    Each value is serialized with ``json.dumps`` so arbitrary (including
    non-string) values round-trip exactly via :func:`from_feature_map`. Entries
    preserve the dict's insertion order so the round-trip is exact.
    """
    entries: tuple[JsonValue, ...] = tuple(
        {"key": key, "value": json.dumps(features[key])} for key in features
    )
    return {"entries": entries}


type _Loaded = (
    str | int | float | bool | None | list["_Loaded"] | dict[str, "_Loaded"]
)


def _tuplify(value: _Loaded) -> MetadataValue:
    """Convert ``json.loads`` output (lists) into the tuple-based MetadataValue."""
    if isinstance(value, list):
        return tuple(_tuplify(item) for item in value)
    if isinstance(value, dict):
        return {str(key): _tuplify(val) for key, val in value.items()}
    return value


def from_feature_map(feature_map: JsonValue) -> dict[str, MetadataValue]:
    """Decode a layers ``featureMap`` back into a feature dict."""
    result: dict[str, MetadataValue] = {}
    if not isinstance(feature_map, dict):
        return result
    entries = feature_map.get("entries")
    if not isinstance(entries, tuple):
        return result
    for entry in entries:
        if isinstance(entry, dict):
            key = entry.get("key")
            value = entry.get("value")
            if isinstance(key, str) and isinstance(value, str):
                result[key] = _tuplify(json.loads(value))
    return result


def from_feature_map_scalar(feature_map: JsonValue) -> dict[str, ProvenanceValue]:
    """Decode a ``featureMap`` whose values are flat provenance scalars."""
    result: dict[str, ProvenanceValue] = {}
    if not isinstance(feature_map, dict):
        return result
    entries = feature_map.get("entries")
    if not isinstance(entries, tuple):
        return result
    for entry in entries:
        if isinstance(entry, dict):
            key = entry.get("key")
            value = entry.get("value")
            if isinstance(key, str) and isinstance(value, str):
                result[key] = json.loads(value)
    return result


def strip_nulls(value: JsonValue) -> JsonValue:
    """Recursively drop dict entries whose value is ``None``.

    The ATProto data model has no null: optional fields are omitted, not set to
    null, and a lexicon rejects an explicit null for a typed optional field.
    Layers views therefore omit absent optionals; the round-trip is unaffected
    because the reverse direction defaults missing keys back to ``None``.
    """
    if isinstance(value, dict):
        return {
            key: strip_nulls(item)
            for key, item in value.items()
            if item is not None
        }
    if isinstance(value, tuple):
        return tuple(strip_nulls(item) for item in value)
    return value


def object_ref(local_id: str) -> JsonValue:
    """Build a layers ``objectRef`` to a local node by id."""
    return {"localId": {"value": local_id}}


def from_object_ref(ref: JsonValue) -> str:
    """Read the local id out of a layers ``objectRef``."""
    if isinstance(ref, dict):
        local = ref.get("localId")
        if isinstance(local, dict):
            value = local.get("value")
            if isinstance(value, str):
                return value
    raise ValueError("objectRef has no localId.value")


def identity_of(model: BeadBaseModel) -> JsonValue:
    """Capture a model's framework identity for a lens complement."""
    return {
        "id": str(model.id),
        "created_at": model.created_at.isoformat(),
        "modified_at": model.modified_at.isoformat(),
        "version": model.version,
        "metadata": dict(model.metadata),
    }


def apply_identity[T: BeadBaseModel](model: T, identity: JsonValue) -> T:
    """Restore a model's captured framework identity onto a fresh instance.

    The model is constructed with content fields (and default identity); this
    overrides the framework identity (id, timestamps, version, metadata) with
    the values captured by :func:`identity_of`, so a round-trip is exact.
    """
    fields = j_obj(identity)
    metadata = fields["metadata"]
    return model.with_(
        id=UUID(_as_str(fields["id"])),
        created_at=datetime.fromisoformat(_as_str(fields["created_at"])),
        modified_at=datetime.fromisoformat(_as_str(fields["modified_at"])),
        version=_as_str(fields["version"]),
        metadata=metadata if isinstance(metadata, dict) else {},
    )


def _as_str(value: JsonValue) -> str:
    if not isinstance(value, str):
        raise ValueError(f"expected str, got {type(value).__name__}")
    return value


def j_obj(value: JsonValue) -> dict[str, JsonValue]:
    """Narrow a ``JsonValue`` to a JSON object, raising otherwise."""
    if not isinstance(value, dict):
        raise ValueError(f"expected JSON object, got {type(value).__name__}")
    return value


def j_list(value: JsonValue) -> tuple[JsonValue, ...]:
    """Narrow a ``JsonValue`` to a JSON array, raising otherwise."""
    if isinstance(value, tuple):
        return value
    raise ValueError(f"expected JSON array, got {type(value).__name__}")


def j_str(value: JsonValue) -> str:
    """Narrow a ``JsonValue`` to a string, raising otherwise."""
    return _as_str(value)


def j_str_or_none(value: JsonValue) -> str | None:
    """Narrow a ``JsonValue`` to ``str | None``."""
    if value is None or isinstance(value, str):
        return value
    raise ValueError(f"expected str or None, got {type(value).__name__}")


def j_float_or_none(value: JsonValue) -> float | None:
    """Narrow a ``JsonValue`` to ``float | None``."""
    if value is None or isinstance(value, (int, float)):
        return value
    raise ValueError(f"expected number or None, got {type(value).__name__}")


def j_bool(value: JsonValue) -> bool:
    """Narrow a ``JsonValue`` to a bool, raising otherwise."""
    if isinstance(value, bool):
        return value
    raise ValueError(f"expected bool, got {type(value).__name__}")


def j_int(value: JsonValue) -> int:
    """Narrow a ``JsonValue`` to an int, raising otherwise."""
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"expected int, got {type(value).__name__}")
    return value
