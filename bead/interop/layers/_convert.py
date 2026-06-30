"""Shared, reversible conversions between bead values and ``lairs`` models.

These helpers centralize the mechanical, lossless conversions every layers lens
relies on. Two sinks are served:

- the layers *view*, built from the canonical ``lairs.records`` models, uses the
  typed :func:`feature_map` / :func:`object_ref` builders.
- the lens *complement*, a plain ``JsonValue``, carries the bead-only remainder
  (framework identity and metadata dicts ``layers`` has no slot for) via
  :func:`identity_of` and :func:`dumps_meta`.
"""

from __future__ import annotations

import json
from collections.abc import Mapping
from datetime import datetime
from typing import TYPE_CHECKING
from uuid import UUID

from lairs.records import defs

from bead.corpus.records import ProvenanceValue
from bead.data.base import BeadBaseModel, JsonValue

if TYPE_CHECKING:
    from bead.items.item import MetadataValue

# layers scales confidence to an integer 0-1000 (to avoid floats on the wire).
CONFIDENCE_SCALE = 1000

# Canonical layers collection NSIDs (record level), used both as fragment record
# NSIDs and as the collection segment of minted corpus AT-URIs.
EXPRESSION_NSID = "pub.layers.expression.expression"
SEGMENTATION_NSID = "pub.layers.segmentation.segmentation"
ANNOTATION_LAYER_NSID = "pub.layers.annotation.annotationLayer"
CORPUS_NSID = "pub.layers.corpus.corpus"
MEMBERSHIP_NSID = "pub.layers.corpus.membership"


# --- typed feature maps (the layers view) -----------------------------------


def feature_map(features: Mapping[str, MetadataValue]) -> defs.FeatureMap | None:
    """Build a layers ``featureMap`` model, or ``None`` for an empty mapping.

    Each value is serialized with ``json.dumps`` so arbitrary (including
    non-string) values round-trip exactly via :func:`read_feature_map`. Entries
    preserve the mapping's iteration order so the round-trip is exact. An empty
    mapping projects to ``None`` (a faithful layers view omits empty optionals).
    """
    if not features:
        return None
    return defs.FeatureMap(
        entries=tuple(
            defs.Feature(key=key, value=json.dumps(features[key])) for key in features
        )
    )


type _Loaded = str | int | float | bool | None | list["_Loaded"] | dict[str, "_Loaded"]


def _tuplify(value: _Loaded) -> MetadataValue:
    """Convert ``json.loads`` output (lists) into the tuple-based MetadataValue."""
    if isinstance(value, list):
        return tuple(_tuplify(item) for item in value)
    if isinstance(value, dict):
        return {str(key): _tuplify(val) for key, val in value.items()}
    return value


def read_feature_map(fm: defs.FeatureMap | None) -> dict[str, MetadataValue]:
    """Decode a layers ``featureMap`` model back into a feature dict."""
    if fm is None:
        return {}
    return {entry.key: _tuplify(json.loads(entry.value)) for entry in fm.entries}


def read_feature_map_scalar(
    fm: defs.FeatureMap | None,
) -> dict[str, ProvenanceValue]:
    """Decode a ``featureMap`` model whose values are flat provenance scalars."""
    if fm is None:
        return {}
    return {entry.key: json.loads(entry.value) for entry in fm.entries}


# --- typed object references (the layers view) ------------------------------


def object_ref(local_id: str) -> defs.ObjectRef:
    """Build a layers ``objectRef`` to a local object by id."""
    return defs.ObjectRef(localId=defs.Uuid(value=local_id))


def from_object_ref(ref: defs.ObjectRef) -> str:
    """Read the local id out of a layers ``objectRef`` model."""
    if ref.localId is None:
        raise ValueError("objectRef has no localId")
    return ref.localId.value


# --- metadata dicts in the lens complement (JsonValue) ----------------------


def dumps_meta(mapping: Mapping[str, MetadataValue]) -> str:
    """Encode a metadata dict as a JSON string for a lens complement.

    ``MetadataValue`` admits tuples (which are not ``JsonValue``); serializing to
    a JSON string keeps the complement a plain ``JsonValue`` and round-trips
    exactly through :func:`loads_meta`. Key insertion order is preserved so the
    reconstructed dict compares equal to the original (didactic compares dict
    fields order-sensitively).
    """
    return json.dumps(dict(mapping))


def loads_meta(value: JsonValue) -> dict[str, MetadataValue]:
    """Decode a :func:`dumps_meta` string back into a metadata dict."""
    loaded = json.loads(_as_str(value))
    if not isinstance(loaded, dict):
        return {}
    return {str(key): _tuplify(val) for key, val in loaded.items()}


# --- framework identity (the lens complement) -------------------------------


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


# --- JsonValue narrowers (for reading complements) --------------------------


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
    """Narrow a ``JsonValue`` to a JSON array, raising otherwise.

    Accepts both tuples (in-process complements) and lists (complements that have
    round-tripped through JSON, as in the codec), normalizing to a tuple.
    """
    if isinstance(value, (tuple, list)):
        return tuple(value)
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
