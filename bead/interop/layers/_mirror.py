"""Generic serialization between mirror models and layers JSON.

The mirror models in :mod:`bead.interop.layers.models` match the ``layers``
schema structurally: snake_case fields correspond to layers' camelCase, nested
objects are embedded models, feature maps are :class:`FeatureMap`, and
confidence is an integer. A single pair of conversions therefore serializes any
of them to and from layers-shaped JSON.

Conversion goes through each model's canonical JSON form (``model_dump_json`` /
``model_validate_json``), so it does not depend on didactic's internal
field-value types.
"""

from __future__ import annotations

import json
import re

import didactic.api as dx

from bead.data.base import JsonValue
from bead.interop.layers._convert import j_obj, strip_nulls

_CAMEL_BOUNDARY = re.compile(r"([A-Z])")

type _Loaded = (
    str | int | float | bool | None | list["_Loaded"] | dict[str, "_Loaded"]
)


def _to_camel(name: str) -> str:
    head, *rest = name.split("_")
    return head + "".join(part[:1].upper() + part[1:] for part in rest)


def _to_snake(name: str) -> str:
    return _CAMEL_BOUNDARY.sub(lambda match: "_" + match.group(1).lower(), name)


def _camel_keys(value: _Loaded) -> JsonValue:
    """Recursively camelCase dict keys and turn JSON arrays into tuples."""
    if isinstance(value, dict):
        return {_to_camel(key): _camel_keys(item) for key, item in value.items()}
    if isinstance(value, list):
        return tuple(_camel_keys(item) for item in value)
    return value


def _snake_keys(value: JsonValue) -> JsonValue:
    """Recursively snake_case dict keys (arrays stay tuples)."""
    if isinstance(value, dict):
        return {_to_snake(key): _snake_keys(item) for key, item in value.items()}
    if isinstance(value, tuple):
        return tuple(_snake_keys(item) for item in value)
    return value


_DEFS_NSID = "pub.layers.defs"

#: The ``externalTarget.selector`` union variants (camelCase mirror keys, each
#: also the layers def name). layers models this as an open ATProto union, so the
#: wire value carries a ``$type`` discriminator rather than a wrapper key.
_SELECTOR_VARIANTS = frozenset(
    {"textQuoteSelector", "textPositionSelector", "fragmentSelector"}
)


def _wrap_unions(value: JsonValue) -> JsonValue:
    """Rewrite ``selector`` wrappers into ATProto ``$type`` union members."""
    if isinstance(value, dict):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            if key == "selector" and isinstance(item, dict) and len(item) == 1:
                variant, payload = next(iter(item.items()))
                if variant in _SELECTOR_VARIANTS and isinstance(payload, dict):
                    member: dict[str, JsonValue] = {"$type": f"{_DEFS_NSID}#{variant}"}
                    for inner_key, inner_item in payload.items():
                        member[inner_key] = _wrap_unions(inner_item)
                    result[key] = member
                    continue
            result[key] = _wrap_unions(item)
        return result
    if isinstance(value, tuple):
        return tuple(_wrap_unions(item) for item in value)
    return value


def _unwrap_unions(value: JsonValue) -> JsonValue:
    """Rewrite ATProto ``$type`` selector union members back to wrappers."""
    if isinstance(value, dict):
        result: dict[str, JsonValue] = {}
        for key, item in value.items():
            type_ref = item.get("$type") if isinstance(item, dict) else None
            if key == "selector" and isinstance(type_ref, str):
                variant = type_ref.rsplit("#", 1)[-1]
                payload: dict[str, JsonValue] = {}
                for inner_key, inner_item in j_obj(item).items():
                    if inner_key != "$type":
                        payload[inner_key] = _unwrap_unions(inner_item)
                result[key] = {variant: payload}
                continue
            result[key] = _unwrap_unions(item)
        return result
    if isinstance(value, tuple):
        return tuple(_unwrap_unions(item) for item in value)
    return value


def mirror_to_layers(model: dx.Model) -> JsonValue:
    """Serialize a faithful mirror model to layers-shaped JSON (camelCase)."""
    return _wrap_unions(strip_nulls(_camel_keys(json.loads(model.model_dump_json()))))


def mirror_from_layers[M: dx.Model](model_type: type[M], data: JsonValue) -> M:
    """Deserialize layers-shaped JSON back into a mirror model."""
    restored = _snake_keys(j_obj(_unwrap_unions(data)))
    return model_type.model_validate_json(json.dumps(restored))
