"""Generic, lossless serialization between faithful mirror models and layers JSON.

The mirror models in :mod:`bead.interop.layers.models` are designed to match the
``layers`` schema structurally (snake_case fields mirroring layers' camelCase,
nested objects as embedded models, feature maps as :class:`FeatureMap`, integer
confidence). That lets a single pair of conversions serialize any of them to and
from layers-shaped JSON, so each model needs only a three-line ``dx.Iso`` and a
round-trip test rather than bespoke code.

Serialization goes through each model's canonical JSON form
(``model_dump_json`` / ``model_validate_json``) so the conversion never depends
on didactic's internal field-value types.
"""

from __future__ import annotations

import json
import re

import didactic.api as dx

from bead.data.base import JsonValue
from bead.interop.layers._convert import j_obj

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


def mirror_to_layers(model: dx.Model) -> JsonValue:
    """Serialize a faithful mirror model to layers-shaped JSON (camelCase)."""
    return _camel_keys(json.loads(model.model_dump_json()))


def mirror_from_layers[M: dx.Model](model_type: type[M], data: JsonValue) -> M:
    """Deserialize layers-shaped JSON back into a mirror model."""
    return model_type.model_validate_json(json.dumps(_snake_keys(j_obj(data))))
