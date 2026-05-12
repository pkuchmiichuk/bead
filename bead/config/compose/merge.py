"""Strict, schema-aware dict merge and dotted-key override application.

This is the one module in the subpackage that imports ``didactic``.
It walks the merged dict against a target ``dx.Model``'s field
specifications to reject unknown keys at merge time — a stricter
behavior than validation alone, with a clearer error location.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import get_args, get_origin

import didactic.api as dx

from bead.config.compose.errors import ConfigError
from bead.config.compose.interpolation import ComposeValue


def strict_merge(
    base: dict[str, ComposeValue],
    overlay: dict[str, ComposeValue],
    *,
    schema: type[dx.Model],
    _path: tuple[str, ...] = (),
) -> dict[str, ComposeValue]:
    """Deep-merge ``overlay`` into ``base`` under ``schema``.

    Mappings overlay key-by-key; non-mapping values overwrite. Any
    key in ``overlay`` not declared in ``schema``'s field specs raises
    :class:`ConfigError` naming the dotted path to the offending key.

    Lists overwrite wholesale (no element-by-element merge); this
    matches OmegaConf's default.

    Parameters
    ----------
    base : dict[str, ComposeValue]
        Lower-precedence dict.
    overlay : dict[str, ComposeValue]
        Higher-precedence dict.
    schema : type[dx.Model]
        didactic model defining allowed keys.
    _path : tuple[str, ...]
        Internal recursion bookkeeping; pass nothing.

    Returns
    -------
    dict[str, ComposeValue]
        A fresh dict; neither input is mutated.
    """
    allowed = _allowed_fields(schema)
    result = dict(base)
    for key, overlay_value in overlay.items():
        if key not in allowed:
            dotted = ".".join((*_path, key))
            raise ConfigError(
                f"Unknown config key {dotted!r}; allowed: {sorted(allowed)}"
            )
        nested_schema = allowed[key]
        if isinstance(overlay_value, dict) and nested_schema is not None:
            existing = result.get(key)
            existing_dict = existing if isinstance(existing, dict) else {}
            result[key] = strict_merge(
                existing_dict,
                overlay_value,
                schema=nested_schema,
                _path=(*_path, key),
            )
        else:
            result[key] = overlay_value
    return result


def apply_override(
    d: dict[str, ComposeValue], dotted_key: str, value: ComposeValue
) -> None:
    """Set ``d[a][b][c] = value`` for ``dotted_key='a.b.c'``.

    Intermediate dicts are created as needed. Existing non-dict
    intermediates raise :class:`ConfigError`.

    Parameters
    ----------
    d : dict[str, ComposeValue]
        Target dict, modified in place.
    dotted_key : str
        Dotted path. Empty segments are not allowed.
    value : ComposeValue
        Value to set.
    """
    if not dotted_key:
        raise ConfigError("Override key cannot be empty")
    parts = dotted_key.split(".")
    cur: dict[str, ComposeValue] = d
    for part in parts[:-1]:
        if not part:
            raise ConfigError(f"Empty segment in override key {dotted_key!r}")
        existing = cur.get(part)
        if existing is None or not isinstance(existing, dict):
            new_dict: dict[str, ComposeValue] = {}
            cur[part] = new_dict
            cur = new_dict
        else:
            cur = existing
    last = parts[-1]
    if not last:
        raise ConfigError(f"Override key {dotted_key!r} ends with an empty segment")
    cur[last] = value


def parse_override(expr: str) -> tuple[str, ComposeValue]:
    """Split a CLI-style ``key=value`` override into its parts.

    The value is parsed as YAML so callers can pass typed primitives
    (``--set foo.bar=0.5`` produces a float, ``--set x.y=true``
    produces a bool, ``--set z=hello`` produces a string).

    Parameters
    ----------
    expr : str
        Override expression, e.g. ``"paths.data_dir=/tmp"``.

    Returns
    -------
    tuple[str, ComposeValue]
        ``(dotted_key, parsed_value)``.

    Raises
    ------
    ConfigError
        If ``expr`` is missing ``=`` or has an empty key.
    """
    if "=" not in expr:
        raise ConfigError(f"Override {expr!r} missing '='; expected 'key=value'")
    key, _, raw_value = expr.partition("=")
    key = key.strip()
    if not key:
        raise ConfigError(f"Override {expr!r} has empty key")

    import yaml  # noqa: PLC0415

    parsed = yaml.safe_load(raw_value)
    return key, parsed


def _allowed_fields(
    schema: type[dx.Model],
) -> dict[str, type[dx.Model] | None]:
    """Return the keys ``schema`` accepts, mapped to nested schemas.

    A field whose type is itself a ``dx.Model`` (directly or wrapped
    in ``dx.Embed[...]`` or ``tuple[dx.Embed[...], ...]``) maps to
    that nested model so the recursive walker can descend. Scalar
    fields map to ``None``.
    """
    field_specs = getattr(schema, "__field_specs__", None)
    if field_specs is None:
        return {}
    allowed: dict[str, type[dx.Model] | None] = {}
    for name, spec in field_specs.items():
        allowed[name] = _nested_schema(spec)
    return allowed


def _nested_schema(spec: object) -> type[dx.Model] | None:
    """Extract a nested ``dx.Model`` type from a field spec, if any."""
    annotation = getattr(spec, "annotation", None)
    if annotation is None:
        return None
    return _unwrap_model(annotation)


def _unwrap_model(annotation: object) -> type[dx.Model] | None:
    """Walk a type annotation looking for a single ``dx.Model`` subclass.

    Handles:
      - bare ``SomeModel``
      - ``dx.Embed[SomeModel]``
      - ``dx.Embed[SomeModel] | None``
      - ``tuple[dx.Embed[SomeModel], ...]``
      - ``dict[str, SomeModel]``
    """
    if isinstance(annotation, type) and issubclass(annotation, dx.Model):
        return annotation
    origin = get_origin(annotation)
    args: Iterable[object] = get_args(annotation)
    if origin is None:
        return None
    candidates: list[type[dx.Model]] = []
    for arg in args:
        inner = _unwrap_model(arg)
        if inner is not None:
            candidates.append(inner)
    if len(candidates) == 1:
        return candidates[0]
    return None
