"""End-to-end compose pipeline.

Ties together :mod:`~bead.config.compose.sources`,
:mod:`~bead.config.compose.merge`, and
:mod:`~bead.config.compose.interpolation` into a single
:func:`compose` entry point that takes file paths and overrides and
returns a validated didactic Model.
"""

from __future__ import annotations

import copy
from collections.abc import Sequence
from pathlib import Path

import didactic.api as dx

# Resolvers are registered as a side effect of importing
# ``bead.config.compose.resolvers``; pull the module in here so that
# any code that imports ``compose`` has the built-in resolvers
# available. The wildcard suppresses pyright's unused-import warning
# while keeping the side-effect explicit.
from bead.config.compose import resolvers as _builtin_resolvers
from bead.config.compose.errors import ConfigError
from bead.config.compose.interpolation import ComposeValue, resolve
from bead.config.compose.merge import (
    apply_override,
    parse_override,
    strict_merge,
)
from bead.config.compose.sources import load_one, resolve_defaults_entry

_ = _builtin_resolvers


def compose[M: dx.Model](
    config_path: Path | str | None = None,
    *,
    schema: type[M],
    profile_dict: dict[str, ComposeValue] | None = None,
    overrides: Sequence[str] = (),
    extra: Sequence[Path | str] = (),
) -> M:
    """Compose, interpolate, and validate a config of type ``schema``.

    Precedence (lowest to highest):

      1. ``profile_dict`` — caller-supplied base. Empty when ``None``.
      2. Each path listed in the YAML's ``defaults: [...]`` key,
         loaded left-to-right (paths resolved relative to the
         primary YAML's parent directory).
      3. The primary YAML body (everything except ``defaults``).
      4. Each ``extra`` overlay file, in order.
      5. ``overrides`` — dotted-key ``key=value`` strings.

    Interpolation is resolved last via
    :func:`~bead.config.compose.interpolation.resolve`; the resolved
    dict is then validated by ``schema.model_validate(...)``.

    Parameters
    ----------
    config_path : Path | str | None, optional
        Primary YAML or TOML file. ``None`` skips file loading and
        merges only ``profile_dict`` + ``overrides``.
    schema : type[M]
        Target didactic model. Drives strict-key enforcement and
        final validation.
    profile_dict : dict[str, ComposeValue] | None, optional
        Pre-loaded base, typically a profile dump. Defaults to
        ``None``.
    overrides : Sequence[str], optional
        CLI-style overrides (``["paths.data_dir=/tmp"]``). YAML-parsed
        values; later entries beat earlier ones.
    extra : Sequence[Path | str], optional
        Additional overlay files merged after the primary YAML.

    Returns
    -------
    M
        Fully composed, interpolated, and validated model.

    Raises
    ------
    ConfigError
        For malformed configs (unknown keys, bad ``defaults`` entries,
        malformed overrides).
    InterpolationError
        For unresolved ``${...}`` expressions or cycles.
    """
    accumulated: dict[str, ComposeValue] = (
        copy.deepcopy(profile_dict) if profile_dict else {}
    )

    if config_path is not None:
        primary_path = Path(config_path)
        primary = load_one(primary_path)
        defaults_list = primary.pop("defaults", None)
        if defaults_list is not None:
            if not isinstance(defaults_list, list):
                raise ConfigError(
                    f"'defaults' in {primary_path} must be a list of "
                    f"strings; got {type(defaults_list).__name__}"
                )
            anchor = primary_path.parent
            for entry in defaults_list:
                if not isinstance(entry, str):
                    raise ConfigError(
                        f"'defaults' entries must be strings; got "
                        f"{type(entry).__name__}: {entry!r}"
                    )
                overlay = load_one(resolve_defaults_entry(entry, anchor=anchor))
                accumulated = strict_merge(accumulated, overlay, schema=schema)
        accumulated = strict_merge(accumulated, primary, schema=schema)

    for extra_path in extra:
        overlay = load_one(extra_path)
        accumulated = strict_merge(accumulated, overlay, schema=schema)

    for raw in overrides:
        key, value = parse_override(raw)
        apply_override(accumulated, key, value)

    resolved = resolve(accumulated, root=accumulated)
    if not isinstance(resolved, dict):
        raise ConfigError(
            f"Resolved config root is not a mapping (got {type(resolved).__name__})"
        )

    return schema.model_validate(resolved)
