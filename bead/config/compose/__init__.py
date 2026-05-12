r"""Composition, interpolation, and validation for didactic configs.

A self-contained subpackage that turns a YAML or TOML file (plus
profile defaults, overlay files, and CLI overrides) into a fully
interpolated, validated ``dx.Model``. The grammar follows OmegaConf's
interpolation conventions:

- ``${section.field}`` absolute and ``${.field}`` / ``${..field}``
  relative dotted-path references
- ``${a.b[0]}`` and ``${a.b.0}`` list indexing
- ``${a.${b.c}}`` nested interpolations
- ``${name:arg1,arg2}`` resolver calls (built-ins plus user-registered)
- ``"prefix_${a.b}_suffix"`` string concatenation with type-preserving
  whole-value substitution
- ``\\${literal}`` escape

The subpackage imports nothing from ``bead`` outside of itself; the
only external dependency in implementation modules is ``didactic``
(used in :mod:`~bead.config.compose.merge` to enforce strict-key
checking against a target schema). It is structured so it can be
lifted into a standalone distribution by relocating the package and
adjusting a couple of internal imports.

Public API
----------
- :func:`compose` — the full pipeline.
- :func:`register_resolver` / :func:`unregister_resolver` /
  :func:`list_resolvers` — manage custom resolvers.
- :func:`resolve` — apply interpolation to an already-merged dict.
- :class:`ConfigError`, :class:`InterpolationError` — exceptions.
"""

from __future__ import annotations

from bead.config.compose.errors import ConfigError, InterpolationError
from bead.config.compose.interpolation import (
    ComposeValue,
    ResolverFn,
    active_root,
    list_resolvers,
    register_resolver,
    resolve,
    unregister_resolver,
)
from bead.config.compose.pipeline import compose

__all__ = [
    "ComposeValue",
    "ConfigError",
    "InterpolationError",
    "ResolverFn",
    "active_root",
    "compose",
    "list_resolvers",
    "register_resolver",
    "resolve",
    "unregister_resolver",
]
