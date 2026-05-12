"""Exceptions raised by the compose subpackage.

These types are part of the public API and survive the eventual
extraction of this subpackage into a standalone distribution.
"""

from __future__ import annotations


class ConfigError(ValueError):
    """Raised when a configuration is malformed or fails validation.

    Common causes
    -------------
    - Unknown keys at any level of the merged config (strict-merge
      rejection).
    - A ``defaults: [...]`` entry that cannot be resolved to a
      loadable YAML or TOML file.
    - A dotted-key override that targets a key not declared in the
      schema.
    """


class InterpolationError(ValueError):
    """Raised when an interpolation expression cannot be resolved.

    Common causes
    -------------
    - A reference like ``${a.b.c}`` that does not exist in the
      composed config.
    - A resolver call to an unregistered name (e.g. ``${unknown:x}``).
    - A cycle detected during evaluation
      (``${a} → ${b} → ${a}``).
    """
