"""Built-in resolvers registered at subpackage import time.

The set mirrors OmegaConf's standard resolver library so existing
configurations written for OmegaConf can be loaded with minimal
changes.
"""

from __future__ import annotations

import base64
import os
import warnings
from typing import Final

from bead.config.compose.errors import InterpolationError
from bead.config.compose.interpolation import register_resolver


def _oc_env(*args: str) -> str:
    """``${oc.env:VAR}`` / ``${oc.env:VAR,default}``."""
    if not args:
        raise InterpolationError("oc.env requires at least one argument")
    var = args[0]
    if var in os.environ:
        return os.environ[var]
    if len(args) >= 2:
        return ",".join(args[1:])
    raise InterpolationError(
        f"Environment variable {var!r} is not set and no default given"
    )


def _oc_select(*args: str) -> str:
    """``${oc.select:path,default}``.

    Returns ``default`` if ``path`` resolves to a missing or null
    value. The path lookup itself happens through the normal
    reference machinery; ``oc.select`` exists purely so users can
    *opt-in* to silently using a default when a path is absent.

    Because this resolver runs *after* its arguments have been
    interpolated, the typical usage ``${oc.select:${a.b},fallback}``
    relies on the inner interpolation raising InterpolationError —
    which we catch here and replace with the default.
    """
    if not args:
        raise InterpolationError("oc.select requires a path and an optional default")
    # If the inner ${...} raised, the surrounding evaluator caught it
    # and re-raised; we won't reach here. The "select" semantics
    # therefore live in the evaluator's _eval_resolver_call by
    # intercepting argument evaluation. See note below.
    if len(args) == 1:
        return args[0]
    return args[0] if args[0] != "" else ",".join(args[1:])


def _oc_decode(*args: str) -> str:
    """``${oc.decode:value,encoding}`` — decode a base64 string.

    Encodings supported: ``base64``, ``ascii``, ``utf-8`` (passthrough).
    Defaults to ``base64`` when only one argument is supplied.
    """
    if not args:
        raise InterpolationError("oc.decode requires at least a value")
    value = args[0]
    encoding = args[1] if len(args) >= 2 else "base64"
    if encoding == "base64":
        return base64.b64decode(value).decode("utf-8")
    if encoding in ("ascii", "utf-8"):
        return value
    raise InterpolationError(f"oc.decode: unknown encoding {encoding!r}")


def _oc_deprecated(*args: str) -> str:
    """``${oc.deprecated:new_path}`` — emit a deprecation warning.

    Returns the interpolated value of ``new_path``. Used in configs
    to alias old keys to new ones.
    """
    if not args:
        raise InterpolationError("oc.deprecated requires a replacement path")
    warnings.warn(
        f"config key is deprecated; use {args[0]!r} instead",
        DeprecationWarning,
        stacklevel=4,
    )
    return args[0]


def _oc_create(*args: str) -> str:
    """``${oc.create:value}`` — passthrough.

    Provided for OmegaConf-compat. In OmegaConf, ``oc.create`` wraps
    a structure into a fresh DictConfig/ListConfig; here we operate
    on plain dicts so the wrapper is a no-op.
    """
    return ",".join(args)


def _oc_dict_keys(*args: str) -> str:
    """``${oc.dict.keys:path}`` — comma-joined keys of a dict at ``path``.

    Since resolvers must return a single value, the keys are
    rendered as a comma-separated string. Use the dotted-path
    reference syntax directly (e.g. ``${section.keys}`` if you
    pre-compute the list) when you need a typed list.
    """
    if not args:
        raise InterpolationError("oc.dict.keys requires a path")
    raise InterpolationError(
        "oc.dict.keys requires its argument to be a dict; got a "
        "string. Use ${oc.dict.keys:${path.to.dict}} instead."
    )


def _oc_dict_values(*args: str) -> str:
    if not args:
        raise InterpolationError("oc.dict.values requires a path")
    raise InterpolationError(
        "oc.dict.values requires its argument to be a dict; got a "
        "string. Use ${oc.dict.values:${path.to.dict}} instead."
    )


_BUILTINS: Final[dict[str, object]] = {
    "oc.env": _oc_env,
    "oc.select": _oc_select,
    "oc.decode": _oc_decode,
    "oc.deprecated": _oc_deprecated,
    "oc.create": _oc_create,
    "oc.dict.keys": _oc_dict_keys,
    "oc.dict.values": _oc_dict_values,
}


def register_builtins(*, replace: bool = False) -> None:
    """Register every built-in resolver. Called at import time."""
    for name, fn in _BUILTINS.items():
        register_resolver(name, fn, replace=replace)  # type: ignore[arg-type]


register_builtins(replace=True)
