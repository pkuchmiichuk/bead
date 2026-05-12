"""Bead-aware interpolation resolvers.

Registered at import time against the
:mod:`bead.config.compose.interpolation` registry. These resolvers
reference bead-side concepts (paths, anchors) and therefore would
*not* be part of an extracted ``didacticonf`` package.

Resolvers
---------
- ``${bead.path:rel}`` — join ``rel`` against the value at
  ``paths.data_dir`` in the composed config. Convenient shorthand
  for ``${paths.data_dir}/rel``.

The compose pipeline sets the in-flight root via a contextvar so
:func:`bead.config.compose.active_root` returns the dict being
interpolated.

Post-validation anchor resolution (``${bead.anchor:name[,attr]}``)
needs a validated :class:`AnnotationProtocol` and therefore lives at
post-validation time. See :func:`resolve_anchor_attributes`.
"""

from __future__ import annotations

import re
from pathlib import PurePosixPath
from typing import cast

from bead.config.compose import active_root, register_resolver, resolve
from bead.config.compose.errors import InterpolationError
from bead.config.compose.interpolation import ComposeValue


def _bead_path(*args: str) -> str:
    """``${bead.path:rel}`` — join ``rel`` against ``paths.data_dir``.

    Reads ``paths.data_dir`` from the in-flight composed root. The
    resulting string uses forward slashes (``PurePosixPath``);
    callers wrap with :class:`pathlib.Path` as needed.
    """
    if not args:
        raise InterpolationError("bead.path requires a relative path")
    rel = ",".join(args)

    root = active_root()
    if root is None:
        raise InterpolationError(
            "bead.path called outside an active compose() pipeline"
        )
    paths_section = root.get("paths")
    if not isinstance(paths_section, dict):
        raise InterpolationError("bead.path requires a 'paths' section in the config")
    data_dir: ComposeValue = paths_section.get("data_dir")
    if data_dir is None:
        raise InterpolationError("bead.path requires paths.data_dir to be set")
    if not isinstance(data_dir, str):
        resolved = resolve(data_dir, root=root)
        if not isinstance(resolved, str):
            raise InterpolationError(
                f"paths.data_dir resolved to {type(resolved).__name__}, expected str"
            )
        data_dir = resolved

    return str(PurePosixPath(data_dir) / rel)


register_resolver("bead.path", _bead_path, replace=True)


# ---------------------------------------------------------------------------
# Post-validation anchor resolution
# ---------------------------------------------------------------------------


_ANCHOR_PATTERN: re.Pattern[str] = re.compile(r"\$\{bead\.anchor:([^}]+)\}")


def resolve_anchor_attributes(
    text: str,
    *,
    protocol: object,
) -> str:
    """Replace ``${bead.anchor:name[,attr]}`` references in ``text``.

    Used by application code after the protocol is materialized.
    ``attr`` defaults to ``"canonical_prompt"`` and may be any
    attribute name on :class:`~bead.protocol.SemanticAnchor`.

    Parameters
    ----------
    text : str
        Text containing ``${bead.anchor:name}`` or
        ``${bead.anchor:name,attr}`` expressions.
    protocol : AnnotationProtocol
        Validated protocol whose ``family_by_name(name).anchor`` is
        consulted.

    Returns
    -------
    str
        ``text`` with every recognized expression substituted.
    """

    def _replace(match: re.Match[str]) -> str:
        spec = match.group(1)
        if "," in spec:
            name, _, attr = spec.partition(",")
            name, attr = name.strip(), attr.strip()
        else:
            name, attr = spec.strip(), "canonical_prompt"
        family = cast("object", protocol.family_by_name(name))  # type: ignore[attr-defined]
        anchor = cast("object", family.anchor)  # type: ignore[attr-defined]
        return str(getattr(anchor, attr))

    return _ANCHOR_PATTERN.sub(_replace, text)
