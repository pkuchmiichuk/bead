"""Bead-specific entrypoint to the compose pipeline.

A thin wrapper around :func:`bead.config.compose.compose` that binds
the schema to :class:`~bead.config.config.BeadConfig` and starts from
the profile defaults declared in :mod:`bead.config.profiles`.
"""

from __future__ import annotations

import json
from collections.abc import Sequence
from pathlib import Path

# Importing this module registers bead-specific resolvers
# (${bead.anchor:...}, ${bead.path:...}) against the compose
# interpolation engine.
from bead.config import resolvers as _bead_resolvers
from bead.config.compose import compose
from bead.config.compose.interpolation import ComposeValue
from bead.config.config import BeadConfig
from bead.config.profiles import get_profile

_ = _bead_resolvers


def load_config(
    config_path: Path | str | None = None,
    *,
    profile: str = "default",
    overrides: Sequence[str] = (),
    extra: Sequence[Path | str] = (),
    **kw_overrides: ComposeValue,
) -> BeadConfig:
    """Compose a :class:`BeadConfig` from a profile, file, and overrides.

    Precedence (lowest to highest):

      1. Profile defaults (``bead.config.profiles.get_profile``).
      2. Each path listed in the primary YAML's ``defaults: [...]``
         key, in order.
      3. The primary YAML body.
      4. Each ``extra`` overlay file, in order.
      5. ``overrides`` — dotted-key ``key=value`` strings.
      6. ``kw_overrides`` — legacy ``key__sub=value`` keyword form.
         Each is rewritten as ``"key.sub=value"`` and merged after
         ``overrides``.

    Interpolation is resolved last; the resolved dict is validated as
    a :class:`BeadConfig`.

    Parameters
    ----------
    config_path : Path | str | None, optional
        Primary YAML or TOML file.
    profile : str, optional
        Profile name (``"default"``, ``"dev"``, ``"prod"``,
        ``"test"``).
    overrides : Sequence[str], optional
        CLI-style overrides (``["paths.data_dir=/tmp"]``).
    extra : Sequence[Path | str], optional
        Additional overlay files merged after the primary YAML.
    **kw_overrides : ComposeValue
        Legacy keyword overrides; ``__`` separates nested levels.

    Returns
    -------
    BeadConfig
        Fully composed and validated configuration.
    """
    profile_dict: dict[str, ComposeValue] = json.loads(
        get_profile(profile).model_dump_json()
    )

    all_overrides: list[str] = list(overrides)
    for key, value in kw_overrides.items():
        dotted = key.replace("__", ".")
        # Use yaml.safe_dump to preserve the value's type when it's
        # parsed back in parse_override (e.g. int / float / bool).
        import yaml  # noqa: PLC0415

        all_overrides.append(f"{dotted}={yaml.safe_dump(value).strip()}")

    return compose(
        config_path,
        schema=BeadConfig,
        profile_dict=profile_dict,
        overrides=all_overrides,
        extra=extra,
    )
