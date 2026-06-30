"""File-format dispatch for the compose pipeline.

Loads YAML (``.yaml`` / ``.yml``) and TOML (``.toml``) files into the
dict-of-:data:`~bead.config.compose.interpolation.ComposeValue` shape
the rest of the subpackage operates on.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

import yaml

from bead.config.compose.errors import ConfigError
from bead.config.compose.interpolation import ComposeValue

_YAML_SUFFIXES = frozenset({".yaml", ".yml"})
_TOML_SUFFIXES = frozenset({".toml"})


def load_one(path: Path | str) -> dict[str, ComposeValue]:
    """Load a YAML or TOML file as a dict.

    Parameters
    ----------
    path : Path | str
        Path to a ``.yaml`` / ``.yml`` / ``.toml`` file.

    Returns
    -------
    dict[str, ComposeValue]
        Loaded content. An empty file yields an empty dict.

    Raises
    ------
    FileNotFoundError
        If ``path`` does not exist.
    ConfigError
        If the suffix is unrecognized or the parsed content is not a
        top-level mapping.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Configuration file not found: {path}")
    suffix = path.suffix.lower()

    if suffix in _YAML_SUFFIXES:
        with path.open(encoding="utf-8") as fp:
            loaded = yaml.safe_load(fp)
    elif suffix in _TOML_SUFFIXES:
        with path.open("rb") as fp_b:
            loaded = tomllib.load(fp_b)
    else:
        raise ConfigError(
            f"Unsupported config suffix {suffix!r} for {path}. "
            f"Expected one of: {sorted(_YAML_SUFFIXES | _TOML_SUFFIXES)}"
        )

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(
            f"Top-level config in {path} must be a mapping; got {type(loaded).__name__}"
        )
    return loaded


def resolve_defaults_entry(entry: str, *, anchor: Path) -> Path:
    """Resolve a ``defaults: [...]`` entry to a concrete file path.

    Entries name a file relative to ``anchor`` (the parent of the
    YAML containing the ``defaults`` list). An entry may include or
    omit the suffix:

    - ``protocol/argument_structure`` is resolved to whichever of
      ``protocol/argument_structure.yaml``,
      ``protocol/argument_structure.yml``, or
      ``protocol/argument_structure.toml`` exists.
    - ``protocol/argument_structure.yaml`` is taken verbatim.

    Parameters
    ----------
    entry : str
        Path string from the YAML ``defaults`` list.
    anchor : Path
        Directory the entry is resolved against.

    Returns
    -------
    Path
        Existing path on disk.

    Raises
    ------
    ConfigError
        If no candidate path exists.
    """
    raw = (anchor / entry).resolve()
    if raw.suffix.lower() in _YAML_SUFFIXES | _TOML_SUFFIXES:
        if raw.exists():
            return raw
        raise ConfigError(f"defaults entry {entry!r} not found at {raw}")
    for suffix in (".yaml", ".yml", ".toml"):
        candidate = raw.with_suffix(suffix)
        if candidate.exists():
            return candidate
    raise ConfigError(
        f"defaults entry {entry!r} not found; tried "
        f"{raw.with_suffix('.yaml')}, "
        f"{raw.with_suffix('.yml')}, "
        f"{raw.with_suffix('.toml')}"
    )
