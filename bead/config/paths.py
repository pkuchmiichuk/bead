"""Path configuration models."""

from __future__ import annotations

from pathlib import Path

import didactic.api as dx


def _default_data_dir() -> Path:
    return Path("data")


def _default_output_dir() -> Path:
    return Path("output")


def _default_cache_dir() -> Path:
    return Path(".cache")


class PathsConfig(dx.Model):
    """Configuration for filesystem paths.

    Attributes
    ----------
    data_dir : Path
        Base directory for data files.
    output_dir : Path
        Base directory for outputs.
    cache_dir : Path
        Cache directory.
    temp_dir : Path | None
        Temporary directory; ``None`` defers to the system default.
    create_dirs : bool
        Whether to create directories if they don't exist.
    """

    data_dir: Path = dx.field(default_factory=_default_data_dir)
    output_dir: Path = dx.field(default_factory=_default_output_dir)
    cache_dir: Path = dx.field(default_factory=_default_cache_dir)
    temp_dir: Path | None = None
    create_dirs: bool = True
