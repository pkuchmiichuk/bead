"""Resource configuration."""

from __future__ import annotations

from pathlib import Path

import didactic.api as dx


class ResourceConfig(dx.Model):
    """Configuration for external resources.

    Attributes
    ----------
    lexicon_path : Path | None
        Path to lexicon file.
    templates_path : Path | None
        Path to templates file.
    constraints_path : Path | None
        Path to constraints file.
    external_adapters : tuple[str, ...]
        External adapters to enable.
    cache_external : bool
        Whether to cache external resource lookups.
    """

    lexicon_path: Path | None = None
    templates_path: Path | None = None
    constraints_path: Path | None = None
    external_adapters: tuple[str, ...] = ()
    cache_external: bool = True
