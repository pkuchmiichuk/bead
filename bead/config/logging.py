"""Logging configuration models."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import didactic.api as dx


class LoggingConfig(dx.Model):
    """Configuration for logging.

    Attributes
    ----------
    level : Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]
        Log level.
    format : str
        Log format string.
    file : Path | None
        Log file path.
    console : bool
        Whether to log to console.
    """

    level: Literal["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"] = "INFO"
    format: str = "%(asctime)s - %(name)s - %(levelname)s - %(message)s"
    file: Path | None = None
    console: bool = True
