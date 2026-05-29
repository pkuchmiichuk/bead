"""JSONLines serialization utilities for didactic Models.

Functions for reading, writing, streaming, and appending didactic Models to
and from JSONLines format files.
"""

from __future__ import annotations

from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import IO

import didactic.api as dx


class SerializationError(Exception):
    """Raised when serialization to JSONLines fails."""


class DeserializationError(Exception):
    """Raised when deserialization from JSONLines fails."""


def _open_text(path: Path) -> IO[str]:
    """Open *path* as a UTF-8 text stream (default JSONL line opener)."""
    return path.open("r", encoding="utf-8")


def iter_jsonl_lines(
    path: Path,
    *,
    open_fn: Callable[[Path], IO[str]] = _open_text,
) -> Iterator[tuple[int, str]]:
    """Yield ``(line_number, stripped_line)`` for each non-empty line.

    Single canonical line-iteration step shared by the JSONLines readers and
    by streaming corpus sources (which pass a decompressing ``open_fn``).

    Parameters
    ----------
    path : Path
        File to read.
    open_fn : Callable[[Path], IO[str]]
        Opener returning a text stream; defaults to UTF-8 text. Pass a
        decompressing opener (e.g. ``zstandard.open``) for compressed files.

    Yields
    ------
    tuple[int, str]
        1-based line number and the stripped line (blank lines skipped).
    """
    with open_fn(path) as handle:
        for line_num, line in enumerate(handle, start=1):
            stripped = line.strip()
            if stripped:
                yield line_num, stripped


def write_jsonlines[T: dx.Model](
    objects: Sequence[T],
    path: Path | str,
    validate: bool = True,
    append: bool = False,
) -> None:
    """Write *objects* to *path* as JSONLines.

    Parameters
    ----------
    objects
        Models to serialize.
    path
        Output file path.
    validate
        Unused; retained for API compatibility.
    append
        Whether to append to an existing file.

    Raises
    ------
    SerializationError
        If writing fails.
    """
    del validate
    path = Path(path)
    mode = "a" if append else "w"
    try:
        with path.open(mode, encoding="utf-8") as f:
            for obj in objects:
                f.write(obj.model_dump_json() + "\n")
    except (OSError, dx.ValidationError) as e:
        raise SerializationError(f"Failed to write to {path}: {e}") from e


def read_jsonlines[T: dx.Model](
    path: Path | str,
    model_class: type[T],
    validate: bool = True,
    skip_errors: bool = False,
) -> list[T]:
    """Read JSONLines from *path* into a list of *model_class* instances."""
    del validate
    path = Path(path)
    objects: list[T] = []
    try:
        for line_num, line in iter_jsonl_lines(path):
            try:
                objects.append(model_class.model_validate_json(line))
            except (dx.ValidationError, ValueError) as e:
                if skip_errors:
                    continue
                raise DeserializationError(
                    f"Failed to parse line {line_num} in {path}: {e}"
                ) from e
    except OSError as e:
        raise DeserializationError(f"Failed to read from {path}: {e}") from e
    return objects


def stream_jsonlines[T: dx.Model](
    path: Path | str,
    model_class: type[T],
    validate: bool = True,
) -> Iterator[T]:
    """Yield *model_class* instances from *path* one line at a time."""
    del validate
    path = Path(path)
    try:
        for line_num, line in iter_jsonl_lines(path):
            try:
                yield model_class.model_validate_json(line)
            except (dx.ValidationError, ValueError) as e:
                raise DeserializationError(
                    f"Failed to parse line {line_num} in {path}: {e}"
                ) from e
    except OSError as e:
        raise DeserializationError(f"Failed to read from {path}: {e}") from e


def append_jsonlines[T: dx.Model](
    objects: Sequence[T],
    path: Path | str,
    validate: bool = True,
) -> None:
    """Append *objects* to *path* as JSONLines."""
    write_jsonlines(objects, path, validate=validate, append=True)
