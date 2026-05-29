"""Concrete corpus sources.

Streaming readers that turn external text data into ``CorpusRecord``s:

- ``JsonlCorpusSource`` streams JSON Lines, transparently decompressing
  ``.zst`` (Zstandard) files.
- ``CsvCorpusSource`` streams rows of a CSV/TSV file.

Both are lazy: records are produced one at a time, so multi-gigabyte corpora
never load into memory.
"""

from __future__ import annotations

import json
from collections.abc import Callable, Iterator
from pathlib import Path
from typing import IO

from bead.corpus.records import CorpusRecord, ProvenanceValue
from bead.data.serialization import iter_jsonl_lines


def _as_scalar(value: object) -> ProvenanceValue:
    """Coerce a parsed value to a flat provenance scalar.

    Scalars pass through; anything else (lists, objects) is stringified so the
    provenance dict stays flat.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _zstd_open(path: Path) -> IO[str]:
    """Open a Zstandard-compressed file as a UTF-8 text stream."""
    try:
        import zstandard  # noqa: PLC0415  # type: ignore[reportMissingImports]
    except ImportError as e:
        raise ImportError(
            "zstandard is required to read .zst corpora. "
            "Install it with: pip install 'bead[corpus]'"
        ) from e
    return zstandard.open(path, "rt", encoding="utf-8")  # type: ignore[no-any-return]


class JsonlCorpusSource:
    """Stream JSON Lines (optionally Zstandard-compressed) as corpus records.

    Parameters
    ----------
    path : str | Path
        Path to the ``.jsonl`` or ``.jsonl.zst`` file.
    source_name : str | None
        Source identifier; defaults to the file name.
    text_field : str
        JSON field holding the record text.
    provenance_fields : tuple[str, ...]
        JSON fields to copy into each record's provenance.
    compression : str
        ``"auto"`` (detect ``.zst`` by suffix), ``"zst"``, or ``"none"``.
    """

    def __init__(
        self,
        path: str | Path,
        *,
        source_name: str | None = None,
        text_field: str = "text",
        provenance_fields: tuple[str, ...] = (),
        compression: str = "auto",
    ) -> None:
        self._path = Path(path)
        self.source_name = source_name if source_name is not None else self._path.name
        self._text_field = text_field
        self._provenance_fields = provenance_fields
        self._compression = compression

    def _open_fn(self) -> Callable[[Path], IO[str]] | None:
        compressed = self._compression == "zst" or (
            self._compression == "auto" and self._path.suffix == ".zst"
        )
        return _zstd_open if compressed else None

    def __iter__(self) -> Iterator[CorpusRecord]:
        """Yield one ``CorpusRecord`` per non-empty JSON line."""
        open_fn = self._open_fn()
        line_iter = (
            iter_jsonl_lines(self._path, open_fn=open_fn)
            if open_fn is not None
            else iter_jsonl_lines(self._path)
        )
        for index, (_line_num, line) in enumerate(line_iter):
            data = json.loads(line)
            if not isinstance(data, dict):
                continue
            raw_text = data.get(self._text_field)
            if raw_text is None:
                continue
            provenance: dict[str, ProvenanceValue] = {
                field: _as_scalar(data[field])
                for field in self._provenance_fields
                if field in data
            }
            yield CorpusRecord(
                text=str(raw_text),
                source_name=self.source_name,
                record_index=index,
                provenance=provenance,
            )


class CsvCorpusSource:
    r"""Stream rows of a CSV/TSV file as corpus records.

    Parameters
    ----------
    path : str | Path
        Path to the CSV/TSV file.
    text_column : str
        Column holding the record text.
    source_name : str | None
        Source identifier; defaults to the file name.
    provenance_columns : tuple[str, ...]
        Columns to copy into each record's provenance.
    sep : str
        Field separator (``","`` for CSV, ``"\\t"`` for TSV).
    """

    def __init__(
        self,
        path: str | Path,
        *,
        text_column: str,
        source_name: str | None = None,
        provenance_columns: tuple[str, ...] = (),
        sep: str = ",",
    ) -> None:
        self._path = Path(path)
        self.source_name = source_name if source_name is not None else self._path.name
        self._text_column = text_column
        self._provenance_columns = provenance_columns
        self._sep = sep

    def __iter__(self) -> Iterator[CorpusRecord]:
        """Yield one ``CorpusRecord`` per CSV row with a non-empty text cell."""
        import pandas as pd  # noqa: PLC0415

        frame = pd.read_csv(self._path, sep=self._sep, dtype=str, keep_default_na=False)
        for index, row in enumerate(frame.to_dict(orient="records")):
            raw_text = row.get(self._text_column, "")
            if raw_text is None or str(raw_text) == "":
                continue
            provenance: dict[str, ProvenanceValue] = {
                column: _as_scalar(row[column])
                for column in self._provenance_columns
                if column in row
            }
            yield CorpusRecord(
                text=str(raw_text),
                source_name=self.source_name,
                record_index=index,
                provenance=provenance,
            )
