"""Concrete corpus sources.

Streaming readers that turn external text data into ``CorpusRecord``s:

- ``JsonlCorpusSource`` streams JSON Lines, transparently decompressing
  ``.zst`` (Zstandard) files.
- ``CsvCorpusSource`` streams rows of a CSV/TSV file.

Both are lazy: records are produced one at a time, so multi-gigabyte corpora
never load into memory.
"""

from __future__ import annotations

import importlib
import json
from collections.abc import Callable, Iterator, Sequence
from pathlib import Path
from typing import IO, TYPE_CHECKING

import pandas as pd

from bead.corpus.records import CorpusRecord, ProvenanceValue
from bead.data.serialization import iter_jsonl_lines

if TYPE_CHECKING:
    from bead.items.adapters.base import TextGenerator

# A value parsed from JSON or a CSV cell (lists, unlike bead's tuple-based
# JsonValue, since json.loads produces lists).
type JsonInput = (
    str | int | float | bool | None | list["JsonInput"] | dict[str, "JsonInput"]
)


def _as_scalar(value: JsonInput) -> ProvenanceValue:
    """Coerce a parsed value to a flat provenance scalar.

    Scalars pass through; anything else (lists, nested objects) is stringified
    so the provenance dict stays flat.
    """
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    return str(value)


def _zstd_open(path: Path) -> IO[str]:
    """Open a Zstandard-compressed file as a UTF-8 text stream."""
    try:
        zstandard = importlib.import_module("zstandard")
    except ImportError as e:
        raise ImportError(
            "zstandard is required to read .zst corpora. "
            "Install it with: pip install 'bead[corpus]'"
        ) from e
    return zstandard.open(path, "rt", encoding="utf-8")


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
        for index, (_, line) in enumerate(line_iter):
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


class CompletionCorpusSource:
    """Generate text from a language model as a corpus source.

    Wraps any ``TextGenerator`` (e.g. an OpenAI or Anthropic adapter) and yields
    one ``CorpusRecord`` per generated completion, with the model and prompt
    recorded as layers-aligned provenance.

    Parameters
    ----------
    generator : TextGenerator
        The model used to generate completions.
    prompts : Sequence[str]
        Prompts to complete.
    source_name : str | None
        Source identifier; defaults to the generator's ``model_name``.
    completions_per_prompt : int
        Number of completions to draw per prompt.
    max_tokens : int
        Maximum tokens per completion.
    temperature : float
        Sampling temperature.
    """

    def __init__(
        self,
        generator: TextGenerator,
        prompts: Sequence[str],
        *,
        source_name: str | None = None,
        completions_per_prompt: int = 1,
        max_tokens: int = 256,
        temperature: float = 1.0,
    ) -> None:
        self._generator = generator
        self._prompts = prompts
        self.source_name = (
            source_name if source_name is not None else generator.model_name
        )
        self._completions_per_prompt = completions_per_prompt
        self._max_tokens = max_tokens
        self._temperature = temperature

    def __iter__(self) -> Iterator[CorpusRecord]:
        """Yield one ``CorpusRecord`` per generated completion."""
        index = 0
        for prompt in self._prompts:
            for _ in range(self._completions_per_prompt):
                text = self._generator.generate_completion(
                    prompt,
                    max_tokens=self._max_tokens,
                    temperature=self._temperature,
                )
                provenance: dict[str, ProvenanceValue] = {
                    "tool": "completion",
                    "model": self._generator.model_name,
                    "prompt": prompt,
                }
                yield CorpusRecord(
                    text=text,
                    source_name=self.source_name,
                    record_index=index,
                    provenance=provenance,
                )
                index += 1


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
