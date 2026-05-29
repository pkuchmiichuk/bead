"""Tests for streaming corpus sources."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from bead.corpus.records import CorpusRecord
from bead.corpus.sources import CsvCorpusSource, JsonlCorpusSource
from bead.data.serialization import (
    read_jsonlines,
    stream_jsonlines,
    write_jsonlines,
)

_REDDIT_ROWS: list[dict[str, object]] = [
    {"body": "The dog chased the cat.", "author": "alice", "score": 12},
    {"body": "The dog slept.", "author": "bob", "score": 3},
    {"author": "carol", "score": 1},  # no body: skipped
]


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.write_text(
        "\n".join(json.dumps(row) for row in rows) + "\n", encoding="utf-8"
    )


class TestJsonlCorpusSource:
    """Tests for plain and compressed JSONL ingestion."""

    def test_plain_jsonl(self, tmp_path: Path) -> None:
        path = tmp_path / "reddit.jsonl"
        _write_jsonl(path, _REDDIT_ROWS)
        source = JsonlCorpusSource(
            path, text_field="body", provenance_fields=("author", "score")
        )
        records = list(source)
        assert len(records) == 2  # row without "body" is skipped
        assert records[0].text == "The dog chased the cat."
        assert records[0].source_name == "reddit.jsonl"
        assert records[0].record_index == 0
        assert records[0].provenance == {"author": "alice", "score": 12}
        assert records[1].provenance["author"] == "bob"

    def test_zst_jsonl(self, tmp_path: Path) -> None:
        zstandard = pytest.importorskip("zstandard")
        path = tmp_path / "reddit.jsonl.zst"
        payload = "\n".join(json.dumps(row) for row in _REDDIT_ROWS) + "\n"
        with zstandard.open(path, "wt", encoding="utf-8") as handle:
            handle.write(payload)

        source = JsonlCorpusSource(
            path, text_field="body", provenance_fields=("author",)
        )
        records = list(source)
        assert [r.text for r in records] == [
            "The dog chased the cat.",
            "The dog slept.",
        ]
        assert records[0].provenance == {"author": "alice"}

    def test_custom_source_name(self, tmp_path: Path) -> None:
        path = tmp_path / "data.jsonl"
        _write_jsonl(path, [{"text": "hello"}])
        source = JsonlCorpusSource(path, source_name="my-corpus")
        assert list(source)[0].source_name == "my-corpus"

    def test_is_lazy(self, tmp_path: Path) -> None:
        path = tmp_path / "data.jsonl"
        _write_jsonl(path, [{"text": "a"}, {"text": "b"}, {"text": "c"}])
        source = JsonlCorpusSource(path)
        iterator = iter(source)
        first = next(iterator)
        assert first.text == "a"  # did not consume the whole file


class TestCsvCorpusSource:
    """Tests for CSV/TSV ingestion."""

    def test_csv(self, tmp_path: Path) -> None:
        path = tmp_path / "items.csv"
        path.write_text(
            "sentence,verb,frequency\n"
            "The dog chased the cat.,chase,100\n"
            "The dog slept.,sleep,50\n",
            encoding="utf-8",
        )
        source = CsvCorpusSource(
            path, text_column="sentence", provenance_columns=("verb", "frequency")
        )
        records = list(source)
        assert len(records) == 2
        assert records[0].text == "The dog chased the cat."
        assert records[0].provenance == {"verb": "chase", "frequency": "100"}

    def test_tsv(self, tmp_path: Path) -> None:
        path = tmp_path / "items.tsv"
        path.write_text("sentence\tverb\nHello world.\tnone\n", encoding="utf-8")
        source = CsvCorpusSource(path, text_column="sentence", sep="\t")
        records = list(source)
        assert len(records) == 1
        assert records[0].text == "Hello world."

    def test_skips_empty_text(self, tmp_path: Path) -> None:
        path = tmp_path / "items.csv"
        path.write_text("sentence\nfull\n\nalso full\n", encoding="utf-8")
        source = CsvCorpusSource(path, text_column="sentence")
        assert [r.text for r in source] == ["full", "also full"]


class TestCorpusRecordRoundTrip:
    """CorpusRecord is a BeadBaseModel and round-trips through JSONLines."""

    def test_round_trip(self, tmp_path: Path) -> None:
        records = [
            CorpusRecord(
                text="hello",
                source_name="s",
                record_index=0,
                provenance={"author": "x", "score": 1},
            )
        ]
        path = tmp_path / "records.jsonl"
        write_jsonlines(records, path)
        loaded = read_jsonlines(path, CorpusRecord)
        assert loaded[0].text == "hello"
        assert loaded[0].provenance == {"author": "x", "score": 1}
        # streaming reader (which now shares iter_jsonl_lines) agrees
        streamed = list(stream_jsonlines(path, CorpusRecord))
        assert streamed[0].id == loaded[0].id
