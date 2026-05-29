"""Tests for streaming corpus sources."""

from __future__ import annotations

import json
from collections.abc import Mapping, Sequence
from pathlib import Path

import pytest

from bead.corpus.records import CorpusRecord
from bead.corpus.sources import (
    CompletionCorpusSource,
    CsvCorpusSource,
    JsonlCorpusSource,
)
from bead.data.serialization import (
    read_jsonlines,
    stream_jsonlines,
    write_jsonlines,
)

type _Json = str | int | float | bool | None | list["_Json"] | dict[str, "_Json"]

_REDDIT_ROWS: list[dict[str, _Json]] = [
    {"body": "The dog chased the cat.", "author": "alice", "score": 12},
    {"body": "The dog slept.", "author": "bob", "score": 3},
    {"author": "carol", "score": 1},  # no body: skipped
]


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, _Json]]) -> None:
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

    def test_retains_all_fields_by_default(self, tmp_path: Path) -> None:
        # The default must not drop ANY field - thread edges (parent_id,
        # link_id) survive without being enumerated, so structure is
        # recoverable downstream.
        path = tmp_path / "reddit.jsonl"
        rows: list[dict[str, _Json]] = [
            {
                "body": "a reply",
                "id": "t1_aaa",
                "parent_id": "t1_root",
                "link_id": "t3_sub",
                "author": "alice",
                "score": 4,
            }
        ]
        _write_jsonl(path, rows)
        record = next(iter(JsonlCorpusSource(path, text_field="body")))
        # every field except the text field is retained
        assert record.provenance == {
            "id": "t1_aaa",
            "parent_id": "t1_root",
            "link_id": "t3_sub",
            "author": "alice",
            "score": 4,
        }
        assert "body" not in record.provenance

    def test_nested_values_round_trip(self, tmp_path: Path) -> None:
        # Non-scalar fields are JSON-serialized (not str()-ified), so they
        # remain recoverable via json.loads.
        path = tmp_path / "nested.jsonl"
        rows: list[dict[str, _Json]] = [
            {"text": "hi", "edits": [1, 2], "meta": {"k": "v"}}
        ]
        _write_jsonl(path, rows)
        record = next(iter(JsonlCorpusSource(path)))
        assert json.loads(str(record.provenance["edits"])) == [1, 2]
        assert json.loads(str(record.provenance["meta"])) == {"k": "v"}


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


class _StubGenerator:
    """A deterministic text generator satisfying TextGenerator."""

    model_name = "stub-model"

    def __init__(self, mapping: dict[str, str]) -> None:
        self._mapping = mapping
        self.calls: list[tuple[str, int, float]] = []

    def generate_completion(
        self, prompt: str, *, max_tokens: int = 256, temperature: float = 1.0
    ) -> str:
        self.calls.append((prompt, max_tokens, temperature))
        return self._mapping[prompt]


class TestCompletionCorpusSource:
    """Tests for generating a corpus from a language model."""

    def test_yields_one_record_per_completion(self) -> None:
        generator = _StubGenerator(
            {"Write a sentence.": "The dog barked.", "Another one.": "Cats sleep."}
        )
        source = CompletionCorpusSource(
            generator, ["Write a sentence.", "Another one."]
        )
        records = list(source)
        assert [r.text for r in records] == ["The dog barked.", "Cats sleep."]
        assert records[0].source_name == "stub-model"
        assert records[0].provenance["model"] == "stub-model"
        assert records[0].provenance["tool"] == "completion"
        assert records[0].provenance["prompt"] == "Write a sentence."
        assert records[1].record_index == 1

    def test_completions_per_prompt(self) -> None:
        generator = _StubGenerator({"p": "out"})
        source = CompletionCorpusSource(
            generator, ["p"], completions_per_prompt=3, max_tokens=10, temperature=0.5
        )
        records = list(source)
        assert len(records) == 3
        assert generator.calls == [("p", 10, 0.5)] * 3


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
