"""Tests for the streaming corpus pipeline."""

from __future__ import annotations

from uuid import uuid4

import pytest

from bead.corpus.pipeline import (
    filter_by_structure,
    parse_records,
    record_to_item,
    sample_corpus,
)
from bead.corpus.records import CorpusRecord
from bead.tokenization.parsers import ParsedSentence, ParsedToken, StanzaParser

# A structural constraint: root is a verb that takes a direct object.
TRANSITIVE = (
    'upos(self, root(self)) == "VERB" and len(dependents(self, root(self), "obj")) > 0'
)


def _transitive_parse() -> ParsedSentence:
    return ParsedSentence(
        original_text="The dog chased the cat",
        tokens=(
            ParsedToken(index=0, text="The", upos="DET", deprel="det", head=1),
            ParsedToken(index=1, text="dog", upos="NOUN", deprel="nsubj", head=2),
            ParsedToken(index=2, text="chased", upos="VERB", deprel="root", head=None),
            ParsedToken(index=3, text="the", upos="DET", deprel="det", head=4),
            ParsedToken(index=4, text="cat", upos="NOUN", deprel="obj", head=2),
        ),
    )


def _intransitive_parse() -> ParsedSentence:
    return ParsedSentence(
        original_text="The dog slept",
        tokens=(
            ParsedToken(index=0, text="The", upos="DET", deprel="det", head=1),
            ParsedToken(index=1, text="dog", upos="NOUN", deprel="nsubj", head=2),
            ParsedToken(index=2, text="slept", upos="VERB", deprel="root", head=None),
        ),
    )


class _StubParser:
    """A deterministic parser keyed on text, satisfying DependencyParser."""

    tool = "stub"

    def __init__(self, mapping: dict[str, tuple[ParsedSentence, ...]]) -> None:
        self._mapping = mapping

    def __call__(self, text: str) -> tuple[ParsedSentence, ...]:
        return self._mapping[text]


def _records() -> list[CorpusRecord]:
    return [
        CorpusRecord(
            text="The dog chased the cat",
            source_name="corpus",
            record_index=0,
            provenance={"author": "alice"},
        ),
        CorpusRecord(
            text="The dog slept",
            source_name="corpus",
            record_index=1,
            provenance={"author": "bob"},
        ),
    ]


def _parser() -> _StubParser:
    return _StubParser(
        {
            "The dog chased the cat": (_transitive_parse(),),
            "The dog slept": (_intransitive_parse(),),
        }
    )


class TestRecordToItem:
    """Tests for building an item from a record and its parse."""

    def test_builds_item_with_provenance(self) -> None:
        template_id = uuid4()
        record = _records()[0]
        item = record_to_item(
            record, _transitive_parse(), item_template_id=template_id, tool="stub"
        )
        assert item.item_template_id == template_id
        assert item.rendered_elements["text"] == "The dog chased the cat"
        assert len(item.spans) == 5
        assert len(item.span_relations) == 4
        # layers-aligned + source provenance on item_metadata
        assert item.item_metadata["author"] == "alice"
        assert item.item_metadata["source_name"] == "corpus"
        assert item.item_metadata["parser_tool"] == "stub"
        assert item.item_metadata["subkind"] == "dependency"
        assert item.item_metadata["corpus_record_id"] == str(record.id)
        assert item.tokenized_elements["text"] == (
            "The",
            "dog",
            "chased",
            "the",
            "cat",
        )


class TestParseRecords:
    """Tests for parsing records into sentence pairs."""

    def test_one_pair_per_sentence(self) -> None:
        multi = CorpusRecord(text="multi", source_name="c")
        parser = _StubParser({"multi": (_transitive_parse(), _intransitive_parse())})
        pairs = list(parse_records([multi], parser))
        assert len(pairs) == 2

    def test_split_sentences_false_skips_multi(self) -> None:
        multi = CorpusRecord(text="multi", source_name="c")
        single = CorpusRecord(text="single", source_name="c")
        parser = _StubParser(
            {
                "multi": (_transitive_parse(), _intransitive_parse()),
                "single": (_transitive_parse(),),
            }
        )
        pairs = list(parse_records([multi, single], parser, split_sentences=False))
        assert len(pairs) == 1
        assert pairs[0][0].text == "single"


class TestFilterByStructure:
    """Tests for structural rejection sampling."""

    def test_keeps_only_transitive(self) -> None:
        pairs = list(parse_records(_records(), _parser()))
        items = list(
            filter_by_structure(
                pairs, TRANSITIVE, item_template_id=uuid4(), tool="stub"
            )
        )
        assert len(items) == 1
        assert items[0].rendered_elements["text"] == "The dog chased the cat"


class TestSampleCorpus:
    """Tests for the end-to-end convenience generator."""

    def test_filters_and_builds_items(self) -> None:
        items = list(
            sample_corpus(
                _records(),
                _parser(),
                TRANSITIVE,
                item_template_id=uuid4(),
            )
        )
        assert len(items) == 1
        assert items[0].item_metadata["author"] == "alice"

    def test_limit(self) -> None:
        # both records match a trivially-true constraint; limit caps output
        items = list(
            sample_corpus(
                _records(),
                _parser(),
                "root(self) >= 0",
                item_template_id=uuid4(),
                limit=1,
            )
        )
        assert len(items) == 1


class TestSampleCorpusStanzaIntegration:
    """End-to-end with a real Stanza parser (skips only if model unavailable)."""

    def test_filters_transitive_with_real_parser(self) -> None:
        stanza = pytest.importorskip("stanza")
        try:
            stanza.download(
                "en", processors="tokenize,pos,lemma,depparse", verbose=False
            )
        except Exception as exc:  # pragma: no cover - network dependent
            pytest.skip(f"Stanza English model unavailable (no network?): {exc}")

        records = [
            CorpusRecord(text="The dog chased the cat.", source_name="c"),
            CorpusRecord(text="The dog slept peacefully.", source_name="c"),
            CorpusRecord(text="She wrote a long letter.", source_name="c"),
        ]
        items = list(
            sample_corpus(
                records,
                StanzaParser(language="en"),
                TRANSITIVE,
                item_template_id=uuid4(),
            )
        )
        kept = {item.rendered_elements["text"] for item in items}
        assert kept == {"The dog chased the cat.", "She wrote a long letter."}
        assert all(it.item_metadata["parser_tool"] == "stanza" for it in items)
