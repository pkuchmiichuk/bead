"""Tests for dependency parsing and span projection."""

from __future__ import annotations

import pytest

from bead.tokenization.config import TokenizerConfig
from bead.tokenization.parsers import (
    UNIVERSAL_DEPENDENCIES,
    ParsedSentence,
    ParsedToken,
    StanzaParser,
    _parse_feats,
    create_parser,
    parse_to_spans,
)


def _known_sentence() -> ParsedSentence:
    """A hand-built parse of 'The dog chased the cat' (UD-style)."""
    return ParsedSentence(
        original_text="The dog chased the cat",
        tokens=(
            ParsedToken(
                index=0, text="The", lemma="the", upos="DET", xpos="DT",
                deprel="det", head=1, start_char=0, end_char=3,
            ),
            ParsedToken(
                index=1, text="dog", lemma="dog", upos="NOUN", xpos="NN",
                deprel="nsubj", head=2, morph={"Number": "Sing"},
                start_char=4, end_char=7,
            ),
            ParsedToken(
                index=2, text="chased", lemma="chase", upos="VERB", xpos="VBD",
                deprel="root", head=None, morph={"Tense": "Past"},
                start_char=8, end_char=14,
            ),
            ParsedToken(
                index=3, text="the", lemma="the", upos="DET", xpos="DT",
                deprel="det", head=4, start_char=15, end_char=18,
            ),
            ParsedToken(
                index=4, text="cat", lemma="cat", upos="NOUN", xpos="NN",
                deprel="obj", head=2, morph={"Number": "Sing"},
                start_char=19, end_char=22,
            ),
        ),
    )


class TestParseFeats:
    """Tests for CoNLL-U feature parsing."""

    def test_empty(self) -> None:
        assert _parse_feats(None) == {}
        assert _parse_feats("_") == {}

    def test_parse(self) -> None:
        assert _parse_feats("Number=Sing|Tense=Past") == {
            "Number": "Sing",
            "Tense": "Past",
        }

    def test_skips_malformed(self) -> None:
        assert _parse_feats("Number=Sing|garbage") == {"Number": "Sing"}


class TestParseToSpans:
    """Tests for projecting a parse onto spans and relations."""

    def test_one_token_span_per_token(self) -> None:
        spans, _ = parse_to_spans(
            _known_sentence(), tokenization_id="tok-1", tool="test"
        )
        assert len(spans) == 5
        assert all(s.span_type == "token" for s in spans)
        assert all(len(s.segments) == 1 for s in spans)
        assert [s.segments[0].indices[0] for s in spans] == [0, 1, 2, 3, 4]

    def test_span_ids_and_metadata(self) -> None:
        spans, _ = parse_to_spans(
            _known_sentence(),
            element_name="text",
            tokenization_id="tok-1",
            tool="stanza",
        )
        chased = spans[2]
        assert chased.span_id == "text:tok:2"
        assert chased.head_index is None  # root
        assert chased.label is not None
        assert chased.label.label == "VERB"
        assert chased.span_metadata["upos"] == "VERB"
        assert chased.span_metadata["xpos"] == "VBD"
        assert chased.span_metadata["lemma"] == "chase"
        assert chased.span_metadata["deprel"] == "root"
        assert chased.span_metadata["formalism"] == UNIVERSAL_DEPENDENCIES
        assert chased.span_metadata["tool"] == "stanza"
        assert chased.span_metadata["tokenization_id"] == "tok-1"
        assert chased.span_metadata["morph"] == {"Tense": "Past"}
        assert chased.span_metadata["start_char"] == 8
        assert chased.span_metadata["end_char"] == 14

    def test_head_index_is_governor(self) -> None:
        spans, _ = parse_to_spans(
            _known_sentence(), tokenization_id="tok-1", tool="test"
        )
        # token 0 ("The") is governed by token 1 ("dog")
        assert spans[0].head_index == 1
        # token 1 ("dog") is governed by token 2 ("chased")
        assert spans[1].head_index == 2

    def test_relations_are_head_to_dependent(self) -> None:
        _, relations = parse_to_spans(
            _known_sentence(),
            element_name="text",
            tokenization_id="tok-1",
            tool="test",
        )
        # 4 arcs (every token except the root)
        assert len(relations) == 4
        arcs = {
            (r.source_span_id, r.target_span_id): (r.label.label if r.label else None)
            for r in relations
        }
        # head ("chased" = tok:2) -> dependent ("dog" = tok:1), labeled nsubj
        assert arcs[("text:tok:2", "text:tok:1")] == "nsubj"
        assert arcs[("text:tok:2", "text:tok:4")] == "obj"
        assert arcs[("text:tok:1", "text:tok:0")] == "det"
        assert arcs[("text:tok:4", "text:tok:3")] == "det"
        assert all(r.directed for r in relations)

    def test_root_has_no_relation(self) -> None:
        _, relations = parse_to_spans(
            _known_sentence(), tokenization_id="tok-1", tool="test"
        )
        targets = {r.target_span_id for r in relations}
        assert "text:tok:2" not in targets  # root is never a dependent


class TestCreateParser:
    """Tests for parser construction."""

    def test_whitespace_cannot_parse(self) -> None:
        with pytest.raises(ValueError, match="cannot produce a dependency parse"):
            create_parser(TokenizerConfig(backend="whitespace"))

    def test_spacy_and_stanza_construct(self) -> None:
        # Construction is lazy; no model is loaded here.
        assert create_parser(TokenizerConfig(backend="spacy")) is not None
        assert create_parser(TokenizerConfig(backend="stanza")) is not None


def _require_stanza_en() -> None:
    """Skip only if Stanza or its English model cannot be obtained.

    Once the model is present, callers run the real parse so genuine parse or
    projection bugs surface as failures rather than being skipped.
    """
    stanza = pytest.importorskip("stanza")
    try:
        stanza.download(
            "en", processors="tokenize,pos,lemma,depparse", verbose=False
        )
    except Exception as exc:  # pragma: no cover - network dependent
        pytest.skip(f"Stanza English model unavailable (no network?): {exc}")


class TestStanzaParserIntegration:
    """End-to-end parse via a real Stanza model (not skipped when available)."""

    def test_parse_transitive_sentence(self) -> None:
        _require_stanza_en()
        # Real parse; errors here are genuine failures, not skips.
        sentences = StanzaParser(language="en")("The dog chased the cat.")

        assert len(sentences) == 1
        tokens = sentences[0].tokens
        roots = [t for t in tokens if t.head is None]
        assert len(roots) == 1
        assert roots[0].upos == "VERB"
        assert roots[0].lemma == "chase"
        obj = [t for t in tokens if t.deprel == "obj" and t.head == roots[0].index]
        assert obj, "expected an object dependent of the root verb"

    def test_parse_projects_to_spans(self) -> None:
        _require_stanza_en()
        sentences = StanzaParser(language="en")("The dog chased the cat.")
        spans, relations = parse_to_spans(
            sentences[0], tokenization_id="tok-1", tool="stanza"
        )
        assert len(spans) == len(sentences[0].tokens)
        # exactly one root (no incoming arc); every other token has one
        assert len(relations) == len(spans) - 1
        assert all(s.span_metadata["tool"] == "stanza" for s in spans)
