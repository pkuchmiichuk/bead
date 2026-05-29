"""Round-trip law tests for the ParsedSentence <-> layers annotation iso."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from bead.interop.layers.parse_lens import PARSED_SENTENCE_LAYERS, parse_to_layers
from bead.tokenization.parsers import ParsedSentence, ParsedToken

ISO = PARSED_SENTENCE_LAYERS


def _assert_roundtrip(sentence: ParsedSentence) -> None:
    view = ISO.forward(sentence)
    assert ISO.backward(view) == sentence
    # PutGet: re-projecting the reconstruction yields the same view.
    assert ISO.forward(ISO.backward(view)) == view


def _known_sentence() -> ParsedSentence:
    return ParsedSentence(
        original_text="The dog chased the cat",
        tokens=(
            ParsedToken(index=0, text="The", lemma="the", upos="DET", xpos="DT",
                        deprel="det", head=1, start_char=0, end_char=3),
            ParsedToken(index=1, text="dog", lemma="dog", upos="NOUN", xpos="NN",
                        deprel="nsubj", head=2, morph={"Number": "Sing"},
                        start_char=4, end_char=7),
            ParsedToken(index=2, text="chased", lemma="chase", upos="VERB",
                        xpos="VBD", deprel="root", head=None,
                        morph={"Tense": "Past"}, start_char=8, end_char=14),
            ParsedToken(index=3, text="the", lemma="the", upos="DET", xpos="DT",
                        deprel="det", head=4, start_char=15, end_char=18),
            ParsedToken(index=4, text="cat", lemma="cat", upos="NOUN", xpos="NN",
                        deprel="obj", head=2, start_char=19, end_char=22),
        ),
    )


class TestExampleRoundTrips:
    """Deterministic round-trips over representative parses."""

    def test_full_parse(self) -> None:
        _assert_roundtrip(_known_sentence())

    def test_root_head_minus_one(self) -> None:
        view = parse_to_layers(_known_sentence())
        # the root token (index 2) is encoded with headIndex -1
        dep = view["dependencyLayer"]["annotations"][2]
        assert dep["headIndex"] == -1
        assert dep["label"] == "root"

    def test_view_is_layers_shaped(self) -> None:
        view = parse_to_layers(_known_sentence())
        assert set(view) == {
            "originalText",
            "tokenization",
            "posLayer",
            "dependencyLayer",
        }
        assert view["posLayer"]["subkind"] == "pos"
        assert view["dependencyLayer"]["subkind"] == "dependency"
        assert view["tokenization"]["tokens"][0]["textSpan"] == {
            "byteStart": 0,
            "byteEnd": 3,
            "charStart": 0,
            "charEnd": 3,
        }

    def test_missing_optionals(self) -> None:
        _assert_roundtrip(
            ParsedSentence(
                original_text="x",
                tokens=(ParsedToken(index=0, text="x", start_char=0, end_char=1),),
            )
        )


_morph = st.dictionaries(
    st.text(alphabet="AB", min_size=1, max_size=2),
    st.text(alphabet="xy", min_size=1, max_size=2),
    max_size=2,
)
_opt = st.one_of(st.none(), st.text(alphabet="pq", min_size=1, max_size=3))


@st.composite
def _sentences(draw: st.DrawFn) -> ParsedSentence:
    n = draw(st.integers(0, 5))
    tokens = tuple(
        ParsedToken(
            index=i,
            text=draw(st.text(max_size=5)),
            lemma=draw(_opt),
            upos=draw(_opt),
            xpos=draw(_opt),
            deprel=draw(_opt),
            head=draw(st.one_of(st.none(), st.integers(0, max(n - 1, 0)))),
            morph=draw(_morph),
            space_after=draw(st.booleans()),
            start_char=draw(st.integers(0, 50)),
            end_char=draw(st.integers(0, 50)),
        )
        for i in range(n)
    )
    return ParsedSentence(original_text=draw(st.text(max_size=20)), tokens=tokens)


@given(_sentences())
def test_iso_round_trip_law(sentence: ParsedSentence) -> None:
    assert ISO.backward(ISO.forward(sentence)) == sentence
