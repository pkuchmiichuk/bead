"""Round-trip law tests for the CorpusRecord <-> layers expression lens."""

from __future__ import annotations

from hypothesis import given
from hypothesis import strategies as st

from bead.corpus.records import CorpusRecord
from bead.interop.layers.bridges import RECORD_EXPRESSION, record_to_expression

LENS = RECORD_EXPRESSION


def _assert_roundtrip(record: CorpusRecord) -> None:
    view, complement = LENS.forward(record)
    assert LENS.backward(view, complement) == record
    view2, complement2 = LENS.forward(LENS.backward(view, complement))
    assert (view2, complement2) == (view, complement)


class TestExampleRoundTrips:
    """Deterministic round-trips over representative records."""

    def test_minimal(self) -> None:
        _assert_roundtrip(CorpusRecord(text="hello", source_name="s"))

    def test_with_scalar_provenance(self) -> None:
        _assert_roundtrip(
            CorpusRecord(
                text="a reply",
                source_name="reddit",
                record_index=3,
                provenance={"author": "alice", "score": 5, "deleted": False},
            )
        )

    def test_view_is_layers_expression(self) -> None:
        view = record_to_expression(
            CorpusRecord(text="hi", source_name="s", provenance={"k": "v"})
        )
        assert view["kind"] == "expression"
        assert view["text"] == "hi"
        assert view["features"]["entries"][0] == {"key": "k", "value": '"v"'}


_scalar = st.one_of(st.text(max_size=6), st.integers(-50, 50), st.booleans(), st.none())


@given(
    text=st.text(max_size=20),
    source_name=st.text(max_size=8),
    record_index=st.integers(0, 1000),
    provenance=st.dictionaries(
        st.text(alphabet="abc", min_size=1, max_size=3), _scalar, max_size=4
    ),
)
def test_get_put_law(
    text: str,
    source_name: str,
    record_index: int,
    provenance: dict[str, str | int | bool | None],
) -> None:
    record = CorpusRecord(
        text=text,
        source_name=source_name,
        record_index=record_index,
        provenance=provenance,
    )
    view, complement = LENS.forward(record)
    assert LENS.backward(view, complement) == record
