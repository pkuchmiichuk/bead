"""Tests for :mod:`bead.protocol.encoding`."""

from __future__ import annotations

import pytest

from bead.protocol.anchor import ResponseSpace, SemanticPoles
from bead.protocol.encoding import (
    ResponseEncoding,
    ScaleType,
    encode_response_space,
)


class TestScaleType:
    """Tests for :class:`ScaleType`."""

    def test_str_values(self) -> None:
        assert ScaleType.BINARY.value == "binary"
        assert ScaleType.ORDINAL.value == "ordinal"
        assert ScaleType.NOMINAL.value == "nominal"


class TestResponseEncoding:
    """Tests for :class:`ResponseEncoding`."""

    def _build(self) -> ResponseEncoding:
        return ResponseEncoding(
            name="completion",
            n_levels=5,
            scale_type=ScaleType.ORDINAL,
            labels=(
                "definitely no",
                "probably no",
                "unsure",
                "probably yes",
                "definitely yes",
            ),
            semantic_poles=SemanticPoles(low="definitely no", high="definitely yes"),
        )

    def test_label_index_round_trip(self) -> None:
        enc = self._build()
        for i, label in enumerate(enc.labels):
            assert enc.label_to_index(label) == i
            assert enc.index_to_label(i) == label

    def test_label_to_index_unknown_raises(self) -> None:
        enc = self._build()
        with pytest.raises(ValueError, match="not found"):
            enc.label_to_index("absent")

    def test_index_out_of_range(self) -> None:
        enc = self._build()
        with pytest.raises(IndexError):
            enc.index_to_label(-1)
        with pytest.raises(IndexError):
            enc.index_to_label(5)

    def test_scale_predicates(self) -> None:
        enc = self._build()
        assert enc.is_ordinal is True
        assert enc.is_binary is False
        assert enc.is_nominal is False

    def test_n_levels_must_match_labels(self) -> None:
        with pytest.raises(Exception, match="n_levels"):
            ResponseEncoding(
                name="bad",
                n_levels=3,
                scale_type=ScaleType.NOMINAL,
                labels=("a", "b"),
            )

    def test_duplicate_labels_rejected(self) -> None:
        with pytest.raises(Exception, match="Duplicate"):
            ResponseEncoding(
                name="dup",
                n_levels=3,
                scale_type=ScaleType.NOMINAL,
                labels=("a", "b", "a"),
            )

    def test_binary_must_have_two_levels(self) -> None:
        with pytest.raises(Exception, match="BINARY"):
            ResponseEncoding(
                name="b",
                n_levels=3,
                scale_type=ScaleType.BINARY,
                labels=("a", "b", "c"),
            )


class TestEncodeResponseSpace:
    """Tests for :func:`encode_response_space`."""

    def test_binary_classification(self) -> None:
        rs = ResponseSpace(options=("no", "yes"), is_ordered=False)
        enc = encode_response_space("dynamicity", rs)
        assert enc.scale_type == ScaleType.BINARY
        assert enc.n_levels == 2
        assert enc.is_binary is True

    def test_ordinal_classification(self) -> None:
        rs = ResponseSpace(
            options=("low", "med", "high"),
            is_ordered=True,
            semantic_poles=SemanticPoles(low="low", high="high"),
        )
        enc = encode_response_space("intensity", rs)
        assert enc.scale_type == ScaleType.ORDINAL
        assert enc.n_levels == 3
        assert enc.semantic_poles is not None
        assert enc.semantic_poles.as_tuple() == ("low", "high")

    def test_nominal_classification(self) -> None:
        rs = ResponseSpace(options=("a", "b", "c"), is_ordered=False)
        enc = encode_response_space("category", rs)
        assert enc.scale_type == ScaleType.NOMINAL
        assert enc.is_nominal is True

    def test_two_options_ordered_is_ordinal_not_binary(self) -> None:
        # A two-option *ordered* space is ordinal, only unordered
        # two-option spaces are classified as binary.
        rs = ResponseSpace(options=("low", "high"), is_ordered=True)
        enc = encode_response_space("polarity", rs)
        assert enc.scale_type == ScaleType.ORDINAL
