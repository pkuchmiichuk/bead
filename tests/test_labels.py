"""Tests for :mod:`bead.labels`."""

from __future__ import annotations

import re

from bead.labels import (
    LABEL_PATTERN,
    LabelRef,
    find_label_names,
    parse_label_refs,
    replace_label_refs,
)


class TestParseLabelRefs:
    """Tests for :func:`parse_label_refs`."""

    def test_no_references(self) -> None:
        assert parse_label_refs("Plain text with no refs.") == ()

    def test_bare_label(self) -> None:
        refs = parse_label_refs("Did [[agent]] act?")
        assert len(refs) == 1
        assert refs[0].label == "agent"
        assert refs[0].display_text is None
        assert refs[0].transforms == ()

    def test_explicit_display_text(self) -> None:
        refs = parse_label_refs("Did [[event:the breaking]] happen?")
        assert refs[0].label == "event"
        assert refs[0].display_text == "the breaking"
        assert refs[0].transforms == ()

    def test_single_transform(self) -> None:
        refs = parse_label_refs("Did [[situation|gerund]] happen?")
        assert refs[0].label == "situation"
        assert refs[0].transforms == ("gerund",)

    def test_chained_transforms(self) -> None:
        refs = parse_label_refs("Did [[situation|gerund|lower]] happen?")
        assert refs[0].transforms == ("gerund", "lower")

    def test_explicit_text_and_transforms(self) -> None:
        refs = parse_label_refs("[[event:the running|upper]]")
        assert refs[0].label == "event"
        assert refs[0].display_text == "the running"
        assert refs[0].transforms == ("upper",)

    def test_offsets_are_correct(self) -> None:
        prompt = "x [[a]] y"
        refs = parse_label_refs(prompt)
        assert refs[0].start_offset == 2
        assert refs[0].end_offset == 7
        assert prompt[refs[0].start_offset : refs[0].end_offset] == "[[a]]"

    def test_multiple_refs_in_order(self) -> None:
        refs = parse_label_refs("[[a]] then [[b:bee]] then [[c|x]]")
        assert [r.label for r in refs] == ["a", "b", "c"]


class TestFindLabelNames:
    """Tests for :func:`find_label_names`."""

    def test_distinct_labels(self) -> None:
        names = find_label_names("[[a]] and [[b:bee]] and [[a|gerund]] and [[c]]")
        assert names == frozenset({"a", "b", "c"})

    def test_empty_prompt(self) -> None:
        assert find_label_names("") == frozenset()


class TestReplaceLabelRefs:
    """Tests for :func:`replace_label_refs`."""

    def test_no_refs_returns_input(self) -> None:
        assert replace_label_refs("plain", lambda r: "X") == "plain"

    def test_replaces_in_order(self) -> None:
        out = replace_label_refs("[[a]] [[b]]", lambda r: f"<{r.label}>")
        assert out == "<a> <b>"

    def test_replacement_uses_explicit_text(self) -> None:
        out = replace_label_refs(
            "Did [[event:the running]] happen?",
            lambda r: r.display_text or r.label,
        )
        assert out == "Did the running happen?"


class TestLabelRef:
    """Tests for the :class:`LabelRef` BeadBaseModel."""

    def test_round_trip_through_with(self) -> None:
        ref = LabelRef(label="x", start_offset=0, end_offset=5)
        ref2 = ref.with_(label="y")
        assert ref.label == "x"
        assert ref2.label == "y"
        assert ref.id == ref2.id


def test_pattern_is_compiled() -> None:
    """The exported regex is a compiled pattern."""
    assert isinstance(LABEL_PATTERN, re.Pattern)
