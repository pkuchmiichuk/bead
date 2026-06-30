"""Tests for :mod:`bead.protocol.context`."""

from __future__ import annotations

import pytest

from bead.protocol import context as context_module
from bead.protocol.context import (
    ContextItem,
    ProtocolContext,
    get_context_predicate,
    list_context_predicates,
    register_context_predicate,
)


class TestContextItem:
    """Tests for :class:`ContextItem`."""

    def test_attribute_lookup(self) -> None:
        item = ContextItem(
            head_lemma="ball",
            attributes={"animacy": 0.0, "definiteness": 0.7},
        )
        assert item.attribute("animacy") == pytest.approx(0.0)
        assert item.attribute("definiteness") == pytest.approx(0.7)
        assert item.attribute("absent") is None

    def test_defaults(self) -> None:
        item = ContextItem()
        assert item.node_id == ""
        assert item.span_positions == ()
        assert item.attributes == {}


class TestProtocolContext:
    """Tests for :class:`ProtocolContext`."""

    def test_with_response_threads(self) -> None:
        ctx = ProtocolContext(sentence="Mary ran fast.")
        ctx2 = ctx.with_response("dynamicity", "yes")
        ctx3 = ctx2.with_response("completion", "no")

        assert ctx.previous_responses == {}
        assert ctx2.previous_responses == {"dynamicity": "yes"}
        assert ctx3.previous_responses == {
            "dynamicity": "yes",
            "completion": "no",
        }

    def test_get_response(self) -> None:
        ctx = ProtocolContext().with_response("q", "yes")
        assert ctx.get_response("q") == "yes"
        assert ctx.get_response("absent") is None

    def test_with_dependents(self) -> None:
        dep = ContextItem(head_lemma="ball", head_upos="NOUN")
        ctx = ProtocolContext(
            sentence="Mary kicked the ball.",
            dependents=(dep,),
        )
        assert len(ctx.dependents) == 1
        assert ctx.dependents[0].head_lemma == "ball"


class TestPredicateRegistry:
    """Tests for the context-predicate registry."""

    def test_always_is_pre_registered(self) -> None:
        always = get_context_predicate("always")
        assert always(ProtocolContext()) is True

    def test_register_and_lookup(self) -> None:
        def has_dependents(ctx: ProtocolContext) -> bool:
            return len(ctx.dependents) > 0

        register_context_predicate("has_dependents_test", has_dependents)
        try:
            fetched = get_context_predicate("has_dependents_test")
            assert fetched is has_dependents
            assert "has_dependents_test" in list_context_predicates()
            assert fetched(ProtocolContext()) is False
            ctx = ProtocolContext(dependents=(ContextItem(),))
            assert fetched(ctx) is True
        finally:
            # cleanup so other tests don't see this predicate
            context_module._PREDICATES.pop("has_dependents_test", None)

    def test_unknown_predicate_raises(self) -> None:
        with pytest.raises(KeyError, match="No context predicate"):
            get_context_predicate("nonexistent_predicate_xyz")

    def test_re_registration_overwrites(self) -> None:
        def first(_ctx: ProtocolContext) -> bool:
            return True

        def second(_ctx: ProtocolContext) -> bool:
            return False

        register_context_predicate("override_test", first)
        register_context_predicate("override_test", second)
        try:
            assert get_context_predicate("override_test") is second
        finally:
            context_module._PREDICATES.pop("override_test", None)
