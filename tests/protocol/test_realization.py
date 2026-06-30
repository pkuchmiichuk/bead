"""Tests for :mod:`bead.protocol.realization`."""

from __future__ import annotations

import pytest

from bead.items.cache import ModelOutputCache
from bead.protocol.anchor import ResponseSpace, SemanticAnchor
from bead.protocol.context import ProtocolContext
from bead.protocol.realization import (
    ContextualTemplateRealization,
    LMRealization,
    RealizationStrategy,
    TemplateRealization,
    TemplateVariant,
    always,
)


def _build_anchor(canonical: str = "Does [[situation]] end?") -> SemanticAnchor:
    return SemanticAnchor(
        name="completion",
        target_property="telicity",
        canonical_prompt=canonical,
        response_space=ResponseSpace(options=("no", "yes"), is_ordered=False),
        required_span_labels=frozenset({"situation"}),
    )


class TestTemplateRealization:
    """Tests for :class:`TemplateRealization`."""

    def test_default_uses_canonical(self) -> None:
        anchor = _build_anchor()
        ctx = ProtocolContext()
        tr = TemplateRealization()
        assert tr.realize(anchor, ctx) == anchor.canonical_prompt

    def test_explicit_template_overrides(self) -> None:
        anchor = _build_anchor()
        ctx = ProtocolContext()
        tr = TemplateRealization(template="Did [[situation]] finish?")
        assert tr.realize(anchor, ctx) == "Did [[situation]] finish?"

    def test_conforms_to_protocol(self) -> None:
        assert isinstance(TemplateRealization(), RealizationStrategy)


class TestContextualTemplateRealization:
    """Tests for :class:`ContextualTemplateRealization`."""

    def test_priority_ordering(self) -> None:
        anchor = _build_anchor()
        ctx = ProtocolContext(target_upos="VERB")

        def is_verb(c: ProtocolContext) -> bool:
            return c.target_upos == "VERB"

        verb_variant = TemplateVariant(
            template="VERB-specific [[situation]]?",
            condition=is_verb,
            priority=10,
        )
        fallback_variant = TemplateVariant(
            template="Generic [[situation]]?",
            condition=always,
            priority=0,
        )
        ctr = ContextualTemplateRealization(variants=(fallback_variant, verb_variant))
        assert ctr.realize(anchor, ctx) == "VERB-specific [[situation]]?"

    def test_fallback_when_no_match(self) -> None:
        anchor = _build_anchor()
        ctx = ProtocolContext(target_upos="NOUN")

        def is_verb(c: ProtocolContext) -> bool:
            return c.target_upos == "VERB"

        ctr = ContextualTemplateRealization(
            variants=(TemplateVariant(template="V", condition=is_verb, priority=10),),
            fallback="Custom fallback?",
        )
        assert ctr.realize(anchor, ctx) == "Custom fallback?"

    def test_fallback_to_canonical_when_no_explicit_fallback(self) -> None:
        anchor = _build_anchor("Canonical [[situation]]?")
        ctx = ProtocolContext(target_upos="NOUN")
        ctr = ContextualTemplateRealization(
            variants=(
                TemplateVariant(
                    template="V",
                    condition=lambda c: c.target_upos == "VERB",
                    priority=10,
                ),
            ),
        )
        assert ctr.realize(anchor, ctx) == "Canonical [[situation]]?"


class _StubLMClient:
    """Stub LM client recording every call."""

    def __init__(self, response: str) -> None:
        self.response = response
        self.calls: list[tuple[str, float, int]] = []

    def complete(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        self.calls.append((prompt, temperature, max_tokens))
        return self.response


def _memory_cache() -> ModelOutputCache:
    """Return an in-memory ModelOutputCache for use in tests."""
    return ModelOutputCache(backend="memory")


class TestLMRealization:
    """Tests for :class:`LMRealization`."""

    def test_realize_appends_question_mark(self) -> None:
        client = _StubLMClient("What does the situation do")
        lm = LMRealization(client, model_name="stub")
        prompt = lm.realize(_build_anchor(), ProtocolContext())
        assert prompt.endswith("?")

    def test_realize_strips_quotes_and_whitespace(self) -> None:
        client = _StubLMClient('  "Did the event end?"  ')
        lm = LMRealization(client, model_name="stub")
        prompt = lm.realize(_build_anchor(), ProtocolContext())
        assert prompt == "Did the event end?"

    def test_caching(self) -> None:
        client = _StubLMClient("Did it end?")
        lm = LMRealization(client, model_name="stub", cache=_memory_cache())
        anchor = _build_anchor()
        ctx = ProtocolContext(sentence="Mary ran.")
        out1 = lm.realize(anchor, ctx)
        out2 = lm.realize(anchor, ctx)
        assert out1 == out2
        assert len(client.calls) == 1

    def test_caching_disabled(self) -> None:
        client = _StubLMClient("Did it end?")
        lm = LMRealization(client, model_name="stub")
        anchor = _build_anchor()
        ctx = ProtocolContext()
        lm.realize(anchor, ctx)
        lm.realize(anchor, ctx)
        assert len(client.calls) == 2

    def test_lm_failure_wraps_runtime_error(self) -> None:
        class FailingClient:
            def complete(
                self,
                prompt: str,
                *,
                temperature: float,
                max_tokens: int,
            ) -> str:
                del prompt, temperature, max_tokens
                raise ConnectionError("network down")

        lm = LMRealization(FailingClient(), model_name="stub")
        with pytest.raises(RuntimeError, match="LM realization failed"):
            lm.realize(_build_anchor(), ProtocolContext())

    def test_empty_response_raises(self) -> None:
        client = _StubLMClient("   ")
        lm = LMRealization(client, model_name="stub")
        with pytest.raises(RuntimeError, match="empty response"):
            lm.realize(_build_anchor(), ProtocolContext())

    def test_quoted_empty_response_raises(self) -> None:
        client = _StubLMClient('  ""  ')
        lm = LMRealization(client, model_name="stub")
        with pytest.raises(RuntimeError, match="empty response"):
            lm.realize(_build_anchor(), ProtocolContext())

    def test_calls_pass_kwargs(self) -> None:
        client = _StubLMClient("Did it end?")
        lm = LMRealization(client, model_name="stub", temperature=0.5, max_tokens=128)
        lm.realize(_build_anchor(), ProtocolContext())
        assert len(client.calls) == 1
        _, temperature, max_tokens = client.calls[0]
        assert temperature == pytest.approx(0.5)
        assert max_tokens == 128

    def test_cache_isolated_by_model_name(self) -> None:
        """Two realizations sharing a cache but different model_names
        do not collide."""
        cache = _memory_cache()
        client_a = _StubLMClient("Answer A?")
        client_b = _StubLMClient("Answer B?")
        lm_a = LMRealization(client_a, model_name="model-a", cache=cache)
        lm_b = LMRealization(client_b, model_name="model-b", cache=cache)
        anchor = _build_anchor()
        ctx = ProtocolContext()
        assert lm_a.realize(anchor, ctx) == "Answer A?"
        assert lm_b.realize(anchor, ctx) == "Answer B?"
