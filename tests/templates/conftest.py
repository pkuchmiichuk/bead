"""Pytest fixtures for template resolver tests."""

from __future__ import annotations

import pytest

from bead.resources.adapters.glazing import GlazingAdapter
from bead.resources.adapters.registry import AdapterRegistry
from bead.resources.adapters.unimorph import UniMorphAdapter
from bead.resources.lexical_item import LexicalItem
from bead.resources.lexicon import Lexicon
from bead.templates.resolver import ConstraintResolver


@pytest.fixture
def sample_lexicon() -> Lexicon:
    """Create sample lexicon with diverse items for testing.

    Returns
    -------
    Lexicon
        Lexicon with multiple items covering different languages,
        parts of speech, and features.
    """
    items_list = [
        LexicalItem(
            lemma="break",
            language_code="en",
            features={
                "pos": "VERB",
                "transitivity": "transitive",
                "causative": True,
                "frequency": "high",
            },
        ),
        LexicalItem(
            lemma="shatter",
            language_code="en",
            features={
                "pos": "VERB",
                "transitivity": "transitive",
                "causative": True,
                "frequency": "medium",
            },
        ),
        LexicalItem(
            lemma="arrive",
            language_code="en",
            features={
                "pos": "VERB",
                "transitivity": "intransitive",
                "causative": False,
                "frequency": "high",
            },
        ),
        LexicalItem(
            lemma="happiness",
            language_code="en",
            features={"pos": "NOUN", "number": "singular", "frequency": "high"},
        ),
        LexicalItem(
            lemma="quickly",
            language_code="en",
            features={"pos": "ADV", "frequency": "high"},
        ),
        # Add multilingual items
        LexicalItem(
            lemma="kkakta",
            language_code="ko",
            features={"pos": "VERB", "transitivity": "transitive", "causative": True},
        ),
        LexicalItem(
            lemma="partir",
            language_code="fr",
            features={"pos": "VERB", "transitivity": "intransitive"},
        ),
    ]

    # Create lexicon and add items
    lexicon = Lexicon(name="test_lexicon")
    for item in items_list:
        lexicon = lexicon.with_item(item)

    return lexicon


@pytest.fixture
def adapter_registry() -> AdapterRegistry:
    """Create adapter registry with glazing and unimorph adapters.

    Returns
    -------
    AdapterRegistry
        Registry with glazing and unimorph adapters registered.
    """
    registry = AdapterRegistry()
    registry.register("glazing", GlazingAdapter)
    registry.register("unimorph", UniMorphAdapter)
    return registry


@pytest.fixture
def resolver() -> ConstraintResolver:
    """Create constraint resolver.

    Returns
    -------
    ConstraintResolver
        Resolver with DSL evaluator.
    """
    return ConstraintResolver()


@pytest.fixture
def resolver_no_cache(
    sample_lexicon: Lexicon,
    adapter_registry: AdapterRegistry,
) -> ConstraintResolver:
    """Create constraint resolver without caching.

    Parameters
    ----------
    sample_lexicon : Lexicon
        Sample lexicon fixture.
    adapter_registry : AdapterRegistry
        Adapter registry fixture.

    Returns
    -------
    ConstraintResolver
        Resolver configured with caching disabled.
    """
    return ConstraintResolver(
        lexicon=sample_lexicon,
        adapter_registry=adapter_registry,
        cache_results=False,
    )
