"""Test UniMorphAdapter."""

from __future__ import annotations

import didactic.api as dx
import pytest

from bead.resources.adapters.cache import AdapterCache
from bead.resources.adapters.unimorph import UniMorphAdapter


def test_unimorph_adapter_initialization() -> None:
    """Test UniMorphAdapter initialization."""
    adapter = UniMorphAdapter()
    assert adapter.cache is None


def test_unimorph_adapter_with_cache(adapter_cache: AdapterCache) -> None:
    """Test UniMorphAdapter with cache."""
    adapter = UniMorphAdapter(cache=adapter_cache)
    assert adapter.cache is adapter_cache


def test_unimorph_adapter_is_available(unimorph_adapter: UniMorphAdapter) -> None:
    """Test that UniMorph adapter is available."""
    assert unimorph_adapter.is_available()


def test_unimorph_adapter_fetch_items_english(
    unimorph_adapter: UniMorphAdapter,
) -> None:
    """Test fetching English items from UniMorph."""
    items = unimorph_adapter.fetch_items(query="walk", language_code="en")
    assert len(items) > 0
    assert all(item.language_code == "eng" for item in items)
    # Check that items have lemma and form
    assert all(item.lemma == "walk" for item in items)
    assert all(item.form is not None for item in items)
    # Check that items have morphological features
    assert any(item.features for item in items)


def test_unimorph_adapter_fetch_items_multilingual(
    unimorph_adapter: UniMorphAdapter,
) -> None:
    """Test fetching items from multiple languages."""
    # Test with Korean (2-letter code)
    items_ko = unimorph_adapter.fetch_items(query=None, language_code="ko")
    # Should normalize to 'kor' internally
    if len(items_ko) > 0:
        assert all(item.language_code == "kor" for item in items_ko)


def test_unimorph_adapter_requires_language_code() -> None:
    """Test that UniMorph adapter requires language_code."""
    adapter = UniMorphAdapter()
    with pytest.raises(
        (ValueError, dx.ValidationError), match="requires language_code"
    ):
        adapter.fetch_items(query="walk", language_code=None)


def test_unimorph_adapter_caching(
    unimorph_adapter: UniMorphAdapter, adapter_cache: AdapterCache
) -> None:
    """Test that UniMorph adapter uses cache."""
    # First fetch (miss)
    items1 = unimorph_adapter.fetch_items(query="walk", language_code="en")

    # Second fetch (hit)
    items2 = unimorph_adapter.fetch_items(query="walk", language_code="en")

    # Should be same (from cache)
    assert items1 == items2


def test_unimorph_adapter_morphological_features(
    unimorph_adapter: UniMorphAdapter,
) -> None:
    """Test that items have morphological features."""
    items = unimorph_adapter.fetch_items(query="walk", language_code="en")
    # At least some items should have features
    items_with_features = [item for item in items if item.features]
    assert len(items_with_features) > 0

    # Check that features are parsed correctly
    for item in items_with_features:
        assert "unimorph_features" in item.features
        # At least one parsed feature should exist
        feature_keys = [k for k in item.features.keys() if k != "unimorph_features"]
        assert len(feature_keys) > 0


def test_unimorph_adapter_language_code_normalization(
    unimorph_adapter: UniMorphAdapter,
) -> None:
    """Test that 2-letter codes are normalized to 3-letter."""
    # Should handle 2-letter code
    items = unimorph_adapter.fetch_items(query=None, language_code="en")
    # All items should be normalized to ISO 639-3 format
    assert all(item.language_code == "eng" for item in items)


def test_unimorph_adapter_feature_parsing() -> None:
    """Test feature parsing logic."""
    adapter = UniMorphAdapter()

    # Test parsing of common features
    features = adapter._parse_features("V;PRS;3;SG")
    assert features["pos"] == "V"
    assert features["tense"] == "PRS"
    assert features["person"] == "3"
    assert features["number"] == "SG"
    assert features["unimorph_features"] == "V;PRS;3;SG"

    # Test parsing past tense
    features_pst = adapter._parse_features("V;PST")
    assert features_pst["pos"] == "V"
    assert features_pst["tense"] == "PST"

    # Test parsing noun plural
    features_n = adapter._parse_features("N;PL")
    assert features_n["pos"] == "N"
    assert features_n["number"] == "PL"
