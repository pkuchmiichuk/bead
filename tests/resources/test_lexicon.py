"""Tests for Lexicon class."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import didactic.api as dx
import pandas as pd
import polars as pl
import pytest

from bead.data.base import BeadBaseModel
from bead.resources import LexicalItem, Lexicon

# ============================================================================
# Creation & Basic Operations (8 tests)
# ============================================================================


def test_creation_with_name() -> None:
    """Test lexicon creation with just a name."""
    lexicon = Lexicon(name="test")
    assert lexicon.name == "test"
    assert lexicon.description is None
    assert lexicon.language_code is None
    assert len(lexicon.items) == 0
    assert len(lexicon.tags) == 0


def test_creation_with_all_fields() -> None:
    """Test lexicon creation with description, language code, and tags."""
    lexicon = Lexicon(
        name="verbs",
        description="A collection of verbs",
        language_code="en",
        tags=["verbs", "test"],
    )
    assert lexicon.name == "verbs"
    assert lexicon.description == "A collection of verbs"
    assert lexicon.language_code == "eng"  # Normalized to ISO 639-3
    assert lexicon.tags == ("verbs", "test")


def test_len_returns_correct_count(sample_lexicon: Lexicon) -> None:
    """Test that __len__ returns correct count."""
    assert len(sample_lexicon) == 3


def test_iter_iterates_over_items(sample_lexicon: Lexicon) -> None:
    """Test that __iter__ iterates over items."""
    items = list(sample_lexicon)
    assert len(items) == 3
    assert all(isinstance(item, LexicalItem) for item in items)


def test_contains_checks_item_presence(sample_lexicon: Lexicon) -> None:
    """Test that __contains__ checks item presence."""
    item = list(sample_lexicon.items)[0]
    assert item.id in sample_lexicon
    assert uuid4() not in sample_lexicon


def test_empty_lexicon() -> None:
    """Test empty lexicon."""
    lexicon = Lexicon(name="empty")
    assert len(lexicon) == 0
    assert list(lexicon) == []


def test_lexicon_with_single_item() -> None:
    """Test lexicon with single item."""
    lexicon = Lexicon(name="single")
    item = LexicalItem(lemma="test", language_code="eng")
    lexicon = lexicon.with_item(item)
    assert len(lexicon) == 1


def test_lexicon_inherits_from_bead_base_model() -> None:
    """Test that Lexicon inherits from BeadBaseModel."""
    lexicon = Lexicon(name="test")
    assert isinstance(lexicon, BeadBaseModel)
    assert hasattr(lexicon, "id")
    assert hasattr(lexicon, "created_at")
    assert hasattr(lexicon, "modified_at")


# ============================================================================
# CRUD Operations (8 tests)
# ============================================================================


def test_add_adds_item_successfully() -> None:
    """Test that add() adds an item successfully."""
    lexicon = Lexicon(name="test")
    item = LexicalItem(lemma="walk", language_code="eng")
    lexicon = lexicon.with_item(item)
    assert len(lexicon) == 1
    assert item.id in lexicon


def test_add_raises_error_on_duplicate_id() -> None:
    """Test that add() raises error on duplicate ID."""
    lexicon = Lexicon(name="test")
    item = LexicalItem(lemma="walk", language_code="eng")
    lexicon = lexicon.with_item(item)
    # Try to add the same item again
    with pytest.raises((ValueError, dx.ValidationError), match="already exists"):
        lexicon = lexicon.with_item(item)


def test_add_many_adds_multiple_items() -> None:
    """Test that add_many() adds multiple items."""
    lexicon = Lexicon(name="test")
    items = [
        LexicalItem(lemma="walk", language_code="eng"),
        LexicalItem(lemma="run", language_code="eng"),
        LexicalItem(lemma="jump", language_code="eng"),
    ]
    lexicon = lexicon.with_items(items)
    assert len(lexicon) == 3


def test_remove_removes_and_returns_item() -> None:
    """Test that remove() removes and returns item."""
    lexicon = Lexicon(name="test")
    item = LexicalItem(lemma="walk", language_code="eng")
    lexicon = lexicon.with_item(item)
    lexicon, removed = lexicon.without_item(item.id)
    assert removed.lemma == "walk"
    assert len(lexicon) == 0


def test_remove_raises_key_error_if_not_found() -> None:
    """Test that remove() raises KeyError if not found."""
    lexicon = Lexicon(name="test")
    with pytest.raises(KeyError, match="not found"):
        lexicon.without_item(uuid4())


def test_get_returns_item_if_exists() -> None:
    """Test that get() returns item if it exists."""
    lexicon = Lexicon(name="test")
    item = LexicalItem(lemma="walk", language_code="eng")
    lexicon = lexicon.with_item(item)
    retrieved = lexicon.by_id(item.id)
    assert retrieved is not None
    assert retrieved.lemma == "walk"


def test_get_returns_none_if_not_exists() -> None:
    """Test that get() returns None if item doesn't exist."""
    lexicon = Lexicon(name="test")
    assert lexicon.by_id(uuid4()) is None


def test_adding_same_item_twice_fails() -> None:
    """Test that adding same item twice fails."""
    lexicon = Lexicon(name="test")
    item = LexicalItem(lemma="walk", language_code="eng")
    lexicon = lexicon.with_item(item)
    with pytest.raises((ValueError, dx.ValidationError), match="already exists"):
        lexicon = lexicon.with_item(item)


# ============================================================================
# Filtering Operations (8 tests)
# ============================================================================


def test_filter_with_custom_predicate(sample_lexicon: Lexicon) -> None:
    """Test filter() with custom predicate."""
    # Filter for items with frequency > 600
    high_freq = sample_lexicon.filter(
        lambda item: "frequency" in item.features and item.features["frequency"] > 600
    )
    assert len(high_freq.items) == 2  # walk (1000) and run (800)


def test_filter_by_pos_filters_correctly(sample_lexicon: Lexicon) -> None:
    """Test that filter_by_pos() filters correctly."""
    verbs = sample_lexicon.filter_by_pos("VERB")
    assert len(verbs.items) == 2

    nouns = sample_lexicon.filter_by_pos("NOUN")
    assert len(nouns.items) == 1


def test_filter_by_lemma_exact_match() -> None:
    """Test that filter_by_lemma() does exact match."""
    lexicon = Lexicon(name="test")
    lexicon = lexicon.with_item(LexicalItem(lemma="walk", language_code="eng"))
    lexicon = lexicon.with_item(LexicalItem(lemma="walked", language_code="eng"))
    results = lexicon.filter_by_lemma("walk")
    assert len(results.items) == 1
    item = list(results.items)[0]
    assert item.lemma == "walk"


def test_filter_by_feature_with_feature_value() -> None:
    """Test that filter_by_feature() filters by feature value."""
    lexicon = Lexicon(name="test")
    lexicon = lexicon.with_item(
        LexicalItem(lemma="walk", language_code="eng", features={"tense": "present"})
    )
    lexicon = lexicon.with_item(
        LexicalItem(lemma="walked", language_code="eng", features={"tense": "past"})
    )

    present = lexicon.filter_by_feature("tense", "present")
    assert len(present.items) == 1


def test_filter_by_attribute_with_attribute_value(
    sample_lexicon: Lexicon,
) -> None:
    """Test that filter_by_attribute() filters by attribute value."""
    high_freq = sample_lexicon.filter_by_attribute("frequency", 1000)
    assert len(high_freq.items) == 1


def test_filter_returns_new_lexicon_instance(sample_lexicon: Lexicon) -> None:
    """Test that filter returns new Lexicon instance."""
    filtered = sample_lexicon.filter(lambda item: True)
    assert isinstance(filtered, Lexicon)
    assert filtered is not sample_lexicon


def test_filter_preserves_lexicon_metadata(sample_lexicon: Lexicon) -> None:
    """Test that filter preserves lexicon name/metadata."""
    filtered = sample_lexicon.filter(lambda item: True)
    assert sample_lexicon.name in filtered.name
    assert filtered.description == sample_lexicon.description
    assert filtered.language_code == sample_lexicon.language_code


def test_filter_with_no_matches_returns_empty_lexicon() -> None:
    """Test that filter with no matches returns empty lexicon."""
    lexicon = Lexicon(name="test")
    lexicon = lexicon.with_item(
        LexicalItem(lemma="walk", language_code="eng", features={"pos": "VERB"})
    )

    results = lexicon.filter_by_pos("NOUN")
    assert len(results.items) == 0


# ============================================================================
# Search Operations (3 tests)
# ============================================================================


def test_search_case_insensitive_substring_match() -> None:
    """Test that search() does case-insensitive substring match."""
    lexicon = Lexicon(name="test")
    lexicon = lexicon.with_item(LexicalItem(lemma="Walking", language_code="eng"))
    lexicon = lexicon.with_item(LexicalItem(lemma="run", language_code="eng"))
    results = lexicon.search("walk")
    assert len(results.items) == 1


def test_search_in_different_fields() -> None:
    """Test that search() works in different fields."""
    lexicon = Lexicon(name="test")
    lexicon = lexicon.with_item(
        LexicalItem(
            lemma="test", form="testing", language_code="eng", features={"pos": "VERB"}
        )
    )

    # Search in lemma
    results_lemma = lexicon.search("test", field="lemma")
    assert len(results_lemma.items) == 1

    # Search in pos
    results_pos = lexicon.search("verb", field="pos")
    assert len(results_pos.items) == 1

    # Search in form
    results_form = lexicon.search("ing", field="form")
    assert len(results_form.items) == 1


def test_search_with_no_matches() -> None:
    """Test that search() with no matches returns empty lexicon."""
    lexicon = Lexicon(name="test")
    lexicon = lexicon.with_item(LexicalItem(lemma="walk", language_code="eng"))
    results = lexicon.search("xyz")
    assert len(results.items) == 0


def test_search_invalid_field_raises_error() -> None:
    """Test that search() with invalid field raises error."""
    lexicon = Lexicon(name="test")
    with pytest.raises((ValueError, dx.ValidationError), match="Invalid field"):
        lexicon.search("test", field="invalid")


# ============================================================================
# Merging Operations (4 tests)
# ============================================================================


def test_merge_with_keep_first_strategy() -> None:
    """Test merge() with 'keep_first' strategy."""
    lex1 = Lexicon(name="lex1")
    item1 = LexicalItem(lemma="walk", language_code="eng")
    lex1 = lex1.with_item(item1)
    lex2 = Lexicon(name="lex2")
    # Add different item with same ID as item1 (simulate duplicate)
    item2 = LexicalItem(lemma="run", language_code="eng")
    lex2 = lex2.with_item(item2)
    # Also add item1 to lex2 to test conflict resolution
    conflict = LexicalItem(id=item1.id, lemma="modified", language_code="eng")
    lex2 = lex2.with_(items=(*lex2.items, conflict))

    merged = lex1.merge(lex2, strategy="keep_first")
    # Should keep item1 from lex1
    assert next(i for i in merged.items if i.id == item1.id).lemma == "walk"


def test_merge_with_keep_second_strategy() -> None:
    """Test merge() with 'keep_second' strategy."""
    lex1 = Lexicon(name="lex1")
    item1 = LexicalItem(lemma="walk", language_code="eng")
    lex1 = lex1.with_item(item1)
    lex2 = Lexicon(name="lex2")
    item2 = LexicalItem(lemma="run", language_code="eng")
    lex2 = lex2.with_item(item2)
    # Add conflicting item
    conflict = LexicalItem(id=item1.id, lemma="modified", language_code="eng")
    lex2 = lex2.with_(items=(*lex2.items, conflict))

    merged = lex1.merge(lex2, strategy="keep_second")
    # Should keep modified version from lex2
    assert next(i for i in merged.items if i.id == item1.id).lemma == "modified"


def test_merge_with_error_strategy_raises_on_duplicates() -> None:
    """Test merge() with 'error' strategy raises on duplicates."""
    lex1 = Lexicon(name="lex1")
    item1 = LexicalItem(lemma="walk", language_code="eng")
    lex1 = lex1.with_item(item1)
    lex2 = Lexicon(name="lex2")
    # Add same item to lex2
    lex2 = lex2.with_(items=(item1,))

    with pytest.raises(
        (ValueError, dx.ValidationError), match="Duplicate item IDs found"
    ):
        lex1.merge(lex2, strategy="error")


def test_merge_with_no_overlapping_ids() -> None:
    """Test merge() with no overlapping IDs."""
    lex1 = Lexicon(name="lex1")
    lex1 = lex1.with_item(LexicalItem(lemma="walk", language_code="eng"))
    lex2 = Lexicon(name="lex2")
    lex2 = lex2.with_item(LexicalItem(lemma="run", language_code="eng"))
    merged = lex1.merge(lex2)
    assert len(merged.items) == 2


def test_merge_preserves_language_code() -> None:
    """Test that merge preserves language code."""
    lex1 = Lexicon(name="lex1", language_code="en")
    lex2 = Lexicon(name="lex2", language_code="es")

    merged = lex1.merge(lex2)
    assert merged.language_code == "eng"  # From lex1 (normalized to ISO 639-3)

    # Test when lex1 has no language code
    lex3 = Lexicon(name="lex3")
    merged2 = lex3.merge(lex2)
    assert merged2.language_code == "spa"  # From lex2 (normalized to ISO 639-3)


# ============================================================================
# DataFrame Conversion (6 tests)
# ============================================================================


def test_to_dataframe_pandas_creates_correct_structure() -> None:
    """Test that to_dataframe() creates correct structure for pandas."""
    lexicon = Lexicon(name="test")
    lexicon = lexicon.with_item(
        LexicalItem(lemma="walk", language_code="eng", features={"pos": "VERB"})
    )

    df = lexicon.to_dataframe(backend="pandas")
    assert isinstance(df, pd.DataFrame)
    assert "lemma" in df.columns
    assert len(df) == 1


def test_to_dataframe_polars_creates_correct_structure() -> None:
    """Test that to_dataframe() creates correct structure for polars."""
    lexicon = Lexicon(name="test")
    lexicon = lexicon.with_item(
        LexicalItem(lemma="walk", language_code="eng", features={"pos": "VERB"})
    )

    df = lexicon.to_dataframe(backend="polars")
    assert isinstance(df, pl.DataFrame)
    assert "lemma" in df.columns
    assert len(df) == 1


def test_to_dataframe_includes_all_fields() -> None:
    """Test that to_dataframe() includes all fields."""
    lexicon = Lexicon(name="test")
    lexicon = lexicon.with_item(
        LexicalItem(
            lemma="walk",
            form="walking",
            source="manual",
            language_code="eng",
            features={"pos": "VERB", "tense": "present", "frequency": 1000},
        )
    )

    df = lexicon.to_dataframe()
    assert "id" in df.columns
    assert "lemma" in df.columns
    assert "language_code" in df.columns
    assert "form" in df.columns
    assert "source" in df.columns
    assert "feature_pos" in df.columns
    assert "feature_tense" in df.columns
    assert "feature_frequency" in df.columns


def test_to_dataframe_flattens_features_and_attributes() -> None:
    """Test that to_dataframe() flattens features/attributes."""
    lexicon = Lexicon(name="test")
    lexicon = lexicon.with_item(
        LexicalItem(
            lemma="walk",
            language_code="eng",
            features={
                "tense": "present",
                "aspect": "progressive",
                "frequency": 1000,
                "register": "formal",
            },
        )
    )

    df = lexicon.to_dataframe()
    assert "feature_tense" in df.columns
    assert "feature_aspect" in df.columns
    assert "feature_frequency" in df.columns
    assert "feature_register" in df.columns


def test_from_dataframe_pandas_roundtrip() -> None:
    """Test from_dataframe() roundtrip with pandas."""
    df = pd.DataFrame(
        {
            "lemma": ["walk", "run"],
            "feature_pos": ["VERB", "VERB"],
            "feature_tense": ["present", "past"],
            "feature_frequency": [1000, 800],
        }
    )

    lexicon = Lexicon.from_dataframe(df, "test")
    assert len(lexicon.items) == 2

    # Check that features were reconstructed
    items = list(lexicon.items)
    assert "tense" in items[0].features
    assert "frequency" in items[0].features
    assert "pos" in items[0].features


def test_from_dataframe_polars_roundtrip() -> None:
    """Test from_dataframe() roundtrip with polars."""
    df = pl.DataFrame(
        {
            "lemma": ["walk", "run"],
            "feature_pos": ["VERB", "VERB"],
            "feature_tense": ["present", "past"],
            "feature_frequency": [1000, 800],
        }
    )

    lexicon = Lexicon.from_dataframe(df, "test")
    assert len(lexicon.items) == 2

    # Check that features were reconstructed
    items = list(lexicon.items)
    assert "tense" in items[0].features
    assert "frequency" in items[0].features
    assert "pos" in items[0].features


def test_from_dataframe_with_minimal_dataframe() -> None:
    """Test from_dataframe() with minimal DataFrame (only lemma)."""
    df = pd.DataFrame({"lemma": ["walk", "run", "jump"]})

    lexicon = Lexicon.from_dataframe(df, "test")
    assert len(lexicon.items) == 3


def test_from_dataframe_raises_on_missing_lemma() -> None:
    """Test that from_dataframe() raises error if no lemma column."""
    df = pd.DataFrame({"pos": ["VERB", "NOUN"]})

    with pytest.raises(
        (ValueError, dx.ValidationError), match="must have a 'lemma' column"
    ):
        Lexicon.from_dataframe(df, "test")


def test_to_dataframe_empty_lexicon() -> None:
    """Test to_dataframe() with empty lexicon."""
    lexicon = Lexicon(name="empty")
    df = lexicon.to_dataframe()
    assert len(df) == 0
    assert "lemma" in df.columns


# ============================================================================
# Serialization (4 tests)
# ============================================================================


def test_to_jsonl_writes_file_correctly(tmp_path: Path) -> None:
    """Test that to_jsonl() writes file correctly."""
    lexicon = Lexicon(name="test")
    lexicon = lexicon.with_item(LexicalItem(lemma="walk", language_code="eng"))
    lexicon = lexicon.with_item(LexicalItem(lemma="run", language_code="eng"))
    file_path = tmp_path / "test.jsonl"
    lexicon.to_jsonl(str(file_path))

    assert file_path.exists()
    lines = file_path.read_text().strip().split("\n")
    assert len(lines) == 2


def test_from_jsonl_reads_file_correctly(tmp_path: Path) -> None:
    """Test that from_jsonl() reads file correctly."""
    # First write a file
    lexicon = Lexicon(name="test")
    lexicon = lexicon.with_item(
        LexicalItem(lemma="walk", language_code="eng", features={"pos": "VERB"})
    )
    lexicon = lexicon.with_item(
        LexicalItem(lemma="run", language_code="eng", features={"pos": "VERB"})
    )
    file_path = tmp_path / "test.jsonl"
    lexicon.to_jsonl(str(file_path))

    # Then read it back
    loaded = Lexicon.from_jsonl(str(file_path), "loaded")
    assert len(loaded.items) == 2


def test_jsonl_roundtrip(tmp_path: Path) -> None:
    """Test roundtrip (save and load)."""
    original = Lexicon(name="test")
    original = original.with_item(
        LexicalItem(
            lemma="walk",
            language_code="eng",
            features={"pos": "VERB", "tense": "present", "frequency": 1000},
        )
    )

    file_path = tmp_path / "test.jsonl"
    original.to_jsonl(str(file_path))

    loaded = Lexicon.from_jsonl(str(file_path), "loaded")
    assert len(loaded.items) == len(original.items)

    # Check that data was preserved
    orig_item = list(original.items)[0]
    loaded_item = list(loaded.items)[0]
    assert loaded_item.lemma == orig_item.lemma
    assert loaded_item.features.get("pos") == orig_item.features.get("pos")


def test_serialization_preserves_all_data(tmp_path: Path) -> None:
    """Test that serialization preserves all data."""
    lexicon = Lexicon(name="test")
    lexicon = lexicon.with_item(
        LexicalItem(
            lemma="walk",
            language_code="eng",
            form="walking",
            features={
                "pos": "VERB",
                "tense": "present",
                "aspect": "progressive",
                "frequency": 1000,
                "register": "informal",
            },
            source="manual",
        )
    )

    file_path = tmp_path / "test.jsonl"
    lexicon.to_jsonl(str(file_path))

    loaded = Lexicon.from_jsonl(str(file_path), "loaded")
    item = list(loaded.items)[0]

    assert item.lemma == "walk"
    assert item.features["pos"] == "VERB"
    assert item.form == "walking"
    assert item.features["tense"] == "present"
    assert item.features["aspect"] == "progressive"
    assert item.features["frequency"] == 1000
    assert item.features["register"] == "informal"
    assert item.source == "manual"


def test_filter_by_language_code() -> None:
    """Test filtering lexicon by language code."""
    lexicon = Lexicon(name="multi", language_code=None)

    # Add items from different languages
    en_item = LexicalItem(lemma="break", language_code="en")
    ko_item = LexicalItem(lemma="깨다", language_code="ko")
    es_item = LexicalItem(lemma="romper", language_code="es")

    lexicon = lexicon.with_item(en_item)
    lexicon = lexicon.with_item(ko_item)
    lexicon = lexicon.with_item(es_item)
    # Filter by English (normalized to ISO 639-3)
    en_items = lexicon.filter(lambda item: item.language_code == "eng")
    assert len(en_items.items) == 1
    assert list(en_items.items)[0].lemma == "break"

    # Filter by Korean (normalized to ISO 639-3)
    ko_items = lexicon.filter(lambda item: item.language_code == "kor")
    assert len(ko_items.items) == 1
    assert list(ko_items.items)[0].lemma == "깨다"

    # Filter by Spanish (normalized to ISO 639-3)
    es_items = lexicon.filter(lambda item: item.language_code == "spa")
    assert len(es_items.items) == 1
    assert list(es_items.items)[0].lemma == "romper"
