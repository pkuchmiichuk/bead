"""Tests for lexical item models."""

from __future__ import annotations

from uuid import UUID

import pytest
from didactic.api import ValidationError

from bead.resources import LexicalItem


class TestLexicalItemCreation:
    """Test lexical item creation."""

    def test_create_with_all_fields(self) -> None:
        """Test creating a lexical item with all fields."""
        item = LexicalItem(
            lemma="walk",
            form="walked",
            language_code="eng",
            features={
                "pos": "VERB",
                "tense": "past",
                "transitive": True,
                "frequency": 1000,
                "rating": 4.5,
            },
            source="manual",
        )
        assert item.lemma == "walk"
        assert item.form == "walked"
        assert item.language_code == "eng"
        assert item.features["pos"] == "VERB"
        assert item.features["tense"] == "past"
        assert item.features["frequency"] == 1000
        assert item.source == "manual"

    def test_create_with_minimal_fields(self) -> None:
        """Test creating a lexical item with minimal fields."""
        item = LexicalItem(lemma="run", language_code="eng")
        assert item.lemma == "run"
        assert item.form is None
        assert item.language_code == "eng"
        assert item.features == {}
        assert item.source is None

    def test_create_with_empty_features_dict(self) -> None:
        """Test creating a lexical item with empty features dict."""
        item = LexicalItem(lemma="jump", language_code="eng", features={})
        assert item.features == {}

    def test_create_with_form_different_from_lemma(self) -> None:
        """Test creating a lexical item with form different from lemma."""
        item = LexicalItem(lemma="go", form="went", language_code="eng")
        assert item.lemma == "go"
        assert item.form == "went"

    def test_create_with_special_characters_in_lemma(self) -> None:
        """Test creating a lexical item with special characters in lemma."""
        item = LexicalItem(lemma="rock-and-roll", language_code="eng")
        assert item.lemma == "rock-and-roll"

    def test_create_with_nested_features(self) -> None:
        """Test creating a lexical item with nested features."""
        item = LexicalItem(
            lemma="test",
            language_code="eng",
            features={"morphology": {"prefix": "re", "suffix": "ed"}},
        )
        assert item.features["morphology"]["prefix"] == "re"


class TestLexicalItemValidation:
    """Test lexical item validation."""

    def test_empty_lemma_fails(self) -> None:
        """Test that empty lemma validation fails."""
        with pytest.raises(ValidationError) as exc_info:
            LexicalItem(lemma="", language_code="eng")
        assert "lemma must be non-empty" in str(exc_info.value)

    def test_whitespace_only_lemma_fails(self) -> None:
        """Test that whitespace-only lemma validation fails."""
        with pytest.raises(ValidationError) as exc_info:
            LexicalItem(lemma="   ", language_code="eng")
        assert "lemma must be non-empty" in str(exc_info.value)


class TestLexicalItemMutability:
    """Test lexical item mutability."""

    def test_features_can_be_extended_via_with_(self) -> None:
        """Test that features can be replaced via ``with_(...)``."""
        item = LexicalItem(lemma="test", language_code="eng")
        item = item.with_(features={**item.features, "new_feature": "value"})
        assert item.features["new_feature"] == "value"


class TestLexicalItemInheritance:
    """Test lexical item inheritance from BeadBaseModel."""

    def test_inherits_uuidv7_id(self) -> None:
        """Test that lexical item inherits UUID id from BeadBaseModel."""
        item = LexicalItem(lemma="test", language_code="eng")
        assert isinstance(item.id, UUID)

    def test_inherits_timestamps(self) -> None:
        """Test that lexical item inherits timestamps from BeadBaseModel."""
        item = LexicalItem(lemma="test", language_code="eng")
        assert hasattr(item, "created_at")
        assert hasattr(item, "modified_at")
        assert item.created_at is not None
        assert item.modified_at is not None

    def test_metadata_tracking(self) -> None:
        """Test that lexical item has metadata tracking."""
        item = LexicalItem(lemma="test", language_code="eng")
        assert hasattr(item, "metadata")


class TestLexicalItemSerialization:
    """Test lexical item serialization."""

    def test_model_dump(self) -> None:
        """Test lexical item serialization with model_dump."""
        item = LexicalItem(
            lemma="walk",
            language_code="eng",
            features={"pos": "VERB", "tense": "present", "frequency": 1000},
        )
        data = item.model_dump()
        assert data["lemma"] == "walk"
        assert data["language_code"] == "eng"
        assert data["features"]["pos"] == "VERB"
        assert data["features"]["tense"] == "present"
        assert data["features"]["frequency"] == 1000

    def test_deserialization(self) -> None:
        """Test lexical item deserialization with model_validate."""
        data = {
            "lemma": "run",
            "language_code": "eng",
            "features": {"pos": "VERB", "tense": "past", "frequency": 500},
        }
        item = LexicalItem.model_validate(data)
        assert item.lemma == "run"
        assert item.language_code == "eng"
        assert item.features["pos"] == "VERB"
        assert item.features["tense"] == "past"
        assert item.features["frequency"] == 500

    def test_model_copy(self) -> None:
        """Test lexical item model_copy."""
        item = LexicalItem(
            lemma="walk",
            language_code="eng",
            features={"pos": "VERB", "tense": "present"},
        )
        copy = item.with_()
        assert copy.lemma == item.lemma
        assert copy.language_code == item.language_code
        assert copy.features["pos"] == item.features["pos"]
        assert copy.id == item.id


class TestLexicalItemAttributeTypes:
    """Test lexical item with various feature types."""

    def test_string_feature(self) -> None:
        """Test lexical item with string feature."""
        item = LexicalItem(
            lemma="test", language_code="eng", features={"category": "motion"}
        )
        assert item.features["category"] == "motion"

    def test_int_feature(self) -> None:
        """Test lexical item with int feature."""
        item = LexicalItem(lemma="test", language_code="eng", features={"count": 42})
        assert item.features["count"] == 42

    def test_float_feature(self) -> None:
        """Test lexical item with float feature."""
        item = LexicalItem(lemma="test", language_code="eng", features={"rating": 4.5})
        assert item.features["rating"] == 4.5

    def test_bool_feature(self) -> None:
        """Test lexical item with bool feature."""
        item = LexicalItem(
            lemma="test", language_code="eng", features={"is_common": True}
        )
        assert item.features["is_common"] is True

    def test_list_feature(self) -> None:
        """Test lexical item with list feature."""
        item = LexicalItem(
            lemma="test", language_code="eng", features={"synonyms": ["run", "jog"]}
        )
        assert item.features["synonyms"] == ("run", "jog")


class TestLexicalItemLanguageCode:
    """Test lexical item language code functionality."""

    def test_create_with_language_code(self) -> None:
        """Test creating a lexical item with language code."""
        item = LexicalItem(lemma="walk", features={"pos": "VERB"}, language_code="en")
        assert item.language_code == "eng"  # Normalized to ISO 639-3

    def test_language_code_normalization(self) -> None:
        """Test that language codes are normalized to ISO 639-3."""
        # English: en → eng
        item1 = LexicalItem(lemma="test", language_code="en")
        assert item1.language_code == "eng"

        # Korean: ko → kor
        item2 = LexicalItem(lemma="테스트", language_code="ko")
        assert item2.language_code == "kor"

        # Already ISO 639-3 stays the same
        item3 = LexicalItem(lemma="test", language_code="eng")
        assert item3.language_code == "eng"

    def test_language_code_validation(self) -> None:
        """Test that invalid language codes are rejected."""
        with pytest.raises(ValidationError) as exc_info:
            LexicalItem(lemma="test", language_code="invalid")
        assert "Invalid language code" in str(exc_info.value)

    def test_language_code_iso639_1(self) -> None:
        """Test ISO 639-1 (2-letter) language codes."""
        item = LexicalItem(lemma="먹다", language_code="ko")
        assert item.language_code == "kor"  # Normalized to ISO 639-3

    def test_language_code_iso639_3(self) -> None:
        """Test ISO 639-3 (3-letter) language codes."""
        item = LexicalItem(lemma="test", language_code="eng")
        assert item.language_code == "eng"
