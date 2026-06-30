"""Tests for LexicalItemClass classification model."""

from __future__ import annotations

from uuid import UUID, uuid4

import didactic.api as dx
import pytest

from bead.resources.classification import LexicalItemClass
from bead.resources.lexical_item import LexicalItem


class TestLexicalItemClassCreation:
    """Tests for LexicalItemClass instantiation."""

    def test_create_empty_class(self) -> None:
        """Test creating an empty lexical item class."""
        cls = LexicalItemClass(
            name="test_class",
            description="Test classification",
            property_name="causative",
            property_value=True,
        )
        assert cls.name == "test_class"
        assert cls.description == "Test classification"
        assert cls.property_name == "causative"
        assert cls.property_value is True
        assert len(cls) == 0
        assert cls.items == ()
        assert cls.tags == ()

    def test_create_with_tags(self) -> None:
        """Test creating a class with tags."""
        cls = LexicalItemClass(
            name="test_class",
            property_name="transitive",
            tags=["verbs", "cross-linguistic"],
        )
        assert "verbs" in cls.tags
        assert "cross-linguistic" in cls.tags

    def test_create_with_metadata(self) -> None:
        """Test creating a class with custom metadata."""
        metadata = {"source": "manual", "version": "1.0"}
        cls = LexicalItemClass(
            name="test_class",
            property_name="stative",
            class_metadata=metadata,
        )
        assert cls.class_metadata["source"] == "manual"
        assert cls.class_metadata["version"] == "1.0"

    def test_validate_empty_name(self) -> None:
        """Test that empty name raises ValueError."""
        with pytest.raises(
            (ValueError, dx.ValidationError), match="name must be non-empty"
        ):
            LexicalItemClass(name="", property_name="test")

    def test_validate_empty_property_name(self) -> None:
        """Test that empty property_name raises ValueError."""
        with pytest.raises(
            (ValueError, dx.ValidationError), match="property_name must be non-empty"
        ):
            LexicalItemClass(name="test", property_name="")

    def test_has_id_and_timestamps(self) -> None:
        """Test that class inherits BeadBaseModel fields."""
        cls = LexicalItemClass(name="test", property_name="causative")
        assert isinstance(cls.id, UUID)
        assert cls.created_at is not None
        assert cls.modified_at is not None
        assert cls.version == "1.0.0"


class TestLexicalItemClassLanguageMethods:
    """Tests for language-related methods."""

    def test_languages_empty_class(self) -> None:
        """Test languages() on empty class."""
        cls = LexicalItemClass(name="test", property_name="causative")
        assert cls.languages() == set()

    def test_languages_monolingual(
        self, monolingual_causative_class: LexicalItemClass
    ) -> None:
        """Test languages() on monolingual class."""
        langs = monolingual_causative_class.languages()
        # Language codes are normalized to ISO 639-3 (3-letter codes)
        assert langs == {"eng"}

    def test_languages_multilingual(
        self, multilingual_causative_class: LexicalItemClass
    ) -> None:
        """Test languages() on multilingual class."""
        langs = multilingual_causative_class.languages()
        # Language codes are normalized to ISO 639-3 (3-letter codes)
        assert langs == {"eng", "kor"}

    def test_languages_with_none_language_code(self) -> None:
        """Test that items without language_code are excluded from languages()."""
        cls = LexicalItemClass(name="test", property_name="causative")
        cls = cls.with_item(LexicalItem(lemma="walk", language_code="en"))
        cls = cls.with_item(
            LexicalItem(lemma="run", language_code="eng")
        )  # No language_code
        # Language codes are normalized to ISO 639-3 (3-letter codes)
        assert cls.languages() == {"eng"}

    def test_get_items_by_language_monolingual(
        self, monolingual_causative_class: LexicalItemClass
    ) -> None:
        """Test get_items_by_language() on monolingual class."""
        # Can query using 2-letter or 3-letter codes (both work)
        en_items = monolingual_causative_class.get_items_by_language("en")
        assert len(en_items) == 3
        # Language codes are stored as ISO 639-3 (3-letter codes)
        assert all(item.language_code == "eng" for item in en_items)
        assert {item.lemma for item in en_items} == {"break", "open", "close"}

    def test_get_items_by_language_multilingual(
        self, multilingual_causative_class: LexicalItemClass
    ) -> None:
        """Test get_items_by_language() on multilingual class."""
        # Can query using 2-letter or 3-letter codes (both work)
        en_items = multilingual_causative_class.get_items_by_language("en")
        ko_items = multilingual_causative_class.get_items_by_language("ko")
        assert len(en_items) == 3
        assert len(ko_items) == 2
        # Language codes are stored as ISO 639-3 (3-letter codes)
        assert all(item.language_code == "eng" for item in en_items)
        assert all(item.language_code == "kor" for item in ko_items)

    def test_get_items_by_language_case_insensitive(
        self, monolingual_causative_class: LexicalItemClass
    ) -> None:
        """Test that language code filtering is case-insensitive."""
        en_items = monolingual_causative_class.get_items_by_language("EN")
        assert len(en_items) == 3

    def test_get_items_by_language_nonexistent(
        self, monolingual_causative_class: LexicalItemClass
    ) -> None:
        """Test filtering by nonexistent language code."""
        zu_items = monolingual_causative_class.get_items_by_language("zu")
        assert zu_items == ()

    def test_is_monolingual_empty(self) -> None:
        """Test is_monolingual() on empty class."""
        cls = LexicalItemClass(name="test", property_name="causative")
        assert cls.is_monolingual() is True

    def test_is_monolingual_one_language(
        self, monolingual_causative_class: LexicalItemClass
    ) -> None:
        """Test is_monolingual() with one language."""
        assert monolingual_causative_class.is_monolingual() is True

    def test_is_monolingual_two_languages(
        self, multilingual_causative_class: LexicalItemClass
    ) -> None:
        """Test is_monolingual() with two languages."""
        assert multilingual_causative_class.is_monolingual() is False

    def test_is_multilingual_empty(self) -> None:
        """Test is_multilingual() on empty class."""
        cls = LexicalItemClass(name="test", property_name="causative")
        assert cls.is_multilingual() is False

    def test_is_multilingual_one_language(
        self, monolingual_causative_class: LexicalItemClass
    ) -> None:
        """Test is_multilingual() with one language."""
        assert monolingual_causative_class.is_multilingual() is False

    def test_is_multilingual_two_languages(
        self, multilingual_causative_class: LexicalItemClass
    ) -> None:
        """Test is_multilingual() with two languages."""
        assert multilingual_causative_class.is_multilingual() is True

    def test_languages_three_languages(
        self,
        multilingual_causative_class: LexicalItemClass,
        zulu_causative_verbs: dict[UUID, LexicalItem],
    ) -> None:
        """Test class with three languages."""
        for item in zulu_causative_verbs.values():
            multilingual_causative_class = multilingual_causative_class.with_item(item)
        # Language codes are normalized to ISO 639-3 (3-letter codes)
        assert multilingual_causative_class.languages() == {"eng", "kor", "zul"}
        assert multilingual_causative_class.is_multilingual() is True


class TestLexicalItemClassCRUDOperations:
    """Tests for add/remove/get operations."""

    def test_add_item(self) -> None:
        """Test adding an item to the class."""
        cls = LexicalItemClass(name="test", property_name="causative")
        item = LexicalItem(lemma="break", language_code="en")
        cls = cls.with_item(item)
        assert len(cls) == 1
        assert item.id in cls

    def test_add_item_updates_modified_time(self) -> None:
        """Test that adding an item updates modified_at."""
        cls = LexicalItemClass(name="test", property_name="causative")
        original_modified = cls.modified_at
        item = LexicalItem(lemma="break", language_code="eng")
        cls = cls.with_item(item)
        assert cls.modified_at > original_modified

    def test_add_duplicate_item_raises_error(self) -> None:
        """Test that adding duplicate item raises ValueError."""
        cls = LexicalItemClass(name="test", property_name="causative")
        item = LexicalItem(lemma="break", language_code="eng")
        cls = cls.with_item(item)
        with pytest.raises(
            (ValueError, dx.ValidationError), match="already exists in class"
        ):
            cls = cls.with_item(item)

    def test_remove_item(self) -> None:
        """Test removing an item from the class."""
        cls = LexicalItemClass(name="test", property_name="causative")
        item = LexicalItem(lemma="break", language_code="eng")
        cls = cls.with_item(item)
        cls, removed = cls.without_item(item.id)
        assert removed.lemma == "break"
        assert len(cls) == 0
        assert item.id not in cls

    def test_remove_item_updates_modified_time(self) -> None:
        """Test that removing an item updates modified_at."""
        cls = LexicalItemClass(name="test", property_name="causative")
        item = LexicalItem(lemma="break", language_code="eng")
        cls = cls.with_item(item)
        original_modified = cls.modified_at
        cls, _ = cls.without_item(item.id)
        assert cls.modified_at > original_modified

    def test_remove_nonexistent_item_raises_error(self) -> None:
        """Test that removing nonexistent item raises KeyError."""
        cls = LexicalItemClass(name="test", property_name="causative")
        with pytest.raises(KeyError, match="not found in class"):
            cls.without_item(uuid4())

    def test_get_item(self) -> None:
        """Test getting an item by ID."""
        cls = LexicalItemClass(name="test", property_name="causative")
        item = LexicalItem(lemma="break", language_code="eng")
        cls = cls.with_item(item)
        retrieved = cls.by_id(item.id)
        assert retrieved is not None
        assert retrieved.lemma == "break"

    def test_get_nonexistent_item_returns_none(self) -> None:
        """Test that getting nonexistent item returns None."""
        cls = LexicalItemClass(name="test", property_name="causative")
        assert cls.by_id(uuid4()) is None

    def test_len_empty(self) -> None:
        """Test __len__ on empty class."""
        cls = LexicalItemClass(name="test", property_name="causative")
        assert len(cls) == 0

    def test_len_with_items(
        self, monolingual_causative_class: LexicalItemClass
    ) -> None:
        """Test __len__ with items."""
        assert len(monolingual_causative_class) == 3

    def test_contains_true(self) -> None:
        """Test __contains__ for existing item."""
        cls = LexicalItemClass(name="test", property_name="causative")
        item = LexicalItem(lemma="break", language_code="eng")
        cls = cls.with_item(item)
        assert item.id in cls

    def test_contains_false(self) -> None:
        """Test __contains__ for nonexistent item."""
        cls = LexicalItemClass(name="test", property_name="causative")
        assert uuid4() not in cls

    def test_iter_empty(self) -> None:
        """Test __iter__ on empty class."""
        cls = LexicalItemClass(name="test", property_name="causative")
        items = list(cls)
        assert items == []

    def test_iter_with_items(
        self, monolingual_causative_class: LexicalItemClass
    ) -> None:
        """Test __iter__ with items."""
        items = list(monolingual_causative_class)
        assert len(items) == 3
        lemmas = {item.lemma for item in items}
        assert lemmas == {"break", "open", "close"}


class TestLexicalItemClassSerialization:
    """Tests for JSONLines serialization."""

    def test_model_dump(self) -> None:
        """Test that LexicalItemClass can be serialized via model_dump."""
        cls = LexicalItemClass(
            name="causative_verbs",
            description="Causative verbs",
            property_name="causative",
            property_value=True,
            tags=["verbs"],
        )
        item = LexicalItem(lemma="break", language_code="en")
        cls = cls.with_item(item)
        # Test model_dump
        data = cls.model_dump()
        assert data["name"] == "causative_verbs"
        assert data["property_name"] == "causative"
        assert data["property_value"] is True
        assert len(data["items"]) == 1

    def test_model_dump_json(self) -> None:
        """Test that LexicalItemClass can be serialized to JSON."""
        cls = LexicalItemClass(
            name="causative_verbs",
            property_name="causative",
        )
        item = LexicalItem(lemma="break", language_code="en")
        cls = cls.with_item(item)
        # Test model_dump_json
        json_str = cls.model_dump_json()
        assert isinstance(json_str, str)
        assert "causative_verbs" in json_str
        assert "break" in json_str

    def test_deserialization_round_trip(self) -> None:
        """Test serialization and deserialization round trip."""
        cls = LexicalItemClass(
            name="causative_verbs",
            description="Test class",
            property_name="causative",
            property_value=True,
            tags=["verbs", "test"],
        )
        item1 = LexicalItem(lemma="break", language_code="en")
        item2 = LexicalItem(lemma="open", language_code="en")
        cls = cls.with_item(item1)
        cls = cls.with_item(item2)
        # Serialize
        data = cls.model_dump()

        # Deserialize
        cls_restored = LexicalItemClass(**data)

        # Verify
        assert cls_restored.name == cls.name
        assert cls_restored.description == cls.description
        assert cls_restored.property_name == cls.property_name
        assert cls_restored.property_value == cls.property_value
        assert cls_restored.tags == cls.tags
        assert len(cls_restored) == 2
        assert item1.id in cls_restored
        assert item2.id in cls_restored


class TestLexicalItemClassEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_class_operations(self) -> None:
        """Test operations on an empty class."""
        cls = LexicalItemClass(name="empty", property_name="test")
        assert len(cls) == 0
        assert list(cls) == []
        assert cls.languages() == set()
        assert cls.is_monolingual() is True
        assert cls.is_multilingual() is False

    def test_items_without_language_code(self) -> None:
        """Test handling of items with various language codes."""
        cls = LexicalItemClass(name="test", property_name="causative")
        item1 = LexicalItem(lemma="break", language_code="en")
        item2 = LexicalItem(lemma="run", language_code="eng")
        item3 = LexicalItem(lemma="walk", language_code="ko")  # Different language

        cls = cls.with_item(item1)
        cls = cls.with_item(item2)
        cls = cls.with_item(item3)
        assert len(cls) == 3
        # Language codes are normalized to ISO 639-3 (3-letter codes)
        assert cls.languages() == {"eng", "kor"}
        # Can query using 2-letter or 3-letter codes (both work)
        # "en" normalizes to "eng", so both item1 and item2 are returned
        eng_items = cls.get_items_by_language("en")
        assert len(eng_items) == 2
        assert item1 in eng_items
        assert item2 in eng_items

    def test_property_value_can_be_none(self) -> None:
        """Test that property_value can be None."""
        cls = LexicalItemClass(
            name="test",
            property_name="transitive",
            property_value=None,
        )
        assert cls.property_value is None

    def test_property_value_various_types(self) -> None:
        """Test that property_value can be various types."""
        # Boolean
        cls1 = LexicalItemClass(
            name="test1", property_name="transitive", property_value=True
        )
        assert cls1.property_value is True

        # String
        cls2 = LexicalItemClass(
            name="test2", property_name="voice", property_value="active"
        )
        assert cls2.property_value == "active"

        # Number
        cls3 = LexicalItemClass(name="test3", property_name="valency", property_value=2)
        assert cls3.property_value == 2

        # Dict
        cls4 = LexicalItemClass(
            name="test4",
            property_name="features",
            property_value={"mood": "indicative", "tense": "present"},
        )
        assert cls4.property_value == {"mood": "indicative", "tense": "present"}

    def test_multiple_add_remove_operations(self) -> None:
        """Test multiple add/remove operations."""
        cls = LexicalItemClass(name="test", property_name="causative")
        items = [
            LexicalItem(lemma="break", language_code="eng"),
            LexicalItem(lemma="open", language_code="eng"),
            LexicalItem(lemma="close", language_code="eng"),
        ]

        # Add all
        for item in items:
            cls = cls.with_item(item)
        assert len(cls) == 3

        # Remove one
        cls, _ = cls.without_item(items[0].id)
        assert len(cls) == 2

        # Add it back
        cls = cls.with_item(items[0])
        assert len(cls) == 3

        # Remove all
        for item in items:
            cls, _ = cls.without_item(item.id)
        assert len(cls) == 0
