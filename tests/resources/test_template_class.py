"""Tests for TemplateClass classification model."""

from __future__ import annotations

from uuid import UUID, uuid4

import didactic.api as dx
import pytest

from bead.resources.classification import TemplateClass
from bead.resources.template import Slot, Template


class TestTemplateClassCreation:
    """Tests for TemplateClass instantiation."""

    def test_create_empty_class(self) -> None:
        """Test creating an empty template class."""
        cls = TemplateClass(
            name="test_class",
            description="Test classification",
            property_name="transitive",
            property_value=True,
        )
        assert cls.name == "test_class"
        assert cls.description == "Test classification"
        assert cls.property_name == "transitive"
        assert cls.property_value is True
        assert len(cls) == 0
        assert cls.templates == ()
        assert cls.tags == ()

    def test_create_with_tags(self) -> None:
        """Test creating a class with tags."""
        cls = TemplateClass(
            name="test_class",
            property_name="transitive",
            tags=["templates", "cross-linguistic"],
        )
        assert "templates" in cls.tags
        assert "cross-linguistic" in cls.tags

    def test_create_with_metadata(self) -> None:
        """Test creating a class with custom metadata."""
        metadata = {"source": "manual", "version": "1.0"}
        cls = TemplateClass(
            name="test_class",
            property_name="wh_question",
            class_metadata=metadata,
        )
        assert cls.class_metadata["source"] == "manual"
        assert cls.class_metadata["version"] == "1.0"

    def test_validate_empty_name(self) -> None:
        """Test that empty name raises ValueError."""
        with pytest.raises(
            (ValueError, dx.ValidationError), match="name must be non-empty"
        ):
            TemplateClass(name="", property_name="test")

    def test_validate_empty_property_name(self) -> None:
        """Test that empty property_name raises ValueError."""
        with pytest.raises(
            (ValueError, dx.ValidationError), match="property_name must be non-empty"
        ):
            TemplateClass(name="test", property_name="")

    def test_has_id_and_timestamps(self) -> None:
        """Test that class inherits BeadBaseModel fields."""
        cls = TemplateClass(name="test", property_name="transitive")
        assert isinstance(cls.id, UUID)
        assert cls.created_at is not None
        assert cls.modified_at is not None
        assert cls.version == "1.0.0"


class TestTemplateClassLanguageMethods:
    """Tests for language-related methods."""

    def test_languages_empty_class(self) -> None:
        """Test languages() on empty class."""
        cls = TemplateClass(name="test", property_name="transitive")
        assert cls.languages() == set()

    def test_languages_monolingual(
        self, monolingual_transitive_template_class: TemplateClass
    ) -> None:
        """Test languages() on monolingual class."""
        langs = monolingual_transitive_template_class.languages()
        # Language codes are normalized to ISO 639-3 (3-letter codes)
        assert langs == {"eng"}

    def test_languages_multilingual(
        self, multilingual_transitive_template_class: TemplateClass
    ) -> None:
        """Test languages() on multilingual class."""
        langs = multilingual_transitive_template_class.languages()
        # Language codes are normalized to ISO 639-3 (3-letter codes)
        assert langs == {"eng", "kor"}

    def test_languages_with_none_language_code(self) -> None:
        """Test that templates without language_code are excluded from languages()."""
        cls = TemplateClass(name="test", property_name="transitive")
        t1 = Template(
            name="svo_en",
            template_string="{s} {v} {o}.",
            slots={"s": Slot(name="s"), "v": Slot(name="v"), "o": Slot(name="o")},
            language_code="en",
        )
        t2 = Template(
            name="svo_none",
            template_string="{s} {v} {o}.",
            slots={"s": Slot(name="s"), "v": Slot(name="v"), "o": Slot(name="o")},
        )
        cls = cls.with_template(t1)
        cls = cls.with_template(t2)
        # Language codes are normalized to ISO 639-3 (3-letter codes)
        assert cls.languages() == {"eng"}

    def test_get_templates_by_language_monolingual(
        self, monolingual_transitive_template_class: TemplateClass
    ) -> None:
        """Test get_templates_by_language() on monolingual class."""
        # Can query using 2-letter or 3-letter codes (both work)
        en_templates = monolingual_transitive_template_class.get_templates_by_language(
            "en"
        )
        assert len(en_templates) == 2
        # Language codes are stored as ISO 639-3 (3-letter codes)
        assert all(template.language_code == "eng" for template in en_templates)
        assert {template.name for template in en_templates} == {
            "svo_simple",
            "svo_with_adverb",
        }

    def test_get_templates_by_language_multilingual(
        self, multilingual_transitive_template_class: TemplateClass
    ) -> None:
        """Test get_templates_by_language() on multilingual class."""
        # Can query using 2-letter or 3-letter codes (both work)
        en_templates = multilingual_transitive_template_class.get_templates_by_language(
            "en"
        )
        ko_templates = multilingual_transitive_template_class.get_templates_by_language(
            "ko"
        )
        assert len(en_templates) == 2
        assert len(ko_templates) == 2
        # Language codes are stored as ISO 639-3 (3-letter codes)
        assert all(template.language_code == "eng" for template in en_templates)
        assert all(template.language_code == "kor" for template in ko_templates)

    def test_get_templates_by_language_case_insensitive(
        self, monolingual_transitive_template_class: TemplateClass
    ) -> None:
        """Test that language code filtering is case-insensitive."""
        en_templates = monolingual_transitive_template_class.get_templates_by_language(
            "EN"
        )
        assert len(en_templates) == 2

    def test_get_templates_by_language_nonexistent(
        self, monolingual_transitive_template_class: TemplateClass
    ) -> None:
        """Test filtering by nonexistent language code."""
        zu_templates = monolingual_transitive_template_class.get_templates_by_language(
            "zu"
        )
        assert zu_templates == ()

    def test_is_monolingual_empty(self) -> None:
        """Test is_monolingual() on empty class."""
        cls = TemplateClass(name="test", property_name="transitive")
        assert cls.is_monolingual() is True

    def test_is_monolingual_one_language(
        self, monolingual_transitive_template_class: TemplateClass
    ) -> None:
        """Test is_monolingual() with one language."""
        assert monolingual_transitive_template_class.is_monolingual() is True

    def test_is_monolingual_two_languages(
        self, multilingual_transitive_template_class: TemplateClass
    ) -> None:
        """Test is_monolingual() with two languages."""
        assert multilingual_transitive_template_class.is_monolingual() is False

    def test_is_multilingual_empty(self) -> None:
        """Test is_multilingual() on empty class."""
        cls = TemplateClass(name="test", property_name="transitive")
        assert cls.is_multilingual() is False

    def test_is_multilingual_one_language(
        self, monolingual_transitive_template_class: TemplateClass
    ) -> None:
        """Test is_multilingual() with one language."""
        assert monolingual_transitive_template_class.is_multilingual() is False

    def test_is_multilingual_two_languages(
        self, multilingual_transitive_template_class: TemplateClass
    ) -> None:
        """Test is_multilingual() with two languages."""
        assert multilingual_transitive_template_class.is_multilingual() is True


class TestTemplateClassCRUDOperations:
    """Tests for add/remove/get operations."""

    def test_add_template(self) -> None:
        """Test adding a template to the class."""
        cls = TemplateClass(name="test", property_name="transitive")
        template = Template(
            name="svo",
            template_string="{s} {v} {o}.",
            slots={"s": Slot(name="s"), "v": Slot(name="v"), "o": Slot(name="o")},
            language_code="en",
        )
        cls = cls.with_template(template)
        assert len(cls) == 1
        assert template.id in cls

    def test_add_template_updates_modified_time(self) -> None:
        """Test that adding a template updates modified_at."""
        cls = TemplateClass(name="test", property_name="transitive")
        original_modified = cls.modified_at
        template = Template(
            name="svo",
            template_string="{s} {v} {o}.",
            slots={"s": Slot(name="s"), "v": Slot(name="v"), "o": Slot(name="o")},
        )
        cls = cls.with_template(template)
        assert cls.modified_at > original_modified

    def test_add_duplicate_template_raises_error(self) -> None:
        """Test that adding duplicate template raises ValueError."""
        cls = TemplateClass(name="test", property_name="transitive")
        template = Template(
            name="svo",
            template_string="{s} {v} {o}.",
            slots={"s": Slot(name="s"), "v": Slot(name="v"), "o": Slot(name="o")},
        )
        cls = cls.with_template(template)
        with pytest.raises(
            (ValueError, dx.ValidationError), match="already exists in class"
        ):
            cls = cls.with_template(template)

    def test_remove_template(self) -> None:
        """Test removing a template from the class."""
        cls = TemplateClass(name="test", property_name="transitive")
        template = Template(
            name="svo",
            template_string="{s} {v} {o}.",
            slots={"s": Slot(name="s"), "v": Slot(name="v"), "o": Slot(name="o")},
        )
        cls = cls.with_template(template)
        cls, removed = cls.without_template(template.id)
        assert removed.name == "svo"
        assert len(cls) == 0
        assert template.id not in cls

    def test_remove_template_updates_modified_time(self) -> None:
        """Test that removing a template updates modified_at."""
        cls = TemplateClass(name="test", property_name="transitive")
        template = Template(
            name="svo",
            template_string="{s} {v} {o}.",
            slots={"s": Slot(name="s"), "v": Slot(name="v"), "o": Slot(name="o")},
        )
        cls = cls.with_template(template)
        original_modified = cls.modified_at
        cls, _ = cls.without_template(template.id)
        assert cls.modified_at > original_modified

    def test_remove_nonexistent_template_raises_error(self) -> None:
        """Test that removing nonexistent template raises KeyError."""
        cls = TemplateClass(name="test", property_name="transitive")
        with pytest.raises(KeyError, match="not found in class"):
            cls.without_template(uuid4())

    def test_get_template(self) -> None:
        """Test getting a template by ID."""
        cls = TemplateClass(name="test", property_name="transitive")
        template = Template(
            name="svo",
            template_string="{s} {v} {o}.",
            slots={"s": Slot(name="s"), "v": Slot(name="v"), "o": Slot(name="o")},
        )
        cls = cls.with_template(template)
        retrieved = cls.by_id(template.id)
        assert retrieved is not None
        assert retrieved.name == "svo"

    def test_get_nonexistent_template_returns_none(self) -> None:
        """Test that getting nonexistent template returns None."""
        cls = TemplateClass(name="test", property_name="transitive")
        assert cls.by_id(uuid4()) is None

    def test_len_empty(self) -> None:
        """Test __len__ on empty class."""
        cls = TemplateClass(name="test", property_name="transitive")
        assert len(cls) == 0

    def test_len_with_templates(
        self, monolingual_transitive_template_class: TemplateClass
    ) -> None:
        """Test __len__ with templates."""
        assert len(monolingual_transitive_template_class) == 2

    def test_contains_true(self) -> None:
        """Test __contains__ for existing template."""
        cls = TemplateClass(name="test", property_name="transitive")
        template = Template(
            name="svo",
            template_string="{s} {v} {o}.",
            slots={"s": Slot(name="s"), "v": Slot(name="v"), "o": Slot(name="o")},
        )
        cls = cls.with_template(template)
        assert template.id in cls

    def test_contains_false(self) -> None:
        """Test __contains__ for nonexistent template."""
        cls = TemplateClass(name="test", property_name="transitive")
        assert uuid4() not in cls

    def test_iter_empty(self) -> None:
        """Test __iter__ on empty class."""
        cls = TemplateClass(name="test", property_name="transitive")
        templates = list(cls)
        assert templates == []

    def test_iter_with_templates(
        self, monolingual_transitive_template_class: TemplateClass
    ) -> None:
        """Test __iter__ with templates."""
        templates = list(monolingual_transitive_template_class)
        assert len(templates) == 2
        names = {template.name for template in templates}
        assert names == {"svo_simple", "svo_with_adverb"}


class TestTemplateClassSerialization:
    """Tests for JSONLines serialization."""

    def test_model_dump(self) -> None:
        """Test that TemplateClass can be serialized via model_dump."""
        cls = TemplateClass(
            name="transitive_templates",
            description="Transitive templates",
            property_name="transitive",
            property_value=True,
            tags=["templates"],
        )
        template = Template(
            name="svo",
            template_string="{s} {v} {o}.",
            slots={"s": Slot(name="s"), "v": Slot(name="v"), "o": Slot(name="o")},
            language_code="en",
        )
        cls = cls.with_template(template)
        # Test model_dump
        data = cls.model_dump()
        assert data["name"] == "transitive_templates"
        assert data["property_name"] == "transitive"
        assert data["property_value"] is True
        assert len(data["templates"]) == 1

    def test_model_dump_json(self) -> None:
        """Test that TemplateClass can be serialized to JSON."""
        cls = TemplateClass(
            name="transitive_templates",
            property_name="transitive",
        )
        template = Template(
            name="svo",
            template_string="{s} {v} {o}.",
            slots={"s": Slot(name="s"), "v": Slot(name="v"), "o": Slot(name="o")},
            language_code="en",
        )
        cls = cls.with_template(template)
        # Test model_dump_json
        json_str = cls.model_dump_json()
        assert isinstance(json_str, str)
        assert "transitive_templates" in json_str
        assert "svo" in json_str

    def test_deserialization_round_trip(self) -> None:
        """Test serialization and deserialization round trip."""
        cls = TemplateClass(
            name="transitive_templates",
            description="Test class",
            property_name="transitive",
            property_value=True,
            tags=["templates", "test"],
        )
        t1 = Template(
            name="svo1",
            template_string="{s} {v} {o}.",
            slots={"s": Slot(name="s"), "v": Slot(name="v"), "o": Slot(name="o")},
            language_code="en",
        )
        t2 = Template(
            name="svo2",
            template_string="{s} {v} {o}.",
            slots={"s": Slot(name="s"), "v": Slot(name="v"), "o": Slot(name="o")},
            language_code="en",
        )
        cls = cls.with_template(t1)
        cls = cls.with_template(t2)
        # Serialize
        data = cls.model_dump()

        # Deserialize
        cls_restored = TemplateClass(**data)

        # Verify
        assert cls_restored.name == cls.name
        assert cls_restored.description == cls.description
        assert cls_restored.property_name == cls.property_name
        assert cls_restored.property_value == cls.property_value
        assert cls_restored.tags == cls.tags
        assert len(cls_restored) == 2
        assert t1.id in cls_restored
        assert t2.id in cls_restored


class TestTemplateClassEdgeCases:
    """Tests for edge cases and special scenarios."""

    def test_empty_class_operations(self) -> None:
        """Test operations on an empty class."""
        cls = TemplateClass(name="empty", property_name="test")
        assert len(cls) == 0
        assert list(cls) == []
        assert cls.languages() == set()
        assert cls.is_monolingual() is True
        assert cls.is_multilingual() is False

    def test_templates_without_language_code(self) -> None:
        """Test handling of templates without language_code."""
        cls = TemplateClass(name="test", property_name="transitive")
        t1 = Template(
            name="svo1",
            template_string="{s} {v} {o}.",
            slots={"s": Slot(name="s"), "v": Slot(name="v"), "o": Slot(name="o")},
            language_code="en",
        )
        t2 = Template(
            name="svo2",
            template_string="{s} {v} {o}.",
            slots={"s": Slot(name="s"), "v": Slot(name="v"), "o": Slot(name="o")},
        )
        t3 = Template(
            name="svo3",
            template_string="{s} {v} {o}.",
            slots={"s": Slot(name="s"), "v": Slot(name="v"), "o": Slot(name="o")},
        )

        cls = cls.with_template(t1)
        cls = cls.with_template(t2)
        cls = cls.with_template(t3)
        assert len(cls) == 3
        # Language codes are normalized to ISO 639-3 (3-letter codes)
        assert cls.languages() == {"eng"}
        # Can query using 2-letter or 3-letter codes (both work)
        assert cls.get_templates_by_language("en") == (t1,)

    def test_property_value_can_be_none(self) -> None:
        """Test that property_value can be None."""
        cls = TemplateClass(
            name="test",
            property_name="transitive",
            property_value=None,
        )
        assert cls.property_value is None

    def test_property_value_various_types(self) -> None:
        """Test that property_value can be various types."""
        # Boolean
        cls1 = TemplateClass(
            name="test1", property_name="transitive", property_value=True
        )
        assert cls1.property_value is True

        # String
        cls2 = TemplateClass(
            name="test2", property_name="voice", property_value="passive"
        )
        assert cls2.property_value == "passive"

        # Number
        cls3 = TemplateClass(name="test3", property_name="slot_count", property_value=3)
        assert cls3.property_value == 3

        # Dict
        cls4 = TemplateClass(
            name="test4",
            property_name="properties",
            property_value={"tense": "present", "aspect": "progressive"},
        )
        assert cls4.property_value == {"tense": "present", "aspect": "progressive"}

    def test_multiple_add_remove_operations(self) -> None:
        """Test multiple add/remove operations."""
        cls = TemplateClass(name="test", property_name="transitive")
        templates = [
            Template(
                name=f"svo{i}",
                template_string="{s} {v} {o}.",
                slots={"s": Slot(name="s"), "v": Slot(name="v"), "o": Slot(name="o")},
            )
            for i in range(3)
        ]

        # Add all
        for template in templates:
            cls = cls.with_template(template)
        assert len(cls) == 3

        # Remove one
        cls, _ = cls.without_template(templates[0].id)
        assert len(cls) == 2

        # Add it back
        cls = cls.with_template(templates[0])
        assert len(cls) == 3

        # Remove all
        for template in templates:
            cls, _ = cls.without_template(template.id)
        assert len(cls) == 0
