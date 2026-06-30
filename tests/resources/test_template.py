"""Tests for template and structure models."""

from __future__ import annotations

import pytest
from didactic.api import ValidationError

from bead.resources import (
    Constraint,
    Slot,
    Template,
    TemplateSequence,
    TemplateTree,
)
from bead.resources.template import slots_match_template


class TestSlot:
    """Test slot model."""

    def test_create_with_all_fields(self) -> None:
        """Test creating a slot with all fields."""
        constraint = Constraint(expression="self.pos == 'VERB'")
        slot = Slot(
            name="subject",
            description="The subject of the sentence",
            constraints=[constraint],
            required=True,
            default_value="default",
        )
        assert slot.name == "subject"
        assert slot.description == "The subject of the sentence"
        assert len(slot.constraints) == 1
        assert slot.required is True
        assert slot.default_value == "default"

    def test_create_with_minimal_fields(self) -> None:
        """Test creating a slot with minimal fields."""
        slot = Slot(name="verb")
        assert slot.name == "verb"
        assert slot.description is None
        assert slot.constraints == ()
        assert slot.required is True
        assert slot.default_value is None

    def test_create_with_multiple_constraints(self) -> None:
        """Test creating a slot with multiple constraints."""
        constraint1 = Constraint(expression="self.pos == 'VERB'")
        constraint2 = Constraint(expression="self.tense == 'present'")
        slot = Slot(name="verb", constraints=[constraint1, constraint2])
        assert len(slot.constraints) == 2

    def test_serialization(self) -> None:
        """Test slot serialization."""
        slot = Slot(name="subject", required=True)
        data = slot.model_dump()
        assert data["name"] == "subject"
        assert data["required"] is True

    def test_deserialization(self) -> None:
        """Test slot deserialization."""
        data = {
            "name": "object",
            "description": "The object",
            "required": False,
        }
        slot = Slot.model_validate(data)
        assert slot.name == "object"
        assert slot.description == "The object"
        assert slot.required is False

    def test_required_vs_optional_slots(self) -> None:
        """Test required vs optional slots."""
        required_slot = Slot(name="subject", required=True)
        optional_slot = Slot(name="modifier", required=False)
        assert required_slot.required is True
        assert optional_slot.required is False

    def test_slot_with_default_value(self) -> None:
        """Test slot with default value."""
        slot = Slot(name="determiner", default_value="the")
        assert slot.default_value == "the"

    def test_invalid_name_fails(self) -> None:
        """Test that invalid slot name validation fails."""
        with pytest.raises(ValidationError) as exc_info:
            Slot(name="invalid-name")
        assert "must be a valid Python identifier" in str(exc_info.value)

    def test_empty_name_fails(self) -> None:
        """Test that empty slot name validation fails."""
        with pytest.raises(ValidationError) as exc_info:
            Slot(name="")
        assert "name must be non-empty" in str(exc_info.value)


class TestTemplate:
    """Test template model."""

    def test_create_with_all_fields(self) -> None:
        """Test creating a template with all fields."""
        slot1 = Slot(name="subject")
        slot2 = Slot(name="verb")
        template = Template(
            name="simple",
            template_string="{subject} {verb}.",
            slots={"subject": slot1, "verb": slot2},
            description="A simple sentence",
            tags=["simple", "declarative"],
            metadata={"author": "test"},
        )
        assert template.name == "simple"
        assert template.template_string == "{subject} {verb}."
        assert len(template.slots) == 2
        assert template.description == "A simple sentence"
        assert len(template.tags) == 2
        assert template.metadata["author"] == "test"

    def test_create_with_minimal_fields(self) -> None:
        """Test creating a template with minimal fields."""
        slot = Slot(name="word")
        template = Template(
            name="minimal",
            template_string="{word}",
            slots={"word": slot},
        )
        assert template.name == "minimal"
        assert template.slots == {"word": slot}
        assert template.description is None
        assert template.language_code is None
        assert template.tags == ()
        assert template.metadata == {}

    def test_create_with_multiple_slots(self) -> None:
        """Test creating a template with multiple slots."""
        slots = {
            "subject": Slot(name="subject"),
            "verb": Slot(name="verb"),
            "object": Slot(name="object"),
        }
        template = Template(
            name="transitive",
            template_string="{subject} {verb} {object}.",
            slots=slots,
        )
        assert len(template.slots) == 3

    def test_template_string_with_placeholders(self) -> None:
        """Test template string with slot placeholders."""
        slot = Slot(name="word")
        template = Template(
            name="test",
            template_string="The {word} is here.",
            slots={"word": slot},
        )
        assert "{word}" in template.template_string

    def test_serialization(self) -> None:
        """Test template serialization."""
        slot = Slot(name="word")
        template = Template(
            name="test",
            template_string="{word}",
            slots={"word": slot},
        )
        data = template.model_dump()
        assert data["name"] == "test"
        assert data["template_string"] == "{word}"

    def test_deserialization(self) -> None:
        """Test template deserialization."""
        data = {
            "name": "test",
            "template_string": "{word}",
            "slots": {
                "word": {
                    "name": "word",
                    "required": True,
                }
            },
        }
        template = Template.model_validate(data)
        assert template.name == "test"
        assert len(template.slots) == 1

    def test_template_tags(self) -> None:
        """Test template tags."""
        slot = Slot(name="word")
        template = Template(
            name="test",
            template_string="{word}",
            slots={"word": slot},
            tags=["simple", "test", "example"],
        )
        assert len(template.tags) == 3
        assert "simple" in template.tags

    def test_template_metadata(self) -> None:
        """Test template metadata."""
        slot = Slot(name="word")
        template = Template(
            name="test",
            template_string="{word}",
            slots={"word": slot},
            metadata={"version": 1, "author": "test"},
        )
        assert template.metadata["version"] == 1
        assert template.metadata["author"] == "test"

    def test_validation_slots_in_template_exist_in_dict(self) -> None:
        """Test validation: all slots in template_string exist in slots dict."""
        slot = Slot(name="word")
        template = Template(
            name="test",
            template_string="{word} {missing}",
            slots={"word": slot},
        )
        with pytest.raises(ValueError) as exc_info:
            slots_match_template(template)
        assert "not in slots dict" in str(exc_info.value)

    def test_validation_slots_in_dict_referenced_in_template(self) -> None:
        """Test validation: all slots in dict referenced in template_string."""
        slot1 = Slot(name="word")
        slot2 = Slot(name="extra")
        template = Template(
            name="test",
            template_string="{word}",
            slots={"word": slot1, "extra": slot2},
        )
        with pytest.raises(ValueError) as exc_info:
            slots_match_template(template)
        assert "not referenced in template" in str(exc_info.value)

    def test_validation_slot_names_valid_identifiers(self) -> None:
        """Test validation: slot names are valid identifiers."""
        # This is tested at the Slot level, but let's verify it works in Template too
        slot = Slot(name="valid_name")
        template = Template(
            name="test",
            template_string="{valid_name}",
            slots={"valid_name": slot},
        )
        assert template.slots["valid_name"].name == "valid_name"

    def test_validation_slot_key_matches_name(self) -> None:
        """Test validation: slot key matches slot name."""
        slot = Slot(name="word")
        template = Template(
            name="test",
            template_string="{word}",
            slots={"different_key": slot},
        )
        with pytest.raises(ValueError) as exc_info:
            slots_match_template(template)
        assert "not in slots dict" in str(exc_info.value)

    def test_template_with_nested_constraints(self) -> None:
        """Test template with nested constraints in slots."""
        constraint = Constraint(expression="self.pos == 'VERB'")
        slot = Slot(name="verb", constraints=[constraint])
        template = Template(
            name="test",
            template_string="{verb}",
            slots={"verb": slot},
        )
        assert len(template.slots["verb"].constraints) == 1
        assert isinstance(template.slots["verb"].constraints[0], Constraint)

    def test_empty_name_fails(self) -> None:
        """Test that empty template name validation fails."""
        slot = Slot(name="word")
        with pytest.raises(ValidationError) as exc_info:
            Template(
                name="",
                template_string="{word}",
                slots={"word": slot},
            )
        assert "name must be non-empty" in str(exc_info.value)

    def test_empty_template_string_fails(self) -> None:
        """Test that empty template_string validation fails."""
        with pytest.raises(ValidationError) as exc_info:
            Template(
                name="test",
                template_string="",
                slots={},
            )
        assert "template_string must be non-empty" in str(exc_info.value)


class TestTemplateLanguageCode:
    """Test template language code functionality."""

    def test_create_with_language_code(self) -> None:
        """Test creating a template with language code."""
        slot = Slot(name="x")
        template = Template(
            name="test",
            template_string="{x}.",
            slots={"x": slot},
            language_code="en",
        )
        assert template.language_code == "eng"  # Normalized to ISO 639-3

    def test_language_code_normalization(self) -> None:
        """Test that language codes are normalized to ISO 639-3."""
        slot = Slot(name="x")
        # English: en → eng
        template1 = Template(
            name="test",
            template_string="{x}.",
            slots={"x": slot},
            language_code="en",
        )
        assert template1.language_code == "eng"

        # Korean: ko → kor
        slot2 = Slot(name="y")
        template2 = Template(
            name="test2",
            template_string="{y}.",
            slots={"y": slot2},
            language_code="ko",
        )
        assert template2.language_code == "kor"

    def test_language_code_validation(self) -> None:
        """Test that invalid language codes are rejected."""
        slot = Slot(name="x")
        with pytest.raises(ValidationError) as exc_info:
            Template(
                name="test",
                template_string="{x}.",
                slots={"x": slot},
                language_code="invalid",
            )
        assert "Invalid language code" in str(exc_info.value)

    def test_language_code_iso639_1(self) -> None:
        """Test ISO 639-1 (2-letter) language codes."""
        slot = Slot(name="x")
        template = Template(
            name="test",
            template_string="{x}.",
            slots={"x": slot},
            language_code="ko",
        )
        assert template.language_code == "kor"  # Normalized to ISO 639-3

    def test_language_code_iso639_3(self) -> None:
        """Test ISO 639-3 (3-letter) language codes."""
        slot = Slot(name="x")
        template = Template(
            name="test",
            template_string="{x}.",
            slots={"x": slot},
            language_code="eng",
        )
        assert template.language_code == "eng"

    def test_language_code_none(self) -> None:
        """Test that None language code is valid (optional)."""
        slot = Slot(name="x")
        template = Template(
            name="test",
            template_string="{x}.",
            slots={"x": slot},
            language_code=None,
        )
        assert template.language_code is None


class TestTemplateSequence:
    """Test template sequence model."""

    def test_create_with_multiple_templates(self) -> None:
        """Test creating a sequence with multiple templates."""
        slot1 = Slot(name="word1")
        slot2 = Slot(name="word2")
        template1 = Template(
            name="t1",
            template_string="{word1}",
            slots={"word1": slot1},
        )
        template2 = Template(
            name="t2",
            template_string="{word2}",
            slots={"word2": slot2},
        )
        sequence = TemplateSequence(
            name="sequence",
            templates=[template1, template2],
        )
        assert sequence.name == "sequence"
        assert len(sequence.templates) == 2

    def test_create_with_cross_template_constraints(self) -> None:
        """Test creating a sequence with cross-template constraints."""
        slot = Slot(name="word")
        template = Template(
            name="t1",
            template_string="{word}",
            slots={"word": slot},
        )
        constraint = Constraint(expression="self.pos == 'VERB'")
        sequence = TemplateSequence(
            name="sequence",
            templates=[template],
            constraints=[constraint],
        )
        assert len(sequence.constraints) == 1

    def test_serialization_deserialization(self) -> None:
        """Test sequence serialization/deserialization."""
        slot = Slot(name="word")
        template = Template(
            name="t1",
            template_string="{word}",
            slots={"word": slot},
        )
        sequence = TemplateSequence(
            name="sequence",
            templates=[template],
        )
        data = sequence.model_dump()
        sequence_reloaded = TemplateSequence.model_validate(data)
        assert sequence_reloaded.name == "sequence"
        assert len(sequence_reloaded.templates) == 1

    def test_empty_name_fails(self) -> None:
        """Test that empty sequence name validation fails."""
        with pytest.raises(ValidationError) as exc_info:
            TemplateSequence(name="", templates=[])
        assert "name must be non-empty" in str(exc_info.value)


class TestTemplateTree:
    """Test template tree model."""

    def test_create_with_root_and_children(self) -> None:
        """Test creating a tree with root and children."""
        slot = Slot(name="word")
        root_template = Template(
            name="root",
            template_string="{word}",
            slots={"word": slot},
        )
        child_template = Template(
            name="child",
            template_string="{word}",
            slots={"word": slot},
        )
        child_tree = TemplateTree(
            name="child_tree",
            root=child_template,
            children=[],
        )
        tree = TemplateTree(
            name="tree",
            root=root_template,
            children=[child_tree],
        )
        assert tree.name == "tree"
        assert tree.root.name == "root"
        assert len(tree.children) == 1

    def test_serialization_deserialization(self) -> None:
        """Test tree serialization/deserialization."""
        slot = Slot(name="word")
        template = Template(
            name="t1",
            template_string="{word}",
            slots={"word": slot},
        )
        tree = TemplateTree(
            name="tree",
            root=template,
            children=[],
        )
        tree_reloaded = TemplateTree.model_validate_json(tree.model_dump_json())
        assert tree_reloaded.name == "tree"
        assert tree_reloaded.root.name == "t1"

    def test_nested_tree_structure(self) -> None:
        """Test nested tree structure (grandchildren)."""
        slot = Slot(name="word")
        grandchild_template = Template(
            name="grandchild",
            template_string="{word}",
            slots={"word": slot},
        )
        child_template = Template(
            name="child",
            template_string="{word}",
            slots={"word": slot},
        )
        root_template = Template(
            name="root",
            template_string="{word}",
            slots={"word": slot},
        )

        grandchild_tree = TemplateTree(
            name="grandchild_tree",
            root=grandchild_template,
            children=[],
        )
        child_tree = TemplateTree(
            name="child_tree",
            root=child_template,
            children=[grandchild_tree],
        )
        tree = TemplateTree(
            name="tree",
            root=root_template,
            children=[child_tree],
        )

        assert len(tree.children) == 1
        assert len(tree.children[0].children) == 1
        assert tree.children[0].children[0].name == "grandchild_tree"

    def test_empty_name_fails(self) -> None:
        """Test that empty tree name validation fails."""
        slot = Slot(name="word")
        template = Template(
            name="t1",
            template_string="{word}",
            slots={"word": slot},
        )
        with pytest.raises(ValidationError) as exc_info:
            TemplateTree(name="", root=template, children=[])
        assert "name must be non-empty" in str(exc_info.value)
