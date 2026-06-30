"""Tests for template generation utilities."""

from __future__ import annotations

from typing import Any

import didactic.api as dx
import pytest

from bead.resources.constraints import Constraint
from bead.resources.template import Slot, Template
from bead.resources.template_generation import FrameToTemplateMapper, MultiFrameMapper


class TestFrameToTemplateMapper:
    """Test FrameToTemplateMapper abstract base class."""

    def test_is_abstract(self) -> None:
        """Test that FrameToTemplateMapper is abstract."""
        with pytest.raises(TypeError):
            FrameToTemplateMapper()  # type: ignore[abstract]

    def test_subclass_must_implement_all_methods(self) -> None:
        """Test that subclasses must implement all abstract methods."""

        class IncompleteMapper(FrameToTemplateMapper):
            def generate_from_frame(self, *args, **kwargs):
                return Template(
                    name="test", template_string="{slot}", slots={"slot": Slot()}
                )

        with pytest.raises(TypeError):
            IncompleteMapper()  # type: ignore[abstract]

    def test_create_template_name(self) -> None:
        """Test create_template_name() utility method."""

        class ConcreteMapper(FrameToTemplateMapper):
            def generate_from_frame(self, *args, **kwargs):
                pass

            def map_frame_to_slots(self, frame_data):
                return {}

            def generate_constraints(self, frame_data, slots):
                return []

        mapper = ConcreteMapper()
        name = mapper.create_template_name("think", "29.9", "that-clause")

        assert name == "think_29_9_that_clause"

    def test_create_template_name_with_spaces(self) -> None:
        """Test create_template_name() sanitizes spaces."""

        class ConcreteMapper(FrameToTemplateMapper):
            def generate_from_frame(self, *args, **kwargs):
                pass

            def map_frame_to_slots(self, frame_data):
                return {}

            def generate_constraints(self, frame_data, slots):
                return []

        mapper = ConcreteMapper()
        name = mapper.create_template_name("verb class", "frame 1")

        assert name == "verb_class_frame_1"
        assert " " not in name

    def test_create_template_name_custom_separator(self) -> None:
        """Test create_template_name() with custom separator."""

        class ConcreteMapper(FrameToTemplateMapper):
            def generate_from_frame(self, *args, **kwargs):
                pass

            def map_frame_to_slots(self, frame_data):
                return {}

            def generate_constraints(self, frame_data, slots):
                return []

        mapper = ConcreteMapper()
        name = mapper.create_template_name("think", "29.9", separator="-")

        assert name == "think-29-9"

    def test_create_template_metadata(self) -> None:
        """Test create_template_metadata() utility method."""

        class ConcreteMapper(FrameToTemplateMapper):
            def generate_from_frame(self, *args, **kwargs):
                pass

            def map_frame_to_slots(self, frame_data):
                return {}

            def generate_constraints(self, frame_data, slots):
                return []

        mapper = ConcreteMapper()
        frame_data = {"id": "29.9-1", "source": "VerbNet"}
        metadata = mapper.create_template_metadata(
            frame_data, verb_lemma="think", language="eng"
        )

        assert metadata["id"] == "29.9-1"
        assert metadata["source"] == "VerbNet"
        assert metadata["verb_lemma"] == "think"
        assert metadata["language"] == "eng"

    def test_create_template_metadata_empty_frame(self) -> None:
        """Test create_template_metadata() with empty frame data."""

        class ConcreteMapper(FrameToTemplateMapper):
            def generate_from_frame(self, *args, **kwargs):
                pass

            def map_frame_to_slots(self, frame_data):
                return {}

            def generate_constraints(self, frame_data, slots):
                return []

        mapper = ConcreteMapper()
        frame_data: dict[str, Any] = {}
        metadata = mapper.create_template_metadata(frame_data, extra_key="value")

        # Should just contain additional metadata
        assert metadata["extra_key"] == "value"
        assert len(metadata) == 1


class TestMultiFrameMapper:
    """Test MultiFrameMapper."""

    def test_is_abstract(self) -> None:
        """Test that MultiFrameMapper is abstract."""
        with pytest.raises(TypeError):
            MultiFrameMapper()  # type: ignore[abstract]

    def test_subclass_must_implement_get_frame_variants(self) -> None:
        """Test that subclasses must implement get_frame_variants()."""

        class IncompleteMultiMapper(MultiFrameMapper):
            def map_frame_to_slots(self, frame_data):
                return {}

            def generate_constraints(self, frame_data, slots):
                return []

        with pytest.raises(TypeError):
            IncompleteMultiMapper()  # type: ignore[abstract]

    def test_generate_from_frame_creates_templates_for_all_variants(self) -> None:
        """Test that generate_from_frame() creates templates for all variants."""

        class TestMultiMapper(MultiFrameMapper):
            def get_frame_variants(self, frame_data):
                return [
                    {"comp": "that", "mood": "declarative"},
                    {"comp": "whether", "mood": "interrogative"},
                    {"comp": "if", "mood": "interrogative"},
                ]

            def map_frame_to_slots(self, frame_data):
                return {
                    "subject": Slot(name="subject"),
                    "verb": Slot(name="verb"),
                    "complement": Slot(name="complement"),
                }

            def generate_constraints(self, frame_data, slots):
                return []

            def _generate_variant(self, *args, **kwargs):
                variant_data = kwargs["variant_data"]
                comp = variant_data["comp"]
                return Template(
                    name=f"test_{comp}",
                    template_string=f"{{subject}} {{verb}} {comp} {{complement}}",
                    slots=self.map_frame_to_slots(kwargs["frame_data"]),
                )

        mapper = TestMultiMapper()
        frame_data = {"base": "NP V COMP S"}
        templates = mapper.generate_from_frame(frame_data=frame_data)

        assert len(templates) == 3
        assert templates[0].name == "test_that"
        assert templates[1].name == "test_whether"
        assert templates[2].name == "test_if"

    def test_generate_from_frame_requires_frame_data(self) -> None:
        """Test that generate_from_frame() requires frame_data in kwargs."""

        class TestMultiMapper(MultiFrameMapper):
            def get_frame_variants(self, frame_data):
                return []

            def map_frame_to_slots(self, frame_data):
                return {}

            def generate_constraints(self, frame_data, slots):
                return []

            def _generate_variant(self, *args, **kwargs):
                return Template(name="test", template_string="{slot}", slots={})

        mapper = TestMultiMapper()

        with pytest.raises(
            (ValueError, dx.ValidationError), match="frame_data must be provided"
        ):
            mapper.generate_from_frame()

    def test_variant_data_passed_to_generate_variant(self) -> None:
        """Test that variant_data is passed to _generate_variant()."""

        class TestMultiMapper(MultiFrameMapper):
            def get_frame_variants(self, frame_data):
                return [{"comp": "that"}, {"comp": "whether"}]

            def map_frame_to_slots(self, frame_data):
                return {"slot": Slot(name="slot")}

            def generate_constraints(self, frame_data, slots):
                return []

            def _generate_variant(self, *args, **kwargs):
                # Verify variant_data is present
                variant_data = kwargs["variant_data"]
                assert "comp" in variant_data
                return Template(
                    name=f"test_{variant_data['comp']}",
                    template_string=f"{{slot}} {variant_data['comp']}",
                    slots=self.map_frame_to_slots(kwargs["frame_data"]),
                )

        mapper = TestMultiMapper()
        frame_data = {"base": "test"}
        templates = mapper.generate_from_frame(frame_data=frame_data)

        assert len(templates) == 2


class TestConcreteMapperImplementation:
    """Test a complete concrete implementation."""

    def test_complete_mapper_implementation(self) -> None:
        """Test a complete FrameToTemplateMapper implementation."""

        class VerbFrameMapper(FrameToTemplateMapper):
            def generate_from_frame(self, verb_lemma: str, frame_data: dict):
                slots = self.map_frame_to_slots(frame_data)
                constraints = self.generate_constraints(frame_data, slots)

                name = self.create_template_name(verb_lemma, frame_data["frame_id"])
                metadata = self.create_template_metadata(frame_data, verb=verb_lemma)

                return Template(
                    name=name,
                    template_string=frame_data["template_string"],
                    slots=slots,
                    constraints=constraints,
                    metadata=metadata,
                )

            def map_frame_to_slots(self, frame_data):
                return {
                    "subject": Slot(name="subject"),
                    "verb": Slot(name="verb"),
                    "object": Slot(name="object"),
                }

            def generate_constraints(self, frame_data, slots):
                # Simple example: subject-verb agreement
                constraint = Constraint(
                    expression=(
                        "subject.features.get('number') == verb.features.get('number')"
                    ),
                    description="Subject-verb number agreement",
                )
                return [constraint]

        mapper = VerbFrameMapper()
        frame_data = {
            "frame_id": "transitive",
            "template_string": "{subject} {verb} {object}",
        }

        template = mapper.generate_from_frame(verb_lemma="break", frame_data=frame_data)

        assert isinstance(template, Template)
        assert template.name == "break_transitive"
        assert len(template.slots) == 3
        assert len(template.constraints) == 1
        assert template.metadata["frame_id"] == "transitive"
        assert template.metadata["verb"] == "break"
