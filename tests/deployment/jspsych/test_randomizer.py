"""Tests for JavaScript randomizer code generation."""

from __future__ import annotations

from uuid import UUID, uuid4

from bead.deployment.jspsych.randomizer import (
    _extract_property_name,
    _generate_distance_constraints,
    _get_nested_property,
    _prepare_template_context,
    _serialize_metadata,
    generate_randomizer_function,
)
from bead.lists.constraints import OrderingConstraint, OrderingPair


class TestGenerateRandomizerFunction:
    """Tests for generate_randomizer_function()."""

    def test_no_constraints(self) -> None:
        """Test randomizer generation with no constraints."""
        item1 = uuid4()
        item2 = uuid4()
        item_ids = [item1, item2]
        constraints: list[OrderingConstraint] = []
        metadata = {
            item1: {"condition": "A"},
            item2: {"condition": "B"},
        }

        js_code = generate_randomizer_function(item_ids, constraints, metadata)

        assert "function randomizeTrials" in js_code
        assert "const ITEM_METADATA" in js_code
        assert str(item1) in js_code
        assert str(item2) in js_code

    def test_with_practice_items(self) -> None:
        """Test randomizer generation with practice items."""
        item1 = uuid4()
        item2 = uuid4()
        item_ids = [item1, item2]
        constraint = OrderingConstraint(
            constraint_type="ordering",
            practice_item_property="item_metadata.is_practice",
        )
        metadata = {
            item1: {"is_practice": True},
            item2: {"is_practice": False},
        }

        js_code = generate_randomizer_function(item_ids, [constraint], metadata)

        assert "practiceTrials" in js_code
        assert "mainTrials" in js_code
        assert "is_practice" in js_code

    def test_with_precedence(self) -> None:
        """Test randomizer generation with precedence constraint."""
        item1 = UUID("12345678-1234-5678-1234-567812345678")
        item2 = UUID("87654321-4321-8765-4321-876543218765")
        item_ids = [item1, item2]
        constraint = OrderingConstraint(
            constraint_type="ordering",
            precedence_pairs=(OrderingPair(before=item1, after=item2),),
        )
        metadata = {
            item1: {},
            item2: {},
        }

        js_code = generate_randomizer_function(item_ids, [constraint], metadata)

        assert "checkPrecedenceConstraints" in js_code
        assert str(item1) in js_code
        assert str(item2) in js_code

    def test_with_no_adjacent(self) -> None:
        """Test randomizer generation with no-adjacency constraint."""
        item1 = uuid4()
        item2 = uuid4()
        item_ids = [item1, item2]
        constraint = OrderingConstraint(
            constraint_type="ordering", no_adjacent_property="item_metadata.condition"
        )
        metadata = {
            item1: {"condition": "A"},
            item2: {"condition": "B"},
        }

        js_code = generate_randomizer_function(item_ids, [constraint], metadata)

        assert "checkNoAdjacentConstraints" in js_code
        assert "item_metadata.condition" in js_code

    def test_with_blocking(self) -> None:
        """Test randomizer generation with blocking constraint."""
        item1 = uuid4()
        item2 = uuid4()
        item_ids = [item1, item2]
        constraint = OrderingConstraint(
            constraint_type="ordering",
            block_by_property="item_metadata.block_type",
            randomize_within_blocks=True,
        )
        metadata = {
            item1: {"block_type": "A"},
            item2: {"block_type": "B"},
        }

        js_code = generate_randomizer_function(item_ids, [constraint], metadata)

        assert "blocks" in js_code or "block_type" in js_code

    def test_with_distance(self) -> None:
        """Test randomizer generation with distance constraint."""
        item1 = uuid4()
        item2 = uuid4()
        item3 = uuid4()
        item_ids = [item1, item2, item3]
        constraint = OrderingConstraint(
            constraint_type="ordering",
            no_adjacent_property="item_metadata.condition",
            min_distance=2,
        )
        metadata = {
            item1: {"condition": "A"},
            item2: {"condition": "A"},
            item3: {"condition": "B"},
        }

        js_code = generate_randomizer_function(item_ids, [constraint], metadata)

        assert "checkDistanceConstraints" in js_code


class TestSerializeMetadata:
    """Tests for _serialize_metadata()."""

    def test_basic_serialization(self) -> None:
        """Test metadata serialization."""
        item1 = uuid4()
        item2 = uuid4()
        metadata = {
            item1: {"condition": "A", "value": 1},
            item2: {"condition": "B", "value": 2},
        }

        json_str = _serialize_metadata(metadata)

        assert str(item1) in json_str
        assert str(item2) in json_str
        assert '"condition"' in json_str
        assert '"A"' in json_str
        assert '"B"' in json_str


class TestExtractPropertyName:
    """Tests for _extract_property_name()."""

    def test_nested_property(self) -> None:
        """Test extraction from nested property path."""
        assert _extract_property_name("item_metadata.is_practice") == "is_practice"

    def test_single_property(self) -> None:
        """Test extraction from single property."""
        assert _extract_property_name("condition") == "condition"

    def test_deeply_nested_property(self) -> None:
        """Test extraction from deeply nested path."""
        assert _extract_property_name("nested.property.path") == "path"


class TestGetNestedProperty:
    """Tests for _get_nested_property()."""

    def test_get_nested_value(self) -> None:
        """Test retrieving nested property value."""
        obj = {"item_metadata": {"condition": "A", "is_practice": True}}

        assert _get_nested_property(obj, "item_metadata.condition") == "A"
        assert _get_nested_property(obj, "item_metadata.is_practice") is True

    def test_missing_property(self) -> None:
        """Test retrieving missing property returns None."""
        obj = {"item_metadata": {"condition": "A"}}

        assert _get_nested_property(obj, "missing.path") is None
        assert _get_nested_property(obj, "item_metadata.missing") is None


class TestGenerateDistanceConstraints:
    """Tests for _generate_distance_constraints()."""

    def test_distance_constraints_generation(self) -> None:
        """Test distance constraint generation."""
        item1 = uuid4()
        item2 = uuid4()
        item3 = uuid4()
        item_ids = [item1, item2, item3]

        constraint = OrderingConstraint(
            constraint_type="ordering",
            no_adjacent_property="item_metadata.condition",
            min_distance=2,
        )

        metadata = {
            item1: {"condition": "A"},
            item2: {"condition": "A"},
            item3: {"condition": "B"},
        }

        distance_constraints = _generate_distance_constraints(
            item_ids, constraint, metadata
        )

        # Should have one constraint between item1 and item2 (both condition A)
        assert len(distance_constraints) == 1
        assert distance_constraints[0]["min_distance"] == 2
        assert distance_constraints[0]["max_distance"] is None


class TestPrepareTemplateContext:
    """Tests for _prepare_template_context()."""

    def test_practice_items(self) -> None:
        """Test template context preparation with practice items."""
        item1 = uuid4()
        item2 = uuid4()
        constraint = OrderingConstraint(
            constraint_type="ordering",
            practice_item_property="item_metadata.is_practice",
        )
        metadata = {
            item1: {"is_practice": True},
            item2: {"is_practice": False},
        }

        context = _prepare_template_context([item1, item2], [constraint], metadata)

        assert context["has_practice_items"] is True
        assert context["practice_property"] == "is_practice"

    def test_blocking(self) -> None:
        """Test template context preparation with blocking."""
        item1 = uuid4()
        item2 = uuid4()
        constraint = OrderingConstraint(
            constraint_type="ordering",
            block_by_property="item_metadata.block_type",
            randomize_within_blocks=False,
        )
        metadata = {
            item1: {"block_type": "A"},
            item2: {"block_type": "B"},
        }

        context = _prepare_template_context([item1, item2], [constraint], metadata)

        assert context["has_blocking"] is True
        assert context["block_property"] == "block_type"
        assert context["randomize_within_blocks"] is False

    def test_precedence(self) -> None:
        """Test template context preparation with precedence."""
        item1 = uuid4()
        item2 = uuid4()
        constraint = OrderingConstraint(
            constraint_type="ordering",
            precedence_pairs=(OrderingPair(before=item1, after=item2),),
        )
        metadata = {item1: {}, item2: {}}

        context = _prepare_template_context([item1, item2], [constraint], metadata)

        assert context["has_precedence"] is True
        assert str(item1) in context["precedence_pairs_json"]
        assert str(item2) in context["precedence_pairs_json"]

    def test_no_adjacent(self) -> None:
        """Test template context preparation with no-adjacency."""
        item1 = uuid4()
        item2 = uuid4()
        constraint = OrderingConstraint(
            constraint_type="ordering", no_adjacent_property="item_metadata.condition"
        )
        metadata = {
            item1: {"condition": "A"},
            item2: {"condition": "B"},
        }

        context = _prepare_template_context([item1, item2], [constraint], metadata)

        assert context["has_no_adjacent"] is True
        # Property name is extracted from path since metadata is already extracted
        assert context["no_adjacent_property"] == "condition"

    def test_multiple_constraints(self) -> None:
        """Test template context preparation with multiple constraints."""
        item1 = uuid4()
        item2 = uuid4()
        item3 = uuid4()

        constraint1 = OrderingConstraint(
            constraint_type="ordering",
            practice_item_property="item_metadata.is_practice",
        )
        constraint2 = OrderingConstraint(
            constraint_type="ordering", no_adjacent_property="item_metadata.condition"
        )

        metadata = {
            item1: {"is_practice": True, "condition": "A"},
            item2: {"is_practice": False, "condition": "B"},
            item3: {"is_practice": False, "condition": "A"},
        }

        context = _prepare_template_context(
            [item1, item2, item3], [constraint1, constraint2], metadata
        )

        assert context["has_practice_items"] is True
        assert context["has_no_adjacent"] is True
