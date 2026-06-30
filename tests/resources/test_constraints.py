"""Tests for constraint models."""

from __future__ import annotations

import didactic.api as dx
import pytest

from bead.resources.constraints import Constraint


class TestConstraint:
    """Test unified Constraint class."""

    def test_create_basic_constraint(self) -> None:
        """Test creating a basic constraint."""
        constraint = Constraint(expression="self.pos == 'VERB'")
        assert constraint.expression == "self.pos == 'VERB'"
        assert constraint.context == {}
        assert constraint.description is None

    def test_create_with_context(self) -> None:
        """Test creating constraint with context."""
        context = {"allowed_verbs": ("break", "shatter")}
        constraint = Constraint(
            expression="self.lemma in allowed_verbs", context=context
        )
        assert constraint.context == context

    def test_create_with_description(self) -> None:
        """Test creating constraint with description."""
        constraint = Constraint(
            expression="self.pos == 'VERB'", description="Verb constraint"
        )
        assert constraint.description == "Verb constraint"

    def test_serialization(self) -> None:
        """Test constraint serialization."""
        constraint = Constraint(
            expression="self.pos == 'VERB'",
            context={"key": "value"},
            description="Test",
        )
        data = constraint.model_dump()
        assert data["expression"] == "self.pos == 'VERB'"
        assert data["context"] == {"key": "value"}
        assert data["description"] == "Test"

    def test_deserialization(self) -> None:
        """Test constraint deserialization."""
        data = {
            "expression": "self.pos == 'NOUN'",
            "context": {"test": "value"},
        }
        constraint = Constraint.model_validate(data)
        assert constraint.expression == "self.pos == 'NOUN'"
        assert constraint.context == {"test": "value"}

    def test_empty_expression_allowed(self) -> None:
        """Test that empty expression is allowed."""
        constraint = Constraint(expression="")
        assert constraint.expression == ""

    def test_context_serialization(self) -> None:
        """Test context with various types serializes correctly."""
        constraint = Constraint(
            expression="test",
            context={
                "str_val": "test",
                "int_val": 42,
                "float_val": 3.14,
                "bool_val": True,
                "list_val": ["a", "b"],
                "set_val": ("x", "y"),
            },
        )
        data = constraint.model_dump()
        assert data["context"]["str_val"] == "test"
        assert data["context"]["int_val"] == 42
        assert data["context"]["bool_val"] is True
        # Sets may be serialized as lists
        assert "x" in data["context"]["set_val"]


class TestConstraintCombine:
    """Test Constraint.combine() class method."""

    def test_combine_two_constraints_with_and(self) -> None:
        """Test combining two constraints with AND logic."""
        c1 = Constraint(expression="self.pos == 'VERB'")
        c2 = Constraint(expression="self.lemma == 'break'")

        combined = Constraint.combine(c1, c2, logic="and")

        assert "(self.pos == 'VERB') and (self.lemma == 'break')" == combined.expression

    def test_combine_two_constraints_with_or(self) -> None:
        """Test combining two constraints with OR logic."""
        c1 = Constraint(expression="self.pos == 'VERB'")
        c2 = Constraint(expression="self.pos == 'NOUN'")

        combined = Constraint.combine(c1, c2, logic="or")

        assert "(self.pos == 'VERB') or (self.pos == 'NOUN')" == combined.expression

    def test_combine_three_constraints(self) -> None:
        """Test combining three constraints."""
        c1 = Constraint(expression="a")
        c2 = Constraint(expression="b")
        c3 = Constraint(expression="c")

        combined = Constraint.combine(c1, c2, c3, logic="and")

        assert combined.expression == "(a) and (b) and (c)"

    def test_combine_merges_context(self) -> None:
        """Test that combine merges context from all constraints."""
        c1 = Constraint(
            expression="self.pos in allowed_pos", context={"allowed_pos": ("VERB",)}
        )
        c2 = Constraint(
            expression="self.lemma in allowed_verbs",
            context={"allowed_verbs": ("break",)},
        )

        combined = Constraint.combine(c1, c2, logic="and")

        assert combined.context["allowed_pos"] == ("VERB",)
        assert combined.context["allowed_verbs"] == ("break",)

    def test_combine_context_key_collision_last_wins(self) -> None:
        """Test that conflicting context keys use last value (dict.update behavior)."""
        c1 = Constraint(expression="expr1", context={"key": "value1", "other": "a"})
        c2 = Constraint(expression="expr2", context={"key": "value2"})

        combined = Constraint.combine(c1, c2, logic="and")

        # Last constraint's value wins for "key"
        assert combined.context["key"] == "value2"
        # Non-conflicting keys are preserved
        assert combined.context["other"] == "a"

    def test_combine_concatenates_descriptions(self) -> None:
        """Test that descriptions are concatenated."""
        c1 = Constraint(expression="a", description="First constraint")
        c2 = Constraint(expression="b", description="Second constraint")

        combined = Constraint.combine(c1, c2, logic="and")

        assert combined.description == "First constraint; Second constraint"

    def test_combine_with_none_descriptions(self) -> None:
        """Test combining constraints with None descriptions."""
        c1 = Constraint(expression="a", description="Only one")
        c2 = Constraint(expression="b", description=None)

        combined = Constraint.combine(c1, c2, logic="and")

        # Should only include non-None descriptions
        assert combined.description == "Only one"

    def test_combine_invalid_logic_raises_error(self) -> None:
        """Test that invalid logic parameter raises ValueError."""
        c1 = Constraint(expression="a")
        c2 = Constraint(expression="b")

        with pytest.raises(
            (ValueError, dx.ValidationError), match="Invalid logic operator"
        ):
            Constraint.combine(c1, c2, logic="xor")

    def test_combine_single_constraint(self) -> None:
        """Test combining a single constraint."""
        c1 = Constraint(expression="self.pos == 'VERB'", description="Test")

        combined = Constraint.combine(c1, logic="and")

        # Single constraint is returned unchanged
        assert combined is c1
        assert combined.expression == "self.pos == 'VERB'"
        assert combined.description == "Test"

    def test_combine_empty_raises_error(self) -> None:
        """Test that combining zero constraints raises error."""
        with pytest.raises(
            (ValueError, dx.ValidationError),
            match="Must provide at least one constraint",
        ):
            Constraint.combine(logic="and")

    def test_combine_preserves_parentheses(self) -> None:
        """Test that expressions are wrapped in parentheses."""
        c1 = Constraint(expression="a or b")
        c2 = Constraint(expression="c and d")

        combined = Constraint.combine(c1, c2, logic="and")

        # Should wrap each expression in parens to preserve precedence
        assert combined.expression == "(a or b) and (c and d)"
