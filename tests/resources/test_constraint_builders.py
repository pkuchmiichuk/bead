"""Tests for constraint builder utilities."""

from __future__ import annotations

import didactic.api as dx
import pytest

from bead.resources.constraint_builders import (
    AgreementConstraintBuilder,
    ConditionalConstraintBuilder,
    ConstraintBuilder,
    SetMembershipConstraintBuilder,
)
from bead.resources.constraints import Constraint


class TestConstraintBuilder:
    """Test abstract ConstraintBuilder base class."""

    def test_is_abstract(self) -> None:
        """Test that ConstraintBuilder is abstract."""
        with pytest.raises(TypeError):
            ConstraintBuilder()  # type: ignore[abstract]

    def test_subclass_must_implement_build(self) -> None:
        """Test that subclasses must implement build()."""

        class IncompleteBuilder(ConstraintBuilder):
            pass

        with pytest.raises(TypeError):
            IncompleteBuilder()  # type: ignore[abstract]


class TestAgreementConstraintBuilder:
    """Test AgreementConstraintBuilder."""

    def test_exact_match_two_slots(self) -> None:
        """Test exact match agreement between two slots."""
        builder = AgreementConstraintBuilder("number")
        constraint = builder.build("subject", "verb")

        assert isinstance(constraint, Constraint)
        assert "subject.features.get('number')" in constraint.expression
        assert "verb.features.get('number')" in constraint.expression
        assert " == " in constraint.expression
        assert constraint.description is not None
        assert "number agreement" in constraint.description.lower()

    def test_exact_match_multiple_slots(self) -> None:
        """Test exact match agreement across multiple slots."""
        builder = AgreementConstraintBuilder("gender")
        constraint = builder.build("adjective", "noun", "pronoun")

        # Should have pairwise equality checks
        assert "adjective.features.get('gender')" in constraint.expression
        assert "noun.features.get('gender')" in constraint.expression
        assert "pronoun.features.get('gender')" in constraint.expression
        assert constraint.expression.count(" and ") >= 1  # Multiple conditions

    def test_with_agreement_rules(self) -> None:
        """Test agreement with equivalence rules."""
        rules = {
            "singular": ["singular", "sing", "sg"],
            "plural": ["plural", "pl"],
        }

        builder = AgreementConstraintBuilder("number", agreement_rules=rules)
        constraint = builder.build("det", "noun")

        assert isinstance(constraint, Constraint)
        # Should have equivalence class checks
        assert "equiv_" in constraint.expression
        assert " or " in constraint.expression  # Multiple equivalence classes
        assert constraint.context is not None
        assert "equiv_singular" in constraint.context
        assert "equiv_plural" in constraint.context

    def test_requires_two_or_more_slots(self) -> None:
        """Test that agreement requires at least 2 slots."""
        builder = AgreementConstraintBuilder("case")

        with pytest.raises(
            (ValueError, dx.ValidationError), match="at least 2 slot names"
        ):
            builder.build("noun")

    def test_feature_name_preserved(self) -> None:
        """Test that feature name is correctly stored."""
        builder = AgreementConstraintBuilder("custom_feature")

        assert builder.feature_name == "custom_feature"

    def test_rules_stored(self) -> None:
        """Test that agreement rules are stored."""
        rules = {"val1": ["a", "b"], "val2": ["c", "d"]}
        builder = AgreementConstraintBuilder("feat", agreement_rules=rules)

        assert builder.agreement_rules == rules


class TestConditionalConstraintBuilder:
    """Test ConditionalConstraintBuilder."""

    def test_basic_conditional(self) -> None:
        """Test basic IF-THEN constraint."""
        builder = ConditionalConstraintBuilder()
        constraint = builder.build(
            condition="det.lemma == 'a'",
            requirement="noun.features.get('number') == 'singular'",
            description="'a' requires singular",
        )

        assert isinstance(constraint, Constraint)
        # IF-THEN encoded as: not condition or requirement
        assert "not (" in constraint.expression
        assert ") or (" in constraint.expression
        assert constraint.description == "'a' requires singular"

    def test_with_context(self) -> None:
        """Test conditional with context variables."""
        builder = ConditionalConstraintBuilder()
        context = {"allowed_nouns": ("cat", "dog")}

        constraint = builder.build(
            condition="det.lemma == 'a'",
            requirement="noun.lemma in allowed_nouns",
            context=context,
        )

        assert dict(constraint.context) == context

    def test_without_description(self) -> None:
        """Test conditional without description."""
        builder = ConditionalConstraintBuilder()
        constraint = builder.build(
            condition="x > 0",
            requirement="y > 0",
        )

        assert constraint.description is None


class TestSetMembershipConstraintBuilder:
    """Test SetMembershipConstraintBuilder."""

    def test_whitelist(self) -> None:
        """Test whitelist (allowed values) constraint."""
        builder = SetMembershipConstraintBuilder()
        allowed = {"walk", "run", "jump"}

        constraint = builder.build(
            slot_name="verb",
            property_path="lemma",
            allowed_values=allowed,
            description="Motion verbs",
        )

        assert isinstance(constraint, Constraint)
        assert "verb.lemma in allowed_values" in constraint.expression
        assert set(constraint.context["allowed_values"]) == allowed
        assert constraint.description == "Motion verbs"

    def test_blacklist(self) -> None:
        """Test blacklist (forbidden values) constraint."""
        builder = SetMembershipConstraintBuilder()
        forbidden = {"be", "have", "do"}

        constraint = builder.build(
            slot_name="verb",
            property_path="lemma",
            forbidden_values=forbidden,
        )

        assert "verb.lemma not in forbidden_values" in constraint.expression
        assert set(constraint.context["forbidden_values"]) == forbidden

    def test_nested_property_path(self) -> None:
        """Test constraint with nested property path."""
        builder = SetMembershipConstraintBuilder()
        allowed = {"singular", "plural"}

        constraint = builder.build(
            slot_name="noun",
            property_path="features.number",
            allowed_values=allowed,
        )

        assert "noun.features.number in allowed_values" in constraint.expression

    def test_requires_exactly_one_of_allowed_or_forbidden(self) -> None:
        """Test that exactly one of allowed/forbidden is required."""
        builder = SetMembershipConstraintBuilder()

        # Neither provided
        with pytest.raises((ValueError, dx.ValidationError), match="Exactly one of"):
            builder.build(
                slot_name="verb",
                property_path="lemma",
            )

        # Both provided
        with pytest.raises((ValueError, dx.ValidationError), match="Exactly one of"):
            builder.build(
                slot_name="verb",
                property_path="lemma",
                allowed_values={"a"},
                forbidden_values={"b"},
            )

    def test_default_description_for_whitelist(self) -> None:
        """Test default description for whitelist."""
        builder = SetMembershipConstraintBuilder()
        constraint = builder.build(
            slot_name="verb",
            property_path="lemma",
            allowed_values={"walk"},
        )

        assert "Restrict verb.lemma to allowed values" in constraint.description

    def test_default_description_for_blacklist(self) -> None:
        """Test default description for blacklist."""
        builder = SetMembershipConstraintBuilder()
        constraint = builder.build(
            slot_name="verb",
            property_path="lemma",
            forbidden_values={"be"},
        )

        assert "Exclude verb.lemma from forbidden values" in constraint.description
