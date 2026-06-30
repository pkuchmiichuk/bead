"""Tests for list constraint models."""

from __future__ import annotations

from uuid import uuid4

import pytest
from didactic.api import ValidationError

from bead.lists.constraints import (
    BalanceConstraint,
    OrderingConstraint,
    OrderingPair,
    QuantileConstraint,
    SizeConstraint,
    UniquenessConstraint,
)


class TestUniquenessConstraint:
    """Tests for UniquenessConstraint model."""

    def test_create_basic(self) -> None:
        """Test creating basic uniqueness constraint."""
        constraint = UniquenessConstraint(
            constraint_type="uniqueness",
            property_expression="item_metadata.target_verb",
        )

        assert constraint.constraint_type == "uniqueness"
        assert constraint.property_expression == "item_metadata.target_verb"
        assert constraint.allow_null is False

    def test_create_with_allow_null(self) -> None:
        """Test creating with allow_null=True."""
        constraint = UniquenessConstraint(
            constraint_type="uniqueness",
            property_expression="item_metadata.target_verb",
            allow_null=True,
        )

        assert constraint.allow_null is True

    def test_property_path_validation_empty(self) -> None:
        """Test empty property_path raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            UniquenessConstraint(constraint_type="uniqueness", property_expression="")
        assert "must be non-empty" in str(exc_info.value)

    def test_property_path_validation_whitespace(self) -> None:
        """Test whitespace-only property_path raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            UniquenessConstraint(
                constraint_type="uniqueness", property_expression="   "
            )
        assert "must be non-empty" in str(exc_info.value)

    def test_property_path_strips_whitespace(self) -> None:
        """Test property_path whitespace is stripped."""
        constraint = UniquenessConstraint(
            constraint_type="uniqueness", property_expression="  test.path  "
        )
        assert constraint.property_expression == "test.path"

    def test_constraint_type_is_uniqueness(self) -> None:
        """Test discriminator is correct."""
        constraint = UniquenessConstraint(
            constraint_type="uniqueness", property_expression="test"
        )
        assert constraint.constraint_type == "uniqueness"

    def test_serialization_roundtrip(self) -> None:
        """Test serialization roundtrip works."""
        constraint = UniquenessConstraint(
            constraint_type="uniqueness",
            property_expression="item_metadata.target_verb",
            allow_null=True,
        )

        data = constraint.model_dump()
        restored = UniquenessConstraint(**data)

        assert restored.property_expression == constraint.property_expression
        assert restored.allow_null == constraint.allow_null

    def test_inherits_beadbasemodel(self) -> None:
        """Test has BeadBaseModel fields."""
        constraint = UniquenessConstraint(
            constraint_type="uniqueness", property_expression="test"
        )

        assert hasattr(constraint, "id")
        assert hasattr(constraint, "created_at")
        assert hasattr(constraint, "modified_at")


class TestBalanceConstraint:
    """Tests for BalanceConstraint model."""

    def test_create_basic(self) -> None:
        """Test creating basic balance constraint."""
        constraint = BalanceConstraint(
            constraint_type="balance", property_expression="item_metadata.transitivity"
        )

        assert constraint.constraint_type == "balance"
        assert constraint.property_expression == "item_metadata.transitivity"
        assert constraint.target_counts is None
        assert constraint.tolerance == 0.1

    def test_create_with_target_counts(self) -> None:
        """Test creating with target_counts."""
        constraint = BalanceConstraint(
            constraint_type="balance",
            property_expression="item_metadata.grammatical",
            target_counts={"true": 20, "false": 10},
        )

        assert constraint.target_counts == {"true": 20, "false": 10}

    def test_create_with_tolerance(self) -> None:
        """Test creating with custom tolerance."""
        constraint = BalanceConstraint(
            constraint_type="balance",
            property_expression="item_metadata.transitivity",
            tolerance=0.05,
        )

        assert constraint.tolerance == 0.05

    def test_property_path_validation_empty(self) -> None:
        """Test empty property_path raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            BalanceConstraint(constraint_type="balance", property_expression="")
        assert "must be non-empty" in str(exc_info.value)

    def test_property_path_validation_whitespace(self) -> None:
        """Test whitespace-only property_path raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            BalanceConstraint(constraint_type="balance", property_expression="   ")
        assert "must be non-empty" in str(exc_info.value)

    def test_tolerance_validation_negative(self) -> None:
        """Test negative tolerance raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            BalanceConstraint(
                constraint_type="balance", property_expression="test", tolerance=-0.1
            )
        assert "non-negative" in str(exc_info.value) or "must be" in str(exc_info.value)

    def test_tolerance_validation_too_large(self) -> None:
        """Test tolerance > 1.0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            BalanceConstraint(
                constraint_type="balance", property_expression="test", tolerance=1.5
            )
        assert "between 0 and 1" in str(exc_info.value)

    def test_tolerance_validation_zero(self) -> None:
        """Test tolerance=0.0 is valid."""
        constraint = BalanceConstraint(
            constraint_type="balance", property_expression="test", tolerance=0.0
        )
        assert constraint.tolerance == 0.0

    def test_tolerance_validation_one(self) -> None:
        """Test tolerance=1.0 is valid."""
        constraint = BalanceConstraint(
            constraint_type="balance", property_expression="test", tolerance=1.0
        )
        assert constraint.tolerance == 1.0

    def test_target_counts_validation_negative_values(self) -> None:
        """Test negative target_counts values raise ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            BalanceConstraint(
                constraint_type="balance",
                property_expression="test",
                target_counts={"cat1": 10, "cat2": -5},
            )
        assert "non-negative" in str(exc_info.value)

    def test_target_counts_validation_zero_values(self) -> None:
        """Test zero target_counts values are valid."""
        constraint = BalanceConstraint(
            constraint_type="balance",
            property_expression="test",
            target_counts={"cat1": 0, "cat2": 10},
        )
        assert constraint.target_counts["cat1"] == 0

    def test_constraint_type_is_balance(self) -> None:
        """Test discriminator is correct."""
        constraint = BalanceConstraint(
            constraint_type="balance", property_expression="test"
        )
        assert constraint.constraint_type == "balance"

    def test_serialization_roundtrip(self) -> None:
        """Test serialization roundtrip works."""
        constraint = BalanceConstraint(
            constraint_type="balance",
            property_expression="test",
            target_counts={"a": 5, "b": 10},
            tolerance=0.15,
        )

        data = constraint.model_dump()
        restored = BalanceConstraint(**data)

        assert restored.property_expression == constraint.property_expression
        assert restored.target_counts == constraint.target_counts
        assert restored.tolerance == constraint.tolerance


class TestQuantileConstraint:
    """Tests for QuantileConstraint model."""

    def test_create_basic(self) -> None:
        """Test creating basic quantile constraint."""
        constraint = QuantileConstraint(
            constraint_type="quantile", property_expression="item_metadata.lm_prob"
        )

        assert constraint.constraint_type == "quantile"
        assert constraint.property_expression == "item_metadata.lm_prob"
        assert constraint.n_quantiles == 5
        assert constraint.items_per_quantile == 2

    def test_create_with_custom_quantiles(self) -> None:
        """Test creating with custom n_quantiles."""
        constraint = QuantileConstraint(
            constraint_type="quantile",
            property_expression="item_metadata.frequency",
            n_quantiles=10,
        )

        assert constraint.n_quantiles == 10

    def test_create_with_custom_items_per_quantile(self) -> None:
        """Test creating with custom items_per_quantile."""
        constraint = QuantileConstraint(
            constraint_type="quantile",
            property_expression="item_metadata.frequency",
            items_per_quantile=5,
        )

        assert constraint.items_per_quantile == 5

    def test_property_path_validation_empty(self) -> None:
        """Test empty property_path raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            QuantileConstraint(constraint_type="quantile", property_expression="")
        assert "must be non-empty" in str(exc_info.value)

    def test_property_path_validation_whitespace(self) -> None:
        """Test whitespace-only property_path raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            QuantileConstraint(constraint_type="quantile", property_expression="   ")
        assert "must be non-empty" in str(exc_info.value)

    def test_n_quantiles_validation_too_small(self) -> None:
        """Test n_quantiles < 2 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            QuantileConstraint(
                constraint_type="quantile", property_expression="test", n_quantiles=0
            )
        assert "must be" in str(exc_info.value)

    def test_n_quantiles_validation_one(self) -> None:
        """Test n_quantiles = 1 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            QuantileConstraint(
                constraint_type="quantile", property_expression="test", n_quantiles=1
            )
        assert "must be" in str(exc_info.value)

    def test_n_quantiles_validation_two(self) -> None:
        """Test n_quantiles = 2 is valid."""
        constraint = QuantileConstraint(
            constraint_type="quantile", property_expression="test", n_quantiles=2
        )
        assert constraint.n_quantiles == 2

    def test_items_per_quantile_validation_zero(self) -> None:
        """Test items_per_quantile = 0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            QuantileConstraint(
                constraint_type="quantile",
                property_expression="test",
                items_per_quantile=0,
            )
        assert "must be" in str(exc_info.value)

    def test_items_per_quantile_validation_negative(self) -> None:
        """Test items_per_quantile < 0 raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            QuantileConstraint(
                constraint_type="quantile",
                property_expression="test",
                items_per_quantile=-1,
            )
        assert "must be" in str(exc_info.value)

    def test_items_per_quantile_validation_one(self) -> None:
        """Test items_per_quantile = 1 is valid."""
        constraint = QuantileConstraint(
            constraint_type="quantile", property_expression="test", items_per_quantile=1
        )
        assert constraint.items_per_quantile == 1

    def test_constraint_type_is_quantile(self) -> None:
        """Test discriminator is correct."""
        constraint = QuantileConstraint(
            constraint_type="quantile", property_expression="test"
        )
        assert constraint.constraint_type == "quantile"

    def test_serialization_roundtrip(self) -> None:
        """Test serialization roundtrip works."""
        constraint = QuantileConstraint(
            constraint_type="quantile",
            property_expression="test",
            n_quantiles=10,
            items_per_quantile=3,
        )

        data = constraint.model_dump()
        restored = QuantileConstraint(**data)

        assert restored.property_expression == constraint.property_expression
        assert restored.n_quantiles == constraint.n_quantiles
        assert restored.items_per_quantile == constraint.items_per_quantile


class TestSizeConstraint:
    """Tests for SizeConstraint model."""

    def test_create_exact_size(self) -> None:
        """Test creating with exact_size."""
        constraint = SizeConstraint(constraint_type="size", exact_size=40)

        assert constraint.constraint_type == "size"
        assert constraint.exact_size == 40
        assert constraint.min_size is None
        assert constraint.max_size is None

    def test_create_min_size_only(self) -> None:
        """Test creating with min_size only."""
        constraint = SizeConstraint(constraint_type="size", min_size=30)

        assert constraint.min_size == 30
        assert constraint.max_size is None
        assert constraint.exact_size is None

    def test_create_max_size_only(self) -> None:
        """Test creating with max_size only."""
        constraint = SizeConstraint(constraint_type="size", max_size=50)

        assert constraint.max_size == 50
        assert constraint.min_size is None
        assert constraint.exact_size is None

    def test_create_min_max_range(self) -> None:
        """Test creating with both min and max."""
        constraint = SizeConstraint(constraint_type="size", min_size=30, max_size=50)

        assert constraint.min_size == 30
        assert constraint.max_size == 50
        assert constraint.exact_size is None

    def test_validation_no_params_set(self) -> None:
        """Test no params set raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SizeConstraint(constraint_type="size")
        assert "Must specify at least one" in str(exc_info.value)

    def test_validation_exact_with_min(self) -> None:
        """Test exact_size with min_size raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SizeConstraint(constraint_type="size", exact_size=40, min_size=30)
        assert "exact_size cannot be used with" in str(exc_info.value)

    def test_validation_exact_with_max(self) -> None:
        """Test exact_size with max_size raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SizeConstraint(constraint_type="size", exact_size=40, max_size=50)
        assert "exact_size cannot be used with" in str(exc_info.value)

    def test_validation_exact_with_both(self) -> None:
        """Test exact_size with both min and max raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SizeConstraint(
                constraint_type="size", exact_size=40, min_size=30, max_size=50
            )
        assert "exact_size cannot be used with" in str(exc_info.value)

    def test_validation_min_greater_than_max(self) -> None:
        """Test min > max raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SizeConstraint(constraint_type="size", min_size=50, max_size=30)
        assert "min_size must be <= max_size" in str(exc_info.value)

    def test_validation_min_equals_max(self) -> None:
        """Test min == max is valid."""
        constraint = SizeConstraint(constraint_type="size", min_size=40, max_size=40)
        assert constraint.min_size == 40
        assert constraint.max_size == 40

    def test_validation_negative_min(self) -> None:
        """Test negative min_size raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SizeConstraint(constraint_type="size", min_size=-1)
        assert "non-negative" in str(exc_info.value) or "must be" in str(exc_info.value)

    def test_validation_negative_max(self) -> None:
        """Test negative max_size raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SizeConstraint(constraint_type="size", max_size=-1)
        assert "non-negative" in str(exc_info.value) or "must be" in str(exc_info.value)

    def test_validation_negative_exact(self) -> None:
        """Test negative exact_size raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            SizeConstraint(constraint_type="size", exact_size=-1)
        assert "non-negative" in str(exc_info.value) or "must be" in str(exc_info.value)

    def test_validation_zero_sizes_valid(self) -> None:
        """Test zero sizes are valid."""
        constraint1 = SizeConstraint(constraint_type="size", min_size=0)
        assert constraint1.min_size == 0

        constraint2 = SizeConstraint(constraint_type="size", max_size=0)
        assert constraint2.max_size == 0

        constraint3 = SizeConstraint(constraint_type="size", exact_size=0)
        assert constraint3.exact_size == 0

    def test_constraint_type_is_size(self) -> None:
        """Test discriminator is correct."""
        constraint = SizeConstraint(constraint_type="size", exact_size=40)
        assert constraint.constraint_type == "size"

    def test_serialization_roundtrip(self) -> None:
        """Test serialization roundtrip works."""
        constraint = SizeConstraint(constraint_type="size", min_size=30, max_size=50)

        data = constraint.model_dump()
        restored = SizeConstraint(**data)

        assert restored.min_size == constraint.min_size
        assert restored.max_size == constraint.max_size


class TestOrderingConstraint:
    """Tests for OrderingConstraint model."""

    def test_create_empty(self) -> None:
        """Test creating empty ordering constraint."""
        constraint = OrderingConstraint(constraint_type="ordering")

        assert constraint.constraint_type == "ordering"
        assert constraint.precedence_pairs == ()
        assert constraint.no_adjacent_property is None
        assert constraint.block_by_property is None
        assert constraint.min_distance is None
        assert constraint.max_distance is None
        assert constraint.practice_item_property is None
        assert constraint.randomize_within_blocks is True

    def test_create_with_precedence(self) -> None:
        """Test creating with precedence pairs."""
        item_a, item_b = uuid4(), uuid4()
        constraint = OrderingConstraint(
            constraint_type="ordering",
            precedence_pairs=(OrderingPair(before=item_a, after=item_b),),
        )

        assert len(constraint.precedence_pairs) == 1
        assert (
            constraint.precedence_pairs[0].before,
            constraint.precedence_pairs[0].after,
        ) == (item_a, item_b)

    def test_create_with_no_adjacent(self) -> None:
        """Test creating with no_adjacent_property."""
        constraint = OrderingConstraint(
            constraint_type="ordering", no_adjacent_property="item_metadata.condition"
        )

        assert constraint.no_adjacent_property == "item_metadata.condition"

    def test_create_with_blocking(self) -> None:
        """Test creating with block_by_property."""
        constraint = OrderingConstraint(
            constraint_type="ordering",
            block_by_property="item_metadata.block_type",
            randomize_within_blocks=False,
        )

        assert constraint.block_by_property == "item_metadata.block_type"
        assert constraint.randomize_within_blocks is False

    def test_create_with_min_distance(self) -> None:
        """Test creating with min_distance."""
        constraint = OrderingConstraint(
            constraint_type="ordering",
            no_adjacent_property="item_metadata.condition",
            min_distance=3,
        )

        assert constraint.min_distance == 3

    def test_create_with_max_distance(self) -> None:
        """Test creating with max_distance."""
        constraint = OrderingConstraint(
            constraint_type="ordering",
            block_by_property="item_metadata.block_type",
            max_distance=5,
        )

        assert constraint.max_distance == 5

    def test_create_with_practice_items(self) -> None:
        """Test creating with practice_item_property."""
        constraint = OrderingConstraint(
            constraint_type="ordering",
            practice_item_property="item_metadata.is_practice",
        )

        assert constraint.practice_item_property == "item_metadata.is_practice"

    def test_validation_min_distance_requires_no_adjacent(self) -> None:
        """Test min_distance without no_adjacent_property raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            OrderingConstraint(constraint_type="ordering", min_distance=3)
        assert "min_distance requires no_adjacent_property" in str(exc_info.value)

    def test_validation_max_distance_requires_blocking(self) -> None:
        """Test max_distance without block_by_property raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            OrderingConstraint(constraint_type="ordering", max_distance=5)
        assert "max_distance requires block_by_property" in str(exc_info.value)

    def test_validation_min_greater_than_max_distance(self) -> None:
        """Test min_distance > max_distance raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            OrderingConstraint(
                constraint_type="ordering",
                no_adjacent_property="item_metadata.condition",
                block_by_property="item_metadata.block_type",
                min_distance=10,
                max_distance=5,
            )
        assert "min_distance cannot be greater than max_distance" in str(exc_info.value)

    def test_validation_min_distance_negative(self) -> None:
        """Test negative min_distance raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            OrderingConstraint(
                constraint_type="ordering",
                no_adjacent_property="item_metadata.condition",
                min_distance=-1,
            )
        assert "must be" in str(exc_info.value)

    def test_validation_max_distance_negative(self) -> None:
        """Test negative max_distance raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            OrderingConstraint(
                constraint_type="ordering",
                block_by_property="item_metadata.block_type",
                max_distance=-1,
            )
        assert "must be" in str(exc_info.value)

    def test_validation_min_distance_zero(self) -> None:
        """Test zero min_distance raises ValidationError."""
        with pytest.raises(ValidationError) as exc_info:
            OrderingConstraint(
                constraint_type="ordering",
                no_adjacent_property="item_metadata.condition",
                min_distance=0,
            )
        assert "must be" in str(exc_info.value)

    def test_validation_min_equals_max_distance(self) -> None:
        """Test min_distance == max_distance is valid."""
        constraint = OrderingConstraint(
            constraint_type="ordering",
            no_adjacent_property="item_metadata.condition",
            block_by_property="item_metadata.block_type",
            min_distance=5,
            max_distance=5,
        )

        assert constraint.min_distance == 5
        assert constraint.max_distance == 5

    def test_constraint_type_is_ordering(self) -> None:
        """Test discriminator is correct."""
        constraint = OrderingConstraint(constraint_type="ordering")
        assert constraint.constraint_type == "ordering"

    def test_serialization_roundtrip(self) -> None:
        """Test serialization roundtrip works."""
        item_a, item_b = uuid4(), uuid4()
        constraint = OrderingConstraint(
            constraint_type="ordering",
            precedence_pairs=(OrderingPair(before=item_a, after=item_b),),
            no_adjacent_property="item_metadata.condition",
            min_distance=2,
        )

        data = constraint.model_dump()
        restored = OrderingConstraint(**data)

        assert len(restored.precedence_pairs) == 1
        assert (
            restored.precedence_pairs[0].before,
            restored.precedence_pairs[0].after,
        ) == (item_a, item_b)
        assert restored.no_adjacent_property == constraint.no_adjacent_property
        assert restored.min_distance == constraint.min_distance

    def test_inherits_beadbasemodel(self) -> None:
        """Test has BeadBaseModel fields."""
        constraint = OrderingConstraint(constraint_type="ordering")

        assert hasattr(constraint, "id")
        assert hasattr(constraint, "created_at")
        assert hasattr(constraint, "modified_at")


class TestListConstraintUnion:
    """Tests for ListConstraint discriminated union."""

    def test_deserialize_uniqueness(self) -> None:
        """Test deserializing uniqueness constraint from dict."""
        data = {
            "constraint_type": "uniqueness",
            "property_expression": "test",
            "allow_null": False,
        }

        # Need to use a model that has ListConstraint field
        # For now, just test the constraint types directly
        constraint = UniquenessConstraint(**data)
        assert isinstance(constraint, UniquenessConstraint)

    def test_deserialize_balance(self) -> None:
        """Test deserializing balance constraint from dict."""
        data = {
            "constraint_type": "balance",
            "property_expression": "test",
            "tolerance": 0.1,
        }

        constraint = BalanceConstraint(**data)
        assert isinstance(constraint, BalanceConstraint)

    def test_deserialize_quantile(self) -> None:
        """Test deserializing quantile constraint from dict."""
        data = {
            "constraint_type": "quantile",
            "property_expression": "test",
            "n_quantiles": 5,
            "items_per_quantile": 2,
        }

        constraint = QuantileConstraint(**data)
        assert isinstance(constraint, QuantileConstraint)

    def test_deserialize_size(self) -> None:
        """Test deserializing size constraint from dict."""
        data = {"constraint_type": "size", "exact_size": 40}

        constraint = SizeConstraint(**data)
        assert isinstance(constraint, SizeConstraint)

    def test_deserialize_ordering(self) -> None:
        """Test deserializing ordering constraint from dict."""
        data = {
            "constraint_type": "ordering",
            "no_adjacent_property": "item_metadata.condition",
        }

        constraint = OrderingConstraint(**data)
        assert isinstance(constraint, OrderingConstraint)

    def test_all_constraints_have_discriminator(self) -> None:
        """Test all constraint types have constraint_type field."""
        constraints = [
            UniquenessConstraint(
                constraint_type="uniqueness", property_expression="test"
            ),
            BalanceConstraint(constraint_type="balance", property_expression="test"),
            QuantileConstraint(constraint_type="quantile", property_expression="test"),
            SizeConstraint(constraint_type="size", exact_size=40),
            OrderingConstraint(constraint_type="ordering"),
        ]

        for constraint in constraints:
            assert hasattr(constraint, "constraint_type")
            assert constraint.constraint_type in [
                "uniqueness",
                "balance",
                "quantile",
                "size",
                "ordering",
            ]

    def test_serialization_preserves_type(self) -> None:
        """Test serialization preserves constraint_type."""
        constraints = [
            UniquenessConstraint(
                constraint_type="uniqueness", property_expression="test"
            ),
            BalanceConstraint(constraint_type="balance", property_expression="test"),
            QuantileConstraint(constraint_type="quantile", property_expression="test"),
            SizeConstraint(constraint_type="size", exact_size=40),
            OrderingConstraint(constraint_type="ordering"),
        ]

        for constraint in constraints:
            data = constraint.model_dump()
            assert "constraint_type" in data
