"""Tests for stratification utilities."""

from __future__ import annotations

from uuid import uuid4

import didactic.api as dx
import pytest

from bead.lists.stratification import assign_quantiles, assign_quantiles_by_uuid


class TestAssignQuantiles:
    """Test assign_quantiles() function."""

    def test_basic_quantiles(self) -> None:
        """Test basic quantile assignment."""
        items = list(range(100))
        result = assign_quantiles(
            items=items,
            property_getter=lambda x: float(x),
            n_quantiles=4,
        )

        assert len(result) == 100
        # Check that quantiles are distributed
        quantiles = list(result.values())
        assert set(quantiles) == {0, 1, 2, 3}

    def test_simple_quartiles(self) -> None:
        """Test quartile assignment with simple list."""
        items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = assign_quantiles(
            items=items,
            property_getter=lambda x: float(x),
            n_quantiles=4,
        )

        # First items should be in lower quartiles
        assert result[1] == 0  # Lowest quartile
        assert result[10] == 3  # Highest quartile

    def test_deciles(self) -> None:
        """Test decile assignment (10 bins)."""
        items = list(range(1, 101))
        result = assign_quantiles(
            items=items,
            property_getter=lambda x: float(x),
            n_quantiles=10,
        )

        # Check all 10 quantiles exist
        quantiles = set(result.values())
        assert quantiles == set(range(10))

    def test_stratified_quantiles(self) -> None:
        """Test quantiles with stratification."""
        # Use tuples (hashable) instead of dicts
        items = [
            (10, "A"),  # value, group
            (20, "A"),
            (30, "A"),
            (40, "A"),
            (5, "B"),
            (15, "B"),
            (25, "B"),
            (35, "B"),
        ]

        result = assign_quantiles(
            items=items,
            property_getter=lambda x: float(x[0]),  # value is first element
            n_quantiles=2,
            stratify_by=lambda x: x[1],  # group is second element
        )

        # Check that quantiles are computed separately per group
        # Within group A: 10, 20 should be in quantile 0; 30, 40 in quantile 1
        # Within group B: 5, 15 should be in quantile 0; 25, 35 in quantile 1
        group_a_items = [item for item in items if item[1] == "A"]
        group_b_items = [item for item in items if item[1] == "B"]

        # Each group should have both quantiles
        group_a_quantiles = {result[item] for item in group_a_items}
        group_b_quantiles = {result[item] for item in group_b_items}

        assert group_a_quantiles == {0, 1}
        assert group_b_quantiles == {0, 1}

    def test_minimum_quantiles_validation(self) -> None:
        """Test that n_quantiles must be >= 2."""
        items = [1, 2, 3]

        with pytest.raises(
            (ValueError, dx.ValidationError), match="n_quantiles must be >= 2"
        ):
            assign_quantiles(
                items=items,
                property_getter=lambda x: float(x),
                n_quantiles=1,
            )

    def test_empty_items_raises_error(self) -> None:
        """Test that empty items list raises ValueError."""
        with pytest.raises(
            (ValueError, dx.ValidationError), match="items list cannot be empty"
        ):
            assign_quantiles(
                items=[],
                property_getter=lambda x: float(x),
                n_quantiles=10,
            )

    def test_single_group_with_stratification(self) -> None:
        """Test stratification with only one group."""
        # Use tuples (hashable) instead of dicts
        items = [(i, "A") for i in range(10)]  # value, group

        result = assign_quantiles(
            items=items,
            property_getter=lambda x: float(x[0]),
            n_quantiles=2,
            stratify_by=lambda x: x[1],
        )

        # Should work fine, just one group
        assert len(result) == 10
        quantiles = set(result.values())
        assert quantiles == {0, 1}


class TestAssignQuantilesByUUID:
    """Test assign_quantiles_by_uuid() convenience function."""

    def test_basic_uuid_quantiles(self) -> None:
        """Test basic quantile assignment by UUID."""
        uuids = [uuid4() for _ in range(20)]
        metadata = {uid: {"score": float(i)} for i, uid in enumerate(uuids)}

        result = assign_quantiles_by_uuid(
            item_ids=uuids,
            item_metadata=metadata,
            property_key="score",
            n_quantiles=4,
        )

        assert len(result) == 20
        # Check that all UUIDs are in result
        assert set(result.keys()) == set(uuids)

    def test_uuid_with_stratification(self) -> None:
        """Test UUID quantiles with stratification."""
        uuids = [uuid4() for _ in range(20)]
        metadata = {
            uid: {
                "score": float(i),
                "group": "A" if i < 10 else "B",
            }
            for i, uid in enumerate(uuids)
        }

        result = assign_quantiles_by_uuid(
            item_ids=uuids,
            item_metadata=metadata,
            property_key="score",
            n_quantiles=2,
            stratify_by_key="group",
        )

        # Check that quantiles are computed per group
        group_a_uuids = [uid for uid in uuids if metadata[uid]["group"] == "A"]
        group_b_uuids = [uid for uid in uuids if metadata[uid]["group"] == "B"]

        group_a_quantiles = {result[uid] for uid in group_a_uuids}
        group_b_quantiles = {result[uid] for uid in group_b_uuids}

        assert group_a_quantiles == {0, 1}
        assert group_b_quantiles == {0, 1}

    def test_missing_property_key_raises_error(self) -> None:
        """Test that missing property_key raises ValueError."""
        uuids = [uuid4() for _ in range(5)]
        metadata = {uid: {"value": float(i)} for i, uid in enumerate(uuids)}

        with pytest.raises(
            (ValueError, dx.ValidationError), match="Property 'score' not found"
        ):
            assign_quantiles_by_uuid(
                item_ids=uuids,
                item_metadata=metadata,
                property_key="score",  # Wrong key
                n_quantiles=2,
            )

    def test_missing_uuid_raises_error(self) -> None:
        """Test that missing UUID raises KeyError."""
        uuids = [uuid4() for _ in range(5)]
        metadata = {
            uid: {"score": float(i)} for i, uid in enumerate(uuids[:-1])
        }  # Missing last

        with pytest.raises(KeyError):
            assign_quantiles_by_uuid(
                item_ids=uuids,
                item_metadata=metadata,
                property_key="score",
                n_quantiles=2,
            )

    def test_missing_stratify_key_raises_error(self) -> None:
        """Test that missing stratify_by_key raises ValueError."""
        uuids = [uuid4() for _ in range(5)]
        metadata = {uid: {"score": float(i)} for i, uid in enumerate(uuids)}

        with pytest.raises(
            (ValueError, dx.ValidationError), match="Stratification key"
        ):
            assign_quantiles_by_uuid(
                item_ids=uuids,
                item_metadata=metadata,
                property_key="score",
                n_quantiles=2,
                stratify_by_key="group",  # Not in metadata
            )

    def test_float_conversion(self) -> None:
        """Test that property values are converted to float."""
        uuids = [uuid4() for _ in range(10)]
        # Store as strings
        metadata = {uid: {"score": str(i)} for i, uid in enumerate(uuids)}

        result = assign_quantiles_by_uuid(
            item_ids=uuids,
            item_metadata=metadata,
            property_key="score",
            n_quantiles=2,
        )

        # Should work fine with string→float conversion
        assert len(result) == 10
