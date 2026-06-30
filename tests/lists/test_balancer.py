"""Tests for QuantileBalancer."""

from __future__ import annotations

from uuid import UUID, uuid4

import didactic.api as dx
import pytest

from bead.lists.balancer import QuantileBalancer


def test_balancer_initialization() -> None:
    """Test QuantileBalancer initialization."""
    balancer = QuantileBalancer(n_quantiles=5, random_seed=42)
    assert balancer.n_quantiles == 5
    assert balancer.random_seed == 42


def test_balancer_invalid_n_quantiles() -> None:
    """Test that n_quantiles < 2 raises ValueError."""
    with pytest.raises(
        (ValueError, dx.ValidationError), match="n_quantiles must be >= 2"
    ):
        QuantileBalancer(n_quantiles=1)


def test_create_strata_basic() -> None:
    """Test basic stratum creation."""
    balancer = QuantileBalancer(n_quantiles=5, random_seed=42)

    # Create 100 items with sequential values
    items = [uuid4() for _ in range(100)]
    values = {item: float(i) for i, item in enumerate(items)}

    def value_func(uid: UUID) -> float:
        return values[uid]

    strata = balancer._create_strata(items, value_func)

    # Should have 5 strata
    assert len(strata) == 5

    # Each stratum should have items
    for q in range(5):
        assert len(strata[q]) > 0

    # All items should be assigned
    total_items = sum(len(strata[q]) for q in range(5))
    assert total_items == 100


def test_balance_basic() -> None:
    """Test basic balancing across lists."""
    balancer = QuantileBalancer(n_quantiles=5, random_seed=42)

    # Create 100 items with sequential values
    items = [uuid4() for _ in range(100)]
    values = {item: float(i) for i, item in enumerate(items)}

    def value_func(uid: UUID) -> float:
        return values[uid]

    # Balance across 4 lists, 5 items per quantile per list
    lists = balancer.balance(
        items, value_func, n_lists=4, items_per_quantile_per_list=5
    )

    # Should have 4 lists
    assert len(lists) == 4

    # Each list should have 25 items (5 quantiles * 5 items)
    for lst in lists:
        assert len(lst) == 25


def test_balance_deterministic() -> None:
    """Test that balancing is deterministic with same seed."""
    items = [uuid4() for _ in range(100)]
    values = {item: float(i) for i, item in enumerate(items)}

    def value_func(uid: UUID) -> float:
        return values[uid]

    # Create two balancers with same seed
    balancer1 = QuantileBalancer(n_quantiles=5, random_seed=42)
    balancer2 = QuantileBalancer(n_quantiles=5, random_seed=42)

    lists1 = balancer1.balance(items, value_func, 4, 5)
    lists2 = balancer2.balance(items, value_func, 4, 5)

    # Should produce identical results
    assert len(lists1) == len(lists2)
    for lst1, lst2 in zip(lists1, lists2, strict=False):
        assert lst1 == lst2


def test_balance_different_seeds() -> None:
    """Test that different seeds produce different results."""
    items = [uuid4() for _ in range(100)]
    values = {item: float(i) for i, item in enumerate(items)}

    def value_func(uid: UUID) -> float:
        return values[uid]

    balancer1 = QuantileBalancer(n_quantiles=5, random_seed=42)
    balancer2 = QuantileBalancer(n_quantiles=5, random_seed=99)

    lists1 = balancer1.balance(items, value_func, 4, 5)
    lists2 = balancer2.balance(items, value_func, 4, 5)

    # Should produce different results (very unlikely to be same)
    different = False
    for lst1, lst2 in zip(lists1, lists2, strict=False):
        if lst1 != lst2:
            different = True
            break
    assert different


def test_balance_invalid_n_lists() -> None:
    """Test that n_lists < 1 raises ValueError."""
    balancer = QuantileBalancer(n_quantiles=5)
    items = [uuid4() for _ in range(10)]
    values = {item: float(i) for i, item in enumerate(items)}

    with pytest.raises((ValueError, dx.ValidationError), match="n_lists must be >= 1"):
        balancer.balance(
            items, lambda uid: values[uid], n_lists=0, items_per_quantile_per_list=1
        )


def test_balance_invalid_items_per_quantile() -> None:
    """Test that items_per_quantile_per_list < 1 raises ValueError."""
    balancer = QuantileBalancer(n_quantiles=5)
    items = [uuid4() for _ in range(10)]
    values = {item: float(i) for i, item in enumerate(items)}

    with pytest.raises(
        (ValueError, dx.ValidationError),
        match="items_per_quantile_per_list must be >= 1",
    ):
        balancer.balance(
            items, lambda uid: values[uid], n_lists=2, items_per_quantile_per_list=0
        )


def test_balance_all_items_assigned() -> None:
    """Test that all items are assigned to lists."""
    balancer = QuantileBalancer(n_quantiles=5, random_seed=42)
    items = [uuid4() for _ in range(100)]
    values = {item: float(i) for i, item in enumerate(items)}

    lists = balancer.balance(items, lambda uid: values[uid], 4, 5)

    # Collect all assigned items
    assigned_items: set[UUID] = set()
    for lst in lists:
        assigned_items.update(lst)

    # Should have all 100 items assigned
    # (4 lists * 5 quantiles * 5 items = 100 items total)
    assert len(assigned_items) == 100


def test_balance_no_duplicates() -> None:
    """Test that no item appears in multiple lists."""
    balancer = QuantileBalancer(n_quantiles=5, random_seed=42)
    items = [uuid4() for _ in range(100)]
    values = {item: float(i) for i, item in enumerate(items)}

    lists = balancer.balance(items, lambda uid: values[uid], 4, 5)

    # Check for duplicates across lists
    all_items: list[UUID] = []
    for lst in lists:
        all_items.extend(lst)

    assert len(all_items) == len(set(all_items))


def test_compute_balance_score_uniform() -> None:
    """Test balance score for uniformly distributed items."""
    balancer = QuantileBalancer(n_quantiles=5, random_seed=42)

    # Create items with uniform distribution
    items = [uuid4() for _ in range(100)]
    values = {item: float(i) for i, item in enumerate(items)}

    def value_func(uid: UUID) -> float:
        return values[uid]

    score = balancer.compute_balance_score(items, value_func)

    # Should be close to 1.0 for uniform distribution
    assert score > 0.9


def test_compute_balance_score_empty() -> None:
    """Test balance score for empty list returns 0.0."""
    balancer = QuantileBalancer(n_quantiles=5)
    score = balancer.compute_balance_score([], lambda uid: 0.0)
    assert score == 0.0


def test_compute_balance_score_imbalanced() -> None:
    """Test balance score for imbalanced distribution."""
    balancer = QuantileBalancer(n_quantiles=5, random_seed=42)

    # Create items with skewed distribution (all same value)
    items = [uuid4() for _ in range(50)]
    values = dict.fromkeys(items, 1.0)

    def value_func(uid: UUID) -> float:
        return values[uid]

    score = balancer.compute_balance_score(items, value_func)

    # Should be lower for imbalanced distribution
    # All items will be in one quantile, so score should be 0
    assert score < 0.5


def test_balance_with_different_n_quantiles() -> None:
    """Test balancing with different number of quantiles."""
    items = [uuid4() for _ in range(100)]
    values = {item: float(i) for i, item in enumerate(items)}

    def value_func(uid: UUID) -> float:
        return values[uid]

    # Test with 2 quantiles
    balancer2 = QuantileBalancer(n_quantiles=2, random_seed=42)
    lists2 = balancer2.balance(items, value_func, 2, 10)
    assert len(lists2) == 2
    assert all(len(lst) == 20 for lst in lists2)  # 2 quantiles * 10 items

    # Test with 10 quantiles
    balancer10 = QuantileBalancer(n_quantiles=10, random_seed=42)
    lists10 = balancer10.balance(items, value_func, 2, 5)
    assert len(lists10) == 2
    assert all(len(lst) == 50 for lst in lists10)  # 10 quantiles * 5 items


def test_balance_insufficient_items() -> None:
    """Test balancing with fewer items than requested."""
    balancer = QuantileBalancer(n_quantiles=5, random_seed=42)

    # Only 20 items but requesting more
    items = [uuid4() for _ in range(20)]
    values = {item: float(i) for i, item in enumerate(items)}

    lists = balancer.balance(items, lambda uid: values[uid], 4, 5)

    # Should still create 4 lists, but some may have fewer items
    assert len(lists) == 4

    # Total items should not exceed 20
    total = sum(len(lst) for lst in lists)
    assert total <= 20


def test_create_strata_edge_values() -> None:
    """Test stratum creation with edge case values."""
    balancer = QuantileBalancer(n_quantiles=5, random_seed=42)

    # All same values
    items = [uuid4() for _ in range(50)]
    values = dict.fromkeys(items, 5.0)

    strata = balancer._create_strata(items, lambda uid: values[uid])

    # All items should be in one stratum
    total = sum(len(strata[q]) for q in range(5))
    assert total == 50


def test_balance_preserves_item_identity() -> None:
    """Test that balancing preserves item UUIDs."""
    balancer = QuantileBalancer(n_quantiles=5, random_seed=42)
    items = [uuid4() for _ in range(100)]
    values = {item: float(i) for i, item in enumerate(items)}

    lists = balancer.balance(items, lambda uid: values[uid], 4, 5)

    # All items in lists should be from original items
    for lst in lists:
        for item_id in lst:
            assert item_id in items


def test_balance_quantile_coverage() -> None:
    """Test that each list gets items from all quantiles."""
    balancer = QuantileBalancer(n_quantiles=5, random_seed=42)
    items = [uuid4() for _ in range(100)]
    values = {item: float(i) for i, item in enumerate(items)}

    def value_func(uid: UUID) -> float:
        return values[uid]

    lists = balancer.balance(items, value_func, 4, 5)

    # Each list should have items from multiple quantiles
    for lst in lists:
        list_values = [values[uid] for uid in lst]
        # Check that values span a range (not all from same quantile)
        assert max(list_values) - min(list_values) > 10
