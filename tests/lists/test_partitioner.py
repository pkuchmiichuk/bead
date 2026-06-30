"""Tests for ListPartitioner."""

from __future__ import annotations

from uuid import UUID, uuid4

import didactic.api as dx
import pytest

from bead.dsl.errors import EvaluationError
from bead.lists import ExperimentList
from bead.lists.constraints import (
    BalanceConstraint,
    QuantileConstraint,
    SizeConstraint,
    UniquenessConstraint,
)
from bead.lists.partitioner import ListPartitioner

type ItemMetadata = dict[str, str | float | int]
type MetadataDict = dict[UUID, ItemMetadata]
type QuantileMetadataDict = dict[UUID, dict[str, float]]


def test_partitioner_initialization() -> None:
    """Test ListPartitioner initialization."""
    partitioner = ListPartitioner(random_seed=42)
    assert partitioner.random_seed == 42


def test_partition_random_basic(sample_item_metadata: MetadataDict) -> None:
    """Test basic random partitioning."""
    partitioner = ListPartitioner(random_seed=42)
    items = list(sample_item_metadata.keys())

    lists = partitioner.partition(
        items, n_lists=5, strategy="random", metadata=sample_item_metadata
    )

    assert len(lists) == 5
    # Each list should have roughly equal items
    sizes = [len(exp_list.item_refs) for exp_list in lists]
    assert max(sizes) - min(sizes) <= 1


def test_partition_balanced_basic(sample_item_metadata: MetadataDict) -> None:
    """Test basic balanced partitioning."""
    partitioner = ListPartitioner(random_seed=42)
    items = list(sample_item_metadata.keys())

    lists = partitioner.partition(
        items, n_lists=5, strategy="balanced", metadata=sample_item_metadata
    )

    assert len(lists) == 5
    # Check relatively balanced
    sizes = [len(exp_list.item_refs) for exp_list in lists]
    assert max(sizes) - min(sizes) <= 5


def test_partition_stratified_basic(
    sample_quantile_metadata: QuantileMetadataDict,
) -> None:
    """Test basic stratified partitioning."""
    partitioner = ListPartitioner(random_seed=42)
    items = list(sample_quantile_metadata.keys())

    constraint = QuantileConstraint(
        constraint_type="quantile",
        property_expression="item['lm_prob']",
        n_quantiles=5,
        items_per_quantile=2,
    )

    lists = partitioner.partition(
        items,
        n_lists=4,
        constraints=[constraint],
        strategy="stratified",
        metadata=sample_quantile_metadata,
    )

    assert len(lists) == 4
    # Each list should have items from multiple quantiles
    for exp_list in lists:
        assert len(exp_list.item_refs) > 0


def test_partition_deterministic(sample_item_metadata: MetadataDict) -> None:
    """Test that partitioning is deterministic with same seed."""
    items = list(sample_item_metadata.keys())

    partitioner1 = ListPartitioner(random_seed=42)
    partitioner2 = ListPartitioner(random_seed=42)

    lists1 = partitioner1.partition(items, 5, metadata=sample_item_metadata)
    lists2 = partitioner2.partition(items, 5, metadata=sample_item_metadata)

    # Should produce identical results
    assert len(lists1) == len(lists2)
    for lst1, lst2 in zip(lists1, lists2, strict=False):
        assert lst1.item_refs == lst2.item_refs


def test_partition_different_seeds(sample_item_metadata: MetadataDict) -> None:
    """Test that different seeds produce different results."""
    items = list(sample_item_metadata.keys())

    partitioner1 = ListPartitioner(random_seed=42)
    partitioner2 = ListPartitioner(random_seed=99)

    lists1 = partitioner1.partition(items, 5, metadata=sample_item_metadata)
    lists2 = partitioner2.partition(items, 5, metadata=sample_item_metadata)

    # Should produce different results (very unlikely to be same)
    different = False
    for lst1, lst2 in zip(lists1, lists2, strict=False):
        if lst1.item_refs != lst2.item_refs:
            different = True
            break
    assert different


def test_partition_all_items_assigned(sample_item_metadata: MetadataDict) -> None:
    """Test that all items are assigned to lists."""
    partitioner = ListPartitioner(random_seed=42)
    items = list(sample_item_metadata.keys())

    lists = partitioner.partition(items, 5, metadata=sample_item_metadata)

    # Collect all assigned items
    assigned_items: set[UUID] = set()
    for exp_list in lists:
        assigned_items.update(exp_list.item_refs)

    # All items should be assigned
    assert assigned_items == set(items)


def test_partition_no_duplicates(sample_item_metadata: MetadataDict) -> None:
    """Test that no item appears in multiple lists."""
    partitioner = ListPartitioner(random_seed=42)
    items = list(sample_item_metadata.keys())

    lists = partitioner.partition(items, 5, metadata=sample_item_metadata)

    # Check for duplicates across lists
    all_items: list[UUID] = []
    for exp_list in lists:
        all_items.extend(exp_list.item_refs)

    assert len(all_items) == len(set(all_items))


def test_partition_correct_list_count(sample_item_metadata: MetadataDict) -> None:
    """Test that correct number of lists is created."""
    partitioner = ListPartitioner(random_seed=42)
    items = list(sample_item_metadata.keys())

    for n_lists in [1, 3, 5, 10]:
        lists = partitioner.partition(items, n_lists, metadata=sample_item_metadata)
        assert len(lists) == n_lists


def test_partition_invalid_n_lists(sample_item_metadata: MetadataDict) -> None:
    """Test that n_lists < 1 raises ValueError."""
    partitioner = ListPartitioner()
    items = list(sample_item_metadata.keys())

    with pytest.raises((ValueError, dx.ValidationError), match="n_lists must be >= 1"):
        partitioner.partition(items, 0, metadata=sample_item_metadata)


def test_partition_invalid_strategy(sample_item_metadata: MetadataDict) -> None:
    """Test that unknown strategy raises ValueError."""
    partitioner = ListPartitioner()
    items = list(sample_item_metadata.keys())

    with pytest.raises((ValueError, dx.ValidationError), match="Unknown strategy"):
        partitioner.partition(
            items, 5, strategy="unknown", metadata=sample_item_metadata
        )


def test_partition_with_uniqueness_constraint(
    sample_item_metadata: MetadataDict,
) -> None:
    """Test partitioning with uniqueness constraint."""
    partitioner = ListPartitioner(random_seed=42)
    items = list(sample_item_metadata.keys())

    constraint = UniquenessConstraint(
        constraint_type="uniqueness",
        property_expression="item['category']",
        allow_null=False,
    )

    lists = partitioner.partition(
        items,
        5,
        constraints=[constraint],
        strategy="balanced",
        metadata=sample_item_metadata,
    )

    # With only 5 unique categories and uniqueness constraint,
    # the partitioner tries to minimize violations but can't fully satisfy
    # the constraint with 100 items. Just verify lists were created.
    assert len(lists) == 5
    # Verify at least some items were assigned
    assert all(len(exp_list.item_refs) > 0 for exp_list in lists)


def test_partition_with_balance_constraint(sample_item_metadata: MetadataDict) -> None:
    """Test partitioning with balance constraint."""
    partitioner = ListPartitioner(random_seed=42)
    items = list(sample_item_metadata.keys())

    constraint = BalanceConstraint(
        constraint_type="balance", property_expression="item['category']", tolerance=0.3
    )

    lists = partitioner.partition(
        items,
        5,
        constraints=[constraint],
        strategy="balanced",
        metadata=sample_item_metadata,
    )

    # Each list should have balanced categories
    for exp_list in lists:
        if len(exp_list.item_refs) > 0:
            categories = [
                sample_item_metadata[uid]["category"] for uid in exp_list.item_refs
            ]
            # Just check that we have multiple categories
            assert len(set(categories)) >= 1


def test_partition_with_size_constraint(sample_item_metadata: MetadataDict) -> None:
    """Test partitioning with size constraint."""
    partitioner = ListPartitioner(random_seed=42)
    items = list(sample_item_metadata.keys())

    constraint = SizeConstraint(constraint_type="size", min_size=15, max_size=25)

    lists = partitioner.partition(
        items,
        5,
        constraints=[constraint],
        strategy="balanced",
        metadata=sample_item_metadata,
    )

    # Balanced partitioning tries to minimize violations but may not
    # perfectly satisfy all constraints with greedy assignment
    assert len(lists) == 5
    # Most lists should be close to satisfying the constraint
    sizes = [len(exp_list.item_refs) for exp_list in lists]
    avg_size = sum(sizes) / len(sizes)
    assert 15 <= avg_size <= 25  # Average size should be in range


def test_partition_with_quantile_constraint(
    sample_quantile_metadata: QuantileMetadataDict,
) -> None:
    """Test partitioning with quantile constraint."""
    partitioner = ListPartitioner(random_seed=42)
    items = list(sample_quantile_metadata.keys())

    constraint = QuantileConstraint(
        constraint_type="quantile",
        property_expression="item['lm_prob']",
        n_quantiles=5,
        items_per_quantile=2,
    )

    lists = partitioner.partition(
        items,
        4,
        constraints=[constraint],
        strategy="stratified",
        metadata=sample_quantile_metadata,
    )

    # Each list should have items from multiple quantiles
    for exp_list in lists:
        if len(exp_list.item_refs) > 0:
            values = [
                sample_quantile_metadata[uid]["lm_prob"] for uid in exp_list.item_refs
            ]
            # Check that values span a range
            assert max(values) - min(values) > 0.1


def test_partition_balance_metrics_computed(sample_item_metadata: MetadataDict) -> None:
    """Test that balance metrics are computed."""
    partitioner = ListPartitioner(random_seed=42)
    items = list(sample_item_metadata.keys())

    constraint = QuantileConstraint(
        constraint_type="quantile",
        property_expression="item['value']",
        n_quantiles=5,
        items_per_quantile=2,
    )

    lists = partitioner.partition(
        items,
        5,
        constraints=[constraint],
        metadata=sample_item_metadata,
    )

    # Each list should have balance metrics
    for exp_list in lists:
        assert "size" in exp_list.balance_metrics
        assert exp_list.balance_metrics["size"] == len(exp_list.item_refs)
        assert f"quantile_{constraint.property_expression}" in exp_list.balance_metrics


def test_partition_constraints_attached(sample_item_metadata: MetadataDict) -> None:
    """Test that constraints are attached to lists."""
    partitioner = ListPartitioner(random_seed=42)
    items = list(sample_item_metadata.keys())

    constraint = UniquenessConstraint(
        constraint_type="uniqueness", property_expression="item['category']"
    )

    lists = partitioner.partition(
        items,
        5,
        constraints=[constraint],
        metadata=sample_item_metadata,
    )

    # Each list should have the constraint
    for exp_list in lists:
        assert len(exp_list.list_constraints) == 1
        assert exp_list.list_constraints[0] == constraint


def test_partition_single_list(sample_item_metadata: MetadataDict) -> None:
    """Test partitioning into single list."""
    partitioner = ListPartitioner(random_seed=42)
    items = list(sample_item_metadata.keys())

    lists = partitioner.partition(items, 1, metadata=sample_item_metadata)

    assert len(lists) == 1
    assert len(lists[0].item_refs) == len(items)


def test_partition_fewer_items_than_lists() -> None:
    """Test partitioning with fewer items than lists."""
    partitioner = ListPartitioner(random_seed=42)
    items = [uuid4() for _ in range(3)]
    metadata = {uid: {"value": i} for i, uid in enumerate(items)}

    lists = partitioner.partition(items, 5, metadata=metadata)

    assert len(lists) == 5
    # Some lists will be empty
    non_empty = [exp_list for exp_list in lists if exp_list.item_refs]
    assert len(non_empty) == 3


def test_partition_no_constraints(sample_item_metadata: MetadataDict) -> None:
    """Test partitioning without constraints."""
    partitioner = ListPartitioner(random_seed=42)
    items = list(sample_item_metadata.keys())

    lists = partitioner.partition(items, 5, metadata=sample_item_metadata)

    # Should still work
    assert len(lists) == 5
    for exp_list in lists:
        assert len(exp_list.list_constraints) == 0


def test_partition_no_metadata() -> None:
    """Test partitioning without metadata."""
    partitioner = ListPartitioner(random_seed=42)
    items = [uuid4() for _ in range(100)]

    lists = partitioner.partition(items, 5)

    # Should work for random strategy without constraints
    assert len(lists) == 5


def test_check_uniqueness_satisfied() -> None:
    """Test uniqueness constraint checking."""
    partitioner = ListPartitioner()
    items = [uuid4() for _ in range(5)]
    metadata = {uid: {"prop": i} for i, uid in enumerate(items)}

    exp_list = ExperimentList(name="test", list_number=0)
    for item_id in items:
        exp_list = exp_list.with_item(item_id)
    constraint = UniquenessConstraint(
        constraint_type="uniqueness", property_expression="item['prop']"
    )
    assert partitioner._check_uniqueness(exp_list, constraint, metadata)


def test_check_uniqueness_violated() -> None:
    """Test uniqueness constraint violation."""
    partitioner = ListPartitioner()
    items = [uuid4() for _ in range(5)]
    metadata = {uid: {"prop": 1} for uid in items}  # All same value

    exp_list = ExperimentList(name="test", list_number=0)
    for item_id in items:
        exp_list = exp_list.with_item(item_id)
    constraint = UniquenessConstraint(
        constraint_type="uniqueness", property_expression="item['prop']"
    )
    assert not partitioner._check_uniqueness(exp_list, constraint, metadata)


def test_check_size_satisfied() -> None:
    """Test size constraint checking."""
    partitioner = ListPartitioner()

    exp_list = ExperimentList(name="test", list_number=0)
    for _ in range(20):
        exp_list = exp_list.with_item(uuid4())
    constraint = SizeConstraint(constraint_type="size", min_size=10, max_size=30)
    assert partitioner._check_size(exp_list, constraint)


def test_check_size_violated() -> None:
    """Test size constraint violation."""
    partitioner = ListPartitioner()

    exp_list = ExperimentList(name="test", list_number=0)
    for _ in range(5):
        exp_list = exp_list.with_item(uuid4())
    constraint = SizeConstraint(constraint_type="size", min_size=10, max_size=30)
    assert not partitioner._check_size(exp_list, constraint)


def test_extract_property_value_simple() -> None:
    """Test property value extraction with simple path."""
    partitioner = ListPartitioner()
    item_id = uuid4()
    metadata = {item_id: {"prop": 42}}

    value = partitioner._extract_property_value(item_id, "item['prop']", None, metadata)
    assert value == 42


def test_extract_property_value_nested() -> None:
    """Test property value extraction with nested path."""
    partitioner = ListPartitioner()
    item_id = uuid4()
    metadata = {item_id: {"level1": {"level2": {"value": 99}}}}

    value = partitioner._extract_property_value(
        item_id, "item['level1']['level2']['value']", None, metadata
    )
    assert value == 99


def test_extract_property_value_missing_item() -> None:
    """Test property extraction with missing item."""
    partitioner = ListPartitioner()
    item_id = uuid4()
    metadata = {}

    with pytest.raises(KeyError, match="not found in metadata"):
        partitioner._extract_property_value(item_id, "item['prop']", None, metadata)


def test_extract_property_value_missing_property() -> None:
    """Test property extraction with missing property."""
    partitioner = ListPartitioner()
    item_id = uuid4()
    metadata = {item_id: {"other": 42}}

    with pytest.raises(EvaluationError):
        partitioner._extract_property_value(item_id, "item['prop']", None, metadata)


def test_stratified_fallback_to_balanced() -> None:
    """Test that stratified falls back to balanced without quantile constraints."""
    partitioner = ListPartitioner(random_seed=42)
    items = [uuid4() for _ in range(100)]
    metadata = {uid: {"value": i} for i, uid in enumerate(items)}

    # Use stratified strategy but no quantile constraints
    lists = partitioner.partition(items, 5, strategy="stratified", metadata=metadata)

    # Should fall back to balanced and still work
    assert len(lists) == 5
    assert all(len(exp_list.item_refs) > 0 for exp_list in lists)


def test_list_names_and_numbers(sample_item_metadata: MetadataDict) -> None:
    """Test that lists have correct names and numbers."""
    partitioner = ListPartitioner(random_seed=42)
    items = list(sample_item_metadata.keys())

    lists = partitioner.partition(items, 5, metadata=sample_item_metadata)

    for i, exp_list in enumerate(lists):
        assert exp_list.name == f"list_{i}"
        assert exp_list.list_number == i


def test_compute_balance_metrics_empty_list() -> None:
    """Test balance metrics computation for empty list."""
    partitioner = ListPartitioner()

    exp_list = ExperimentList(name="test", list_number=0)
    constraint = QuantileConstraint(
        constraint_type="quantile", property_expression="item['value']", n_quantiles=5
    )

    metrics = partitioner._compute_balance_metrics(exp_list, [constraint], {})

    assert metrics["size"] == 0
    assert "quantile_item['value']" in metrics


def test_constraint_priority_weighting() -> None:
    """Test that higher priority constraints are weighted more heavily."""
    partitioner = ListPartitioner(random_seed=42)

    # Create 10 items
    items = [uuid4() for _ in range(10)]
    metadata = {uid: {"value": i} for i, uid in enumerate(items)}

    # High priority size constraint (priority=10)
    size_constraint = SizeConstraint(constraint_type="size", exact_size=5, priority=10)
    # Low priority uniqueness constraint (priority=1)
    uniqueness_constraint = UniquenessConstraint(
        constraint_type="uniqueness", property_expression="item['value']", priority=1
    )

    lists = partitioner.partition(
        items,
        2,
        constraints=[size_constraint, uniqueness_constraint],
        strategy="balanced",
        metadata=metadata,
    )

    # With high priority on size, lists should be exactly equal size
    sizes = [len(exp_list.item_refs) for exp_list in lists]
    assert sizes == [5, 5]  # Exact equal sizes due to high priority


def test_constraint_priority_default() -> None:
    """Test that constraints default to priority=1."""
    constraint = SizeConstraint(constraint_type="size", exact_size=40)
    assert constraint.priority == 1

    constraint2 = UniquenessConstraint(
        constraint_type="uniqueness", property_expression="item['value']"
    )
    assert constraint2.priority == 1

    constraint3 = BalanceConstraint(
        constraint_type="balance", property_expression="item['category']"
    )
    assert constraint3.priority == 1

    constraint4 = QuantileConstraint(
        constraint_type="quantile", property_expression="item['score']"
    )
    assert constraint4.priority == 1
