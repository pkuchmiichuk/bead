"""Shared pytest fixtures for list model tests."""

from __future__ import annotations

from typing import Any
from uuid import UUID, uuid4

import pytest

from bead.lists import ExperimentList, ListCollection
from bead.lists.constraints import (
    BalanceConstraint,
    OrderingConstraint,
    OrderingPair,
    QuantileConstraint,
    SizeConstraint,
    UniquenessConstraint,
)
from bead.lists.partitioner import ListPartitioner


@pytest.fixture
def sample_uuid() -> UUID:
    """Provide a fixed UUID for testing.

    Returns
    -------
    UUID
        Sample UUID.
    """
    return UUID("12345678-1234-5678-1234-567812345678")


@pytest.fixture
def sample_item_uuids() -> list[UUID]:
    """Generate list of 100 UUIDs for testing.

    Returns
    -------
    list[UUID]
        List of 100 UUIDs.
    """
    return [uuid4() for _ in range(100)]


@pytest.fixture
def empty_experiment_list() -> ExperimentList:
    """Create an empty experiment list.

    Returns
    -------
    ExperimentList
        Empty list with default values.
    """
    return ExperimentList(name="empty_list", list_number=0)


@pytest.fixture
def experiment_list_with_items(sample_item_uuids: list[UUID]) -> ExperimentList:
    """Create an experiment list with 20 items.

    Parameters
    ----------
    sample_item_uuids : list[UUID]
        Pool of UUIDs to draw from.

    Returns
    -------
    ExperimentList
        List with 20 items.
    """
    exp_list = ExperimentList(name="list_with_items", list_number=1)
    for item_id in sample_item_uuids[:20]:
        exp_list = exp_list.with_item(item_id)
    return exp_list


@pytest.fixture
def uniqueness_constraint() -> UniquenessConstraint:
    """Create a uniqueness constraint.

    Returns
    -------
    UniquenessConstraint
        Constraint requiring unique target verbs.
    """
    return UniquenessConstraint(
        constraint_type="uniqueness",
        property_expression="item['target_verb']",
        allow_null=False,
    )


@pytest.fixture
def balance_constraint() -> BalanceConstraint:
    """Create a balance constraint.

    Returns
    -------
    BalanceConstraint
        Constraint for balanced transitivity.
    """
    return BalanceConstraint(
        constraint_type="balance",
        property_expression="item['transitivity']",
        tolerance=0.1,
    )


@pytest.fixture
def quantile_constraint() -> QuantileConstraint:
    """Create a quantile constraint.

    Returns
    -------
    QuantileConstraint
        Constraint for LM probability quantiles.
    """
    return QuantileConstraint(
        constraint_type="quantile",
        property_expression="item['lm_prob']",
        n_quantiles=5,
        items_per_quantile=2,
    )


@pytest.fixture
def size_constraint_exact() -> SizeConstraint:
    """Create a size constraint with exact size.

    Returns
    -------
    SizeConstraint
        Constraint requiring exactly 40 items.
    """
    return SizeConstraint(constraint_type="size", exact_size=40)


@pytest.fixture
def size_constraint_range() -> SizeConstraint:
    """Create a size constraint with range.

    Returns
    -------
    SizeConstraint
        Constraint requiring 30-50 items.
    """
    return SizeConstraint(constraint_type="size", min_size=30, max_size=50)


@pytest.fixture
def experiment_list_with_constraints(
    uniqueness_constraint: UniquenessConstraint,
    balance_constraint: BalanceConstraint,
) -> ExperimentList:
    """Create an experiment list with constraints.

    Parameters
    ----------
    uniqueness_constraint : UniquenessConstraint
        Uniqueness constraint to add.
    balance_constraint : BalanceConstraint
        Balance constraint to add.

    Returns
    -------
    ExperimentList
        List with constraints.
    """
    return ExperimentList(
        name="list_with_constraints",
        list_number=2,
        list_constraints=[uniqueness_constraint, balance_constraint],
    )


@pytest.fixture
def sample_list_collection(
    sample_uuid: UUID, experiment_list_with_items: ExperimentList
) -> ListCollection:
    """Create a list collection with one list.

    Parameters
    ----------
    sample_uuid : UUID
        UUID for source items.
    experiment_list_with_items : ExperimentList
        List to add to collection.

    Returns
    -------
    ListCollection
        Collection with one list.
    """
    collection = ListCollection(
        name="sample_collection",
        source_items_id=sample_uuid,
        partitioning_strategy="balanced",
        partitioning_config={"n_lists": 1, "seed": 42},
    )
    collection = collection.with_list(experiment_list_with_items)
    return collection


@pytest.fixture
def ordering_constraint_precedence() -> OrderingConstraint:
    """Create ordering constraint with precedence pairs.

    Returns
    -------
    OrderingConstraint
        Constraint with precedence pairs.
    """
    return OrderingConstraint(
        constraint_type="ordering",
        precedence_pairs=(OrderingPair(before=uuid4(), after=uuid4()),),
    )


@pytest.fixture
def ordering_constraint_no_adjacent() -> OrderingConstraint:
    """Create ordering constraint with no-adjacent property.

    Returns
    -------
    OrderingConstraint
        Constraint preventing adjacent items with same condition.
    """
    return OrderingConstraint(
        constraint_type="ordering",
        no_adjacent_property="item_metadata.condition",
        min_distance=2,
    )


@pytest.fixture
def ordering_constraint_blocking() -> OrderingConstraint:
    """Create ordering constraint with blocking.

    Returns
    -------
    OrderingConstraint
        Constraint that groups items by block type.
    """
    return OrderingConstraint(
        constraint_type="ordering",
        block_by_property="item_metadata.block_type",
        randomize_within_blocks=True,
    )


@pytest.fixture
def ordering_constraint_practice() -> OrderingConstraint:
    """Create ordering constraint for practice items.

    Returns
    -------
    OrderingConstraint
        Constraint ensuring practice items appear first.
    """
    return OrderingConstraint(
        constraint_type="ordering", practice_item_property="item_metadata.is_practice"
    )


@pytest.fixture
def sample_item_metadata() -> dict[UUID, dict[str, str | float | int]]:
    """Create sample metadata for 100 items.

    Returns
    -------
    dict[UUID, dict[str, str | float | int]]
        Metadata dict with various properties for testing.
    """
    items = [uuid4() for _ in range(100)]
    metadata: dict[UUID, dict[str, str | float | int]] = {}

    for i, item_id in enumerate(items):
        metadata[item_id] = {
            "value": float(i),
            "category": f"cat_{i % 5}",  # 5 categories
            "group": i % 10,  # 10 groups
            "score": i * 0.1,
        }

    return metadata


@pytest.fixture
def sample_quantile_metadata() -> dict[UUID, dict[str, float]]:
    """Create metadata with numeric values for quantile testing.

    Returns
    -------
    dict[UUID, dict[str, float]]
        Metadata with lm_prob values.
    """
    import numpy as np  # noqa: PLC0415

    items = [uuid4() for _ in range(100)]
    metadata: dict[UUID, dict[str, float]] = {}

    # Create values with known distribution
    for i, item_id in enumerate(items):
        metadata[item_id] = {
            "lm_prob": float(np.log(i + 1) / 10),  # Log-scaled values
            "value": float(i),
        }

    return metadata


@pytest.fixture
def partitioner_default() -> Any:
    """Create partitioner with default settings.

    Returns
    -------
    ListPartitioner
        Partitioner with default settings.
    """
    return ListPartitioner()


@pytest.fixture
def partitioner_seeded() -> Any:
    """Create partitioner with fixed seed.

    Returns
    -------
    ListPartitioner
        Partitioner with seed=42.
    """
    return ListPartitioner(random_seed=42)
