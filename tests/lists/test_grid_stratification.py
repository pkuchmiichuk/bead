"""Tests for N-dimensional mixed-variable grid stratification."""

from __future__ import annotations

from uuid import uuid4

import didactic.api as dx
import pytest

from bead.lists.balancer import QuantileBalancer
from bead.lists.constraints import (
    CategoricalBinning,
    EqualWidthBinning,
    GridDimension,
    GridStratificationConstraint,
    QuantileBinning,
    StdDevBinning,
    ThresholdBinning,
)
from bead.lists.partitioner import ListPartitioner
from bead.lists.stratification import (
    assign_grid_cells,
    assign_grid_cells_by_uuid,
    assign_quantiles,
    flatten_cell,
    grid_shape,
    unflatten_cell,
)


def _q(n: int) -> QuantileBinning:
    return QuantileBinning(binning="quantile", n_quantiles=n)


def _c(**kwargs: object) -> CategoricalBinning:
    return CategoricalBinning(binning="categorical", **kwargs)  # type: ignore[arg-type]


class TestBinningStrategies:
    """Each binning strategy maps values to the expected bin indices."""

    def test_quantile_equal_frequency(self) -> None:
        items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
        result = assign_grid_cells(items, [lambda x: float(x)], [_q(4)])
        assert result[1] == (0,)
        assert result[10] == (3,)
        assert {c[0] for c in result.values()} == {0, 1, 2, 3}

    def test_equal_width(self) -> None:
        items = [0.0, 0.4, 0.6, 1.0]
        binning = EqualWidthBinning(
            binning="equal_width", n_bins=2, range_min=0.0, range_max=1.0
        )
        result = assign_grid_cells(items, [lambda x: x], [binning])
        assert result[0.0] == (0,)
        assert result[0.4] == (0,)
        assert result[0.6] == (1,)
        assert result[1.0] == (1,)

    def test_threshold(self) -> None:
        items = [0.2, 0.5, 0.8]
        binning = ThresholdBinning(binning="threshold", edges=(0.5,))
        result = assign_grid_cells(items, [lambda x: x], [binning])
        assert result[0.2] == (0,)
        assert result[0.5] == (1,)  # at the edge -> upper bin
        assert result[0.8] == (1,)

    def test_threshold_multiple_edges(self) -> None:
        items = [-1.0, 0.5, 1.5, 3.0]
        binning = ThresholdBinning(binning="threshold", edges=(0.0, 1.0, 2.0))
        result = assign_grid_cells(items, [lambda x: x], [binning])
        assert result[-1.0] == (0,)
        assert result[0.5] == (1,)
        assert result[1.5] == (2,)
        assert result[3.0] == (3,)

    def test_stddev(self) -> None:
        items = [0.0, 1.0, 2.0, 3.0, 4.0]
        binning = StdDevBinning(binning="stddev", k_values=(-1.0, 1.0))
        result = assign_grid_cells(items, [lambda x: x], [binning])
        # mean=2, sd~1.41; below mean-sd -> 0, around mean -> 1, above mean+sd -> 2
        assert result[0.0] == (0,)
        assert result[2.0] == (1,)
        assert result[4.0] == (2,)

    def test_categorical_declared(self) -> None:
        items = ["a", "b", "a", "c"]
        binning = _c(categories=("a", "b"), include_other=True)
        result = assign_grid_cells(items, [lambda x: x], [binning])
        assert result["a"] == (0,)
        assert result["b"] == (1,)
        assert result["c"] == (2,)  # catch-all bin

    def test_categorical_no_other_raises(self) -> None:
        items = ["a", "b", "c"]
        binning = _c(categories=("a", "b"), include_other=False)
        with pytest.raises(ValueError, match="outside the declared categories"):
            assign_grid_cells(items, [lambda x: x], [binning])

    def test_categorical_auto_discovery(self) -> None:
        items = ["z", "a", "m", "a"]
        result = assign_grid_cells(items, [lambda x: x], [_c()])
        # categories discovered and sorted: a=0, m=1, z=2
        assert result["a"] == (0,)
        assert result["m"] == (1,)
        assert result["z"] == (2,)


class TestAssignGridCells:
    """Grid assignment composes dimensions and validates inputs."""

    def test_reduces_to_quantiles_at_n1(self) -> None:
        items = list(range(100))
        grid = assign_grid_cells(items, [lambda x: float(x)], [_q(4)])
        quant = assign_quantiles(items, lambda x: float(x), n_quantiles=4)
        assert all(grid[i][0] == quant[i] for i in items)

    def test_mixed_continuous_and_discrete(self) -> None:
        data = [(0.1, "a"), (0.9, "a"), (0.2, "b"), (0.8, "b")]
        getters = [lambda x: x[0], lambda x: x[1]]
        binnings = [_q(2), _c()]
        result = assign_grid_cells(list(data), getters, binnings)
        assert result[(0.1, "a")] == (0, 0)
        assert result[(0.9, "a")] == (1, 0)
        assert result[(0.2, "b")] == (0, 1)
        assert result[(0.8, "b")] == (1, 1)

    def test_stratify_by_groups_continuous(self) -> None:
        data = [(10, "A"), (20, "A"), (30, "A"), (40, "A"), (5, "B"), (35, "B")]
        result = assign_grid_cells(
            list(data),
            [lambda x: float(x[0])],
            [_q(2)],
            stratify_by=lambda x: x[1],
        )
        # within each group both quantiles appear
        a = {result[item][0] for item in data if item[1] == "A"}
        b = {result[item][0] for item in data if item[1] == "B"}
        assert a == {0, 1}
        assert b == {0, 1}

    def test_empty_items_raises(self) -> None:
        with pytest.raises(ValueError, match="items list cannot be empty"):
            assign_grid_cells([], [lambda x: x], [_q(2)])

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            assign_grid_cells([1, 2], [lambda x: float(x)], [_q(2), _q(2)])


class TestFlattenCell:
    """Flatten/unflatten round-trips with heterogeneous cardinalities."""

    def test_round_trip(self) -> None:
        shape = (3, 4, 2)
        seen = set()
        for i in range(3):
            for j in range(4):
                for k in range(2):
                    cell = flatten_cell((i, j, k), shape)
                    assert unflatten_cell(cell, shape) == (i, j, k)
                    seen.add(cell)
        # bijection onto 0..23
        assert seen == set(range(24))

    def test_length_mismatch_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            flatten_cell((1, 2), (3,))


class TestAssignGridCellsByUUID:
    """The UUID convenience returns flattened cell ids and validates inputs."""

    def test_basic(self) -> None:
        uuids = [uuid4() for _ in range(8)]
        meta = {
            uid: {
                "acc": float(i) / 8,
                "lm": float(7 - i) / 8,
                "pair_type": "x" if i % 2 else "y",
            }
            for i, uid in enumerate(uuids)
        }
        cells = assign_grid_cells_by_uuid(
            uuids,
            meta,
            ["acc", "lm", "pair_type"],
            [_q(2), _q(2), _c(categories=("x", "y"))],
        )
        assert set(cells.keys()) == set(uuids)
        assert all(0 <= c < 2 * 2 * 2 for c in cells.values())

    def test_shape_matches_grid_shape(self) -> None:
        uuids = [uuid4() for _ in range(6)]
        meta = {uid: {"a": float(i)} for i, uid in enumerate(uuids)}
        getters = [lambda uid: meta[uid]["a"]]
        shape = grid_shape(uuids, getters, [_q(3)])
        assert shape == (3,)

    def test_missing_property_raises(self) -> None:
        uuids = [uuid4() for _ in range(4)]
        meta = {uid: {"a": float(i)} for i, uid in enumerate(uuids)}
        with pytest.raises(ValueError, match="Property 'b' not found"):
            assign_grid_cells_by_uuid(uuids, meta, ["b"], [_q(2)])

    def test_missing_uuid_raises(self) -> None:
        uuids = [uuid4() for _ in range(4)]
        meta = {uid: {"a": float(i)} for i, uid in enumerate(uuids[:-1])}
        with pytest.raises(KeyError):
            assign_grid_cells_by_uuid(uuids, meta, ["a"], [_q(2)])

    def test_missing_stratify_key_raises(self) -> None:
        uuids = [uuid4() for _ in range(4)]
        meta = {uid: {"a": float(i)} for i, uid in enumerate(uuids)}
        with pytest.raises(ValueError, match="Stratification key"):
            assign_grid_cells_by_uuid(
                uuids, meta, ["a"], [_q(2)], stratify_by_key="grp"
            )


class TestGridStratificationConstraint:
    """The constraint validates its dimensions and parameters."""

    def test_valid(self) -> None:
        dim = GridDimension(property_expression="item['a']", binning=_q(3))
        constraint = GridStratificationConstraint(
            constraint_type="grid_stratification",
            dimensions=(dim,),
            items_per_cell=2,
        )
        assert constraint.constraint_type == "grid_stratification"
        assert len(constraint.dimensions) == 1

    def test_empty_dimensions_raises(self) -> None:
        with pytest.raises((ValueError, dx.ValidationError)):
            GridStratificationConstraint(
                constraint_type="grid_stratification", dimensions=()
            )

    def test_items_per_cell_axiom(self) -> None:
        dim = GridDimension(property_expression="item['a']", binning=_q(2))
        with pytest.raises((ValueError, dx.ValidationError)):
            GridStratificationConstraint(
                constraint_type="grid_stratification",
                dimensions=(dim,),
                items_per_cell=0,
            )

    def test_threshold_edges_must_increase(self) -> None:
        with pytest.raises((ValueError, dx.ValidationError)):
            ThresholdBinning(binning="threshold", edges=(1.0, 0.5))


class TestBalancerByCell:
    """The balancer spreads each grid cell uniformly across lists."""

    def test_distribute_across_cells(self) -> None:
        uuids = [uuid4() for _ in range(40)]
        # 4 cells, 10 items each
        cell_of = {uid: i % 4 for i, uid in enumerate(uuids)}
        balancer = QuantileBalancer(n_quantiles=2, random_seed=0)
        lists = balancer.balance_by_cell(
            uuids,
            lambda uid: cell_of[uid],
            n_cells=4,
            n_lists=2,
            items_per_cell_per_list=5,
        )
        assert len(lists) == 2
        # each list gets 5 items per cell * 4 cells = 20
        assert all(len(lst) == 20 for lst in lists)

    def test_invalid_n_cells(self) -> None:
        balancer = QuantileBalancer(n_quantiles=2)
        with pytest.raises(ValueError, match="n_cells must be >= 1"):
            balancer.balance_by_cell([], lambda uid: 0, 0, 2, 1)


class TestPartitionerGrid:
    """The partitioner uses the grid constraint when present."""

    def test_grid_partition(self) -> None:
        items = [uuid4() for _ in range(40)]
        meta = {
            uid: {
                "metadata": {
                    "acc": float(i % 10) / 10.0,
                    "lm": float((i * 3) % 10) / 10.0,
                    "pair_type": "same" if i % 2 == 0 else "diff",
                }
            }
            for i, uid in enumerate(items)
        }
        constraint = GridStratificationConstraint(
            constraint_type="grid_stratification",
            dimensions=(
                GridDimension(
                    property_expression="item['metadata']['acc']", binning=_q(2)
                ),
                GridDimension(
                    property_expression="item['metadata']['lm']", binning=_q(2)
                ),
                GridDimension(
                    property_expression="item['metadata']['pair_type']",
                    binning=_c(categories=("same", "diff")),
                ),
            ),
            items_per_cell=1,
        )
        partitioner = ListPartitioner(random_seed=42)
        lists = partitioner.partition(
            items,
            n_lists=4,
            constraints=[constraint],
            strategy="stratified",
            metadata=meta,
        )
        assert len(lists) == 4
        # no item appears in two lists
        all_refs = [ref for lst in lists for ref in lst.item_refs]
        assert len(all_refs) == len(set(all_refs))
