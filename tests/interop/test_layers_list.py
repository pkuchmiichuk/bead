"""Round-trip law tests for the list-composition overlap lenses."""

from __future__ import annotations

from uuid import uuid4

from lairs.records import judgment

from bead.interop.layers.list_lens import (
    EXPERIMENT_LIST_LAYERS,
    LIST_CONSTRAINT,
)
from bead.lists.constraints import (
    BalanceConstraint,
    GridDimension,
    GridStratificationConstraint,
    ListConstraint,
    QuantileBinning,
    UniquenessConstraint,
)
from bead.lists.experiment_list import ConstraintSatisfaction, ExperimentList


def _uniqueness() -> UniquenessConstraint:
    return UniquenessConstraint(
        constraint_type="uniqueness",
        property_expression="item['lemma']",
        allow_null=True,
        priority=3,
    )


def _balance() -> BalanceConstraint:
    return BalanceConstraint(
        constraint_type="balance",
        property_expression="item['condition']",
        target_counts={"grammatical": 4, "ungrammatical": 4},
        tolerance=0.2,
        priority=2,
    )


def _grid() -> GridStratificationConstraint:
    return GridStratificationConstraint(
        constraint_type="grid_stratification",
        dimensions=(
            GridDimension(
                property_expression="item['acceptability_score_diff']",
                binning=QuantileBinning(binning="quantile", n_quantiles=4),
            ),
            GridDimension(
                property_expression="item['length']",
                binning=QuantileBinning(binning="quantile", n_quantiles=3),
            ),
        ),
        items_per_cell=3,
        priority=5,
    )


class TestListConstraint:
    """ListConstraint <-> layers list constraint."""

    def test_uniqueness_roundtrip_exact(self) -> None:
        constraint = _uniqueness()
        view, complement = LIST_CONSTRAINT.forward(constraint)
        assert view.kind == "uniqueness"
        assert view.targetProperty == "item['lemma']"
        restored = LIST_CONSTRAINT.backward(view, complement)
        assert restored == constraint
        assert isinstance(restored, UniquenessConstraint)

    def test_balance_roundtrip_exact(self) -> None:
        constraint = _balance()
        view, complement = LIST_CONSTRAINT.forward(constraint)
        assert view.kind == "balance"
        assert view.targetProperty == "item['condition']"
        restored = LIST_CONSTRAINT.backward(view, complement)
        assert restored == constraint
        assert isinstance(restored, BalanceConstraint)

    def test_grid_stratification_roundtrip_exact(self) -> None:
        constraint = _grid()
        view, complement = LIST_CONSTRAINT.forward(constraint)
        assert view.kind == "grid_stratification"
        # grid stratification has no single property_expression
        assert view.targetProperty is None
        restored = LIST_CONSTRAINT.backward(view, complement)
        assert restored == constraint
        assert isinstance(restored, GridStratificationConstraint)

    def test_uniqueness_roundtrip_through_serialization(self) -> None:
        constraint = _uniqueness()
        view, complement = LIST_CONSTRAINT.forward(constraint)
        view2 = judgment.ListConstraint.model_validate_json(view.model_dump_json())
        assert LIST_CONSTRAINT.backward(view2, complement) == constraint

    def test_balance_roundtrip_through_serialization(self) -> None:
        constraint = _balance()
        view, complement = LIST_CONSTRAINT.forward(constraint)
        view2 = judgment.ListConstraint.model_validate_json(view.model_dump_json())
        assert LIST_CONSTRAINT.backward(view2, complement) == constraint

    def test_grid_roundtrip_through_serialization(self) -> None:
        constraint = _grid()
        view, complement = LIST_CONSTRAINT.forward(constraint)
        view2 = judgment.ListConstraint.model_validate_json(view.model_dump_json())
        restored = LIST_CONSTRAINT.backward(view2, complement)
        assert restored == constraint

    def test_backward_returns_union_root_type(self) -> None:
        constraint: ListConstraint = _balance()
        view, complement = LIST_CONSTRAINT.forward(constraint)
        restored = LIST_CONSTRAINT.backward(view, complement)
        assert type(restored) is BalanceConstraint


class TestExperimentListLayers:
    """ExperimentList <-> layers collection aggregate."""

    def _experiment_list(self) -> tuple[ExperimentList, tuple[str, ...]]:
        item_a = uuid4()
        item_b = uuid4()
        item_c = uuid4()
        uniqueness = _uniqueness()
        balance = _balance()
        experiment_list = ExperimentList(
            name="list_0",
            list_number=2,
            item_refs=(item_a, item_b, item_c),
            list_constraints=(uniqueness, balance),
            constraint_satisfaction=(
                ConstraintSatisfaction(constraint_id=uniqueness.id, satisfied=True),
                ConstraintSatisfaction(constraint_id=balance.id, satisfied=False),
            ),
            presentation_order=(item_c, item_a, item_b),
            list_metadata={"block": 1, "notes": ("a", "b")},
            balance_metrics={"entropy": 0.87, "balanced": True},
        )
        entry_refs = (str(item_a), str(item_b), str(item_c))
        return experiment_list, entry_refs

    def test_roundtrip_exact(self) -> None:
        experiment_list, _ = self._experiment_list()
        view, complement = EXPERIMENT_LIST_LAYERS.forward(experiment_list)
        assert view.collection.kind == "stimulus-pool"
        assert view.collection.name == "list_0"
        assert len(view.list_constraints) == 2
        restored = EXPERIMENT_LIST_LAYERS.backward(view, complement)
        assert restored == experiment_list

    def test_membership_count_and_order(self) -> None:
        experiment_list, entry_refs = self._experiment_list()
        view, _ = EXPERIMENT_LIST_LAYERS.forward(experiment_list)
        assert len(view.memberships) == len(experiment_list.item_refs)
        assert tuple(m.entryRef for m in view.memberships) == entry_refs
        assert tuple(m.ordinal for m in view.memberships) == (0, 1, 2)
        for membership in view.memberships:
            assert membership.collectionRef == str(experiment_list.id)

    def test_roundtrip_through_serialization(self) -> None:
        experiment_list, _ = self._experiment_list()
        view, complement = EXPERIMENT_LIST_LAYERS.forward(experiment_list)
        collection2 = view.collection.__class__.model_validate_json(
            view.collection.model_dump_json()
        )
        view2 = view.with_(collection=collection2)
        restored = EXPERIMENT_LIST_LAYERS.backward(view2, complement)
        assert restored == experiment_list

    def test_empty_roundtrip(self) -> None:
        experiment_list = ExperimentList(name="empty", list_number=0)
        view, complement = EXPERIMENT_LIST_LAYERS.forward(experiment_list)
        assert view.memberships == ()
        assert view.list_constraints == ()
        restored = EXPERIMENT_LIST_LAYERS.backward(view, complement)
        assert restored == experiment_list

    def test_presentation_order_none_recovered(self) -> None:
        item_a = uuid4()
        experiment_list = ExperimentList(
            name="list_1",
            list_number=1,
            item_refs=(item_a,),
        )
        view, complement = EXPERIMENT_LIST_LAYERS.forward(experiment_list)
        restored = EXPERIMENT_LIST_LAYERS.backward(view, complement)
        assert restored.presentation_order is None
        assert restored == experiment_list
