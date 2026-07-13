"""Lenses between bead list-composition models and layers judgment records.

Maps bead's experimental list models to their canonical
:mod:`lairs.records` counterparts:

- ``ListConstraint`` <-> a layers ``listConstraint`` (``judgment.ListConstraint``)
- ``ExperimentList`` <-> a layers ``collection`` with its ``collectionMembership``
  records and per-list ``listConstraint`` records, bundled as the bead-side
  :class:`ExperimentListLayers` aggregate view

The bead ``ListConstraint`` union is far richer than the four scalar fields of the
layers ``listConstraint`` record, so the lens carries the entire bead constraint
(serialized with ``model_dump_json``) in the lens complement and reconstructs the
exact subclass with ``ListConstraint.model_validate_json`` on the way back. The
``ExperimentList`` lens keeps the bead framework identity, the numeric list id, the
ordered item references, the presentation order, the list and balance metadata, the
per-constraint satisfaction records, and the per-constraint complements in its own
complement, so reconstruction is exact.
"""

from __future__ import annotations

from uuid import UUID

import didactic.api as dx
from lairs.records import defs, judgment, resource

from bead.data.base import JsonValue
from bead.interop.layers._convert import (
    apply_identity,
    dumps_meta,
    identity_of,
    j_int,
    j_list,
    j_obj,
    j_str,
    loads_meta,
)
from bead.lists.constraints import (
    BalanceConstraint,
    ConditionalUniquenessConstraint,
    DiversityConstraint,
    GroupedQuantileConstraint,
    ListConstraint,
    QuantileConstraint,
    UniquenessConstraint,
)
from bead.lists.experiment_list import ConstraintSatisfaction, ExperimentList

# An experiment list is a curated pool of stimuli; "stimulus-pool" is the
# layers collection kind documented for exactly this (the bead list number and
# identity ride in the lens complement).
_EXPERIMENT_LIST_KIND = "stimulus-pool"


def _constraint_kind(constraint: ListConstraint) -> str:
    """Read a list constraint's discriminator tag as a plain string."""
    value = constraint.model_dump().get("constraint_type")
    return value if isinstance(value, str) else ""


def _target_property(constraint: ListConstraint) -> str | None:
    """Return a constraint's ``property_expression`` when its variant has one."""
    if isinstance(
        constraint,
        (
            UniquenessConstraint,
            ConditionalUniquenessConstraint,
            BalanceConstraint,
            QuantileConstraint,
            GroupedQuantileConstraint,
            DiversityConstraint,
        ),
    ):
        return constraint.property_expression
    return None


class ListConstraintLens(dx.Lens[ListConstraint, judgment.ListConstraint, JsonValue]):
    """Lossless lens ``ListConstraint <-> (layers list constraint, complement)``.

    The layers ``listConstraint`` record (``judgment.ListConstraint``) keeps only a
    ``kind`` slug, an optional ``targetProperty``, an optional formal ``constraint``
    expression, and an optional ``parameters`` map. The bead union carries many more
    variant-specific fields, so the projection is lossy on its own; the complement
    stores the entire bead constraint as ``model_dump_json`` and reconstruction
    resolves the exact subclass through the discriminated-union root.
    """

    def forward(
        self, constraint: ListConstraint
    ) -> tuple[judgment.ListConstraint, JsonValue]:
        """Project a list constraint to a layers list constraint and complement."""
        kind = _constraint_kind(constraint)
        target_property = _target_property(constraint)
        view = judgment.ListConstraint(
            kind=kind,
            targetProperty=target_property,
            constraint=defs.Constraint(
                expression=target_property if target_property is not None else kind,
                description=kind,
            ),
        )
        complement: JsonValue = {"bead": constraint.model_dump_json()}
        return view, complement

    def backward(
        self, view: judgment.ListConstraint, complement: JsonValue
    ) -> ListConstraint:
        """Reconstruct a list constraint from its complement.

        The layers view is purely a projection; the exact bead constraint comes
        from the serialized model in the complement, whose discriminated-union root
        resolves the correct subclass.
        """
        comp = j_obj(complement)
        return ListConstraint.model_validate_json(j_str(comp["bead"]))


LIST_CONSTRAINT = ListConstraintLens()


class ExperimentListLayers(dx.Model):
    """A layers view of an experiment list.

    Attributes
    ----------
    collection : resource.Collection
        The list itself, projected to a layers ``collection`` of kind
        ``stimulus-pool``.
    memberships : tuple[resource.CollectionMembership, ...]
        One membership per item reference, in presentation-independent item order,
        carrying the item ordinal.
    list_constraints : tuple[judgment.ListConstraint, ...]
        The per-list constraints, projected via :data:`LIST_CONSTRAINT`.
    """

    collection: dx.Embed[resource.Collection] = dx.field()
    memberships: tuple[dx.Embed[resource.CollectionMembership], ...] = dx.field(
        default=()
    )
    list_constraints: tuple[dx.Embed[judgment.ListConstraint], ...] = dx.field(
        default=()
    )


class ExperimentListLens(dx.Lens[ExperimentList, ExperimentListLayers, JsonValue]):
    """Lossless lens ``ExperimentList <-> (layers collection aggregate, complement)``.

    The layers view bundles a ``collection`` (the list), one
    ``collectionMembership`` per item reference (preserving order via the membership
    ordinal), and the per-list ``listConstraint`` records. The bead-only remainder
    (framework identity, the numeric list id, the ordered item references, the
    presentation order, the list and balance metadata, the constraint-satisfaction
    records, and the per-constraint complements) travels in the lens complement, so
    a round-trip reconstructs the original list exactly.
    """

    def forward(
        self, experiment_list: ExperimentList
    ) -> tuple[ExperimentListLayers, JsonValue]:
        """Project an experiment list to a layers collection aggregate."""
        memberships: list[resource.CollectionMembership] = []
        for ordinal, item_id in enumerate(experiment_list.item_refs):
            memberships.append(
                resource.CollectionMembership(
                    collectionRef=str(experiment_list.id),
                    entryRef=str(item_id),
                    createdAt=experiment_list.created_at,
                    ordinal=ordinal,
                )
            )
        constraint_views: list[judgment.ListConstraint] = []
        constraint_complements: list[JsonValue] = []
        for constraint in experiment_list.list_constraints:
            constraint_view, constraint_complement = LIST_CONSTRAINT.forward(constraint)
            constraint_views.append(constraint_view)
            constraint_complements.append(constraint_complement)
        view = ExperimentListLayers(
            collection=resource.Collection(
                name=experiment_list.name,
                kind=_EXPERIMENT_LIST_KIND,
                createdAt=experiment_list.created_at,
            ),
            memberships=tuple(memberships),
            list_constraints=tuple(constraint_views),
        )
        presentation_order: JsonValue = (
            tuple(str(item_id) for item_id in experiment_list.presentation_order)
            if experiment_list.presentation_order is not None
            else None
        )
        complement: JsonValue = {
            "identity": identity_of(experiment_list),
            "list_number": experiment_list.list_number,
            "item_refs": tuple(str(item_id) for item_id in experiment_list.item_refs),
            "presentation_order": presentation_order,
            "list_metadata": dumps_meta(experiment_list.list_metadata),
            "balance_metrics": dumps_meta(experiment_list.balance_metrics),
            "constraint_satisfaction": tuple(
                record.model_dump_json()
                for record in experiment_list.constraint_satisfaction
            ),
            "constraint_complements": tuple(constraint_complements),
        }
        return view, complement

    def backward(
        self, view: ExperimentListLayers, complement: JsonValue
    ) -> ExperimentList:
        """Reconstruct an experiment list from its layers aggregate and complement."""
        comp = j_obj(complement)
        item_refs = tuple(UUID(j_str(value)) for value in j_list(comp["item_refs"]))
        order_value = comp["presentation_order"]
        presentation_order = (
            None
            if order_value is None
            else tuple(UUID(j_str(value)) for value in j_list(order_value))
        )
        constraint_complements = j_list(comp["constraint_complements"])
        list_constraints = tuple(
            LIST_CONSTRAINT.backward(constraint_view, constraint_complement)
            for constraint_view, constraint_complement in zip(
                view.list_constraints, constraint_complements, strict=True
            )
        )
        constraint_satisfaction = tuple(
            ConstraintSatisfaction.model_validate_json(j_str(value))
            for value in j_list(comp["constraint_satisfaction"])
        )
        experiment_list = ExperimentList(
            name=view.collection.name,
            list_number=j_int(comp["list_number"]),
            item_refs=item_refs,
            list_constraints=list_constraints,
            constraint_satisfaction=constraint_satisfaction,
            presentation_order=presentation_order,
            list_metadata=loads_meta(comp["list_metadata"]),
            balance_metrics=loads_meta(comp["balance_metrics"]),
        )
        return apply_identity(experiment_list, comp["identity"])


EXPERIMENT_LIST_LAYERS = ExperimentListLens()
