"""List construction module for experimental list partitioning.

Provides data models for organizing experimental items into balanced lists
for presentation to participants. Includes ExperimentList, ListCollection,
constraint types (uniqueness, balance, quantile, grid stratification, size,
diversity, ordering), and the stratification engine that bins items along one
or more dimensions.
"""

from bead.lists.constraints import (
    BalanceConstraint,
    BinningSpec,
    CategoricalBinning,
    DiversityConstraint,
    EqualWidthBinning,
    GridDimension,
    GridStratificationConstraint,
    ListConstraint,
    OrderingConstraint,
    QuantileBinning,
    QuantileConstraint,
    SizeConstraint,
    StdDevBinning,
    ThresholdBinning,
    UniquenessConstraint,
)
from bead.lists.experiment_list import ExperimentList
from bead.lists.list_collection import ListCollection
from bead.lists.stratification import (
    assign_grid_cells,
    assign_grid_cells_by_uuid,
    assign_quantiles,
    assign_quantiles_by_uuid,
    flatten_cell,
    grid_shape,
    unflatten_cell,
)

__all__ = [
    "ExperimentList",
    "ListCollection",
    "ListConstraint",
    "UniquenessConstraint",
    "BalanceConstraint",
    "QuantileConstraint",
    "GridStratificationConstraint",
    "GridDimension",
    "BinningSpec",
    "QuantileBinning",
    "EqualWidthBinning",
    "ThresholdBinning",
    "StdDevBinning",
    "CategoricalBinning",
    "DiversityConstraint",
    "SizeConstraint",
    "OrderingConstraint",
    "assign_quantiles",
    "assign_quantiles_by_uuid",
    "assign_grid_cells",
    "assign_grid_cells_by_uuid",
    "grid_shape",
    "flatten_cell",
    "unflatten_cell",
]
