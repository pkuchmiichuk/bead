# bead.lists

Stage 4 of the bead pipeline: list partitioning with constraint satisfaction.

Items can be stratified along several dimensions at once. Each dimension of a
`GridStratificationConstraint` declares a binning strategy (`QuantileBinning`,
`EqualWidthBinning`, `ThresholdBinning`, or `StdDevBinning` for continuous
values; `CategoricalBinning` for finite discrete values), and items are
distributed uniformly across the cells of the resulting grid. The
`assign_grid_cells` and `assign_grid_cells_by_uuid` helpers compute the grid
cell for each item; one-dimensional quantile stratification via
`assign_quantiles` is the single-dimension quantile case of the same engine.

## Core Classes

::: bead.lists.experiment_list
    options:
      show_root_heading: true
      show_source: false

::: bead.lists.list_collection
    options:
      show_root_heading: true
      show_source: false

## Constraints

::: bead.lists.constraints
    options:
      show_root_heading: true
      show_source: false

## Partitioning

::: bead.lists.partitioner
    options:
      show_root_heading: true
      show_source: false

::: bead.lists.stratification
    options:
      show_root_heading: true
      show_source: false

## Balancing

::: bead.lists.balancer
    options:
      show_root_heading: true
      show_source: false
