#!/usr/bin/env python3
"""Generate experiment lists from 2AFC pairs.

This script loads 2AFC pairs and partitions them into balanced experimental lists
according to the configuration in config.yaml.
"""

from __future__ import annotations

import sys
import traceback
from pathlib import Path
from uuid import UUID, uuid4

import layers_io
import yaml

from bead.cli.display import (
    console,
    create_summary_table,
    print_error,
    print_header,
    print_success,
    print_warning,
)
from bead.items.item import Item
from bead.lists import (
    BalanceConstraint,
    CategoricalBinning,
    DiversityConstraint,
    EqualWidthBinning,
    GridDimension,
    GridStratificationConstraint,
    ListCollection,
    QuantileBinning,
    StdDevBinning,
    ThresholdBinning,
    UniquenessConstraint,
)
from bead.lists.constraints import (
    BatchBalanceConstraint,
    BatchCoverageConstraint,
    BatchDiversityConstraint,
    BatchMinOccurrenceConstraint,
    BinningSpec,
)
from bead.lists.partitioner import ListPartitioner


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_2afc_pairs(pairs_path: Path) -> tuple[list[Item], dict[UUID, dict]]:
    """Load 2AFC pairs and extract metadata for constraint checking.

    Returns
    -------
    tuple[list[Item], dict[UUID, dict]]
        Tuple of (items list, metadata dict keyed by item UUID).
    """
    with console.status(f"[bold]Loading 2AFC pairs from {pairs_path}...[/bold]"):
        items = []
        metadata_dict = {}

        with open(pairs_path) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = Item.model_validate_json(line)
                items.append(item)

                # Build metadata dict for constraint checking
                # The DSL expects: item.metadata.pair_type format
                # So we need: {"metadata": {item_metadata values}}
                metadata_dict[item.id] = {"metadata": dict(item.item_metadata)}

    print_success(f"Loaded {len(items)} 2AFC pairs")
    return items, metadata_dict


def build_binning(spec: dict) -> BinningSpec:
    """Build a binning strategy from a config dict.

    Parameters
    ----------
    spec : dict
        Mapping with a ``type`` key (quantile, equal_width, threshold, stddev,
        or categorical) and strategy-specific parameters.

    Returns
    -------
    BinningSpec
        The constructed binning strategy.
    """
    binning_type = spec["type"]
    if binning_type == "quantile":
        return QuantileBinning(
            binning="quantile", n_quantiles=spec.get("n_quantiles", 5)
        )
    if binning_type == "equal_width":
        return EqualWidthBinning(
            binning="equal_width",
            n_bins=spec.get("n_bins", 5),
            range_min=spec.get("range_min"),
            range_max=spec.get("range_max"),
        )
    if binning_type == "threshold":
        return ThresholdBinning(binning="threshold", edges=tuple(spec["edges"]))
    if binning_type == "stddev":
        return StdDevBinning(
            binning="stddev", k_values=tuple(spec.get("k_values", (-1.0, 0.0, 1.0)))
        )
    if binning_type == "categorical":
        categories = spec.get("categories")
        return CategoricalBinning(
            binning="categorical",
            categories=tuple(categories) if categories else None,
            include_other=spec.get("include_other", False),
        )
    raise ValueError(f"Unknown binning type: {binning_type}")


def build_constraints(config: dict) -> tuple[list, list]:
    """Build list and batch constraints from config.

    Returns
    -------
    tuple[list, list]
        Tuple of (list_constraints, batch_constraints).
    """
    list_constraints = []
    batch_constraints = []

    # Parse list constraints from config
    for constraint_spec in config.get("lists", {}).get("constraints", []):
        constraint_type = constraint_spec["type"]

        if constraint_type == "balance":
            list_constraints.append(
                BalanceConstraint(
                    constraint_type="balance",
                    property_expression=constraint_spec["property_expression"],
                    target_counts=constraint_spec.get("target_counts"),
                )
            )
        elif constraint_type == "uniqueness":
            list_constraints.append(
                UniquenessConstraint(
                    constraint_type="uniqueness",
                    property_expression=constraint_spec["property_expression"],
                )
            )
        elif constraint_type == "grid_stratification":
            list_constraints.append(
                GridStratificationConstraint(
                    constraint_type="grid_stratification",
                    dimensions=tuple(
                        GridDimension(
                            property_expression=dim["property_expression"],
                            binning=build_binning(dim["binning"]),
                        )
                        for dim in constraint_spec["dimensions"]
                    ),
                    group_by_expression=constraint_spec.get("group_by_expression"),
                    items_per_cell=constraint_spec.get("items_per_cell", 2),
                )
            )
        elif constraint_type == "diversity":
            list_constraints.append(
                DiversityConstraint(
                    constraint_type="diversity",
                    property_expression=constraint_spec["property_expression"],
                    min_unique_values=constraint_spec["min_unique_values"],
                )
            )

    # Parse batch constraints from config
    for constraint_spec in config.get("lists", {}).get("batch_constraints", []):
        constraint_type = constraint_spec["type"]

        if constraint_type == "coverage":
            batch_constraints.append(
                BatchCoverageConstraint(
                    constraint_type="coverage",
                    property_expression=constraint_spec["property_expression"],
                    target_values=tuple(constraint_spec["target_values"]),
                    min_coverage=constraint_spec.get("min_coverage", 1.0),
                )
            )
        elif constraint_type == "balance":
            batch_constraints.append(
                BatchBalanceConstraint(
                    constraint_type="balance",
                    property_expression=constraint_spec["property_expression"],
                    target_distribution=constraint_spec.get("target_distribution", {}),
                    tolerance=constraint_spec.get("tolerance", 0.05),
                )
            )
        elif constraint_type == "min_occurrence":
            batch_constraints.append(
                BatchMinOccurrenceConstraint(
                    constraint_type="min_occurrence",
                    property_expression=constraint_spec["property_expression"],
                    min_occurrences=constraint_spec["min_occurrences"],
                )
            )
        elif constraint_type == "diversity":
            batch_constraints.append(
                BatchDiversityConstraint(
                    constraint_type="diversity",
                    property_expression=constraint_spec["property_expression"],
                    max_lists_per_value=constraint_spec.get("max_lists_per_value"),
                )
            )

    return list_constraints, batch_constraints


def main() -> None:
    """Generate experiment lists from 2AFC pairs."""
    # Determine base directory
    base_dir = Path(__file__).parent
    config_path = base_dir / "config.yaml"

    print_header("Experiment List Generation")
    console.print(f"Base directory: [cyan]{base_dir}[/cyan]")
    console.print(f"Configuration: [cyan]{config_path}[/cyan]\n")

    # Load configuration
    print_header("[1/5] Loading Configuration")
    config = load_config(config_path)
    list_config = config["lists"]
    n_lists = list_config["n_lists"]
    items_per_list = list_config["items_per_list"]
    strategy = list_config.get("strategy", "balanced")

    print_success("Configuration loaded")
    console.print(f"  - Lists to generate: [cyan]{n_lists}[/cyan]")
    console.print(f"  - Items per list: [cyan]{items_per_list}[/cyan]")
    console.print(f"  - Strategy: [cyan]{strategy}[/cyan]\n")

    # Load 2AFC pairs, preferring the canonical layers fragment
    print_header("[2/5] Loading 2AFC Pairs")
    fragment_path = base_dir / config["paths"].get("2afc_pairs_fragment", "")
    pairs_path = base_dir / config["paths"]["2afc_pairs"]
    if config["paths"].get("2afc_pairs_fragment") and fragment_path.exists():
        print_success(f"Loading from layers fragment {fragment_path.name}")
        items = layers_io.read_items(fragment_path)
        metadata_dict = {
            item.id: {"metadata": dict(item.item_metadata)} for item in items
        }
    else:
        items, metadata_dict = load_2afc_pairs(pairs_path)

    # Build constraints
    console.print()
    print_header("[3/5] Building Constraints")
    list_constraints, batch_constraints = build_constraints(config)
    print_success(f"Built {len(list_constraints)} list constraints")
    print_success(f"Built {len(batch_constraints)} batch constraints\n")

    # Partition items into lists
    print_header("[4/5] Partitioning Items")
    partitioner = ListPartitioner(random_seed=42)
    item_uuids = [item.id for item in items]

    try:
        # Use the appropriate partitioning method based on batch constraints
        if batch_constraints:
            with console.status("[bold]Using batch-constrained partitioning...[/bold]"):
                # Select total items needed
                total_items_needed = n_lists * items_per_list
                selected_uuids = item_uuids[:total_items_needed]

                experiment_lists = partitioner.partition_with_batch_constraints(
                    items=selected_uuids,
                    n_lists=n_lists,
                    list_constraints=list_constraints,
                    batch_constraints=batch_constraints,
                    metadata=metadata_dict,
                )
        else:
            with console.status("[bold]Using standard partitioning...[/bold]"):
                # For standard partitioning, we need to select subset first
                total_items_needed = n_lists * items_per_list
                selected_uuids = item_uuids[:total_items_needed]

                experiment_lists = partitioner.partition(
                    items=selected_uuids,
                    n_lists=n_lists,
                    constraints=list_constraints,
                    strategy=strategy,
                    metadata=metadata_dict,
                )

        print_success(f"Created {len(experiment_lists)} lists")
        for i, exp_list in enumerate(experiment_lists):
            console.print(
                f"  - List {i + 1}: [cyan]{len(exp_list.item_refs)}[/cyan] items"
            )

    except Exception as e:
        print_error(f"Error during partitioning: {e}")
        traceback.print_exc()
        sys.exit(1)

    # Save lists
    console.print()
    print_header("[5/5] Saving Lists")
    output_path = base_dir / config["paths"]["experiment_lists"]
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Create list collection
    list_collection = ListCollection(
        name="argument_structure_lists",
        source_items_id=uuid4(),  # Generate a UUID for the source items
        lists=experiment_lists,
        partitioning_strategy=strategy,
        partitioning_config={
            "n_lists": len(experiment_lists),
            "items_per_list": items_per_list,
            "n_list_constraints": len(list_constraints),
            "n_batch_constraints": len(batch_constraints),
        },
        partitioning_stats={
            "total_items": sum(len(lst.item_refs) for lst in experiment_lists),
        },
    )

    # Save as JSONL
    list_collection.to_jsonl(output_path)

    print_success(f"Saved {len(experiment_lists)} lists to {output_path}")

    # also persist the lists as layers collection aggregates
    fragment_rel = config["paths"].get("experiment_lists_fragment")
    if fragment_rel:
        fragment_path = base_dir / fragment_rel
        layers_io.write_experiment_lists_layers(list(experiment_lists), fragment_path)
        print_success(f"Wrote layers list collections to {fragment_path.name}")
    console.print()

    # Print summary table
    print_header("Summary")
    table = create_summary_table(
        {
            "Total lists": str(len(experiment_lists)),
            "Total items distributed": str(
                sum(len(lst.item_refs) for lst in experiment_lists)
            ),
            "Items per list": str([len(lst.item_refs) for lst in experiment_lists]),
        }
    )
    console.print(table)


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_warning("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
