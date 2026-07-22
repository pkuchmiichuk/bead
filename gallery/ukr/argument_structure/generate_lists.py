#!/usr/bin/env python3
"""Partition the 2AFC pairs into experiment lists.

Each list is one version of the experiment: a participant judges its pairs, and
several participants can take the same list. Pairs are drawn from the pool in
the proportions the balance constraint declares, then distributed so that no
verb repeats within a list and difficulty stays spread.

Lists reference pairs by UUID, so the pair text lives only in the 2AFC file.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from random import Random
from uuid import uuid4

import yaml

from bead.cli.display import (
    display_file_stats,
    print_header,
    print_info,
    print_success,
    print_warning,
)
from bead.items.item import Item
from bead.lists.constraints import (
    BalanceConstraint,
    BatchConstraint,
    BatchCoverageConstraint,
    BatchDiversityConstraint,
    BatchMinOccurrenceConstraint,
    DiversityConstraint,
    ListConstraint,
    UniquenessConstraint,
)
from bead.lists.list_collection import ListCollection
from bead.lists.partitioner import ListPartitioner, MetadataDict

BASE_DIR = Path(__file__).parent


def load_config(path: Path) -> dict:
    """Load the YAML configuration file.

    Parameters
    ----------
    path : Path
        Path to the configuration file.

    Returns
    -------
    dict
        Parsed configuration.
    """
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_pairs(path: Path) -> list[Item]:
    """Load 2AFC pairs from JSONL.

    Parameters
    ----------
    path : Path
        Path to the pairs file.

    Returns
    -------
    list[Item]
        The loaded pairs.
    """
    with path.open(encoding="utf-8") as f:
        return [Item.model_validate_json(line) for line in f if line.strip()]


def build_list_constraints(specs: list[dict]) -> list[ListConstraint]:
    """Build per-list constraints from their config entries.

    Parameters
    ----------
    specs : list[dict]
        Constraint entries, each with a ``type`` key.

    Returns
    -------
    list[ListConstraint]
        The constructed constraints.

    Raises
    ------
    ValueError
        If a constraint type is unknown.
    """
    constraints: list[ListConstraint] = []
    for spec in specs:
        kind = spec["type"]
        if kind == "balance":
            constraints.append(
                BalanceConstraint(
                    property_expression=spec["property_expression"],
                    target_counts=spec.get("target_counts"),
                )
            )
        elif kind == "uniqueness":
            constraints.append(
                UniquenessConstraint(property_expression=spec["property_expression"])
            )
        elif kind == "diversity":
            constraints.append(
                DiversityConstraint(
                    property_expression=spec["property_expression"],
                    min_unique_values=spec["min_unique_values"],
                )
            )
        else:
            raise ValueError(f"Unknown list constraint type: {kind!r}")
    return constraints


def build_batch_constraints(specs: list[dict]) -> list[BatchConstraint]:
    """Build cross-list constraints from their config entries.

    Parameters
    ----------
    specs : list[dict]
        Constraint entries, each with a ``type`` key.

    Returns
    -------
    list[BatchConstraint]
        The constructed constraints.

    Raises
    ------
    ValueError
        If a constraint type is unknown.
    """
    constraints: list[BatchConstraint] = []
    for spec in specs:
        kind = spec["type"]
        if kind == "coverage":
            constraints.append(
                BatchCoverageConstraint(
                    property_expression=spec["property_expression"],
                    target_values=spec["target_values"],
                    min_coverage=spec.get("min_coverage", 1.0),
                )
            )
        elif kind == "min_occurrence":
            constraints.append(
                BatchMinOccurrenceConstraint(
                    property_expression=spec["property_expression"],
                    min_occurrences=spec["min_occurrences"],
                )
            )
        elif kind == "diversity":
            constraints.append(
                BatchDiversityConstraint(
                    property_expression=spec["property_expression"],
                    max_lists_per_value=spec["max_lists_per_value"],
                )
            )
        else:
            raise ValueError(f"Unknown batch constraint type: {kind!r}")
    return constraints


def draw_pool(
    pairs: list[Item], target_counts: dict[str, int], n_lists: int, seed: int
) -> list[Item]:
    """Draw the pairs to partition, sampling each type in proportion.

    Sampling is random rather than positional: the pairs file is ordered by
    verb and frame, so taking a prefix would cover only a handful of verbs.

    Parameters
    ----------
    pairs : list[Item]
        All available pairs.
    target_counts : dict[str, int]
        Pairs of each type per list.
    n_lists : int
        Number of lists to fill.
    seed : int
        Seed for the draw.

    Returns
    -------
    list[Item]
        The sampled pairs.

    Raises
    ------
    ValueError
        If the pool holds too few pairs of a type.
    """
    rng = Random(seed)
    by_type: dict[str, list[Item]] = {}
    for pair in pairs:
        kind = str(pair.item_metadata.get("pair_type"))
        by_type.setdefault(kind, []).append(pair)

    drawn: list[Item] = []
    for kind, per_list in target_counts.items():
        wanted = per_list * n_lists
        available = by_type.get(kind, [])
        if len(available) < wanted:
            raise ValueError(
                f"Need {wanted} {kind} pairs but only {len(available)} available"
            )
        drawn.extend(rng.sample(available, wanted))
        print_info(f"Drew {wanted:,} {kind} pairs from {len(available):,}")

    rng.shuffle(drawn)
    return drawn


def main(config_path: Path, output: Path | None = None) -> None:
    """Partition the pairs and write the lists to the output path.

    Parameters
    ----------
    config_path : Path
        Path to the configuration file.
    output : Path | None
        Override the output path from the config.
    """
    config = load_config(config_path)
    logging.basicConfig(
        level=getattr(logging, config["logging"]["level"]),
        format=config["logging"]["format"],
    )

    print_header("Experiment Lists")

    list_config = config["lists"]
    n_lists = list_config["n_lists"]
    items_per_list = list_config["items_per_list"]
    strategy = list_config["strategy"]
    seed = list_config["random_seed"]

    list_constraints = build_list_constraints(list_config["constraints"])
    batch_constraints = build_batch_constraints(list_config["batch_constraints"])
    print_success(
        f"Built {len(list_constraints)} list and "
        f"{len(batch_constraints)} batch constraints"
    )

    target_counts = next(
        spec["target_counts"]
        for spec in list_config["constraints"]
        if spec["type"] == "balance"
    )
    if sum(target_counts.values()) != items_per_list:
        raise ValueError(
            f"Balance targets sum to {sum(target_counts.values())}, "
            f"but items_per_list is {items_per_list}"
        )

    pairs = load_pairs(BASE_DIR / config["paths"]["2afc_pairs"])
    print_success(f"Loaded {len(pairs):,} pairs")
    pool = draw_pool(pairs, target_counts, n_lists, seed)

    metadata: MetadataDict = {pair.id: dict(pair.item_metadata) for pair in pool}
    partitioner = ListPartitioner(random_seed=seed)
    lists = partitioner.partition_with_batch_constraints(
        items=[pair.id for pair in pool],
        n_lists=n_lists,
        list_constraints=list_constraints,
        batch_constraints=batch_constraints,
        strategy=strategy,
        metadata=metadata,
    )
    sizes = [len(exp_list.item_refs) for exp_list in lists]
    print_success(f"Built {len(lists)} lists of {min(sizes)} to {max(sizes)} pairs")
    if min(sizes) != items_per_list:
        print_warning(f"Lists are uneven: {sizes}")

    collection = ListCollection(
        name="argument_structure_lists",
        source_items_id=uuid4(),
        lists=tuple(lists),
        partitioning_strategy=strategy,
        partitioning_config={
            "n_lists": n_lists,
            "items_per_list": items_per_list,
            "random_seed": seed,
        },
        partitioning_stats={"total_items": sum(sizes)},
    )

    output_path = output or (BASE_DIR / config["paths"]["experiment_lists"])
    collection.to_jsonl(output_path)
    display_file_stats(output_path, len(lists), "lists")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Partition the 2AFC pairs into experiment lists."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=BASE_DIR / "config.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the output path from the config.",
    )
    args = parser.parse_args()
    main(config_path=args.config, output=args.output)
