"""List constraint builder commands for bead CLI.

This module provides CLI commands for creating list and batch constraints.
List constraints apply to individual lists, while batch constraints apply
across all lists in a batch.

List constraints (8 types):
- uniqueness: No duplicate property values
- balance: Balanced distribution
- quantile: Uniform across quantiles
- grouped-quantile: Quantile distribution within groups
- conditional-uniqueness: Conditional uniqueness via DSL
- diversity: Minimum unique values
- size: List size requirements
- ordering: Presentation order (runtime)

Batch constraints (4 types):
- coverage: All values appear somewhere
- balance: Balanced distribution across batch
- diversity: Prevent values in too many lists
- min-occurrence: Minimum occurrences per value
"""

from __future__ import annotations

from pathlib import Path

import click

from bead.cli.display import print_error, print_success
from bead.cli.utils import parse_key_value_pairs, parse_list_option
from bead.lists.constraints import (
    BalanceConstraint,
    BatchBalanceConstraint,
    BatchCoverageConstraint,
    BatchDiversityConstraint,
    BatchMinOccurrenceConstraint,
    DiversityConstraint,
    GroupedQuantileConstraint,
    QuantileConstraint,
    SizeConstraint,
    UniquenessConstraint,
)

# ==================== List Constraint Commands ====================


@click.command()
@click.option(
    "--property-expression",
    type=str,
    required=True,
    help="Property expression (e.g., 'item.metadata.verb')",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSONL file",
)
@click.option(
    "--priority",
    type=int,
    default=5,
    help="Constraint priority (default: 5)",
)
def create_uniqueness(
    property_expression: str,
    output: Path,
    priority: int,
) -> None:
    r"""Create uniqueness constraint.

    Ensures no duplicate values for a property within each list.

    Examples
    --------
    $ bead lists create-uniqueness \\
        --property-expression "item.metadata.verb" \\
        -o constraints/unique_verbs.jsonl
    """
    try:
        constraint = UniquenessConstraint(
            constraint_type="uniqueness",
            property_expression=property_expression,
            priority=priority,
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            f.write(constraint.model_dump_json() + "\n")

        print_success(f"Created uniqueness constraint: {output}")

    except Exception as e:
        print_error(f"Failed to create uniqueness constraint: {e}")


@click.command()
@click.option(
    "--property-expression",
    type=str,
    required=True,
    help="Property expression",
)
@click.option(
    "--target-counts",
    type=str,
    help=(
        "Target counts (key=value pairs, comma-separated, e.g., 'a=20,b=10'). "
        "Omit for equal distribution."
    ),
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSONL file",
)
@click.option(
    "--tolerance",
    type=float,
    default=0.1,
    help="Tolerance (default: 0.1)",
)
@click.option(
    "--priority",
    type=int,
    default=5,
    help="Constraint priority (default: 5)",
)
def create_balance(
    property_expression: str,
    target_counts: str | None,
    output: Path,
    tolerance: float,
    priority: int,
) -> None:
    r"""Create balance constraint.

    Ensures balanced distribution of property values.

    Examples
    --------
    $ bead lists create-balance \\
        --property-expression "item.metadata.condition" \\
        --target-counts "control=20,experimental=10" \\
        -o constraints/balance.jsonl
    """
    try:
        # Parse target counts (None means equal distribution)
        counts_dict = None
        if target_counts:
            parsed = parse_key_value_pairs(target_counts)
            counts_dict = {k: int(v) for k, v in parsed.items()}

        constraint = BalanceConstraint(
            constraint_type="balance",
            property_expression=property_expression,
            target_counts=counts_dict,
            tolerance=tolerance,
            priority=priority,
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            f.write(constraint.model_dump_json() + "\n")

        print_success(f"Created balance constraint: {output}")

    except Exception as e:
        print_error(f"Failed to create balance constraint: {e}")


@click.command()
@click.option(
    "--property-expression",
    type=str,
    required=True,
    help="Property expression",
)
@click.option(
    "--n-quantiles",
    type=int,
    required=True,
    help="Number of quantiles",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSONL file",
)
@click.option(
    "--priority",
    type=int,
    default=5,
    help="Constraint priority (default: 5)",
)
def create_quantile(
    property_expression: str,
    n_quantiles: int,
    output: Path,
    priority: int,
) -> None:
    r"""Create quantile constraint.

    Ensures uniform distribution across quantiles.

    Examples
    --------
    $ bead lists create-quantile \\
        --property-expression "item.metadata.word_length" \\
        --n-quantiles 4 \\
        -o constraints/quantile.jsonl
    """
    try:
        constraint = QuantileConstraint(
            constraint_type="quantile",
            property_expression=property_expression,
            n_quantiles=n_quantiles,
            priority=priority,
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            f.write(constraint.model_dump_json() + "\n")

        print_success(f"Created quantile constraint: {output}")

    except Exception as e:
        print_error(f"Failed to create quantile constraint: {e}")


@click.command()
@click.option(
    "--property-expression",
    type=str,
    required=True,
    help="Property expression",
)
@click.option(
    "--group-by-expression",
    type=str,
    required=True,
    help="Group-by expression",
)
@click.option(
    "--n-quantiles",
    type=int,
    required=True,
    help="Number of quantiles",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSONL file",
)
@click.option(
    "--priority",
    type=int,
    default=5,
    help="Constraint priority (default: 5)",
)
def create_grouped_quantile(
    property_expression: str,
    group_by_expression: str,
    n_quantiles: int,
    output: Path,
    priority: int,
) -> None:
    r"""Create grouped quantile constraint.

    Ensures quantile distribution within groups.

    Examples
    --------
    $ bead lists create-grouped-quantile \\
        --property-expression "item.metadata.frequency" \\
        --group-by-expression "item.metadata.condition" \\
        --n-quantiles 3 \\
        -o constraints/grouped_quantile.jsonl
    """
    try:
        constraint = GroupedQuantileConstraint(
            constraint_type="grouped_quantile",
            property_expression=property_expression,
            group_by_expression=group_by_expression,
            n_quantiles=n_quantiles,
            priority=priority,
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            f.write(constraint.model_dump_json() + "\n")

        print_success(f"Created grouped quantile constraint: {output}")

    except Exception as e:
        print_error(f"Failed to create grouped quantile constraint: {e}")


@click.command()
@click.option(
    "--property-expression",
    type=str,
    required=True,
    help="Property expression",
)
@click.option(
    "--min-unique",
    type=int,
    required=True,
    help="Minimum unique values",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSONL file",
)
@click.option(
    "--priority",
    type=int,
    default=5,
    help="Constraint priority (default: 5)",
)
def create_diversity(
    property_expression: str,
    min_unique: int,
    output: Path,
    priority: int,
) -> None:
    r"""Create diversity constraint.

    Ensures minimum unique values for property.

    Examples
    --------
    $ bead lists create-diversity \\
        --property-expression "item.metadata.verb_class" \\
        --min-unique 10 \\
        -o constraints/diversity.jsonl
    """
    try:
        constraint = DiversityConstraint(
            constraint_type="diversity",
            property_expression=property_expression,
            min_unique_values=min_unique,
            priority=priority,
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            f.write(constraint.model_dump_json() + "\n")

        print_success(f"Created diversity constraint: {output}")

    except Exception as e:
        print_error(f"Failed to create diversity constraint: {e}")


@click.command()
@click.option(
    "--exact-size",
    type=int,
    help="Exact number of items per list (mutually exclusive with min/max)",
)
@click.option(
    "--min-size",
    type=int,
    help="Minimum items per list",
)
@click.option(
    "--max-size",
    type=int,
    help="Maximum items per list",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSONL file",
)
@click.option(
    "--priority",
    type=int,
    default=5,
    help="Constraint priority (default: 5)",
)
def create_size(
    exact_size: int | None,
    min_size: int | None,
    max_size: int | None,
    output: Path,
    priority: int,
) -> None:
    r"""Create size constraint.

    Ensures list size requirements.

    Examples
    --------
    $ bead lists create-size --exact-size 40 -o constraints/size.jsonl
    $ bead lists create-size --min-size 40 --max-size 60 \\
        -o constraints/size.jsonl
    """
    try:
        if exact_size is not None and (min_size is not None or max_size is not None):
            raise ValueError(
                "Cannot specify --exact-size with --min-size or --max-size"
            )

        if exact_size is None and min_size is None and max_size is None:
            raise ValueError(
                "Must specify --exact-size or at least one of --min-size/--max-size"
            )

        constraint = SizeConstraint(
            constraint_type="size",
            exact_size=exact_size,
            min_size=min_size,
            max_size=max_size,
            priority=priority,
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            f.write(constraint.model_dump_json() + "\n")

        print_success(f"Created size constraint: {output}")

    except Exception as e:
        print_error(f"Failed to create size constraint: {e}")


# ==================== Batch Constraint Commands ====================


@click.command()
@click.option(
    "--property-expression",
    type=str,
    required=True,
    help="Property expression",
)
@click.option(
    "--target-values",
    type=str,
    required=True,
    help="Target values (comma-separated)",
)
@click.option(
    "--min-coverage",
    type=float,
    default=1.0,
    help="Minimum coverage (default: 1.0 = 100%%)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSONL file",
)
@click.option(
    "--priority",
    type=int,
    default=5,
    help="Constraint priority (default: 5)",
)
def create_batch_coverage(
    property_expression: str,
    target_values: str,
    min_coverage: float,
    output: Path,
    priority: int,
) -> None:
    r"""Create batch coverage constraint.

    Ensures all target values appear somewhere across all lists.

    Examples
    --------
    $ bead lists create-batch-coverage \\
        --property-expression "item.metadata.template_id" \\
        --target-values "0,1,2,3,4,5" \\
        -o constraints/coverage.jsonl
    """
    try:
        values_list = parse_list_option(target_values)

        constraint = BatchCoverageConstraint(
            constraint_type="coverage",
            property_expression=property_expression,
            target_values=values_list,  # type: ignore[arg-type]
            min_coverage=min_coverage,
            priority=priority,
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            f.write(constraint.model_dump_json() + "\n")

        print_success(f"Created batch coverage constraint: {output}")

    except Exception as e:
        print_error(f"Failed to create batch coverage constraint: {e}")


@click.command()
@click.option(
    "--property-expression",
    type=str,
    required=True,
    help="Property expression",
)
@click.option(
    "--target-distribution",
    type=str,
    required=True,
    help="Target distribution (key=value pairs, comma-separated)",
)
@click.option(
    "--tolerance",
    type=float,
    default=0.05,
    help="Tolerance (default: 0.05)",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSONL file",
)
@click.option(
    "--priority",
    type=int,
    default=5,
    help="Constraint priority (default: 5)",
)
def create_batch_balance(
    property_expression: str,
    target_distribution: str,
    tolerance: float,
    output: Path,
    priority: int,
) -> None:
    r"""Create batch balance constraint.

    Ensures balanced distribution across entire batch.

    Examples
    --------
    $ bead lists create-batch-balance \\
        --property-expression "item.metadata.condition" \\
        --target-distribution "control=0.5,experimental=0.5" \\
        -o constraints/batch_balance.jsonl
    """
    try:
        dist_dict = parse_key_value_pairs(target_distribution)
        dist_float = {k: float(v) for k, v in dist_dict.items()}

        constraint = BatchBalanceConstraint(
            constraint_type="balance",
            property_expression=property_expression,
            target_distribution=dist_float,
            tolerance=tolerance,
            priority=priority,
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            f.write(constraint.model_dump_json() + "\n")

        print_success(f"Created batch balance constraint: {output}")

    except Exception as e:
        print_error(f"Failed to create batch balance constraint: {e}")


@click.command()
@click.option(
    "--property-expression",
    type=str,
    required=True,
    help="Property expression",
)
@click.option(
    "--max-lists-per-value",
    type=int,
    required=True,
    help="Maximum lists per value",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSONL file",
)
@click.option(
    "--priority",
    type=int,
    default=5,
    help="Constraint priority (default: 5)",
)
def create_batch_diversity(
    property_expression: str,
    max_lists_per_value: int,
    output: Path,
    priority: int,
) -> None:
    r"""Create batch diversity constraint.

    Prevents values from appearing in too many lists.

    Examples
    --------
    $ bead lists create-batch-diversity \\
        --property-expression "item.metadata.target_word" \\
        --max-lists-per-value 3 \\
        -o constraints/batch_diversity.jsonl
    """
    try:
        constraint = BatchDiversityConstraint(
            constraint_type="diversity",
            property_expression=property_expression,
            max_lists_per_value=max_lists_per_value,
            priority=priority,
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            f.write(constraint.model_dump_json() + "\n")

        print_success(f"Created batch diversity constraint: {output}")

    except Exception as e:
        print_error(f"Failed to create batch diversity constraint: {e}")


@click.command()
@click.option(
    "--property-expression",
    type=str,
    required=True,
    help="Property expression",
)
@click.option(
    "--min-occurrences",
    type=int,
    required=True,
    help="Minimum occurrences per value",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output JSONL file",
)
@click.option(
    "--priority",
    type=int,
    default=5,
    help="Constraint priority (default: 5)",
)
def create_batch_min_occurrence(
    property_expression: str,
    min_occurrences: int,
    output: Path,
    priority: int,
) -> None:
    r"""Create batch minimum occurrence constraint.

    Ensures minimum occurrences per value across batch.

    Examples
    --------
    $ bead lists create-batch-min-occurrence \\
        --property-expression "item.metadata.construction" \\
        --min-occurrences 5 \\
        -o constraints/min_occurrence.jsonl
    """
    try:
        constraint = BatchMinOccurrenceConstraint(
            constraint_type="min_occurrence",
            property_expression=property_expression,
            min_occurrences=min_occurrences,
            priority=priority,
        )

        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w") as f:
            f.write(constraint.model_dump_json() + "\n")

        print_success(f"Created batch min occurrence constraint: {output}")

    except Exception as e:
        print_error(f"Failed to create batch min occurrence constraint: {e}")


# Export all commands
__all__ = [
    "create_uniqueness",
    "create_balance",
    "create_quantile",
    "create_grouped_quantile",
    "create_diversity",
    "create_size",
    "create_batch_coverage",
    "create_batch_balance",
    "create_batch_diversity",
    "create_batch_min_occurrence",
]
