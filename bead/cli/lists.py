"""List partitioning commands for bead CLI.

This module provides commands for partitioning items into experiment lists
(Stage 4 of the bead pipeline).
"""

from __future__ import annotations

import json
from pathlib import Path

import click
from didactic.api import ValidationError
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from bead.cli.utils import print_error, print_info, print_success
from bead.items.item import Item
from bead.lists import ExperimentList
from bead.lists.constraints import BatchConstraint, ListConstraint
from bead.lists.partitioner import ListPartitioner

console = Console()


@click.group()
def lists() -> None:
    r"""List construction commands (Stage 4).

    Commands for partitioning items into experiment lists.

    \b
    Examples:
        $ bead lists partition items.jsonl lists/ --n-lists 5 --strategy balanced
        $ bead lists list lists/
        $ bead lists validate lists/list_0.jsonl
        $ bead lists show-stats lists/
    """


@click.command()
@click.argument(
    "items_file", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument("output_file", type=click.Path(dir_okay=False, path_type=Path))
@click.option(
    "--strategy",
    type=click.Choice(["balanced", "random", "stratified"]),
    default="balanced",
    help="Partitioning strategy",
)
@click.option(
    "--n-lists",
    type=int,
    required=True,
    help="Number of lists to create",
)
@click.option(
    "--list-constraints",
    "list_constraint_files",
    type=click.Path(exists=True, path_type=Path),
    multiple=True,
    help="List constraint files (JSONL, can specify multiple)",
)
@click.option(
    "--batch-constraints",
    "batch_constraint_files",
    type=click.Path(exists=True, path_type=Path),
    multiple=True,
    help="Batch constraint files (JSONL, can specify multiple)",
)
@click.option(
    "--max-iterations",
    type=int,
    default=1000,
    help="Maximum iterations for batch constraint satisfaction (default: 1000)",
)
@click.option(
    "--random-seed",
    type=int,
    help="Random seed for reproducibility",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be done without writing files",
)
@click.pass_context
def partition(
    ctx: click.Context,
    items_file: Path,
    output_file: Path,
    strategy: str,
    n_lists: int,
    list_constraint_files: tuple[Path, ...],
    batch_constraint_files: tuple[Path, ...],
    max_iterations: int,
    random_seed: int | None,
    dry_run: bool,
) -> None:
    r"""Partition items into experiment lists.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    items_file : Path
        Path to items JSONL file.
    output_file : Path
        Output JSONL file for experiment lists (one list per line).
    strategy : str
        Partitioning strategy.
    n_lists : int
        Number of lists to create.
    list_constraint_files : tuple[Path, ...]
        List constraint files (JSONL).
    batch_constraint_files : tuple[Path, ...]
        Batch constraint files (JSONL).
    max_iterations : int
        Maximum iterations for batch constraint satisfaction.
    random_seed : int | None
        Random seed for reproducibility.
    dry_run : bool
        Show what would be done without writing files.

    Examples
    --------
    # Balanced partitioning
    $ bead lists partition items.jsonl lists.jsonl --n-lists 5 --strategy balanced

    # With list constraints
    $ bead lists partition items.jsonl lists.jsonl --n-lists 5 \\
        --list-constraints constraints/unique.jsonl

    # With batch constraints
    $ bead lists partition items.jsonl lists.jsonl --n-lists 5 \\
        --batch-constraints constraints/coverage.jsonl

    # With both constraint types
    $ bead lists partition items.jsonl lists.jsonl --n-lists 5 \\
        --list-constraints constraints/unique.jsonl constraints/balance.jsonl \\
        --batch-constraints constraints/coverage.jsonl \\
        --max-iterations 10000

    # Dry run to preview
    $ bead lists partition items.jsonl lists.jsonl \\
        --n-lists 5 --strategy balanced --dry-run
    """
    try:
        if n_lists < 1:
            print_error("--n-lists must be >= 1")
            ctx.exit(1)

        # Load items
        print_info(f"Loading items from {items_file}")
        items: list[Item] = []
        with open(items_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = Item.model_validate_json(line)
                items.append(item)

        if len(items) == 0:
            print_error("No items found in file")
            ctx.exit(1)

        print_info(f"Loaded {len(items)} items")

        # Extract item UUIDs and create metadata dict with all item data
        item_uuids = [item.id for item in items]
        metadata = {}
        for item in items:
            item_meta = {
                **item.item_metadata,
                "template_id": str(item.item_template_id),
            }
            # Add task_type if it exists (optional field for backwards compatibility)
            if hasattr(item, "task_type") and item.task_type is not None:
                item_meta["task_type"] = item.task_type
            metadata[item.id] = item_meta

        # Load list constraints if provided
        list_constraints = []
        if list_constraint_files:
            print_info(f"Loading {len(list_constraint_files)} list constraint file(s)")
            for constraint_file in list_constraint_files:
                with open(constraint_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        constraint = ListConstraint.model_validate_json(line)
                        list_constraints.append(constraint)
            print_info(f"Loaded {len(list_constraints)} list constraint(s)")

        # Load batch constraints if provided
        batch_constraints = []
        if batch_constraint_files:
            print_info(
                f"Loading {len(batch_constraint_files)} batch constraint file(s)"
            )
            for constraint_file in batch_constraint_files:
                with open(constraint_file, encoding="utf-8") as f:
                    for line in f:
                        line = line.strip()
                        if not line:
                            continue
                        constraint = BatchConstraint.model_validate_json(line)
                        batch_constraints.append(constraint)
            print_info(f"Loaded {len(batch_constraints)} batch constraint(s)")

        # Create partitioner
        partitioner = ListPartitioner(random_seed=random_seed)

        # Partition items (choose method based on constraints)
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(
                f"Partitioning {len(items)} items into {n_lists} lists...", total=None
            )

            if batch_constraints:
                # Use batch-constrained partitioning
                experiment_lists = partitioner.partition_with_batch_constraints(
                    items=item_uuids,
                    n_lists=n_lists,
                    list_constraints=list_constraints if list_constraints else None,
                    batch_constraints=batch_constraints,
                    strategy=strategy,
                    metadata=metadata,
                    max_iterations=max_iterations,
                )
            else:
                # Use standard partitioning (with optional list constraints)
                experiment_lists = partitioner.partition(
                    items=item_uuids,
                    n_lists=n_lists,
                    constraints=list_constraints if list_constraints else None,
                    strategy=strategy,
                    metadata=metadata,
                )

        # Save lists (or show dry-run preview)
        if dry_run:
            print_info(f"[DRY RUN] Would write {len(experiment_lists)} lists to:")
            console.print(f"  [dim]{output_file}[/dim]")
            for exp_list in experiment_lists:
                console.print(
                    f"    list_{exp_list.list_number}: {len(exp_list.item_refs)} items"
                )
            print_info(
                f"[DRY RUN] Total: {len(experiment_lists)} lists, {len(items)} items"
            )
        else:
            # Ensure parent directory exists
            output_file.parent.mkdir(parents=True, exist_ok=True)
            # Write all lists to single JSONL file (one list per line)
            with open(output_file, "w", encoding="utf-8") as f:
                for exp_list in experiment_lists:
                    f.write(exp_list.model_dump_json() + "\n")

            print_success(
                f"Created {len(experiment_lists)} lists "
                f"with {len(items)} items: {output_file}"
            )

        # Show distribution
        console.print("\n[cyan]Distribution:[/cyan]")
        for exp_list in experiment_lists:
            console.print(
                f"  list_{exp_list.list_number}: {len(exp_list.item_refs)} items"
            )

    except ValidationError as e:
        print_error(f"Validation error: {e}")
        ctx.exit(1)
    except Exception as e:
        print_error(f"Failed to partition items: {e}")
        ctx.exit(1)


@click.command(name="list")
@click.argument(
    "lists_file", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.pass_context
def list_lists(
    ctx: click.Context,
    lists_file: Path,
) -> None:
    """List experiment lists in a JSONL file.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    lists_file : Path
        JSONL file containing experiment lists (one list per line).

    Examples
    --------
    $ bead lists list lists.jsonl
    """
    try:
        table = Table(title=f"Experiment Lists in {lists_file}")
        table.add_column("List #", justify="right", style="yellow")
        table.add_column("Name", style="cyan")
        table.add_column("Items", justify="right", style="green")

        with open(lists_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    exp_list = ExperimentList.model_validate_json(line)
                    table.add_row(
                        str(exp_list.list_number),
                        exp_list.name,
                        str(len(exp_list.item_refs)),
                    )
                except Exception:
                    continue

        console.print(table)

    except Exception as e:
        print_error(f"Failed to list experiment lists: {e}")
        ctx.exit(1)


@click.command()
@click.argument("list_file", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def validate(ctx: click.Context, list_file: Path) -> None:
    """Validate an experiment list file.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    list_file : Path
        Path to experiment list file.

    Examples
    --------
    $ bead lists validate list_0.jsonl
    """
    try:
        print_info(f"Validating experiment list: {list_file}")

        with open(list_file, encoding="utf-8") as f:
            first_line = f.readline().strip()
            if not first_line:
                print_error("File is empty")
                ctx.exit(1)

        exp_list = ExperimentList.model_validate_json(first_line)

        print_success(
            f"Experiment list is valid: {exp_list.name} "
            f"({len(exp_list.item_refs)} items)"
        )

    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON: {e}")
        ctx.exit(1)
    except ValidationError as e:
        print_error(f"Validation error: {e}")
        ctx.exit(1)
    except Exception as e:
        print_error(f"Failed to validate experiment list: {e}")
        ctx.exit(1)


@click.command()
@click.argument(
    "lists_file", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.pass_context
def show_stats(ctx: click.Context, lists_file: Path) -> None:
    """Show statistics about experiment lists in a JSONL file.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    lists_file : Path
        JSONL file containing experiment lists (one list per line).

    Examples
    --------
    $ bead lists show-stats lists.jsonl
    """
    try:
        print_info(f"Analyzing experiment lists in: {lists_file}")

        lists_data: list[ExperimentList] = []
        with open(lists_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    exp_list = ExperimentList.model_validate_json(line)
                    lists_data.append(exp_list)
                except Exception:
                    continue

        if not lists_data:
            print_error("No valid experiment lists found")
            ctx.exit(1)

        # Calculate statistics
        total_lists = len(lists_data)
        item_counts = [len(exp_list.item_refs) for exp_list in lists_data]
        total_items = sum(item_counts)
        avg_items = total_items / total_lists if total_lists > 0 else 0
        min_items = min(item_counts) if item_counts else 0
        max_items = max(item_counts) if item_counts else 0

        # Display statistics
        table = Table(title="Experiment List Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Total Lists", str(total_lists))
        table.add_row("Total Items", str(total_items))
        table.add_row("", "")  # Separator
        table.add_row("Avg Items per List", f"{avg_items:.1f}")
        table.add_row("Min Items per List", str(min_items))
        table.add_row("Max Items per List", str(max_items))

        console.print(table)

        # Show per-list breakdown
        console.print("\n[cyan]Per-List Breakdown:[/cyan]")
        for exp_list in sorted(lists_data, key=lambda x: x.list_number):
            console.print(f"  {exp_list.name}: {len(exp_list.item_refs)} items")

    except Exception as e:
        print_error(f"Failed to show statistics: {e}")
        ctx.exit(1)


# Import constraint builder commands
from bead.cli.list_constraints import (  # noqa: E402
    create_balance,
    create_batch_balance,
    create_batch_coverage,
    create_batch_diversity,
    create_batch_min_occurrence,
    create_diversity,
    create_grouped_quantile,
    create_quantile,
    create_size,
    create_uniqueness,
)

# Register core commands
lists.add_command(partition)
lists.add_command(list_lists)
lists.add_command(validate)
lists.add_command(show_stats)

# Register list constraint commands
lists.add_command(create_uniqueness, name="create-uniqueness")
lists.add_command(create_balance, name="create-balance")
lists.add_command(create_quantile, name="create-quantile")
lists.add_command(create_grouped_quantile, name="create-grouped-quantile")
lists.add_command(create_diversity, name="create-diversity")
lists.add_command(create_size, name="create-size")

# Register batch constraint commands
lists.add_command(create_batch_coverage, name="create-batch-coverage")
lists.add_command(create_batch_balance, name="create-batch-balance")
lists.add_command(create_batch_diversity, name="create-batch-diversity")
lists.add_command(create_batch_min_occurrence, name="create-batch-min-occurrence")
