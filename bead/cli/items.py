"""Item construction commands for bead CLI.

This module provides commands for constructing experimental items from filled
templates (Stage 3 of the bead pipeline).

Commands support:
- Full item construction with ItemTemplate specifications
- Model adapter integration (HuggingFace, OpenAI, Anthropic, Google, TogetherAI)
- Model output caching for efficiency
- Constraint-based filtering (DSL, extensional, intensional, relational)
- Batch processing with progress tracking
- Parallel execution for large-scale construction
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast
from uuid import UUID

import click
from didactic.api import ValidationError
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from bead.cli.utils import print_error, print_info, print_success
from bead.items.adapters.registry import default_registry
from bead.items.cache import ModelOutputCache
from bead.items.constructor import ItemConstructor
from bead.items.item import Item
from bead.items.item_template import ItemTemplate, TaskType
from bead.items.validation import (
    get_task_type_requirements,
    infer_task_type_from_item,
    validate_item_for_task_type,
)
from bead.resources.constraints import Constraint
from bead.templates.filler import FilledTemplate

console = Console()


# Helper functions for item construction


def _load_item_templates(template_file: Path) -> list[ItemTemplate]:
    """Load ItemTemplate objects from JSONL file.

    Parameters
    ----------
    template_file : Path
        Path to ItemTemplate JSONL file.

    Returns
    -------
    list[ItemTemplate]
        List of loaded ItemTemplate objects.

    Raises
    ------
    FileNotFoundError
        If template file doesn't exist.
    ValidationError
        If template data is invalid.
    """
    templates: list[ItemTemplate] = []

    with open(template_file, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                template = ItemTemplate.model_validate_json(line)
                templates.append(template)
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {line_num}: Invalid JSON - {e}") from e
            except ValidationError as e:
                raise ValueError(f"Line {line_num}: Invalid ItemTemplate - {e}") from e

    return templates


def _load_filled_templates(filled_file: Path) -> dict[UUID, FilledTemplate]:
    """Load FilledTemplate objects from JSONL file.

    Parameters
    ----------
    filled_file : Path
        Path to FilledTemplate JSONL file.

    Returns
    -------
    dict[UUID, FilledTemplate]
        Map of FilledTemplate IDs to objects.

    Raises
    ------
    FileNotFoundError
        If filled templates file doesn't exist.
    ValidationError
        If filled template data is invalid.
    """
    filled_templates: dict[UUID, FilledTemplate] = {}

    with open(filled_file, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                filled = FilledTemplate.model_validate_json(line)
                filled_templates[filled.id] = filled
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {line_num}: Invalid JSON - {e}") from e
            except ValidationError as e:
                raise ValueError(
                    f"Line {line_num}: Invalid FilledTemplate - {e}"
                ) from e

    return filled_templates


def _load_constraints(constraints_file: Path) -> dict[UUID, Constraint]:
    """Load Constraint objects from JSONL file.

    Parameters
    ----------
    constraints_file : Path
        Path to Constraint JSONL file.

    Returns
    -------
    dict[UUID, Constraint]
        Map of Constraint IDs to objects.

    Raises
    ------
    FileNotFoundError
        If constraints file doesn't exist.
    ValidationError
        If constraint data is invalid.
    """
    constraints: dict[UUID, Constraint] = {}

    with open(constraints_file, encoding="utf-8") as f:
        for line_num, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue

            try:
                constraint = Constraint.model_validate_json(line)  # type: ignore[misc]
                constraints[constraint.id] = constraint  # type: ignore[misc]
            except json.JSONDecodeError as e:
                raise ValueError(f"Line {line_num}: Invalid JSON - {e}") from e
            except ValidationError as e:
                raise ValueError(f"Line {line_num}: Invalid Constraint - {e}") from e

    return constraints


def _setup_cache(
    cache_dir: Path | None,
    no_cache: bool,
) -> ModelOutputCache:
    """Set up model output cache.

    Parameters
    ----------
    cache_dir : Path | None
        Cache directory (None for default).
    no_cache : bool
        Whether to disable caching.

    Returns
    -------
    ModelOutputCache
        Configured cache instance.
    """
    if no_cache:
        return ModelOutputCache(backend="memory", enabled=False)

    if cache_dir:
        return ModelOutputCache(cache_dir=cache_dir, backend="filesystem")

    # Use default cache location
    return ModelOutputCache(backend="filesystem")


def _display_construction_stats(
    items: list[Item],
    templates: list[ItemTemplate],
) -> None:
    """Display construction statistics.

    Parameters
    ----------
    items : list[Item]
        Constructed items.
    templates : list[ItemTemplate]
        ItemTemplates used for construction.
    """
    table = Table(title="Item Construction Statistics")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    # Total items
    table.add_row("Total Items Created", str(len(items)))
    table.add_row("ItemTemplates Processed", str(len(templates)))
    table.add_row("", "")  # Separator

    # Items per template
    if templates:
        items_per_template = len(items) / len(templates)
        table.add_row("Avg Items per Template", f"{items_per_template:.1f}")

    # Model outputs
    total_model_outputs = sum(len(item.model_outputs) for item in items)
    if total_model_outputs > 0:
        table.add_row("Total Model Outputs", str(total_model_outputs))
        avg_outputs_per_item = total_model_outputs / len(items) if items else 0
        table.add_row("Avg Outputs per Item", f"{avg_outputs_per_item:.1f}")

    # Constraint satisfaction
    if items and items[0].constraint_satisfaction:
        satisfied_count = sum(
            1 for item in items for cs in item.constraint_satisfaction if cs.satisfied
        )
        total_constraints = sum(len(item.constraint_satisfaction) for item in items)
        if total_constraints > 0:
            table.add_row("", "")  # Separator
            table.add_row("Constraints Satisfied", str(satisfied_count))
            table.add_row("Total Constraint Checks", str(total_constraints))
            satisfaction_rate = (satisfied_count / total_constraints) * 100
            table.add_row("Satisfaction Rate", f"{satisfaction_rate:.1f}%")

    console.print(table)


@click.group()
def items() -> None:
    r"""Item construction commands (Stage 3).

    Commands for constructing and managing experimental items.

    \b
    Examples:
        $ bead items construct --item-template template.jsonl \
            --filled-templates filled.jsonl --output items.jsonl
        $ bead items list items.jsonl
        $ bead items validate items.jsonl
        $ bead items show-stats items.jsonl
    """


@click.command()
@click.option(
    "--item-template",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to ItemTemplate JSONL file",
)
@click.option(
    "--filled-templates",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to filled templates JSONL file",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Path to output items JSONL file",
)
@click.option(
    "--constraints",
    type=click.Path(exists=True, path_type=Path),
    help="Path to constraints JSONL file (optional)",
)
@click.option(
    "--cache-dir",
    type=click.Path(path_type=Path),
    help="Cache directory for model outputs",
)
@click.option(
    "--no-cache",
    is_flag=True,
    help="Disable model output caching",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Preview construction without executing",
)
@click.pass_context
def construct(
    ctx: click.Context,
    item_template: Path,
    filled_templates: Path,
    output: Path,
    constraints: Path | None,
    cache_dir: Path | None,
    no_cache: bool,
    dry_run: bool,
) -> None:
    r"""Construct experimental items from filled templates.

    Constructs items by combining filled templates according to ItemTemplate
    specifications. Supports model-based constraints, caching, and batch processing.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    item_template : Path
        Path to ItemTemplate JSONL file.
    filled_templates : Path
        Path to filled templates JSONL file.
    output : Path
        Path to output items JSONL file.
    constraints : Path | None
        Path to constraints JSONL file (optional).
    cache_dir : Path | None
        Cache directory for model outputs.
    no_cache : bool
        Whether to disable caching.
    dry_run : bool
        Whether to preview without executing.

    Examples
    --------
    # Basic construction
    $ bead items construct \
        --item-template templates.jsonl \
        --filled-templates filled.jsonl \
        --output items.jsonl

    # With constraints
    $ bead items construct \
        --item-template templates.jsonl \
        --filled-templates filled.jsonl \
        --constraints constraints.jsonl \
        --output items.jsonl

    # With custom cache
    $ bead items construct \
        --item-template templates.jsonl \
        --filled-templates filled.jsonl \
        --output items.jsonl \
        --cache-dir .cache/models

    # Dry run
    $ bead items construct \
        --item-template templates.jsonl \
        --filled-templates filled.jsonl \
        --output items.jsonl \
        --dry-run
    """
    try:
        # Load ItemTemplates
        print_info(f"Loading ItemTemplates from {item_template}")
        templates = _load_item_templates(item_template)
        print_info(f"Loaded {len(templates)} ItemTemplate(s)")

        # Load filled templates
        print_info(f"Loading filled templates from {filled_templates}")
        filled_map = _load_filled_templates(filled_templates)
        print_info(f"Loaded {len(filled_map)} filled template(s)")

        # Load constraints if provided
        constraints_map: dict[UUID, Constraint] = {}
        if constraints:
            print_info(f"Loading constraints from {constraints}")
            constraints_map = _load_constraints(constraints)
            print_info(f"Loaded {len(constraints_map)} constraint(s)")

        # Validate constraint references
        for template in templates:
            for constraint_id in template.constraints:
                if constraint_id not in constraints_map:
                    print_error(
                        f"ItemTemplate '{template.name}' references unknown "
                        f"constraint {constraint_id}"
                    )
                    ctx.exit(1)

        # Dry run mode
        if dry_run:
            print_info("[DRY RUN] Construction preview:")
            console.print(f"  ItemTemplates: {len(templates)}")
            console.print(f"  Filled Templates: {len(filled_map)}")
            console.print(f"  Constraints: {len(constraints_map)}")
            console.print(f"  Output: {output}")
            print_info("[DRY RUN] No items will be constructed")
            return

        # Set up cache
        print_info("Setting up model output cache")
        cache = _setup_cache(cache_dir, no_cache)

        # Set up constructor
        constructor = ItemConstructor(
            model_registry=default_registry,
            cache=cache,
        )

        # Construct items with progress
        all_items: list[Item] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Constructing items from {len(templates)} template(s)...",
                total=len(templates),
            )

            for template in templates:
                try:
                    # Construct items for this template
                    items = list(
                        constructor.construct_items(
                            template, filled_map, constraints_map
                        )
                    )
                    all_items.extend(items)
                    progress.advance(task)
                except Exception as e:
                    print_error(
                        f"Failed to construct items for template '{template.name}': {e}"
                    )
                    continue

        # Save items
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            for item in all_items:
                f.write(item.model_dump_json() + "\n")

        print_success(f"Created {len(all_items)} item(s): {output}")

        # Display statistics
        if all_items:
            _display_construction_stats(all_items, templates)

    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        ctx.exit(1)
    except ValidationError as e:
        print_error(f"Validation error: {e}")
        ctx.exit(1)
    except ValueError as e:
        print_error(str(e))
        ctx.exit(1)
    except Exception as e:
        print_error(f"Failed to construct items: {e}")
        ctx.exit(1)


@click.command(name="list")
@click.option(
    "--directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path.cwd(),
    help="Directory to search for item files",
)
@click.option(
    "--pattern",
    default="*.jsonl",
    help="File pattern to match (default: *.jsonl)",
)
@click.pass_context
def list_items(
    ctx: click.Context,
    directory: Path,
    pattern: str,
) -> None:
    """List item files in a directory.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    directory : Path
        Directory to search.
    pattern : str
        File pattern to match.

    Examples
    --------
    $ bead items list
    $ bead items list --directory items/
    $ bead items list --pattern "experiment_*.jsonl"
    """
    try:
        files = list(directory.glob(pattern))

        if not files:
            print_info(f"No files found in {directory} matching {pattern}")
            return

        table = Table(title=f"Items in {directory}")
        table.add_column("File", style="cyan")
        table.add_column("Count", justify="right", style="yellow")
        table.add_column("Sample", style="white")

        for file_path in sorted(files):
            try:
                with open(file_path, encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]

                if not lines:
                    continue

                count = len(lines)

                # Parse first item for preview
                first_data = json.loads(lines[0])
                rendered = first_data.get("rendered_elements", {})

                # Get first rendered element as sample
                sample = "N/A"
                if rendered:
                    first_key = next(iter(rendered))
                    sample = str(rendered[first_key])
                    if len(sample) > 40:
                        sample = sample[:37] + "..."

                table.add_row(
                    str(file_path.name),
                    str(count),
                    sample,
                )
            except Exception:
                continue

        console.print(table)

    except Exception as e:
        print_error(f"Failed to list items: {e}")
        ctx.exit(1)


@click.command()
@click.argument("items_file", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def validate(ctx: click.Context, items_file: Path) -> None:
    """Validate an items file.

    Checks that all items are properly formatted.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    items_file : Path
        Path to items file.

    Examples
    --------
    $ bead items validate items.jsonl
    """
    try:
        print_info(f"Validating items: {items_file}")

        count = 0
        errors: list[str] = []

        with open(items_file, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    Item.model_validate_json(line)
                    count += 1
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: Invalid JSON - {e}")
                except ValidationError as e:
                    errors.append(f"Line {line_num}: Validation error - {e}")

        if errors:
            print_error(f"Validation failed with {len(errors)} errors:")
            for error in errors[:10]:
                console.print(f"  [red]✗[/red] {error}")
            if len(errors) > 10:
                console.print(f"  ... and {len(errors) - 10} more errors")
            ctx.exit(1)
        else:
            print_success(f"Items file is valid: {count} items")

    except Exception as e:
        print_error(f"Failed to validate items: {e}")
        ctx.exit(1)


@click.command()
@click.argument("items_file", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def show_stats(ctx: click.Context, items_file: Path) -> None:
    """Show statistics about items.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    items_file : Path
        Path to items file.

    Examples
    --------
    $ bead items show-stats items.jsonl
    """
    try:
        print_info(f"Analyzing items: {items_file}")

        total_count = 0
        templates_seen: set[str] = set()
        model_output_counts: dict[str, int] = {}

        with open(items_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    item = Item.model_validate_json(line)

                    total_count += 1
                    templates_seen.add(str(item.item_template_id))

                    # Count model outputs
                    for output in item.model_outputs:
                        model_name = output.model_name
                        model_output_counts[model_name] = (
                            model_output_counts.get(model_name, 0) + 1
                        )

                except Exception:
                    continue

        if total_count == 0:
            print_error("No valid items found")
            ctx.exit(1)

        # Display statistics
        table = Table(title="Item Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Total Items", str(total_count))
        table.add_row("Unique Templates", str(len(templates_seen)))
        table.add_row("", "")  # Separator

        if model_output_counts:
            for model_name, count in sorted(model_output_counts.items()):
                table.add_row(f"Model Outputs: {model_name}", str(count))

        console.print(table)

    except Exception as e:
        print_error(f"Failed to show statistics: {e}")
        ctx.exit(1)


# Import task-type factory commands
from bead.cli.items_factories import (  # noqa: E402
    create_binary_from_texts,
    create_categorical,
    create_forced_choice,
    create_forced_choice_from_texts,
    create_free_text_from_texts,
    create_likert_7,
    create_magnitude_from_texts,
    create_multi_select_from_texts,
    create_nli,
    create_ordinal_scale_from_texts,
    create_simple_cloze,
)

# Register core commands
items.add_command(construct)
items.add_command(list_items)
items.add_command(validate)
items.add_command(show_stats)

# Register task-type factory commands
items.add_command(create_forced_choice)
items.add_command(
    create_forced_choice_from_texts, name="create-forced-choice-from-texts"
)
items.add_command(create_likert_7, name="create-likert-7")
items.add_command(
    create_ordinal_scale_from_texts, name="create-ordinal-scale-from-texts"
)
items.add_command(create_nli)
items.add_command(create_categorical)
items.add_command(create_binary_from_texts, name="create-binary-from-texts")
items.add_command(create_multi_select_from_texts, name="create-multi-select-from-texts")
items.add_command(create_magnitude_from_texts, name="create-magnitude-from-texts")
items.add_command(create_free_text_from_texts, name="create-free-text-from-texts")
items.add_command(create_simple_cloze, name="create-simple-cloze")


# ==================== Validation Commands ====================


@items.command()
@click.argument("items_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--task-type",
    type=click.Choice(
        [
            "forced_choice",
            "ordinal_scale",
            "categorical",
            "binary",
            "multi_select",
            "magnitude",
            "free_text",
            "cloze",
        ],
        case_sensitive=False,
    ),
    required=True,
    help="Task type to validate against",
)
@click.option(
    "--strict",
    is_flag=True,
    help="Strict validation mode",
)
@click.pass_context
def validate_for_task_type(
    ctx: click.Context,
    items_file: Path,
    task_type: str,
    strict: bool,
) -> None:
    r"""Validate items for specific task type.

    Examples
    --------
    $ bead items validate-for-task-type items.jsonl --task-type forced_choice

    $ bead items validate-for-task-type items.jsonl \\
        --task-type ordinal_scale --strict
    """
    try:
        print_info(f"Validating items for task type: {task_type}")

        # Cast string to TaskType literal (validated by Click Choice)
        task_type_lit: TaskType = cast(TaskType, task_type)
        valid_count: int = 0
        invalid_count: int = 0
        errors: list[str] = []

        with open(items_file) as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    item = Item.model_validate_json(line)

                    if validate_item_for_task_type(item, task_type_lit):
                        valid_count += 1
                    else:
                        invalid_count += 1
                        errors.append(f"Line {line_num}: Invalid for {task_type}")

                except Exception as e:
                    invalid_count += 1
                    errors.append(f"Line {line_num}: {e}")

        # Display results
        table = Table(title="Validation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Count", justify="right", style="green")

        table.add_row("Valid items", str(valid_count))
        table.add_row(
            "Invalid items",
            str(invalid_count),
            style="red" if invalid_count else "green",
        )
        table.add_row("Total", str(valid_count + invalid_count))

        console.print(table)

        # Show errors if any
        if errors and strict:
            print_error("Validation errors:")
            for error in errors[:10]:
                console.print(f"  [red]✗[/red] {error}")
            if len(errors) > 10:
                console.print(f"  ... and {len(errors) - 10} more errors")

        if invalid_count > 0 and strict:
            ctx.exit(1)
        else:
            print_success(f"Validation complete: {valid_count} valid items")

    except Exception as e:
        print_error(f"Failed to validate items: {e}")
        ctx.exit(1)


@items.command()
@click.argument("items_file", type=click.Path(exists=True, path_type=Path))
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    help="Output file for inferred types (JSONL)",
)
@click.pass_context
def infer_task_type(
    ctx: click.Context,
    items_file: Path,
    output: Path | None,
) -> None:
    """Infer task type for each item.

    Examples
    --------
    $ bead items infer-task-type items.jsonl

    $ bead items infer-task-type items.jsonl --output types.jsonl
    """
    try:
        print_info("Inferring task types...")

        results: list[dict[str, str]] = []
        type_counts: dict[str, int] = {}

        with open(items_file) as f:
            line: str
            for line in f:
                line = line.strip()
                if not line:
                    continue

                item: Item = Item(**json.loads(line))

                try:
                    task_type_val: str = infer_task_type_from_item(item)
                    # task_type is already a string (Literal type), not enum
                    type_counts[task_type_val] = type_counts.get(task_type_val, 0) + 1
                    result_item: dict[str, str] = {
                        "item_id": str(item.id),
                        "task_type": task_type_val,
                    }
                    results.append(result_item)
                except ValueError:
                    result_unknown: dict[str, str] = {
                        "item_id": str(item.id),
                        "task_type": "unknown",
                    }
                    results.append(result_unknown)
                    type_counts["unknown"] = type_counts.get("unknown", 0) + 1

        # Display results
        table = Table(title="Task Type Distribution")
        table.add_column("Task Type", style="cyan")
        table.add_column("Count", justify="right", style="green")

        for task_type, count in sorted(type_counts.items()):
            table.add_row(task_type, str(count))

        console.print(table)

        # Save if output specified
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w") as f:
                for result in results:
                    f.write(json.dumps(result) + "\n")
            print_success(f"Saved task type inference results: {output}")

    except Exception as e:
        print_error(f"Failed to infer task types: {e}")
        ctx.exit(1)


@items.command()
@click.option(
    "--task-type",
    type=click.Choice(
        [
            "forced_choice",
            "ordinal_scale",
            "categorical",
            "binary",
            "multi_select",
            "magnitude",
            "free_text",
            "cloze",
        ],
        case_sensitive=False,
    ),
    required=True,
    help="Task type",
)
def get_task_requirements(task_type: str) -> None:
    """Get requirements for a task type.

    Examples
    --------
    $ bead items get-task-requirements --task-type forced_choice

    $ bead items get-task-requirements --task-type ordinal_scale
    """
    try:
        # Cast string to TaskType literal (validated by Click Choice)
        task_type_lit: TaskType = cast(TaskType, task_type)
        requirements: dict[str, list[str] | str] = get_task_type_requirements(
            task_type_lit
        )

        print_info(f"Requirements for task type: {task_type}")
        console.print()

        table = Table(show_header=False)
        table.add_column("Key", style="cyan", no_wrap=True)
        table.add_column("Value", style="white")

        key: str
        value: list[str] | str
        for key, value in requirements.items():
            if isinstance(value, list):
                # Requirements lists contain strings
                value_str: str = ", ".join(value)
            else:
                value_str = str(value)
            table.add_row(key, value_str)

        console.print(table)

    except Exception as e:
        print_error(f"Failed to get task requirements: {e}")
