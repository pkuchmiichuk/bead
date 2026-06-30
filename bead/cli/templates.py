"""Template filling commands for bead CLI.

This module provides commands for filling templates with lexical items
(Stage 2 of the bead pipeline).
"""

from __future__ import annotations

import csv as csv_module
import json
from pathlib import Path

import click
from didactic.api import ValidationError
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from bead.cli.utils import print_error, print_info, print_success
from bead.data.base import JsonValue
from bead.dsl.evaluator import DSLEvaluator
from bead.dsl.parser import parse
from bead.resources.constraints import Constraint
from bead.resources.lexicon import Lexicon
from bead.resources.template_collection import TemplateCollection
from bead.templates.combinatorics import count_combinations
from bead.templates.filler import FilledTemplate
from bead.templates.strategies import (
    ExhaustiveStrategy,
    RandomStrategy,
    StrategyFiller,
    StratifiedStrategy,
)

console = Console()


@click.group()
def templates() -> None:
    r"""Template filling commands (Stage 2).

    Commands for filling templates with lexical items using various strategies.

    \b
    Examples:
        $ bead templates fill template.jsonl lexicon.jsonl filled.jsonl \\
            --strategy exhaustive
        $ bead templates fill template.jsonl lexicon.jsonl filled.jsonl \\
            --strategy random --max-combinations 100
        $ bead templates list-filled filled.jsonl
        $ bead templates validate-filled filled.jsonl
        $ bead templates show-stats filled.jsonl
    """


@click.command()
@click.argument("template_file", type=click.Path(exists=True, path_type=Path))
@click.argument(
    "lexicon_files",
    nargs=-1,
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.argument("output_file", type=click.Path(path_type=Path))
@click.option(
    "--strategy",
    type=click.Choice(["exhaustive", "random", "stratified"]),
    default="exhaustive",
    help="Filling strategy to use",
)
@click.option(
    "--max-combinations",
    type=int,
    help="Maximum combinations for random/stratified strategies",
)
@click.option(
    "--random-seed",
    type=int,
    help="Random seed for reproducibility",
)
@click.option(
    "--grouping-property",
    help="Property for stratified strategy (e.g., 'pos', 'features.tense')",
)
@click.option(
    "--language-code",
    help="ISO 639 language code to filter items",
)
@click.option(
    "--constraints",
    type=click.Path(exists=True, path_type=Path),
    help="Path to constraints file (JSONL) to apply during filling",
)
@click.pass_context
def fill(
    ctx: click.Context,
    template_file: Path,
    lexicon_files: tuple[Path, ...],
    output_file: Path,
    strategy: str,
    max_combinations: int | None,
    random_seed: int | None,
    grouping_property: str | None,
    language_code: str | None,
    constraints: Path | None,
) -> None:
    r"""Fill templates with lexical items.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    template_file : Path
        Path to template file.
    lexicon_files : tuple[Path, ...]
        Paths to one or more lexicon files to merge.
    output_file : Path
        Path to output filled templates file.
    strategy : str
        Filling strategy name.
    max_combinations : int | None
        Maximum number of combinations.
    random_seed : int | None
        Random seed for reproducibility.
    grouping_property : str | None
        Property for stratified sampling.
    language_code : str | None
        ISO 639 language code filter.
    constraints : Path | None
        Path to constraints file (JSONL) to apply.

    Examples
    --------
    # Exhaustive filling with single lexicon
    $ bead templates fill template.jsonl lexicon.jsonl filled.jsonl \\
        --strategy exhaustive

    # Multiple lexicons
    $ bead templates fill tpl.jsonl nouns.jsonl verbs.jsonl filled.jsonl \\
        --strategy exhaustive

    # Random sampling
    $ bead templates fill template.jsonl lexicon.jsonl filled.jsonl \\
        --strategy random --max-combinations 100 --random-seed 42

    # Stratified sampling
    $ bead templates fill template.jsonl lexicon.jsonl filled.jsonl \\
        --strategy stratified --max-combinations 100 --grouping-property pos

    # With constraints
    $ bead templates fill template.jsonl lexicon.jsonl filled.jsonl \\
        --strategy exhaustive --constraints constraints.jsonl
    """
    try:
        # Validate strategy-specific options
        if strategy in ("random", "stratified") and max_combinations is None:
            print_error(f"--max-combinations required for {strategy} strategy")
            ctx.exit(1)

        if strategy == "stratified" and grouping_property is None:
            print_error("--grouping-property required for stratified strategy")
            ctx.exit(1)

        # Load and merge lexicons
        if not lexicon_files:
            print_error("At least one lexicon file is required")
            ctx.exit(1)

        print_info(f"Loading {len(lexicon_files)} lexicon(s)")
        merged_lexicon = Lexicon(name="merged", items=())

        for lex_file in lexicon_files:
            lex = Lexicon.from_jsonl(str(lex_file), lex_file.stem)
            print_info(f"  Loaded {len(lex)} items from {lex_file.name}")
            # Merge items
            merged_lexicon = merged_lexicon.with_(
                items=(*merged_lexicon.items, *lex.items)
            )

        print_info(f"Total merged lexicon: {len(merged_lexicon)} items")
        lexicon = merged_lexicon

        # Load templates
        print_info(f"Loading templates from {template_file}")
        template_collection = TemplateCollection.from_jsonl(
            str(template_file), "templates"
        )
        print_info(f"Loaded {len(template_collection)} templates")

        # Load and apply constraints if provided
        if constraints:
            print_info(f"Loading constraints from {constraints}")
            loaded_constraints: list[Constraint] = []

            with open(constraints, encoding="utf-8") as f:
                for line_num, line in enumerate(f, start=1):
                    line = line.strip()
                    if not line:
                        continue

                    try:
                        constraint = Constraint.model_validate_json(line)
                        loaded_constraints.append(constraint)
                    except json.JSONDecodeError as e:
                        print_error(f"Invalid JSON on line {line_num}: {e}")
                        ctx.exit(1)
                    except ValidationError as e:
                        print_error(f"Invalid constraint on line {line_num}: {e}")
                        ctx.exit(1)

            print_info(f"Loaded {len(loaded_constraints)} constraints")

            # Apply constraints to all templates
            template_collection = template_collection.with_(
                templates=tuple(
                    t.with_(constraints=(*t.constraints, *loaded_constraints))
                    for t in template_collection
                )
            )

            print_info(f"Applied constraints to {len(template_collection)} templates")

        # Create strategy
        filling_strategy: ExhaustiveStrategy | RandomStrategy | StratifiedStrategy
        if strategy == "exhaustive":
            filling_strategy = ExhaustiveStrategy()
        elif strategy == "random":
            assert max_combinations is not None
            filling_strategy = RandomStrategy(
                n_samples=max_combinations,
                seed=random_seed,
            )
        elif strategy == "stratified":
            assert max_combinations is not None
            assert grouping_property is not None
            filling_strategy = StratifiedStrategy(
                n_samples=max_combinations,
                grouping_property=grouping_property,
                seed=random_seed,
            )
        else:
            print_error(f"Unknown strategy: {strategy}")
            ctx.exit(1)

        # Create filler
        filler = StrategyFiller(lexicon=lexicon, strategy=filling_strategy)

        # Fill templates with progress
        all_filled: list[FilledTemplate] = []

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            task = progress.add_task(
                f"Filling {len(template_collection)} templates...",
                total=len(template_collection),
            )

            for template in template_collection:
                try:
                    filled_templates = filler.fill(template, language_code)
                    all_filled.extend(filled_templates)
                    progress.advance(task)
                except ValueError as e:
                    print_error(f"Failed to fill template '{template.name}': {e}")
                    continue

        # Save filled templates
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for filled in all_filled:
                f.write(filled.model_dump_json() + "\n")

        print_success(
            f"Created {len(all_filled)} filled templates from "
            f"{len(template_collection)} templates: {output_file}"
        )

    except ValidationError as e:
        print_error(f"Validation error: {e}")
        ctx.exit(1)
    except Exception as e:
        print_error(f"Failed to fill templates: {e}")
        ctx.exit(1)


@click.command()
@click.option(
    "--directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path.cwd(),
    help="Directory to search for filled template files",
)
@click.option(
    "--pattern",
    default="*.jsonl",
    help="File pattern to match (default: *.jsonl)",
)
@click.option(
    "--filter",
    "filter_expr",
    help="DSL expression to filter (e.g., 'slot_fillers.noun.lemma == \"cat\"')",
)
@click.pass_context
def list_filled(
    ctx: click.Context,
    directory: Path,
    pattern: str,
    filter_expr: str | None,
) -> None:
    """List filled template files in a directory.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    directory : Path
        Directory to search.
    pattern : str
        File pattern to match.
    filter_expr : str | None
        DSL expression to filter filled templates.

    Examples
    --------
    $ bead templates list-filled
    $ bead templates list-filled --directory filled_templates/
    $ bead templates list-filled --pattern "filled_*.jsonl"
    $ bead templates list-filled --filter "slot_fillers.noun.lemma == 'cat'"
    $ bead templates list-filled --filter "len(slot_fillers) > 2"
    """
    try:
        files = list(directory.glob(pattern))

        if not files:
            print_info(f"No files found in {directory} matching {pattern}")
            return

        # Parse filter expression if provided
        filter_ast = None
        evaluator = None
        if filter_expr:
            try:
                filter_ast = parse(filter_expr)
                evaluator = DSLEvaluator()
                print_info(f"Filtering with expression: {filter_expr}")
            except Exception as e:
                print_error(f"Invalid filter expression: {e}")
                ctx.exit(1)

        table = Table(title=f"Filled Templates in {directory}")
        table.add_column("File", style="cyan")
        table.add_column("Count", justify="right", style="yellow")
        table.add_column("Filtered", justify="right", style="magenta")
        table.add_column("Strategy", style="green")
        table.add_column("Sample", style="white")

        for file_path in sorted(files):
            try:
                # Count filled templates and get metadata
                with open(file_path, encoding="utf-8") as f:
                    lines = [line.strip() for line in f if line.strip()]

                if not lines:
                    continue

                # Apply filter if provided
                filtered_count = 0
                if filter_ast and evaluator:
                    for line in lines:
                        try:
                            filled_template = FilledTemplate.model_validate_json(line)
                            # Create evaluation context
                            context = {"self": filled_template}
                            # Evaluate filter
                            if evaluator.evaluate(filter_ast, context):
                                filtered_count += 1
                        except Exception:
                            continue
                else:
                    filtered_count = len(lines)

                if filtered_count == 0:
                    continue

                # Parse first filled template for metadata
                first_data = json.loads(lines[0])
                strategy_name = first_data.get("strategy_name", "N/A")
                rendered = first_data.get("rendered_text", "N/A")

                # Truncate long rendered text
                if len(rendered) > 40:
                    rendered = rendered[:37] + "..."

                table.add_row(
                    str(file_path.name),
                    str(len(lines)),
                    str(filtered_count) if filter_expr else "N/A",
                    strategy_name,
                    rendered,
                )
            except Exception:
                # Skip files that can't be parsed
                continue

        console.print(table)

    except Exception as e:
        print_error(f"Failed to list filled templates: {e}")
        ctx.exit(1)


@click.command()
@click.argument("filled_file", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def validate_filled(ctx: click.Context, filled_file: Path) -> None:
    """Validate a filled templates file.

    Checks that all filled templates are properly formatted.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    filled_file : Path
        Path to filled templates file.

    Examples
    --------
    $ bead templates validate-filled filled.jsonl
    """
    try:
        print_info(f"Validating filled templates: {filled_file}")

        count = 0
        errors: list[str] = []

        with open(filled_file, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    FilledTemplate.model_validate_json(line)
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
            print_success(f"Filled templates file is valid: {count} filled templates")

    except Exception as e:
        print_error(f"Failed to validate filled templates: {e}")
        ctx.exit(1)


@click.command()
@click.argument("filled_file", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def show_stats(ctx: click.Context, filled_file: Path) -> None:
    """Show statistics about filled templates.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    filled_file : Path
        Path to filled templates file.

    Examples
    --------
    $ bead templates show-stats filled.jsonl
    """
    try:
        print_info(f"Analyzing filled templates: {filled_file}")

        # Collect statistics
        total_count = 0
        templates_seen: set[str] = set()
        strategies_used: dict[str, int] = {}
        text_lengths: list[int] = []

        with open(filled_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    filled = FilledTemplate.model_validate_json(line)

                    total_count += 1
                    templates_seen.add(filled.template_name)
                    strategies_used[filled.strategy_name] = (
                        strategies_used.get(filled.strategy_name, 0) + 1
                    )
                    text_lengths.append(len(filled.rendered_text))

                except Exception:
                    continue

        if total_count == 0:
            print_error("No valid filled templates found")
            ctx.exit(1)

        # Calculate statistics
        avg_length = sum(text_lengths) / len(text_lengths) if text_lengths else 0
        min_length = min(text_lengths) if text_lengths else 0
        max_length = max(text_lengths) if text_lengths else 0

        # Display statistics table
        table = Table(title="Filled Template Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Total Filled Templates", str(total_count))
        table.add_row("Unique Template Names", str(len(templates_seen)))
        table.add_row("", "")  # Separator

        for strategy, count in sorted(strategies_used.items()):
            table.add_row(f"Strategy: {strategy}", str(count))

        table.add_row("", "")  # Separator
        table.add_row("Avg Text Length", f"{avg_length:.1f}")
        table.add_row("Min Text Length", str(min_length))
        table.add_row("Max Text Length", str(max_length))

        console.print(table)

        # Show sample templates
        if templates_seen:
            console.print("\n[cyan]Sample Template Names:[/cyan]")
            for name in sorted(templates_seen)[:5]:
                console.print(f"  • {name}")
            if len(templates_seen) > 5:
                console.print(f"  ... and {len(templates_seen) - 5} more")

    except Exception as e:
        print_error(f"Failed to show statistics: {e}")
        ctx.exit(1)


@click.command()
@click.argument("template_file", type=click.Path(exists=True, path_type=Path))
@click.argument(
    "lexicon_files",
    nargs=-1,
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.option(
    "--language-code",
    help="ISO 639 language code to filter items",
)
@click.pass_context
def estimate(
    ctx: click.Context,
    template_file: Path,
    lexicon_files: tuple[Path, ...],
    language_code: str | None,
) -> None:
    r"""Estimate total combinations for exhaustive filling.

    Calculates the total number of combinations that would be generated
    by exhaustive template filling without actually generating them.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    template_file : Path
        Path to template file.
    lexicon_files : tuple[Path, ...]
        Paths to one or more lexicon files to merge.
    language_code : str | None
        ISO 639 language code filter.

    Examples
    --------
    # Estimate combinations with single lexicon
    $ bead templates estimate template.jsonl lexicon.jsonl

    # With multiple lexicons
    $ bead templates estimate template.jsonl nouns.jsonl verbs.jsonl

    # With language filter
    $ bead templates estimate template.jsonl lexicon.jsonl --language-code eng
    """
    try:
        # Load and merge lexicons
        if not lexicon_files:
            print_error("At least one lexicon file is required")
            ctx.exit(1)

        print_info(f"Loading {len(lexicon_files)} lexicon(s)")
        merged_lexicon = Lexicon(name="merged", items=())

        for lex_file in lexicon_files:
            lex = Lexicon.from_jsonl(str(lex_file), lex_file.stem)
            merged_lexicon = merged_lexicon.with_(
                items=(*merged_lexicon.items, *lex.items)
            )

        print_info(f"Total merged lexicon: {len(merged_lexicon)} items")
        lexicon = merged_lexicon

        # Load templates
        print_info(f"Loading templates from {template_file}")
        template_collection = TemplateCollection.from_jsonl(
            str(template_file), "templates"
        )

        # Calculate estimates for each template
        table = Table(title="Combination Estimates")
        table.add_column("Template", style="cyan")
        table.add_column("Slots", justify="right", style="yellow")
        table.add_column("Combinations", justify="right", style="green")

        total_combinations = 0

        for template in template_collection:
            # Get lexical items for each slot
            slot_lists: list[list[str]] = []
            for _slot_name in template.slots:
                items = [
                    item.lemma
                    for item in lexicon
                    if language_code is None or item.language_code == language_code
                ]
                slot_lists.append(items)

            # Estimate combinations
            num_combos = count_combinations(*slot_lists)
            total_combinations += num_combos

            table.add_row(
                template.name,
                str(len(template.slots)),
                f"{num_combos:,}",
            )

        # Add total row
        table.add_section()
        table.add_row(
            "[bold]TOTAL[/bold]",
            "",
            f"[bold]{total_combinations:,}[/bold]",
        )

        console.print(table)

        # Warn if combinations are very large
        if total_combinations > 1_000_000:
            print_info(
                "\n⚠️  Warning: Exhaustive filling will generate over 1 million "
                "combinations. Consider using random or stratified strategies instead."
            )
        elif total_combinations > 100_000:
            print_info(
                "\n⚠️  Warning: Exhaustive filling will generate over 100K "
                "combinations. This may take significant time."
            )

    except Exception as e:
        print_error(f"Failed to estimate combinations: {e}")
        ctx.exit(1)


@click.command()
@click.argument("filled_file", type=click.Path(exists=True, path_type=Path))
@click.argument("output_file", type=click.Path(path_type=Path))
@click.option(
    "--expression",
    help="Filter expression (DSL) to apply to filled templates",
)
@click.option(
    "--min-length",
    type=int,
    help="Minimum text length",
)
@click.option(
    "--max-length",
    type=int,
    help="Maximum text length",
)
@click.option(
    "--template-name",
    help="Filter by template name (exact match)",
)
@click.option(
    "--strategy",
    help="Filter by strategy name",
)
@click.pass_context
def filter_filled(
    ctx: click.Context,
    filled_file: Path,
    output_file: Path,
    expression: str | None,
    min_length: int | None,
    max_length: int | None,
    template_name: str | None,
    strategy: str | None,
) -> None:
    """Filter filled templates by various criteria.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    filled_file : Path
        Path to filled templates file.
    output_file : Path
        Path to output filtered file.
    expression : str | None
        DSL expression for filtering.
    min_length : int | None
        Minimum text length.
    max_length : int | None
        Maximum text length.
    template_name : str | None
        Template name filter.
    strategy : str | None
        Strategy name filter.

    Examples
    --------
    $ bead templates filter-filled filled.jsonl filtered.jsonl --min-length 10
    $ bead templates filter-filled filled.jsonl filtered.jsonl --template-name active
    """
    try:
        print_info(f"Filtering filled templates from: {filled_file}")

        filtered_count = 0
        total_count = 0

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as out_f:
            with open(filled_file, encoding="utf-8") as in_f:
                for line in in_f:
                    line = line.strip()
                    if not line:
                        continue

                    total_count += 1

                    try:
                        filled = FilledTemplate.model_validate_json(line)

                        # Apply filters
                        if min_length and len(filled.rendered_text) < min_length:
                            continue
                        if max_length and len(filled.rendered_text) > max_length:
                            continue
                        if template_name and filled.template_name != template_name:
                            continue
                        if strategy and filled.strategy_name != strategy:
                            continue

                        # DSL expression filtering would go here
                        if expression:
                            print_info(
                                "DSL expression filtering not yet implemented, skipping"
                            )

                        # Passed all filters
                        out_f.write(line + "\n")
                        filtered_count += 1

                    except Exception as e:
                        print_error(f"Error processing line: {e}")
                        continue

        print_success(
            f"Filtered {filtered_count} of {total_count} templates: {output_file}"
        )

    except Exception as e:
        print_error(f"Failed to filter filled templates: {e}")
        ctx.exit(1)


@click.command()
@click.argument("input_files", nargs=-1, type=click.Path(exists=True, path_type=Path))
@click.argument("output_file", type=click.Path(path_type=Path))
@click.option(
    "--deduplicate",
    is_flag=True,
    help="Remove duplicates by UUID",
)
@click.pass_context
def merge_filled(
    ctx: click.Context,
    input_files: tuple[Path, ...],
    output_file: Path,
    deduplicate: bool,
) -> None:
    """Merge multiple filled template files.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    input_files : tuple[Path, ...]
        Input filled template files.
    output_file : Path
        Output merged file.
    deduplicate : bool
        Remove duplicates by UUID.

    Examples
    --------
    $ bead templates merge-filled file1.jsonl file2.jsonl merged.jsonl
    $ bead templates merge-filled *.jsonl merged.jsonl --deduplicate
    """
    try:
        if not input_files:
            print_error("No input files provided")
            ctx.exit(1)

        print_info(f"Merging {len(input_files)} filled template files")

        seen_ids: set[str] = set()
        merged_count = 0
        duplicate_count = 0

        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as out_f:
            for input_file in input_files:
                print_info(f"  Processing: {input_file}")
                with open(input_file, encoding="utf-8") as in_f:
                    for line in in_f:
                        line = line.strip()
                        if not line:
                            continue

                        try:
                            filled = FilledTemplate.model_validate_json(line)

                            if deduplicate:
                                if str(filled.id) in seen_ids:
                                    duplicate_count += 1
                                    continue
                                seen_ids.add(str(filled.id))

                            out_f.write(line + "\n")
                            merged_count += 1

                        except Exception as e:
                            print_error(f"Error processing line from {input_file}: {e}")
                            continue

        print_success(f"Merged {merged_count} filled templates: {output_file}")
        if deduplicate and duplicate_count > 0:
            print_info(f"Removed {duplicate_count} duplicates")

    except Exception as e:
        print_error(f"Failed to merge filled templates: {e}")
        ctx.exit(1)


@click.command()
@click.argument("filled_file", type=click.Path(exists=True, path_type=Path))
@click.argument("output_file", type=click.Path(path_type=Path))
@click.pass_context
def export_csv(
    ctx: click.Context,
    filled_file: Path,
    output_file: Path,
) -> None:
    """Export filled templates to CSV format.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    filled_file : Path
        Input filled templates file (JSONL).
    output_file : Path
        Output CSV file.

    Examples
    --------
    $ bead templates export-csv filled.jsonl filled.csv
    """
    try:
        print_info(f"Exporting filled templates to CSV: {output_file}")

        filled_templates: list[FilledTemplate] = []

        with open(filled_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    filled = FilledTemplate.model_validate_json(line)
                    filled_templates.append(filled)
                except Exception:
                    continue

        if not filled_templates:
            print_error("No valid filled templates found")
            ctx.exit(1)

        # Write to CSV
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", newline="", encoding="utf-8") as f:
            writer = csv_module.writer(f)

            # Header
            writer.writerow(
                [
                    "id",
                    "template_id",
                    "template_name",
                    "rendered_text",
                    "strategy_name",
                    "slot_count",
                ]
            )

            # Data
            for filled in filled_templates:
                writer.writerow(
                    [
                        str(filled.id),
                        str(filled.template_id),
                        filled.template_name,
                        filled.rendered_text,
                        filled.strategy_name,
                        len(filled.slot_fillers),
                    ]
                )

        print_success(
            f"Exported {len(filled_templates)} filled templates to CSV: {output_file}"
        )

    except Exception as e:
        print_error(f"Failed to export to CSV: {e}")
        ctx.exit(1)


@click.command()
@click.argument("filled_file", type=click.Path(exists=True, path_type=Path))
@click.argument("output_file", type=click.Path(path_type=Path))
@click.option(
    "--pretty",
    is_flag=True,
    help="Pretty-print JSON with indentation",
)
@click.pass_context
def export_json(
    ctx: click.Context,
    filled_file: Path,
    output_file: Path,
    pretty: bool,
) -> None:
    """Export filled templates to JSON array format.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    filled_file : Path
        Input filled templates file (JSONL).
    output_file : Path
        Output JSON file.
    pretty : bool
        Pretty-print with indentation.

    Examples
    --------
    $ bead templates export-json filled.jsonl filled.json
    $ bead templates export-json filled.jsonl filled.json --pretty
    """
    try:
        print_info(f"Exporting filled templates to JSON: {output_file}")

        filled_templates: list[dict[str, JsonValue]] = []

        with open(filled_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue

                try:
                    filled = FilledTemplate.model_validate_json(line)
                    filled_templates.append(json.loads(filled.model_dump_json()))
                except Exception:
                    continue

        if not filled_templates:
            print_error("No valid filled templates found")
            ctx.exit(1)

        # Write to JSON
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            if pretty:
                json.dump(filled_templates, f, indent=2, ensure_ascii=False)
            else:
                json.dump(filled_templates, f, ensure_ascii=False)

        print_success(
            f"Exported {len(filled_templates)} filled templates to JSON: {output_file}"
        )

    except Exception as e:
        print_error(f"Failed to export to JSON: {e}")
        ctx.exit(1)


@click.command()
@click.argument("template_file", type=click.Path(exists=True, path_type=Path))
@click.argument(
    "lexicon_files",
    nargs=-1,
    type=click.Path(exists=True, path_type=Path),
    required=True,
)
@click.argument("output_file", type=click.Path(path_type=Path))
@click.option(
    "--n-samples",
    type=int,
    required=True,
    help="Number of samples to generate",
)
@click.option(
    "--seed",
    type=int,
    help="Random seed for reproducibility",
)
@click.option(
    "--language-code",
    help="ISO 639 language code to filter items",
)
@click.pass_context
def sample_combinations(
    ctx: click.Context,
    template_file: Path,
    lexicon_files: tuple[Path, ...],
    output_file: Path,
    n_samples: int,
    seed: int | None,
    language_code: str | None,
) -> None:
    r"""Sample template-lexicon combinations with stratified sampling.

    Uses stratified sampling to ensure diverse coverage of the combination space
    without exhaustive generation.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    template_file : Path
        Path to template file.
    lexicon_files : tuple[Path, ...]
        Paths to one or more lexicon files to merge.
    output_file : Path
        Path to output sampled combinations.
    n_samples : int
        Number of samples to generate.
    seed : int | None
        Random seed.
    language_code : str | None
        Language code filter.

    Examples
    --------
    # Single lexicon
    $ bead templates sample-combinations template.jsonl lexicon.jsonl samples.jsonl \\
        --n-samples 1000 --seed 42

    # Multiple lexicons
    $ bead templates sample-combinations tpl.jsonl nouns.jsonl verbs.jsonl out.jsonl \\
        --n-samples 1000 --seed 42
    """
    try:
        # Load and merge lexicons
        if not lexicon_files:
            print_error("At least one lexicon file is required")
            ctx.exit(1)

        print_info(f"Loading {len(lexicon_files)} lexicon(s)")
        merged_lexicon = Lexicon(name="merged", items=())

        for lex_file in lexicon_files:
            lex = Lexicon.from_jsonl(str(lex_file), lex_file.stem)
            print_info(f"  Loaded {len(lex)} items from {lex_file.name}")
            merged_lexicon = merged_lexicon.with_(
                items=(*merged_lexicon.items, *lex.items)
            )

        print_info(f"Total merged lexicon: {len(merged_lexicon)} items")
        lexicon = merged_lexicon

        # Load templates
        print_info(f"Loading templates from {template_file}")
        template_collection = TemplateCollection.from_jsonl(
            str(template_file), "templates"
        )

        # Use random strategy for sampling
        print_info(f"Generating {n_samples} stratified samples")
        strategy = RandomStrategy(n_samples=n_samples, seed=seed)
        filler = StrategyFiller(lexicon=lexicon, strategy=strategy)

        # Fill templates
        all_filled: list[FilledTemplate] = []
        for template in template_collection:
            try:
                filled_templates = filler.fill(template, language_code)
                all_filled.extend(filled_templates)
            except ValueError as e:
                print_error(f"Failed to fill template '{template.name}': {e}")
                continue

        # Save sampled combinations
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for filled in all_filled:
                f.write(filled.model_dump_json() + "\n")

        print_success(
            f"Generated {len(all_filled)} sampled combinations: {output_file}"
        )

    except Exception as e:
        print_error(f"Failed to sample combinations: {e}")
        ctx.exit(1)


# Register commands
templates.add_command(fill)
templates.add_command(list_filled)
templates.add_command(validate_filled)
templates.add_command(show_stats)
templates.add_command(estimate, name="estimate-combinations")
templates.add_command(filter_filled)
templates.add_command(merge_filled)
templates.add_command(export_csv)
templates.add_command(export_json)
templates.add_command(sample_combinations)
