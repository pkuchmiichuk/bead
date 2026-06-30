"""Constraint builder commands for bead CLI.

This module provides commands for creating constraints for template slot filling
using three types: extensional (value whitelists), intensional (feature-based
expressions), and relational (cross-slot constraints).
"""

from __future__ import annotations

from pathlib import Path

import click
from rich.console import Console

from bead.cli.utils import print_error, print_info, print_success
from bead.resources.constraints import Constraint

console = Console()


@click.command()
@click.argument("output_file", type=click.Path(path_type=Path))
@click.option(
    "--type",
    "constraint_type",
    type=click.Choice(["extensional", "intensional", "relational"]),
    required=True,
    help="Type of constraint to create",
)
@click.option(
    "--slot",
    help="Slot name for single-slot constraints (extensional, intensional)",
)
@click.option(
    "--expression",
    help="DSL expression for intensional constraints (e.g., \"self.pos == 'VERB'\")",
)
@click.option(
    "--relation",
    help='DSL expression for relational constraints (e.g., "a.number == b.number")',
)
@click.option(
    "--values-file",
    type=click.Path(exists=True, path_type=Path),
    help="File with values for extensional constraints (one value per line)",
)
@click.option(
    "--values",
    help="Comma-separated values for extensional constraints",
)
@click.option(
    "--context-var-name",
    default="allowed_values",
    help="Context variable name for extensional constraints (default: allowed_values)",
)
@click.option(
    "--description",
    help="Human-readable description of the constraint",
)
@click.option(
    "--prop-name",
    default="lemma",
    help="Property to check for extensional constraints (default: lemma)",
)
@click.pass_context
def create_constraint(
    ctx: click.Context,
    output_file: Path,
    constraint_type: str,
    slot: str | None,
    expression: str | None,
    relation: str | None,
    values_file: Path | None,
    values: str | None,
    context_var_name: str,
    description: str | None,
    prop_name: str,
) -> None:
    r"""Create a constraint for template slot filling.

    Supports three types of constraints:

    \b
    1. EXTENSIONAL: Whitelist of allowed values
       $ bead resources create-constraint constraints.jsonl \
           --type extensional \
           --slot verb \
           --values "walk,run,jump" \
           --description "Motion verbs"

    \b
    2. INTENSIONAL: Feature-based DSL expression
       $ bead resources create-constraint constraints.jsonl \
           --type intensional \
           --slot verb \
           --expression "self.pos == 'VERB' and self.features.tense == 'past'" \
           --description "Past tense verbs"

    \b
    3. RELATIONAL: Cross-slot agreement
       $ bead resources create-constraint constraints.jsonl \
           --type relational \
           --relation "subject.features.number == verb.features.number" \
           --description "Subject-verb agreement"

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    output_file : Path
        Path to output constraints file (JSONL).
    constraint_type : str
        Type of constraint (extensional, intensional, relational).
    slot : str | None
        Slot name for single-slot constraints.
    expression : str | None
        DSL expression for intensional constraints.
    relation : str | None
        DSL expression for relational constraints.
    values_file : Path | None
        File with values for extensional constraints.
    values : str | None
        Comma-separated values for extensional constraints.
    context_var_name : str
        Context variable name for extensional constraints.
    description : str | None
        Description of the constraint.
    prop_name : str
        Property to check for extensional constraints.

    Examples
    --------
    # Extensional constraint from file
    $ bead resources create-constraint constraints.jsonl \\
        --type extensional \\
        --slot verb \\
        --values-file motion_verbs.txt \\
        --description "Motion verbs only"

    # Extensional constraint from values
    $ bead resources create-constraint constraints.jsonl \\
        --type extensional \\
        --slot noun \\
        --values "cat,dog,bird" \\
        --prop-name lemma

    # Intensional constraint
    $ bead resources create-constraint constraints.jsonl \\
        --type intensional \\
        --slot verb \\
        --expression "self.pos == 'VERB' and self.features.number == 'singular'"

    # Relational constraint
    $ bead resources create-constraint constraints.jsonl \\
        --type relational \\
        --relation "det.lemma != 'a' or noun.features.number == 'singular'" \\
        --description "Article-noun agreement"
    """
    try:
        constraint: Constraint | None = None

        if constraint_type == "extensional":
            # Validate required parameters
            if not values_file and not values:
                print_error("Extensional constraints require --values-file or --values")
                ctx.exit(1)

            if not slot:
                print_error("Extensional constraints require --slot")
                ctx.exit(1)

            # Load values
            value_set: set[str]
            if values_file:
                print_info(f"Loading values from {values_file}")
                with open(values_file, encoding="utf-8") as f:
                    value_set = {line.strip() for line in f if line.strip()}
            else:
                assert values is not None
                value_set = {v.strip() for v in values.split(",") if v.strip()}

            if not value_set:
                print_error("No values provided")
                ctx.exit(1)

            # Create constraint
            # Expression uses 'self' for single-slot constraints
            expr = f"self.{prop_name} in {context_var_name}"
            constraint = Constraint(
                expression=expr,
                context={context_var_name: tuple(sorted(value_set))},
                description=description
                or f"Allowed {prop_name} values for {slot}: {len(value_set)} values",
            )
            print_success(
                f"Created extensional constraint for '{slot}' "
                f"with {len(value_set)} values"
            )

        elif constraint_type == "intensional":
            # Validate required parameters
            if not expression:
                print_error("Intensional constraints require --expression")
                ctx.exit(1)

            if not slot:
                print_info(
                    "No --slot specified; constraint will apply to template level"
                )

            # Validate expression starts with 'self.' for slot-level constraints
            if slot and not expression.startswith("self."):
                print_error(
                    f"Intensional expression for slot '{slot}' must start with 'self.' "
                    f"to refer to the slot filler.\n\n"
                    f"Example: self.pos == 'VERB' and self.features.tense == 'past'"
                )
                ctx.exit(1)

            # Create constraint
            constraint = Constraint(
                expression=expression,
                description=description or f"Intensional constraint: {expression}",
            )
            print_success(
                f"Created intensional constraint{' for slot ' + slot if slot else ''}"
            )

        elif constraint_type == "relational":
            # Validate required parameters
            if not relation:
                print_error("Relational constraints require --relation")
                ctx.exit(1)

            # Validate expression does NOT start with 'self.' (multi-slot)
            if relation.startswith("self."):
                print_error(
                    "Relational constraints are multi-slot; do not use 'self.'.\n\n"
                    "Use slot names as variables instead.\n\n"
                    "Example: subject.features.number == verb.features.number"
                )
                ctx.exit(1)

            # Create constraint
            constraint = Constraint(
                expression=relation,
                description=description or f"Relational constraint: {relation}",
            )
            print_success("Created relational constraint")

        else:
            print_error(f"Unknown constraint type: {constraint_type}")
            ctx.exit(1)

        # Save constraint to JSONL
        assert constraint is not None
        output_file.parent.mkdir(parents=True, exist_ok=True)

        # Append to file if it exists, create new file otherwise
        mode = "a" if output_file.exists() else "w"
        with open(output_file, mode, encoding="utf-8") as f:
            f.write(constraint.model_dump_json() + "\n")

        print_info(f"Constraint written to: {output_file}")

        # Show example usage
        if constraint_type == "extensional":
            console.print("\n[cyan]Usage in template:[/cyan]")
            console.print(
                f"  Add to Slot(name='{slot}', constraints=[...loaded from file...])"
            )
        elif constraint_type == "intensional":
            console.print("\n[cyan]Usage in template:[/cyan]")
            if slot:
                console.print(
                    f"  Add to Slot(name='{slot}', constraints=[...from file...])"
                )
            else:
                console.print("  Add to Template.constraints=[...loaded from file...]")
        elif constraint_type == "relational":
            console.print("\n[cyan]Usage in template:[/cyan]")
            console.print("  Add to Template.constraints=[...loaded from file...]")

    except Exception as e:
        print_error(f"Failed to create constraint: {e}")
        ctx.exit(1)
