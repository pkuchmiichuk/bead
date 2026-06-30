"""Resource management commands for bead CLI.

This module provides commands for creating, listing, and validating
lexicons and templates (Stage 1 of the bead pipeline).
"""

from __future__ import annotations

import csv
import json
import re
from itertools import product
from pathlib import Path
from typing import Any, cast

import click
from didactic.api import ValidationError
from rich.console import Console
from rich.table import Table

from bead.cli.constraint_builders import create_constraint
from bead.cli.resource_loaders import (
    import_framenet,
    import_propbank,
    import_unimorph,
    import_verbnet,
)
from bead.cli.utils import print_error, print_info, print_success
from bead.data.base import JsonValue
from bead.resources.lexical_item import LexicalItem
from bead.resources.lexicon import Lexicon
from bead.resources.template import Slot, Template
from bead.resources.template_collection import TemplateCollection

console = Console()


@click.group()
def resources() -> None:
    r"""Resource management commands (Stage 1).

    Commands for creating, validating, and managing lexicons and templates.

    \b
    Examples:
        $ bead resources create-lexicon lexicon.jsonl --name verbs \\
            --from-csv verbs.csv
        $ bead resources create-template template.jsonl --name transitive \\
            --template-string "{subject} {verb} {object}"
        $ bead resources list-lexicons --directory lexicons/
        $ bead resources validate-lexicon lexicon.jsonl
    """


@resources.command()
@click.argument("output_file", type=click.Path(path_type=Path))
@click.option("--name", required=True, help="Lexicon name")
@click.option(
    "--from-csv",
    "csv_file",
    type=click.Path(exists=True, path_type=Path),
    help="Create from CSV file (requires 'lemma' column, optional 'pos', 'form', etc.)",
)
@click.option(
    "--from-json",
    "json_file",
    type=click.Path(exists=True, path_type=Path),
    help="Create from JSON file (array of lexical item objects)",
)
@click.option("--language-code", help="ISO 639 language code (e.g., 'eng', 'en')")
@click.option("--description", help="Description of the lexicon")
@click.pass_context
def create_lexicon(
    ctx: click.Context,
    output_file: Path,
    name: str,
    csv_file: Path | None,
    json_file: Path | None,
    language_code: str | None,
    description: str | None,
) -> None:
    r"""Create a lexicon from various sources.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    output_file : Path
        Path to output lexicon file.
    name : str
        Name for the lexicon.
    csv_file : Path | None
        Path to CSV source file.
    json_file : Path | None
        Path to JSON source file.
    language_code : str | None
        ISO 639 language code.
    description : str | None
        Description of the lexicon.

    Examples
    --------
    # Create from CSV file
    $ bead resources create-lexicon lexicon.jsonl --name verbs --from-csv verbs.csv

    # Create from JSON file
    $ bead resources create-lexicon lexicon.jsonl --name verbs --from-json verbs.json

    # With language code
    $ bead resources create-lexicon lexicon.jsonl --name verbs \\
        --from-csv verbs.csv --language-code eng
    """
    try:
        # Validate that exactly one source is provided
        sources = [csv_file, json_file]
        provided_sources = [s for s in sources if s is not None]

        if len(provided_sources) == 0:
            print_error("Must provide one source: --from-csv or --from-json")
            ctx.exit(1)
        elif len(provided_sources) > 1:
            print_error("Only one source allowed: --from-csv or --from-json")
            ctx.exit(1)

        # Create lexicon
        lexicon = Lexicon(
            name=name,
            language_code=language_code,
            description=description,
        )

        # Determine language code for items (default to "eng" if not specified)
        item_language_code = language_code or "eng"

        # Load from source
        if csv_file:
            print_info(f"Loading lexical items from CSV: {csv_file}")
            with open(csv_file, encoding="utf-8") as f:
                reader = csv.DictReader(f)
                for row in reader:
                    if "lemma" not in row:
                        print_error("CSV must have 'lemma' column")
                        ctx.exit(1)

                    item_data: dict[str, Any] = {
                        "lemma": row["lemma"],
                        "language_code": item_language_code,
                    }

                    if "form" in row and row["form"]:
                        item_data["form"] = row["form"]
                    if "source" in row and row["source"]:
                        item_data["source"] = row["source"]

                    # Build features dict from pos, feature_ columns, and attr_ columns
                    features: dict[str, Any] = {}

                    # Add pos to features if present
                    if "pos" in row and row["pos"]:
                        features["pos"] = row["pos"]

                    # Extract features (columns with feature_ prefix)
                    for key, value in row.items():
                        if key.startswith("feature_") and value:
                            features[key[8:]] = value

                    # Extract attributes (columns with attr_ prefix) into features
                    for key, value in row.items():
                        if key.startswith("attr_") and value:
                            features[key[5:]] = value

                    if features:
                        item_data["features"] = features

                    item = LexicalItem(**item_data)
                    lexicon = lexicon.with_item(item)
        elif json_file:
            print_info(f"Loading lexical items from JSON: {json_file}")
            with open(json_file, encoding="utf-8") as f:
                raw_data = json.load(f)

            if not isinstance(raw_data, list):
                print_error("JSON file must contain an array of lexical items")
                ctx.exit(1)

            data = cast(list[dict[str, JsonValue]], raw_data)

            for raw_item_untyped in data:
                # Extract required lemma field
                if "lemma" not in raw_item_untyped or not isinstance(
                    raw_item_untyped["lemma"], str
                ):
                    continue
                lemma: str = raw_item_untyped["lemma"]

                # Extract optional form field
                form: str | None = None
                if "form" in raw_item_untyped and isinstance(
                    raw_item_untyped["form"], str
                ):
                    form = raw_item_untyped["form"]

                # Extract language_code
                lang_code: str = item_language_code
                if "language_code" in raw_item_untyped and isinstance(
                    raw_item_untyped["language_code"], str
                ):
                    lang_code = raw_item_untyped["language_code"]

                # Extract optional source field
                source: str | None = None
                if "source" in raw_item_untyped and isinstance(
                    raw_item_untyped["source"], str
                ):
                    source = raw_item_untyped["source"]

                # Handle features dict - copy all key-value pairs
                json_features: dict[str, str | int | float | bool | None] = {}
                if "features" in raw_item_untyped:
                    features_value = raw_item_untyped["features"]
                    if isinstance(features_value, dict):
                        for k, v in features_value.items():
                            if isinstance(v, str | int | float | bool) or v is None:
                                json_features[k] = v

                # Move pos to features if present at top level
                if "pos" in raw_item_untyped and isinstance(
                    raw_item_untyped["pos"], str
                ):
                    json_features["pos"] = raw_item_untyped["pos"]

                # Build LexicalItem
                if form is None and source is None:
                    item = LexicalItem(
                        lemma=lemma, language_code=lang_code, features=json_features
                    )  # type: ignore[arg-type]
                elif form is None:
                    item = LexicalItem(
                        lemma=lemma,
                        language_code=lang_code,
                        features=json_features,
                        source=source,
                    )  # type: ignore[arg-type]
                elif source is None:
                    item = LexicalItem(
                        lemma=lemma,
                        form=form,
                        language_code=lang_code,
                        features=json_features,
                    )  # type: ignore[arg-type]
                else:
                    item = LexicalItem(
                        lemma=lemma,
                        form=form,
                        language_code=lang_code,
                        features=json_features,
                        source=source,
                    )  # type: ignore[arg-type]

                lexicon = lexicon.with_item(item)
        # Save lexicon
        output_file.parent.mkdir(parents=True, exist_ok=True)
        lexicon.to_jsonl(str(output_file))

        print_success(
            f"Created lexicon '{name}' with {len(lexicon)} items: {output_file}"
        )

    except ValidationError as e:
        print_error(f"Validation error: {e}")
        ctx.exit(1)
    except Exception as e:
        print_error(f"Failed to create lexicon: {e}")
        ctx.exit(1)


@resources.command()
@click.argument("output_file", type=click.Path(path_type=Path))
@click.option("--name", required=True, help="Template name")
@click.option(
    "--template-string",
    required=True,
    help="Template string with {slot_name} placeholders",
)
@click.option("--language-code", help="ISO 639 language code")
@click.option("--description", help="Template description")
@click.option(
    "--slot",
    "slots",
    multiple=True,
    help=(
        "Slot definition in format: name:required "
        "(e.g., 'subject:true', 'object:false')"
    ),
)
@click.pass_context
def create_template(
    ctx: click.Context,
    output_file: Path,
    name: str,
    template_string: str,
    language_code: str | None,
    description: str | None,
    slots: tuple[str, ...],
) -> None:
    r"""Create a template with slots and constraints.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    output_file : Path
        Path to output template file.
    name : str
        Name for the template.
    template_string : str
        Template string with {slot_name} placeholders.
    language_code : str | None
        ISO 639 language code.
    description : str | None
        Description of the template.
    slots : tuple[str, ...]
        Slot definitions in format "name:required".

    Examples
    --------
    # Create simple template
    $ bead resources create-template template.jsonl \\
        --name transitive \\
        --template-string "{subject} {verb} {object}"

    # With slot specifications
    $ bead resources create-template template.jsonl \\
        --name transitive \\
        --template-string "{subject} {verb} {object}" \\
        --slot subject:true \\
        --slot verb:true \\
        --slot object:false
    """
    try:
        # Parse slot definitions
        slot_dict: dict[str, Slot] = {}

        # Extract slot names from template string
        slot_names = re.findall(r"\{(\w+)\}", template_string)

        if not slot_names:
            print_error(
                "Template string must contain at least one {slot_name} placeholder"
            )
            ctx.exit(1)

        # Parse explicit slot definitions
        explicit_slots: dict[str, bool] = {}
        for slot_def in slots:
            if ":" not in slot_def:
                print_error(
                    f"Invalid slot definition: {slot_def}. Use format 'name:required'"
                )
                ctx.exit(1)

            slot_name, required_str = slot_def.split(":", 1)
            required = required_str.lower() in ("true", "yes", "1")
            explicit_slots[slot_name] = required

        # Create slot objects for all slot names in template
        for slot_name in slot_names:
            required = explicit_slots.get(slot_name, True)
            slot_dict[slot_name] = Slot(name=slot_name, required=required)

        # Create template
        template = Template(
            name=name,
            template_string=template_string,
            slots=slot_dict,
            language_code=language_code,
            description=description,
        )

        # Create collection and add template
        collection = TemplateCollection(
            name=f"{name}_collection",
            language_code=language_code,
        )
        collection = collection.with_template(template)
        # Save collection
        output_file.parent.mkdir(parents=True, exist_ok=True)
        collection.to_jsonl(str(output_file))

        print_success(
            f"Created template '{name}' with {len(slot_dict)} slots: {output_file}"
        )

    except ValidationError as e:
        print_error(f"Validation error: {e}")
        ctx.exit(1)
    except Exception as e:
        print_error(f"Failed to create template: {e}")
        ctx.exit(1)


@resources.command()
@click.option(
    "--directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path.cwd(),
    help="Directory to search for lexicon files",
)
@click.option(
    "--pattern",
    default="*.jsonl",
    help="File pattern to match (default: *.jsonl)",
)
@click.pass_context
def list_lexicons(
    ctx: click.Context,
    directory: Path,
    pattern: str,
) -> None:
    """List available lexicons in a directory.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    directory : Path
        Directory to search for lexicon files.
    pattern : str
        File pattern to match.

    Examples
    --------
    $ bead resources list-lexicons
    $ bead resources list-lexicons --directory lexicons/
    $ bead resources list-lexicons --pattern "verb*.jsonl"
    """
    try:
        lexicon_files = list(directory.glob(pattern))

        if not lexicon_files:
            print_info(f"No lexicon files found in {directory} matching {pattern}")
            return

        table = Table(title=f"Lexicons in {directory}")
        table.add_column("File", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Items", justify="right", style="yellow")
        table.add_column("Language", style="magenta")

        for file_path in sorted(lexicon_files):
            try:
                # Try to load first item to get lexicon metadata
                with open(file_path, encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if not first_line:
                        continue

                # Count total lines
                with open(file_path, encoding="utf-8") as f:
                    item_count = sum(1 for line in f if line.strip())

                # Parse first item to get metadata
                item_data = json.loads(first_line)
                lexicon_name = file_path.stem
                language = item_data.get("language_code", "N/A")

                table.add_row(
                    str(file_path.name),
                    lexicon_name,
                    str(item_count),
                    language,
                )
            except Exception:
                # Skip files that can't be parsed
                continue

        console.print(table)

    except Exception as e:
        print_error(f"Failed to list lexicons: {e}")
        ctx.exit(1)


@resources.command()
@click.option(
    "--directory",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    default=Path.cwd(),
    help="Directory to search for template files",
)
@click.option(
    "--pattern",
    default="*.jsonl",
    help="File pattern to match (default: *.jsonl)",
)
@click.pass_context
def list_templates(
    ctx: click.Context,
    directory: Path,
    pattern: str,
) -> None:
    """List available templates in a directory.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    directory : Path
        Directory to search for template files.
    pattern : str
        File pattern to match.

    Examples
    --------
    $ bead resources list-templates
    $ bead resources list-templates --directory templates/
    $ bead resources list-templates --pattern "trans*.jsonl"
    """
    try:
        template_files = list(directory.glob(pattern))

        if not template_files:
            print_info(f"No template files found in {directory} matching {pattern}")
            return

        table = Table(title=f"Templates in {directory}")
        table.add_column("File", style="cyan")
        table.add_column("Name", style="green")
        table.add_column("Slots", justify="right", style="yellow")
        table.add_column("Template String", style="white")

        for file_path in sorted(template_files):
            try:
                # Load first template
                with open(file_path, encoding="utf-8") as f:
                    first_line = f.readline().strip()
                    if not first_line:
                        continue

                # Parse template
                template_data = json.loads(first_line)
                template_name = template_data.get("name", file_path.stem)
                slot_count = len(template_data.get("slots", {}))
                template_str = template_data.get("template_string", "N/A")

                # Truncate long template strings
                if len(template_str) > 50:
                    template_str = template_str[:47] + "..."

                table.add_row(
                    str(file_path.name),
                    template_name,
                    str(slot_count),
                    template_str,
                )
            except Exception:
                # Skip files that can't be parsed
                continue

        console.print(table)

    except Exception as e:
        print_error(f"Failed to list templates: {e}")
        ctx.exit(1)


@resources.command()
@click.argument("lexicon_file", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def validate_lexicon(ctx: click.Context, lexicon_file: Path) -> None:
    """Validate a lexicon file.

    Checks that the lexicon file is properly formatted and all items are valid.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    lexicon_file : Path
        Path to lexicon file to validate.

    Examples
    --------
    $ bead resources validate-lexicon lexicon.jsonl
    """
    try:
        print_info(f"Validating lexicon: {lexicon_file}")

        item_count = 0
        errors: list[str] = []

        with open(lexicon_file, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    LexicalItem.model_validate_json(line)
                    item_count += 1
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: Invalid JSON - {e}")
                except ValidationError as e:
                    errors.append(f"Line {line_num}: Validation error - {e}")

        if errors:
            print_error(f"Validation failed with {len(errors)} errors:")
            for error in errors[:10]:  # Show first 10 errors
                console.print(f"  [red]✗[/red] {error}")
            if len(errors) > 10:
                console.print(f"  ... and {len(errors) - 10} more errors")
            ctx.exit(1)
        else:
            print_success(f"Lexicon is valid: {item_count} items")

    except Exception as e:
        print_error(f"Failed to validate lexicon: {e}")
        ctx.exit(1)


# Add resource loader commands to resources group
resources.add_command(import_verbnet, name="import-verbnet")
resources.add_command(import_unimorph, name="import-unimorph")
resources.add_command(import_propbank, name="import-propbank")
resources.add_command(import_framenet, name="import-framenet")


@resources.command()
@click.argument("template_file", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def validate_template(ctx: click.Context, template_file: Path) -> None:
    """Validate a template file.

    Checks that the template file is properly formatted and all templates are valid.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    template_file : Path
        Path to template file to validate.

    Examples
    --------
    $ bead resources validate-template templates.jsonl
    """
    try:
        print_info(f"Validating template: {template_file}")

        template_count = 0
        errors: list[str] = []

        with open(template_file, encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue

                try:
                    Template.model_validate_json(line)
                    template_count += 1
                except json.JSONDecodeError as e:
                    errors.append(f"Line {line_num}: Invalid JSON - {e}")
                except ValidationError as e:
                    errors.append(f"Line {line_num}: Validation error - {e}")

        if errors:
            print_error(f"Validation failed with {len(errors)} errors:")
            for error in errors[:10]:  # Show first 10 errors
                console.print(f"  [red]✗[/red] {error}")
            if len(errors) > 10:
                console.print(f"  ... and {len(errors) - 10} more errors")
            ctx.exit(1)
        else:
            print_success(f"Template file is valid: {template_count} templates")

    except Exception as e:
        print_error(f"Failed to validate template: {e}")
        ctx.exit(1)


@resources.command()
@click.argument("output_file", type=click.Path(path_type=Path))
@click.option(
    "--pattern",
    required=True,
    help="Template pattern with {slot_name} placeholders (e.g., '{subj} {verb}')",
)
@click.option(
    "--name",
    required=True,
    help="Template name",
)
@click.option(
    "--slot",
    "slots",
    multiple=True,
    help="Slot specification: name:required (e.g., subject:true, object:false)",
)
@click.option(
    "--description",
    help="Description of the template",
)
@click.option(
    "--language-code",
    help="ISO 639 language code (e.g., 'eng', 'en')",
)
@click.option(
    "--tags",
    help="Comma-separated tags for categorization",
)
@click.pass_context
def generate_templates(
    ctx: click.Context,
    output_file: Path,
    pattern: str,
    name: str,
    slots: tuple[str, ...],
    description: str | None,
    language_code: str | None,
    tags: str | None,
) -> None:
    r"""Generate templates from pattern specifications.

    Creates template objects from a pattern string with slot placeholders.
    Slots are automatically extracted from the pattern or explicitly specified.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    output_file : Path
        Path to output template file (JSONL).
    pattern : str
        Template pattern with {slot_name} placeholders.
    name : str
        Template name.
    slots : tuple[str, ...]
        Slot specifications (name:required).
    description : str | None
        Template description.
    language_code : str | None
        ISO 639 language code.
    tags : str | None
        Comma-separated tags.

    Examples
    --------
    # Generate simple template (auto-detect slots)
    $ bead resources generate-templates template.jsonl \\
        --pattern "{subject} {verb} {object}" \\
        --name simple_transitive

    # With explicit slot specifications
    $ bead resources generate-templates template.jsonl \\
        --pattern "{subject} {verb} {object}" \\
        --name transitive \\
        --slot subject:true \\
        --slot verb:true \\
        --slot object:false \\
        --description "Transitive sentence template"

    # With language and tags
    $ bead resources generate-templates template.jsonl \\
        --pattern "{subject} {verb} {object}" \\
        --name transitive \\
        --language-code eng \\
        --tags "transitive,simple"
    """
    try:
        # Extract slot names from pattern
        slot_names_in_pattern = set(re.findall(r"\{(\w+)\}", pattern))

        if not slot_names_in_pattern:
            print_error(
                "No slot placeholders found in pattern.\n\n"
                "Pattern must contain {slot_name} placeholders.\n\n"
                "Example: '{subject} {verb} {object}'"
            )
            ctx.exit(1)

        # Build slot dictionary
        slot_dict: dict[str, Slot] = {}

        if slots:
            # Use explicit slot specifications
            for slot_spec in slots:
                if ":" not in slot_spec:
                    print_error(
                        f"Invalid slot specification: {slot_spec}\n\n"
                        f"Format: name:required (e.g., subject:true, object:false)"
                    )
                    ctx.exit(1)

                slot_name, required_str = slot_spec.split(":", 1)
                required = required_str.lower() in ("true", "yes", "1", "t", "y")

                if slot_name not in slot_names_in_pattern:
                    print_error(
                        f"Slot '{slot_name}' not found in pattern.\n\n"
                        f"Available slots: {', '.join(sorted(slot_names_in_pattern))}"
                    )
                    ctx.exit(1)

                slot_dict[slot_name] = Slot(name=slot_name, required=required)
        else:
            # Auto-generate slots (all required)
            for slot_name in slot_names_in_pattern:
                slot_dict[slot_name] = Slot(name=slot_name, required=True)

        # Build template
        template_data: dict[str, Any] = {
            "name": name,
            "template_string": pattern,
            "slots": slot_dict,
        }

        if description:
            template_data["description"] = description
        if language_code:
            template_data["language_code"] = language_code
        if tags:
            template_data["tags"] = [t.strip() for t in tags.split(",") if t.strip()]

        template = Template.model_validate(template_data)

        # Save to JSONL
        output_file.parent.mkdir(parents=True, exist_ok=True)
        mode = "a" if output_file.exists() else "w"
        with open(output_file, mode, encoding="utf-8") as f:
            f.write(template.model_dump_json() + "\n")

        print_success(
            f"Created template '{name}' with {len(slot_dict)} slots: {output_file}"
        )

        # Show slot details
        console.print("\n[cyan]Slots:[/cyan]")
        for slot_name, slot in sorted(slot_dict.items()):
            required_str = (
                "[green]required[/green]"
                if slot.required
                else "[yellow]optional[/yellow]"
            )
            console.print(f"  • {slot_name}: {required_str}")

    except ValidationError as e:
        print_error(f"Validation error: {e}")
        ctx.exit(1)
    except Exception as e:
        print_error(f"Failed to generate template: {e}")
        ctx.exit(1)


@resources.command()
@click.argument("base_template_file", type=click.Path(exists=True, path_type=Path))
@click.argument("output_file", type=click.Path(path_type=Path))
@click.option(
    "--slot-variants",
    help="JSON file with slot variant specs: {slot_name: [variant1, variant2]}",
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--name-pattern",
    default="{base_name}_variant_{index}",
    help="Pattern for variant names (default: {base_name}_variant_{index})",
)
@click.option(
    "--max-variants",
    type=int,
    help="Maximum number of variants to generate",
)
@click.pass_context
def generate_template_variants(
    ctx: click.Context,
    base_template_file: Path,
    output_file: Path,
    slot_variants: Path | None,
    name_pattern: str,
    max_variants: int | None,
) -> None:
    r"""Generate systematic variations of a base template.

    Creates template variants by substituting slot configurations or
    reordering slots while preserving the base structure.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    base_template_file : Path
        Path to base template file (JSONL).
    output_file : Path
        Path to output variants file (JSONL).
    slot_variants : Path | None
        JSON file with slot variant specifications.
    name_pattern : str
        Pattern for variant names.
    max_variants : int | None
        Maximum number of variants to generate.

    Examples
    --------
    # Generate variants with slot permutations
    $ bead resources generate-template-variants base.jsonl variants.jsonl \\
        --slot-variants slot_variants.json \\
        --max-variants 10

    Where slot_variants.json contains:
    {
      "subject": ["{subject}", "{object}"],
      "object": ["{object}", "{subject}"]
    }

    This creates templates with swapped subject/object positions.
    """
    try:
        print_info(f"Loading base template from {base_template_file}")

        # Load base template
        with open(base_template_file, encoding="utf-8") as f:
            first_line = f.readline().strip()
            if not first_line:
                print_error("Base template file is empty")
                ctx.exit(1)

            base_template = Template.model_validate_json(first_line)

        variants: list[Template] = []
        base_name = base_template.name
        base_template_string = base_template.template_string

        if slot_variants:
            # Load slot variant specifications
            print_info(f"Loading slot variants from {slot_variants}")
            with open(slot_variants, encoding="utf-8") as f:
                variant_spec = json.load(f)

            # Generate all combinations of slot substitutions
            slot_names = list(variant_spec.keys())
            slot_options = [variant_spec[slot] for slot in slot_names]

            # Generate all combinations
            combinations = list(product(*slot_options))

            # Limit to max_variants if specified
            if max_variants and len(combinations) > max_variants:
                print_info(
                    f"Limiting to {max_variants} variants "
                    f"(out of {len(combinations)} possible)"
                )
                combinations = combinations[:max_variants]

            for idx, combo in enumerate(combinations):
                # Create substitution map
                substitution_map = dict(zip(slot_names, combo, strict=False))

                # Apply substitutions to template_string
                variant_template_string = base_template_string
                for slot_name, replacement in substitution_map.items():
                    variant_template_string = variant_template_string.replace(
                        f"{{{slot_name}}}", replacement
                    )

                # Skip if template_string didn't change (original)
                if idx == 0 and variant_template_string == base_template_string:
                    continue

                # Create variant template
                variant_name = name_pattern.format(base_name=base_name, index=idx)
                variant_data = base_template.model_dump()
                variant_data["name"] = variant_name
                variant_data["template_string"] = variant_template_string
                variant_data["metadata"] = {
                    **variant_data.get("metadata", {}),
                    "variant_index": idx,
                    "base_template": base_name,
                    "substitutions": substitution_map,
                }

                variant = Template.model_validate(variant_data)
                variants.append(variant)

            print_success(f"Generated {len(variants)} slot-based template variants")

        else:
            # Generate simple metadata-only variants
            print_info("No slot variants specified, generating metadata variants")
            num_variants = max_variants or 3

            for i in range(num_variants):
                variant_name = name_pattern.format(base_name=base_name, index=i)

                variant_data = base_template.model_dump()
                variant_data["name"] = variant_name
                variant_data["metadata"] = {
                    **variant_data.get("metadata", {}),
                    "variant_index": i,
                    "base_template": base_name,
                }

                variant = Template.model_validate(variant_data)
                variants.append(variant)

            print_success(f"Generated {len(variants)} metadata-only template variants")

        # Save variants
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for variant in variants:
                f.write(variant.model_dump_json() + "\n")

        print_success(f"Saved variants to {output_file}")

    except ValidationError as e:
        print_error(f"Validation error: {e}")
        ctx.exit(1)
    except Exception as e:
        print_error(f"Failed to generate template variants: {e}")
        ctx.exit(1)


# Register external resource loader commands
resources.add_command(import_verbnet)
resources.add_command(import_unimorph)
resources.add_command(import_propbank)
resources.add_command(import_framenet)

# Register constraint builder command
resources.add_command(create_constraint)
