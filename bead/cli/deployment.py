"""Deployment commands for bead CLI.

This module provides commands for generating and deploying jsPsych experiments
(Stage 5 of the bead pipeline).
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

from bead.cli.utils import print_error, print_info, print_success
from bead.data.base import JsonValue
from bead.deployment.distribution import (
    DistributionStrategyType,
    ListDistributionStrategy,
)
from bead.deployment.jatos.api import JATOSClient
from bead.deployment.jatos.exporter import JATOSExporter
from bead.deployment.jspsych.config import ExperimentConfig
from bead.deployment.jspsych.generator import JsPsychExperimentGenerator
from bead.items.item import Item
from bead.items.item_template import (
    ItemTemplate,
    PresentationSpec,
    ScaleBounds,
    TaskSpec,
)
from bead.lists import ExperimentList

console = Console()


@click.group()
def deployment() -> None:
    r"""Deployment commands (Stage 5).

    Commands for generating and deploying jsPsych experiments.

    \b
    Examples:
        $ bead deployment generate lists.jsonl items.jsonl experiment/
        $ bead deployment export-jatos experiment/ study.jzip \\
            --title "My Study"
        $ bead deployment upload-jatos study.jzip \\
            --jatos-url https://jatos.example.com --api-token TOKEN
        $ bead deployment validate experiment/
    """


@click.command()
@click.argument(
    "lists_file", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument(
    "items_file", type=click.Path(exists=True, dir_okay=False, path_type=Path)
)
@click.argument("output_dir", type=click.Path(path_type=Path))
@click.option(
    "--experiment-type",
    type=click.Choice(["likert_rating", "forced_choice", "magnitude_estimation"]),
    default="likert_rating",
    help="Type of experiment",
)
@click.option("--title", default="Experiment", help="Experiment title")
@click.option("--description", default="", help="Experiment description")
@click.option(
    "--instructions", default="Please complete the task.", help="Instructions text"
)
@click.option(
    "--distribution-strategy",
    type=click.Choice(
        [
            "random",
            "sequential",
            "balanced",
            "latin_square",
            "stratified",
            "weighted_random",
            "quota_based",
            "metadata_based",
        ],
        case_sensitive=False,
    ),
    required=True,
    help="List distribution strategy (REQUIRED, no default). "
    "random: Random selection. "
    "sequential: Round-robin. "
    "balanced: Assign to least-used list. "
    "latin_square: Counterbalancing. "
    "stratified: Balance across factors. "
    "weighted_random: Non-uniform probabilities. "
    "quota_based: Fixed quota per list. "
    "metadata_based: Filter/rank by metadata.",
)
@click.option(
    "--distribution-config",
    type=str,
    help="Strategy-specific configuration (JSON format). "
    "Examples: "
    'quota_based: \'{"participants_per_list": 25, "allow_overflow": false}\'. '
    'weighted_random: \'{"weight_expression": "list_metadata.priority || 1.0"}\'. '
    'stratified: \'{"factors": ["condition", "verb_type"]}\'. '
    "metadata_based: "
    "'{\"filter_expression\": \"list_metadata.difficulty === 'easy'\"}'. ",
)
@click.option(
    "--max-participants",
    type=int,
    help="Maximum total participants across all lists (unlimited if not specified)",
)
@click.option(
    "--debug-mode",
    is_flag=True,
    help="Enable debug mode (always assign same list for testing)",
)
@click.option(
    "--debug-list-index",
    type=int,
    default=0,
    help="List index to use in debug mode (default: 0)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    help="Show what would be done without generating files",
)
@click.pass_context
def generate(
    ctx: click.Context,
    lists_file: Path,
    items_file: Path,
    output_dir: Path,
    experiment_type: str,
    title: str,
    description: str,
    instructions: str,
    distribution_strategy: str,
    distribution_config: str | None,
    max_participants: int | None,
    debug_mode: bool,
    debug_list_index: int,
    dry_run: bool,
) -> None:
    r"""Generate jsPsych experiment from lists and items.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    lists_file : Path
        JSONL file containing experiment lists (one list per line).
    items_file : Path
        JSONL file containing items (one item per line).
    output_dir : Path
        Output directory for generated experiment.
    experiment_type : str
        Type of experiment to generate.
    title : str
        Experiment title.
    description : str
        Experiment description.
    instructions : str
        Instructions text.
    distribution_strategy : str
        Distribution strategy type (required).
    distribution_config : str | None
        Strategy-specific configuration as JSON string.
    max_participants : int | None
        Maximum total participants.
    debug_mode : bool
        Enable debug mode.
    debug_list_index : int
        List index for debug mode.
    dry_run : bool
        Show what would be done without generating files.

    Examples
    --------
    # Basic balanced distribution
    $ bead deployment generate lists.jsonl items.jsonl experiment/ \\
        --experiment-type forced_choice \\
        --title "Acceptability Study" \\
        --distribution-strategy balanced

    # Quota-based with config
    $ bead deployment generate lists.jsonl items.jsonl experiment/ \\
        --experiment-type forced_choice \\
        --distribution-strategy quota_based \\
        --distribution-config '{"participants_per_list": 25, "allow_overflow": false}'

    # Stratified by factors
    $ bead deployment generate lists.jsonl items.jsonl experiment/ \\
        --experiment-type forced_choice \\
        --distribution-strategy stratified \\
        --distribution-config '{"factors": ["condition", "verb_type"]}'

    # Dry run to preview
    $ bead deployment generate lists.jsonl items.jsonl experiment/ \\
        --experiment-type forced_choice \\
        --distribution-strategy balanced \\
        --dry-run
    """
    try:
        # Parse distribution config if provided
        strategy_config_dict: dict[str, JsonValue] = {}
        if distribution_config:
            try:
                strategy_config_dict = json.loads(distribution_config)
            except json.JSONDecodeError as e:
                print_error(
                    f"Invalid JSON in --distribution-config: {e}\n"
                    f"Provided: {distribution_config}\n"
                    f"Example: '{{\"participants_per_list\": 25}}'"
                )
                ctx.exit(1)

        # Create distribution strategy
        try:
            dist_strategy = ListDistributionStrategy(
                strategy_type=DistributionStrategyType(distribution_strategy),
                strategy_config=strategy_config_dict,
                max_participants=max_participants,
                debug_mode=debug_mode,
                debug_list_index=debug_list_index,
            )
        except ValueError as e:
            print_error(f"Invalid distribution strategy configuration: {e}")
            ctx.exit(1)
        # Load experiment lists from JSONL file (one list per line)
        print_info(f"Loading experiment lists from {lists_file}")
        experiment_lists: list[ExperimentList] = []
        with open(lists_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                exp_list = ExperimentList.model_validate_json(line)
                experiment_lists.append(exp_list)

        if not experiment_lists:
            print_error(f"No lists found in {lists_file}")
            ctx.exit(1)

        print_info(f"Loaded {len(experiment_lists)} experiment lists")

        # Load items
        print_info(f"Loading items from {items_file}")
        items_dict: dict[UUID, Item] = {}
        with open(items_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = Item.model_validate_json(line)
                items_dict[item.id] = item

        print_info(f"Loaded {len(items_dict)} items")

        # Create stub templates for each unique item_template_id (simplified for CLI)
        # Extract unique template IDs from items
        unique_template_ids = {item.item_template_id for item in items_dict.values()}
        templates_dict: dict[UUID, ItemTemplate] = {}
        for template_id in unique_template_ids:
            # Create minimal stub template (no actual template
            # structure needed for deployment)
            templates_dict[template_id] = ItemTemplate(
                id=template_id,
                name=f"template_{template_id}",
                description="Auto-generated stub template for CLI deployment",
                judgment_type="acceptability",
                task_type="ordinal_scale",
                task_spec=TaskSpec(
                    prompt="Rate this item.",
                    scale_bounds=ScaleBounds(min=1, max=7),
                ),
                presentation_spec=PresentationSpec(mode="static"),
            )

        print_info(f"Created {len(templates_dict)} stub templates for deployment")

        # Create experiment config with distribution strategy
        from bead.deployment.jspsych.config import InstructionsConfig  # noqa: PLC0415

        config = ExperimentConfig(
            experiment_type=experiment_type,  # type: ignore
            title=title,
            description=description,
            instructions=(
                instructions
                if not isinstance(instructions, str)
                else InstructionsConfig.from_text(instructions)
            ),
            distribution_strategy=dist_strategy,
        )

        # Generate experiment (or show dry-run preview)
        if dry_run:
            print_info("[DRY RUN] Would generate jsPsych experiment with:")
            console.print(f"  [dim]Output directory:[/dim] {output_dir}")
            console.print(f"  [dim]Experiment type:[/dim] {experiment_type}")
            console.print(f"  [dim]Title:[/dim] {title}")
            console.print(
                f"  [dim]Distribution strategy:[/dim] {distribution_strategy}"
            )
            console.print(f"  [dim]Number of lists:[/dim] {len(experiment_lists)}")
            console.print(f"  [dim]Number of items:[/dim] {len(items_dict)}")
            console.print(f"  [dim]Number of templates:[/dim] {len(templates_dict)}")
            if max_participants:
                console.print(f"  [dim]Max participants:[/dim] {max_participants}")
            if debug_mode:
                console.print(
                    f"  [dim]Debug mode:[/dim] Enabled (list index: {debug_list_index})"
                )
            print_info("[DRY RUN] Files that would be created:")
            console.print(f"  [dim]{output_dir}/index.html[/dim]")
            console.print(f"  [dim]{output_dir}/js/experiment.js[/dim]")
            console.print(f"  [dim]{output_dir}/js/list_distributor.js[/dim]")
            console.print(f"  [dim]{output_dir}/css/experiment.css[/dim]")
            console.print(f"  [dim]{output_dir}/data/config.json[/dim]")
            console.print(f"  [dim]{output_dir}/data/lists.jsonl[/dim]")
            console.print(f"  [dim]{output_dir}/data/items.jsonl[/dim]")
            console.print(f"  [dim]{output_dir}/data/distribution.json[/dim]")
        else:
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("Generating jsPsych experiment...", total=None)

                generator = JsPsychExperimentGenerator(
                    config=config,
                    output_dir=output_dir,
                )
                output_path = generator.generate(
                    lists=experiment_lists,
                    items=items_dict,
                    templates=templates_dict,
                )

            print_success(f"Generated jsPsych experiment: {output_path}")

    except ValidationError as e:
        print_error(f"Validation error: {e}")
        ctx.exit(1)
    except Exception as e:
        import traceback  # noqa: PLC0415

        print_error(
            f"Failed to generate experiment: {type(e).__name__}: {e}\n"
            + traceback.format_exc()
        )
        ctx.exit(1)


@click.command()
@click.argument(
    "experiment_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.argument("output_file", type=click.Path(path_type=Path))
@click.option("--title", required=True, help="Study title for JATOS")
@click.option("--description", default="", help="Study description")
@click.option("--component-title", default="Main Experiment", help="Component title")
@click.pass_context
def export_jatos(
    ctx: click.Context,
    experiment_dir: Path,
    output_file: Path,
    title: str,
    description: str,
    component_title: str,
) -> None:
    r"""Export experiment to JATOS .jzip file.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    experiment_dir : Path
        Directory containing generated experiment.
    output_file : Path
        Output path for .jzip file.
    title : str
        Study title for JATOS.
    description : str
        Study description.
    component_title : str
        Component title.

    Examples
    --------
    $ bead deployment export-jatos experiment/ study.jzip \\
        --title "Acceptability Study" \\
        --description "Rating task for linguistic acceptability"
    """
    try:
        print_info(f"Exporting experiment from {experiment_dir}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Creating JATOS package...", total=None)

            exporter = JATOSExporter(
                study_title=title,
                study_description=description,
            )
            exporter.export(
                experiment_dir=experiment_dir,
                output_path=output_file,
                component_title=component_title,
            )

        print_success(f"Created JATOS package: {output_file}")

    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        ctx.exit(1)
    except ValueError as e:
        print_error(f"Invalid experiment: {e}")
        ctx.exit(1)
    except Exception as e:
        print_error(f"Failed to export to JATOS: {e}")
        ctx.exit(1)


@click.command()
@click.argument("jzip_file", type=click.Path(exists=True, path_type=Path))
@click.option("--jatos-url", required=True, help="JATOS server URL")
@click.option("--api-token", required=True, help="JATOS API token")
@click.pass_context
def upload_jatos(
    ctx: click.Context,
    jzip_file: Path,
    jatos_url: str,
    api_token: str,
) -> None:
    r"""Upload .jzip file to JATOS server.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    jzip_file : Path
        Path to .jzip file.
    jatos_url : str
        JATOS server URL.
    api_token : str
        JATOS API token.

    Examples
    --------
    $ bead deployment upload-jatos study.jzip \\
        --jatos-url https://jatos.example.com \\
        --api-token my-api-token
    """
    try:
        print_info(f"Uploading {jzip_file} to {jatos_url}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Uploading to JATOS...", total=None)

            client = JATOSClient(base_url=jatos_url, api_token=api_token)
            study_id: int = client.import_study(jzip_file)  # type: ignore[attr-defined]

        print_success(f"Uploaded study to JATOS (Study ID: {study_id})")

    except Exception as e:
        print_error(f"Failed to upload to JATOS: {e}")
        ctx.exit(1)


@click.command()
@click.argument(
    "experiment_dir", type=click.Path(exists=True, file_okay=False, path_type=Path)
)
@click.option(
    "--check-distribution",
    is_flag=True,
    default=False,
    help="Validate distribution strategy configuration",
)
@click.option(
    "--check-trials",
    is_flag=True,
    default=False,
    help="Validate trial configurations (if present)",
)
@click.option(
    "--check-data-structure",
    is_flag=True,
    default=False,
    help="Validate JSONL data structure and schemas",
)
@click.option(
    "--strict",
    is_flag=True,
    default=False,
    help="Enable all validation checks (strict mode)",
)
@click.pass_context
def validate(
    ctx: click.Context,
    experiment_dir: Path,
    check_distribution: bool,
    check_trials: bool,
    check_data_structure: bool,
    strict: bool,
) -> None:
    """Validate generated experiment structure.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    experiment_dir : Path
        Directory containing generated experiment.
    check_distribution : bool
        Validate distribution strategy configuration.
    check_trials : bool
        Validate trial configurations (if present).
    check_data_structure : bool
        Validate JSONL data structure and schemas.
    strict : bool
        Enable all validation checks.

    Examples
    --------
    $ bead deployment validate experiment/

    $ bead deployment validate experiment/ --check-distribution

    $ bead deployment validate experiment/ --strict
    """
    try:
        # Enable all checks if strict mode is enabled
        if strict:
            check_distribution = True
            check_trials = True
            check_data_structure = True

        print_info(f"Validating experiment: {experiment_dir}")
        if strict:
            print_info("Running in strict mode (all checks enabled)")

        validation_errors: list[str] = []
        validation_warnings: list[str] = []

        # Check required files (batch mode)
        required_files = [
            "index.html",
            "css/experiment.css",
            "js/experiment.js",
            "js/list_distributor.js",
            "data/config.json",
            "data/lists.jsonl",
            "data/items.jsonl",
            "data/distribution.json",
        ]

        missing_files: list[str] = []
        for file_path in required_files:
            full_path = experiment_dir / file_path
            if not full_path.exists():
                missing_files.append(file_path)

        if missing_files:
            for file_path in missing_files:
                validation_errors.append(f"Missing required file: {file_path}")

        # Validate lists.jsonl
        lists_file = experiment_dir / "data" / "lists.jsonl"
        if lists_file.exists():
            with open(lists_file, encoding="utf-8") as f:
                lists_data = [json.loads(line) for line in f if line.strip()]

            if not lists_data:
                validation_errors.append("lists.jsonl must contain at least one list")
        else:
            lists_data = []

        # Validate items.jsonl
        items_file = experiment_dir / "data" / "items.jsonl"
        if items_file.exists():
            with open(items_file, encoding="utf-8") as f:
                items_data = [json.loads(line) for line in f if line.strip()]

            if not items_data:
                validation_errors.append("items.jsonl must contain at least one item")
        else:
            items_data = []

        # Validate distribution.json
        dist_file = experiment_dir / "data" / "distribution.json"
        dist_data: dict[str, JsonValue] | None = None
        if dist_file.exists():
            with open(dist_file, encoding="utf-8") as f:
                dist_data_obj: JsonValue = json.load(f)
                if isinstance(dist_data_obj, dict):
                    dist_data = dist_data_obj  # type: ignore[assignment]

            if dist_data is None or "strategy_type" not in dist_data:
                validation_errors.append(
                    "distribution.json must be a dict with strategy_type field"
                )

        # Additional validation checks
        if check_distribution and dist_data:
            _validate_distribution_config(
                dist_data, validation_errors, validation_warnings
            )

        if check_trials:
            _validate_trial_configs(
                experiment_dir, validation_errors, validation_warnings
            )

        if check_data_structure and items_data and lists_data:
            _validate_data_structure(
                items_data, lists_data, validation_errors, validation_warnings
            )

        # Report results
        if validation_errors:
            print_error(f"Validation failed with {len(validation_errors)} error(s):")
            for error in validation_errors:
                console.print(f"  [red]✗[/red] {error}")
            if validation_warnings:
                console.print()
                console.print(
                    f"[yellow]⚠[/yellow] {len(validation_warnings)} warning(s):"
                )
                for warning in validation_warnings:
                    console.print(f"  [yellow]⚠[/yellow] {warning}")
            ctx.exit(1)

        if validation_warnings:
            console.print(f"[yellow]⚠[/yellow] {len(validation_warnings)} warning(s):")
            for warning in validation_warnings:
                console.print(f"  [yellow]⚠[/yellow] {warning}")

        print_success(
            f"Experiment structure is valid "
            f"({len(lists_data)} lists, {len(items_data)} items)"
        )

    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON in experiment data files: {e}")
        ctx.exit(1)
    except Exception as e:
        print_error(f"Failed to validate experiment: {e}")
        ctx.exit(1)


def _validate_distribution_config(
    dist_data: dict[str, JsonValue],
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate distribution strategy configuration.

    Parameters
    ----------
    dist_data : dict[str, JsonValue]
        Distribution configuration data.
    errors : list[str]
        List to append errors to.
    warnings : list[str]
        List to append warnings to.
    """
    strategy_type = dist_data.get("strategy_type")

    # Validate strategy type
    valid_strategies = [
        "random",
        "sequential",
        "balanced",
        "latin_square",
        "stratified",
        "weighted_random",
        "quota_based",
        "metadata_based",
    ]

    if strategy_type not in valid_strategies:
        errors.append(
            f"Invalid strategy_type: {strategy_type}. "
            f"Must be one of: {', '.join(valid_strategies)}"
        )
        return

    # Validate strategy-specific configuration
    strategy_config_raw = dist_data.get("strategy_config")
    strategy_config: dict[str, JsonValue] | None = (
        strategy_config_raw if isinstance(strategy_config_raw, dict) else None
    )

    if strategy_type == "quota_based":
        if not strategy_config:
            errors.append("quota_based strategy requires strategy_config")
        elif "participants_per_list" not in strategy_config:
            errors.append(
                "quota_based requires participants_per_list in strategy_config"
            )

    elif strategy_type == "weighted_random":
        if strategy_config and "weight_expression" not in strategy_config:
            warnings.append(
                "weighted_random without weight_expression uses uniform weights"
            )

    elif strategy_type == "metadata_based":
        if not strategy_config:
            errors.append("metadata_based strategy requires strategy_config")
        elif (
            "filter_expression" not in strategy_config
            and "rank_expression" not in strategy_config
        ):
            warnings.append(
                "metadata_based without filter or rank expressions has no effect"
            )

    elif strategy_type == "stratified":
        if not strategy_config:
            warnings.append(
                "stratified strategy without factors uses random assignment"
            )
        elif "factors" not in strategy_config:
            warnings.append(
                "stratified strategy without factors uses random assignment"
            )


def _validate_trial_configs(
    experiment_dir: Path,
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate trial configuration files if present.

    Parameters
    ----------
    experiment_dir : Path
        Experiment directory.
    errors : list[str]
        List to append errors to.
    warnings : list[str]
        List to append warnings to.
    """
    # Check for trial configuration files
    config_dir = experiment_dir / "config"
    if not config_dir.exists():
        return  # No config directory, skip trial validation

    trial_configs = list(config_dir.glob("*_config.json"))

    for config_file in trial_configs:
        try:
            config_data: JsonValue = json.loads(config_file.read_text(encoding="utf-8"))

            if not isinstance(config_data, dict):
                errors.append(f"Trial config {config_file.name} must be a JSON object")
                continue

            config_dict: dict[str, JsonValue] = config_data  # type: ignore[assignment]

            # Validate config type
            config_type = config_dict.get("type")
            if not config_type:
                errors.append(f"Trial config {config_file.name} missing 'type' field")
                continue

            # Validate type-specific fields
            if config_type == "rating_scale":
                required_fields = ["min_value", "max_value", "step"]
                for field in required_fields:
                    if field not in config_dict:
                        errors.append(
                            f"Rating config {config_file.name} missing '{field}' field"
                        )

            elif config_type == "choice":
                if "button_html" not in config_dict:
                    warnings.append(
                        f"Choice config {config_file.name} missing button_html "
                        f"(will use default)"
                    )

        except json.JSONDecodeError:
            errors.append(f"Trial config {config_file.name} contains invalid JSON")


def _validate_data_structure(
    items_data: list[dict[str, JsonValue]],
    lists_data: list[dict[str, JsonValue]],
    errors: list[str],
    warnings: list[str],
) -> None:
    """Validate JSONL data structure and schemas.

    Parameters
    ----------
    items_data : list[dict[str, JsonValue]]
        Items data from items.jsonl.
    lists_data : list[dict[str, JsonValue]]
        Lists data from lists.jsonl.
    errors : list[str]
        List to append errors to.
    warnings : list[str]
        List to append warnings to.
    """
    # Validate items structure
    for i, item in enumerate(items_data):
        if "id" not in item:
            errors.append(f"Item {i} missing 'id' field")

        if "item_template_id" not in item:
            warnings.append(f"Item {i} missing 'item_template_id' field")

        if "rendered_elements" not in item:
            errors.append(f"Item {i} missing 'rendered_elements' field")

    # Validate lists structure
    for i, exp_list in enumerate(lists_data):
        if "id" not in exp_list:
            errors.append(f"List {i} missing 'id' field")

        if "item_refs" not in exp_list:
            errors.append(f"List {i} missing 'item_refs' field")
        elif not isinstance(exp_list["item_refs"], list):
            errors.append(f"List {i} 'item_refs' must be a list")
        elif not exp_list["item_refs"]:
            warnings.append(f"List {i} has no items (empty item_refs)")

    # Check that all item_refs in lists exist in items
    item_ids = {item.get("id") for item in items_data if "id" in item}

    for i, exp_list in enumerate(lists_data):
        if "item_refs" in exp_list and isinstance(exp_list["item_refs"], list):
            # item_refs are UUID strings from JSON
            item_refs_list = cast(list[str], exp_list["item_refs"])
            for item_ref in item_refs_list:
                if item_ref not in item_ids:
                    errors.append(f"List {i} references non-existent item: {item_ref}")


# Import nested command groups
from bead.cli.deployment_trials import deployment_trials  # noqa: E402
from bead.cli.deployment_ui import deployment_ui  # noqa: E402

# Register commands
deployment.add_command(generate)
deployment.add_command(export_jatos)
deployment.add_command(upload_jatos)
deployment.add_command(validate)

# Register nested command groups
deployment.add_command(deployment_trials)
deployment.add_command(deployment_ui)
