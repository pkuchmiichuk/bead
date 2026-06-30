"""Configuration commands for bead CLI.

This module provides commands for viewing, validating, and managing configuration.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import click
import yaml
from didactic.api import ValidationError

from bead.cli.utils import (
    format_output,
    get_nested_value,
    load_config_for_cli,
    merge_config_dicts,
    print_error,
    print_info,
    print_success,
    redact_sensitive_values,
)
from bead.config import list_profiles, validate_config


@click.group()
def config() -> None:
    r"""Manage configuration commands.

    Provides commands for viewing, validating, and exporting configuration.

    \b
    Examples:
        $ bead config show
        $ bead config show --format json
        $ bead config show --key paths.data_dir
        $ bead config validate
        $ bead config export --output my-config.yaml
        $ bead config profiles
    """


@config.command()
@click.option(
    "--format",
    "-f",
    "format_type",
    type=click.Choice(["yaml", "json", "table"], case_sensitive=False),
    default="yaml",
    help="Output format (default: yaml)",
)
@click.option(
    "--key",
    "-k",
    type=str,
    default=None,
    help="Show specific config value (e.g., paths.data_dir)",
)
@click.option(
    "--no-redact",
    is_flag=True,
    default=False,
    help="Show sensitive values (API keys, etc.)",
)
@click.pass_context
def show(
    ctx: click.Context,
    format_type: str,
    key: str | None,
    no_redact: bool,
) -> None:
    r"""Display current configuration.

    Shows the merged configuration from profile, file, and environment variables.

    \b
    Examples:
        $ bead config show
        $ bead config show --format json
        $ bead config show --key paths.data_dir
        $ bead config show --no-redact  # Show API keys
    """
    config_file = ctx.obj.get("config_file")
    profile = ctx.obj.get("profile", "default")
    verbose = ctx.obj.get("verbose", False)

    try:
        cfg = load_config_for_cli(
            config_file=str(config_file) if config_file else None,
            profile=profile,
            verbose=verbose,
        )

        # Convert to dict
        config_dict = cfg.model_dump()

        # Redact sensitive values unless --no-redact
        if not no_redact:
            config_dict = redact_sensitive_values(config_dict)

        # Show specific key if requested
        if key:
            try:
                value = get_nested_value(config_dict, key)
                click.echo(value)
            except KeyError as e:
                print_error(f"Configuration key not found: {e}")
            return

        # Format and display
        try:
            output = format_output(config_dict, format_type)  # type: ignore[arg-type]
            click.echo(output)
        except ValueError as e:
            print_error(f"Failed to format output: {e}")

    except Exception as e:
        print_error(f"Failed to load configuration: {e}")


@config.command()
@click.option(
    "--config-file",
    "-c",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    default=None,
    help="Configuration file to validate",
)
@click.pass_context
def validate(ctx: click.Context, config_file: Path | None) -> None:
    r"""Validate configuration file.

    Checks YAML syntax and validates against bead configuration schema.

    \b
    Examples:
        $ bead config validate
        $ bead config validate --config-file my-config.yaml

    \b
    Exit codes:
        0 - Configuration is valid
        1 - Configuration is invalid
    """
    # Use CLI context config-file if not explicitly provided
    if config_file is None:
        config_file = ctx.obj.get("config_file")

    if config_file is None:
        print_error("No configuration file specified. Use --config-file or -c.")
        return

    profile = ctx.obj.get("profile", "default")
    verbose = ctx.obj.get("verbose", False)

    try:
        # Load and validate
        cfg = load_config_for_cli(
            config_file=str(config_file),
            profile=profile,
            verbose=verbose,
        )

        # Additional validation
        errors = validate_config(cfg)

        if errors:
            print_error("Configuration validation failed:")
            for error in errors:
                click.echo(f"  • {error}", err=True)
            click.get_current_context().exit(1)
        else:
            print_success(f"Configuration is valid: {config_file}")

    except ValidationError as e:
        print_error("Configuration validation failed:")
        for error in e.errors():
            location = " → ".join(str(loc) for loc in error["loc"])
            click.echo(f"  • {location}: {error['msg']}", err=True)
        click.get_current_context().exit(1)

    except Exception as e:
        print_error(f"Failed to validate configuration: {e}")
        click.get_current_context().exit(1)


@config.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    default=None,
    help="Output file (default: stdout)",
)
@click.option(
    "--comments",
    is_flag=True,
    default=False,
    help="Include comments explaining each field",
)
@click.option(
    "--no-redact",
    is_flag=True,
    default=False,
    help="Include sensitive values (API keys, etc.)",
)
@click.pass_context
def export(
    ctx: click.Context,
    output: Path | None,
    comments: bool,
    no_redact: bool,
) -> None:
    r"""Export current configuration to YAML.

    Exports the merged configuration (profile + file + env) to a YAML file.

    \b
    Examples:
        $ bead config export
        $ bead config export --output my-config.yaml
        $ bead config export --comments  # Include field explanations
        $ bead config export --no-redact --output full-config.yaml
    """
    config_file = ctx.obj.get("config_file")
    profile = ctx.obj.get("profile", "default")
    verbose = ctx.obj.get("verbose", False)

    try:
        cfg = load_config_for_cli(
            config_file=str(config_file) if config_file else None,
            profile=profile,
            verbose=verbose,
        )

        # Convert to dict
        config_dict = cfg.model_dump()

        # Redact sensitive values unless --no-redact
        if not no_redact:
            config_dict = redact_sensitive_values(config_dict)

        # Add comments if requested
        yaml_content = _generate_yaml_with_comments(config_dict) if comments else None

        # Save or print
        if output:
            if yaml_content:
                output.write_text(yaml_content)
            else:
                # Write config dict directly to YAML file
                with open(output, "w") as f:
                    yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)
            print_success(f"Configuration exported to: {output}")
        else:
            if yaml_content:
                click.echo(yaml_content)
            else:
                click.echo(
                    yaml.dump(config_dict, default_flow_style=False, sort_keys=False)
                )

    except Exception as e:
        print_error(f"Failed to export configuration: {e}")


@config.command()
def profiles() -> None:
    r"""List available configuration profiles.

    Shows all built-in profiles with descriptions.

    \b
    Examples:
        $ bead config profiles
    """
    available_profiles = list_profiles()

    print_info("Available configuration profiles:")
    click.echo()

    for profile_name in available_profiles:
        click.echo(f"  • {profile_name}")

    click.echo()
    print_info("Use --profile to select a profile:")
    click.echo("  $ bead --profile dev config show")


def _generate_yaml_with_comments(config_dict: dict[str, Any]) -> str:
    """Generate YAML with comments explaining fields.

    Parameters
    ----------
    config_dict : dict[str, Any]
        Configuration dictionary.

    Returns
    -------
    str
        YAML content with comments.
    """
    lines = ["# bead Configuration", "# Generated with comments", ""]

    # Add commented sections
    sections = {
        "profile": "Configuration profile (default, dev, prod, test)",
        "logging": "Logging configuration (level, format, file)",
        "paths": "Path configuration (directories for data, models, cache)",
        "resources": "Resource management (auto-download, caching, language)",
        "templates": "Template filling (strategy, constraints, MLM settings)",
        "models": "Model configuration (default models, GPU, API keys)",
        "items": "Item construction (validation, auto-save)",
        "lists": "List construction (partitioning, balancing)",
        "deployment": "Deployment configuration (platform, jsPsych, plugins)",
        "training": "Training configuration (framework, hyperparameters)",
    }

    for section, description in sections.items():
        if section in config_dict:
            lines.append(f"# {description}")
            section_yaml = yaml.dump(
                {section: config_dict[section]},
                default_flow_style=False,
                sort_keys=False,
            )
            lines.append(section_yaml.rstrip())
            lines.append("")

    return "\n".join(lines)


@config.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Output configuration file path",
)
@click.option(
    "--selection-strategy",
    type=click.Choice(["uncertainty", "diversity", "hybrid"], case_sensitive=False),
    default="uncertainty",
    help="Selection strategy for active learning",
)
@click.option(
    "--budget",
    type=int,
    default=1000,
    help="Annotation budget (default: 1000)",
)
@click.option(
    "--convergence-threshold",
    type=float,
    default=0.85,
    help="Convergence threshold (default: 0.85)",
)
@click.option(
    "--checkpoint-interval",
    type=int,
    default=100,
    help="Checkpoint interval (default: 100)",
)
def create_active_learning(
    output: Path,
    selection_strategy: str,
    budget: int,
    convergence_threshold: float,
    checkpoint_interval: int,
) -> None:
    r"""Create active learning configuration file.

    Examples
    --------
    $ bead config create-active-learning --output al_config.yaml
    $ bead config create-active-learning --output al_config.yaml \
        --selection-strategy hybrid --budget 2000
    """
    config_dict = {
        "active_learning": {
            "selection_strategy": selection_strategy,
            "budget": budget,
            "convergence_threshold": convergence_threshold,
            "checkpoint_interval": checkpoint_interval,
            "min_iterations": 10,
        }
    }

    with open(output, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    print_success(f"Created active learning configuration: {output}")


@config.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Output configuration file path",
)
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
    help="Task type for model",
)
@click.option(
    "--base-model",
    type=str,
    default="bert-base-uncased",
    help="Base model name (default: bert-base-uncased)",
)
@click.option(
    "--mixed-effects-mode",
    type=click.Choice(
        ["fixed-only", "random-intercepts", "random-slopes"],
        case_sensitive=False,
    ),
    default="fixed-only",
    help="Mixed effects mode (default: fixed-only)",
)
@click.option(
    "--use-lora",
    is_flag=True,
    help="Use LoRA parameter-efficient fine-tuning",
)
def create_model(
    output: Path,
    task_type: str,
    base_model: str,
    mixed_effects_mode: str,
    use_lora: bool,
) -> None:
    r"""Create model configuration file.

    Examples
    --------
    $ bead config create-model --output model_config.yaml \
        --task-type forced_choice
    $ bead config create-model --output model_config.yaml \
        --task-type ordinal_scale --mixed-effects-mode random-intercepts
    """
    config_dict: dict[str, Any] = {
        "model": {
            "task_type": task_type,
            "base_model": base_model,
            "mixed_effects_mode": mixed_effects_mode,
        }
    }

    if use_lora:
        config_dict["model"]["lora"] = {
            "enabled": True,
            "rank": 8,
            "alpha": 16,
            "dropout": 0.1,
        }

    if mixed_effects_mode in ("random-intercepts", "random-slopes"):
        config_dict["model"]["mixed_effects_config"] = {
            "participant_intercept": True,
            "item_intercept": True,
            "interaction": mixed_effects_mode == "random-slopes",
            "variance_components": {
                "participant_variance": 1.0,
                "item_variance": 1.0,
                "interaction_variance": (
                    0.5 if mixed_effects_mode == "random-slopes" else 0.0
                ),
            },
        }

    with open(output, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    print_success(f"Created model configuration: {output}")


@config.command()
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Output configuration file path",
)
@click.option(
    "--annotator-type",
    type=click.Choice(
        ["oracle", "random", "lm-based", "distance-based"],
        case_sensitive=False,
    ),
    default="oracle",
    help="Annotator type (default: oracle)",
)
@click.option(
    "--model-name",
    type=str,
    help="Model name for lm-based annotator",
)
@click.option(
    "--noise-model",
    type=click.Choice(["random", "systematic", "temperature"], case_sensitive=False),
    help="Noise model type",
)
@click.option(
    "--noise-level",
    type=float,
    default=0.05,
    help="Noise level (default: 0.05)",
)
@click.option(
    "--n-annotators",
    type=int,
    default=20,
    help="Number of simulated annotators (default: 20)",
)
def create_simulation(
    output: Path,
    annotator_type: str,
    model_name: str | None,
    noise_model: str | None,
    noise_level: float,
    n_annotators: int,
) -> None:
    r"""Create simulation configuration file.

    Examples
    --------
    $ bead config create-simulation --output sim_config.yaml
    $ bead config create-simulation --output sim_config.yaml \
        --annotator-type lm-based --model-name gpt-4
    $ bead config create-simulation --output sim_config.yaml \
        --noise-model random --noise-level 0.1
    """
    config_dict: dict[str, Any] = {
        "simulation": {
            "annotator_type": annotator_type,
            "n_annotators": n_annotators,
        }
    }

    if annotator_type == "lm-based" and model_name:
        config_dict["simulation"]["model_name"] = model_name

    if noise_model:
        config_dict["simulation"]["noise_model"] = {
            "type": noise_model,
            "level": noise_level,
        }

    with open(output, "w") as f:
        yaml.dump(config_dict, f, default_flow_style=False, sort_keys=False)

    print_success(f"Created simulation configuration: {output}")


@config.command()
@click.option(
    "--base",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Base configuration file",
)
@click.option(
    "--override",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Override configuration file",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Output merged configuration file",
)
def merge(
    base: Path,
    override: Path,
    output: Path,
) -> None:
    """Merge two configuration files.

    Examples
    --------
    $ bead config merge --base base.yaml --override custom.yaml --output merged.yaml
    """
    try:
        # Load both files
        with open(base) as f:
            base_config = yaml.safe_load(f)

        with open(override) as f:
            override_config = yaml.safe_load(f)

        # Merge recursively
        merged = merge_config_dicts(base_config, override_config)

        # Write merged config
        with open(output, "w") as f:
            yaml.dump(merged, f, default_flow_style=False, sort_keys=False)

        print_success(f"Merged configurations: {output}")

    except Exception as e:
        print_error(f"Failed to merge configurations: {e}")
