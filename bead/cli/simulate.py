"""Simulation commands for bead CLI.

This module provides commands for running multi-annotator simulations with
configurable annotator strategies and noise models.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path

import click
import numpy as np
import yaml
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from bead.cli.utils import JsonValue, print_error, print_info, print_success
from bead.config.simulation import (
    NoiseModelConfig,
    SimulatedAnnotatorConfig,
    SimulationRunnerConfig,
)
from bead.data.serialization import read_jsonlines
from bead.evaluation.interannotator import InterAnnotatorMetrics
from bead.items.item import Item
from bead.items.item_template import ItemTemplate
from bead.simulation.runner import SimulationRunner

console = Console()


@click.group()
def simulate() -> None:
    r"""Run multi-annotator simulation experiments.

    Commands for running simulations with various annotator types
    (oracle, random, LM-based, distance-based) and noise models.

    \b
    AVAILABLE COMMANDS:
        run                 Run simulation with configured annotators
        configure           Create simulation configuration file
        analyze             Analyze simulation results
        list-annotators     List available annotator types
        list-noise-models   List available noise models

    \b
    Examples:
        # Run simulation with LM-based annotator
        $ bead simulate run \\
            --items items.jsonl \\
            --templates templates.jsonl \\
            --annotator lm_score \\
            --n-annotators 5 \\
            --output results.jsonl

        # Create configuration file
        $ bead simulate configure \\
            --strategy lm_score \\
            --noise-type temperature \\
            --temperature 1.5 \\
            --output simulation_config.yaml
    """


@click.command()
@click.option(
    "--items",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to items file (JSONL)",
)
@click.option(
    "--templates",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to templates file (JSONL, optional if shared template)",
)
@click.option(
    "--annotator",
    type=click.Choice(["lm_score", "distance", "random", "oracle"]),
    default="lm_score",
    help="Annotator strategy (default: lm_score)",
)
@click.option(
    "--n-annotators",
    type=int,
    default=5,
    help="Number of simulated annotators (default: 5)",
)
@click.option(
    "--noise-type",
    type=click.Choice(["temperature", "systematic", "random", "none"]),
    default="temperature",
    help="Type of noise model (default: temperature)",
)
@click.option(
    "--temperature",
    type=float,
    default=1.0,
    help="Temperature for scaling (default: 1.0)",
)
@click.option(
    "--bias-strength",
    type=float,
    default=0.0,
    help="Systematic bias strength 0.0-1.0 (default: 0.0)",
)
@click.option(
    "--bias-type",
    type=str,
    help="Type of systematic bias (length, frequency, position)",
)
@click.option(
    "--random-seed",
    type=int,
    help="Random seed for reproducibility",
)
@click.option(
    "--model-output-key",
    type=str,
    default="lm_score",
    help="Key to extract from model outputs (default: lm_score)",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Output path for simulation results (JSONL)",
)
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    help="Path to simulation configuration file (JSON/YAML, overrides CLI options)",
)
@click.pass_context
def run(
    ctx: click.Context,
    items: Path,
    templates: Path | None,
    annotator: str,
    n_annotators: int,
    noise_type: str,
    temperature: float,
    bias_strength: float,
    bias_type: str | None,
    random_seed: int | None,
    model_output_key: str,
    output: Path,
    config: Path | None,
) -> None:
    r"""Run multi-annotator simulation.

    Simulates annotations from multiple annotators using specified strategy
    and noise model. Results are saved in JSONL format with one annotation
    per rater per item.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    items : Path
        Path to items file.
    templates : Path | None
        Path to templates file (optional).
    annotator : str
        Annotator strategy name.
    n_annotators : int
        Number of annotators to simulate.
    noise_type : str
        Type of noise model.
    temperature : float
        Temperature for scaling.
    bias_strength : float
        Systematic bias strength.
    bias_type : str | None
        Type of systematic bias.
    random_seed : int | None
        Random seed.
    model_output_key : str
        Key for model outputs.
    output : Path
        Output path for results.
    config : Path | None
        Configuration file path.

    Examples
    --------
    $ bead simulate run \\
        --items items.jsonl \\
        --templates templates.jsonl \\
        --annotator lm_score \\
        --n-annotators 10 \\
        --noise-type temperature \\
        --temperature 1.5 \\
        --output simulation_results.jsonl

    $ bead simulate run \\
        --items items.jsonl \\
        --annotator oracle \\
        --n-annotators 5 \\
        --noise-type none \\
        --output oracle_baseline.jsonl

    $ bead simulate run \\
        --items items.jsonl \\
        --config simulation_config.yaml \\
        --output results.jsonl
    """
    try:
        console.rule("[bold]Simulation Runner[/bold]")

        # Load configuration if provided
        if config:
            print_info(f"Loading configuration from {config}")
            with open(config, encoding="utf-8") as f:
                if config.suffix in [".yaml", ".yml"]:
                    config_dict = yaml.safe_load(f)
                else:
                    config_dict = json.load(f)

            sim_config = SimulationRunnerConfig(**config_dict)
            print_success("Configuration loaded")
        else:
            # Build configuration from CLI options
            noise_model = NoiseModelConfig(
                noise_type=noise_type,  # type: ignore[arg-type]
                temperature=temperature,
                bias_strength=bias_strength,
                bias_type=bias_type,
            )

            annotator_config = SimulatedAnnotatorConfig(
                strategy=annotator,  # type: ignore[arg-type]
                noise_model=noise_model,
                random_state=random_seed,
                model_output_key=model_output_key,
            )

            sim_config = SimulationRunnerConfig(
                annotator_configs=[annotator_config],
                n_annotators=n_annotators,
            )

        # Load items
        print_info(f"Loading items from {items}")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Loading items...", total=None)
            items_list = read_jsonlines(items, Item)

        print_success(f"Loaded {len(items_list)} items")

        # Load templates (optional)
        templates_list: list[ItemTemplate] | ItemTemplate
        if templates:
            print_info(f"Loading templates from {templates}")
            with Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                console=console,
            ) as progress:
                progress.add_task("Loading templates...", total=None)
                loaded_templates = read_jsonlines(templates, ItemTemplate)

            print_success(f"Loaded {len(loaded_templates)} templates")

            # Use single template if only one, otherwise list
            if len(loaded_templates) == 1:
                templates_list = loaded_templates[0]
            else:
                templates_list = loaded_templates
        else:
            # Create minimal template for items without explicit templates
            print_info("No templates provided, using minimal template")
            templates_list = ItemTemplate(
                name="default_template",
                judgment_type="acceptability",
                task_type="forced_choice",
                task_spec={"prompt": "Default simulation prompt"},
                presentation_spec={"mode": "static"},
            )

        # Create simulation runner
        print_info(f"Creating simulation with {sim_config.n_annotators} annotators")
        runner = SimulationRunner(config=sim_config)

        # Run simulation
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Running simulation...", total=None)
            results = runner.run(items=items_list, templates=templates_list)

        # Display summary
        table = Table(title="Simulation Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Items", str(len(items_list)))
        table.add_row("Annotators", str(sim_config.n_annotators))
        table.add_row("Strategy", annotator if not config else "from config")
        table.add_row("Noise Type", noise_type if not config else "from config")
        table.add_row(
            "Total Annotations", str(len(items_list) * sim_config.n_annotators)
        )

        console.print(table)

        # Save results
        print_info(f"Saving results to {output}")

        # Convert results to JSONL format (one record per item per annotator)
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            for i, item_id in enumerate(results["item_ids"]):
                for annotator_idx in range(sim_config.n_annotators):
                    annotation = results[f"annotator_{annotator_idx}"][i]
                    record = {
                        "item_id": item_id,
                        "annotator_id": f"annotator_{annotator_idx}",
                        "annotation": annotation,
                    }
                    f.write(json.dumps(record) + "\n")

        print_success(f"Simulation complete! Results saved to {output}")

    except FileNotFoundError as e:
        print_error(f"File not found: {e}", exit_code=0)
        ctx.exit(1)
    except KeyError as e:
        print_error(f"Missing required field: {e}", exit_code=0)
        ctx.exit(1)
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON in configuration: {e}", exit_code=0)
        ctx.exit(1)
    except ValueError as e:
        print_error(f"Validation error: {e}", exit_code=0)
        ctx.exit(1)


@click.command()
@click.option(
    "--strategy",
    type=click.Choice(["lm_score", "distance", "random", "oracle"]),
    default="lm_score",
    help="Annotator strategy (default: lm_score)",
)
@click.option(
    "--n-annotators",
    type=int,
    default=5,
    help="Number of annotators (default: 5)",
)
@click.option(
    "--noise-type",
    type=click.Choice(["temperature", "systematic", "random", "none"]),
    default="temperature",
    help="Noise model type (default: temperature)",
)
@click.option(
    "--temperature",
    type=float,
    default=1.0,
    help="Temperature for scaling (default: 1.0)",
)
@click.option(
    "--bias-strength",
    type=float,
    default=0.0,
    help="Systematic bias strength (default: 0.0)",
)
@click.option(
    "--bias-type",
    type=str,
    help="Type of systematic bias (length, frequency, position)",
)
@click.option(
    "--random-seed",
    type=int,
    help="Random seed for reproducibility",
)
@click.option(
    "--model-output-key",
    type=str,
    default="lm_score",
    help="Key for model outputs (default: lm_score)",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    required=True,
    help="Output path for configuration file (YAML/JSON)",
)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["yaml", "json"]),
    default="yaml",
    help="Output format (default: yaml)",
)
@click.pass_context
def configure(
    ctx: click.Context,
    strategy: str,
    n_annotators: int,
    noise_type: str,
    temperature: float,
    bias_strength: float,
    bias_type: str | None,
    random_seed: int | None,
    model_output_key: str,
    output: Path,
    output_format: str,
) -> None:
    r"""Create simulation configuration file.

    Generates a configuration file that can be used with the 'run' command
    via the --config option. Configuration includes annotator strategy,
    noise model parameters, and simulation settings.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    strategy : str
        Annotator strategy.
    n_annotators : int
        Number of annotators.
    noise_type : str
        Type of noise model.
    temperature : float
        Temperature parameter.
    bias_strength : float
        Bias strength.
    bias_type : str | None
        Type of bias.
    random_seed : int | None
        Random seed.
    model_output_key : str
        Key for model outputs.
    output : Path
        Output path for config.
    format : str
        Output format (yaml or json).

    Examples
    --------
    $ bead simulate configure \\
        --strategy lm_score \\
        --n-annotators 10 \\
        --noise-type temperature \\
        --temperature 2.0 \\
        --random-seed 42 \\
        --output simulation_config.yaml

    $ bead simulate configure \\
        --strategy systematic \\
        --bias-strength 0.3 \\
        --bias-type length \\
        --output config.json \\
        --format json
    """
    try:
        console.rule("[bold]Simulation Configuration[/bold]")

        # Build configuration
        noise_model = NoiseModelConfig(
            noise_type=noise_type,  # type: ignore[arg-type]
            temperature=temperature,
            bias_strength=bias_strength,
            bias_type=bias_type,
        )

        annotator_config = SimulatedAnnotatorConfig(
            strategy=strategy,  # type: ignore[arg-type]
            noise_model=noise_model,
            random_state=random_seed,
            model_output_key=model_output_key,
        )

        sim_config = SimulationRunnerConfig(
            annotator_configs=[annotator_config],
            n_annotators=n_annotators,
        )

        # Display configuration
        table = Table(title="Configuration Summary")
        table.add_column("Setting", style="cyan")
        table.add_column("Value", style="green")

        table.add_row("Strategy", strategy)
        table.add_row("Number of Annotators", str(n_annotators))
        table.add_row("Noise Type", noise_type)
        table.add_row("Temperature", f"{temperature:.2f}")
        table.add_row("Bias Strength", f"{bias_strength:.2f}")
        if bias_type:
            table.add_row("Bias Type", bias_type)
        if random_seed is not None:
            table.add_row("Random Seed", str(random_seed))

        console.print(table)

        # Save configuration
        output.parent.mkdir(parents=True, exist_ok=True)

        config_dict = json.loads(sim_config.model_dump_json())

        if output_format == "yaml":
            with open(output, "w", encoding="utf-8") as f:
                yaml.safe_dump(config_dict, f, default_flow_style=False, indent=2)
        else:
            with open(output, "w", encoding="utf-8") as f:
                json.dump(config_dict, f, indent=2)

        print_success(f"Configuration saved to {output}")

    except Exception as e:
        print_error(f"Failed to create configuration: {e}", exit_code=0)
        ctx.exit(1)


@click.command()
@click.option(
    "--results",
    type=click.Path(dir_okay=False, path_type=Path),
    required=True,
    help="Path to simulation results (JSONL)",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    help="Output path for analysis report (JSON)",
)
@click.pass_context
def analyze(
    ctx: click.Context,
    results: Path,
    output: Path | None,
) -> None:
    r"""Analyze simulation results.

    Computes statistics and agreement metrics from simulation results,
    including per-annotator statistics and inter-annotator agreement.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    results : Path
        Path to simulation results.
    output : Path | None
        Optional output path for report.

    Examples
    --------
    $ bead simulate analyze \\
        --results simulation_results.jsonl \\
        --output analysis_report.json

    $ bead simulate analyze \\
        --results results.jsonl
    """
    try:
        console.rule("[bold]Simulation Analysis[/bold]")

        # Load results
        print_info(f"Loading simulation results from {results}")
        with open(results, encoding="utf-8") as f:
            records = [json.loads(line) for line in f if line.strip()]

        print_success(f"Loaded {len(records)} annotation records")

        # Organize by item and annotator
        # Annotations can be int, float, str, or list[str] depending on task type
        items_dict: dict[str, dict[str, int | float | str | list[str]]] = {}
        annotators_dict: dict[str, list[int | float | str | list[str]]] = {}

        for record in records:
            item_id = record["item_id"]
            annotator_id = record["annotator_id"]
            annotation = record["annotation"]

            if item_id not in items_dict:
                items_dict[item_id] = {}
            items_dict[item_id][annotator_id] = annotation

            if annotator_id not in annotators_dict:
                annotators_dict[annotator_id] = []
            annotators_dict[annotator_id].append(annotation)

        n_items = len(items_dict)
        n_annotators = len(annotators_dict)

        print_info(f"Found {n_items} items and {n_annotators} annotators")

        # Compute statistics
        # Per-annotator statistics
        annotator_stats: dict[str, dict[str, JsonValue]] = {}
        for annotator_id, annotations in annotators_dict.items():
            # Basic statistics (depends on annotation type)
            if annotations and isinstance(annotations[0], int | float):
                # Type narrowing: we know these are numeric now
                numeric_annotations = [
                    a for a in annotations if isinstance(a, int | float)
                ]
                annotator_stats[annotator_id] = {
                    "count": len(annotations),
                    "mean": float(np.mean(numeric_annotations)),
                    "std": float(np.std(numeric_annotations)),
                    "min": float(np.min(numeric_annotations)),
                    "max": float(np.max(numeric_annotations)),
                }
            else:
                counter = Counter(annotations)
                # Convert Counter.most_common() result to JSON-serializable format
                most_common_list = [
                    {"value": str(val), "count": count}
                    for val, count in counter.most_common(3)
                ]
                annotator_stats[annotator_id] = {
                    "count": len(annotations),
                    "unique_values": len(counter),
                    "most_common": most_common_list,
                }

        # Display summary
        table = Table(title="Analysis Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Total Items", str(n_items))
        table.add_row("Total Annotators", str(n_annotators))
        table.add_row("Total Annotations", str(len(records)))
        table.add_row("Annotations per Item", f"{len(records) / n_items:.1f}")

        console.print(table)

        # Compute inter-annotator agreement if applicable
        if n_annotators >= 2:
            print_info("Computing inter-annotator agreement...")

            # Organize for agreement computation
            rater_data: dict[str, list[int | float | str | list[str] | None]] = {}
            for item_id in sorted(items_dict.keys()):
                for annotator_id in sorted(annotators_dict.keys()):
                    if annotator_id not in rater_data:
                        rater_data[annotator_id] = []
                    rater_data[annotator_id].append(
                        items_dict[item_id].get(annotator_id)
                    )

            try:
                # Type: ignore needed because InterAnnotatorMetrics expects
                # specific Label type but annotations vary by task type
                alpha = InterAnnotatorMetrics.krippendorff_alpha(
                    rater_data,
                    metric="nominal",  # type: ignore[arg-type]
                )

                agreement_table = Table(title="Inter-Annotator Agreement")
                agreement_table.add_column("Metric", style="cyan")
                agreement_table.add_column("Value", style="green", justify="right")

                agreement_table.add_row("Krippendorff's Alpha", f"{alpha:.4f}")

                console.print(agreement_table)

                annotator_stats["inter_annotator_agreement"] = {
                    "krippendorff_alpha": float(alpha)
                }
            except Exception as e:
                print_info(f"Could not compute agreement: {e}")

        # Save analysis report
        if output:
            analysis_report: dict[str, JsonValue] = {
                "n_items": n_items,
                "n_annotators": n_annotators,
                "total_annotations": len(records),
                "annotator_statistics": annotator_stats,
            }

            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w", encoding="utf-8") as f:
                json.dump(analysis_report, f, indent=2)

            print_success(f"Analysis report saved to {output}")

        print_success("Analysis complete!")

    except FileNotFoundError as e:
        print_error(f"File not found: {e}", exit_code=0)
        ctx.exit(1)
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON: {e}", exit_code=0)
        ctx.exit(1)
    except Exception as e:
        print_error(f"Analysis failed: {e}", exit_code=0)
        ctx.exit(1)


@click.command()
def list_annotators() -> None:
    """List available annotator types with descriptions.

    Displays all available annotator strategies, their descriptions,
    and typical use cases.

    Examples
    --------
    $ bead simulate list-annotators
    """
    console.rule("[bold]Available Annotator Types[/bold]")

    table = Table(show_header=True)
    table.add_column("Strategy", style="cyan", width=15)
    table.add_column("Description", style="green", width=50)
    table.add_column("Use Case", style="yellow", width=30)

    annotators = [
        (
            "lm_score",
            "Uses language model scores from item.model_outputs",
            "Simulate LM-based judgments",
        ),
        (
            "distance",
            "Uses distance metrics (embeddings, edit distance)",
            "Similarity-based judgments",
        ),
        (
            "random",
            "Random selection from valid options",
            "Baseline / control condition",
        ),
        (
            "oracle",
            "Uses ground truth labels (requires labels file)",
            "Gold standard simulation",
        ),
    ]

    for strategy, description, use_case in annotators:
        table.add_row(strategy, description, use_case)

    console.print(table)

    # Print usage examples
    console.print("\n[bold]Example Usage:[/bold]")
    console.print(
        "  $ bead simulate run --items items.jsonl --annotator lm_score "
        "--n-annotators 5"
    )
    console.print(
        "  $ bead simulate run --items items.jsonl --annotator oracle "
        "--ground-truth labels.jsonl"
    )


@click.command()
def list_noise_models() -> None:
    """List available noise models with descriptions.

    Displays all available noise model types, their parameters,
    and effects on simulation results.

    Examples
    --------
    $ bead simulate list-noise-models
    """
    console.rule("[bold]Available Noise Models[/bold]")

    table = Table(show_header=True)
    table.add_column("Noise Type", style="cyan", width=15)
    table.add_column("Description", style="green", width=40)
    table.add_column("Key Parameters", style="yellow", width=25)

    noise_models = [
        (
            "temperature",
            "Scales decision probabilities (higher = more random)",
            "temperature (0.1-10.0)",
        ),
        (
            "systematic",
            "Applies systematic biases (length, frequency, position)",
            "bias_strength, bias_type",
        ),
        (
            "random",
            "Adds Gaussian noise to scores",
            "random_noise_stddev",
        ),
        (
            "none",
            "No noise (deterministic)",
            "N/A",
        ),
    ]

    for noise_type, description, parameters in noise_models:
        table.add_row(noise_type, description, parameters)

    console.print(table)

    # Print parameter details
    console.print("\n[bold]Parameter Details:[/bold]")
    console.print(
        "  • temperature: Controls randomness (1.0 = unchanged, >1.0 = more random)"
    )
    console.print("  • bias_strength: Strength of systematic bias (0.0-1.0)")
    console.print("  • bias_type: Type of bias (length/frequency/position)")
    console.print("  • random_noise_stddev: Standard deviation for random noise")

    # Print usage examples
    console.print("\n[bold]Example Usage:[/bold]")
    console.print(
        "  $ bead simulate run --items items.jsonl --noise-type temperature "
        "--temperature 2.0"
    )
    console.print(
        "  $ bead simulate run --items items.jsonl --noise-type systematic "
        "--bias-strength 0.3 --bias-type length"
    )


# Register commands
simulate.add_command(run)
simulate.add_command(configure)
simulate.add_command(analyze)
simulate.add_command(list_annotators)
simulate.add_command(list_noise_models)
