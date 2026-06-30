"""Additional active learning CLI commands.

This module contains the select-items and run commands that were too large
to include in the main active_learning.py file.
"""

from __future__ import annotations

import json
import traceback
from pathlib import Path
from typing import Any, Literal

import click
import didactic.api as dx
import numpy as np
import yaml
from didactic.api import ValidationError
from rich.console import Console
from rich.progress import (
    BarColumn,
    Progress,
    SpinnerColumn,
    TaskProgressColumn,
    TextColumn,
)
from rich.table import Table

from bead.active_learning.loop import ActiveLearningLoop
from bead.active_learning.models.base import ActiveLearningModel
from bead.active_learning.models.binary import BinaryModel
from bead.active_learning.models.categorical import CategoricalModel
from bead.active_learning.models.forced_choice import ForcedChoiceModel
from bead.active_learning.selection import ItemSelector, UncertaintySampler
from bead.cli.utils import print_error, print_info, print_success
from bead.config.active_learning import (
    ActiveLearningLoopConfig,
    BinaryModelConfig,
    CategoricalModelConfig,
    ForcedChoiceModelConfig,
    UncertaintySamplerConfig,
)
from bead.items.item import Item
from bead.items.item_template import ItemTemplate

console = Console()


# Configuration models for the run command
StoppingCriterion = Literal["max_iterations", "convergence", "performance_threshold"]


class RunLoopConfig(dx.Model):
    """Loop configuration for the active learning run command."""

    max_iterations: int = 10
    budget_per_iteration: int = 100
    stopping_criterion: StoppingCriterion = "max_iterations"
    performance_threshold: float | None = None
    metric_name: str = "accuracy"
    convergence_patience: int = 3
    convergence_threshold: float = 0.01


class RunModelConfig(dx.Model):
    """Model configuration for the active learning run command."""

    type: Literal["binary", "categorical", "forced_choice"] = "binary"
    model_name: str = "bert-base-uncased"
    max_length: int = 128
    learning_rate: float = 2e-5
    batch_size: int = 16
    num_epochs: int = 3
    device: Literal["cpu", "cuda", "mps"] = "cpu"


class RunSelectionConfig(dx.Model):
    """Selection configuration for the active learning run command."""

    method: Literal["entropy", "margin", "least_confidence"] = "entropy"
    batch_size: int | None = None


class RunDataConfig(dx.Model):
    """Data paths configuration for the active learning run command."""

    initial_items: str
    unlabeled_pool: str
    item_template: str
    human_ratings: str | None = None


def _default_run_loop_config() -> RunLoopConfig:
    return RunLoopConfig()


def _default_run_model_config() -> RunModelConfig:
    return RunModelConfig()


def _default_run_selection_config() -> RunSelectionConfig:
    return RunSelectionConfig()


class ActiveLearningRunConfig(dx.Model):
    """Full configuration for the active learning run command."""

    data: dx.Embed[RunDataConfig]
    loop: dx.Embed[RunLoopConfig] = dx.field(default_factory=_default_run_loop_config)
    model: dx.Embed[RunModelConfig] = dx.field(
        default_factory=_default_run_model_config
    )
    selection: dx.Embed[RunSelectionConfig] = dx.field(
        default_factory=_default_run_selection_config
    )


def load_run_config(config_path: Path) -> ActiveLearningRunConfig:
    """Load active learning run configuration from YAML file.

    Parameters
    ----------
    config_path : Path
        Path to YAML configuration file.

    Returns
    -------
    ActiveLearningRunConfig
        Validated configuration.

    Raises
    ------
    FileNotFoundError
        If configuration file doesn't exist.
    ValidationError
        If configuration is invalid.
    """
    with open(config_path, encoding="utf-8") as f:
        config_dict = yaml.safe_load(f)
    return ActiveLearningRunConfig(**config_dict)


def create_model_from_config(model_config: RunModelConfig) -> ActiveLearningModel:
    """Create model instance from configuration.

    Parameters
    ----------
    model_config : RunModelConfig
        Model configuration.

    Returns
    -------
    ActiveLearningModel
        Configured model instance.

    Raises
    ------
    ValueError
        If model type is unknown.
    """
    model_type = model_config.type

    if model_type == "binary":
        config = BinaryModelConfig(
            model_name=model_config.model_name,
            max_length=model_config.max_length,
            learning_rate=model_config.learning_rate,
            batch_size=model_config.batch_size,
            num_epochs=model_config.num_epochs,
            device=model_config.device,
        )
        return BinaryModel(config=config)
    elif model_type == "categorical":
        config = CategoricalModelConfig(
            model_name=model_config.model_name,
            max_length=model_config.max_length,
            learning_rate=model_config.learning_rate,
            batch_size=model_config.batch_size,
            num_epochs=model_config.num_epochs,
            device=model_config.device,
        )
        return CategoricalModel(config=config)
    elif model_type == "forced_choice":
        config = ForcedChoiceModelConfig(
            model_name=model_config.model_name,
            max_length=model_config.max_length,
            learning_rate=model_config.learning_rate,
            batch_size=model_config.batch_size,
            num_epochs=model_config.num_epochs,
            device=model_config.device,
        )
        return ForcedChoiceModel(config=config)
    else:
        raise ValueError(f"Unknown model type: {model_type}")


def _load_items(path: Path) -> list[Item]:
    """Load items from JSONL file.

    Parameters
    ----------
    path : Path
        Path to JSONL file.

    Returns
    -------
    list[Item]
        List of loaded items.
    """
    items: list[Item] = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            items.append(Item.model_validate_json(line))
    return items


def _load_item_template(path: Path) -> ItemTemplate:
    """Load item template from JSONL file.

    Parameters
    ----------
    path : Path
        Path to JSONL file containing template.

    Returns
    -------
    ItemTemplate
        Loaded item template.

    Raises
    ------
    ValueError
        If no template found in file.
    """
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            template_data = json.loads(line)
            return ItemTemplate(**template_data)
    raise ValueError(f"No item template found in {path}")


def _load_ratings(path: Path) -> dict[str, Any]:
    """Load human ratings from JSONL file.

    Parameters
    ----------
    path : Path
        Path to JSONL file with ratings.

    Returns
    -------
    dict[str, Any]
        Mapping from item_id to label.
    """
    ratings: dict[str, Any] = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            item_id = str(record["item_id"])
            label = record["label"]
            ratings[item_id] = label
    return ratings


def _save_iteration_results(
    output_dir: Path,
    iteration: int,
    selected_items: list[Item],
    metrics: dict[str, float] | None,
) -> None:
    """Save results from a single iteration.

    Parameters
    ----------
    output_dir : Path
        Output directory.
    iteration : int
        Iteration number.
    selected_items : list[Item]
        Items selected in this iteration.
    metrics : dict[str, float] | None
        Training metrics (if available).
    """
    iter_dir = output_dir / f"iteration_{iteration}"
    iter_dir.mkdir(parents=True, exist_ok=True)

    # Save selected items
    items_path = iter_dir / "selected_items.jsonl"
    with open(items_path, "w", encoding="utf-8") as f:
        for item in selected_items:
            f.write(item.model_dump_json() + "\n")

    # Save metrics if available
    if metrics:
        metrics_path = iter_dir / "metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)


def _display_run_summary(
    iterations_completed: int,
    total_items_selected: int,
    final_metrics: dict[str, float] | None,
) -> None:
    """Display summary table of active learning run.

    Parameters
    ----------
    iterations_completed : int
        Number of iterations completed.
    total_items_selected : int
        Total number of items selected.
    final_metrics : dict[str, float] | None
        Final model metrics.
    """
    table = Table(title="Active Learning Run Summary")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    table.add_row("Iterations Completed", str(iterations_completed))
    table.add_row("Total Items Selected", str(total_items_selected))

    if final_metrics:
        for metric_name, value in final_metrics.items():
            if isinstance(value, float):
                table.add_row(f"Final {metric_name}", f"{value:.4f}")

    console.print(table)


def _show_dry_run_plan(
    config: ActiveLearningRunConfig,
    output_dir: Path,
) -> None:
    """Show what would be executed in dry-run mode.

    Parameters
    ----------
    config : ActiveLearningRunConfig
        Run configuration.
    output_dir : Path
        Output directory.
    """
    console.print("\n[yellow]DRY RUN MODE - No commands will be executed[/yellow]\n")

    console.print("[bold]Configuration Summary:[/bold]")
    console.print(f"  Model type: {config.model.type}")
    console.print(f"  Model name: {config.model.model_name}")
    console.print(f"  Max iterations: {config.loop.max_iterations}")
    console.print(f"  Budget per iteration: {config.loop.budget_per_iteration}")
    console.print(f"  Stopping criterion: {config.loop.stopping_criterion}")
    console.print(f"  Selection method: {config.selection.method}")

    console.print("\n[bold]Data Paths:[/bold]")
    console.print(f"  Initial items: {config.data.initial_items}")
    console.print(f"  Unlabeled pool: {config.data.unlabeled_pool}")
    console.print(f"  Item template: {config.data.item_template}")
    ratings_path = config.data.human_ratings or "None (required for simulation)"
    console.print(f"  Human ratings: {ratings_path}")

    console.print(f"\n[bold]Output directory:[/bold] {output_dir}")


@click.command()
@click.option(
    "--items",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to unlabeled items file (JSONL)",
)
@click.option(
    "--model",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to trained model directory",
)
@click.option(
    "--output",
    "-o",
    type=click.Path(path_type=Path),
    required=True,
    help="Output file for selected items (JSONL)",
)
@click.option(
    "--budget",
    type=int,
    required=True,
    help="Number of items to select",
)
@click.option(
    "--method",
    type=click.Choice(["entropy", "margin", "least_confidence"]),
    default="entropy",
    help="Uncertainty sampling method (default: entropy)",
)
@click.pass_context
def select_items(
    ctx: click.Context,
    items: Path,
    model: Path,
    output: Path,
    budget: int,
    method: str,
) -> None:
    r"""Select items for annotation using active learning.

    Uses uncertainty sampling to select the most informative items from
    an unlabeled pool for human annotation.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    items : Path
        Path to unlabeled items file (JSONL).
    model : Path
        Path to trained model directory.
    output : Path
        Output file for selected items (JSONL).
    budget : int
        Number of items to select.
    method : str
        Uncertainty sampling method.

    Examples
    --------
    $ bead active-learning select-items \\
        --items unlabeled_items.jsonl \\
        --model models/binary_model \\
        --output selected_items.jsonl \\
        --budget 50 \\
        --method entropy
    """
    try:
        console.rule("[bold]Item Selection[/bold]")

        # Load items
        print_info(f"Loading items from {items}")
        unlabeled_items: list[Item] = []
        with open(items, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                item = Item.model_validate_json(line)
                unlabeled_items.append(item)

        if len(unlabeled_items) == 0:
            print_error("No items found in file")
            ctx.exit(1)

        print_success(f"Loaded {len(unlabeled_items)} unlabeled items")

        if budget > len(unlabeled_items):
            n_available = len(unlabeled_items)
            print_error(f"Budget ({budget}) exceeds available items ({n_available})")
            ctx.exit(1)

        # Load model
        print_info(f"Loading model from {model}")
        # Try to determine model type from config
        config_path = model / "config.json"
        if not config_path.exists():
            print_error(f"Model config not found at {config_path}")
            ctx.exit(1)

        with open(config_path, encoding="utf-8") as f:
            config_dict = json.load(f)

        # Determine model type and load
        model_type = config_dict.get("model_type") or config_dict.get("task_type")
        cfg = str(config_dict)
        is_binary = model_type == "binary" or "BinaryModelConfig" in cfg
        is_categorical = model_type == "categorical" or "CategoricalModelConfig" in cfg
        is_forced = model_type == "forced_choice" or "ForcedChoiceModelConfig" in cfg

        if is_binary:
            loaded_model = BinaryModel()
            loaded_model.load(str(model))
        elif is_categorical:
            loaded_model = CategoricalModel()
            loaded_model.load(str(model))
        elif is_forced:
            loaded_model = ForcedChoiceModel()
            loaded_model.load(str(model))
        else:
            # Default to binary
            loaded_model = BinaryModel()
            loaded_model.load(str(model))

        print_success("Model loaded successfully")

        # Create item selector
        sampler_config = UncertaintySamplerConfig(method=method)
        selector: ItemSelector = UncertaintySampler(config=sampler_config)

        # Define predict function
        def predict_fn(model_instance: object, item: Item) -> np.ndarray:
            """Get prediction probabilities for an item."""
            predictions = model_instance.predict_proba([item], participant_ids=None)
            return predictions[0]

        # Select items
        print_info(f"Selecting {budget} items using {method} method...")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Selecting items...", total=None)
            selected_items = selector.select(
                items=unlabeled_items,
                model=loaded_model,
                predict_fn=predict_fn,
                budget=budget,
            )

        print_success(f"Selected {len(selected_items)} items")

        # Save selected items
        print_info(f"Writing selected items to {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        with open(output, "w", encoding="utf-8") as f:
            for item in selected_items:
                f.write(item.model_dump_json() + "\n")

        print_success(f"Selected items written to {output}")

        # Display summary
        table = Table(title="Selection Summary")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Total unlabeled items", str(len(unlabeled_items)))
        table.add_row("Budget", str(budget))
        table.add_row("Selected items", str(len(selected_items)))
        table.add_row("Method", method)

        console.print(table)

    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        ctx.exit(1)
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON: {e}")
        ctx.exit(1)
    except Exception as e:
        print_error(f"Item selection failed: {e}")
        traceback.print_exc()
        ctx.exit(1)


@click.command()
@click.option(
    "--config",
    type=click.Path(exists=True, path_type=Path),
    required=True,
    help="Path to active learning configuration file (YAML)",
)
@click.option(
    "--output-dir",
    type=click.Path(path_type=Path),
    required=True,
    help="Output directory for active learning results",
)
@click.option(
    "--mode",
    type=click.Choice(["simulation"]),
    default="simulation",
    help="Execution mode: simulation (with ratings file)",
)
@click.option(
    "--dry-run",
    is_flag=True,
    default=False,
    help="Show what would be done without executing",
)
@click.pass_context
def run(
    ctx: click.Context,
    config: Path,
    output_dir: Path,
    mode: str,
    dry_run: bool,
) -> None:
    r"""Run full active learning loop.

    Orchestrates the complete active learning workflow:
    1. Select informative items using uncertainty sampling
    2. Simulate data collection using provided human ratings
    3. Train model on labeled data
    4. Check convergence
    5. Repeat until convergence or max iterations

    Note: Currently only simulation mode is supported, which requires
    a human_ratings file in the configuration. Automated data collection
    via JATOS/Prolific is not yet implemented.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    config : Path
        Path to active learning configuration file (YAML).
    output_dir : Path
        Output directory for results.
    mode : str
        Execution mode (currently only "simulation").
    dry_run : bool
        If True, show plan without executing.

    Examples
    --------
    $ bead active-learning run \\
        --config configs/active_learning.yaml \\
        --output-dir results/

    $ bead active-learning run \\
        --config configs/active_learning.yaml \\
        --output-dir results/ \\
        --dry-run
    """
    try:
        console.rule("[bold]Active Learning Loop[/bold]")

        # Step 1: Load and validate configuration
        print_info(f"Loading configuration from {config}")
        try:
            run_config = load_run_config(config)
        except ValidationError as e:
            print_error(f"Configuration validation error: {e}")
            ctx.exit(1)
            return

        # Step 2: Validate mode requirements
        if mode == "simulation" and run_config.data.human_ratings is None:
            print_error("Simulation mode requires human_ratings path in config")
            print_info("Add 'human_ratings: path/to/ratings.jsonl' to data section")
            ctx.exit(1)
            return

        # Step 3: Create output directory
        output_dir.mkdir(parents=True, exist_ok=True)

        # Step 4: Handle dry run
        if dry_run:
            _show_dry_run_plan(run_config, output_dir)
            return

        # Step 5: Load data
        print_info(f"Loading initial items from {run_config.data.initial_items}")
        initial_items = _load_items(run_config.data.initial_items)
        print_success(f"Loaded {len(initial_items)} initial items")

        print_info(f"Loading unlabeled pool from {run_config.data.unlabeled_pool}")
        unlabeled_pool = _load_items(run_config.data.unlabeled_pool)
        print_success(f"Loaded {len(unlabeled_pool)} unlabeled items")

        print_info(f"Loading item template from {run_config.data.item_template}")
        item_template = _load_item_template(run_config.data.item_template)
        print_success("Loaded item template")

        human_ratings: dict[str, Any] | None = None
        if run_config.data.human_ratings:
            print_info(f"Loading human ratings from {run_config.data.human_ratings}")
            human_ratings = _load_ratings(run_config.data.human_ratings)
            print_success(f"Loaded {len(human_ratings)} human ratings")

        # Step 6: Create model
        print_info(f"Creating {run_config.model.type} model...")
        model = create_model_from_config(run_config.model)
        print_success("Model created")

        # Step 7: Create item selector
        sampler_config = UncertaintySamplerConfig(
            method=run_config.selection.method,
            batch_size=run_config.selection.batch_size,
        )
        item_selector: ItemSelector = UncertaintySampler(config=sampler_config)

        # Step 8: Create loop config
        loop_config = ActiveLearningLoopConfig(
            max_iterations=run_config.loop.max_iterations,
            budget_per_iteration=run_config.loop.budget_per_iteration,
            stopping_criterion=run_config.loop.stopping_criterion,
            performance_threshold=run_config.loop.performance_threshold,
            metric_name=run_config.loop.metric_name,
            convergence_patience=run_config.loop.convergence_patience,
            convergence_threshold=run_config.loop.convergence_threshold,
        )

        # Step 9: Create and run loop
        print_info("Initializing active learning loop...")
        loop = ActiveLearningLoop(
            item_selector=item_selector,
            config=loop_config,
        )

        # Step 10: Run with progress reporting
        print_info("Starting active learning loop...")
        console.print()

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            BarColumn(),
            TaskProgressColumn(),
            console=console,
        ) as progress:
            task = progress.add_task(
                "Running active learning...",
                total=run_config.loop.max_iterations,
            )

            try:
                loop.run(
                    initial_items=initial_items,
                    initial_model=model,
                    item_template=item_template,
                    unlabeled_pool=unlabeled_pool,
                    human_ratings=human_ratings,
                )

                # Update progress based on actual iterations completed
                iterations_completed = len(loop.iteration_history)
                progress.update(task, completed=iterations_completed)

            except Exception as e:
                print_error(f"Active learning loop failed: {e}")
                traceback.print_exc()
                ctx.exit(1)
                return

        # Step 11: Save results
        print_info("Saving results...")
        total_items_selected = 0
        final_metrics: dict[str, float] | None = None

        for i, iteration_result in enumerate(loop.iteration_history):
            selected_items = iteration_result.get("selected_items", [])
            total_items_selected += len(selected_items)
            metrics = iteration_result.get("metrics")

            _save_iteration_results(
                output_dir=output_dir,
                iteration=i,
                selected_items=selected_items,
                metrics=metrics,
            )

            if metrics:
                final_metrics = metrics

        # Save run summary
        summary = {
            "iterations_completed": len(loop.iteration_history),
            "total_items_selected": total_items_selected,
            "config": run_config.model_dump(),
        }
        summary_path = output_dir / "run_summary.json"
        with open(summary_path, "w", encoding="utf-8") as f:
            json.dump(summary, f, indent=2, default=str)

        print_success(f"Results saved to {output_dir}")

        # Step 12: Display summary
        console.print()
        _display_run_summary(
            iterations_completed=len(loop.iteration_history),
            total_items_selected=total_items_selected,
            final_metrics=final_metrics,
        )

        print_success("Active learning completed!")

    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        ctx.exit(1)
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON: {e}")
        ctx.exit(1)
    except Exception as e:
        print_error(f"Active learning run failed: {e}")
        traceback.print_exc()
        ctx.exit(1)
