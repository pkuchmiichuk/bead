"""Training commands for bead CLI.

This module provides commands for collecting data, training judgment prediction
models, and evaluating model performance (Stage 6 of the bead pipeline).
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import cast

import click
import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn, track
from rich.table import Table
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.model_selection import KFold

from bead.active_learning.models import (
    config_class_for_task_type,
    model_class_for_task_type,
)
from bead.cli.utils import print_error, print_info, print_success
from bead.data.base import JsonValue
from bead.data.serialization import read_jsonlines
from bead.data_collection.jatos import JATOSDataCollector
from bead.evaluation.interannotator import InterAnnotatorMetrics
from bead.items.item import Item
from bead.items.item_template import TaskType

console = Console()


@click.group()
def training() -> None:
    r"""Training commands (Stage 6).

    Commands for collecting data and training judgment prediction models.

    \b
    Examples:
        $ bead training collect-data results.jsonl \\
            --jatos-url https://jatos.example.com \\
            --api-token TOKEN --study-id 123
        $ bead training show-data-stats results.jsonl
    """


@click.command()
@click.argument("output_file", type=click.Path(path_type=Path))
@click.option("--jatos-url", required=True, help="JATOS server URL")
@click.option("--api-token", required=True, help="JATOS API token")
@click.option("--study-id", required=True, type=int, help="JATOS study ID")
@click.option("--component-id", type=int, help="Filter by component ID (optional)")
@click.option("--worker-type", help="Filter by worker type (optional)")
@click.pass_context
def collect_data(
    ctx: click.Context,
    output_file: Path,
    jatos_url: str,
    api_token: str,
    study_id: int,
    component_id: int | None,
    worker_type: str | None,
) -> None:
    r"""Collect judgment data from JATOS.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    output_file : Path
        Output path for collected data.
    jatos_url : str
        JATOS server URL.
    api_token : str
        JATOS API token.
    study_id : int
        JATOS study ID.
    component_id : int | None
        Component ID to filter by.
    worker_type : str | None
        Worker type to filter by.

    Examples
    --------
    $ bead training collect-data results.jsonl \\
        --jatos-url https://jatos.example.com \\
        --api-token my-token \\
        --study-id 123

    $ bead training collect-data results.jsonl \\
        --jatos-url https://jatos.example.com \\
        --api-token my-token \\
        --study-id 123 \\
        --component-id 456 \\
        --worker-type Prolific
    """
    try:
        print_info(f"Collecting data from JATOS study {study_id}")

        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Downloading results from JATOS...", total=None)

            collector = JATOSDataCollector(
                base_url=jatos_url,
                api_token=api_token,
                study_id=study_id,
            )

            results = collector.download_results(
                output_path=output_file,
                component_id=component_id,
                worker_type=worker_type,
            )

        print_success(f"Collected {len(results)} results: {output_file}")

    except Exception as e:
        print_error(f"Failed to collect data: {e}")
        ctx.exit(1)


@click.command()
@click.argument("data_file", type=click.Path(exists=True, path_type=Path))
@click.pass_context
def show_data_stats(ctx: click.Context, data_file: Path) -> None:
    """Show statistics about collected data.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    data_file : Path
        Path to data file.

    Examples
    --------
    $ bead training show-data-stats results.jsonl
    """
    try:
        print_info(f"Analyzing data: {data_file}")

        # Load and analyze data
        results: list[dict[str, JsonValue]] = []
        with open(data_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                result: dict[str, JsonValue] = json.loads(line)
                results.append(result)

        if not results:
            print_error("No data found in file")
            ctx.exit(1)

        # Calculate statistics
        total_results = len(results)

        # Count unique workers if available
        worker_ids: set[str] = set()
        for result in results:
            if "worker_id" in result and isinstance(result["worker_id"], str):
                worker_ids.add(result["worker_id"])

        # Count response types if available
        response_types: dict[str, int] = {}
        for result in results:
            if "data" in result:
                data: JsonValue = result["data"]
                if isinstance(data, dict):
                    for key in data.keys():  # type: ignore[var-annotated]
                        key_str = str(key)  # type: ignore[arg-type]
                        response_types[key_str] = response_types.get(key_str, 0) + 1

        # Display statistics
        table = Table(title="Data Statistics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Total Results", str(total_results))
        if worker_ids:
            table.add_row("Unique Workers", str(len(worker_ids)))

        if response_types:
            table.add_row("", "")  # Separator
            for resp_type, count in sorted(response_types.items()):
                table.add_row(f"Response Type: {resp_type}", str(count))

        console.print(table)

    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON in data file: {e}")
        ctx.exit(1)
    except Exception as e:
        print_error(f"Failed to show statistics: {e}")
        ctx.exit(1)


@click.command()
@click.option(
    "--model-dir",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
    required=True,
    help="Directory containing trained model",
)
@click.option(
    "--test-items",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to test items (JSONL)",
)
@click.option(
    "--test-labels",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to test labels (JSONL, one label per line)",
)
@click.option(
    "--participant-ids",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to participant IDs (JSONL, one ID per line, optional)",
)
@click.option(
    "--metrics",
    default="accuracy,precision,recall,f1",
    help="Comma-separated list of metrics (accuracy,precision,recall,f1)",
)
@click.option(
    "--average",
    type=click.Choice(["macro", "micro", "weighted"]),
    default="macro",
    help="Averaging strategy for multi-class metrics",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    help="Output path for evaluation report (JSON)",
)
@click.pass_context
def evaluate(
    ctx: click.Context,
    model_dir: Path,
    test_items: Path,
    test_labels: Path,
    participant_ids: Path | None,
    metrics: str,
    average: str,
    output: Path | None,
) -> None:
    r"""Evaluate trained model on test set.

    Loads a trained model and computes evaluation metrics (accuracy, precision,
    recall, F1) on a held-out test set.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    model_dir : Path
        Directory containing trained model.
    test_items : Path
        Path to test items (JSONL).
    test_labels : Path
        Path to test labels (JSONL, one label per line).
    participant_ids : Path | None
        Path to participant IDs (optional, for random effects models).
    metrics : str
        Comma-separated list of metrics to compute.
    average : str
        Averaging strategy for multi-class metrics.
    output : Path | None
        Output path for evaluation report (JSON).

    Examples
    --------
    $ bead training evaluate \\
        --model-dir models/my_model/ \\
        --test-items data/test_items.jsonl \\
        --test-labels data/test_labels.jsonl \\
        --metrics accuracy,f1 \\
        --output evaluation_report.json
    """
    try:
        print_info(f"Evaluating model: {model_dir}")

        # Load model config
        config_path = model_dir / "config.json"
        if not config_path.exists():
            print_error(f"Model config not found: {config_path}")
            ctx.exit(1)

        with open(config_path, encoding="utf-8") as f:
            model_config = json.load(f)

        task_type = model_config.get("task_type")
        if not task_type:
            print_error("Model config missing 'task_type' field")
            ctx.exit(1)

        # Load test items
        items_list = read_jsonlines(test_items, Item)
        print_info(f"Loaded {len(items_list)} test items")

        # Load test labels
        with open(test_labels, encoding="utf-8") as f:
            labels: list[str | int | float] = [
                json.loads(line.strip()) for line in f if line.strip()
            ]

        if len(items_list) != len(labels):
            print_error(f"Mismatch: {len(items_list)} items but {len(labels)} labels")
            ctx.exit(1)

        # Load participant IDs if provided
        participant_ids_list: list[str] | None = None
        if participant_ids:
            with open(participant_ids, encoding="utf-8") as f:
                participant_ids_list = [
                    json.loads(line.strip()) for line in f if line.strip()
                ]
            if len(participant_ids_list) != len(items_list):
                print_error(
                    f"Mismatch: {len(items_list)} items "
                    f"but {len(participant_ids_list)} participant IDs"
                )
                ctx.exit(1)

        # Load model
        model_class = model_class_for_task_type(cast(TaskType, task_type))

        model_instance = model_class.load(model_dir)
        print_success(f"Loaded model from {model_dir}")

        # Make predictions
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Making predictions...", total=None)
            predictions = model_instance.predict(items_list, participant_ids_list)

        # Compute requested metrics
        metrics_list = [m.strip().lower() for m in metrics.split(",")]
        results: dict[str, float] = {}

        for metric_name in metrics_list:
            if metric_name == "accuracy":
                acc = accuracy_score(labels, predictions)
                results["accuracy"] = acc
            elif metric_name in ["precision", "recall", "f1"]:
                precision, recall, f1, support = precision_recall_fscore_support(
                    labels, predictions, average=average, zero_division=0.0
                )
                if "precision" not in results:
                    results["precision"] = float(precision)
                    results["recall"] = float(recall)
                    results["f1"] = float(f1)
                    # support is None when using averaging
                    if support is not None:
                        results["support"] = (
                            float(support)
                            if isinstance(support, int | float)
                            else float(sum(support))
                        )
            else:
                print_error(f"Unknown metric: {metric_name}")
                ctx.exit(1)

        # Display results
        table = Table(title="Evaluation Results")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        for metric_name, value in results.items():
            if metric_name == "support":
                table.add_row(metric_name.capitalize(), f"{int(value)}")
            else:
                table.add_row(metric_name.capitalize(), f"{value:.4f}")

        console.print(table)

        # Save to file if requested
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "model_dir": str(model_dir),
                        "test_items": str(test_items),
                        "test_labels": str(test_labels),
                        "metrics": results,
                        "average": average,
                    },
                    f,
                    indent=2,
                )
            print_success(f"Evaluation report saved: {output}")

    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        ctx.exit(1)
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON: {e}")
        ctx.exit(1)
    except ValueError as e:
        print_error(f"Validation error: {e}")
        ctx.exit(1)
    except ImportError as e:
        print_error(f"Failed to import model class: {e}")
        ctx.exit(1)


@click.command()
@click.option(
    "--items",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to items (JSONL)",
)
@click.option(
    "--labels",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to labels (JSONL, one label per line)",
)
@click.option(
    "--participant-ids",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    help="Path to participant IDs (JSONL, optional)",
)
@click.option(
    "--model-config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to model configuration (JSON/YAML)",
)
@click.option(
    "--k-folds",
    type=int,
    default=5,
    help="Number of folds for cross-validation",
)
@click.option(
    "--stratify-by",
    type=click.Choice(["participant_id", "label", "none"]),
    default="none",
    help="Stratification strategy",
)
@click.option(
    "--random-seed",
    type=int,
    help="Random seed for reproducibility",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    help="Output path for CV results (JSON)",
)
@click.pass_context
def cross_validate(
    ctx: click.Context,
    items: Path,
    labels: Path,
    participant_ids: Path | None,
    model_config: Path,
    k_folds: int,
    stratify_by: str,
    random_seed: int | None,
    output: Path | None,
) -> None:
    r"""Perform K-fold cross-validation.

    Trains model with K-fold cross-validation and reports metrics for each fold.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    items : Path
        Path to items (JSONL).
    labels : Path
        Path to labels (JSONL).
    participant_ids : Path | None
        Path to participant IDs (optional).
    model_config : Path
        Path to model configuration file.
    k_folds : int
        Number of folds.
    stratify_by : str
        Stratification strategy.
    random_seed : int | None
        Random seed for reproducibility.
    output : Path | None
        Output path for results (JSON).

    Examples
    --------
    $ bead training cross-validate \\
        --items data/items.jsonl \\
        --labels data/labels.jsonl \\
        --model-config config.yaml \\
        --k-folds 5 \\
        --stratify-by label \\
        --output cv_results.json
    """
    try:
        print_info(f"Running {k_folds}-fold cross-validation")

        # Load items
        items_list = read_jsonlines(items, Item)
        print_info(f"Loaded {len(items_list)} items")

        # Load labels
        with open(labels, encoding="utf-8") as f:
            labels_list: list[JsonValue] = [
                json.loads(line.strip()) for line in f if line.strip()
            ]

        if len(items_list) != len(labels_list):
            print_error(
                f"Mismatch: {len(items_list)} items but {len(labels_list)} labels"
            )
            ctx.exit(1)

        # Load participant IDs if provided
        participant_ids_list: list[str] | None = None
        if participant_ids:
            with open(participant_ids, encoding="utf-8") as f:
                participant_ids_list = [
                    json.loads(line.strip()) for line in f if line.strip()
                ]
            if len(participant_ids_list) != len(items_list):
                print_error(
                    f"Mismatch: {len(items_list)} items "
                    f"but {len(participant_ids_list)} participant IDs"
                )
                ctx.exit(1)

        # Load model config
        with open(model_config, encoding="utf-8") as f:
            config_dict = json.load(f)

        task_type = config_dict.get("task_type")
        if not task_type:
            print_error("Model config missing 'task_type' field")
            ctx.exit(1)

        # Import model and config classes

        model_class = model_class_for_task_type(cast(TaskType, task_type))
        config_class = config_class_for_task_type(cast(TaskType, task_type))

        # Create cross-validator
        cv = KFold(n_splits=k_folds, shuffle=True, random_state=random_seed)

        # Generate fold indices
        fold_indices = list(cv.split(items_list))

        print_info(f"Generated {len(fold_indices)} folds")

        # Train and evaluate on each fold
        fold_results: list[dict[str, float | int]] = []

        for fold_idx, (train_indices, test_indices) in enumerate(fold_indices, start=1):
            print_info(f"\n[Fold {fold_idx}/{k_folds}]")
            print_info(f"  Train: {len(train_indices)} items")
            print_info(f"  Test: {len(test_indices)} items")

            # Get items for train and test sets
            train_items = [items_list[i] for i in train_indices]
            test_items = [items_list[i] for i in test_indices]

            # Get labels for this fold
            train_labels = [labels_list[i] for i in train_indices]
            test_labels = [labels_list[i] for i in test_indices]

            # Get participant IDs for this fold (if provided)
            train_pids: list[str] | None = None
            test_pids: list[str] | None = None
            if participant_ids_list is not None:
                train_pids = [participant_ids_list[i] for i in train_indices]
                test_pids = [participant_ids_list[i] for i in test_indices]

            # Create and train model for this fold
            print_info("  Training model...")
            model_config_obj = config_class(**config_dict)
            model_instance = model_class(config=model_config_obj)
            model_instance.train(train_items, train_labels, participant_ids=train_pids)

            # Make predictions on test set
            predictions = model_instance.predict(test_items, participant_ids=test_pids)
            pred_labels = [p.predicted_class for p in predictions]

            # Compute metrics
            accuracy = accuracy_score(test_labels, pred_labels)
            precision, recall, f1, support = precision_recall_fscore_support(
                test_labels, pred_labels, average="macro", zero_division=0.0
            )
            prf: dict[str, float] = {
                "precision": float(precision),
                "recall": float(recall),
                "f1": float(f1),
            }
            # support is None when using averaging
            if support is not None:
                prf["support"] = (
                    float(support)
                    if isinstance(support, int | float)
                    else float(sum(support))
                )

            fold_result: dict[str, float | int] = {
                "fold": fold_idx,
                "accuracy": float(accuracy),
                "precision": prf["precision"],
                "recall": prf["recall"],
                "f1": prf["f1"],
            }
            if "support" in prf:
                fold_result["support"] = prf["support"]
            fold_results.append(fold_result)

            print_success(f"  Accuracy: {accuracy:.4f}, F1: {prf['f1']:.4f}")

        # Compute average metrics
        avg_results = {
            "accuracy": np.mean([r["accuracy"] for r in fold_results]),
            "precision": np.mean([r["precision"] for r in fold_results]),
            "recall": np.mean([r["recall"] for r in fold_results]),
            "f1": np.mean([r["f1"] for r in fold_results]),
        }

        # Display summary
        console.rule("[bold]Cross-Validation Summary[/bold]")
        table = Table()
        table.add_column("Metric", style="cyan")
        table.add_column("Mean", style="green", justify="right")
        table.add_column("Std", style="yellow", justify="right")

        for metric_name in ["accuracy", "precision", "recall", "f1"]:
            values = [r[metric_name] for r in fold_results]
            mean_val = np.mean(values)
            std_val = np.std(values)
            table.add_row(metric_name.capitalize(), f"{mean_val:.4f}", f"{std_val:.4f}")

        console.print(table)

        # Save results
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "k_folds": k_folds,
                        "stratify_by": stratify_by,
                        "fold_results": fold_results,
                        "average_metrics": avg_results,
                    },
                    f,
                    indent=2,
                )
            print_success(f"CV results saved: {output}")

    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        ctx.exit(1)
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON: {e}")
        ctx.exit(1)
    except ValueError as e:
        print_error(f"Validation error: {e}")
        ctx.exit(1)


@click.command()
@click.option(
    "--items",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to items (JSONL)",
)
@click.option(
    "--labels",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to labels (JSONL)",
)
@click.option(
    "--model-config",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to model configuration",
)
@click.option(
    "--train-sizes",
    default="0.1,0.2,0.5,0.8,1.0",
    help="Comma-separated training set sizes (fractions)",
)
@click.option(
    "--random-seed",
    type=int,
    help="Random seed for reproducibility",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    help="Output path for learning curve data (JSON)",
)
@click.pass_context
def learning_curve(
    ctx: click.Context,
    items: Path,
    labels: Path,
    model_config: Path,
    train_sizes: str,
    random_seed: int | None,
    output: Path | None,
) -> None:
    r"""Generate learning curve with varying training set sizes.

    Trains models with increasing amounts of training data and plots
    training/validation performance.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    items : Path
        Path to items (JSONL).
    labels : Path
        Path to labels (JSONL).
    model_config : Path
        Path to model configuration.
    train_sizes : str
        Comma-separated training set sizes (fractions).
    random_seed : int | None
        Random seed for reproducibility.
    output : Path | None
        Output path for results (JSON).

    Examples
    --------
    $ bead training learning-curve \\
        --items data/items.jsonl \\
        --labels data/labels.jsonl \\
        --model-config config.yaml \\
        --train-sizes 0.1,0.2,0.5,1.0 \\
        --output learning_curve.json
    """
    try:
        print_info("Generating learning curve")

        # Load items
        items_list = read_jsonlines(items, Item)
        print_info(f"Loaded {len(items_list)} items")

        # Load labels
        with open(labels, encoding="utf-8") as f:
            labels_list: list[str | int | float] = [
                json.loads(line.strip()) for line in f if line.strip()
            ]

        # Load model config
        with open(model_config, encoding="utf-8") as f:
            config_dict = json.load(f)

        task_type = config_dict.get("task_type")
        if not task_type:
            print_error("Model config missing 'task_type' field")
            ctx.exit(1)

        # Import model and config classes

        model_class = model_class_for_task_type(cast(TaskType, task_type))
        config_class = config_class_for_task_type(cast(TaskType, task_type))

        # Parse train sizes
        sizes = [float(s.strip()) for s in train_sizes.split(",")]
        if any(s <= 0 or s > 1 for s in sizes):
            print_error("Train sizes must be in range (0, 1]")
            ctx.exit(1)

        # Train with different data sizes
        curve_results: list[dict[str, float]] = []

        for size in track(sizes, description="Training with varying data sizes"):
            n_samples = int(len(items_list) * size)
            print_info(f"\nTraining with {n_samples} samples ({size:.0%})")

            # Split into train/test (80/20)
            split_idx = int(n_samples * 0.8)
            train_items_subset = items_list[:split_idx]
            test_items_subset = items_list[split_idx:n_samples]
            train_labels_subset = labels_list[:split_idx]
            test_labels_subset = labels_list[split_idx:n_samples]

            # Train model
            print_info("  Training...")
            model_config_obj = config_class(**config_dict)
            model_instance = model_class(config=model_config_obj)
            # Note: participant_ids=None for fixed effects models
            model_instance.train(
                train_items_subset, train_labels_subset, participant_ids=None
            )

            # Make predictions
            train_predictions = model_instance.predict(
                train_items_subset, participant_ids=None
            )
            test_predictions = model_instance.predict(
                test_items_subset, participant_ids=None
            )

            # Compute metrics
            train_acc = accuracy_score(train_labels_subset, train_predictions)
            test_acc = accuracy_score(test_labels_subset, test_predictions)

            curve_results.append(
                {
                    "train_size": size,
                    "n_samples": n_samples,
                    "train_accuracy": train_acc,
                    "test_accuracy": test_acc,
                }
            )

            print_success(f"  Train acc: {train_acc:.4f}, Test acc: {test_acc:.4f}")

        # Display summary
        console.rule("[bold]Learning Curve Summary[/bold]")
        table = Table()
        table.add_column("Train Size", style="cyan")
        table.add_column("N Samples", style="blue", justify="right")
        table.add_column("Train Acc", style="green", justify="right")
        table.add_column("Test Acc", style="yellow", justify="right")

        for result in curve_results:
            table.add_row(
                f"{result['train_size']:.0%}",
                str(result["n_samples"]),
                f"{result['train_accuracy']:.4f}",
                f"{result['test_accuracy']:.4f}",
            )

        console.print(table)

        # Save results
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w", encoding="utf-8") as f:
                json.dump({"curve_data": curve_results}, f, indent=2)
            print_success(f"Learning curve data saved: {output}")

    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        ctx.exit(1)
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON: {e}")
        ctx.exit(1)
    except ValueError as e:
        print_error(f"Validation error: {e}")
        ctx.exit(1)


@click.command()
@click.option(
    "--annotations",
    type=click.Path(exists=True, dir_okay=False, path_type=Path),
    required=True,
    help="Path to annotations (JSONL with 'rater_id' and 'label' fields)",
)
@click.option(
    "--metric",
    type=click.Choice(
        [
            "krippendorff_alpha",
            "fleiss_kappa",
            "cohens_kappa",
            "percentage_agreement",
        ]
    ),
    default="krippendorff_alpha",
    help="Agreement metric to compute",
)
@click.option(
    "--data-type",
    type=click.Choice(["nominal", "ordinal", "interval", "ratio"]),
    default="nominal",
    help="Data type for Krippendorff's alpha",
)
@click.option(
    "--output",
    type=click.Path(path_type=Path),
    help="Output path for agreement report (JSON)",
)
@click.pass_context
def compute_agreement(
    ctx: click.Context,
    annotations: Path,
    metric: str,
    data_type: str,
    output: Path | None,
) -> None:
    r"""Compute inter-annotator agreement.

    Calculates agreement metrics (Cohen's kappa, Fleiss' kappa, Krippendorff's
    alpha, or percentage agreement) from multi-rater annotations.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    annotations : Path
        Path to annotations file (JSONL).
    metric : str
        Agreement metric to compute.
    data_type : str
        Data type for Krippendorff's alpha.
    output : Path | None
        Output path for report (JSON).

    Examples
    --------
    $ bead training compute-agreement \\
        --annotations data/annotations.jsonl \\
        --metric krippendorff_alpha \\
        --data-type nominal \\
        --output agreement_report.json

    $ bead training compute-agreement \\
        --annotations data/annotations.jsonl \\
        --metric cohens_kappa
    """
    try:
        print_info(f"Computing {metric.replace('_', ' ').title()}")

        # Load annotations
        with open(annotations, encoding="utf-8") as f:
            annotation_records = [json.loads(line) for line in f if line.strip()]

        print_info(f"Loaded {len(annotation_records)} annotation records")

        # Organize annotations by rater
        rater_annotations: dict[str, list[str | int | float]] = {}
        for record in annotation_records:
            rater_id = str(record.get("rater_id", "unknown"))
            label = record.get("label")
            if rater_id not in rater_annotations:
                rater_annotations[rater_id] = []
            rater_annotations[rater_id].append(label)

        n_raters = len(rater_annotations)
        print_info(f"Found {n_raters} raters")

        # Compute agreement metric
        agreement_score: float
        if metric == "percentage_agreement":
            if n_raters != 2:
                print_error("Percentage agreement requires exactly 2 raters")
                ctx.exit(1)
            rater_ids = list(rater_annotations.keys())
            agreement_score = InterAnnotatorMetrics.percentage_agreement(
                rater_annotations[rater_ids[0]], rater_annotations[rater_ids[1]]
            )
        elif metric == "cohens_kappa":
            if n_raters != 2:
                print_error("Cohen's kappa requires exactly 2 raters")
                ctx.exit(1)
            rater_ids = list(rater_annotations.keys())
            agreement_score = InterAnnotatorMetrics.cohens_kappa(
                rater_annotations[rater_ids[0]], rater_annotations[rater_ids[1]]
            )
        elif metric == "fleiss_kappa":
            # Convert to ratings matrix format
            # Matrix shape: (n_items, n_categories)
            all_labels = set()
            for labels in rater_annotations.values():
                all_labels.update(labels)
            categories = sorted(all_labels)
            n_items = len(next(iter(rater_annotations.values())))

            ratings_matrix = np.zeros((n_items, len(categories)), dtype=int)
            for labels in rater_annotations.values():
                for item_idx, label in enumerate(labels):
                    cat_idx = categories.index(label)
                    ratings_matrix[item_idx, cat_idx] += 1

            agreement_score = InterAnnotatorMetrics.fleiss_kappa(
                cast(np.ndarray[int, np.dtype[np.int_]], ratings_matrix)  # type: ignore[misc,valid-type]
            )
        elif metric == "krippendorff_alpha":
            agreement_score = InterAnnotatorMetrics.krippendorff_alpha(
                rater_annotations, metric=data_type
            )
        else:
            print_error(f"Unknown metric: {metric}")
            ctx.exit(1)

        # Display result
        table = Table(title="Inter-Annotator Agreement")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")
        table.add_column("Interpretation", style="yellow")

        # Interpretation guidelines (Landis & Koch, 1977)
        if agreement_score < 0:
            interpretation = "Poor"
        elif agreement_score < 0.2:
            interpretation = "Slight"
        elif agreement_score < 0.4:
            interpretation = "Fair"
        elif agreement_score < 0.6:
            interpretation = "Moderate"
        elif agreement_score < 0.8:
            interpretation = "Substantial"
        else:
            interpretation = "Almost Perfect"

        table.add_row(
            metric.replace("_", " ").title(),
            f"{agreement_score:.4f}",
            interpretation,
        )
        table.add_row("N Raters", str(n_raters), "")
        table.add_row("N Items", str(len(annotation_records) // n_raters), "")

        console.print(table)

        # Save results
        if output:
            output.parent.mkdir(parents=True, exist_ok=True)
            with open(output, "w", encoding="utf-8") as f:
                data_type_value = data_type if metric == "krippendorff_alpha" else None
                json.dump(
                    {
                        "metric": metric,
                        "data_type": data_type_value,
                        "score": agreement_score,
                        "interpretation": interpretation,
                        "n_raters": n_raters,
                        "n_items": len(annotation_records) // n_raters,
                    },
                    f,
                    indent=2,
                )
            print_success(f"Agreement report saved: {output}")

    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        ctx.exit(1)
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON: {e}")
        ctx.exit(1)
    except ValueError as e:
        print_error(f"Validation error: {e}")
        ctx.exit(1)


# Register commands
training.add_command(collect_data)
training.add_command(show_data_stats)
training.add_command(evaluate)
training.add_command(cross_validate)
training.add_command(learning_curve)
training.add_command(compute_agreement)
