"""Model training commands for bead CLI.

This module provides commands for training GLMM models across all 8 task types
with support for fixed effects, random intercepts, and random slopes.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Literal, cast

import click
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from bead.active_learning.config import MixedEffectsConfig
from bead.active_learning.models import (
    MODEL_CLASSES,
    config_class_for_task_type,
    model_class_for_task_type,
)
from bead.cli.display import (
    print_error,
    print_info,
    print_success,
)
from bead.data.serialization import read_jsonlines
from bead.items.item import Item
from bead.items.item_template import TaskType

console = Console()


@click.group()
def models() -> None:
    r"""Model training commands.

    Commands for training GLMM models for judgment prediction across all 8
    task types with support for mixed effects modeling.

    \b
    Task Types:
      • forced_choice  - 2AFC, 3AFC, N-way forced choice
      • categorical    - Unordered categories (NLI, semantic relations)
      • binary         - Yes/No, True/False
      • multi_select   - Multiple selection (checkboxes)
      • ordinal_scale  - Likert scales, sliders
      • magnitude      - Unbounded numeric (reading time, confidence)
      • free_text      - Open-ended text responses
      • cloze          - Fill-in-the-blank

    \b
    Mixed Effects Modes:
      • fixed              - Fixed effects only (no participant variability)
      • random_intercepts  - Participant-specific biases
      • random_slopes      - Participant-specific model parameters

    \b
    Examples:
        # Train forced choice model with fixed effects
        $ bead models train-model \\
            --task-type forced_choice \\
            --items items.jsonl \\
            --labels labels.jsonl \\
            --output-dir models/fc_model/

        # Train with random intercepts
        $ bead models train-model \\
            --task-type ordinal_scale \\
            --items items.jsonl \\
            --labels labels.jsonl \\
            --participant-ids participant_ids.txt \\
            --mixed-effects-mode random_intercepts \\
            --output-dir models/os_model/

        # Make predictions
        $ bead models predict \\
            --model-dir models/fc_model/ \\
            --items test_items.jsonl \\
            --output predictions.jsonl
    """


@click.command()
@click.option(
    "--task-type",
    required=True,
    type=click.Choice(list(MODEL_CLASSES.keys())),
    help="Task type for model",
)
@click.option(
    "--items",
    "items_file",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to items JSONL file",
)
@click.option(
    "--labels",
    "labels_file",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to labels JSONL file (list of response strings)",
)
@click.option(
    "--participant-ids",
    "participant_ids_file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to participant IDs file (one ID per line, aligned with labels)",
)
@click.option(
    "--validation-items",
    type=click.Path(exists=True, path_type=Path),
    help="Path to validation items JSONL file (optional)",
)
@click.option(
    "--validation-labels",
    type=click.Path(exists=True, path_type=Path),
    help="Path to validation labels JSONL file (optional)",
)
@click.option(
    "--output-dir",
    required=True,
    type=click.Path(path_type=Path),
    help="Output directory for trained model",
)
@click.option(
    "--model-name",
    default="bert-base-uncased",
    help="HuggingFace model name",
)
@click.option(
    "--mixed-effects-mode",
    type=click.Choice(["fixed", "random_intercepts", "random_slopes"]),
    default="fixed",
    help="Mixed effects mode",
)
@click.option(
    "--max-length",
    type=int,
    default=128,
    help="Maximum sequence length for tokenization",
)
@click.option(
    "--learning-rate",
    type=float,
    default=2e-5,
    help="Learning rate for AdamW optimizer",
)
@click.option(
    "--batch-size",
    type=int,
    default=16,
    help="Batch size for training",
)
@click.option(
    "--num-epochs",
    type=int,
    default=3,
    help="Number of training epochs",
)
@click.option(
    "--device",
    type=click.Choice(["cpu", "cuda", "mps"]),
    default="cpu",
    help="Device to train on",
)
@click.option(
    "--use-lora",
    is_flag=True,
    help="Use LoRA parameter-efficient fine-tuning",
)
@click.option(
    "--lora-rank",
    type=int,
    default=8,
    help="LoRA rank (r)",
)
@click.option(
    "--lora-alpha",
    type=int,
    default=16,
    help="LoRA alpha scaling parameter",
)
@click.pass_context
def train_model(
    ctx: click.Context,
    task_type: str,
    items_file: Path,
    labels_file: Path,
    participant_ids_file: Path | None,
    validation_items: Path | None,
    validation_labels: Path | None,
    output_dir: Path,
    model_name: str,
    mixed_effects_mode: str,
    max_length: int,
    learning_rate: float,
    batch_size: int,
    num_epochs: int,
    device: str,
    use_lora: bool,
    lora_rank: int,
    lora_alpha: int,
) -> None:
    r"""Train GLMM model for judgment prediction.

    Trains a generalized linear mixed model (GLMM) with support for:
    - Fixed effects (population-level parameters)
    - Random intercepts (participant-specific biases)
    - Random slopes (participant-specific model parameters)

    The model uses a transformer encoder (default: BERT) with optional
    LoRA parameter-efficient fine-tuning.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    task_type : str
        Task type (forced_choice, categorical, binary, etc.).
    items_file : Path
        Path to items JSONL file.
    labels_file : Path
        Path to labels JSONL file (one label per line).
    participant_ids_file : Path | None
        Path to participant IDs file (required for random effects).
    validation_items : Path | None
        Path to validation items JSONL file (optional).
    validation_labels : Path | None
        Path to validation labels JSONL file (optional).
    output_dir : Path
        Output directory for trained model.
    model_name : str
        HuggingFace model name.
    mixed_effects_mode : str
        Mixed effects mode (fixed, random_intercepts, random_slopes).
    max_length : int
        Maximum sequence length for tokenization.
    learning_rate : float
        Learning rate for AdamW optimizer.
    batch_size : int
        Batch size for training.
    num_epochs : int
        Number of training epochs.
    device : str
        Device to train on (cpu, cuda, mps).
    use_lora : bool
        Whether to use LoRA fine-tuning.
    lora_rank : int
        LoRA rank.
    lora_alpha : int
        LoRA alpha scaling parameter.

    Examples
    --------
    $ bead models train-model \\
        --task-type forced_choice \\
        --items items.jsonl \\
        --labels labels.jsonl \\
        --output-dir models/fc_model/ \\
        --num-epochs 5

    $ bead models train-model \\
        --task-type ordinal_scale \\
        --items items.jsonl \\
        --labels labels.jsonl \\
        --participant-ids participant_ids.txt \\
        --mixed-effects-mode random_intercepts \\
        --output-dir models/os_model/ \\
        --device cuda \\
        --use-lora \\
        --lora-rank 8
    """
    try:
        # Validate mixed effects mode requirements
        if mixed_effects_mode != "fixed" and participant_ids_file is None:
            print_error(
                f"Mixed effects mode '{mixed_effects_mode}' requires "
                "--participant-ids parameter"
            )
            print_info(
                "Provide a file with one participant ID per line, "
                "aligned with the labels file"
            )
            ctx.exit(1)

        print_info(f"Training {task_type} model with {mixed_effects_mode} mode")

        # Load items
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Loading items...", total=None)
            items = read_jsonlines(items_file, Item)

        print_success(f"Loaded {len(items)} items")

        # Load labels
        with open(labels_file, encoding="utf-8") as f:
            labels = [line.strip() for line in f if line.strip()]

        if len(labels) != len(items):
            print_error(
                f"Number of labels ({len(labels)}) does not match "
                f"number of items ({len(items)})"
            )
            ctx.exit(1)

        print_success(f"Loaded {len(labels)} labels")

        # Load participant IDs if provided
        participant_ids = None
        if participant_ids_file:
            with open(participant_ids_file, encoding="utf-8") as f:
                participant_ids = [line.strip() for line in f if line.strip()]

            if len(participant_ids) != len(items):
                print_error(
                    f"Number of participant IDs ({len(participant_ids)}) does not "
                    f"match number of items ({len(items)})"
                )
                ctx.exit(1)

            unique_participants = len(set(participant_ids))
            print_success(
                f"Loaded {len(participant_ids)} participant IDs "
                f"({unique_participants} unique participants)"
            )

        # Load validation data if provided
        val_items = None
        val_labels = None
        if validation_items and validation_labels:
            val_items = read_jsonlines(validation_items, Item)

            with open(validation_labels, encoding="utf-8") as f:
                val_labels = [line.strip() for line in f if line.strip()]

            if len(val_labels) != len(val_items):
                print_error(
                    f"Number of validation labels ({len(val_labels)}) does not "
                    f"match number of validation items ({len(val_items)})"
                )
                ctx.exit(1)

            print_success(f"Loaded {len(val_items)} validation items")

        # Build mixed effects config
        # Cast to proper Literal type since Click validates the value
        mode = cast(
            Literal["fixed", "random_intercepts", "random_slopes"],
            mixed_effects_mode,
        )
        mixed_effects_config = MixedEffectsConfig(mode=mode)

        # Import model class and config dynamically
        model_class = model_class_for_task_type(cast(TaskType, task_type))
        config_class = config_class_for_task_type(cast(TaskType, task_type))

        # Build model config
        config_dict = {
            "model_name": model_name,
            "max_length": max_length,
            "learning_rate": learning_rate,
            "batch_size": batch_size,
            "num_epochs": num_epochs,
            "device": device,
            "mixed_effects": mixed_effects_config,
        }

        # Add LoRA config if enabled
        if use_lora:
            config_dict["use_lora"] = True
            config_dict["lora_rank"] = lora_rank
            config_dict["lora_alpha"] = lora_alpha

        model_config = config_class(**config_dict)

        # Initialize model
        console.rule("[bold]Initializing Model[/bold]")
        model = model_class(config=model_config)

        # Train model
        console.rule("[bold]Training Model[/bold]")
        print_info(
            f"Training for {num_epochs} epochs on {device} "
            f"(batch_size={batch_size}, lr={learning_rate})"
        )

        if use_lora:
            print_info(f"Using LoRA fine-tuning (rank={lora_rank}, alpha={lora_alpha})")

        metrics = model.train(
            items=items,
            labels=labels,
            participant_ids=participant_ids,
            validation_items=val_items,
            validation_labels=val_labels,
        )

        # Display training metrics
        console.rule("[bold]Training Results[/bold]")
        table = Table(title="Training Metrics")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        for metric_name, metric_value in metrics.items():
            if isinstance(metric_value, float):
                table.add_row(metric_name, f"{metric_value:.4f}")
            else:
                table.add_row(metric_name, str(metric_value))

        console.print(table)

        # Save model
        console.rule("[bold]Saving Model[/bold]")
        output_dir.mkdir(parents=True, exist_ok=True)

        # Save model weights
        model_path = output_dir / "model.pt"
        model.save(model_path)
        print_success(f"Saved model weights: {model_path}")

        # Save config with task_type for later inference
        config_path = output_dir / "config.json"
        config_with_task_type = model_config.model_dump()
        config_with_task_type["task_type"] = task_type  # Add task type to config
        with open(config_path, "w", encoding="utf-8") as f:
            json.dump(config_with_task_type, f, indent=2)
        print_success(f"Saved config: {config_path}")

        # Save training metrics
        metrics_path = output_dir / "training_metrics.json"
        with open(metrics_path, "w", encoding="utf-8") as f:
            json.dump(metrics, f, indent=2)
        print_success(f"Saved training metrics: {metrics_path}")

        console.rule("[bold green]✓ Training Complete[/bold green]")

    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        ctx.exit(1)
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON in file: {e}")
        ctx.exit(1)
    except ValueError as e:
        print_error(f"Invalid configuration or data: {e}")
        ctx.exit(1)
    except (ImportError, AttributeError) as e:
        print_error(f"Failed to import model class: {e}")
        print_info(
            "This may indicate a corrupted installation. "
            "Try reinstalling bead with: pip install --force-reinstall bead"
        )
        ctx.exit(1)


@click.command()
@click.option(
    "--model-dir",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to trained model directory",
)
@click.option(
    "--items",
    "items_file",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to items JSONL file",
)
@click.option(
    "--participant-ids",
    "participant_ids_file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to participant IDs file (required for random effects models)",
)
@click.option(
    "--output",
    "output_file",
    required=True,
    type=click.Path(path_type=Path),
    help="Output path for predictions JSONL",
)
@click.pass_context
def predict(
    ctx: click.Context,
    model_dir: Path,
    items_file: Path,
    participant_ids_file: Path | None,
    output_file: Path,
) -> None:
    r"""Make predictions with trained model.

    Predicts class labels for items using a trained GLMM model.
    For random effects models, participant IDs are required to compute
    participant-specific predictions.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    model_dir : Path
        Path to trained model directory.
    items_file : Path
        Path to items JSONL file.
    participant_ids_file : Path | None
        Path to participant IDs file (required for random effects).
    output_file : Path
        Output path for predictions JSONL.

    Examples
    --------
    $ bead models predict \\
        --model-dir models/fc_model/ \\
        --items test_items.jsonl \\
        --output predictions.jsonl

    $ bead models predict \\
        --model-dir models/os_model/ \\
        --items test_items.jsonl \\
        --participant-ids participant_ids.txt \\
        --output predictions.jsonl
    """
    try:
        print_info(f"Loading model from {model_dir}")

        # Load config
        config_path = model_dir / "config.json"
        if not config_path.exists():
            print_error(f"Model config not found: {config_path}")
            ctx.exit(1)

        with open(config_path, encoding="utf-8") as f:
            config_dict = json.load(f)

        # Get task type from config
        if "task_type" not in config_dict:
            print_error(
                "Model config missing 'task_type' field. "
                "This model may have been trained with an older version of bead."
            )
            print_info("Valid task types: " + ", ".join(MODEL_CLASSES.keys()))
            ctx.exit(1)

        task_type = config_dict["task_type"]
        if task_type not in MODEL_CLASSES:
            print_error(
                f"Unknown task type '{task_type}' in model config. "
                f"Valid types: {', '.join(MODEL_CLASSES.keys())}"
            )
            ctx.exit(1)

        print_success(f"Detected task type: {task_type}")

        # Import model class
        model_class = model_class_for_task_type(cast(TaskType, task_type))
        config_class = config_class_for_task_type(cast(TaskType, task_type))
        model_config = config_class(**config_dict)

        # Initialize model and load weights
        model = model_class(config=model_config)
        model_path = model_dir / "model.pt"
        if not model_path.exists():
            print_error(f"Model weights not found: {model_path}")
            ctx.exit(1)

        model.load(model_path)
        print_success(f"Loaded model: {model_path}")

        # Load items
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Loading items...", total=None)
            items = read_jsonlines(items_file, Item)

        print_success(f"Loaded {len(items)} items")

        # Load participant IDs if provided
        participant_ids = None
        if participant_ids_file:
            with open(participant_ids_file, encoding="utf-8") as f:
                participant_ids = [line.strip() for line in f if line.strip()]

            if len(participant_ids) != len(items):
                print_error(
                    f"Number of participant IDs ({len(participant_ids)}) does not "
                    f"match number of items ({len(items)})"
                )
                ctx.exit(1)

            print_success(f"Loaded {len(participant_ids)} participant IDs")

        # Make predictions
        console.rule("[bold]Making Predictions[/bold]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Predicting...", total=None)
            predictions = model.predict(items=items, participant_ids=participant_ids)

        # Save predictions
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            for pred in predictions:
                f.write(pred.model_dump_json() + "\n")

        print_success(f"Saved {len(predictions)} predictions: {output_file}")

        # Display sample predictions
        console.rule("[bold]Sample Predictions[/bold]")
        table = Table(title="First 5 Predictions")
        table.add_column("Index", style="cyan", justify="right")
        table.add_column("Predicted Label", style="green")
        table.add_column("Confidence", style="yellow", justify="right")

        for i, pred in enumerate(predictions[:5]):
            confidence = pred.confidence if hasattr(pred, "confidence") else "N/A"
            if isinstance(confidence, float):
                confidence_str = f"{confidence:.3f}"
            else:
                confidence_str = str(confidence)
            table.add_row(str(i), str(pred.predicted_label), confidence_str)

        console.print(table)

    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        ctx.exit(1)
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON in file: {e}")
        ctx.exit(1)
    except ValueError as e:
        print_error(f"Invalid configuration or data: {e}")
        ctx.exit(1)
    except (ImportError, AttributeError) as e:
        print_error(f"Failed to import model class: {e}")
        print_info(
            "This may indicate a corrupted installation. "
            "Try reinstalling bead with: pip install --force-reinstall bead"
        )
        ctx.exit(1)


@click.command()
@click.option(
    "--model-dir",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to trained model directory",
)
@click.option(
    "--items",
    "items_file",
    required=True,
    type=click.Path(exists=True, path_type=Path),
    help="Path to items JSONL file",
)
@click.option(
    "--participant-ids",
    "participant_ids_file",
    type=click.Path(exists=True, path_type=Path),
    help="Path to participant IDs file (required for random effects models)",
)
@click.option(
    "--output",
    "output_file",
    required=True,
    type=click.Path(path_type=Path),
    help="Output path for probabilities JSON",
)
@click.pass_context
def predict_proba(
    ctx: click.Context,
    model_dir: Path,
    items_file: Path,
    participant_ids_file: Path | None,
    output_file: Path,
) -> None:
    r"""Predict class probabilities with trained model.

    Predicts class probability distributions for items using a trained GLMM
    model. For random effects models, participant IDs are required.

    Parameters
    ----------
    ctx : click.Context
        Click context object.
    model_dir : Path
        Path to trained model directory.
    items_file : Path
        Path to items JSONL file.
    participant_ids_file : Path | None
        Path to participant IDs file (required for random effects).
    output_file : Path
        Output path for probabilities JSON.

    Examples
    --------
    $ bead models predict-proba \\
        --model-dir models/fc_model/ \\
        --items test_items.jsonl \\
        --output probabilities.json
    """
    try:
        print_info(f"Loading model from {model_dir}")

        # Load config
        config_path = model_dir / "config.json"
        if not config_path.exists():
            print_error(f"Model config not found: {config_path}")
            ctx.exit(1)

        with open(config_path, encoding="utf-8") as f:
            config_dict = json.load(f)

        # Get task type from config
        if "task_type" not in config_dict:
            print_error(
                "Model config missing 'task_type' field. "
                "This model may have been trained with an older version of bead."
            )
            print_info("Valid task types: " + ", ".join(MODEL_CLASSES.keys()))
            ctx.exit(1)

        task_type = config_dict["task_type"]
        if task_type not in MODEL_CLASSES:
            print_error(
                f"Unknown task type '{task_type}' in model config. "
                f"Valid types: {', '.join(MODEL_CLASSES.keys())}"
            )
            ctx.exit(1)

        print_success(f"Detected task type: {task_type}")

        # Import model class
        model_class = model_class_for_task_type(cast(TaskType, task_type))
        config_class = config_class_for_task_type(cast(TaskType, task_type))
        model_config = config_class(**config_dict)

        # Initialize model and load weights
        model = model_class(config=model_config)
        model_path = model_dir / "model.pt"
        if not model_path.exists():
            print_error(f"Model weights not found: {model_path}")
            ctx.exit(1)

        model.load(model_path)
        print_success(f"Loaded model: {model_path}")

        # Load items
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Loading items...", total=None)
            items = read_jsonlines(items_file, Item)

        print_success(f"Loaded {len(items)} items")

        # Load participant IDs if provided
        participant_ids = None
        if participant_ids_file:
            with open(participant_ids_file, encoding="utf-8") as f:
                participant_ids = [line.strip() for line in f if line.strip()]

            if len(participant_ids) != len(items):
                print_error(
                    f"Number of participant IDs ({len(participant_ids)}) does not "
                    f"match number of items ({len(items)})"
                )
                ctx.exit(1)

            print_success(f"Loaded {len(participant_ids)} participant IDs")

        # Predict probabilities
        console.rule("[bold]Predicting Probabilities[/bold]")
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task("Predicting...", total=None)
            probabilities = model.predict_proba(
                items=items, participant_ids=participant_ids
            )

        # Save probabilities
        output_file.parent.mkdir(parents=True, exist_ok=True)
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(probabilities.tolist(), f, indent=2)

        print_success(
            f"Saved {len(probabilities)} probability distributions: {output_file}"
        )

        # Display sample probabilities
        console.rule("[bold]Sample Probabilities[/bold]")
        table = Table(title="First 5 Probability Distributions")
        table.add_column("Index", style="cyan", justify="right")
        table.add_column("Probabilities", style="green")

        for i, prob in enumerate(probabilities[:5]):
            prob_str = ", ".join([f"{p:.3f}" for p in prob])
            table.add_row(str(i), f"[{prob_str}]")

        console.print(table)

    except FileNotFoundError as e:
        print_error(f"File not found: {e}")
        ctx.exit(1)
    except json.JSONDecodeError as e:
        print_error(f"Invalid JSON in file: {e}")
        ctx.exit(1)
    except ValueError as e:
        print_error(f"Invalid configuration or data: {e}")
        ctx.exit(1)
    except (ImportError, AttributeError) as e:
        print_error(f"Failed to import model class: {e}")
        print_info(
            "This may indicate a corrupted installation. "
            "Try reinstalling bead with: pip install --force-reinstall bead"
        )
        ctx.exit(1)


# Register commands
models.add_command(train_model)
models.add_command(predict)
models.add_command(predict_proba)
