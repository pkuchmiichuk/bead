#!/usr/bin/env python3
"""Train the initial acceptability model on MegaAcceptability-derived 2AFC pairs.

This script consumes the forced-choice training items written by
``prepare_megaacceptability.py`` and fits a :class:`ForcedChoiceModel` with
participant random effects. Each item carries its gold ``label`` (``option_a`` or
``option_b``) and the annotator ``participant_id`` in metadata, so the model can fit
a per-annotator random intercept on top of the shared acceptability classifier. The
trained model is saved to a checkpoint directory for use as the active-learning
initializer.

Real training pulls torch, transformers, and BERT weights and is slow on CPU, so the
heavy imports are deferred into the training path. Run ``--self-test`` to validate
the config wiring and confirm a model config can be built without loading any model
weights or running training.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import TYPE_CHECKING

import yaml

from bead.cli.display import (
    confirm,
    console,
    create_summary_table,
    print_error,
    print_header,
    print_info,
    print_success,
    print_warning,
)
from bead.items.item import Item

if TYPE_CHECKING:
    from bead.config.active_learning import ForcedChoiceModelConfig

DEFAULT_MIXED_EFFECTS_MODE = "random_intercepts"
DEFAULT_MODEL_NAME = "bert-base-uncased"
DEFAULT_DEVICE = "cpu"
DEFAULT_LEARNING_RATE = 2e-5
DEFAULT_BATCH_SIZE = 16
DEFAULT_EPOCHS = 3
DEFAULT_CHECKPOINT_DIR = "checkpoints/acceptability_init"
DEFAULT_TRAINING_ITEMS = "items/megaacceptability_2afc.jsonl"


def load_config(config_path: Path) -> dict[str, object]:
    """Load the YAML configuration file."""
    with open(config_path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_acceptability_config(config: dict[str, object]) -> dict[str, object]:
    """Return the ``acceptability_model`` section, or an empty dict if absent."""
    section = config.get("acceptability_model")
    return section if isinstance(section, dict) else {}


def load_training_items(path: Path, *, limit: int | None = None) -> list[Item]:
    """Load derived 2AFC training items from JSONL written by the prepare script."""
    items: list[Item] = []
    with open(path, encoding="utf-8") as handle:
        for index, line in enumerate(handle):
            if limit is not None and index >= limit:
                break
            line = line.strip()
            if not line:
                continue
            items.append(Item.model_validate_json(line))
    return items


def extract_labels_and_participants(
    items: list[Item],
) -> tuple[list[str], list[str]]:
    """Pull gold labels and participant ids from each item's metadata.

    Parameters
    ----------
    items
        Forced-choice training items, each with ``label`` and ``participant_id``
        keys in ``item_metadata``.

    Returns
    -------
    tuple[list[str], list[str]]
        Parallel lists of option-name labels and participant identifiers.

    Raises
    ------
    ValueError
        If any item is missing a ``label`` or ``participant_id``.
    """
    labels: list[str] = []
    participant_ids: list[str] = []
    for item in items:
        label = item.item_metadata.get("label")
        participant_id = item.item_metadata.get("participant_id")
        if label is None or participant_id is None:
            raise ValueError(
                "Each training item must carry 'label' and 'participant_id' in "
                "item_metadata. Regenerate items with prepare_megaacceptability.py."
            )
        labels.append(str(label))
        participant_ids.append(str(participant_id))
    return labels, participant_ids


def build_model_config(section: dict[str, object]) -> ForcedChoiceModelConfig:
    """Build a ForcedChoiceModelConfig from the acceptability config section.

    Parameters
    ----------
    section
        The ``acceptability_model`` config section (possibly empty), read with
        defensive defaults.

    Returns
    -------
    ForcedChoiceModelConfig
        Config wired for single-encoder 2AFC training with participant random
        effects.
    """
    from bead.active_learning.config import MixedEffectsConfig  # noqa: PLC0415
    from bead.config.active_learning import ForcedChoiceModelConfig  # noqa: PLC0415

    mode = str(section.get("mixed_effects_mode", DEFAULT_MIXED_EFFECTS_MODE))
    return ForcedChoiceModelConfig(
        model_name=str(section.get("model_name", DEFAULT_MODEL_NAME)),
        encoder_mode="single_encoder",
        learning_rate=float(section.get("learning_rate", DEFAULT_LEARNING_RATE)),
        batch_size=int(section.get("batch_size", DEFAULT_BATCH_SIZE)),
        num_epochs=int(section.get("epochs", DEFAULT_EPOCHS)),
        device=str(section.get("device", DEFAULT_DEVICE)),
        early_stopping_patience=int(section.get("early_stopping_patience", 2)),
        mixed_effects=MixedEffectsConfig(mode=mode),
    )


def run_self_test(config_path: Path) -> int:
    """Validate config wiring by building a model config without loading weights."""
    print_header("train_acceptability_model self-test")

    config: dict[str, object] = {}
    if config_path.exists():
        config = load_config(config_path)
        print_info(f"Loaded config from {config_path}")
    else:
        print_warning(f"Config not found at {config_path}; using built-in defaults")

    section = get_acceptability_config(config)
    model_config = build_model_config(section)

    table = create_summary_table(
        {
            "model_name": model_config.model_name,
            "encoder_mode": model_config.encoder_mode,
            "learning_rate": str(model_config.learning_rate),
            "batch_size": str(model_config.batch_size),
            "num_epochs": str(model_config.num_epochs),
            "device": model_config.device,
            "mixed_effects.mode": model_config.mixed_effects.mode,
        }
    )
    console.print(table)
    print_success("Self-test passed: model config built without loading weights")
    return 0


def main(
    config_path: Path = Path("config.yaml"),
    item_limit: int | None = None,
    *,
    self_test: bool = False,
    yes: bool = False,
) -> None:
    """Train the initial acceptability model on derived 2AFC pairs.

    Parameters
    ----------
    config_path
        Path to the gallery configuration file.
    item_limit
        Optional cap on the number of training items to load (for quick testing).
    self_test
        When True, validate config wiring and exit without loading model weights.
    yes
        Skip overwrite confirmation prompts for non-interactive use.
    """
    base_dir = Path(__file__).parent
    if not config_path.exists():
        config_path = base_dir / config_path.name

    if self_test:
        sys.exit(run_self_test(config_path))

    if not config_path.exists():
        print_error(f"Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    section = get_acceptability_config(config)

    paths_section = section.get("paths")
    paths_section = paths_section if isinstance(paths_section, dict) else {}
    training_items_path = base_dir / str(
        paths_section.get("training_items", DEFAULT_TRAINING_ITEMS)
    )
    dev_items_path = base_dir / str(
        paths_section.get("dev_items", "items/megaacceptability_2afc_dev.jsonl")
    )
    checkpoint_dir = base_dir / str(
        section.get("checkpoint_dir", DEFAULT_CHECKPOINT_DIR)
    )

    print_header("Acceptability Model Training")
    console.print(f"Training items: [cyan]{training_items_path}[/cyan]")
    console.print(f"Checkpoint dir: [cyan]{checkpoint_dir}[/cyan]\n")

    if not training_items_path.exists():
        print_error(
            f"Training items not found at {training_items_path}. "
            "Run prepare_megaacceptability.py first."
        )
        sys.exit(1)

    if checkpoint_dir.exists() and not yes:
        if not confirm(f"Overwrite checkpoint at {checkpoint_dir}?", default=False):
            print_info("Operation cancelled.")
            return

    # 1. Load derived training items
    print_header("1/3 Loading Training Items")
    try:
        items = load_training_items(training_items_path, limit=item_limit)
        labels, participant_ids = extract_labels_and_participants(items)
    except Exception as exc:  # noqa: BLE001
        print_error(f"Failed to load training items: {exc}")
        sys.exit(1)
    if not items:
        print_error("No training items loaded. Exiting.")
        sys.exit(1)
    n_participants = len(set(participant_ids))
    print_success(f"Loaded {len(items):,} items from {n_participants:,} annotators")

    # Load the held-out dev set (unseen sentences) for early stopping, if present
    dev_items: list[Item] = []
    dev_labels: list[str] = []
    if dev_items_path.exists():
        dev_items = load_training_items(dev_items_path, limit=item_limit)
        dev_labels, _ = extract_labels_and_participants(dev_items)
        print_success(f"Loaded {len(dev_items):,} dev items from {dev_items_path.name}")
    else:
        print_warning(
            "No dev set found; training without early stopping. "
            "Run prepare_megaacceptability.py to create one."
        )
    console.print()

    # 2. Build config and train (heavy imports deferred to here)
    print_header("2/3 Training Forced-Choice Model")
    model_config = build_model_config(section)
    console.print(
        f"Model: [cyan]{model_config.model_name}[/cyan] | "
        f"mode: [cyan]{model_config.mixed_effects.mode}[/cyan] | "
        f"epochs: [cyan]{model_config.num_epochs}[/cyan]\n"
    )
    try:
        from bead.active_learning.models.forced_choice import (  # noqa: PLC0415
            ForcedChoiceModel,
        )

        model = ForcedChoiceModel(model_config)
        with console.status("[bold]Training (this can be slow on CPU)...[/bold]"):
            metrics = model.train(
                items,
                labels,
                participant_ids=participant_ids,
                validation_items=dev_items or None,
                validation_labels=dev_labels or None,
            )
    except Exception as exc:  # noqa: BLE001
        print_error(f"Training failed: {exc}")
        sys.exit(1)
    print_success("Training complete\n")

    # 3. Save the checkpoint
    print_header("3/3 Saving Checkpoint")
    try:
        checkpoint_dir.parent.mkdir(parents=True, exist_ok=True)
        model.save(str(checkpoint_dir))
        print_success(f"Saved model to {checkpoint_dir}\n")
    except Exception as exc:  # noqa: BLE001
        print_error(f"Failed to save checkpoint: {exc}")
        sys.exit(1)

    # Summary
    print_header("Summary")
    summary: dict[str, str] = {
        "Training items": f"{len(items):,}",
        "Annotators": f"{n_participants:,}",
        "Checkpoint": str(checkpoint_dir),
    }
    for name, value in metrics.items():
        summary[name] = f"{value:.4f}" if isinstance(value, float) else str(value)
    console.print(create_summary_table(summary))
    print_info("Next: use this checkpoint to seed the active-learning loop")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Train the initial acceptability model on derived 2AFC pairs"
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit number of training items to load (default: all)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Validate config wiring without loading model weights, then exit",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompts (for non-interactive use)",
    )
    args = parser.parse_args()

    main(
        config_path=args.config,
        item_limit=args.limit,
        self_test=args.self_test,
        yes=args.yes,
    )
