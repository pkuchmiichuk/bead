#!/usr/bin/env python3
"""Run the argument-structure active-learning pipeline.

Seeds the active-learning loop with the acceptability model pretrained on
MegaAcceptability (see train_acceptability_model.py) and fine-tunes it on
collected human 2AFC judgments until the model converges to human-level
inter-annotator agreement. Pairs are loaded from the canonical layers fragment
produced by create_2afc_pairs.py.

The active-learning loop trains transformer models, so a real run needs torch,
transformers, and a trained checkpoint. Use --dry-run to validate the wiring
(item template, loop config, selector, checkpoint presence) without training.
"""

from __future__ import annotations

import argparse
import json
import sys
import traceback
from pathlib import Path
from typing import Any

import layers_io
import yaml
from protocol import acceptability_family, build_protocol

from bead.active_learning.loop import ActiveLearningLoop
from bead.active_learning.selection import UncertaintySampler
from bead.cli.display import (
    console,
    create_summary_table,
    print_error,
    print_header,
    print_info,
    print_success,
    print_warning,
)
from bead.config.active_learning import (
    ActiveLearningLoopConfig,
    UncertaintySamplerConfig,
)
from bead.evaluation.convergence import ConvergenceDetector
from bead.items.item import Item
from bead.items.item_template import ItemTemplate
from bead.protocol.items import family_to_item_template


def load_config(config_path: Path) -> dict[str, Any]:
    """Load configuration from a YAML file."""
    print_info(f"Loading configuration from {config_path}")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    print_success("Configuration loaded")
    return config


def load_pairs(config: dict[str, Any], base_dir: Path) -> list[Item]:
    """Load 2AFC pairs from the canonical layers fragment, else the JSONL.

    Parameters
    ----------
    config : dict[str, Any]
        Loaded configuration.
    base_dir : Path
        Directory the config paths are relative to.

    Returns
    -------
    list[Item]
        The 2AFC pairs.
    """
    fragment_rel = config["paths"].get("2afc_pairs_fragment")
    fragment_path = base_dir / fragment_rel if fragment_rel else None
    if fragment_path is not None and fragment_path.exists():
        print_info(f"Loading pairs from layers fragment {fragment_path.name}")
        return layers_io.read_items(fragment_path)

    pairs_path = base_dir / config["paths"]["2afc_pairs"]
    print_info(f"Loading pairs from {pairs_path.name}")
    items: list[Item] = []
    with open(pairs_path) as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(Item.model_validate_json(line))
    return items


def build_item_template(config_path: Path) -> ItemTemplate:
    """Build the 2AFC ItemTemplate that the forced-choice model validates against.

    The protocol declares per-item (FORCED_CHOICE) options, so its task spec
    leaves ``options`` unset. The active-learning model, however, validates each
    item against the template's ``task_spec.options`` and reads those option
    names out of ``rendered_elements``. The 2AFC pairs use ``option_a`` /
    ``option_b``, so we set those on the model-facing template.
    """
    family = acceptability_family(build_protocol(config_path))
    template = family_to_item_template(family, judgment_type="acceptability")
    return template.with_(
        task_spec=template.task_spec.with_(options=("option_a", "option_b"))
    )


def build_loop_config(config: dict[str, Any]) -> ActiveLearningLoopConfig:
    """Build the active-learning loop config from the configuration."""
    al = config["active_learning"]
    conv = config["training"]["convergence"]
    return ActiveLearningLoopConfig(
        max_iterations=al.get("max_iterations", 10),
        budget_per_iteration=al.get("budget_per_iteration", 100),
        stopping_criterion=al.get("stopping_criterion", "convergence"),
        metric_name=conv.get("metric", "accuracy"),
    )


def load_acceptability_model(checkpoint_dir: Path, device: str) -> Any:
    """Load the pretrained acceptability ForcedChoiceModel from a checkpoint.

    Imports torch-backed modules lazily so the rest of the pipeline (and
    --dry-run) stays light.

    Parameters
    ----------
    checkpoint_dir : Path
        Directory written by train_acceptability_model.py.
    device : str
        Device to load the model onto.

    Returns
    -------
    ForcedChoiceModel
        The pretrained model, ready to seed the active-learning loop.
    """
    from bead.active_learning.config import MixedEffectsConfig  # noqa: PLC0415
    from bead.active_learning.models.forced_choice import (  # noqa: PLC0415
        ForcedChoiceModel,
    )
    from bead.config.active_learning import ForcedChoiceModelConfig  # noqa: PLC0415

    model = ForcedChoiceModel(ForcedChoiceModelConfig(device=device))  # type: ignore[arg-type]
    model.load(str(checkpoint_dir))
    # The model was pretrained with per-annotator random intercepts. The loop
    # fine-tunes at the population level (it trains without participant ids), so
    # switch the seeded model to fixed effects; the pretrained encoder carries
    # over.
    model.config = model.config.with_(mixed_effects=MixedEffectsConfig(mode="fixed"))
    return model


def load_human_ratings(path: Path) -> dict[str, str] | None:
    """Load human 2AFC ratings mapping item id to the chosen option name.

    Parameters
    ----------
    path : Path
        JSONL file of ``{"item_id": ..., "option": "option_a"|"option_b"}``.

    Returns
    -------
    dict[str, str] | None
        Mapping from item id to chosen option, or None when absent.
    """
    if not path.exists():
        console.print("  - Human ratings: not available (collected during deployment)")
        return None
    ratings: dict[str, str] = {}
    with open(path) as f:
        for line in f:
            record = json.loads(line)
            ratings[str(record["item_id"])] = str(record["option"])
    print_success(f"Loaded {len(ratings)} human ratings")
    return ratings


def print_results(results: list[Any]) -> None:
    """Print per-iteration active-learning results."""
    print_header("Active Learning Results")
    if not results:
        print_info("No training iterations completed.")
        return
    console.print(f"\nCompleted {len(results)} iterations")
    for i, metadata in enumerate(results, 1):
        metrics = metadata.metrics
        acc = metrics.get("accuracy", 0.0)
        console.print(f"  Iteration {i}: accuracy={acc:.4f}")


def main(args: argparse.Namespace) -> None:
    """Run the active-learning pipeline."""
    print_header("Argument Structure Active Learning Pipeline")

    if args.config:
        config_path = Path(args.config)
        base_dir = config_path.parent
    else:
        base_dir = Path(__file__).parent
        config_path = base_dir / "config.yaml"

    print_header("[1/6] Loading Configuration")
    config = load_config(config_path)

    print_header("[2/6] Building Item Template + Loop Config")
    item_template = build_item_template(config_path)
    loop_config = build_loop_config(config)
    sampler_config = UncertaintySamplerConfig(
        method=config["active_learning"].get("method", "entropy")
    )
    selector = UncertaintySampler(config=sampler_config)
    loop = ActiveLearningLoop(item_selector=selector, config=loop_config)
    print_success(
        f"Loop ready: {loop_config.max_iterations} iterations, "
        f"stop on {loop_config.stopping_criterion}"
    )

    print_header("[3/6] Loading 2AFC Pairs")
    pairs = load_pairs(config, base_dir)
    initial_size = args.initial_size or config["active_learning"].get(
        "initial_training_size", 100
    )
    initial_items = pairs[:initial_size]
    unlabeled_pool = pairs[initial_size:]
    print_success(
        f"{len(initial_items)} initial items, {len(unlabeled_pool)} in the pool"
    )

    print_header("[4/6] Seeding Pretrained Acceptability Model")
    checkpoint_dir = base_dir / config["acceptability_model"]["checkpoint_dir"]
    if not checkpoint_dir.exists():
        print_warning(
            f"No acceptability checkpoint at {checkpoint_dir}. "
            "Run prepare_megaacceptability.py then train_acceptability_model.py."
        )
    else:
        print_success(f"Found pretrained checkpoint at {checkpoint_dir}")

    print_header("[5/6] Convergence Detection")
    conv = config["training"]["convergence"]
    convergence_detector = ConvergenceDetector(
        human_agreement_metric=conv["metric"],
        convergence_threshold=conv["threshold"],
        min_iterations=conv["min_iterations"],
        alpha=conv.get("alpha", 0.05),
    )
    human_ratings = load_human_ratings(base_dir / "data" / "human_ratings.jsonl")

    print_header("[6/6] Running Active Learning Loop")
    if args.dry_run:
        print_warning("DRY RUN: validated wiring; not training.")
        print_success("Wiring OK (template, loop, selector, convergence detector).")
        return

    if not checkpoint_dir.exists():
        print_error("Cannot run without a pretrained checkpoint. See step [4/6].")
        sys.exit(1)

    device = config["acceptability_model"].get("device", "cpu")
    initial_model = load_acceptability_model(checkpoint_dir, device)

    try:
        results = loop.run(
            initial_items=initial_items,
            initial_model=initial_model,
            item_template=item_template,
            unlabeled_pool=unlabeled_pool,
            human_ratings=human_ratings,
            convergence_detector=convergence_detector,
        )
    except Exception as e:
        print_error(f"Error during active learning: {e}")
        traceback.print_exc()
        sys.exit(1)

    print_results(results)
    table = create_summary_table(
        {
            "Iterations": str(len(results)),
            "Initial items": str(len(initial_items)),
            "Pool size": str(len(unlabeled_pool)),
        }
    )
    console.print(table)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run the argument structure active learning pipeline"
    )
    parser.add_argument(
        "--config",
        type=str,
        help="Path to config.yaml (default: alongside this script)",
    )
    parser.add_argument(
        "--initial-size", type=int, help="Size of the initial training set"
    )
    parser.add_argument(
        "--dry-run", action="store_true", help="Validate wiring without training"
    )
    args = parser.parse_args()

    try:
        main(args)
    except KeyboardInterrupt:
        print_warning("Pipeline interrupted by user")
        sys.exit(130)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
