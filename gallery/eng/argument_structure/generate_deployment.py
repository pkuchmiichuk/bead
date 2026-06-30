#!/usr/bin/env python3
"""Generate jsPsych/JATOS deployment from experiment lists.

This script loads experiment lists and 2AFC pairs, randomly selects a subset of lists,
and generates a jsPsych experiment that can be exported to JATOS.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import traceback
from pathlib import Path
from uuid import UUID

import yaml
from rich.progress import track

from bead.cli.display import (
    console,
    create_summary_table,
    print_error,
    print_header,
    print_info,
    print_success,
    print_warning,
)
from bead.deployment.distribution import (
    DistributionStrategyType,
    ListDistributionStrategy,
)
from bead.deployment.jatos.exporter import JATOSExporter
from bead.deployment.jspsych.config import ChoiceConfig, ExperimentConfig
from bead.deployment.jspsych.generator import JsPsychExperimentGenerator
from bead.items.item import Item
from bead.items.item_template import ItemTemplate
from bead.lists import ExperimentList, ListCollection
from bead.protocol.items import family_to_item_template

from protocol import acceptability_family, build_protocol


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_experiment_lists(lists_path: Path) -> list[ExperimentList]:
    """Load experiment lists from JSONL file."""
    with console.status(f"[bold]Loading experiment lists from {lists_path}...[/bold]"):
        collection = ListCollection.from_jsonl(lists_path)

    print_success(f"Loaded {len(collection.lists)} experiment lists")
    return collection.lists


def load_items_by_uuid(pairs_path: Path) -> dict[UUID, Item]:
    """Load all 2AFC pairs and index by UUID."""
    with console.status(f"[bold]Loading 2AFC pairs from {pairs_path}...[/bold]"):
        items_dict = {}
        with open(pairs_path) as f:
            for line in f:
                data = json.loads(line)
                item = Item(**data)
                items_dict[item.id] = item

    print_success(f"Loaded {len(items_dict)} 2AFC pairs")
    return items_dict


def create_minimal_item_template(config_path: Path) -> ItemTemplate:
    """Build the 2AFC ItemTemplate from the protocol declared in config.yaml.

    The template's prompt, response options, and judgment type are pulled
    from the ``protocol.families[].anchor`` block via the canonical
    :func:`bead.protocol.items.family_to_item_template` bridge.
    """
    family = acceptability_family(build_protocol(config_path))
    return family_to_item_template(family, judgment_type="acceptability")


def main() -> None:
    """Generate jsPsych/JATOS deployment."""
    parser = argparse.ArgumentParser(
        description="Generate jsPsych/JATOS deployment from experiment lists"
    )
    parser.add_argument(
        "--n-lists",
        type=int,
        default=20,
        help="Number of lists to randomly select (default: 20)",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="deployment",
        help="Output directory for deployment files (default: deployment)",
    )
    parser.add_argument(
        "--no-jatos",
        action="store_true",
        help="Skip JATOS .jzip export",
    )
    args = parser.parse_args()

    # Determine base directory
    base_dir = Path(__file__).parent
    config_path = base_dir / "config.yaml"

    print_header("jsPsych/JATOS Deployment Generation")
    console.print(f"Base directory: [cyan]{base_dir}[/cyan]")
    console.print(f"Configuration: [cyan]{config_path}[/cyan]")
    console.print(f"Lists to select: [cyan]{args.n_lists}[/cyan]")
    console.print(f"Output directory: [cyan]{args.output_dir}[/cyan]\n")

    # Load configuration
    print_header("[1/6] Loading Configuration")
    config = load_config(config_path)
    deployment_config = config.get("deployment", {})
    jspsych_config = deployment_config.get("jspsych", {})
    experiment_config_dict = deployment_config.get("experiment", {})

    print_success("Configuration loaded")
    console.print(
        f"  - Platform: [cyan]{deployment_config.get('platform', 'jatos')}[/cyan]"
    )
    console.print("  - Experiment type: [cyan]forced_choice (2AFC)[/cyan]\n")

    # Load experiment lists
    print_header("[2/6] Loading Experiment Lists")
    lists_path = base_dir / config["paths"]["experiment_lists"]
    all_lists = load_experiment_lists(lists_path)

    # Randomly select subset of lists
    random.seed(42)
    n_lists = min(args.n_lists, len(all_lists))
    selected_lists = random.sample(all_lists, n_lists)
    print_success(
        f"Randomly selected {n_lists} lists from {len(all_lists)} available\n"
    )

    # Load all 2AFC pairs
    print_header("[3/6] Loading 2AFC Pairs")
    pairs_path = base_dir / config["paths"]["2afc_pairs"]
    items_dict = load_items_by_uuid(pairs_path)

    # Create minimal template
    console.print()
    print_header("[4/6] Creating Item Template")
    template = create_minimal_item_template(config_path)
    templates_dict = {template.id: template}
    print_success("Created minimal ItemTemplate for 2AFC items\n")

    # Update all items to reference this template
    for item in items_dict.values():
        item.item_template_id = template.id

    # Create ExperimentConfig for jsPsych (base configuration)
    print_header("[5/6] Generating jsPsych Experiments")

    # Extract distribution strategy from config
    dist_config_dict = deployment_config.get("distribution_strategy", {})
    strategy_type = dist_config_dict.get("strategy_type", "balanced")
    distribution_strategy = ListDistributionStrategy(
        strategy_type=DistributionStrategyType(strategy_type),
        max_participants=dist_config_dict.get("max_participants"),
        error_on_exhaustion=dist_config_dict.get("error_on_exhaustion", True),
        debug_mode=dist_config_dict.get("debug_mode", False),
        debug_list_index=dist_config_dict.get("debug_list_index", 0),
    )

    base_config_dict = {
        "experiment_type": "forced_choice",
        "title": experiment_config_dict.get(
            "title", "Sentence Acceptability Judgments"
        ),
        "description": experiment_config_dict.get(
            "description", "Rate which sentence sounds more natural"
        ),
        "instructions": experiment_config_dict.get(
            "instructions",
            "You will see pairs of sentences. "
            "Please select the sentence that sounds more natural to you.",
        ),
        "randomize_trial_order": jspsych_config.get("randomize_order", True),
        "show_progress_bar": True,
        "distribution_strategy": distribution_strategy,
    }

    choice_config = ChoiceConfig(
        randomize_choice_order=jspsych_config.get("randomize_choices", True),
        required=True,
    )

    # Generate TWO versions: local (standalone) and jatos (deployment)
    versions = [
        ("local", False),  # Standalone version for local testing
        ("jatos", True),  # JATOS version for deployment
    ]

    for version_name, use_jatos in versions:
        console.print(
            f"\n[bold]Generating {version_name} version (use_jatos={use_jatos})[/bold]"
        )

        # Create version-specific config
        experiment_config = ExperimentConfig(
            **base_config_dict,
            use_jatos=use_jatos,
        )

        # Create version-specific output directory
        output_dir = base_dir / args.output_dir / version_name
        output_dir.mkdir(parents=True, exist_ok=True)

        for i, exp_list in track(
            enumerate(selected_lists),
            description=f"Generating {version_name} experiments",
            total=len(selected_lists),
        ):
            list_output_dir = output_dir / f"list_{i + 1:02d}"

            generator = JsPsychExperimentGenerator(
                config=experiment_config,
                output_dir=list_output_dir,
                choice_config=choice_config,
            )

            try:
                generator.generate(
                    lists=[exp_list],
                    items=items_dict,
                    templates=templates_dict,
                )
            except Exception as e:
                print_error(f"Error generating {version_name} list {i + 1}: {e}")
                traceback.print_exc()
                sys.exit(1)

        print_success(f"Generated {n_lists} {version_name} experiments")

    console.print()

    # Export to JATOS if requested
    if not args.no_jatos:
        print_header("[6/6] Exporting to JATOS")

        exporter = JATOSExporter(
            study_title=config["project"]["name"],
            study_description=config["project"].get(
                "description", "Argument structure acceptability study"
            ),
        )

        jatos_dir = base_dir / args.output_dir / "jatos"

        for i, _ in track(
            enumerate(selected_lists),
            description="Exporting to JATOS",
            total=len(selected_lists),
        ):
            list_dir = jatos_dir / f"list_{i + 1:02d}"
            jzip_path = jatos_dir / f"list_{i + 1:02d}.jzip"

            try:
                exporter.export(
                    experiment_dir=list_dir,
                    output_path=jzip_path,
                    component_title=f"List {i + 1}",
                )
            except Exception as e:
                print_error(f"Error exporting list {i + 1}: {e}")
                traceback.print_exc()

        print_success(f"Exported {n_lists} JATOS packages\n")
    else:
        print_header("[6/6] Skipping JATOS Export")
        print_info("Use --no-jatos flag to disable JATOS export\n")

    # Print summary table
    print_header("Deployment Summary")
    summary_data = {
        "Lists generated": str(n_lists),
        "Versions": "local (standalone) + jatos (deployment)",
        "Output directory": str(base_dir / args.output_dir),
        "Total items deployed": str(sum(len(lst.item_refs) for lst in selected_lists)),
        "Items per list": str([len(lst.item_refs) for lst in selected_lists]),
    }

    if not args.no_jatos:
        jatos_path = base_dir / args.output_dir / "jatos"
        summary_data["JATOS packages"] = f"{jatos_path}/*.jzip"

    table = create_summary_table(summary_data)
    console.print(table)

    print_info("Local version: Open deployment/local/list_XX/index.html in browser")
    if not args.no_jatos:
        print_info("JATOS version: Upload .jzip files to your JATOS server")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print_warning("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        print_error(f"Unexpected error: {e}")
        traceback.print_exc()
        sys.exit(1)
