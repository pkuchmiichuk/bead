#!/usr/bin/env python3
"""Generate jsPsych/JATOS deployment from experiment lists.

This script loads experiment lists and 2AFC pairs, randomly selects a subset of lists,
and generates a jsPsych experiment that can be exported to JATOS.
"""

from __future__ import annotations

import sys
from pathlib import Path
from uuid import UUID

import yaml
from rich.console import Console
from rich.progress import track
from rich.table import Table

from bead.deployment.distribution import (
    DistributionStrategyType,
    ListDistributionStrategy,
)
from bead.deployment.jatos.exporter import JATOSExporter
from bead.deployment.jspsych.config import ChoiceConfig, ExperimentConfig, InstructionsConfig
from bead.deployment.jspsych.generator import JsPsychExperimentGenerator
from bead.items.item import Item
from bead.items.item_template import (
    ItemTemplate,
    PresentationSpec,
    TaskSpec,
)
from bead.lists import ExperimentList

console = Console()

_INSTRUCTIONS_PLUGIN_TAG = (
    '<script src="https://unpkg.com/@jspsych/plugin-instructions@2.0.0"></script>'
)


def _inject_instructions_plugin(html_path: Path) -> None:
    """Insert the jsPsych instructions plugin script tag if missing."""
    html = html_path.read_text(encoding="utf-8")
    if "plugin-instructions" not in html:
        html = html.replace(
            '<script src="https://unpkg.com/@jspsych/plugin-preload',
            f'{_INSTRUCTIONS_PLUGIN_TAG}\n    <script src="https://unpkg.com/@jspsych/plugin-preload',
        )
        html_path.write_text(html, encoding="utf-8")


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_experiment_lists(lists_path: Path) -> list[ExperimentList]:
    """Load experiment lists from JSONL file."""
    with console.status(f"[bold]Loading experiment lists from {lists_path}...[/bold]"):
        lists = []
        with open(lists_path) as f:
            for line in f:
                lists.append(ExperimentList.model_validate_json(line))

    console.print(f"[green]✓[/green] Loaded {len(lists)} experiment lists")
    return lists


def load_items_by_uuid(pairs_path: Path) -> dict[UUID, Item]:
    """Load all 2AFC pairs and index by UUID."""
    with console.status(f"[bold]Loading 2AFC pairs from {pairs_path}...[/bold]"):
        items_dict = {}
        with open(pairs_path) as f:
            for line in f:
                item = Item.model_validate_json(line)
                items_dict[item.id] = item

    console.print(f"[green]✓[/green] Loaded {len(items_dict)} 2AFC pairs")
    return items_dict


def create_minimal_item_template() -> ItemTemplate:
    """Create a minimal ItemTemplate for 2AFC forced choice items.

    Since our 2AFC items are already fully rendered, we just need a minimal
    template to satisfy the deployment generator's requirements.
    """
    return ItemTemplate(
        name="2afc_forced_choice",
        description="Two-alternative forced choice item",
        judgment_type="acceptability",
        task_type="forced_choice",
        task_spec=TaskSpec(
            prompt="Which sentence sounds more natural?",
            options=["Option A", "Option B"],
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )


def main() -> None:
    """Generate jsPsych/JATOS deployment."""
    import argparse

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

    console.rule("[bold]jsPsych/JATOS Deployment Generation[/bold]")
    console.print(f"Base directory: [cyan]{base_dir}[/cyan]")
    console.print(f"Configuration: [cyan]{config_path}[/cyan]")
    console.print(f"Lists to select: [cyan]{args.n_lists}[/cyan]")
    console.print(f"Output directory: [cyan]{args.output_dir}[/cyan]\n")

    # Load configuration
    console.rule("[1/6] Loading Configuration")
    config = load_config(config_path)
    deployment_config = config.get("deployment", {})
    jspsych_config = deployment_config.get("jspsych", {})
    experiment_config_dict = deployment_config.get("experiment", {})

    console.print("[green]✓[/green] Configuration loaded")
    console.print(
        f"  • Platform: [cyan]{deployment_config.get('platform', 'jatos')}[/cyan]"
    )
    console.print("  • Experiment type: [cyan]forced_choice (2AFC)[/cyan]\n")

    # Load experiment lists
    console.rule("[2/6] Loading Experiment Lists")
    lists_path = base_dir / config["paths"]["experiment_lists"]
    all_lists = load_experiment_lists(lists_path)

    # Randomly select subset of lists
    import random

    random.seed(42)
    n_lists = min(args.n_lists, len(all_lists))
    selected_lists = random.sample(all_lists, n_lists)
    console.print(
        f"[green]✓[/green] Randomly selected {n_lists} lists from {len(all_lists)} available\n"
    )

    # Load all 2AFC pairs
    console.rule("[3/6] Loading 2AFC Pairs")
    pairs_path = base_dir / config["paths"]["2afc_pairs"]
    items_dict = load_items_by_uuid(pairs_path)

    # Create minimal template
    console.print()
    console.rule("[4/6] Creating Item Template")
    template = create_minimal_item_template()
    templates_dict = {template.id: template}
    console.print("[green]✓[/green] Created minimal ItemTemplate for 2AFC items\n")

    # Update all items to reference this template (immutable update)
    items_dict = {
        uid: item.with_(item_template_id=template.id)
        for uid, item in items_dict.items()
    }

    # Create ExperimentConfig for jsPsych (base configuration)
    console.rule("[5/6] Generating jsPsych Experiments")

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
        "title": experiment_config_dict.get("title", "Sentence Acceptability Judgments"),
        "description": experiment_config_dict.get(
            "description", "Rate which sentence sounds more natural"
        ),
        "instructions": InstructionsConfig.from_text(
            experiment_config_dict.get(
                "instructions",
                "You will see pairs of sentences. Please select the sentence that sounds more natural to you.",
            )
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
        ("local", False),   # Standalone version for local testing
        ("jatos", True),    # JATOS version for deployment
    ]

    for version_name, use_jatos in versions:
        console.print(f"\n[bold]Generating {version_name} version (use_jatos={use_jatos})[/bold]")

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
                _inject_instructions_plugin(list_output_dir / "index.html")
            except Exception as e:
                console.print(f"[red]✗[/red] Error generating {version_name} list {i + 1}: {e}")
                import traceback

                traceback.print_exc()
                sys.exit(1)

        console.print(f"[green]✓[/green] Generated {n_lists} {version_name} experiments")

    console.print()

    # Export to JATOS if requested
    if not args.no_jatos:
        console.rule("[6/6] Exporting to JATOS")

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
                console.print(f"[red]✗[/red] Error exporting list {i + 1}: {e}")
                import traceback

                traceback.print_exc()

        console.print(f"[green]✓[/green] Exported {n_lists} JATOS packages\n")
    else:
        console.rule("[6/6] Skipping JATOS Export")
        console.print("[dim]  Use --no-jatos flag to disable JATOS export[/dim]\n")

    # Print summary table
    console.rule("[bold]Deployment Summary[/bold]")
    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("Lists generated:", f"[cyan]{n_lists}[/cyan]")
    table.add_row("Versions:", f"[cyan]local (standalone) + jatos (deployment)[/cyan]")
    table.add_row("Output directory:", f"[cyan]{base_dir / args.output_dir}[/cyan]")
    table.add_row(
        "Total items deployed:",
        f"[cyan]{sum(len(lst.item_refs) for lst in selected_lists)}[/cyan]",
    )
    table.add_row(
        "Items per list:",
        f"[cyan]{[len(lst.item_refs) for lst in selected_lists]}[/cyan]",
    )

    if not args.no_jatos:
        table.add_row("", "")
        table.add_row("JATOS packages:", f"[cyan]{base_dir / args.output_dir / 'jatos'}/*.jzip[/cyan]")

    console.print(table)

    console.print(
        "\n[dim]Local version: Open deployment/local/list_XX/index.html in browser[/dim]"
    )
    if not args.no_jatos:
        console.print(
            "[dim]JATOS version: Upload .jzip files to your JATOS server[/dim]"
        )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]⚠️  Interrupted by user[/yellow]")
        sys.exit(130)
    except Exception as e:
        console.print(f"\n[red]✗ Unexpected error: {e}[/red]")
        import traceback

        traceback.print_exc()
        sys.exit(1)
