#!/usr/bin/env python3
"""Generate the jsPsych/JATOS deployment from the experiment lists.

Builds one jsPsych experiment per selected list in two versions: a standalone
local build for testing in a browser, and a JATOS build packaged as a ``.jzip``
archive for upload. Text shown to participants comes from the config.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from random import Random
from uuid import UUID

import yaml

from bead.cli.display import (
    create_progress,
    print_header,
    print_info,
    print_success,
)
from bead.deployment.distribution import (
    DistributionStrategyType,
    ListDistributionStrategy,
)
from bead.deployment.jatos.exporter import JATOSExporter
from bead.deployment.jspsych.config import (
    ChoiceConfig,
    ExperimentConfig,
    InstructionsConfig,
)
from bead.deployment.jspsych.generator import JsPsychExperimentGenerator
from bead.items.item import Item
from bead.items.item_template import ItemTemplate, PresentationSpec, TaskSpec
from bead.lists.experiment_list import ExperimentList

BASE_DIR = Path(__file__).parent
INSTRUCTIONS_PLUGIN = (
    '<script src="https://unpkg.com/@jspsych/plugin-instructions@2.0.0"></script>'
)


def load_config(path: Path) -> dict:
    """Load the YAML configuration file.

    Parameters
    ----------
    path : Path
        Path to the configuration file.

    Returns
    -------
    dict
        Parsed configuration.
    """
    with path.open(encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_lists(path: Path) -> list[ExperimentList]:
    """Load the experiment lists from JSONL.

    Parameters
    ----------
    path : Path
        Path to the lists file.

    Returns
    -------
    list[ExperimentList]
        The loaded lists.
    """
    with path.open(encoding="utf-8") as f:
        return [ExperimentList.model_validate_json(line) for line in f if line.strip()]


def load_pairs(path: Path) -> dict[UUID, Item]:
    """Load the 2AFC pairs, keyed by the id the lists reference.

    Parameters
    ----------
    path : Path
        Path to the pairs file.

    Returns
    -------
    dict[UUID, Item]
        Pairs indexed by their id.
    """
    pairs: dict[UUID, Item] = {}
    with path.open(encoding="utf-8") as f:
        for line in f:
            if line.strip():
                item = Item.model_validate_json(line)
                pairs[item.id] = item
    return pairs


def build_item_template(prompt: str) -> ItemTemplate:
    """Build the template the generator renders each pair through.

    Parameters
    ----------
    prompt : str
        Question shown above the two options.

    Returns
    -------
    ItemTemplate
        Template for the forced-choice items.
    """
    return ItemTemplate(
        name="2afc_case_contrast",
        description="Two-alternative forced choice over object case",
        judgment_type="acceptability",
        task_type="forced_choice",
        task_spec=TaskSpec(prompt=prompt, options=["Речення 1", "Речення 2"]),
        presentation_spec=PresentationSpec(mode="static"),
    )


def _inject_instructions_plugin(html_path: Path) -> None:
    """Add the jsPsych instructions plugin tag when the generator omits it.

    Parameters
    ----------
    html_path : Path
        Generated ``index.html`` to patch.
    """
    html = html_path.read_text(encoding="utf-8")
    if "plugin-instructions" in html:
        return
    preload = '<script src="https://unpkg.com/@jspsych/plugin-preload'
    html = html.replace(preload, f"{INSTRUCTIONS_PLUGIN}\n    {preload}")
    html_path.write_text(html, encoding="utf-8")


def main(
    config_path: Path, n_lists: int | None = None, no_jatos: bool = False
) -> None:
    """Generate the deployment builds and package them for JATOS.

    Parameters
    ----------
    config_path : Path
        Path to the configuration file.
    n_lists : int | None
        Override how many lists to deploy.
    no_jatos : bool
        Skip the JATOS packaging step.
    """
    config = load_config(config_path)
    deployment = config["deployment"]
    experiment = deployment["experiment"]
    jspsych = deployment["jspsych"]

    print_header("Deployment")

    lists = load_lists(BASE_DIR / config["paths"]["experiment_lists"])
    pairs = load_pairs(BASE_DIR / config["paths"]["2afc_pairs"])
    print_success(f"Loaded {len(lists)} lists and {len(pairs):,} pairs")

    wanted = n_lists if n_lists is not None else deployment["n_lists_to_deploy"]
    selected = Random(deployment["random_seed"]).sample(lists, min(wanted, len(lists)))
    print_info(f"Deploying {len(selected)} of {len(lists)} lists")

    template = build_item_template(experiment["prompt"])
    templates = {template.id: template}
    pairs = {
        pair_id: pair.with_(item_template_id=template.id)
        for pair_id, pair in pairs.items()
    }

    strategy = ListDistributionStrategy(
        strategy_type=DistributionStrategyType(
            deployment.get("distribution_strategy", {}).get("strategy_type", "balanced")
        )
    )
    settings = {
        "experiment_type": "forced_choice",
        "title": experiment["title"],
        "description": experiment["description"],
        "instructions": InstructionsConfig.from_text(experiment["instructions"]),
        "randomize_trial_order": jspsych["randomize_order"],
        "show_progress_bar": True,
        "distribution_strategy": strategy,
    }
    choices = ChoiceConfig(
        randomize_choice_order=jspsych["randomize_choices"], required=True
    )

    output_dir = BASE_DIR / deployment["output_dir"]
    for version, use_jatos in (("local", False), ("jatos", True)):
        experiment_config = ExperimentConfig(**settings, use_jatos=use_jatos)
        with create_progress() as progress:
            task = progress.add_task(f"Building {version}", total=len(selected))
            for index, exp_list in enumerate(selected, start=1):
                list_dir = output_dir / version / f"list_{index:02d}"
                # Ship only the pairs this list references, not the whole pool.
                used = {
                    ref: pairs[ref] for ref in exp_list.item_refs if ref in pairs
                }
                JsPsychExperimentGenerator(
                    config=experiment_config,
                    output_dir=list_dir,
                    choice_config=choices,
                ).generate(lists=[exp_list], items=used, templates=templates)
                _inject_instructions_plugin(list_dir / "index.html")
                progress.advance(task)
        print_success(f"Built {len(selected)} {version} experiments")

    if no_jatos:
        print_info("Skipping JATOS packaging")
        return

    exporter = JATOSExporter(
        study_title=experiment["title"],
        study_description=experiment["description"],
    )
    jatos_dir = output_dir / "jatos"
    with create_progress() as progress:
        task = progress.add_task("Packaging", total=len(selected))
        for index in range(1, len(selected) + 1):
            exporter.export(
                experiment_dir=jatos_dir / f"list_{index:02d}",
                output_path=jatos_dir / f"list_{index:02d}.jzip",
                component_title=f"List {index}",
            )
            progress.advance(task)
    print_success(f"Packaged {len(selected)} JATOS archives in {jatos_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the jsPsych/JATOS deployment from the experiment lists."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=BASE_DIR / "config.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--n-lists",
        type=int,
        default=None,
        help="Override how many lists to deploy.",
    )
    parser.add_argument(
        "--no-jatos",
        action="store_true",
        help="Skip the JATOS packaging step.",
    )
    args = parser.parse_args()
    main(config_path=args.config, n_lists=args.n_lists, no_jatos=args.no_jatos)
