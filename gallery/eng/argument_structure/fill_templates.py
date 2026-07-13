#!/usr/bin/env python3
"""Fill templates using configuration-driven MLM strategy.

This script loads templates and lexicons, creates a TemplateFiller with
MixedFillingStrategy using slot_strategies from config.yaml, and outputs
filled templates.

All parameters are configurable via config.yaml, with optional CLI overrides.
"""

import argparse
import logging
import sys
from pathlib import Path

import layers_io
import yaml
from utils.renderers import OtherNounRenderer

from bead.cli.display import (
    confirm,
    console,
    create_progress,
    create_summary_table,
    display_dry_run_summary,
    print_error,
    print_header,
    print_info,
    print_success,
    print_warning,
)
from bead.data.serialization import write_jsonlines
from bead.resources.lexicon import Lexicon
from bead.resources.template import Template
from bead.resources.template_collection import TemplateCollection
from bead.templates.adapters.cache import ModelOutputCache
from bead.templates.adapters.huggingface import HuggingFaceMLMAdapter
from bead.templates.filler import FilledTemplate
from bead.templates.resolver import ConstraintResolver
from bead.templates.strategies import MixedFillingStrategy

logger = logging.getLogger(__name__)


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def main() -> None:
    """Fill templates using config-driven MLM strategy."""
    parser = argparse.ArgumentParser(description="Fill templates with MLM strategy")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="Path to configuration file",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Dry run: use 10 verbs with 1 simple and 1 progressive template",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output path from config",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompts (for non-interactive use)",
    )
    args = parser.parse_args()

    # load configuration
    try:
        config = load_config(args.config)
    except Exception as e:
        print_error(f"Failed to load config: {e}")
        sys.exit(1)

    # setup logging
    logging.basicConfig(
        level=getattr(logging, config["logging"]["level"]),
        format=config["logging"]["format"],
    )

    print_header("Template Filling")

    # resolve paths
    templates_path = Path(config["resources"]["templates"][0]["path"])
    output_path = args.output or Path(config["template"]["output_path"])

    console.print(f"Config: [cyan]{args.config}[/cyan]")
    console.print(f"Templates: [cyan]{templates_path}[/cyan]")
    console.print(f"Output: [cyan]{output_path}[/cyan]\n")

    # Check for existing output
    if output_path.exists() and not args.dry_run and not args.yes:
        if not confirm(f"Overwrite {output_path}?", default=False):
            print_info("Operation cancelled.")
            return

    # load templates
    print_header("Loading Templates")
    try:
        template_collection = TemplateCollection.from_jsonl(
            templates_path, "generic_frames"
        )
        num_templates = len(template_collection.templates)
        print_success(f"Loaded {num_templates} templates from {templates_path}")
    except Exception as e:
        print_error(f"Failed to load templates: {e}")
        sys.exit(1)

    # apply dry-run mode: select specific templates
    templates = list(template_collection.templates.values())
    if args.dry_run:
        print_warning("DRY RUN: Selecting 1 simple + 1 progressive template")

        # find templates with 3 noun slots
        def count_noun_slots(t: Template) -> int:
            return sum(1 for s in t.slots if s.startswith("noun_"))

        templates_with_3_nouns = [t for t in templates if count_noun_slots(t) == 3]

        # separate simple and progressive
        simple_templates = [
            t for t in templates_with_3_nouns if "progressive" not in t.name
        ]
        progressive_templates = [
            t for t in templates_with_3_nouns if "progressive" in t.name
        ]

        # select one of each
        selected = []
        if simple_templates:
            selected.append(simple_templates[0])
            print_info(f"Simple template: {simple_templates[0].name}")
        if progressive_templates:
            selected.append(progressive_templates[0])
            print_info(f"Progressive template: {progressive_templates[0].name}")

        templates = selected if selected else templates[:2]
        print_success(f"Selected {len(templates)} templates for dry run")

    # load lexicons
    print_header("Loading Lexicons")
    lexicons: list[Lexicon] = []
    try:
        for lex_config in config["resources"]["lexicons"]:
            lex_path = Path(lex_config["path"])
            lexicon = Lexicon.from_jsonl(lex_path, lex_config["name"])

            # in dry-run mode, limit verb lexicon to 10 verbs
            if args.dry_run and lex_config["name"] == "verbnet_verbs":
                # get first 10 unique verb lemmas
                verb_lemmas = []
                limited_items = {}
                for item_id, item in lexicon.items.items():
                    if item.lemma not in verb_lemmas:
                        verb_lemmas.append(item.lemma)
                        if len(verb_lemmas) >= 10:
                            break
                    # keep all forms of verbs we're including
                    if item.lemma in verb_lemmas[:10]:
                        limited_items[item_id] = item

                lexicon.items = limited_items
                num_forms = len(lexicon.items)
                print_warning(f"DRY RUN: Limited to 10 verb lemmas ({num_forms} forms)")

            lexicons.append(lexicon)
            num_items = len(lexicon.items)
            print_success(f"Loaded {num_items} items from {lex_config['name']}")
    except Exception as e:
        print_error(f"Failed to load lexicons: {e}")
        sys.exit(1)

    # Show dry run summary before continuing
    if args.dry_run:
        display_dry_run_summary(
            {
                "Templates": len(templates),
                "Lexicons": len(lexicons),
                "Output": str(output_path),
            }
        )
        console.print()

    # initialize constraint resolver
    resolver = ConstraintResolver()

    # initialize model adapter for MLM
    print_header("Loading MLM Model")
    mlm_config = config["template"]["mlm"]
    print_info(f"Loading MLM model: {mlm_config['model_name']}...")
    try:
        model_adapter = HuggingFaceMLMAdapter(
            model_name=mlm_config["model_name"],
            device=mlm_config.get("device", "cpu"),
        )
        model_adapter.load_model()
        print_success("MLM model loaded successfully")
    except Exception as e:
        print_error(f"Failed to load MLM model: {e}")
        sys.exit(1)

    # initialize cache
    cache_dir = Path(config["paths"]["cache_dir"])
    cache = ModelOutputCache(cache_dir=cache_dir)

    # build slot_strategies dict for MixedFillingStrategy
    # format: {slot_name: (strategy_name, config_dict)}
    slot_strategies: dict[str, tuple[str, dict]] = {}

    for slot_name, slot_config in config["template"]["slot_strategies"].items():
        strategy_name = slot_config["strategy"]

        if strategy_name == "mlm":
            # MLM strategy needs special config with resolver, model_adapter, etc.
            mlm_slot_config = {
                "resolver": resolver,
                "model_adapter": model_adapter,
                "cache": cache,
                "beam_size": mlm_config.get("beam_size", 5),
                "top_k": mlm_config.get("top_k", 10),
            }
            # add per-slot max_fills and enforce_unique if specified
            if "max_fills" in slot_config:
                mlm_slot_config["max_fills"] = slot_config["max_fills"]
            if "enforce_unique" in slot_config:
                mlm_slot_config["enforce_unique"] = slot_config["enforce_unique"]

            slot_strategies[slot_name] = ("mlm", mlm_slot_config)
        else:
            # for other strategies (exhaustive, random, etc.)
            slot_strategies[slot_name] = (strategy_name, {})

    # create renderer for English-specific noun handling
    # uses OtherNounRenderer for "another"/"the other" patterns with repeated nouns
    renderer = OtherNounRenderer()
    print_info("Using OtherNounRenderer for English-specific noun handling")

    # create filler with MixedFillingStrategy
    print_header("Filling Templates")
    strategy = MixedFillingStrategy(
        slot_strategies=slot_strategies,
    )

    # fill templates
    filled_templates = []
    try:
        with create_progress() as progress:
            task = progress.add_task("Filling templates", total=len(templates))

            for template in templates:
                try:
                    combos = list(
                        strategy.generate_from_template(
                            template=template, lexicons=lexicons, language_code="en"
                        )
                    )

                    # convert combinations to FilledTemplate objects
                    for combo in combos:
                        # render text with English-specific noun handling
                        rendered = renderer.render(
                            template.template_string, combo, template.slots
                        )

                        slots_required = {
                            name: slot.required for name, slot in template.slots.items()
                        }
                        filled = FilledTemplate(
                            template_id=str(template.id),
                            template_name=template.name,
                            slot_fillers=combo,
                            rendered_text=rendered,
                            strategy_name="mixed",
                            template_slots=slots_required,
                        )
                        filled_templates.append(filled)

                    num_combos = len(combos)
                    logger.info(f"Generated {num_combos} for {template.name}")
                except Exception as e:
                    print_warning(f"Failed to fill template {template.name}: {e}")
                    continue

                progress.advance(task)

        print_success(f"Total filled templates: {len(filled_templates)}")
    except Exception as e:
        print_error(f"Failed to fill templates: {e}")
        sys.exit(1)

    # save filled templates
    print_header("Saving Results")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        write_jsonlines(filled_templates, output_path)
        print_success(f"Saved filled templates to {output_path}")

        # also persist as layers filling records
        layers_path = output_path.with_suffix(".layers.json")
        layers_io.write_fillings_layers(filled_templates, layers_path)
        print_success(f"Wrote layers fillings to {layers_path}")
    except Exception as e:
        print_error(f"Failed to save output: {e}")
        sys.exit(1)

    # Summary
    print_header("Summary")
    table = create_summary_table(
        {
            "Templates processed": str(len(templates)),
            "Filled templates": f"{len(filled_templates):,}",
            "Output file": str(output_path),
        }
    )
    console.print(table)

    print_info("Next: Run create_2afc_pairs.py to generate forced-choice pairs")


if __name__ == "__main__":
    main()
