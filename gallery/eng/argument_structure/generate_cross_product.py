#!/usr/bin/env python3
"""Generate cross-product of all verbs × all generic templates.

This script creates the foundational item set for the argument structure
experiment by testing every VerbNet verb in every generic frame structure.

Output: items/cross_product_items.jsonl
"""

import argparse
import json
import sys
from pathlib import Path

import layers_io

from bead.cli.display import (
    confirm,
    console,
    create_progress,
    create_summary_table,
    print_error,
    print_header,
    print_info,
    print_success,
    print_warning,
)
from bead.items.item import Item
from bead.resources.lexicon import Lexicon


def main(
    templates_file: str = "templates/generic_frames.jsonl",
    verbs_file: str = "lexicons/verbnet_verbs.jsonl",
    output_limit: int | None = None,
    *,
    yes: bool = False,
) -> None:
    """Generate cross-product items.

    Parameters
    ----------
    templates_file : str
        Path to generic templates file.
    verbs_file : str
        Path to verb lexicon file.
    output_limit : int | None
        Limit output to first N items (for testing).
    yes : bool
        Skip confirmation prompts (for non-interactive use).
    """
    base_dir = Path(__file__).parent
    templates_path = base_dir / templates_file
    verbs_path = base_dir / verbs_file
    output_dir = base_dir / "items"
    output_dir.mkdir(exist_ok=True)
    output_path = output_dir / "cross_product_items.jsonl"

    print_header("Cross-Product Generation")
    console.print(f"Base directory: [cyan]{base_dir}[/cyan]")
    console.print(f"Templates: [cyan]{templates_path}[/cyan]")
    console.print(f"Verbs: [cyan]{verbs_path}[/cyan]")
    console.print(f"Output: [cyan]{output_path}[/cyan]\n")

    # Check for existing output
    if output_path.exists() and not yes:
        if not confirm(f"Overwrite {output_path}?", default=False):
            print_info("Operation cancelled.")
            return

    # Load generic templates
    print_header("1/3 Loading Generic Templates")
    try:
        with console.status("[bold]Loading templates...[/bold]"):
            templates = []
            with open(templates_path) as f:
                for line in f:
                    template = json.loads(line)
                    templates.append(template)

        print_success(f"Loaded {len(templates)} generic templates\n")
    except Exception as e:
        print_error(f"Failed to load templates: {e}")
        sys.exit(1)

    # Load verb lexicon
    print_header("2/3 Loading Verb Lexicon")
    try:
        with console.status("[bold]Loading verb lexicon...[/bold]"):
            verb_lexicon = Lexicon.from_jsonl(str(verbs_path), "verbnet_verbs")

        print_success(f"Loaded {len(verb_lexicon.items)} verb forms")

        # Get unique verb lemmas (we only need base forms for cross-product)
        verb_lemmas = sorted({item.lemma for item in verb_lexicon.items.values()})
        print_success(f"Found {len(verb_lemmas)} unique verb lemmas\n")
    except Exception as e:
        print_error(f"Failed to load verb lexicon: {e}")
        sys.exit(1)

    # Generate cross-product
    print_header("3/3 Generating Cross-Product")
    total_combinations = len(verb_lemmas) * len(templates)

    if output_limit:
        print_warning(f"Test mode: Limiting output to {output_limit:,} items")
        total_combinations = min(output_limit, total_combinations)

    items_generated = 0
    items_for_layers: list[Item] = []

    try:
        with open(output_path, "w") as f:
            with create_progress() as progress:
                task = progress.add_task("Processing templates", total=len(templates))

                for template in templates:
                    template_id = template["id"]
                    template_name = template["name"]
                    template_string = template["template_string"]

                    for verb_lemma in verb_lemmas:
                        # Create Item for this verb×template combination
                        item = Item(
                            item_template_id=template_id,
                            rendered_elements={
                                "template_name": template_name,
                                "template_string": template_string,
                                "verb_lemma": verb_lemma,
                            },
                            item_metadata={
                                "verb_lemma": verb_lemma,
                                "template_id": str(template_id),
                                "template_name": template_name,
                                "template_structure": template_string,
                                "combination_type": "verb_frame_cross_product",
                            },
                        )

                        # Write to file and buffer for the layers corpus
                        f.write(item.model_dump_json() + "\n")
                        items_for_layers.append(item)
                        items_generated += 1

                        # Check limit
                        if output_limit and items_generated >= output_limit:
                            break

                    progress.advance(task)

                    if output_limit and items_generated >= output_limit:
                        break

        print_success(f"Generated {items_generated:,} cross-product items\n")

        # also persist as a canonical layers fragment and Arrow/Parquet corpus
        fragment_path = output_dir / "cross_product_items.layers.json"
        corpus_dir = output_dir / "cross_product_corpus"
        with console.status("[bold]Encoding layers fragment + corpus...[/bold]"):
            layers_io.write_items(
                items_for_layers,
                name="argument_structure_cross_product",
                fragment_path=fragment_path,
                materialize_dir=corpus_dir,
            )
        print_success(f"Wrote layers fragment to {fragment_path}\n")
    except Exception as e:
        print_error(f"Failed to generate items: {e}")
        sys.exit(1)

    # Summary
    print_header("Summary")
    table = create_summary_table(
        {
            "Verb lemmas": str(len(verb_lemmas)),
            "Generic templates": str(len(templates)),
            "Cross-product items": f"{items_generated:,}",
            "Output file": str(output_path),
        }
    )
    console.print(table)

    print_info("Next: Run create_2afc_pairs.py to generate forced-choice pairs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate cross-product of verbs × templates"
    )
    parser.add_argument(
        "--templates",
        type=str,
        default="templates/generic_frames.jsonl",
        help="Path to generic templates file",
    )
    parser.add_argument(
        "--verbs",
        type=str,
        default="lexicons/verbnet_verbs.jsonl",
        help="Path to verb lexicon file",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit output to first N items (for testing)",
    )
    parser.add_argument(
        "-y",
        "--yes",
        action="store_true",
        help="Skip confirmation prompts (for non-interactive use)",
    )
    args = parser.parse_args()

    main(
        templates_file=args.templates,
        verbs_file=args.verbs,
        output_limit=args.limit,
        yes=args.yes,
    )
