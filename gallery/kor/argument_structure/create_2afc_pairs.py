#!/usr/bin/env python3
"""Generate 2AFC (two-alternative forced choice) pairs from filled templates.

This script:
1. Loads filled templates from fill_templates.py output
2. Scores filled items with language model (uses bead/items/scoring.py)
3. Creates forced-choice items (uses bead/items/forced_choice.py)
4. Assigns quantiles (uses bead/lists/stratification.py)

All parameters are configurable via config.yaml.
"""

from __future__ import annotations

import argparse
from pathlib import Path
from uuid import UUID, uuid4

import yaml
from rich.console import Console
from rich.table import Table

from bead.items.forced_choice import create_forced_choice_items_from_groups
from bead.items.item import Item
from bead.items.scoring import LanguageModelScorer
from bead.lists.stratification import assign_quantiles_by_uuid
from bead.templates.filler import FilledTemplate

console = Console()


def load_config(config_path: Path) -> dict:
    """Load configuration from YAML file."""
    with open(config_path) as f:
        return yaml.safe_load(f)


def load_filled_templates(path: str, limit: int | None = None) -> list[FilledTemplate]:
    """Load filled templates from JSONL."""
    filled_templates = []
    with open(path) as f:
        for i, line in enumerate(f):
            if limit and i >= limit:
                break
            filled_templates.append(FilledTemplate.model_validate_json(line))
    return filled_templates


def convert_filled_templates_to_items(
    filled_templates: list[FilledTemplate],
) -> list[Item]:
    """Convert FilledTemplate objects to Item objects for scoring and pairing.

    Extracts verb lemma from slot_fillers and creates Item with metadata.
    """
    items = []

    for ft in filled_templates:
        # Extract verb lemma from slot_fillers
        verb_lemma = None
        if "verb" in ft.slot_fillers:
            verb_lemma = ft.slot_fillers["verb"].lemma

        # Create Item with filled text in rendered_elements (sentence-cased)
        text = ft.rendered_text
        if text:
            text = text[0].upper() + text[1:] if len(text) > 1 else text.upper()

        item = Item(
            item_template_id=UUID(ft.template_id),
            rendered_elements={"text": text},
            item_metadata={
                "filled_template_id": str(ft.id),
                "template_id": ft.template_id,
                "template_name": ft.template_name,
                "template_structure": ft.template_name,  # Use name as structure identifier
                "verb_lemma": verb_lemma,
                "strategy": ft.strategy_name,
            },
        )
        items.append(item)

    return items


def score_filled_items_with_lm(
    items: list[Item],
    cache_dir: Path,
    model_name: str = "gpt2",
) -> dict[str, float]:
    """Score filled items with language model using bead/items/scoring.py."""
    # Use bead's LanguageModelScorer
    scorer = LanguageModelScorer(
        model_name=model_name,
        cache_dir=cache_dir,
        device="cpu",
        text_key="text",
    )

    # Create temporary items with filled text in rendered_elements
    temp_items = []
    item_id_map = {}
    for item in items:
        temp_item = Item(
            item_template_id=uuid4(),
            rendered_elements={"text": item.rendered_elements.get("text", "")},
        )
        temp_items.append(temp_item)
        item_id_map[temp_item.id] = str(item.id)

    # Score batch (progress bar shown by scorer)
    scores_list = scorer.score_batch(temp_items)

    # Map back to original item IDs
    scores = {}
    for temp_item, score in zip(temp_items, scores_list, strict=True):
        original_id = item_id_map[temp_item.id]
        scores[original_id] = score

    return scores


def create_forced_choice_pairs(
    items: list[Item],
    lm_scores: dict[str, float],
) -> list[Item]:
    """Create 2AFC items using bead/items/forced_choice.py.

    Creates two types of forced-choice items:
    1. Same-verb pairs (same verb, different frames)
    2. Different-verb pairs (different verbs, same frame)
    """
    # Create lookup dict to avoid O(n) scans for each pair
    item_lookup = {str(item.id): item for item in items}

    # Helper to extract text from items
    def extract_text(item: Item) -> str:
        return item.rendered_elements.get("text", "")

    # 1. Create same-verb pairs (group by verb_lemma)
    with console.status("[bold]Creating same-verb pairs...[/bold]"):
        same_verb_items = create_forced_choice_items_from_groups(
            items=items,
            group_by=lambda item: item.item_metadata.get("verb_lemma", "unknown"),
            n_alternatives=2,
            extract_text=extract_text,
            include_group_metadata=True,
        )

    # Add pair_type and additional metadata
    updated_same_verb_items = []
    for fc_item in same_verb_items:
        item1_id = fc_item.item_metadata.get("source_item_0_id")
        item2_id = fc_item.item_metadata.get("source_item_1_id")

        source_items = [item_lookup.get(item1_id), item_lookup.get(item2_id)]
        if all(source_items) and len(source_items) == 2:
            fc_item = fc_item.with_(item_metadata={
                **fc_item.item_metadata,
                "pair_type": "same_verb",
                "verb": source_items[0].item_metadata.get("verb_lemma"),
                "template1": source_items[0].item_metadata.get("template_structure"),
                "template2": source_items[1].item_metadata.get("template_structure"),
                "lm_score_a": lm_scores.get(str(source_items[0].id), float("-inf")),
                "lm_score_b": lm_scores.get(str(source_items[1].id), float("-inf")),
                "lm_score_diff": abs(
                    lm_scores.get(str(source_items[0].id), 0)
                    - lm_scores.get(str(source_items[1].id), 0)
                ),
            })
        updated_same_verb_items.append(fc_item)
    same_verb_items = updated_same_verb_items

    console.print(f"[green]✓[/green] Created {len(same_verb_items):,} same-verb pairs")

    # 2. Create different-verb pairs (group by template_id)
    with console.status("[bold]Creating different-verb pairs...[/bold]"):
        different_verb_items = create_forced_choice_items_from_groups(
            items=items,
            group_by=lambda item: str(item.item_template_id),
            n_alternatives=2,
            extract_text=extract_text,
            include_group_metadata=True,
        )

    # Add pair_type and additional metadata
    updated_different_verb_items = []
    for fc_item in different_verb_items:
        item1_id = fc_item.item_metadata.get("source_item_0_id")
        item2_id = fc_item.item_metadata.get("source_item_1_id")

        source_items = [item_lookup.get(item1_id), item_lookup.get(item2_id)]
        if all(source_items) and len(source_items) == 2:
            fc_item = fc_item.with_(item_metadata={
                **fc_item.item_metadata,
                "pair_type": "different_verb",
                "template_id": str(source_items[0].item_template_id),
                "template_structure": source_items[0].item_metadata.get(
                    "template_structure"
                ),
                "verb1": source_items[0].item_metadata.get("verb_lemma"),
                "verb2": source_items[1].item_metadata.get("verb_lemma"),
                "lm_score_a": lm_scores.get(str(source_items[0].id), float("-inf")),
                "lm_score_b": lm_scores.get(str(source_items[1].id), float("-inf")),
                "lm_score_diff": abs(
                    lm_scores.get(str(source_items[0].id), 0)
                    - lm_scores.get(str(source_items[1].id), 0)
                ),
            })
        updated_different_verb_items.append(fc_item)
    different_verb_items = updated_different_verb_items

    console.print(
        f"[green]✓[/green] Created {len(different_verb_items):,} different-verb pairs"
    )

    return same_verb_items + different_verb_items


def assign_quantiles_to_pairs(
    pair_items: list[Item],
    n_quantiles: int = 10,
) -> list[Item]:
    """Assign quantile bins using bead/lists/stratification.py.

    Stratifies by pair_type so same-verb and different-verb pairs
    get separate quantile distributions.
    """
    with console.status(
        "[bold]Assigning quantiles (stratified by pair_type)...[/bold]"
    ):
        # Build metadata dict for quantile assignment
        item_metadata = {item.id: item.item_metadata for item in pair_items}

        # Get item IDs
        item_ids = [item.id for item in pair_items]

        # Assign quantiles stratified by pair_type
        quantile_assignments = assign_quantiles_by_uuid(
            item_ids=item_ids,
            item_metadata=item_metadata,
            property_key="lm_score_diff",
            n_quantiles=n_quantiles,
            stratify_by_key="pair_type",
        )

        # Add quantile to each item's metadata (immutable update)
        pair_items = [
            item.with_(item_metadata={
                **item.item_metadata,
                "quantile": quantile_assignments[item.id],
            })
            for item in pair_items
        ]

    console.print(f"[green]✓[/green] Assigned quantiles to {len(pair_items):,} pairs")
    return pair_items


def main(
    config_path: Path = Path("config.yaml"),
    item_limit: int | None = None,
    output_path: Path | None = None,
) -> None:
    """Generate 2AFC pairs from filled templates.

    Parameters
    ----------
    config_path : Path
        Path to configuration file
    item_limit : int | None
        Limit number of filled templates to process
    output_path : Path | None
        Override output path from config
    """
    # Load configuration
    config = load_config(config_path)

    # Paths from config
    base_dir = Path(__file__).parent
    filled_templates_path = base_dir / config["template"]["output_path"]
    if output_path is None:
        output_path = base_dir / config["paths"]["2afc_pairs"]
    cache_dir = base_dir / config["paths"]["cache_dir"]

    console.rule("[bold]2AFC Pair Generation[/bold]")
    console.print(f"Base directory: [cyan]{base_dir}[/cyan]")
    console.print(f"Filled templates: [cyan]{filled_templates_path}[/cyan]")
    console.print(f"Output: [cyan]{output_path}[/cyan]\n")

    if item_limit:
        console.print(
            f"[yellow]⚠[/yellow]  Test mode: Limiting to {item_limit:,} filled templates\n"
        )

    # Load filled templates
    console.rule("[1/4] Loading Filled Templates")
    with console.status("[bold]Loading filled templates...[/bold]"):
        filled_templates = load_filled_templates(
            str(filled_templates_path), limit=item_limit
        )
    console.print(
        f"[green]✓[/green] Loaded {len(filled_templates):,} filled templates\n"
    )

    # Convert to Items
    console.rule("[2/4] Converting to Items")
    with console.status("[bold]Converting filled templates to items...[/bold]"):
        items = convert_filled_templates_to_items(filled_templates)
    console.print(f"[green]✓[/green] Created {len(items):,} items")

    # Show examples
    console.print("\n[dim]Example filled texts:[/dim]")
    for i, item in enumerate(items[:3]):
        console.print(f"  [dim]{i + 1}.[/dim] {item.rendered_elements['text']}")
    console.print()

    # Score with LM
    console.rule("[3/4] Scoring with Language Model")
    model_name = config["items"]["models"][0]["name"]
    lm_scores = score_filled_items_with_lm(
        items, cache_dir=cache_dir, model_name=model_name
    )
    console.print(f"[green]✓[/green] Scored {len(lm_scores):,} items\n")

    # Create forced-choice pairs
    console.rule("[4/4] Creating Forced-Choice Pairs")
    pair_items = create_forced_choice_pairs(items, lm_scores)

    if not pair_items:
        console.print("[red]✗[/red] No pairs were created. Exiting.")
        return

    console.print()

    # Assign quantiles
    quantile_bins = config["lists"].get("quantile_bins", 10)
    pair_items = assign_quantiles_to_pairs(pair_items, n_quantiles=quantile_bins)
    console.print()

    # Save
    console.rule("Saving Results")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with console.status(f"[bold]Writing to {output_path}...[/bold]"):
        with open(output_path, "w") as f:
            for item in pair_items:
                f.write(item.model_dump_json() + "\n")

    console.print(f"[green]✓[/green] Saved {len(pair_items):,} 2AFC pairs\n")

    # Summary
    console.rule("[bold]Summary[/bold]")
    same_verb_count = sum(
        1 for item in pair_items if item.item_metadata.get("pair_type") == "same_verb"
    )
    different_verb_count = sum(
        1
        for item in pair_items
        if item.item_metadata.get("pair_type") == "different_verb"
    )

    table = Table(show_header=False, box=None, padding=(0, 2))
    table.add_row("Same-verb pairs:", f"[cyan]{same_verb_count:,}[/cyan]")
    table.add_row("Different-verb pairs:", f"[cyan]{different_verb_count:,}[/cyan]")
    table.add_row("Total pairs:", f"[cyan]{len(pair_items):,}[/cyan]")
    table.add_row("Output file:", f"[cyan]{output_path}[/cyan]")
    console.print(table)

    console.print(
        "\n[dim]Next: Run generate_lists.py to partition pairs into experiment lists[/dim]"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate 2AFC pairs from filled templates"
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
        help="Limit number of filled templates to process (default: all)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override output path from config",
    )
    args = parser.parse_args()

    main(config_path=args.config, item_limit=args.limit, output_path=args.output)
