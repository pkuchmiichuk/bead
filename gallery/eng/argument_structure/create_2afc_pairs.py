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
import json
import sys
from pathlib import Path
from uuid import uuid4

import layers_io
import yaml
from protocol import ACCEPTABILITY_ANCHOR_NAME

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
from bead.items.forced_choice import create_forced_choice_items_from_groups
from bead.items.item import Item
from bead.items.scoring import AcceptabilityScorer, LanguageModelScorer
from bead.lists.constraints import CategoricalBinning, QuantileBinning
from bead.lists.stratification import assign_grid_cells_by_uuid
from bead.templates.filler import FilledTemplate


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
            data = json.loads(line)
            filled_templates.append(FilledTemplate(**data))
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

        # Use template_name as structure identifier
        item = Item(
            item_template_id=ft.template_id,
            rendered_elements={"text": text},
            item_metadata={
                "filled_template_id": str(ft.id),
                "template_id": ft.template_id,
                "template_name": ft.template_name,
                "template_structure": ft.template_name,
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


def with_metadata(item: Item, extra: dict[str, object]) -> Item:
    """Return a copy of ``item`` with ``extra`` merged into its metadata.

    Items are frozen, so metadata is updated through ``with_`` rather than by
    mutating the dict in place (which would not persist). ``None`` values are
    skipped so absent fields do not fail metadata validation.
    """
    merged: dict[str, object] = dict(item.item_metadata)
    for key, value in extra.items():
        if value is not None:
            merged[key] = value
    return item.with_(item_metadata=merged)


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
        text = item.rendered_elements.get("text", "")
        return text if isinstance(text, str) else ""

    def lm_diff(a: Item, b: Item) -> float:
        return abs(lm_scores.get(str(a.id), 0.0) - lm_scores.get(str(b.id), 0.0))

    def source_pair(fc_item: Item) -> tuple[Item, Item] | None:
        id0 = fc_item.item_metadata.get("source_item_0_id")
        id1 = fc_item.item_metadata.get("source_item_1_id")
        s0 = item_lookup.get(str(id0)) if id0 is not None else None
        s1 = item_lookup.get(str(id1)) if id1 is not None else None
        if s0 is None or s1 is None:
            return None
        return s0, s1

    # 1. Create same-verb pairs (group by verb_lemma)
    with console.status("[bold]Creating same-verb pairs...[/bold]"):
        same_verb_raw = create_forced_choice_items_from_groups(
            items=items,
            group_by=lambda item: item.item_metadata.get("verb_lemma", "unknown"),
            n_alternatives=2,
            extract_text=extract_text,
            include_group_metadata=True,
        )

    same_verb_items: list[Item] = []
    for fc_item in same_verb_raw:
        pair = source_pair(fc_item)
        if pair is None:
            continue
        s0, s1 = pair
        same_verb_items.append(
            with_metadata(
                fc_item,
                {
                    "pair_type": "same_verb",
                    "verb": s0.item_metadata.get("verb_lemma"),
                    "template1": s0.item_metadata.get("template_structure"),
                    "template2": s1.item_metadata.get("template_structure"),
                    "lm_score_a": lm_scores.get(str(s0.id), 0.0),
                    "lm_score_b": lm_scores.get(str(s1.id), 0.0),
                    "lm_score_diff": lm_diff(s0, s1),
                    "anchor": ACCEPTABILITY_ANCHOR_NAME,
                },
            )
        )

    print_success(f"Created {len(same_verb_items):,} same-verb pairs")

    # 2. Create different-verb pairs (group by template_id)
    with console.status("[bold]Creating different-verb pairs...[/bold]"):
        different_verb_raw = create_forced_choice_items_from_groups(
            items=items,
            group_by=lambda item: str(item.item_template_id),
            n_alternatives=2,
            extract_text=extract_text,
            include_group_metadata=True,
        )

    different_verb_items: list[Item] = []
    for fc_item in different_verb_raw:
        pair = source_pair(fc_item)
        if pair is None:
            continue
        s0, s1 = pair
        different_verb_items.append(
            with_metadata(
                fc_item,
                {
                    "pair_type": "different_verb",
                    "template_id": str(s0.item_template_id),
                    "template_structure": s0.item_metadata.get("template_structure"),
                    "verb1": s0.item_metadata.get("verb_lemma"),
                    "verb2": s1.item_metadata.get("verb_lemma"),
                    "lm_score_a": lm_scores.get(str(s0.id), 0.0),
                    "lm_score_b": lm_scores.get(str(s1.id), 0.0),
                    "lm_score_diff": lm_diff(s0, s1),
                    "anchor": ACCEPTABILITY_ANCHOR_NAME,
                },
            )
        )

    print_success(f"Created {len(different_verb_items):,} different-verb pairs")

    return same_verb_items + different_verb_items


def score_pairs_with_acceptability(
    pair_items: list[Item],
    checkpoint_dir: Path,
) -> list[Item]:
    """Score 2AFC pairs with the trained acceptability model.

    Returns new pairs carrying ``acceptability_score_diff`` (the predicted
    preference margin) and ``accept_p_prefer_a``. When the checkpoint is
    missing, the acceptability dimension is filled with zeros so the grid still
    partitions.

    Parameters
    ----------
    pair_items : list[Item]
        The 2AFC pairs to score.
    checkpoint_dir : Path
        Directory holding the trained ForcedChoiceModel checkpoint.

    Returns
    -------
    list[Item]
        The scored pairs.
    """
    if not checkpoint_dir.exists():
        print_warning(
            f"Acceptability checkpoint not found at {checkpoint_dir}; "
            "filling acceptability_score_diff with 0.0. Train the model first "
            "with train_acceptability_model.py for grid stratification."
        )
        return [
            with_metadata(
                item,
                {"acceptability_score_diff": 0.0, "accept_p_prefer_a": 0.5},
            )
            for item in pair_items
        ]

    with console.status("[bold]Scoring pairs with acceptability model...[/bold]"):
        scorer = AcceptabilityScorer.from_checkpoint(checkpoint_dir)
        scored = scorer.score_with_metadata(pair_items)
        result = [
            with_metadata(
                item,
                {
                    "acceptability_score_diff": float(
                        scored[item.id]["acceptability_margin"]
                    ),
                    "accept_p_prefer_a": float(scored[item.id]["p_first"]),
                },
            )
            for item in pair_items
        ]

    print_success(f"Scored {len(pair_items):,} pairs with the acceptability model")
    return result


def assign_grid_cells_to_pairs(
    pair_items: list[Item],
    n_quantiles: int = 5,
) -> list[Item]:
    """Stratify pairs across a grid of acceptability margin x LM score x pair type.

    Bins the acceptability-model margin and the language-model score difference
    into quantiles and crosses them with the categorical pair type, storing the
    flattened grid cell id as ``stratum_cell`` on each returned pair.
    """
    with console.status("[bold]Assigning grid cells (acceptability x LM)...[/bold]"):
        item_metadata = {item.id: dict(item.item_metadata) for item in pair_items}
        item_ids = [item.id for item in pair_items]

        cell_ids = assign_grid_cells_by_uuid(
            item_ids=item_ids,
            item_metadata=item_metadata,
            property_keys=[
                "acceptability_score_diff",
                "lm_score_diff",
                "pair_type",
            ],
            binnings=[
                QuantileBinning(binning="quantile", n_quantiles=n_quantiles),
                QuantileBinning(binning="quantile", n_quantiles=n_quantiles),
                CategoricalBinning(
                    binning="categorical",
                    categories=("same_verb", "different_verb"),
                ),
            ],
        )

        result = [
            with_metadata(item, {"stratum_cell": cell_ids[item.id]})
            for item in pair_items
        ]

    n_cells = len(set(cell_ids.values()))
    print_success(f"Assigned {len(pair_items):,} pairs across {n_cells} grid cells")
    return result


def main(
    config_path: Path = Path("config.yaml"),
    item_limit: int | None = None,
    output_path: Path | None = None,
    *,
    yes: bool = False,
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
    yes : bool
        Skip confirmation prompts (for non-interactive use).
    """
    # Load configuration
    try:
        config = load_config(config_path)
    except Exception as e:
        print_error(f"Failed to load config: {e}")
        sys.exit(1)

    # Paths from config
    base_dir = Path(__file__).parent
    filled_templates_path = base_dir / config["template"]["output_path"]
    if output_path is None:
        output_path = base_dir / config["paths"]["2afc_pairs"]
    cache_dir = base_dir / config["paths"]["cache_dir"]

    print_header("2AFC Pair Generation")
    console.print(f"Base directory: [cyan]{base_dir}[/cyan]")
    console.print(f"Filled templates: [cyan]{filled_templates_path}[/cyan]")
    console.print(f"Output: [cyan]{output_path}[/cyan]\n")

    # Check for existing output
    if output_path.exists() and not yes:
        if not confirm(f"Overwrite {output_path}?", default=False):
            print_info("Operation cancelled.")
            return

    if item_limit:
        print_warning(f"Test mode: Limiting to {item_limit:,} filled templates\n")

    # Load filled templates
    print_header("1/4 Loading Filled Templates")
    try:
        with console.status("[bold]Loading filled templates...[/bold]"):
            filled_templates = load_filled_templates(
                str(filled_templates_path), limit=item_limit
            )
        print_success(f"Loaded {len(filled_templates):,} filled templates\n")
    except Exception as e:
        print_error(f"Failed to load filled templates: {e}")
        sys.exit(1)

    # Convert to Items
    print_header("2/4 Converting to Items")
    try:
        with console.status("[bold]Converting filled templates to items...[/bold]"):
            items = convert_filled_templates_to_items(filled_templates)
        print_success(f"Created {len(items):,} items")

        # Show examples
        console.print("\n[dim]Example filled texts:[/dim]")
        for i, item in enumerate(items[:3]):
            console.print(f"  [dim]{i + 1}.[/dim] {item.rendered_elements['text']}")
        console.print()
    except Exception as e:
        print_error(f"Failed to convert templates: {e}")
        sys.exit(1)

    # Score with LM
    print_header("3/4 Scoring with Language Model")
    try:
        model_name = config["items"]["models"][0]["name"]
        lm_scores = score_filled_items_with_lm(
            items, cache_dir=cache_dir, model_name=model_name
        )
        print_success(f"Scored {len(lm_scores):,} items\n")
    except Exception as e:
        print_error(f"Failed to score items: {e}")
        sys.exit(1)

    # Create forced-choice pairs
    print_header("4/4 Creating Forced-Choice Pairs")
    try:
        pair_items = create_forced_choice_pairs(items, lm_scores)

        if not pair_items:
            print_error("No pairs were created. Exiting.")
            sys.exit(1)

        console.print()

        # Score with the acceptability model, then stratify across the grid of
        # acceptability margin x language-model score x pair type.
        checkpoint_dir = base_dir / config["acceptability_model"]["checkpoint_dir"]
        pair_items = score_pairs_with_acceptability(pair_items, checkpoint_dir)
        quantile_bins = config["lists"].get("quantile_bins", 5)
        pair_items = assign_grid_cells_to_pairs(pair_items, n_quantiles=quantile_bins)
        console.print()
    except Exception as e:
        print_error(f"Failed to create pairs: {e}")
        sys.exit(1)

    # Save: a bead-native JSONL plus the canonical layers fragment and an
    # Arrow/Parquet corpus through the bead lairs codec.
    print_header("Saving Results")
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with console.status(f"[bold]Writing to {output_path}...[/bold]"):
            with open(output_path, "w") as f:
                for item in pair_items:
                    f.write(item.model_dump_json() + "\n")
        print_success(f"Saved {len(pair_items):,} 2AFC pairs to {output_path.name}")

        if config.get("layers", {}).get("enabled", True):
            fragment_path = base_dir / config["paths"]["2afc_pairs_fragment"]
            corpus_dir = (
                base_dir / config["paths"]["2afc_corpus_dir"]
                if config["layers"].get("materialize", True)
                else None
            )
            with console.status("[bold]Encoding layers fragment + corpus...[/bold]"):
                layers_io.write_items(
                    pair_items,
                    name="argument_structure_2afc",
                    fragment_path=fragment_path,
                    materialize_dir=corpus_dir,
                )
            print_success(f"Wrote layers fragment to {fragment_path.name}")
            if corpus_dir is not None:
                print_success(f"Materialized layers corpus to {corpus_dir.name}\n")
    except Exception as e:
        print_error(f"Failed to save output: {e}")
        sys.exit(1)

    # Summary
    print_header("Summary")
    same_verb_count = sum(
        1 for item in pair_items if item.item_metadata.get("pair_type") == "same_verb"
    )
    different_verb_count = sum(
        1
        for item in pair_items
        if item.item_metadata.get("pair_type") == "different_verb"
    )

    table = create_summary_table(
        {
            "Same-verb pairs": f"{same_verb_count:,}",
            "Different-verb pairs": f"{different_verb_count:,}",
            "Total pairs": f"{len(pair_items):,}",
            "Output file": str(output_path),
        }
    )
    console.print(table)

    print_info("Next: Run generate_lists.py to partition pairs into experiment lists")


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
        output_path=args.output,
        yes=args.yes,
    )
