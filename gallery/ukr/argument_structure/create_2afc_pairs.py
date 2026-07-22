#!/usr/bin/env python3
"""Create 2AFC minimal pairs from the filled sentences.

Loads the filled sentences, scores each with the selected language model, then
pairs sentences that share a verb and object noun but differ in the object's
case, so each pair isolates the case the verb governs. Pairs are tagged with the
score difference and assigned difficulty quantiles, and written to the configured
output path.
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from uuid import UUID

import yaml
from utils.scoring import MaskedLanguageModelScorer

from bead.cli.display import (
    create_live_status,
    display_file_stats,
    print_header,
    print_info,
    print_success,
    print_warning,
)
from bead.items.forced_choice import (
    create_filtered_forced_choice_items,
    create_forced_choice_item,
)
from bead.items.item import Item
from bead.items.scoring import ItemScorer, LanguageModelScorer
from bead.lists.stratification import assign_quantiles_by_uuid
from bead.templates.filler import FilledTemplate

BASE_DIR = Path(__file__).parent


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


def load_filled(path: Path, limit: int | None = None) -> list[FilledTemplate]:
    """Load filled sentences from JSONL.

    Parameters
    ----------
    path : Path
        Path to the filled-sentences file.
    limit : int | None
        Read at most this many lines.

    Returns
    -------
    list[FilledTemplate]
        The loaded filled sentences.
    """
    filled: list[FilledTemplate] = []
    with path.open(encoding="utf-8") as f:
        for i, line in enumerate(f):
            if limit is not None and i >= limit:
                break
            filled.append(FilledTemplate.model_validate_json(line))
    return filled


def to_items(filled: list[FilledTemplate]) -> list[Item]:
    """Convert filled sentences with an object into scorable items.

    Intransitive sentences (no object) are dropped, since the case contrast is
    defined only over object-bearing frames.

    Parameters
    ----------
    filled : list[FilledTemplate]
        The filled sentences.

    Returns
    -------
    list[Item]
        One item per object-bearing sentence, tagged with verb, object class,
        and case.
    """
    items: list[Item] = []
    for ft in filled:
        verb = ft.slot_fillers.get("verb")
        obj = next(
            (f for name, f in ft.slot_fillers.items() if name.startswith("obj_")),
            None,
        )
        if verb is None or obj is None:
            continue
        items.append(
            Item(
                item_template_id=UUID(ft.template_id),
                rendered_elements={"text": ft.rendered_text},
                item_metadata={
                    "filled_template_id": str(ft.id),
                    "template_name": ft.template_name,
                    "verb_lemma": verb.lemma,
                    "object_lemma": obj.lemma,
                    "object_class": obj.features.get("semantic_class"),
                    "case": obj.features.get("case"),
                },
            )
        )
    return items


def select_model(config: dict, override: str | None) -> dict:
    """Return the model config to score with.

    Parameters
    ----------
    config : dict
        Parsed configuration.
    override : str | None
        Model name to force, ignoring the ``use_for_scoring`` flag.

    Returns
    -------
    dict
        The selected model's configuration entry.

    Raises
    ------
    ValueError
        If ``override`` names no configured model.
    """
    models = config["items"]["models"]
    if override is not None:
        for model in models:
            if model["name"] == override:
                return model
        raise ValueError(f"No configured model named '{override}'")
    for model in models:
        if model.get("use_for_scoring"):
            return model
    return models[0]


def build_scorer(model: dict, cache_dir: Path) -> ItemScorer:
    """Build the scorer matching a model's type.

    Parameters
    ----------
    model : dict
        A model configuration entry (``name``, ``type``, ``device``).
    cache_dir : Path
        Directory for the model output cache.

    Returns
    -------
    ItemScorer
        A masked or causal scorer.
    """
    device = model.get("device", "cpu")
    if model["type"] == "masked":
        return MaskedLanguageModelScorer(
            model_name=model["name"], cache_dir=cache_dir, device=device
        )
    return LanguageModelScorer(
        model_name=model["name"],
        cache_dir=cache_dir,
        device=device,
        dtype=model.get("dtype", "auto"),
    )


def make_pairs(items: list[Item]) -> list[Item]:
    """Pair object-bearing items by verb and object class, contrasting case.

    Groups items sharing a verb and object noun, forms every within-group pair,
    and drops pairs whose two sentences render identically (case syncretism).

    Parameters
    ----------
    items : list[Item]
        Scored, object-bearing items.

    Returns
    -------
    list[Item]
        Forced-choice pairs tagged with case contrast and score difference.
    """
    by_id = {str(item.id): item for item in items}

    def text_of(item: Item) -> str:
        """Return an item's rendered sentence text."""
        return item.rendered_elements.get("text", "")

    pairs = create_filtered_forced_choice_items(
        items=items,
        group_by=lambda item: (
            item.item_metadata["verb_lemma"],
            item.item_metadata["object_lemma"],
        ),
        n_alternatives=2,
        combination_filter=lambda combo: text_of(combo[0]) != text_of(combo[1]),
        extract_text=text_of,
    )

    enriched: list[Item] = []
    for pair in pairs:
        ids = pair.item_metadata.get("source_item_ids", ())
        first, second = by_id.get(ids[0]), by_id.get(ids[1])
        if first is None or second is None:
            continue
        case_a = first.item_metadata["case"]
        case_b = second.item_metadata["case"]
        score_a = first.item_metadata["lm_score"]
        score_b = second.item_metadata["lm_score"]
        pair = pair.with_(
            item_metadata={
                **pair.item_metadata,
                "pair_type": "case_contrast",
                "verb": first.item_metadata["verb_lemma"],
                "object_lemma": first.item_metadata["object_lemma"],
                "object_class": first.item_metadata["object_class"],
                "case_a": case_a,
                "case_b": case_b,
                "contrast": "-".join(sorted([str(case_a), str(case_b)])),
                "lm_score_a": score_a,
                "lm_score_b": score_b,
                "lm_score_diff": abs(score_a - score_b),
            }
        )
        enriched.append(pair)
    return enriched


def make_anchor_pairs(items: list[Item], anchors: dict[str, str]) -> list[Item]:
    """Pair each verb against the anchor verb for its case, noun held fixed.

    The anchor's government is known, so the comparison puts test verbs on a
    scale shared across verbs and exposes verbs that fit no frame at all.

    Parameters
    ----------
    items : list[Item]
        Scored, object-bearing items.
    anchors : dict[str, str]
        Anchor lemma for each case.

    Returns
    -------
    list[Item]
        Forced-choice pairs contrasting a verb with its case's anchor.
    """
    by_key = {
        (
            item.item_metadata["verb_lemma"],
            item.item_metadata["case"],
            item.item_metadata["object_lemma"],
        ): item
        for item in items
    }

    pairs: list[Item] = []
    for item in items:
        case_name = item.item_metadata["case"]
        verb = item.item_metadata["verb_lemma"]
        anchor_lemma = anchors.get(str(case_name))
        if anchor_lemma is None or verb == anchor_lemma:
            continue
        key = (anchor_lemma, case_name, item.item_metadata["object_lemma"])
        anchor = by_key.get(key)
        if anchor is None:
            continue

        text = item.rendered_elements.get("text", "")
        anchor_text = anchor.rendered_elements.get("text", "")
        if text == anchor_text:
            continue
        score = item.item_metadata["lm_score"]
        anchor_score = anchor.item_metadata["lm_score"]
        pairs.append(
            create_forced_choice_item(
                text,
                anchor_text,
                metadata={
                    "pair_type": "anchor_contrast",
                    "verb": verb,
                    "anchor": anchor_lemma,
                    "case": case_name,
                    "contrast": f"anchor-{case_name}",
                    "object_lemma": item.item_metadata["object_lemma"],
                    "object_class": item.item_metadata["object_class"],
                    "lm_score_a": score,
                    "lm_score_b": anchor_score,
                    "lm_score_diff": abs(score - anchor_score),
                    "source_item_ids": (str(item.id), str(anchor.id)),
                },
            )
        )
    return pairs


def main(
    config_path: Path,
    limit: int | None = None,
    output: Path | None = None,
    model_override: str | None = None,
) -> None:
    """Score sentences, build case-contrast pairs, and write them out.

    Parameters
    ----------
    config_path : Path
        Path to the configuration file.
    limit : int | None
        Read at most this many filled sentences.
    output : Path | None
        Override the output path from the config.
    model_override : str | None
        Score with this model name instead of the configured default.
    """
    config = load_config(config_path)
    logging.basicConfig(
        level=getattr(logging, config["logging"]["level"]),
        format=config["logging"]["format"],
    )

    print_header("2AFC Pair Creation")

    cache_dir = BASE_DIR / config["paths"]["cache_dir"]
    filled_path = BASE_DIR / config["template"]["output_path"]
    output_path = output or (BASE_DIR / config["paths"]["2afc_pairs"])

    filled = load_filled(filled_path, limit)
    items = to_items(filled)
    print_success(f"Loaded {len(items):,} object-bearing sentences")
    if not items:
        print_warning("No object-bearing sentences to pair.")
        return

    model = select_model(config, model_override)
    scorer = build_scorer(model, cache_dir)
    print_info(f"Scoring with {model['name']} ({model['type']})")
    with create_live_status("Scoring sentences..."):
        scores = scorer.score_batch(items)
    items = [
        item.with_(item_metadata={**item.item_metadata, "lm_score": score})
        for item, score in zip(items, scores, strict=True)
    ]

    case_pairs = make_pairs(items)
    anchor_pairs = make_anchor_pairs(
        items, config["items"]["construction"]["anchors"]
    )
    pairs = case_pairs + anchor_pairs
    print_success(
        f"Built {len(case_pairs):,} case-contrast and "
        f"{len(anchor_pairs):,} anchor pairs"
    )
    if not pairs:
        print_warning("No pairs were created.")
        return

    n_quantiles = config["lists"]["quantile_bins"]
    quantiles = assign_quantiles_by_uuid(
        item_ids=[pair.id for pair in pairs],
        item_metadata={pair.id: pair.item_metadata for pair in pairs},
        property_key="lm_score_diff",
        n_quantiles=n_quantiles,
        stratify_by_key="contrast",
    )
    pairs = [
        pair.with_(item_metadata={**pair.item_metadata, "quantile": quantiles[pair.id]})
        for pair in pairs
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        for pair in pairs:
            f.write(pair.model_dump_json() + "\n")
    display_file_stats(output_path, len(pairs), "pairs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Create 2AFC case-contrast pairs from the filled sentences."
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=BASE_DIR / "config.yaml",
        help="Path to the configuration file.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Read at most this many filled sentences (for testing).",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the output path from the config.",
    )
    parser.add_argument(
        "--model",
        type=str,
        default=None,
        help="Score with this model name instead of the configured default.",
    )
    args = parser.parse_args()
    main(
        config_path=args.config,
        limit=args.limit,
        output=args.output,
        model_override=args.model,
    )
