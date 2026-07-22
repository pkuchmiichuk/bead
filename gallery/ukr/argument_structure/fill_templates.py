#!/usr/bin/env python3
"""Fill the Ukrainian frames with lexical items.

Loads the frames and lexicons, then for each frame produces every combination of
items satisfying the slot constraints, rendered into a sentence, and writes them
to the configured output path. Per-slot strategies come from ``config.yaml``;
every slot is exhaustive by default, but a slot may instead be filled by a
masked language model.
"""

from __future__ import annotations

import argparse
import logging
from collections.abc import Iterable
from itertools import islice
from pathlib import Path

import yaml
from utils.frequency import most_frequent
from utils.renderers import UkrainianRenderer

from bead.cli.display import (
    create_progress,
    display_file_stats,
    print_header,
    print_info,
    print_success,
    print_warning,
)
from bead.data.serialization import write_jsonlines
from bead.resources.lexicon import Lexicon
from bead.resources.template_collection import TemplateCollection
from bead.templates.filler import FilledTemplate
from bead.templates.resolver import ConstraintResolver
from bead.templates.strategies import MixedFillingStrategy, StrategyConfig

BASE_DIR = Path(__file__).parent
LANGUAGE = "ukr"


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


def build_strategy(config: dict) -> MixedFillingStrategy:
    """Build the filling strategy from the config's per-slot strategies.

    The masked language model is loaded only when some slot requests it, so an
    all-exhaustive run needs no model.

    Parameters
    ----------
    config : dict
        Parsed configuration.

    Returns
    -------
    MixedFillingStrategy
        Strategy dispatching each slot to its configured filler.
    """
    slot_configs = config["template"]["slot_strategies"]
    mlm = config["template"]["mlm"]
    resolver = ConstraintResolver()

    model_adapter = None
    cache = None
    if any(sc["strategy"] == "mlm" for sc in slot_configs.values()):
        from bead.templates.adapters.cache import ModelOutputCache
        from bead.templates.adapters.huggingface import HuggingFaceMLMAdapter

        print_info(f"Loading MLM model: {mlm['model_name']}...")
        model_adapter = HuggingFaceMLMAdapter(
            model_name=mlm["model_name"], device=mlm.get("device", "cpu")
        )
        model_adapter.load_model()
        cache = ModelOutputCache(cache_dir=BASE_DIR / config["paths"]["cache_dir"])

    slot_strategies: dict[str, tuple[str, StrategyConfig]] = {}
    for slot_name, slot_config in slot_configs.items():
        if slot_config["strategy"] == "mlm":
            settings: StrategyConfig = {
                "resolver": resolver,
                "model_adapter": model_adapter,
                "cache": cache,
                "beam_size": mlm.get("beam_size", 5),
                "top_k": mlm.get("top_k", 10),
            }
            if "max_fills" in slot_config:
                settings["max_fills"] = slot_config["max_fills"]
            if "enforce_unique" in slot_config:
                settings["enforce_unique"] = slot_config["enforce_unique"]
            slot_strategies[slot_name] = ("mlm", settings)
        else:
            slot_strategies[slot_name] = (slot_config["strategy"], {})

    return MixedFillingStrategy(slot_strategies=slot_strategies)


def limit_verbs(
    lexicon: Lexicon,
    limit: int,
    *,
    by_frequency: bool = False,
    keep: Iterable[str] = (),
) -> Lexicon:
    """Return a lexicon keeping only ``limit`` unique verb lemmas.

    Parameters
    ----------
    lexicon : Lexicon
        Verb lexicon.
    limit : int
        Number of unique lemmas to keep.
    by_frequency : bool
        Keep the most frequent lemmas instead of the first ones, since VESUM is
        alphabetical and the first lemmas are rare rather than everyday verbs.
    keep : Iterable[str]
        Lemmas retained whatever the limit, for the anchor verbs the pair stage
        compares against.

    Returns
    -------
    Lexicon
        Lexicon restricted to ``limit`` lemmas plus ``keep``.
    """
    if by_frequency:
        allowed = set(most_frequent((item.lemma for item in lexicon.items), limit))
    else:
        lemmas: list[str] = []
        for item in lexicon.items:
            if item.lemma not in lemmas:
                lemmas.append(item.lemma)
            if len(lemmas) >= limit:
                break
        allowed = set(lemmas)
    allowed |= set(keep)
    return lexicon.filter(lambda item: item.lemma in allowed)


def main(
    config_path: Path,
    limit: int | None = None,
    max_per_template: int | None = None,
    output: Path | None = None,
    *,
    by_frequency: bool = False,
) -> None:
    """Fill every frame and write the sentences to the output path.

    Parameters
    ----------
    config_path : Path
        Path to the configuration file.
    limit : int | None
        Keep only this many unique verb lemmas (for quick runs).
    max_per_template : int | None
        Keep at most this many sentences per frame.
    output : Path | None
        Override the output path from the config.
    by_frequency : bool
        Select the most frequent verbs rather than the first ones.
    """
    config = load_config(config_path)
    logging.basicConfig(
        level=getattr(logging, config["logging"]["level"]),
        format=config["logging"]["format"],
    )

    print_header("Template Filling")

    templates_config = config["resources"]["templates"][0]
    templates = list(
        TemplateCollection.from_jsonl(
            str(BASE_DIR / templates_config["path"]), templates_config["name"]
        )
    )
    print_success(f"Loaded {len(templates)} frames")

    lexicons: list[Lexicon] = []
    for lex_config in config["resources"]["lexicons"]:
        lexicon = Lexicon.from_jsonl(
            str(BASE_DIR / lex_config["path"]), lex_config["name"]
        )
        if limit is not None and lex_config["name"] == "verbs":
            anchors = config["items"]["construction"]["anchors"].values()
            lexicon = limit_verbs(
                lexicon, limit, by_frequency=by_frequency, keep=anchors
            )
            selection = "most frequent" if by_frequency else "first"
            print_warning(
                f"Limited verbs to the {selection} {limit} lemmas "
                f"plus {len(set(anchors))} anchors ({len(lexicon)} forms)"
            )
        lexicons.append(lexicon)
        print_info(f"Loaded {len(lexicon)} items from {lex_config['name']}")

    strategy = build_strategy(config)
    renderer = UkrainianRenderer()

    filled: list[FilledTemplate] = []
    counts: dict[str, int] = {}
    with create_progress() as progress:
        task = progress.add_task("Filling frames", total=len(templates))
        for template in templates:
            combos = strategy.generate_from_template(
                template=template, lexicons=lexicons, language_code=LANGUAGE
            )
            if max_per_template is not None:
                combos = islice(combos, max_per_template)

            start = len(filled)
            for combo in combos:
                rendered = renderer.render(
                    template.template_string, combo, template.slots
                )
                filled.append(
                    FilledTemplate(
                        template_id=str(template.id),
                        template_name=template.name,
                        slot_fillers=combo,
                        rendered_text=rendered,
                        strategy_name="mixed",
                        template_slots={
                            name: slot.required
                            for name, slot in template.slots.items()
                        },
                    )
                )
            counts[template.name] = len(filled) - start
            progress.advance(task)

    for name, count in counts.items():
        print_info(f"{name}: {count:,} sentences")
    print_success(f"Filled {len(filled):,} sentences")

    output_path = output or (BASE_DIR / config["template"]["output_path"])
    output_path.parent.mkdir(parents=True, exist_ok=True)
    write_jsonlines(filled, output_path)
    display_file_stats(output_path, len(filled), "sentences")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Fill the Ukrainian frames with lexical items."
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
        help="Keep only this many unique verb lemmas (for testing).",
    )
    parser.add_argument(
        "--max-per-template",
        type=int,
        default=None,
        help="Keep at most this many sentences per frame.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="Override the output path from the config.",
    )
    parser.add_argument(
        "--by-frequency",
        action="store_true",
        help="Select the most frequent verbs rather than the first ones.",
    )
    args = parser.parse_args()
    main(
        config_path=args.config,
        limit=args.limit,
        max_per_template=args.max_per_template,
        output=args.output,
        by_frequency=args.by_frequency,
    )
