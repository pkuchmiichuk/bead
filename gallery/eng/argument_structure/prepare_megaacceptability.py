#!/usr/bin/env python3
"""Prepare MegaAcceptability data and derive a per-annotator 2AFC training set.

MegaAcceptability collects single-sentence ordinal (1-7 Likert) ratings from many
annotators. The 2AFC acceptability experiment in this gallery instead asks which of
two sentences sounds more natural. This script bridges the two via per-annotator
within-rater pairing: for each annotator, every pair of sentences they both rated
becomes one forced-choice training item whose gold label is the option holding the
sentence they rated higher. The annotator id rides along as ``participant_id`` so a
downstream ForcedChoiceModel can fit participant random effects.

The script:

1. Loads MegaAcceptability rows from a local CSV (or downloads one named in config).
2. Emits the raw ratings as a layers corpus (guarded; never fatal).
3. Builds per-annotator 2AFC training items via :func:`build_per_annotator_pairs`.
4. Writes the derived items as JSONL and as a layers corpus (guarded).

Run ``--self-test`` to exercise the pairing logic on a small in-memory dataset
without any network access, file IO, or heavy model dependencies.
"""

from __future__ import annotations

import argparse
import csv
import shutil
import sys
import urllib.error
import urllib.request
from collections.abc import Iterator
from dataclasses import dataclass
from itertools import combinations
from pathlib import Path
from random import Random
from typing import Literal
from uuid import uuid4

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
from bead.items.item import Item, ItemCollection, MetadataValue

type PairingScope = Literal["all", "same_verb", "same_frame"]

OPTION_A = "option_a"
OPTION_B = "option_b"
DEFAULT_SEED = 42
DEFAULT_SCOPE: PairingScope = "all"
VALID_SCOPES: tuple[PairingScope, ...] = ("all", "same_verb", "same_frame")


@dataclass(frozen=True)
class RatingRow:
    """A single MegaAcceptability rating by one annotator.

    Attributes
    ----------
    participant_id
        Identifier of the annotator who produced the rating.
    verb
        Verb associated with the sentence.
    frame
        Syntactic frame associated with the sentence.
    sentence
        Surface text that was rated.
    rating
        Ordinal acceptability rating, coerced to float (1-7 Likert).
    """

    participant_id: str
    verb: str
    frame: str
    sentence: str
    rating: float


# --- pure pairing logic -----------------------------------------------------


def _pair_type(row_a: RatingRow, row_b: RatingRow) -> str:
    """Classify a pair as ``same_verb``, ``same_frame``, or ``cross``."""
    if row_a.verb == row_b.verb:
        return "same_verb"
    if row_a.frame == row_b.frame:
        return "same_frame"
    return "cross"


def _group_rows(
    rows: list[RatingRow], key: Literal["verb", "frame"]
) -> list[list[RatingRow]]:
    """Group rows by verb or frame, returning groups in sorted key order."""
    groups: dict[str, list[RatingRow]] = {}
    for row in rows:
        group_key = row.verb if key == "verb" else row.frame
        groups.setdefault(group_key, []).append(row)
    return [groups[group_key] for group_key in sorted(groups)]


def _candidate_pairs(
    rows: list[RatingRow], scope: PairingScope
) -> Iterator[tuple[RatingRow, RatingRow]]:
    """Yield within-scope unordered pairs of an annotator's rated sentences."""
    if scope == "same_verb":
        groups = _group_rows(rows, "verb")
    elif scope == "same_frame":
        groups = _group_rows(rows, "frame")
    else:
        groups = [rows]
    for group in groups:
        yield from combinations(group, 2)


def _build_pair_item(
    row_i: RatingRow,
    row_j: RatingRow,
    *,
    participant_id: str,
    rng: Random,
) -> Item:
    """Build one forced-choice item from a within-rater sentence pair.

    The orientation (which sentence becomes ``option_a``) is drawn from ``rng`` so
    that, across many pairs, the gold label is balanced between the two options.

    Parameters
    ----------
    row_i, row_j
        The two rows rated by the same annotator.
    participant_id
        Identifier of the annotator, attached as item metadata.
    rng
        Seeded random generator driving deterministic orientation.

    Returns
    -------
    Item
        A 2AFC item with ``option_a`` / ``option_b`` rendered elements and a
        ``label`` metadata key naming the higher-rated option.
    """
    if rng.random() < 0.5:
        option_a, option_b = row_i, row_j
    else:
        option_a, option_b = row_j, row_i

    if option_a.rating > option_b.rating:
        label = OPTION_A
    elif option_b.rating > option_a.rating:
        label = OPTION_B
    else:
        # Exact tie: only reached when drop_ties is False. Pick a deterministic
        # option and flag the pair so callers can filter ties downstream.
        label = OPTION_A

    is_tie = option_a.rating == option_b.rating
    shared_verb = option_a.verb if option_a.verb == option_b.verb else None
    shared_frame = option_a.frame if option_a.frame == option_b.frame else None

    metadata: dict[str, MetadataValue] = {
        "participant_id": participant_id,
        "verb": shared_verb,
        "frame": shared_frame,
        "verb_option_a": option_a.verb,
        "verb_option_b": option_b.verb,
        "frame_option_a": option_a.frame,
        "frame_option_b": option_b.frame,
        "rating_option_a": option_a.rating,
        "rating_option_b": option_b.rating,
        "pair_type": _pair_type(option_a, option_b),
        "label": label,
        "is_tie": is_tie,
        "anchor": ACCEPTABILITY_ANCHOR_NAME,
    }

    return Item(
        item_template_id=uuid4(),
        rendered_elements={OPTION_A: option_a.sentence, OPTION_B: option_b.sentence},
        options=(option_a.sentence, option_b.sentence),
        item_metadata=metadata,
    )


def build_per_annotator_pairs(
    rows: list[RatingRow],
    *,
    scope: PairingScope = DEFAULT_SCOPE,
    drop_ties: bool = True,
    max_per_annotator: int | None = None,
    seed: int = DEFAULT_SEED,
) -> list[Item]:
    """Derive 2AFC training items by within-rater pairing of MegaAcceptability rows.

    For each annotator, every within-scope pair of sentences they both rated yields
    one forced-choice item. The gold label names the option holding the sentence the
    annotator rated higher; pairs with equal ratings are dropped unless ``drop_ties``
    is False. Pairs per annotator are capped at ``max_per_annotator`` by shuffling and
    keeping the first N. All randomness derives from ``seed`` so the output is
    reproducible.

    Parameters
    ----------
    rows
        Flat list of per-annotator ratings.
    scope
        Which pairs to form: ``all`` pairs, only ``same_verb`` pairs, or only
        ``same_frame`` pairs.
    drop_ties
        Whether to drop pairs with equal ratings. When False, ties are kept with a
        deterministic label and an ``is_tie`` metadata flag.
    max_per_annotator
        Optional cap on the number of pairs emitted per annotator.
    seed
        Seed for the deterministic random generator.

    Returns
    -------
    list[Item]
        Forced-choice training items, one per retained sentence pair.

    Examples
    --------
    >>> rows = [
    ...     RatingRow("p1", "give", "NP_NP", "She gave him a book.", 7.0),
    ...     RatingRow("p1", "give", "NP_PP", "She gave a book to him.", 5.0),
    ... ]
    >>> items = build_per_annotator_pairs(rows, scope="same_verb")
    >>> items[0].item_metadata["participant_id"]
    'p1'
    """
    by_participant: dict[str, list[RatingRow]] = {}
    for row in rows:
        by_participant.setdefault(row.participant_id, []).append(row)

    rng = Random(seed)
    items: list[Item] = []
    for participant_id in sorted(by_participant):
        participant_rows = sorted(
            by_participant[participant_id], key=lambda row: row.sentence
        )
        participant_items: list[Item] = []
        for row_i, row_j in _candidate_pairs(participant_rows, scope):
            if drop_ties and row_i.rating == row_j.rating:
                continue
            participant_items.append(
                _build_pair_item(row_i, row_j, participant_id=participant_id, rng=rng)
            )
        if max_per_annotator is not None and len(participant_items) > max_per_annotator:
            rng.shuffle(participant_items)
            participant_items = participant_items[:max_per_annotator]
        items.extend(participant_items)
    return items


def split_rows_by_sentence(
    rows: list[RatingRow], dev_fraction: float, seed: int
) -> tuple[list[RatingRow], list[RatingRow]]:
    """Split rows into train and dev so no sentence appears in both.

    Unique sentences are partitioned: a ``dev_fraction`` of them (and every
    rating of those sentences) is held out as the dev split, the rest form the
    train split. Holding out whole sentences guarantees the dev set contains
    only sentences never seen during training, so dev performance measures
    generalization to new stimuli rather than new annotator pairings.

    Parameters
    ----------
    rows : list[RatingRow]
        All rating rows.
    dev_fraction : float
        Fraction of unique sentences to hold out for the dev split.
    seed : int
        Seed for the deterministic sentence shuffle.

    Returns
    -------
    tuple[list[RatingRow], list[RatingRow]]
        The train rows and dev rows.
    """
    if dev_fraction <= 0.0:
        return list(rows), []
    sentences = sorted({row.sentence for row in rows})
    Random(seed).shuffle(sentences)
    n_dev = max(1, round(len(sentences) * dev_fraction))
    dev_sentences = set(sentences[:n_dev])
    train_rows = [row for row in rows if row.sentence not in dev_sentences]
    dev_rows = [row for row in rows if row.sentence in dev_sentences]
    return train_rows, dev_rows


# --- raw item construction --------------------------------------------------


def build_raw_items(rows: list[RatingRow]) -> list[Item]:
    """Build one raw item per rating, carrying the rating in item metadata."""
    items: list[Item] = []
    for row in rows:
        metadata: dict[str, MetadataValue] = {
            "participant_id": row.participant_id,
            "verb": row.verb,
            "frame": row.frame,
            "rating": row.rating,
            "anchor": ACCEPTABILITY_ANCHOR_NAME,
        }
        items.append(
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": row.sentence},
                item_metadata=metadata,
            )
        )
    return items


# --- IO helpers -------------------------------------------------------------


def load_config(config_path: Path) -> dict[str, object]:
    """Load the YAML configuration file."""
    with open(config_path, encoding="utf-8") as handle:
        return yaml.safe_load(handle)


def get_acceptability_config(config: dict[str, object]) -> dict[str, object]:
    """Return the ``acceptability_model`` section, or an empty dict if absent."""
    section = config.get("acceptability_model")
    return section if isinstance(section, dict) else {}


def resolve_columns(section: dict[str, object]) -> dict[str, str]:
    """Resolve CSV column names from config, falling back to sensible defaults."""
    columns = section.get("columns")
    columns = columns if isinstance(columns, dict) else {}
    return {
        "verb": str(columns.get("verb", "verb")),
        "frame": str(columns.get("frame", "frame")),
        "sentence": str(columns.get("sentence", "sentence")),
        "participant_id": str(columns.get("participant_id", "participant")),
        "rating": str(columns.get("rating", "rating")),
    }


def resolve_scope(section: dict[str, object]) -> PairingScope:
    """Resolve the pairing scope, defaulting to ``all`` for unknown values."""
    scope = section.get("pairing_scope", DEFAULT_SCOPE)
    if scope in VALID_SCOPES:
        return scope  # type: ignore[return-value]
    print_warning(f"Unknown pairing_scope '{scope}'; using '{DEFAULT_SCOPE}'")
    return DEFAULT_SCOPE


def download_csv(url: str, target: Path) -> Path | None:
    """Download a CSV from ``url`` into ``target``, returning None on failure."""
    target.parent.mkdir(parents=True, exist_ok=True)
    try:
        with console.status(f"[bold]Downloading {url}...[/bold]"):
            with urllib.request.urlopen(url) as response, open(target, "wb") as out:
                shutil.copyfileobj(response, out)
    except (urllib.error.URLError, OSError, ValueError) as exc:
        print_error(f"Could not download {url}: {exc}")
        return None
    return target


def ensure_source_csv(
    *,
    source_path: Path | None,
    source_url: str | None,
    cache_dir: Path,
) -> Path | None:
    """Resolve the MegaAcceptability CSV, downloading into the cache if needed."""
    if source_path is not None and source_path.exists():
        return source_path
    if source_url is None:
        print_error(
            "No MegaAcceptability source found. Set acceptability_model.source_path "
            "to a local CSV, or acceptability_model.source_url to download one."
        )
        return None
    cached = cache_dir / "megaacceptability.csv"
    if cached.exists():
        return cached
    return download_csv(source_url, cached)


def load_rating_rows(
    csv_path: Path,
    *,
    columns: dict[str, str],
    limit: int | None = None,
) -> list[RatingRow]:
    """Read MegaAcceptability rows from CSV, coercing ratings to float."""
    rows: list[RatingRow] = []
    with open(csv_path, newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        for index, record in enumerate(reader):
            if limit is not None and index >= limit:
                break
            raw_rating = record.get(columns["rating"])
            try:
                rating = float(raw_rating)
            except (TypeError, ValueError):
                continue
            rows.append(
                RatingRow(
                    participant_id=str(record.get(columns["participant_id"], "")),
                    verb=str(record.get(columns["verb"], "")),
                    frame=str(record.get(columns["frame"], "")),
                    sentence=str(record.get(columns["sentence"], "")),
                    rating=rating,
                )
            )
    return rows


def write_items_jsonl(items: list[Item], path: Path) -> None:
    """Write items to JSONL, one ``model_dump_json`` line each."""
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as handle:
        for item in items:
            handle.write(item.model_dump_json() + "\n")


def emit_layers_corpus(items: list[Item], *, corpus_name: str, out_dir: Path) -> None:
    """Emit items as a materialized layers corpus, warning (not failing) on error."""
    try:
        from bead.interop.layers import (  # noqa: PLC0415
            items_to_corpus,
            materialize_corpus,
        )
    except Exception as exc:  # noqa: BLE001
        print_warning(f"Skipping layers corpus '{corpus_name}': {exc}")
        return
    try:
        collection = ItemCollection(
            name=corpus_name,
            source_template_collection_id=uuid4(),
            source_filled_collection_id=uuid4(),
            items=tuple(items),
        )
        out_dir.mkdir(parents=True, exist_ok=True)
        corpus = items_to_corpus(collection, corpus_name=corpus_name)
        written = materialize_corpus(corpus, out_dir)
        print_success(
            f"Wrote layers corpus '{corpus_name}' ({len(written)} files) to {out_dir}"
        )
    except Exception as exc:  # noqa: BLE001
        print_warning(f"Could not emit layers corpus '{corpus_name}': {exc}")


# --- self-test --------------------------------------------------------------


def _synthetic_rows() -> list[RatingRow]:
    """Build a tiny synthetic dataset: 3 annotators each rating 4 sentences."""
    verbs = ("give", "send", "throw")
    frames = ("NP_NP", "NP_PP")
    rows: list[RatingRow] = []
    for participant_index in range(3):
        participant_id = f"annotator_{participant_index}"
        for sentence_index in range(4):
            verb = verbs[sentence_index % len(verbs)]
            frame = frames[sentence_index % len(frames)]
            sentence = f"The {verb} sentence number {sentence_index}."
            rating = float(1 + (sentence_index + participant_index) % 7)
            rows.append(RatingRow(participant_id, verb, frame, sentence, rating))
    return rows


def run_self_test() -> int:
    """Exercise the pairing logic on synthetic data without IO or heavy imports."""
    print_header("prepare_megaacceptability self-test")
    rows = _synthetic_rows()
    expected_participants = {row.participant_id for row in rows}

    pairs = build_per_annotator_pairs(
        rows,
        scope="all",
        drop_ties=True,
        max_per_annotator=None,
        seed=DEFAULT_SEED,
    )

    assert pairs, "expected at least one derived 2AFC pair"

    seen_participants: set[str] = set()
    for item in pairs:
        metadata = item.item_metadata
        seen_participants.add(str(metadata["participant_id"]))

        rating_a = metadata["rating_option_a"]
        rating_b = metadata["rating_option_b"]
        higher_option = OPTION_A if rating_a > rating_b else OPTION_B
        assert metadata["label"] == higher_option, (
            "gold label must point to the higher-rated sentence "
            f"(option_a={rating_a}, option_b={rating_b}, label={metadata['label']})"
        )
        assert item.rendered_elements[OPTION_A], "option_a text must be present"
        assert item.rendered_elements[OPTION_B], "option_b text must be present"

    assert seen_participants == expected_participants, (
        "participant_ids did not propagate to the derived pairs: "
        f"saw {sorted(seen_participants)}, expected {sorted(expected_participants)}"
    )

    print_success(
        f"Self-test passed: {len(pairs)} pairs from "
        f"{len(expected_participants)} annotators, all labels and "
        "participant_ids verified"
    )
    return 0


# --- main -------------------------------------------------------------------


def main(
    config_path: Path = Path("config.yaml"),
    item_limit: int | None = None,
    *,
    self_test: bool = False,
    yes: bool = False,
) -> None:
    """Prepare MegaAcceptability data and derive a per-annotator 2AFC training set.

    Parameters
    ----------
    config_path
        Path to the gallery configuration file.
    item_limit
        Optional cap on the number of CSV rows to read (for quick testing).
    self_test
        When True, run the in-memory pairing self-test and exit.
    yes
        Skip overwrite confirmation prompts for non-interactive use.
    """
    if self_test:
        sys.exit(run_self_test())

    base_dir = Path(__file__).parent
    if not config_path.exists():
        config_path = base_dir / config_path.name
    if not config_path.exists():
        print_error(f"Config file not found: {config_path}")
        sys.exit(1)

    config = load_config(config_path)
    section = get_acceptability_config(config)
    columns = resolve_columns(section)
    scope = resolve_scope(section)
    drop_ties = bool(section.get("drop_ties", True))
    seed = int(section.get("seed", DEFAULT_SEED))
    max_per_annotator = section.get("max_pairs_per_annotator")
    if max_per_annotator is not None:
        max_per_annotator = int(max_per_annotator)

    paths_section = section.get("paths")
    paths_section = paths_section if isinstance(paths_section, dict) else {}
    base_paths = config.get("paths")
    base_paths = base_paths if isinstance(base_paths, dict) else {}

    dev_fraction = float(section.get("dev_fraction", 0.15))

    cache_dir = base_dir / str(base_paths.get("cache_dir", ".cache"))
    training_items_path = base_dir / str(
        paths_section.get("training_items", "items/megaacceptability_2afc.jsonl")
    )
    dev_items_path = base_dir / str(
        paths_section.get("dev_items", "items/megaacceptability_2afc_dev.jsonl")
    )
    raw_corpus_dir = base_dir / str(
        paths_section.get("raw_corpus_dir", "items/megaacceptability_raw_corpus")
    )
    training_corpus_dir = base_dir / str(
        paths_section.get("training_corpus_dir", "items/megaacceptability_2afc_corpus")
    )

    source_path_value = section.get("source_path")
    source_path = base_dir / str(source_path_value) if source_path_value else None
    source_url = section.get("source_url")
    source_url = str(source_url) if source_url else None

    print_header("MegaAcceptability Preparation")
    console.print(f"Base directory: [cyan]{base_dir}[/cyan]")
    console.print(f"Pairing scope: [cyan]{scope}[/cyan]")
    console.print(f"Output items: [cyan]{training_items_path}[/cyan]\n")

    if training_items_path.exists() and not yes:
        if not confirm(f"Overwrite {training_items_path}?", default=False):
            print_info("Operation cancelled.")
            return

    if item_limit:
        print_warning(f"Test mode: limiting to {item_limit:,} rows\n")

    # 1. Resolve and load the raw dataset
    print_header("1/4 Loading MegaAcceptability")
    csv_path = ensure_source_csv(
        source_path=source_path, source_url=source_url, cache_dir=cache_dir
    )
    if csv_path is None:
        sys.exit(1)
    try:
        rows = load_rating_rows(csv_path, columns=columns, limit=item_limit)
    except Exception as exc:  # noqa: BLE001
        print_error(f"Failed to read {csv_path}: {exc}")
        sys.exit(1)
    if not rows:
        print_error(f"No usable rows read from {csv_path}.")
        sys.exit(1)
    n_annotators = len({row.participant_id for row in rows})
    print_success(f"Loaded {len(rows):,} ratings from {n_annotators:,} annotators\n")

    # 2. Emit the raw dataset as a layers corpus (guarded)
    print_header("2/4 Emitting Raw Layers Corpus")
    emit_layers_corpus(
        build_raw_items(rows),
        corpus_name="megaacceptability_raw",
        out_dir=raw_corpus_dir,
    )
    console.print()

    # 3. Split by sentence (dev sentences are unseen during training), then
    #    derive per-annotator 2AFC pairs within each split
    print_header("3/4 Building Per-Annotator 2AFC Pairs")
    train_rows, dev_rows = split_rows_by_sentence(rows, dev_fraction, seed)
    with console.status("[bold]Pairing within-rater sentences...[/bold]"):
        pair_items = build_per_annotator_pairs(
            train_rows,
            scope=scope,
            drop_ties=drop_ties,
            max_per_annotator=max_per_annotator,
            seed=seed,
        )
        dev_items = build_per_annotator_pairs(
            dev_rows,
            scope=scope,
            drop_ties=drop_ties,
            max_per_annotator=max_per_annotator,
            seed=seed,
        )
    if not pair_items:
        print_error("No 2AFC pairs were derived. Exiting.")
        sys.exit(1)
    print_success(
        f"Derived {len(pair_items):,} train and {len(dev_items):,} dev items "
        f"(dev sentences held out)\n"
    )

    # 4. Persist the derived train and dev sets (JSONL + layers corpus)
    print_header("4/4 Persisting Derived Training Set")
    try:
        write_items_jsonl(pair_items, training_items_path)
        print_success(f"Wrote {len(pair_items):,} train items to {training_items_path}")
        if dev_items:
            write_items_jsonl(dev_items, dev_items_path)
            print_success(f"Wrote {len(dev_items):,} dev items to {dev_items_path}")
    except Exception as exc:  # noqa: BLE001
        print_error(f"Failed to write items: {exc}")
        sys.exit(1)
    emit_layers_corpus(
        pair_items,
        corpus_name="megaacceptability_2afc",
        out_dir=training_corpus_dir,
    )
    console.print()

    # Summary
    print_header("Summary")
    n_option_a = sum(
        1 for item in pair_items if item.item_metadata.get("label") == OPTION_A
    )
    n_option_b = len(pair_items) - n_option_a
    table = create_summary_table(
        {
            "Ratings loaded": f"{len(rows):,}",
            "Annotators": f"{n_annotators:,}",
            "Derived pairs": f"{len(pair_items):,}",
            "Gold option_a / option_b": f"{n_option_a:,} / {n_option_b:,}",
            "Training items (JSONL)": str(training_items_path),
        }
    )
    console.print(table)
    print_info("Next: run train_acceptability_model.py to fit the initial model")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Prepare MegaAcceptability and derive per-annotator 2AFC pairs"
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
        help="Limit number of CSV rows to read (default: all)",
    )
    parser.add_argument(
        "--self-test",
        action="store_true",
        help="Run the in-memory pairing self-test and exit (no network or IO)",
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
        self_test=args.self_test,
        yes=args.yes,
    )
