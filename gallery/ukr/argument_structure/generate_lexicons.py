#!/usr/bin/env python3
"""Generate the Ukrainian verb lexicon from VESUM UniMorph data.

Writes ``lexicons/verbs.jsonl`` with one present-tense 3rd-person-singular form
per imperfective verb (perfective verbs have no present tense in Ukrainian).
"""

from __future__ import annotations

import argparse
from pathlib import Path

from utils.vesum_adapter import VesumUniMorphAdapter

from bead.cli.display import (
    create_live_status,
    display_file_stats,
    print_header,
    print_info,
    print_success,
)
from bead.resources.lexical_item import LexicalItem
from bead.resources.lexicon import Lexicon

LANGUAGE = "ukr"


def build_verb_lexicon(limit: int | None = None) -> list[LexicalItem]:
    """Return one present-3sg finite indicative form per imperfective verb.

    Parameters
    ----------
    limit : int | None
        Stop after this many unique verb lemmas. None means all.

    Returns
    -------
    list[LexicalItem]
        One item per verb, carrying parsed morphological features.
    """
    adapter = VesumUniMorphAdapter()
    with create_live_status("Loading VESUM data (ukr.xz, ~7.5M rows)..."):
        frame = adapter._load_dataset(LANGUAGE)

    # Narrow the frame before building items to avoid materializing every row.
    features = frame["features"].astype(str)
    candidates = frame[
        features.str.startswith("V;") & features.str.contains("PRS", regex=False)
    ]

    items: list[LexicalItem] = []
    seen: set[str] = set()
    for row in candidates.itertuples(index=False):
        parsed = adapter._parse_features(str(row.features))
        keep = (
            parsed.get("pos") == "V"
            and parsed.get("tense") == "PRS"
            and parsed.get("person") == "3"
            and parsed.get("number") == "SG"
            and parsed.get("mood") != "IMP"
            and parsed.get("finiteness") != "NFIN"
        )
        if not keep:
            continue
        lemma = str(row.lemma)
        if lemma in seen:
            continue
        seen.add(lemma)
        items.append(
            LexicalItem(
                lemma=lemma,
                form=str(row.form),
                language_code=LANGUAGE,
                features=parsed,
                source="UniMorph:ukr.xz",
            )
        )
        if limit is not None and len(items) >= limit:
            break
    return items


def main(limit: int | None = None) -> None:
    """Generate the verb lexicon and write it to ``lexicons/verbs.jsonl``.

    Parameters
    ----------
    limit : int | None
        Truncate to this many unique verb lemmas. None means all.
    """
    print_header("Verb Lexicon (VESUM)")
    if limit is not None:
        print_info(f"Limiting to {limit} verb lemmas")

    items = build_verb_lexicon(limit)
    print_success(f"Selected {len(items):,} present-3sg verb forms")

    lexicon = Lexicon(name="verbs", language_code=LANGUAGE).with_items(items)

    output_path = Path(__file__).parent / "lexicons" / "verbs.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lexicon.to_jsonl(str(output_path))
    display_file_stats(output_path, len(items), "verbs")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the Ukrainian verb lexicon from VESUM UniMorph data."
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of verb lemmas (for testing).",
    )
    args = parser.parse_args()
    main(limit=args.limit)
