#!/usr/bin/env python3
"""Generate the Ukrainian lexicons from VESUM UniMorph data.

Writes two files:

``lexicons/verbs.jsonl``
    One present-tense 3rd-person-singular form per imperfective verb
    (perfective verbs have no present tense in Ukrainian).

``lexicons/bleached_nouns.jsonl``
    A small hand-authored set of bleached nouns, expanded from UniMorph into one
    form per case each noun's role uses, tagged with ``semantic_class`` and
    ``role``.

The VESUM file (~7.5M rows) is parsed once and both lexicons are built from that
single in-memory frame.

Object forms are chosen to be unambiguous: for each case a form is preferred
whose surface realizes no other case that could be read as an object. Two nouns
have no such form for the genitive and are kept deliberately: людина (людини is
also accusative plural) and випадок (випадку is also dative singular). Their
genitive carries a ``case_collides_with`` feature naming the competing reading.
"""

from __future__ import annotations

import argparse
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import pandas as pd
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


@dataclass(frozen=True)
class NounSpec:
    """A hand-authored bleached noun and its experimental annotation.

    Parameters
    ----------
    lemma : str
        Nominative-singular dictionary form.
    role : str
        Argument role the noun fills (``"subject"`` or ``"object"``).
    semantic_class : str
        Coarse meaning label carried through the pipeline.
    gloss : str
        English gloss, carried through as ``eng_gloss``.
    """

    lemma: str
    role: str
    semantic_class: str
    gloss: str


# One subject and a set of objects spanning the core meaning classes. The
# subject appears only in the nominative; objects in each case a frame governs.
NOUN_SPECS: tuple[NounSpec, ...] = (
    NounSpec("людина", "subject", "animate", "person"),
    NounSpec("гурт", "object", "group", "group"),
    NounSpec("людина", "object", "animate", "person"),
    NounSpec("інструмент", "object", "inanimate_object", "instrument"),
    NounSpec("зразок", "object", "abstract", "sample"),
    NounSpec("двір", "object", "location", "yard"),
    NounSpec("день", "object", "temporal", "day"),
    NounSpec("шматок", "object", "quantity", "piece"),
    NounSpec("випадок", "object", "event", "incident"),
)

CASES_BY_ROLE: dict[str, tuple[str, ...]] = {
    "subject": ("NOM",),
    "object": ("ACC", "GEN", "DAT", "INS"),
}

# Readings a bare form could carry in object position, used only to detect
# ambiguity. Plural cells count because a singular form that also spells a
# plural (людини is genitive singular and accusative plural) is ambiguous.
COMPETING_READINGS: frozenset[str] = frozenset(
    f"{case}.{number}"
    for case in ("ACC", "GEN", "DAT", "INS")
    for number in ("SG", "PL")
)


def build_verb_lexicon(
    adapter: VesumUniMorphAdapter, frame: pd.DataFrame, limit: int | None = None
) -> list[LexicalItem]:
    """Return one present-3sg finite indicative form per imperfective verb.

    Parameters
    ----------
    adapter : VesumUniMorphAdapter
        Adapter supplying the feature parser.
    frame : pd.DataFrame
        The parsed VESUM frame (``lemma``, ``form``, ``features``).
    limit : int | None
        Stop after this many unique verb lemmas. None means all.

    Returns
    -------
    list[LexicalItem]
        One item per verb, carrying parsed morphological features.
    """
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


def build_noun_lexicon(
    adapter: VesumUniMorphAdapter, frame: pd.DataFrame
) -> list[LexicalItem]:
    """Return the bleached noun forms for the cases each role's frames use.

    For every noun in :data:`NOUN_SPECS`, its singular forms are read from the
    frame and one form is kept per case the role governs, tagged with the parsed
    morphology plus ``semantic_class`` and ``role``.

    Parameters
    ----------
    adapter : VesumUniMorphAdapter
        Adapter supplying the feature parser.
    frame : pd.DataFrame
        The parsed VESUM frame (``lemma``, ``form``, ``features``).

    Returns
    -------
    list[LexicalItem]
        One item per (noun, case), carrying parsed features and annotation.
    """
    items: list[LexicalItem] = []
    for spec in NOUN_SPECS:
        wanted = CASES_BY_ROLE[spec.role]
        rows = frame[frame["lemma"] == spec.lemma]

        readings: dict[str, set[str]] = defaultdict(set)
        candidates: dict[str, list[str]] = defaultdict(list)
        parsed_by_form: dict[tuple[str, str], dict[str, str]] = {}
        for row in rows.itertuples(index=False):
            parsed = adapter._parse_features(str(row.features))
            case_name, number = parsed.get("case"), parsed.get("number")
            if parsed.get("pos") != "N" or case_name is None or number is None:
                continue
            form = str(row.form)
            readings[form].add(f"{case_name}.{number}")
            if number == "SG" and case_name in wanted:
                if form not in candidates[case_name]:
                    candidates[case_name].append(form)
                parsed_by_form[case_name, form] = parsed

        for case_name in wanted:
            forms = candidates.get(case_name)
            if not forms:
                continue
            cell = f"{case_name}.SG"
            unambiguous = [
                form for form in forms if readings[form] & COMPETING_READINGS == {cell}
            ]
            form = unambiguous[0] if unambiguous else forms[0]

            features = dict(parsed_by_form[case_name, form])
            features["semantic_class"] = spec.semantic_class
            features["role"] = spec.role
            features["eng_gloss"] = spec.gloss
            collisions = sorted((readings[form] & COMPETING_READINGS) - {cell})
            if collisions:
                features["case_collides_with"] = ",".join(collisions)
            items.append(
                LexicalItem(
                    lemma=spec.lemma,
                    form=form,
                    language_code=LANGUAGE,
                    features=features,
                    source="UniMorph:ukr.xz",
                )
            )
    return items


def generate_verbs(
    adapter: VesumUniMorphAdapter, frame: pd.DataFrame, limit: int | None = None
) -> None:
    """Build the verb lexicon and write ``lexicons/verbs.jsonl``.

    Parameters
    ----------
    adapter : VesumUniMorphAdapter
        Adapter supplying the feature parser.
    frame : pd.DataFrame
        The parsed VESUM frame.
    limit : int | None
        Truncate to this many unique verb lemmas. None means all.
    """
    print_header("Verb Lexicon (VESUM)")
    if limit is not None:
        print_info(f"Limiting to {limit} verb lemmas")

    items = build_verb_lexicon(adapter, frame, limit)
    print_success(f"Selected {len(items):,} present-3sg verb forms")

    lexicon = Lexicon(name="verbs", language_code=LANGUAGE).with_items(items)
    output_path = Path(__file__).parent / "lexicons" / "verbs.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lexicon.to_jsonl(str(output_path))
    display_file_stats(output_path, len(items), "verbs")


def generate_nouns(adapter: VesumUniMorphAdapter, frame: pd.DataFrame) -> None:
    """Build the bleached noun lexicon and write ``lexicons/bleached_nouns.jsonl``.

    Parameters
    ----------
    adapter : VesumUniMorphAdapter
        Adapter supplying the feature parser.
    frame : pd.DataFrame
        The parsed VESUM frame.
    """
    print_header("Bleached Noun Lexicon (VESUM)")

    items = build_noun_lexicon(adapter, frame)
    print_success(f"Selected {len(items)} noun forms across {len(NOUN_SPECS)} lemmas")
    for item in items:
        f = item.features
        print_info(
            f"{item.lemma} -> {item.form} "
            f"[{f.get('case')}/{f.get('role')}/{f.get('semantic_class')}]"
        )

    lexicon = Lexicon(name="bleached_nouns", language_code=LANGUAGE).with_items(items)
    output_path = Path(__file__).parent / "lexicons" / "bleached_nouns.jsonl"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lexicon.to_jsonl(str(output_path))
    display_file_stats(output_path, len(items), "noun forms")


def main(target: str = "all", limit: int | None = None) -> None:
    """Generate the requested lexicons from a single VESUM load.

    Parameters
    ----------
    target : str
        Which lexicon to build: ``"verbs"``, ``"nouns"``, or ``"all"``.
    limit : int | None
        Truncate the verb lexicon to this many lemmas. None means all.
    """
    adapter = VesumUniMorphAdapter()
    with create_live_status("Loading VESUM data (ukr.xz, ~7.5M rows)..."):
        frame = adapter._load_dataset(LANGUAGE)

    if target in ("verbs", "all"):
        generate_verbs(adapter, frame, limit)
    if target in ("nouns", "all"):
        generate_nouns(adapter, frame)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Generate the Ukrainian lexicons from VESUM UniMorph data."
    )
    parser.add_argument(
        "--target",
        choices=("verbs", "nouns", "all"),
        default="all",
        help="Which lexicon to generate (default: all).",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Limit the number of verb lemmas (for testing).",
    )
    args = parser.parse_args()
    main(target=args.target, limit=args.limit)
