#!/usr/bin/env python3
"""Write the verb frequency table used to choose which verbs to test.

Run once, with wordfreq supplied for the invocation::

    uv run --project <repo root> --with wordfreq python build_frequencies.py

Frequency is the Zipf value of the infinitive, from the wordfreq package
(Ukrainian, which wordfreq keys by the ISO 639-1 code "uk"). Summing over a
verb's whole paradigm was tried and rejected: infinitives in -ти are distinctive,
while finite forms collide with frequent homographs, so колоти collects the count
of the conjunction коли and маяти collects має from мати.

The generated CSV is committed so the pipeline needs no wordfreq dependency and
the ranking stays fixed across runs. Unattested lemmas are omitted and read
as 0.0.
"""

from __future__ import annotations

import csv
from pathlib import Path

from wordfreq import zipf_frequency

from bead.cli.display import print_header, print_success
from bead.resources.lexicon import Lexicon

BASE_DIR = Path(__file__).parent
WORDFREQ_LANGUAGE = "uk"


def main() -> None:
    """Rank every verb lemma by the Zipf frequency of its infinitive."""
    print_header("Verb Frequencies (wordfreq)")

    lexicon = Lexicon.from_jsonl(str(BASE_DIR / "lexicons" / "verbs.jsonl"), "verbs")
    lemmas = {item.lemma for item in lexicon.items}
    attested = sorted(
        (
            (lemma, zipf_frequency(lemma, WORDFREQ_LANGUAGE))
            for lemma in lemmas
        ),
        key=lambda row: (-row[1], row[0]),
    )
    attested = [(lemma, zipf) for lemma, zipf in attested if zipf > 0.0]

    output_path = BASE_DIR / "resources" / "verb_frequencies.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(["lemma", "zipf"])
        writer.writerows((lemma, f"{zipf:.2f}") for lemma, zipf in attested)

    print_success(
        f"Wrote {len(attested):,} attested lemmas of {len(lemmas):,} to {output_path}"
    )


if __name__ == "__main__":
    main()
