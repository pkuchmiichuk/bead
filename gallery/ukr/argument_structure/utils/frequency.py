"""Verb frequency lookup.

Frequencies are read from ``resources/verb_frequencies.csv``, a committed table
derived from the wordfreq package. Keeping it as data means the pipeline needs no
wordfreq dependency and the ranking stays fixed across runs. Regenerate it with
``build_frequencies.py``.
"""

from __future__ import annotations

import csv
from collections.abc import Iterable
from functools import cache
from pathlib import Path

FREQUENCY_PATH = Path(__file__).parent.parent / "resources" / "verb_frequencies.csv"


@cache
def _frequencies() -> dict[str, float]:
    """Return the lemma to Zipf frequency table, read once.

    Returns
    -------
    dict[str, float]
        Zipf frequency keyed by lemma, for attested lemmas only.
    """
    with FREQUENCY_PATH.open(encoding="utf-8") as f:
        return {row["lemma"]: float(row["zipf"]) for row in csv.DictReader(f)}


def zipf(lemma: str) -> float:
    """Return the Zipf frequency of a lemma.

    Parameters
    ----------
    lemma : str
        Dictionary form to look up.

    Returns
    -------
    float
        Zipf frequency, or 0.0 when the lemma is unattested.
    """
    return _frequencies().get(lemma, 0.0)


def most_frequent(lemmas: Iterable[str], limit: int) -> list[str]:
    """Return the most frequent lemmas, most frequent first.

    Parameters
    ----------
    lemmas : Iterable[str]
        Lemmas to rank; duplicates are ignored.
    limit : int
        Number of lemmas to keep.

    Returns
    -------
    list[str]
        At most ``limit`` lemmas, ordered by decreasing frequency then
        alphabetically so ties are stable.
    """
    return sorted(set(lemmas), key=lambda lemma: (-zipf(lemma), lemma))[:limit]
