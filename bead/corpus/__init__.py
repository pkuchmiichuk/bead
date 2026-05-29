"""Streaming corpus ingestion and structural sampling.

Turns raw external text (JSONL, optionally Zstandard-compressed; CSV/TSV) into
structurally filtered experimental ``Item``s: stream ``CorpusRecord``s from a
``CorpusSource``, dependency-parse them, and keep only those whose parse
satisfies a structural DSL constraint.
"""

from __future__ import annotations

from bead.corpus.base import CorpusSource
from bead.corpus.pipeline import (
    filter_by_structure,
    parse_records,
    record_to_item,
    sample_corpus,
)
from bead.corpus.records import CorpusRecord, ProvenanceValue
from bead.corpus.sources import CsvCorpusSource, JsonlCorpusSource

__all__ = [
    "CorpusRecord",
    "CorpusSource",
    "CsvCorpusSource",
    "JsonlCorpusSource",
    "ProvenanceValue",
    "filter_by_structure",
    "parse_records",
    "record_to_item",
    "sample_corpus",
]
