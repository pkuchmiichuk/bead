"""Lossless, law-verified lenses between bead models and the ``layers`` schema.

Maps bead's corpus and annotation models to ``layers``-shaped JSON and back via
didactic lenses (``dx.Lens``/``dx.Iso``): the layers view is a faithful,
standalone projection; the lens complement holds the bead-only round-trip
remainder. Round-trip fidelity is guaranteed by the didactic GetPut/PutGet laws.
"""

from __future__ import annotations

from bead.interop.layers.bridges import (
    RECORD_EXPRESSION,
    RecordExpressionLens,
    record_to_expression,
)
from bead.interop.layers.graph_lens import (
    CORPUS_GRAPH_LAYERS,
    CorpusGraphLayersLens,
    graph_to_layers,
)
from bead.interop.layers.model_lenses import (
    ALL_MIRROR_ISOS,
    RECORD_ISOS,
    RECORD_MODELS,
    SHARED_DEF_ISOS,
    SHARED_DEF_MODELS,
    MirrorIso,
    mirror_iso,
)
from bead.interop.layers.parse_lens import (
    PARSED_SENTENCE_LAYERS,
    ParsedSentenceLayersIso,
    parse_to_layers,
)

__all__ = [
    "ALL_MIRROR_ISOS",
    "CORPUS_GRAPH_LAYERS",
    "PARSED_SENTENCE_LAYERS",
    "RECORD_EXPRESSION",
    "RECORD_ISOS",
    "RECORD_MODELS",
    "SHARED_DEF_ISOS",
    "SHARED_DEF_MODELS",
    "CorpusGraphLayersLens",
    "MirrorIso",
    "ParsedSentenceLayersIso",
    "RecordExpressionLens",
    "graph_to_layers",
    "mirror_iso",
    "parse_to_layers",
    "record_to_expression",
]
