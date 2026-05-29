"""Lossless, law-verified lenses between bead models and the ``layers`` schema.

Maps bead's corpus and annotation models to ``layers``-shaped JSON and back via
didactic lenses (``dx.Lens``/``dx.Iso``): the layers view is a faithful,
standalone projection; the lens complement holds the bead-only round-trip
remainder. Round-trip fidelity is guaranteed by the didactic GetPut/PutGet laws.
"""

from __future__ import annotations

from bead.interop.layers.graph_lens import (
    CORPUS_GRAPH_LAYERS,
    CorpusGraphLayersLens,
    graph_to_layers,
)

__all__ = [
    "CORPUS_GRAPH_LAYERS",
    "CorpusGraphLayersLens",
    "graph_to_layers",
]
