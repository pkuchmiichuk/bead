"""Lossless, law-verified lenses between bead models and the ``layers`` schema.

Maps bead's corpus and annotation models to ``layers``-shaped JSON and back via
didactic lenses (``dx.Lens``/``dx.Iso``): the layers view is a faithful,
standalone projection; the lens complement holds the bead-only round-trip
remainder. Round-trip fidelity is guaranteed by the didactic GetPut/PutGet laws.

Coverage:

- Every linguistic ``layers`` construct (shared defs + records) is mirrored as a
  faithful didactic model with a generic ``MirrorIso`` (see ``models`` /
  ``models_records`` / ``model_lenses``).
- bead's pipeline outputs bridge directly: ``CorpusGraph`` <-> the property
  graph, ``CorpusRecord`` <-> ``expression``, a dependency parse <->
  ``tokenization`` + annotation layers.
- The resource overlap is mapped over bead's existing models:
  ``LexicalItem`` <-> ``entry``, ``Lexicon`` <-> ``collection``, ``Template``
  <-> ``template`` (see ``resource_lens``).

The remaining experiment/publishing overlaps are intentionally NOT mapped: a
feasibility review found ``persona`` orthogonal to bead participants (who took
part vs. how one annotates), and ``judgment``, ``corpus``, and ``changelog``
too divergent from bead's response, list, and changelog representations to yield
a faithful layers view. Their data is reachable through the constructs above.
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
from bead.interop.layers.resource_lens import (
    LEXICAL_ITEM_ENTRY,
    LEXICON_COLLECTION,
    TEMPLATE_LAYERS,
    LexicalItemEntryLens,
    LexiconCollectionLens,
    TemplateLayersLens,
)

__all__ = [
    "ALL_MIRROR_ISOS",
    "CORPUS_GRAPH_LAYERS",
    "LEXICAL_ITEM_ENTRY",
    "LEXICON_COLLECTION",
    "PARSED_SENTENCE_LAYERS",
    "RECORD_EXPRESSION",
    "TEMPLATE_LAYERS",
    "LexicalItemEntryLens",
    "LexiconCollectionLens",
    "TemplateLayersLens",
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
