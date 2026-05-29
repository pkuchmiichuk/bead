# Layers Interoperability

bead maps its corpus and annotation data to the
[layers](https://github.com/layers-pub/layers) linguistic-annotation schema and
back **losslessly**, via didactic lenses (`dx.Iso` / `dx.Lens`). The forward
direction produces faithful, standalone layers-shaped JSON; the reverse
reconstructs the exact bead value. Because the mappings are lenses, the
round-trip is guaranteed by the didactic GetPut/PutGet laws (verified in the
test suite with `verify_iso` / `check_lens_laws`).

There is no ATProto wire/network dependency: the lenses produce and consume
plain layers-shaped Python/JSON.

## What is covered

- Every linguistic `pub.layers` construct is mirrored as a faithful didactic
  model in `bead.interop.layers.models` / `models_records` (the anchor union,
  temporal/spatial expressions, token/text/page/external anchors, the
  polymorphic annotation and annotation layer, the property graph, media
  descriptors, ontology type definitions, knowledge references, and the shared
  objects). Each has a lossless `MirrorIso` to layers JSON.
- bead's own pipeline outputs bridge directly to layers:
  - `CorpusGraph` ↔ a layers property graph (expressions + graph nodes + a
    `graphEdgeSet`).
  - `CorpusRecord` ↔ a layers `expression`.
  - a dependency `ParsedSentence` ↔ a layers `tokenization` plus part-of-speech
    and dependency annotation layers.

## Mapping a corpus graph

```python
from bead.corpus.assemble import EdgeSpec, assemble_graph
from bead.corpus.records import CorpusRecord
from bead.interop.layers import CORPUS_GRAPH_LAYERS, graph_to_layers

records = [
    CorpusRecord(text="the submission", source_name="r", provenance={"id": "sub"}),
    CorpusRecord(
        text="a reply",
        source_name="r",
        provenance={"id": "c1", "parent_id": "t3_sub"},
    ),
]
graph = assemble_graph(
    records,
    node_id_field="id",
    edge_specs=[
        EdgeSpec(
            target_field="parent_id", edge_type="reply-to", strip_prefixes=("t3_",)
        )
    ],
)

# Faithful, standalone layers-shaped projection.
view = graph_to_layers(graph)
assert set(view) == {"expressions", "graphNodes", "graphEdgeSet"}

# Lossless round-trip via the lens (view + complement reconstruct exactly).
layers_view, complement = CORPUS_GRAPH_LAYERS.forward(graph)
assert CORPUS_GRAPH_LAYERS.backward(layers_view, complement) == graph
```

## Mapping a dependency parse

```python
from bead.interop.layers import PARSED_SENTENCE_LAYERS, parse_to_layers
from bead.tokenization.parsers import ParsedSentence, ParsedToken

sentence = ParsedSentence(
    original_text="dogs bark",
    tokens=(
        ParsedToken(
            index=0,
            text="dogs",
            upos="NOUN",
            deprel="nsubj",
            head=1,
            start_char=0,
            end_char=4,
        ),
        ParsedToken(
            index=1,
            text="bark",
            upos="VERB",
            deprel="root",
            head=None,
            start_char=5,
            end_char=9,
        ),
    ),
)

view = parse_to_layers(sentence)
assert view["dependencyLayer"]["subkind"] == "dependency"
# The root token is encoded with headIndex -1 (the layers convention).
assert view["dependencyLayer"]["annotations"][1]["headIndex"] == -1
# Iso: the parse reconstructs exactly (no complement needed).
assert PARSED_SENTENCE_LAYERS.backward(view) == sentence
```

## Working with the mirror models directly

Any layers construct can be built as a bead model and serialized to/from layers
JSON with its `MirrorIso`:

```python
from bead.interop.layers import mirror_iso
from bead.interop.layers.models import Anchor, LayersUuid, TokenRef

anchor = Anchor(
    token_ref=TokenRef(tokenization_id=LayersUuid(value="tok-1"), token_index=4)
)
iso = mirror_iso(Anchor)

layers_json = iso.forward(anchor)
assert layers_json["tokenRef"]["tokenIndex"] == 4  # camelCased, layers-shaped
assert iso.backward(layers_json) == anchor  # exact round-trip
```

`bead.interop.layers.ALL_MIRROR_ISOS` maps every mirror model type to its iso,
and a coverage test guards that every targeted layers construct has a
law-passing mapping.

## Validating against the layers lexicons

The mappings are checked against the canonical layers lexicons, vendored as the
`vendor/layers` git submodule pointing at
[`layers-pub/layers`](https://github.com/layers-pub/layers). The interop test
suite feeds every mapping's output through the ATProto lexicon validator
(`@atproto/lexicon`) and asserts each record validates against its lexicon, so a
schema drift in layers surfaces as a failing test.

Fetch the lexicons with `git submodule update --init vendor/layers`, and pull the
latest published schemas with `git submodule update --remote vendor/layers`. The
validation tests skip when the submodule is not checked out.
