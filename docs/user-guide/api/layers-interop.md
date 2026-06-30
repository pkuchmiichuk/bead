# Layers Interoperability

bead maps its corpus, annotation, and resource data to the
[layers](https://github.com/layers-pub/layers) linguistic-annotation schema and
back **losslessly**, via didactic lenses (`dx.Iso` / `dx.Lens`). The forward
direction produces a faithful, standalone projection built from the canonical
`layers` record models that [`lairs`](https://pypi.org/project/lairs/) generates
from the layers lexicons; the reverse reconstructs the exact bead value. Because
the mappings are lenses, the round-trip is guaranteed by the didactic
GetPut/PutGet laws (verified in the test suite).

bead depends on `lairs` directly, so the layers record models come from one
canonical source (`lairs.records`) rather than a hand-maintained copy. Importing
`bead` does not import `lairs`; the dependency loads only when you reach into
`bead.interop.layers`.

## What is covered

bead maps its pipeline outputs and resources onto the canonical `lairs.records`
models (`expression`, `segmentation`, `annotation`, `graph`, `resource`, and the
shared `defs` objects):

- `CorpusRecord` to a layers `expression`.
- `CorpusGraph` to the layers property graph (expressions, graph nodes, and a
  `graphEdgeSet`), bundled as a `CorpusGraphLayers` view.
- a dependency `ParsedSentence` to a layers `tokenization` plus part-of-speech
  and dependency annotation layers, bundled as a `ParsedSentenceLayers` view.
- an `Item`'s span and relation annotations to span and relation
  `AnnotationLayer` records over per-element expressions and tokenizations.
- bead resources to their layers counterparts: `LexicalItem` to an `entry`,
  `Lexicon` to a `collection`, and `Template` to a `template`.

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

# Faithful, standalone layers projection of canonical lairs models.
view = graph_to_layers(graph)
assert view.expressions[0].kind == "expression"
assert view.edge_set.edges[0].edgeType == "reply-to"

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
assert view.dependency_layer.subkind == "dependency"
# The root token is encoded with headIndex -1 (the layers convention).
assert view.dependency_layer.annotations[1].headIndex == -1
# Iso: the parse reconstructs exactly (no complement needed).
assert PARSED_SENTENCE_LAYERS.backward(view) == sentence
```

The layers `token` has no space-after slot, so each token's `space_after` flag
travels in its part-of-speech annotation's features.

## Mapping an item's spans

An `Item`'s standoff spans and relations project to span and relation
`AnnotationLayer` records. A span anchors by `tokenRefSequence` (its
`head_index` becomes the sequence's `anchorTokenIndex`, and a Wikidata
`label_id` becomes a `knowledgeRef`); a relation carries `ArgumentRef` source and
target arguments.

```python
from uuid import uuid4

from bead.interop.layers import ITEM_LAYERS, item_to_layers
from bead.items.item import Item
from bead.items.spans import Span, SpanLabel, SpanSegment

item = Item(
    item_template_id=uuid4(),
    rendered_elements={"text": "Einstein won"},
    tokenized_elements={"text": ("Einstein", "won")},
    spans=(
        Span(
            span_id="s1",
            segments=(SpanSegment(element_name="text", indices=(0,)),),
            label=SpanLabel(label="PERSON", label_id="Q937"),
        ),
    ),
)

fragment = item_to_layers(item)  # a lairs CorpusFragment of canonical records
layers_view, complement = ITEM_LAYERS.forward(item)
assert ITEM_LAYERS.backward(layers_view, complement) == item
```

## Using bead as a lairs codec

bead registers a `bead` codec on the `lairs.codecs` entry point, so any tool with
both packages installed can round-trip a bead `ItemCollection` through the
`layers` schema:

```python
from uuid import uuid4

import lairs

from bead.items.item import Item, ItemCollection

collection = ItemCollection(
    name="study-1",
    source_template_collection_id=uuid4(),
    source_filled_collection_id=uuid4(),
    items=(Item(item_template_id=uuid4(), rendered_elements={"text": "dogs bark"}),),
)

codec = lairs.codec("bead")()
fragment = codec.decode(collection.model_dump_json())
assert codec.encode(fragment.records) == collection.model_dump_json()  # lossless
```

## Loading and emitting a corpus

`bead.interop.layers.corpus_io` ingests a `lairs.data.Corpus` into bead models
and emits bead data as a corpus:

```python
from pathlib import Path
from uuid import uuid4

from bead.interop.layers import corpus_io
from bead.items.item import Item, ItemCollection

collection = ItemCollection(
    name="study-1",
    source_template_collection_id=uuid4(),
    source_filled_collection_id=uuid4(),
    items=(Item(item_template_id=uuid4(), rendered_elements={"text": "dogs bark"}),),
)

corpus = corpus_io.items_to_corpus(collection, corpus_name="study-1")
paths = corpus_io.materialize_corpus(corpus, Path("corpus_views"))  # Arrow/Parquet
revision = corpus_io.save_corpus_repo(corpus, Path("corpus_repo"))  # VCS commit

graph = corpus_io.corpus_to_graph(corpus)  # parentRef -> parent edges
records = list(corpus_io.corpus_to_records(corpus))
```

The same operations are available from the command line:

```console
$ bead layers encode items.json --out fragment.json
$ bead layers decode fragment.json --out items.json
$ bead layers materialize items.json --out corpus/
```

The PDS publish path (`corpus_io.publish_corpus` and `bead layers publish`) is
opt-in and defaults to a dry run.

## Validation

The lenses construct real `lairs.records` models, which validate their structure,
required fields, and types on construction. Conformance to the layers lexicons is
owned upstream by `lairs`, which generates those models from the lexicons and
tests them in its own suite, so bead does not re-host a separate validator.
