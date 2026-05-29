# Corpus Ingestion

The `bead.corpus` package turns raw text corpora into experimental `Item`s. You
stream records from a source, dependency-parse them, and keep only those whose
syntactic structure matches a constraint. This is the natural way to build
naturalistic stimuli (for example, transitive-verb sentences drawn from a large
corpus) that then flow into the rest of the pipeline (items, lists, deployment).

## Installation

```bash
# Streaming sources, including .zst corpora
uv sync --extra corpus

# Dependency parsing (spaCy, Stanza)
uv sync --extra tokenization

# Structural sampling needs both
uv sync --extra corpus --extra tokenization
```

## Sources

A `CorpusSource` streams `CorpusRecord`s, each carrying `text`, a `source_name`,
a `record_index`, and a flat `provenance` dict.

```python
from bead.corpus import JsonlCorpusSource, CsvCorpusSource

# JSON Lines, transparently decompressing .jsonl.zst
reddit = JsonlCorpusSource(
    "comments.jsonl.zst",
    text_field="body",
    provenance_fields=("author", "subreddit", "score"),
)

# CSV / TSV
items = CsvCorpusSource(
    "sentences.csv",
    text_column="sentence",
    provenance_columns=("verb", "frequency"),
)

for record in reddit:
    print(record.text, record.provenance["author"])
```

Sources are lazy iterators, so multi-gigabyte corpora are never loaded into
memory.

## Structural Sampling

`sample_corpus` streams a source through a dependency parser and yields only the
items whose parse satisfies a structural DSL constraint. The constraint is a
normal bead DSL expression with the item bound as `self`, using the structural
builtins (`root`, `dependents`, `upos`, `head`, `has_relation`, ...).

```python
from uuid import uuid4
from bead.corpus import JsonlCorpusSource, sample_corpus
from bead.tokenization.parsers import StanzaParser

source = JsonlCorpusSource("comments.jsonl", text_field="body")
parser = StanzaParser(language="en")

# Keep only sentences whose root verb takes a direct object.
constraint = (
    'upos(self, root(self)) == "VERB" '
    'and len(dependents(self, root(self), "obj")) > 0'
)

items = list(
    sample_corpus(
        source,
        parser,
        constraint,
        item_template_id=uuid4(),
        limit=200,
    )
)
```

Each resulting `Item` carries the parse as standoff annotations: one token-level
`Span` per token (with its governor as `head_index` and its POS, lemma, deprel,
morphology, and character offsets in `span_metadata`) and one directed
head-to-dependent `SpanRelation` per syntactic arc. The record's provenance plus
the parser tool and formalism are recorded on `item.item_metadata`.

## Composing the Pipeline by Hand

`sample_corpus` is a convenience wrapper. The underlying generators can be
composed directly when you want to inspect or transform intermediate results:

```python
from bead.corpus import parse_records, filter_by_structure

pairs = parse_records(source, parser, split_sentences=True)
items = filter_by_structure(pairs, constraint, item_template_id=uuid4(), tool=parser.tool)
```

`parse_records` yields one `(record, sentence)` pair per sentence; set
`split_sentences=False` to keep only records that parse to a single sentence.

## Cleaning Source Text

Web and forum text often needs cleanup before parsing. The text transforms in
`bead.transforms` help:

```python
from bead.transforms.base import TransformContext
from bead.transforms.text import RedditCleanupTransform, split_sentences

clean = RedditCleanupTransform()
text = clean("see [the thread](http://x) &amp; more", TransformContext())
# -> "see the thread & more"

sentences = split_sentences("First one. Second one.")
# -> ("First one.", "Second one.")
```

## Generated Corpora

A language model can also act as a corpus source via `CompletionCorpusSource`,
which wraps any adapter implementing the `TextGenerator` protocol (for example
the OpenAI or Anthropic adapters):

```python
from bead.corpus import CompletionCorpusSource
from bead.items.adapters import OpenAIAdapter  # requires the `api` extra

generator = OpenAIAdapter(model_name="gpt-4o", cache=...)
source = CompletionCorpusSource(
    generator,
    prompts=["Write a sentence about a cat.", "Write a sentence about a dog."],
    completions_per_prompt=5,
)
```
