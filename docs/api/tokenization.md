# bead.tokenization

Configurable multilingual tokenization for span annotation and UI display.

## Configuration

::: bead.tokenization.config
    options:
      show_root_heading: true
      show_source: false

## Tokenizers

::: bead.tokenization.tokenizers
    options:
      show_root_heading: true
      show_source: false

## Dependency Parsing

Dependency parsers (spaCy, Stanza) produce a per-sentence `ParsedSentence` of
`ParsedToken` records, and `parse_to_spans` projects a parse onto the standoff
`Span` + `SpanRelation` models used by `bead.items.Item`: one single-token
`Span` per token (carrying its governor as `head_index` and its
`upos`/`xpos`/`lemma`/`deprel`/morphology plus character offsets in
`span_metadata`), and one directed head-to-dependent `SpanRelation` per
syntactic arc labeled with the dependency relation.

::: bead.tokenization.parsers
    options:
      show_root_heading: true
      show_source: false

## Display-to-Subword Alignment

::: bead.tokenization.alignment
    options:
      show_root_heading: true
      show_source: false
