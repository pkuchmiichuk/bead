# bead.corpus

Streaming corpus ingestion and structural sampling. Turns raw external text
(JSON Lines, optionally Zstandard-compressed; CSV/TSV; or language-model
completions) into structurally filtered experimental `Item`s: stream
`CorpusRecord`s from a `CorpusSource`, dependency-parse them, and keep only those
whose parse satisfies a structural DSL constraint.

The whole pipeline is lazy, so a structural query can run over a multi-gigabyte
corpus without loading it into memory.

## Records

::: bead.corpus.records
    options:
      show_root_heading: true
      show_source: false

## Source Protocol

::: bead.corpus.base
    options:
      show_root_heading: true
      show_source: false

## Sources

::: bead.corpus.sources
    options:
      show_root_heading: true
      show_source: false

## Pipeline

::: bead.corpus.pipeline
    options:
      show_root_heading: true
      show_source: false
