# bead.transforms

Value-level text transforms (`str -> str`, parameterised by a
`TransformContext`) used when rendering template slots and item prompts.
Transforms are registered by name in a `TransformRegistry`; any callable
conforming to the `SpanTextTransform` protocol can be registered.

## Core Abstractions

::: bead.transforms.base
    options:
      show_root_heading: true
      show_source: false

## Text Transforms

Pure surface-string transforms. In addition to case transforms (`lower`,
`upper`, `capitalize`, `title`), this module provides `MarkdownStripTransform`
and `RedditCleanupTransform` for cleaning web/markdown text into plain prose,
and `split_sentences` for sentence segmentation (parser-backed when a
spaCy/Stanza config is given, with a regular-expression fallback otherwise).

::: bead.transforms.text
    options:
      show_root_heading: true
      show_source: false

## Morphological Transforms

::: bead.transforms.morphology
    options:
      show_root_heading: true
      show_source: false
