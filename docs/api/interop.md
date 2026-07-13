# bead.interop

Lossless, law-verified interoperability mappings between bead models and the
[layers](https://github.com/layers-pub/layers) linguistic-annotation schema. The
`layers` subpackage maps bead's corpus, annotation, and resource data to and from
the canonical `lairs.records` models via didactic lenses (`dx.Iso` / `dx.Lens`);
round-trip fidelity is guaranteed by the GetPut/PutGet lens laws.

See the [Layers Interoperability guide](../user-guide/api/layers-interop.md) for
usage.

## Bridge lenses (bead-native <-> layers)

::: bead.interop.layers.graph_lens
    options:
      show_root_heading: true
      show_source: false

::: bead.interop.layers.bridges
    options:
      show_root_heading: true
      show_source: false

::: bead.interop.layers.parse_lens
    options:
      show_root_heading: true
      show_source: false

::: bead.interop.layers.item_bridge
    options:
      show_root_heading: true
      show_source: false

## Resource lenses

Lenses between bead's resource models and their layers counterparts: lexical
items and lexicons to entries and collections, templates to layers templates,
and filled templates to layers fillings.

::: bead.interop.layers.resource_lens
    options:
      show_root_heading: true
      show_source: false

## Judgment and list lenses

Lenses between bead's response and list-composition models and their layers
counterparts: annotation records to judgments (and judgment sets), bead list
constraints to layers list constraints, and experiment lists to layers
collections with their memberships and constraints.

::: bead.interop.layers.judgment_lens
    options:
      show_root_heading: true
      show_source: false

::: bead.interop.layers.list_lens
    options:
      show_root_heading: true
      show_source: false

## Codec and corpus I/O

::: bead.interop.layers.codec
    options:
      show_root_heading: true
      show_source: false

::: bead.interop.layers.corpus_io
    options:
      show_root_heading: true
      show_source: false
