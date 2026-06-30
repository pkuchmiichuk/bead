# bead.interop

Lossless, law-verified interoperability mappings between bead models and
external schemas. The `layers` subpackage maps bead's corpus and annotation data
to the [layers](https://github.com/layers-pub/layers) linguistic-annotation
schema and back via didactic lenses (`dx.Iso` / `dx.Lens`); round-trip fidelity
is guaranteed by the GetPut/PutGet lens laws.

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

## Mirror models

Faithful didactic mirrors of the layers constructs.

::: bead.interop.layers.models
    options:
      show_root_heading: true
      show_source: false

::: bead.interop.layers.models_records
    options:
      show_root_heading: true
      show_source: false

## Generic mirror iso

::: bead.interop.layers.model_lenses
    options:
      show_root_heading: true
      show_source: false
