"""Streamed corpus records with provenance.

A ``CorpusRecord`` is the raw ingress of the corpus pipeline: one text unit
drawn from an external source (a JSONL/CSV file, a language model) together
with the provenance needed to trace it. Provenance keys follow the ``layers``
``AnnotationMetadata`` shape (``source_name``, ``tool``, ``model``,
``created_at``, ``confidence``, ``formalism``) alongside any raw source fields,
so corpus-derived items carry layers-ready provenance from ingestion onward.
"""

from __future__ import annotations

import didactic.api as dx

from bead.data.base import BeadBaseModel

type ProvenanceValue = str | int | float | bool | None


class CorpusRecord(BeadBaseModel):
    """A single streamed text record with provenance.

    Attributes
    ----------
    text : str
        The text of the record.
    source_name : str
        Identifier of the source the record was drawn from (e.g. a file
        basename, a corpus name, or a model name).
    record_index : int
        0-based position of the record within its source stream.
    provenance : dict[str, ProvenanceValue]
        Flat scalar provenance. Conventionally includes layers-aligned keys
        (``source_name``, ``tool``, ``model``, ``created_at``, ``confidence``,
        ``formalism``) plus any raw source fields.
    """

    text: str
    source_name: str
    record_index: int = 0
    provenance: dict[str, ProvenanceValue] = dx.field(default_factory=dict)
