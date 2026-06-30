"""Buffer a record stream into a typed multidigraph.

``assemble_graph`` is the opt-in buffering tier that sits on top of the lazy
streaming sources: it consumes ``CorpusRecord``s and reconstructs the structure
between them (e.g. a Reddit reply tree from ``parent_id``, or an arbitrary typed
graph) as a :class:`~bead.corpus.graph.CorpusGraph`. It holds the records in
memory, so it is a deliberate, explicit step distinct from streaming.
"""

from __future__ import annotations

from collections.abc import Callable, Iterable, Sequence

import didactic.api as dx

from bead.corpus.graph import CorpusEdge, CorpusGraph, CorpusNode
from bead.corpus.records import CorpusRecord
from bead.data.base import BeadBaseModel


class EdgeSpec(BeadBaseModel):
    """Declarative rule for deriving one typed edge per record from a field.

    For each record, if ``target_field`` is present in the record's provenance,
    an edge ``record_node -> target`` is created with type ``edge_type``. The
    target id is the field value with any matching ``strip_prefixes`` removed
    (e.g. Reddit's ``t1_``/``t3_`` fullname prefixes).

    Attributes
    ----------
    target_field : str
        Provenance field naming the other endpoint (e.g. ``"parent_id"``).
    edge_type : str
        Edge type slug for the created edge (e.g. ``"reply-to"``).
    edge_type_uri : str | None
        Optional canonical edge-type URI.
    strip_prefixes : tuple[str, ...]
        Prefixes to strip from the field value to recover the bare node id.
    directed : bool
        Whether the created edge is directed.
    """

    target_field: str
    edge_type: str
    edge_type_uri: str | None = None
    strip_prefixes: tuple[str, ...] = ()
    directed: bool = True

    @dx.validates("target_field", "edge_type")
    def _check_non_empty(self, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("must be non-empty")
        return value.strip()


def _strip_prefix(value: str, prefixes: tuple[str, ...]) -> str:
    """Strip the first matching prefix from *value*."""
    for prefix in prefixes:
        if prefix and value.startswith(prefix):
            return value[len(prefix) :]
    return value


def assemble_graph(
    records: Iterable[CorpusRecord],
    *,
    node_id_field: str,
    edge_specs: Sequence[EdgeSpec] = (),
    edge_fn: Callable[[CorpusRecord, str], Iterable[CorpusEdge]] | None = None,
) -> CorpusGraph:
    """Buffer a record stream into a typed multidigraph.

    Each record with a ``node_id_field`` value becomes one expression node.
    Edges are derived from the declarative ``edge_specs`` and/or a runtime
    ``edge_fn`` (given the record and its node id) for arbitrary extraction.

    Parameters
    ----------
    records : Iterable[CorpusRecord]
        The records to buffer (typically a streaming source).
    node_id_field : str
        Provenance field holding each record's stable node id.
    edge_specs : Sequence[EdgeSpec]
        Declarative field-to-edge rules (the common case).
    edge_fn : Callable[[CorpusRecord, str], Iterable[CorpusEdge]] | None
        Optional function yielding extra edges for arbitrary structure.

    Returns
    -------
    CorpusGraph
        The assembled graph. Edges may reference target ids that have no node
        (dangling references are preserved, not dropped).
    """
    nodes: list[CorpusNode] = []
    edges: list[CorpusEdge] = []
    for record in records:
        node_id_raw = record.provenance.get(node_id_field)
        if node_id_raw is None:
            continue
        node_id = str(node_id_raw)
        nodes.append(CorpusNode(node_id=node_id, record=record))
        for spec in edge_specs:
            target_raw = record.provenance.get(spec.target_field)
            if target_raw is None:
                continue
            target_id = _strip_prefix(str(target_raw), spec.strip_prefixes)
            edges.append(
                CorpusEdge(
                    source_id=node_id,
                    target_id=target_id,
                    edge_type=spec.edge_type,
                    edge_type_uri=spec.edge_type_uri,
                    directed=spec.directed,
                )
            )
        if edge_fn is not None:
            edges.extend(edge_fn(record, node_id))
    return CorpusGraph(nodes=tuple(nodes), edges=tuple(edges))
