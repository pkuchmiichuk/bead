"""Typed multidigraph over corpus expressions (buffering tier).

On top of the lazy streaming tier, a :class:`CorpusGraph` materializes the
structure *between* records: a directed, typed multigraph whose nodes are
expressions (one per :class:`~bead.corpus.records.CorpusRecord`) or abstract
entities, and whose edges are typed, directed relations (multiple edges may
connect the same pair). A reply tree (Reddit) is the special case of a graph
whose edges all share one type; arbitrarily complex corpora (typed relations
between expressions) are the general case.

This model is aligned with the ``layers`` property graph (``graphNode`` /
``graphEdgeSet``) so it maps losslessly; see ``bead.interop.layers``.
"""

from __future__ import annotations

import didactic.api as dx

from bead.corpus.records import CorpusRecord
from bead.data.base import BeadBaseModel
from bead.items.item import MetadataValue


class CorpusNode(BeadBaseModel):
    """A node in a corpus graph.

    Attributes
    ----------
    node_id : str
        Stable identifier, unique within the graph (e.g. a Reddit comment id).
    node_type : str
        Node type slug (``"expression"`` for a text record, or an abstract type
        such as ``"entity"``/``"concept"``). Mirrors layers' ``nodeType``.
    node_type_uri : str | None
        Optional canonical type URI (the layers slug+uri pattern).
    label : str | None
        Human-readable node label.
    record : CorpusRecord | None
        The expression this node wraps, if it is a text node.
    properties : dict[str, MetadataValue]
        Arbitrary node properties (maps to a layers feature map).
    """

    node_id: str
    node_type: str = "expression"
    node_type_uri: str | None = None
    label: str | None = None
    record: dx.Embed[CorpusRecord] | None = None
    properties: dict[str, MetadataValue] = dx.field(default_factory=dict)

    @dx.validates("node_id")
    def _check_node_id(self, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("node_id cannot be empty")
        return value.strip()


class CorpusEdge(BeadBaseModel):
    """A typed, directed edge between two corpus nodes.

    Attributes
    ----------
    source_id : str
        ``node_id`` of the source node.
    target_id : str
        ``node_id`` of the target node.
    edge_type : str
        Edge type slug (e.g. ``"reply-to"``, ``"coreference"``).
    edge_type_uri : str | None
        Optional canonical edge-type URI (the layers slug+uri pattern).
    directed : bool
        Whether the edge is directed (``True``) or symmetric.
    confidence : float | None
        Optional confidence in ``[0, 1]``.
    features : dict[str, MetadataValue]
        Arbitrary edge features (maps to a layers feature map).
    """

    source_id: str
    target_id: str
    edge_type: str
    edge_type_uri: str | None = None
    directed: bool = True
    confidence: float | None = None
    features: dict[str, MetadataValue] = dx.field(default_factory=dict)


class CorpusGraph(BeadBaseModel):
    """A directed, typed multigraph over corpus nodes.

    Edges are directed ``source -> target``. Multiple edges (of the same or
    different types) may connect a pair, so this is a multidigraph; a tree is
    the special case where every node has at most one out-edge of the tree's
    edge type.

    Attributes
    ----------
    nodes : tuple[CorpusNode, ...]
        The graph's nodes.
    edges : tuple[CorpusEdge, ...]
        The graph's directed edges.
    graph_metadata : dict[str, MetadataValue]
        Graph-level metadata.
    """

    nodes: tuple[dx.Embed[CorpusNode], ...] = ()
    edges: tuple[dx.Embed[CorpusEdge], ...] = ()
    graph_metadata: dict[str, MetadataValue] = dx.field(default_factory=dict)

    def node_by_id(self, node_id: str) -> CorpusNode | None:
        """Return the node with ``node_id``, or ``None`` if absent."""
        for node in self.nodes:
            if node.node_id == node_id:
                return node
        return None

    def out_edges(
        self, node_id: str, edge_type: str | None = None
    ) -> tuple[CorpusEdge, ...]:
        """Edges whose source is ``node_id`` (optionally filtered by type)."""
        return tuple(
            edge
            for edge in self.edges
            if edge.source_id == node_id
            and (edge_type is None or edge.edge_type == edge_type)
        )

    def in_edges(
        self, node_id: str, edge_type: str | None = None
    ) -> tuple[CorpusEdge, ...]:
        """Edges whose target is ``node_id`` (optionally filtered by type)."""
        return tuple(
            edge
            for edge in self.edges
            if edge.target_id == node_id
            and (edge_type is None or edge.edge_type == edge_type)
        )

    def successors(self, node_id: str, edge_type: str | None = None) -> tuple[str, ...]:
        """Target ids of ``node_id``'s out-edges, in edge order."""
        return tuple(edge.target_id for edge in self.out_edges(node_id, edge_type))

    def predecessors(
        self, node_id: str, edge_type: str | None = None
    ) -> tuple[str, ...]:
        """Source ids of ``node_id``'s in-edges, in edge order."""
        return tuple(edge.source_id for edge in self.in_edges(node_id, edge_type))

    def roots(self, edge_type: str | None = None) -> tuple[str, ...]:
        """Node ids with no in-edges (of the given type)."""
        return tuple(
            node.node_id
            for node in self.nodes
            if not self.in_edges(node.node_id, edge_type)
        )

    def descendants(
        self, node_id: str, edge_type: str | None = None
    ) -> tuple[str, ...]:
        """Transitive successors of ``node_id`` (cycle-guarded, excludes self)."""
        seen: set[str] = {node_id}
        order: list[str] = []
        queue: list[str] = list(self.successors(node_id, edge_type))
        while queue:
            current = queue.pop(0)
            if current in seen:
                continue
            seen.add(current)
            order.append(current)
            queue.extend(self.successors(current, edge_type))
        return tuple(order)

    def reverse(self) -> CorpusGraph:
        """Return a copy of the graph with every edge's direction flipped."""
        flipped = tuple(
            edge.with_(source_id=edge.target_id, target_id=edge.source_id)
            for edge in self.edges
        )
        return self.with_(edges=flipped).touched()
