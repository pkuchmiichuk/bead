"""Lens between a ``CorpusGraph`` and the layers property graph.

The lens projects a :class:`~bead.corpus.graph.CorpusGraph` to a
:class:`CorpusGraphLayers` view (a bundle of canonical
:class:`lairs.records.expression.Expression` and
:class:`lairs.records.graph.GraphNode` records plus a
:class:`lairs.records.graph.GraphEdgeSet`) and keeps a complement holding the
information layers' graph does not express directly (the bead framework
identity, the bead node ids, edge directedness, and exact float confidence).
Together the view and complement reconstruct the graph exactly.
"""

from __future__ import annotations

import didactic.api as dx
from lairs.records import defs, expression, graph

from bead.corpus.graph import CorpusEdge, CorpusGraph, CorpusNode
from bead.corpus.records import CorpusRecord
from bead.data.base import JsonValue
from bead.interop.layers._convert import (
    CONFIDENCE_SCALE,
    apply_identity,
    dumps_meta,
    feature_map,
    from_object_ref,
    identity_of,
    j_bool,
    j_float_or_none,
    j_int,
    j_list,
    j_obj,
    j_str,
    j_str_or_none,
    loads_meta,
    object_ref,
    read_feature_map,
    read_feature_map_scalar,
)


class CorpusGraphLayers(dx.Model):
    """A layers view of a corpus graph: expressions, graph nodes, and an edge set.

    The records are ordered tuples; the bead node ids that key them live in the
    lens complement (``layers`` records carry no bead node id). An empty graph
    projects to empty tuples and an empty edge set.
    """

    expressions: tuple[dx.Embed[expression.Expression], ...] = dx.field(default=())
    graph_nodes: tuple[dx.Embed[graph.GraphNode], ...] = dx.field(default=())
    edge_set: dx.Embed[graph.GraphEdgeSet] = dx.field()


class CorpusGraphLayersLens(dx.Lens[CorpusGraph, CorpusGraphLayers, JsonValue]):
    """Lossless lens ``CorpusGraph <-> (layers graph view, bead complement)``."""

    def forward(self, graph_in: CorpusGraph) -> tuple[CorpusGraphLayers, JsonValue]:
        """Project a graph to its layers view and bead complement."""
        expressions: list[expression.Expression] = []
        graph_nodes: list[graph.GraphNode] = []
        node_complements: dict[str, JsonValue] = {}
        node_order: list[JsonValue] = []

        for node in graph_in.nodes:
            node_order.append(node.node_id)
            if node.record is not None:
                expressions.append(
                    expression.Expression(
                        id=node.node_id,
                        kind=node.node_type,
                        kindUri=node.node_type_uri,
                        text=node.record.text,
                        features=feature_map(node.record.provenance),
                        createdAt=node.record.created_at,
                    )
                )
                node_complements[node.node_id] = {
                    "is_expression": True,
                    "identity": identity_of(node),
                    "label": node.label,
                    "properties": dumps_meta(node.properties),
                    "record_identity": identity_of(node.record),
                    "record_source_name": node.record.source_name,
                    "record_index": node.record.record_index,
                }
            else:
                graph_nodes.append(
                    graph.GraphNode(
                        nodeType=node.node_type,
                        nodeTypeUri=node.node_type_uri,
                        label=node.label,
                        properties=feature_map(node.properties),
                        createdAt=node.created_at,
                    )
                )
                node_complements[node.node_id] = {
                    "is_expression": False,
                    "identity": identity_of(node),
                }

        edge_entries: list[graph.GraphEdgeEntry] = []
        edge_complements: list[JsonValue] = []
        for edge in graph_in.edges:
            confidence = (
                round(edge.confidence * CONFIDENCE_SCALE)
                if edge.confidence is not None
                else None
            )
            edge_entries.append(
                graph.GraphEdgeEntry(
                    uuid=defs.Uuid(value=str(edge.id)),
                    edgeType=edge.edge_type,
                    edgeTypeUri=edge.edge_type_uri,
                    source=object_ref(edge.source_id),
                    target=object_ref(edge.target_id),
                    confidence=confidence,
                    features=feature_map(edge.features),
                )
            )
            edge_complements.append(
                {
                    "identity": identity_of(edge),
                    "directed": edge.directed,
                    "confidence": edge.confidence,
                }
            )

        view = CorpusGraphLayers(
            expressions=tuple(expressions),
            graph_nodes=tuple(graph_nodes),
            edge_set=graph.GraphEdgeSet(
                createdAt=graph_in.created_at, edges=tuple(edge_entries)
            ),
        )
        complement: JsonValue = {
            "graph_identity": identity_of(graph_in),
            "graph_metadata": dumps_meta(graph_in.graph_metadata),
            "node_order": tuple(node_order),
            "node_complements": node_complements,
            "edge_complements": tuple(edge_complements),
        }
        return view, complement

    def backward(self, view: CorpusGraphLayers, complement: JsonValue) -> CorpusGraph:
        """Reconstruct the graph from its layers view and bead complement."""
        comp = j_obj(complement)
        node_complements = j_obj(comp["node_complements"])

        nodes: list[CorpusNode] = []
        expr_index = 0
        graph_node_index = 0
        for node_id_value in j_list(comp["node_order"]):
            node_id = j_str(node_id_value)
            node_comp = j_obj(node_complements[node_id])
            if j_bool(node_comp["is_expression"]):
                entry = view.expressions[expr_index]
                expr_index += 1
                record = apply_identity(
                    CorpusRecord(
                        text=entry.text if entry.text is not None else "",
                        source_name=j_str(node_comp["record_source_name"]),
                        record_index=j_int(node_comp["record_index"]),
                        provenance=read_feature_map_scalar(entry.features),
                    ),
                    node_comp["record_identity"],
                )
                node = CorpusNode(
                    node_id=node_id,
                    node_type=entry.kind,
                    node_type_uri=entry.kindUri,
                    label=j_str_or_none(node_comp["label"]),
                    record=record,
                    properties=loads_meta(node_comp["properties"]),
                )
            else:
                gnode = view.graph_nodes[graph_node_index]
                graph_node_index += 1
                node = CorpusNode(
                    node_id=node_id,
                    node_type=gnode.nodeType,
                    node_type_uri=gnode.nodeTypeUri,
                    label=gnode.label,
                    record=None,
                    properties=read_feature_map(gnode.properties),
                )
            nodes.append(apply_identity(node, node_comp["identity"]))

        edge_complements = j_list(comp["edge_complements"])
        edges: list[CorpusEdge] = []
        for entry, edge_comp_value in zip(
            view.edge_set.edges, edge_complements, strict=True
        ):
            edge_comp = j_obj(edge_comp_value)
            edges.append(
                apply_identity(
                    CorpusEdge(
                        source_id=from_object_ref(entry.source),
                        target_id=from_object_ref(entry.target),
                        edge_type=entry.edgeType,
                        edge_type_uri=entry.edgeTypeUri,
                        directed=j_bool(edge_comp["directed"]),
                        confidence=j_float_or_none(edge_comp["confidence"]),
                        features=read_feature_map(entry.features),
                    ),
                    edge_comp["identity"],
                )
            )

        return apply_identity(
            CorpusGraph(
                nodes=tuple(nodes),
                edges=tuple(edges),
                graph_metadata=loads_meta(comp["graph_metadata"]),
            ),
            comp["graph_identity"],
        )


CORPUS_GRAPH_LAYERS = CorpusGraphLayersLens()


def graph_to_layers(graph_in: CorpusGraph) -> CorpusGraphLayers:
    """Return the standalone layers-shaped view of a corpus graph."""
    view, _complement = CORPUS_GRAPH_LAYERS.forward(graph_in)
    return view
