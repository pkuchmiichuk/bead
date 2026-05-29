"""Lens between a ``CorpusGraph`` and the layers property graph.

The lens projects a :class:`~bead.corpus.graph.CorpusGraph` to a layers-shaped
view (expression records, graph nodes, and a ``graphEdgeSet``) and keeps a
complement holding the information layers' graph does not express directly (the
bead framework identity, edge directedness, and exact float confidence).
Together the view and complement reconstruct the graph exactly.
"""

from __future__ import annotations

import didactic.api as dx

from bead.corpus.graph import CorpusEdge, CorpusGraph, CorpusNode
from bead.corpus.records import CorpusRecord
from bead.data.base import JsonValue
from bead.interop.layers._convert import (
    apply_identity,
    from_feature_map,
    from_feature_map_scalar,
    from_object_ref,
    identity_of,
    j_bool,
    j_float_or_none,
    j_int,
    j_list,
    j_obj,
    j_str,
    j_str_or_none,
    object_ref,
    to_feature_map,
)

_CONFIDENCE_SCALE = 1000


class CorpusGraphLayersLens(dx.Lens[CorpusGraph, JsonValue, JsonValue]):
    """Lossless lens ``CorpusGraph <-> (layers graph view, bead complement)``."""

    def forward(self, graph: CorpusGraph) -> tuple[JsonValue, JsonValue]:
        """Project a graph to its layers view and bead complement."""
        expressions: dict[str, JsonValue] = {}
        graph_nodes: dict[str, JsonValue] = {}
        node_complements: dict[str, JsonValue] = {}
        node_order: list[JsonValue] = []

        for node in graph.nodes:
            node_order.append(node.node_id)
            if node.record is not None:
                expr: dict[str, JsonValue] = {
                    "id": node.node_id,
                    "kind": node.node_type,
                    "text": node.record.text,
                    "features": to_feature_map(node.record.provenance),
                    "createdAt": node.record.created_at.isoformat(),
                }
                if node.node_type_uri is not None:
                    expr["kindUri"] = node.node_type_uri
                expressions[node.node_id] = expr
                node_complements[node.node_id] = {
                    "is_expression": True,
                    "identity": identity_of(node),
                    "label": node.label,
                    "properties": to_feature_map(node.properties),
                    "record_identity": identity_of(node.record),
                    "record_source_name": node.record.source_name,
                    "record_index": node.record.record_index,
                }
            else:
                graph_node: dict[str, JsonValue] = {
                    "nodeType": node.node_type,
                    "properties": to_feature_map(node.properties),
                    "createdAt": node.created_at.isoformat(),
                }
                if node.node_type_uri is not None:
                    graph_node["nodeTypeUri"] = node.node_type_uri
                if node.label is not None:
                    graph_node["label"] = node.label
                graph_nodes[node.node_id] = graph_node
                node_complements[node.node_id] = {
                    "is_expression": False,
                    "identity": identity_of(node),
                }

        edge_views: list[JsonValue] = []
        edge_complements: list[JsonValue] = []
        for edge in graph.edges:
            edge_view: dict[str, JsonValue] = {
                "edgeType": edge.edge_type,
                "source": object_ref(edge.source_id),
                "target": object_ref(edge.target_id),
                "features": to_feature_map(edge.features),
            }
            if edge.edge_type_uri is not None:
                edge_view["edgeTypeUri"] = edge.edge_type_uri
            if edge.confidence is not None:
                edge_view["confidence"] = round(edge.confidence * _CONFIDENCE_SCALE)
            edge_view["uuid"] = {"value": str(edge.id)}
            edge_views.append(edge_view)
            edge_complements.append(
                {
                    "identity": identity_of(edge),
                    "directed": edge.directed,
                    "confidence": edge.confidence,
                }
            )

        view: JsonValue = {
            "expressions": expressions,
            "graphNodes": graph_nodes,
            "graphEdgeSet": {
                "edges": tuple(edge_views),
                "createdAt": graph.created_at.isoformat(),
            },
        }
        complement: JsonValue = {
            "graph_identity": identity_of(graph),
            "graph_metadata": to_feature_map(graph.graph_metadata),
            "node_order": tuple(node_order),
            "node_complements": node_complements,
            "edge_complements": tuple(edge_complements),
        }
        return view, complement

    def backward(self, view: JsonValue, complement: JsonValue) -> CorpusGraph:
        """Reconstruct the graph from its layers view and bead complement."""
        view_obj = j_obj(view)
        comp = j_obj(complement)
        expressions = j_obj(view_obj["expressions"])
        graph_nodes = j_obj(view_obj["graphNodes"])
        node_complements = j_obj(comp["node_complements"])

        nodes: list[CorpusNode] = []
        for node_id_value in j_list(comp["node_order"]):
            node_id = j_str(node_id_value)
            node_comp = j_obj(node_complements[node_id])
            if j_bool(node_comp["is_expression"]):
                entry = j_obj(expressions[node_id])
                record = apply_identity(
                    CorpusRecord(
                        text=j_str(entry["text"]),
                        source_name=j_str(node_comp["record_source_name"]),
                        record_index=j_int(node_comp["record_index"]),
                        provenance=from_feature_map_scalar(entry["features"]),
                    ),
                    node_comp["record_identity"],
                )
                node = CorpusNode(
                    node_id=node_id,
                    node_type=j_str(entry["kind"]),
                    node_type_uri=j_str_or_none(entry.get("kindUri")),
                    label=j_str_or_none(node_comp["label"]),
                    record=record,
                    properties=from_feature_map(node_comp["properties"]),
                )
            else:
                entry = j_obj(graph_nodes[node_id])
                node = CorpusNode(
                    node_id=node_id,
                    node_type=j_str(entry["nodeType"]),
                    node_type_uri=j_str_or_none(entry.get("nodeTypeUri")),
                    label=j_str_or_none(entry.get("label")),
                    record=None,
                    properties=from_feature_map(entry["properties"]),
                )
            nodes.append(apply_identity(node, node_comp["identity"]))

        edge_set = j_obj(view_obj["graphEdgeSet"])
        edge_views = j_list(edge_set["edges"])
        edge_complements = j_list(comp["edge_complements"])
        edges: list[CorpusEdge] = []
        for edge_view_value, edge_comp_value in zip(
            edge_views, edge_complements, strict=True
        ):
            edge_view = j_obj(edge_view_value)
            edge_comp = j_obj(edge_comp_value)
            edges.append(
                apply_identity(
                    CorpusEdge(
                        source_id=from_object_ref(edge_view["source"]),
                        target_id=from_object_ref(edge_view["target"]),
                        edge_type=j_str(edge_view["edgeType"]),
                        edge_type_uri=j_str_or_none(edge_view.get("edgeTypeUri")),
                        directed=j_bool(edge_comp["directed"]),
                        confidence=j_float_or_none(edge_comp["confidence"]),
                        features=from_feature_map(edge_view["features"]),
                    ),
                    edge_comp["identity"],
                )
            )

        return apply_identity(
            CorpusGraph(
                nodes=tuple(nodes),
                edges=tuple(edges),
                graph_metadata=from_feature_map(comp["graph_metadata"]),
            ),
            comp["graph_identity"],
        )


CORPUS_GRAPH_LAYERS = CorpusGraphLayersLens()


def graph_to_layers(graph: CorpusGraph) -> JsonValue:
    """Return the standalone layers-shaped view of a corpus graph."""
    view, _complement = CORPUS_GRAPH_LAYERS.forward(graph)
    return view
