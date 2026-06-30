"""Round-trip law tests for the CorpusGraph <-> layers graph lens."""

from __future__ import annotations

from hypothesis import HealthCheck, given, settings
from hypothesis import strategies as st
from lairs.records import defs

from bead.corpus.assemble import EdgeSpec, assemble_graph
from bead.corpus.graph import CorpusEdge, CorpusGraph, CorpusNode
from bead.corpus.records import CorpusRecord
from bead.interop.layers.graph_lens import CORPUS_GRAPH_LAYERS, graph_to_layers

LENS = CORPUS_GRAPH_LAYERS


def _assert_roundtrip(graph: CorpusGraph) -> None:
    view, complement = LENS.forward(graph)
    # GetPut: reconstructing from view + complement yields the original exactly.
    assert LENS.backward(view, complement) == graph
    # PutGet: re-projecting the reconstruction yields the same view + complement.
    view2, complement2 = LENS.forward(LENS.backward(view, complement))
    assert (view2, complement2) == (view, complement)


class TestExampleRoundTrips:
    """Deterministic round-trips over representative graphs."""

    def test_empty_graph(self) -> None:
        _assert_roundtrip(CorpusGraph())

    def test_reddit_thread(self) -> None:
        records = [
            CorpusRecord(text="sub", source_name="r", provenance={"id": "sub"}),
            CorpusRecord(
                text="reply one",
                source_name="r",
                provenance={"id": "c1", "parent_id": "t3_sub", "score": 5},
            ),
            CorpusRecord(
                text="reply two",
                source_name="r",
                provenance={"id": "c2", "parent_id": "t1_c1"},
            ),
        ]
        graph = assemble_graph(
            records,
            node_id_field="id",
            edge_specs=[
                EdgeSpec(
                    target_field="parent_id",
                    edge_type="reply-to",
                    strip_prefixes=("t1_", "t3_"),
                )
            ],
        )
        _assert_roundtrip(graph)

    def test_abstract_nodes_and_typed_multidigraph(self) -> None:
        graph = CorpusGraph(
            nodes=(
                CorpusNode(node_id="a", node_type="entity", label="Alice"),
                CorpusNode(
                    node_id="b",
                    node_type="concept",
                    node_type_uri="at://x#concept",
                    properties={"weight": 3, "tags": ("x", "y")},
                ),
            ),
            edges=(
                CorpusEdge(source_id="a", target_id="b", edge_type="mentions"),
                CorpusEdge(
                    source_id="a",
                    target_id="b",
                    edge_type="mentions",
                    edge_type_uri="at://x#mentions",
                    directed=False,
                    confidence=0.875,
                    features={"note": "parallel edge"},
                ),
            ),
            graph_metadata={"corpus": "demo"},
        )
        _assert_roundtrip(graph)

    def test_expression_node_preserves_provenance(self) -> None:
        graph = CorpusGraph(
            nodes=(
                CorpusNode(
                    node_id="x",
                    record=CorpusRecord(
                        text="hello world",
                        source_name="src",
                        record_index=7,
                        provenance={"author": "a", "score": 2, "deleted": False},
                    ),
                    label="kept",
                    properties={"k": "v"},
                ),
            ),
        )
        _assert_roundtrip(graph)

    def test_view_is_layers_shaped(self) -> None:
        graph = CorpusGraph(
            nodes=(
                CorpusNode(node_id="x", record=CorpusRecord(text="t", source_name="s")),
            ),
            edges=(CorpusEdge(source_id="x", target_id="y", edge_type="e"),),
        )
        view = graph_to_layers(graph)
        edge = view.edge_set.edges[0]
        assert edge.edgeType == "e"
        assert edge.source == defs.ObjectRef(localId=defs.Uuid(value="x"))
        assert edge.target == defs.ObjectRef(localId=defs.Uuid(value="y"))
        assert view.expressions[0].kind == "expression"
        assert view.expressions[0].text == "t"


# --- property-based lens-law verification -----------------------------------

_scalar = st.one_of(st.text(max_size=6), st.integers(-50, 50), st.booleans(), st.none())
_features = st.dictionaries(
    st.text(alphabet="klm", min_size=1, max_size=3), _scalar, max_size=3
)
_node_ids = st.lists(
    st.text(alphabet="abcde", min_size=1, max_size=4), max_size=5, unique=True
)


@st.composite
def _graphs(draw: st.DrawFn) -> CorpusGraph:
    ids = draw(_node_ids)
    nodes: list[CorpusNode] = []
    for node_id in ids:
        if draw(st.booleans()):
            record = CorpusRecord(
                text=draw(st.text(max_size=8)),
                source_name=draw(st.text(max_size=4)),
                record_index=draw(st.integers(0, 20)),
                provenance=draw(_features),
            )
            nodes.append(
                CorpusNode(node_id=node_id, record=record, properties=draw(_features))
            )
        else:
            nodes.append(
                CorpusNode(
                    node_id=node_id,
                    node_type=draw(st.sampled_from(["entity", "concept"])),
                    label=draw(st.one_of(st.none(), st.text(max_size=5))),
                    properties=draw(_features),
                )
            )
    endpoint = (
        st.sampled_from(ids)
        if ids
        else st.text(alphabet="abcde", min_size=1, max_size=4)
    )
    edges: list[CorpusEdge] = []
    for _ in range(draw(st.integers(0, 4))):
        edges.append(
            CorpusEdge(
                source_id=draw(endpoint),
                target_id=draw(endpoint),
                edge_type=draw(st.sampled_from(["e1", "e2"])),
                directed=draw(st.booleans()),
                confidence=draw(st.one_of(st.none(), st.floats(0.0, 1.0))),
                features=draw(_features),
            )
        )
    return CorpusGraph(nodes=tuple(nodes), edges=tuple(edges))


class TestLensLaws:
    """The didactic GetPut/PutGet laws hold across generated graphs."""

    @settings(max_examples=60, suppress_health_check=[HealthCheck.too_slow])
    @given(_graphs())
    def test_get_put_law(self, graph: CorpusGraph) -> None:
        view, complement = LENS.forward(graph)
        assert LENS.backward(view, complement) == graph
