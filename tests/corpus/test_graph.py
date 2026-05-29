"""Tests for the corpus graph (typed multidigraph) and its traversal."""

from __future__ import annotations

from bead.corpus.graph import CorpusEdge, CorpusGraph, CorpusNode


def _graph() -> CorpusGraph:
    # a -> b -> c, plus a parallel typed edge a =mentions=> c
    nodes = (
        CorpusNode(node_id="a"),
        CorpusNode(node_id="b"),
        CorpusNode(node_id="c"),
    )
    edges = (
        CorpusEdge(source_id="a", target_id="b", edge_type="next"),
        CorpusEdge(source_id="b", target_id="c", edge_type="next"),
        CorpusEdge(source_id="a", target_id="c", edge_type="mentions"),
    )
    return CorpusGraph(nodes=nodes, edges=edges)


class TestTraversal:
    """Tests for the graph traversal helpers."""

    def test_node_by_id(self) -> None:
        g = _graph()
        assert g.node_by_id("b") is not None
        assert g.node_by_id("missing") is None

    def test_out_in_edges_typed(self) -> None:
        g = _graph()
        assert len(g.out_edges("a")) == 2
        assert len(g.out_edges("a", "next")) == 1
        assert len(g.in_edges("c")) == 2
        assert len(g.in_edges("c", "mentions")) == 1

    def test_successors_predecessors(self) -> None:
        g = _graph()
        assert set(g.successors("a")) == {"b", "c"}
        assert g.successors("a", "next") == ("b",)
        assert g.predecessors("c", "next") == ("b",)
        assert g.predecessors("c", "mentions") == ("a",)

    def test_roots(self) -> None:
        g = _graph()
        # only 'a' has no incoming edge
        assert g.roots() == ("a",)

    def test_descendants_follows_type(self) -> None:
        g = _graph()
        assert g.descendants("a", "next") == ("b", "c")
        assert g.descendants("a", "mentions") == ("c",)

    def test_descendants_cycle_guarded(self) -> None:
        nodes = (CorpusNode(node_id="x"), CorpusNode(node_id="y"))
        edges = (
            CorpusEdge(source_id="x", target_id="y", edge_type="e"),
            CorpusEdge(source_id="y", target_id="x", edge_type="e"),
        )
        g = CorpusGraph(nodes=nodes, edges=edges)
        # does not loop forever; visits the other node once
        assert g.descendants("x") == ("y",)

    def test_reverse(self) -> None:
        g = _graph().reverse()
        # edges flipped: b->a, c->b, c->a
        assert g.successors("c") == ("b", "a")
        assert g.roots() == ("c",)


class TestMultidigraph:
    """Parallel edges of the same type between a pair are permitted."""

    def test_parallel_edges_same_type(self) -> None:
        nodes = (CorpusNode(node_id="a"), CorpusNode(node_id="b"))
        edges = (
            CorpusEdge(source_id="a", target_id="b", edge_type="cites"),
            CorpusEdge(source_id="a", target_id="b", edge_type="cites"),
        )
        g = CorpusGraph(nodes=nodes, edges=edges)
        assert len(g.out_edges("a", "cites")) == 2
        assert g.successors("a") == ("b", "b")
