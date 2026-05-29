"""Tests for assembling a corpus graph from a record stream."""

from __future__ import annotations

from collections.abc import Iterable

from bead.corpus.assemble import EdgeSpec, assemble_graph
from bead.corpus.graph import CorpusEdge
from bead.corpus.records import CorpusRecord, ProvenanceValue


def _record(text: str, **provenance: ProvenanceValue) -> CorpusRecord:
    return CorpusRecord(text=text, source_name="reddit", provenance=dict(provenance))


def _reddit_thread() -> list[CorpusRecord]:
    # submission + three comments forming a reply tree
    return [
        _record("the submission", id="sub"),
        _record("top reply", id="c1", parent_id="t3_sub"),
        _record("nested reply", id="c2", parent_id="t1_c1"),
        _record("another nested reply", id="c3", parent_id="t1_c1"),
    ]


_REPLY = EdgeSpec(
    target_field="parent_id", edge_type="reply-to", strip_prefixes=("t1_", "t3_")
)


class TestRedditReplyTree:
    """Reconstructs a Reddit reply tree (edges child -> parent)."""

    def test_edges_and_prefix_stripping(self) -> None:
        g = assemble_graph(
            _reddit_thread(), node_id_field="id", edge_specs=[_REPLY]
        )
        assert {n.node_id for n in g.nodes} == {"sub", "c1", "c2", "c3"}
        # c1 replies to the submission (t3_ prefix stripped)
        assert g.successors("c1", "reply-to") == ("sub",)
        # c2 and c3 reply to c1 (t1_ prefix stripped)
        assert set(g.predecessors("c1", "reply-to")) == {"c2", "c3"}
        # the submission replies to nothing
        assert g.out_edges("sub", "reply-to") == ()

    def test_full_tree_via_reverse(self) -> None:
        # Reverse the child->parent edges to get parent->child, then the
        # submission is the unique root and its descendants are the thread.
        g = assemble_graph(
            _reddit_thread(), node_id_field="id", edge_specs=[_REPLY]
        ).reverse()
        assert g.roots("reply-to") == ("sub",)
        assert set(g.descendants("sub", "reply-to")) == {"c1", "c2", "c3"}

    def test_records_preserved_on_nodes(self) -> None:
        g = assemble_graph(
            _reddit_thread(), node_id_field="id", edge_specs=[_REPLY]
        )
        node = g.node_by_id("c2")
        assert node is not None
        assert node.record is not None
        assert node.record.text == "nested reply"
        # losslessly retained provenance still present on the wrapped record
        assert node.record.provenance["parent_id"] == "t1_c1"


class TestGeneralGraph:
    """Arbitrary typed multidigraphs, dangling targets, and edge_fn."""

    def test_multiple_edge_specs(self) -> None:
        records = [
            _record("x", id="x", parent_id="root", author="alice"),
            _record("y", id="y", parent_id="x", author="alice"),
        ]
        specs = [
            EdgeSpec(target_field="parent_id", edge_type="reply-to"),
            EdgeSpec(target_field="author", edge_type="authored-by"),
        ]
        g = assemble_graph(records, node_id_field="id", edge_specs=specs)
        assert g.successors("y", "reply-to") == ("x",)
        assert g.successors("y", "authored-by") == ("alice",)

    def test_dangling_target_preserved(self) -> None:
        # parent_id 'root' has no node; the edge is kept, not dropped.
        records = [_record("x", id="x", parent_id="root")]
        g = assemble_graph(records, node_id_field="id", edge_specs=[_REPLY])
        assert g.successors("x", "reply-to") == ("root",)
        assert g.node_by_id("root") is None

    def test_edge_fn(self) -> None:
        def link_pairs(
            record: CorpusRecord, node_id: str
        ) -> Iterable[CorpusEdge]:
            mentions = record.provenance.get("mentions")
            if isinstance(mentions, str):
                return [
                    CorpusEdge(
                        source_id=node_id, target_id=mentions, edge_type="mentions"
                    )
                ]
            return []

        records = [_record("x", id="x", mentions="y"), _record("y", id="y")]
        g = assemble_graph(records, node_id_field="id", edge_fn=link_pairs)
        assert g.successors("x", "mentions") == ("y",)

    def test_records_without_node_id_skipped(self) -> None:
        records = [_record("x", id="x"), _record("no id")]
        g = assemble_graph(records, node_id_field="id")
        assert {n.node_id for n in g.nodes} == {"x"}
