"""Tests for the layers corpus ingest and egress helpers."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

from lairs.store import Repository

from bead.corpus.graph import CorpusGraph, CorpusNode
from bead.corpus.records import CorpusRecord
from bead.interop.layers import corpus_io
from bead.items.item import Item, ItemCollection
from bead.items.spans import Span, SpanLabel, SpanSegment


def _collection() -> ItemCollection:
    return ItemCollection(
        name="demo",
        source_template_collection_id=uuid4(),
        source_filled_collection_id=uuid4(),
        items=(
            Item(
                item_template_id=uuid4(),
                rendered_elements={"text": "The cat sat"},
                tokenized_elements={"text": ("The", "cat", "sat")},
                token_space_after={"text": (True, True, False)},
                spans=(
                    Span(
                        span_id="s1",
                        segments=(SpanSegment(element_name="text", indices=(1,)),),
                        label=SpanLabel(label="ANIMAL", label_id="Q5"),
                    ),
                ),
            ),
        ),
    )


class TestEgress:
    """Bead data projects to a coherent layers corpus."""

    def test_items_to_corpus_has_expressions(self) -> None:
        corpus = corpus_io.items_to_corpus(_collection(), corpus_name="demo")
        records = list(corpus_io.corpus_to_records(corpus))
        assert [record.text for record in records] == ["The cat sat"]

    def test_graph_to_corpus_has_expressions(self) -> None:
        graph = CorpusGraph(
            nodes=(
                CorpusNode(
                    node_id="a", record=CorpusRecord(text="root", source_name="s")
                ),
                CorpusNode(
                    node_id="b",
                    record=CorpusRecord(
                        text="child", source_name="s", provenance={"score": 3}
                    ),
                ),
            )
        )
        corpus = corpus_io.graph_to_corpus(graph, corpus_name="g")
        assert len(list(corpus_io.corpus_to_records(corpus))) == 2

    def test_materialize_writes_parquet(self, tmp_path: Path) -> None:
        corpus = corpus_io.items_to_corpus(_collection(), corpus_name="demo")
        paths = corpus_io.materialize_corpus(corpus, tmp_path)
        names = {path.name for path in paths}
        assert "expressions.parquet" in names
        assert "annotations.parquet" in names

    def test_save_to_repo_returns_revision(self, tmp_path: Path) -> None:
        corpus = corpus_io.items_to_corpus(_collection(), corpus_name="demo")
        revision = corpus_io.save_corpus_repo(corpus, tmp_path / "repo")
        assert isinstance(revision, str)
        assert revision

    def test_publish_dry_run_plans_without_network(self, tmp_path: Path) -> None:
        corpus = corpus_io.items_to_corpus(_collection(), corpus_name="demo")
        revision = corpus_io.save_corpus_repo(corpus, tmp_path / "repo")
        repo = Repository.open(tmp_path / "repo")
        plan = corpus_io.publish_corpus(
            repo, revision, to="did:plc:example", dry_run=True
        )
        assert type(plan).__name__ == "PublishPlan"
        assert plan.creates


class TestIngest:
    """A layers corpus reconstructs into bead corpus models and items."""

    def test_corpus_to_items_recovers_spans(self) -> None:
        corpus = corpus_io.items_to_corpus(_collection(), corpus_name="demo")
        items = list(corpus_io.corpus_to_items(corpus, item_template_id=uuid4()))
        assert len(items) == 1
        span = items[0].spans[0]
        assert span.label is not None
        assert span.label.label == "ANIMAL"
        assert span.label.label_id == "Q5"

    def test_corpus_to_graph_derives_parent_edges(self) -> None:
        graph = CorpusGraph(
            nodes=(
                CorpusNode(
                    node_id="a", record=CorpusRecord(text="root", source_name="s")
                ),
            )
        )
        corpus = corpus_io.graph_to_corpus(graph, corpus_name="g")
        rebuilt = corpus_io.corpus_to_graph(corpus)
        assert len(rebuilt.nodes) == 1
