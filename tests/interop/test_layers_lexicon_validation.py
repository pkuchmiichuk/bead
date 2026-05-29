"""Validate bead's layers-mapping output against the layers lexicons.

Uses the ATProto lexicon validation machinery (``@atproto/lexicon``) against the
layers lexicons vendored as the ``vendor/layers`` git submodule (checked out with
``git submodule update --init``) to prove every mapping produces schema-valid
layers records. The validator runs in Node; the suite skips if Node, the
validator dependency, or the submodule checkout is unavailable.
"""

from __future__ import annotations

import json
import shutil
import subprocess
from pathlib import Path

import pytest

from bead.corpus.graph import CorpusEdge, CorpusGraph, CorpusNode
from bead.corpus.records import CorpusRecord
from bead.interop.layers import models as m
from bead.interop.layers import models_records as r
from bead.interop.layers.bridges import RECORD_EXPRESSION
from bead.interop.layers.graph_lens import graph_to_layers
from bead.interop.layers.model_lenses import ALL_MIRROR_ISOS
from bead.interop.layers.parse_lens import parse_to_layers
from bead.tokenization.parsers import ParsedSentence, ParsedToken

# Reuse the exact instances exercised by the round-trip suites.
from tests.interop.test_layers_defs import _EXAMPLES as _DEF_EXAMPLES
from tests.interop.test_layers_records import _EXAMPLES as _RECORD_EXAMPLES

_VALIDATOR = Path(__file__).parent / "lexicon_validator"
_INSTALLED = _VALIDATOR / "node_modules" / "@atproto" / "lexicon"
_REPO_ROOT = Path(__file__).resolve().parents[2]
_LEXICON_DIR = _REPO_ROOT / "vendor" / "layers" / "lexicons" / "pub" / "layers"

# Mirror model type -> the lexicon URI its JSON must validate against.
_LEX_URI: dict[type, str] = {
    m.LayersUuid: "pub.layers.defs#uuid",
    m.Feature: "pub.layers.defs#feature",
    m.FeatureMap: "pub.layers.defs#featureMap",
    m.KnowledgeRef: "pub.layers.defs#knowledgeRef",
    m.BoundingBox: "pub.layers.defs#boundingBox",
    m.TemporalSpan: "pub.layers.defs#temporalSpan",
    m.AgentRef: "pub.layers.defs#agentRef",
    m.ObjectRef: "pub.layers.defs#objectRef",
    m.LayersSpan: "pub.layers.defs#span",
    m.TokenRef: "pub.layers.defs#tokenRef",
    m.TokenRefSequence: "pub.layers.defs#tokenRefSequence",
    m.Keyframe: "pub.layers.defs#keyframe",
    m.SpatioTemporalAnchor: "pub.layers.defs#spatioTemporalAnchor",
    m.TemporalEntity: "pub.layers.defs#temporalEntity",
    m.TemporalModifier: "pub.layers.defs#temporalModifier",
    m.TemporalExpression: "pub.layers.defs#temporalExpression",
    m.SpatialEntity: "pub.layers.defs#spatialEntity",
    m.SpatialModifier: "pub.layers.defs#spatialModifier",
    m.SpatialExpression: "pub.layers.defs#spatialExpression",
    m.PageAnchor: "pub.layers.defs#pageAnchor",
    m.TextQuoteSelector: "pub.layers.defs#textQuoteSelector",
    m.TextPositionSelector: "pub.layers.defs#textPositionSelector",
    m.FragmentSelector: "pub.layers.defs#fragmentSelector",
    m.ExternalTarget: "pub.layers.defs#externalTarget",
    m.Anchor: "pub.layers.defs#anchor",
    m.AlignmentLink: "pub.layers.defs#alignmentLink",
    m.AnnotationMetadata: "pub.layers.defs#annotationMetadata",
    m.LayersConstraint: "pub.layers.defs#constraint",
    r.Expression: "pub.layers.expression.expression",
    r.Token: "pub.layers.segmentation.defs#token",
    r.Tokenization: "pub.layers.segmentation.defs#tokenization",
    r.ArgumentRef: "pub.layers.annotation.defs#argumentRef",
    r.Annotation: "pub.layers.annotation.defs#annotation",
    r.Cluster: "pub.layers.annotation.defs#cluster",
    r.AnnotationLayer: "pub.layers.annotation.annotationLayer",
    r.GraphNode: "pub.layers.graph.graphNode",
    r.GraphEdge: "pub.layers.graph.graphEdge",
    r.GraphEdgeEntry: "pub.layers.graph.defs#graphEdgeEntry",
    r.GraphEdgeSet: "pub.layers.graph.graphEdgeSet",
    r.AudioInfo: "pub.layers.media.defs#audioInfo",
    r.VideoInfo: "pub.layers.media.defs#videoInfo",
    r.DocumentInfo: "pub.layers.media.defs#documentInfo",
    r.RoleSlot: "pub.layers.ontology.defs#roleSlot",
    r.TypeDef: "pub.layers.ontology.typeDef",
}


@pytest.fixture(scope="module")
def validate_layers():  # noqa: ANN202 - returns an internal validator callable
    """Provide a callable validating ``(lexUri, value)`` pairs via @atproto/lexicon."""
    if not _LEXICON_DIR.is_dir():
        pytest.skip(
            "layers lexicons missing; run `git submodule update --init vendor/layers`"
        )
    node = shutil.which("node")
    if node is None:
        pytest.skip("node is required for ATProto lexicon validation")
    if not _INSTALLED.exists():
        npm = shutil.which("npm")
        if npm is None:
            pytest.skip("npm is required to install @atproto/lexicon")
        proc = subprocess.run(
            [npm, "install", "--no-audit", "--no-fund"],
            cwd=_VALIDATOR,
            capture_output=True,
            text=True,
            timeout=300,
        )
        if proc.returncode != 0 or not _INSTALLED.exists():
            pytest.skip(f"could not install @atproto/lexicon: {proc.stderr[:200]}")

    def _validate(pairs: list[tuple[str, object]]) -> list[dict[str, object]]:
        payload = json.dumps([{"lexUri": uri, "value": value} for uri, value in pairs])
        proc = subprocess.run(
            [node, str(_VALIDATOR / "validate.mjs")],
            input=payload,
            capture_output=True,
            text=True,
            timeout=120,
        )
        assert proc.returncode == 0, proc.stderr
        return json.loads(proc.stdout)

    return _validate


def _failures(results, pairs):  # noqa: ANN001, ANN202
    return [
        {"lexUri": uri, "error": res.get("error")}
        for (uri, _value), res in zip(pairs, results, strict=True)
        if not res["ok"]
    ]


def test_all_mirror_models_validate(validate_layers) -> None:  # noqa: ANN001
    pairs: list[tuple[str, object]] = []
    for example in (*_DEF_EXAMPLES, *_RECORD_EXAMPLES):
        lex_uri = _LEX_URI.get(type(example))
        if lex_uri is None:  # the Selector union has no standalone lexicon def
            continue
        pairs.append((lex_uri, ALL_MIRROR_ISOS[type(example)].forward(example)))
    assert not _failures(validate_layers(pairs), pairs)


def _graph() -> CorpusGraph:
    return CorpusGraph(
        nodes=(
            CorpusNode(
                node_id="sub", record=CorpusRecord(text="submission", source_name="r")
            ),
            CorpusNode(node_id="alice", node_type="entity", label="Alice"),
        ),
        edges=(
            CorpusEdge(
                source_id="sub",
                target_id="alice",
                edge_type="authored-by",
                confidence=0.9,
            ),
        ),
    )


def test_graph_bridge_outputs_validate(validate_layers) -> None:  # noqa: ANN001
    view = graph_to_layers(_graph())
    assert isinstance(view, dict)
    expressions = view["expressions"]
    graph_nodes = view["graphNodes"]
    assert isinstance(expressions, dict) and isinstance(graph_nodes, dict)
    pairs: list[tuple[str, object]] = []
    for expression in expressions.values():
        pairs.append(("pub.layers.expression.expression", expression))
    for graph_node in graph_nodes.values():
        pairs.append(("pub.layers.graph.graphNode", graph_node))
    pairs.append(("pub.layers.graph.graphEdgeSet", view["graphEdgeSet"]))
    assert not _failures(validate_layers(pairs), pairs)


def test_record_bridge_output_validates(validate_layers) -> None:  # noqa: ANN001
    view, _complement = RECORD_EXPRESSION.forward(
        CorpusRecord(text="hello", source_name="s", provenance={"author": "a"})
    )
    pairs = [("pub.layers.expression.expression", view)]
    assert not _failures(validate_layers(pairs), pairs)


def test_parse_bridge_content_validates(validate_layers) -> None:  # noqa: ANN001
    sentence = ParsedSentence(
        original_text="dogs bark",
        tokens=(
            ParsedToken(
                index=0,
                text="dogs",
                upos="NOUN",
                deprel="nsubj",
                head=1,
                start_char=0,
                end_char=4,
            ),
            ParsedToken(
                index=1,
                text="bark",
                upos="VERB",
                deprel="root",
                head=None,
                start_char=5,
                end_char=9,
            ),
        ),
    )
    view = parse_to_layers(sentence)
    assert isinstance(view, dict)
    tokenization = view["tokenization"]
    pos_layer = view["posLayer"]
    dep_layer = view["dependencyLayer"]
    assert isinstance(tokenization, dict)
    assert isinstance(pos_layer, dict) and isinstance(dep_layer, dict)
    pairs: list[tuple[str, object]] = [
        ("pub.layers.segmentation.defs#tokenization", tokenization)
    ]
    tokens = tokenization["tokens"]
    assert isinstance(tokens, tuple)
    for token in tokens:
        pairs.append(("pub.layers.segmentation.defs#token", token))
    for layer in (pos_layer, dep_layer):
        annotations = layer["annotations"]
        assert isinstance(annotations, tuple)
        for annotation in annotations:
            pairs.append(("pub.layers.annotation.defs#annotation", annotation))
    assert not _failures(validate_layers(pairs), pairs)
