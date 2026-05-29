"""Tests for DSL structural-query builtins over a dependency parse.

Also includes the layers no-drop smoke test: every field a layers dependency
``AnnotationLayer``/``Annotation`` needs must be reconstructable from a parsed
``Item``.
"""

from __future__ import annotations

from uuid import uuid4

from bead.dsl.evaluator import DSLEvaluator
from bead.items.item import Item
from bead.tokenization.parsers import (
    UNIVERSAL_DEPENDENCIES,
    ParsedSentence,
    ParsedToken,
    parse_to_spans,
)


def _known_sentence() -> ParsedSentence:
    """Hand-built parse of 'The dog chased the cat'."""
    return ParsedSentence(
        original_text="The dog chased the cat",
        tokens=(
            ParsedToken(index=0, text="The", lemma="the", upos="DET",
                        deprel="det", head=1, start_char=0, end_char=3),
            ParsedToken(index=1, text="dog", lemma="dog", upos="NOUN",
                        deprel="nsubj", head=2, morph={"Number": "Sing"},
                        start_char=4, end_char=7),
            ParsedToken(index=2, text="chased", lemma="chase", upos="VERB",
                        deprel="root", head=None, morph={"Tense": "Past"},
                        start_char=8, end_char=14),
            ParsedToken(index=3, text="the", lemma="the", upos="DET",
                        deprel="det", head=4, start_char=15, end_char=18),
            ParsedToken(index=4, text="cat", lemma="cat", upos="NOUN",
                        deprel="obj", head=2, morph={"Number": "Sing"},
                        start_char=19, end_char=22),
        ),
    )


def _parsed_item() -> Item:
    sentence = _known_sentence()
    spans, relations = parse_to_spans(
        sentence, element_name="text", tokenization_id="tok-1", tool="test"
    )
    return Item(
        item_template_id=uuid4(),
        rendered_elements={"text": sentence.original_text},
        spans=spans,
        span_relations=relations,
        tokenized_elements={"text": tuple(t.text for t in sentence.tokens)},
    )


def _eval(expression: str):
    item = _parsed_item()
    return DSLEvaluator().evaluate(expression, {"self": item, "item": item})


class TestTokenAttributeBuiltins:
    """Tests for per-token attribute accessors."""

    def test_upos(self) -> None:
        assert _eval("upos(self, 2)") == "VERB"
        assert _eval("upos(self, 1)") == "NOUN"

    def test_lemma_and_deprel(self) -> None:
        assert _eval("lemma_of(self, 2)") == "chase"
        assert _eval("deprel(self, 1)") == "nsubj"
        assert _eval("deprel(self, 2)") == "root"

    def test_morph(self) -> None:
        assert _eval("morph(self, 1, 'Number')") == "Sing"
        assert _eval("morph(self, 2, 'Tense')") == "Past"
        assert _eval("morph(self, 0, 'Number')") is None

    def test_missing_token(self) -> None:
        assert _eval("upos(self, 99)") is None


class TestGraphBuiltins:
    """Tests for graph traversal accessors."""

    def test_root(self) -> None:
        assert _eval("root(self)") == 2

    def test_head(self) -> None:
        assert _eval("head(self, 1)") == 2
        assert _eval("head(self, 0)") == 1
        assert _eval("head(self, 2)") is None  # root

    def test_dependents(self) -> None:
        assert _eval("dependents(self, 2)") == [1, 4]
        assert _eval("dependents(self, 2, 'obj')") == [4]
        assert _eval("dependents(self, 2, 'nsubj')") == [1]
        assert _eval("dependents(self, 0)") == []

    def test_has_relation(self) -> None:
        assert _eval("has_relation(self, 2, 4, 'obj')") is True
        assert _eval("has_relation(self, 2, 1, 'obj')") is False
        assert _eval("has_relation(self, 2, 4)") is True

    def test_tokens_with(self) -> None:
        assert _eval("tokens_with_upos(self, 'NOUN')") == [1, 4]
        assert _eval("tokens_with_deprel(self, 'det')") == [0, 3]

    def test_path_to_root(self) -> None:
        assert _eval("path_to_root(self, 0)") == [0, 1, 2]

    def test_subtree(self) -> None:
        assert _eval("subtree(self, 2)") == [0, 1, 2, 3, 4]
        assert _eval("subtree(self, 4)") == [3, 4]

    def test_helpers_avoid_comprehensions(self) -> None:
        assert _eval("any_deprel(self, [0, 1], 'nsubj')") is True
        assert _eval("filter_upos(self, [0, 1, 2], 'DET')") == [0]


class TestStructuralConstraints:
    """Tests for full constraint expressions over structure."""

    def test_transitive_verb_constraint(self) -> None:
        expr = (
            'upos(self, root(self)) == "VERB" '
            'and len(dependents(self, root(self), "obj")) > 0'
        )
        assert _eval(expr) is True

    def test_intransitive_fails_object_check(self) -> None:
        # 'cat' (index 4) has no object dependent
        assert _eval('len(dependents(self, 4, "obj")) > 0') is False


class TestLayersNoDropSmoke:
    """Every field a layers dependency annotation needs is reconstructable."""

    def test_all_layers_fields_present(self) -> None:
        item = _parsed_item()
        token_spans = {
            span.segments[0].indices[0]: span
            for span in item.spans
            if span.span_type == "token"
        }
        # one token span per token
        assert set(token_spans) == {0, 1, 2, 3, 4}

        for span in token_spans.values():
            md = span.span_metadata
            # layer-level discriminators
            assert md["tokenization_id"] == "tok-1"
            assert md["formalism"] == UNIVERSAL_DEPENDENCIES
            assert md["tool"] == "test"
            # per-token annotation fields
            assert "upos" in md
            assert "lemma" in md
            assert "deprel" in md
            # char offsets (layers' canonical byte offsets derive from these)
            assert isinstance(md["start_char"], int)
            assert isinstance(md["end_char"], int)
            # head_index present (None only for the root)
            if md["deprel"] != "root":
                assert span.head_index is not None

        # arcs reconstructable as head -> dependent with a deprel label
        for relation in item.span_relations:
            assert relation.directed
            assert relation.label is not None
            assert relation.source_span_id in {s.span_id for s in item.spans}
            assert relation.target_span_id in {s.span_id for s in item.spans}

    def test_reconstruct_conllu_like_rows(self) -> None:
        """Reconstruct (id, form, upos, head, deprel) rows from the Item."""
        item = _parsed_item()
        evaluator = DSLEvaluator()
        rows = []
        for index in range(5):
            ctx = {"self": item, "item": item}
            rows.append(
                (
                    index,
                    evaluator.evaluate(f"upos(self, {index})", ctx),
                    evaluator.evaluate(f"head(self, {index})", ctx),
                    evaluator.evaluate(f"deprel(self, {index})", ctx),
                )
            )

        assert rows == [
            (0, "DET", 1, "det"),
            (1, "NOUN", 2, "nsubj"),
            (2, "VERB", None, "root"),
            (3, "DET", 4, "det"),
            (4, "NOUN", 2, "obj"),
        ]
