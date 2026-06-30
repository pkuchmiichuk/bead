"""Lossless iso between a dependency parse and layers annotation records.

A :class:`~bead.tokenization.parsers.ParsedSentence` maps to a
:class:`ParsedSentenceLayers` view: a canonical
:class:`lairs.records.segmentation.Tokenization` plus two
:class:`lairs.records.annotation.AnnotationLayer` records (a part-of-speech
``token-tag`` layer and a dependency ``relation`` layer).
``ParsedToken``/``ParsedSentence`` carry no framework identity, so the mapping is
a true bijection (``dx.Iso``).

The layers ``token`` has no ``spaceAfter`` slot, so each token's space-after flag
travels in its part-of-speech annotation's features alongside the lemma, xpos,
and morphology. The annotation layers require an expression reference and a
timestamp that the parse does not carry; these are emitted as fixed constants and
ignored on the way back.
"""

from __future__ import annotations

from datetime import UTC, datetime

import didactic.api as dx
from lairs.records import annotation, defs, segmentation

from bead.interop.layers._convert import feature_map, read_feature_map
from bead.tokenization.parsers import (
    UNIVERSAL_DEPENDENCIES,
    ParsedSentence,
    ParsedToken,
)

_ROOT_HEAD = -1
_TOKENIZATION_UUID = "tokenization"
# The annotation layers require an expression reference and timestamp that a bare
# parse does not carry; these fixed constants are emitted forward and ignored
# backward, so the bijection holds.
_PARSE_EXPRESSION_URI = "at://local/parse"
_PARSE_CREATED_AT = datetime(1970, 1, 1, tzinfo=UTC)


def _opt_str(value: object) -> str | None:
    if value is None or isinstance(value, str):
        return value
    return str(value)


def _char(value: int | None) -> int:
    """Return a character offset, defaulting absent offsets to 0."""
    return value if value is not None else 0


class ParsedSentenceLayers(dx.Model):
    """A layers view of a parse: a tokenization plus pos and dependency layers."""

    original_text: str = dx.field()
    tokenization: dx.Embed[segmentation.Tokenization] = dx.field()
    pos_layer: dx.Embed[annotation.AnnotationLayer] = dx.field()
    dependency_layer: dx.Embed[annotation.AnnotationLayer] = dx.field()


class ParsedSentenceLayersIso(dx.Iso[ParsedSentence, ParsedSentenceLayers]):
    """Lossless ``ParsedSentence <-> layers tokenization + annotation layers``."""

    def forward(self, sentence: ParsedSentence) -> ParsedSentenceLayers:
        """Project a parsed sentence to layers tokenization + annotations."""
        text = sentence.original_text

        def _byte(char_index: int) -> int:
            return len(text[:char_index].encode("utf-8"))

        tokens = tuple(
            segmentation.Token(
                tokenIndex=token.index,
                text=token.text,
                textSpan=defs.Span(
                    byteStart=_byte(token.start_char),
                    byteEnd=_byte(token.end_char),
                    charStart=token.start_char,
                    charEnd=token.end_char,
                ),
            )
            for token in sentence.tokens
        )
        pos_annotations = tuple(
            annotation.Annotation(
                uuid=defs.Uuid(value=f"pos-{token.index}"),
                tokenIndex=token.index,
                label=token.upos,
                features=feature_map(
                    {
                        "xpos": token.xpos,
                        "lemma": token.lemma,
                        "morph": dict(token.morph),
                        "spaceAfter": token.space_after,
                    }
                ),
            )
            for token in sentence.tokens
        )
        dependency_annotations = tuple(
            annotation.Annotation(
                uuid=defs.Uuid(value=f"dep-{token.index}"),
                tokenIndex=token.index,
                label=token.deprel,
                headIndex=token.head if token.head is not None else _ROOT_HEAD,
            )
            for token in sentence.tokens
        )
        tokenization = segmentation.Tokenization(
            uuid=defs.Uuid(value=_TOKENIZATION_UUID),
            kind="custom",
            tokens=tokens,
        )
        return ParsedSentenceLayers(
            original_text=text,
            tokenization=tokenization,
            pos_layer=annotation.AnnotationLayer(
                annotations=pos_annotations,
                createdAt=_PARSE_CREATED_AT,
                expression=_PARSE_EXPRESSION_URI,
                kind="token-tag",
                subkind="pos",
                formalism=UNIVERSAL_DEPENDENCIES,
                tokenizationId=defs.Uuid(value=_TOKENIZATION_UUID),
            ),
            dependency_layer=annotation.AnnotationLayer(
                annotations=dependency_annotations,
                createdAt=_PARSE_CREATED_AT,
                expression=_PARSE_EXPRESSION_URI,
                kind="relation",
                subkind="dependency",
                formalism=UNIVERSAL_DEPENDENCIES,
                tokenizationId=defs.Uuid(value=_TOKENIZATION_UUID),
            ),
        )

    def backward(self, view: ParsedSentenceLayers) -> ParsedSentence:
        """Reconstruct a parsed sentence from its layers projection."""
        tokens: list[ParsedToken] = []
        for token, pos, dep in zip(
            view.tokenization.tokens,
            view.pos_layer.annotations,
            view.dependency_layer.annotations,
            strict=True,
        ):
            features = read_feature_map(pos.features)
            raw_morph = features.get("morph")
            morph = (
                {key: str(value) for key, value in raw_morph.items()}
                if isinstance(raw_morph, dict)
                else {}
            )
            head_index = dep.headIndex if dep.headIndex is not None else _ROOT_HEAD
            span = token.textSpan
            tokens.append(
                ParsedToken(
                    index=token.tokenIndex,
                    text=token.text if token.text is not None else "",
                    lemma=_opt_str(features.get("lemma")),
                    upos=pos.label,
                    xpos=_opt_str(features.get("xpos")),
                    deprel=dep.label,
                    head=None if head_index == _ROOT_HEAD else head_index,
                    morph=morph,
                    space_after=bool(features.get("spaceAfter")),
                    start_char=_char(span.charStart if span is not None else None),
                    end_char=_char(span.charEnd if span is not None else None),
                )
            )
        return ParsedSentence(original_text=view.original_text, tokens=tuple(tokens))


PARSED_SENTENCE_LAYERS = ParsedSentenceLayersIso()


def parse_to_layers(sentence: ParsedSentence) -> ParsedSentenceLayers:
    """Return the layers tokenization + annotation-layer view of a parse."""
    return PARSED_SENTENCE_LAYERS.forward(sentence)
