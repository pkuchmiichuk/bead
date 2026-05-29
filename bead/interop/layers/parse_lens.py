"""Lossless iso between a dependency parse and layers annotation records.

A :class:`~bead.tokenization.parsers.ParsedSentence` maps to a layers
``tokenization`` plus two annotation layers (a part-of-speech ``token-tag``
layer and a ``dependency`` ``relation`` layer). ``ParsedToken``/``ParsedSentence``
carry no framework identity, so the mapping is a true bijection (``dx.Iso``):
the layers view captures everything and reconstructs the parse exactly.
"""

from __future__ import annotations

import didactic.api as dx

from bead.data.base import JsonValue
from bead.interop.layers._convert import (
    from_feature_map,
    j_bool,
    j_int,
    j_list,
    j_obj,
    j_str,
    j_str_or_none,
    strip_nulls,
    to_feature_map,
)
from bead.tokenization.parsers import (
    UNIVERSAL_DEPENDENCIES,
    ParsedSentence,
    ParsedToken,
)

_ROOT_HEAD = -1


def _opt_str(value: JsonValue) -> str | None:
    if value is None or isinstance(value, str):
        return value
    return str(value)


class ParsedSentenceLayersIso(dx.Iso[ParsedSentence, JsonValue]):
    """Lossless ``ParsedSentence <-> layers tokenization + annotation layers``."""

    def forward(self, sentence: ParsedSentence) -> JsonValue:
        """Project a parsed sentence to layers tokenization + annotations."""
        text = sentence.original_text

        def _byte(char_index: int) -> int:
            return len(text[:char_index].encode("utf-8"))

        token_views: tuple[JsonValue, ...] = tuple(
            {
                "tokenIndex": token.index,
                "text": token.text,
                "textSpan": {
                    "byteStart": _byte(token.start_char),
                    "byteEnd": _byte(token.end_char),
                    "charStart": token.start_char,
                    "charEnd": token.end_char,
                },
                "spaceAfter": token.space_after,
            }
            for token in sentence.tokens
        )
        pos_annotations: tuple[JsonValue, ...] = tuple(
            {
                "uuid": {"value": f"pos-{token.index}"},
                "tokenIndex": token.index,
                "label": token.upos,
                "features": to_feature_map(
                    {
                        "xpos": token.xpos,
                        "lemma": token.lemma,
                        "morph": dict(token.morph),
                    }
                ),
            }
            for token in sentence.tokens
        )
        dependency_annotations: tuple[JsonValue, ...] = tuple(
            {
                "uuid": {"value": f"dep-{token.index}"},
                "tokenIndex": token.index,
                "label": token.deprel,
                "headIndex": token.head if token.head is not None else _ROOT_HEAD,
            }
            for token in sentence.tokens
        )
        return strip_nulls(
            {
                "originalText": sentence.original_text,
                "tokenization": {
                    "uuid": {"value": "tokenization"},
                    "kind": "parser",
                    "tokens": token_views,
                },
                "posLayer": {
                    "kind": "token-tag",
                    "subkind": "pos",
                    "formalism": UNIVERSAL_DEPENDENCIES,
                    "annotations": pos_annotations,
                },
                "dependencyLayer": {
                    "kind": "relation",
                    "subkind": "dependency",
                    "formalism": UNIVERSAL_DEPENDENCIES,
                    "annotations": dependency_annotations,
                },
            }
        )

    def backward(self, view: JsonValue) -> ParsedSentence:
        """Reconstruct a parsed sentence from its layers projection."""
        view_obj = j_obj(view)
        tokenization = j_obj(view_obj["tokenization"])
        token_views = j_list(tokenization["tokens"])
        pos_annotations = j_list(j_obj(view_obj["posLayer"])["annotations"])
        dep_annotations = j_list(j_obj(view_obj["dependencyLayer"])["annotations"])

        tokens: list[ParsedToken] = []
        for token_value, pos_value, dep_value in zip(
            token_views, pos_annotations, dep_annotations, strict=True
        ):
            token_obj = j_obj(token_value)
            pos_obj = j_obj(pos_value)
            dep_obj = j_obj(dep_value)
            span = j_obj(token_obj["textSpan"])
            features = from_feature_map(pos_obj["features"])
            raw_morph = features.get("morph")
            morph = (
                {key: str(value) for key, value in raw_morph.items()}
                if isinstance(raw_morph, dict)
                else {}
            )
            head_index = j_int(dep_obj["headIndex"])
            tokens.append(
                ParsedToken(
                    index=j_int(token_obj["tokenIndex"]),
                    text=j_str(token_obj["text"]),
                    lemma=_opt_str(features.get("lemma")),
                    upos=j_str_or_none(pos_obj.get("label")),
                    xpos=_opt_str(features.get("xpos")),
                    deprel=j_str_or_none(dep_obj.get("label")),
                    head=None if head_index == _ROOT_HEAD else head_index,
                    morph=morph,
                    space_after=j_bool(token_obj["spaceAfter"]),
                    start_char=j_int(span["charStart"]),
                    end_char=j_int(span["charEnd"]),
                )
            )
        return ParsedSentence(
            original_text=j_str(view_obj["originalText"]), tokens=tuple(tokens)
        )


PARSED_SENTENCE_LAYERS = ParsedSentenceLayersIso()


def parse_to_layers(sentence: ParsedSentence) -> JsonValue:
    """Return the layers tokenization + annotation-layer view of a parse."""
    return PARSED_SENTENCE_LAYERS.forward(sentence)
