"""Configurable multilingual tokenization for span annotation.

This package provides display-level tokenization that splits text into
word-level tokens for span annotation and UI display. Supports multiple
NLP backends (spaCy, Stanza, whitespace) for multilingual coverage.

Display tokens are distinct from model (subword) tokens used in active
learning. The alignment module maps between the two.
"""

from __future__ import annotations

from bead.tokenization.config import TokenizerBackend, TokenizerConfig
from bead.tokenization.parsers import (
    UNIVERSAL_DEPENDENCIES,
    ParsedSentence,
    ParsedToken,
    SpacyParser,
    StanzaParser,
    create_parser,
    parse_to_spans,
)
from bead.tokenization.tokenizers import (
    DisplayToken,
    SpacyTokenizer,
    StanzaTokenizer,
    TokenizedText,
    WhitespaceTokenizer,
    create_tokenizer,
)

__all__ = [
    "UNIVERSAL_DEPENDENCIES",
    "DisplayToken",
    "ParsedSentence",
    "ParsedToken",
    "SpacyParser",
    "SpacyTokenizer",
    "StanzaParser",
    "StanzaTokenizer",
    "TokenizedText",
    "TokenizerBackend",
    "TokenizerConfig",
    "WhitespaceTokenizer",
    "create_parser",
    "create_tokenizer",
    "parse_to_spans",
]
