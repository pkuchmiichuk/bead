"""Tokenizer configuration model."""

from __future__ import annotations

from typing import Literal

import didactic.api as dx

TokenizerBackend = Literal["spacy", "stanza", "whitespace"]


class TokenizerConfig(dx.Model):
    """Configuration for display-level tokenization.

    Attributes
    ----------
    backend : TokenizerBackend
        Tokenization backend to use. ``spacy`` (default) supports 49+
        languages. ``stanza`` covers 80+ languages including
        morphologically rich ones. ``whitespace`` is a simple fallback for
        pre-tokenized text.
    language : str
        ISO 639 language code (e.g. ``"en"``, ``"zh"``).
    model_name : str | None
        Explicit model name; auto-resolved when ``None``.
    """

    backend: TokenizerBackend = "spacy"
    language: str = "en"
    model_name: str | None = None
