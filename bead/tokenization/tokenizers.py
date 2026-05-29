"""Concrete tokenizer implementations.

Provides display-level tokenizers for span annotation. Each tokenizer
converts raw text into a sequence of ``DisplayToken`` objects that carry
rendering metadata (``space_after``) for artifact-free reconstruction.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Iterator
from typing import Protocol

import didactic.api as dx

from bead.tokenization.config import TokenizerConfig


class DisplayToken(dx.Model):
    """A word-level token with rendering metadata.

    Attributes
    ----------
    text : str
        The token text.
    space_after : bool
        Whether whitespace follows this token in the original text.
    start_char : int
        Character offset of the token start in the original text.
    end_char : int
        Character offset of the token end in the original text.
    """

    text: str
    start_char: int
    end_char: int
    space_after: bool = True


class TokenizedText(dx.Model):
    """Result of display-level tokenization.

    Attributes
    ----------
    tokens : tuple[DisplayToken, ...]
        The sequence of display tokens.
    original_text : str
        The original input text.
    """

    original_text: str
    tokens: tuple[dx.Embed[DisplayToken], ...] = ()

    @property
    def token_texts(self) -> tuple[str, ...]:
        """Plain token strings (for ``Item.tokenized_elements``)."""
        return tuple(t.text for t in self.tokens)

    @property
    def space_after_flags(self) -> tuple[bool, ...]:
        """Per-token ``space_after`` flags (for ``Item.token_space_after``)."""
        return tuple(t.space_after for t in self.tokens)

    def render(self) -> str:
        """Reconstruct display text from tokens with correct spacing.

        Guarantees identical rendering to original when round-tripped.

        Returns
        -------
        str
            Reconstructed text.
        """
        parts: list[str] = []
        for token in self.tokens:
            parts.append(token.text)
            if token.space_after:
                parts.append(" ")
        return "".join(parts).rstrip()


def spacy_space_after(token: _SpacyTokenProtocol) -> bool:
    """Whether whitespace follows a spaCy token in the source text.

    Shared by ``SpacyTokenizer`` and ``SpacyParser`` (single canonical site).
    """
    return token.whitespace_ != ""


def _stanza_space_after(token: _StanzaTokenProtocol, text: str) -> bool:
    """Whether whitespace follows a Stanza token in the source text.

    Prefers the CoNLL-U ``SpaceAfter=No`` annotation when present, falling
    back to inspecting the character immediately after the token.
    """
    if getattr(token, "misc", None):
        return "SpaceAfter=No" not in (token.misc or "")
    if token.end_char < len(text):
        return text[token.end_char] == " "
    return True


class WhitespaceTokenizer:
    """Simple whitespace-split tokenizer.

    Fallback for pre-tokenized text or languages not supported by spaCy
    or Stanza. Splits on whitespace boundaries and infers ``space_after``
    from the original character offsets.
    """

    def __call__(self, text: str) -> TokenizedText:
        """Tokenize text by splitting on whitespace.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        TokenizedText
            Tokenized result.
        """
        tokens: list[DisplayToken] = []
        for match in re.finditer(r"\S+", text):
            start = match.start()
            end = match.end()
            space_after = end < len(text) and text[end] == " "
            tokens.append(
                DisplayToken(
                    text=match.group(),
                    space_after=space_after,
                    start_char=start,
                    end_char=end,
                )
            )
        return TokenizedText(tokens=tuple(tokens), original_text=text)


class SpacyTokenizer:
    """spaCy-based tokenizer.

    Supports 49+ languages. Auto-resolves model from language code if
    ``model_name`` is not specified. Handles punctuation attachment and
    multi-word token (MWT) expansion correctly.

    Parameters
    ----------
    language : str
        ISO 639 language code.
    model_name : str | None
        Explicit spaCy model name. When None, uses ``{language}_core_web_sm``
        for common languages, falling back to a blank model.
    """

    def __init__(self, language: str = "en", model_name: str | None = None) -> None:
        self._language = language
        self._model_name = model_name
        self._nlp: Callable[..., _SpacyDocProtocol] | None = None

    def _load(self) -> Callable[..., _SpacyDocProtocol]:
        if self._nlp is not None:
            return self._nlp

        try:
            import spacy  # noqa: PLC0415  # type: ignore[reportMissingImports]
        except ImportError as e:
            raise ImportError(
                "spaCy is required for SpacyTokenizer. "
                "Install it with: pip install 'bead[tokenization]'"
            ) from e

        model = self._model_name
        if model is None:
            model = f"{self._language}_core_web_sm"

        try:
            nlp: Callable[..., _SpacyDocProtocol] = spacy.load(model)  # type: ignore[assignment]
        except OSError:
            # fall back to blank model
            nlp = spacy.blank(self._language)  # type: ignore[assignment]

        self._nlp = nlp
        return nlp

    def __call__(self, text: str) -> TokenizedText:
        """Tokenize text using spaCy.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        TokenizedText
            Tokenized result with correct ``space_after`` metadata.
        """
        nlp = self._load()
        doc = nlp(text)
        tokens: list[DisplayToken] = []
        for token in doc:
            tokens.append(
                DisplayToken(
                    text=token.text,
                    space_after=spacy_space_after(token),
                    start_char=token.idx,
                    end_char=token.idx + len(token.text),
                )
            )
        return TokenizedText(tokens=tuple(tokens), original_text=text)


class StanzaTokenizer:
    """Stanza-based tokenizer.

    Supports 80+ languages. Handles multi-word token (MWT) expansion for
    languages like German, French, and Arabic. Better coverage for
    low-resource and morphologically rich languages.

    Parameters
    ----------
    language : str
        ISO 639 language code.
    model_name : str | None
        Explicit Stanza model/package name. When None, uses the default
        package for the language.
    """

    def __init__(self, language: str = "en", model_name: str | None = None) -> None:
        self._language = language
        self._model_name = model_name
        self._nlp: _StanzaPipelineProtocol | None = None

    def _load(self) -> _StanzaPipelineProtocol:
        if self._nlp is not None:
            return self._nlp

        try:
            import stanza  # noqa: PLC0415  # type: ignore[reportMissingImports]
        except ImportError as e:
            raise ImportError(
                "Stanza is required for StanzaTokenizer. "
                "Install it with: pip install 'bead[tokenization]'"
            ) from e

        pkg = self._model_name
        pkg_kwarg = {"package": pkg} if pkg is not None else {}

        try:
            nlp: _StanzaPipelineProtocol = stanza.Pipeline(  # type: ignore[assignment]
                lang=self._language,
                processors="tokenize",
                verbose=False,
                **pkg_kwarg,  # type: ignore[reportArgumentType]
            )
        except Exception:
            # download model and retry
            stanza.download(self._language, verbose=False)
            nlp = stanza.Pipeline(  # type: ignore[assignment]
                lang=self._language,
                processors="tokenize",
                verbose=False,
                **pkg_kwarg,  # type: ignore[reportArgumentType]
            )

        self._nlp = nlp
        return nlp

    def __call__(self, text: str) -> TokenizedText:
        """Tokenize text using Stanza.

        Parameters
        ----------
        text : str
            Input text.

        Returns
        -------
        TokenizedText
            Tokenized result with correct ``space_after`` metadata.
        """
        nlp = self._load()
        doc = nlp(text)
        tokens: list[DisplayToken] = []
        for sentence in doc.sentences:
            for token in sentence.tokens:
                tokens.append(
                    DisplayToken(
                        text=token.text,
                        space_after=_stanza_space_after(token, text),
                        start_char=token.start_char,
                        end_char=token.end_char,
                    )
                )
        return TokenizedText(tokens=tuple(tokens), original_text=text)


def create_tokenizer(config: TokenizerConfig) -> Callable[[str], TokenizedText]:
    """Return a tokenization function for the given config.

    Lazy-loads the NLP backend (spaCy/Stanza) on first call.

    Parameters
    ----------
    config : TokenizerConfig
        Tokenizer configuration.

    Returns
    -------
    Callable[[str], TokenizedText]
        A callable that tokenizes text.

    Raises
    ------
    ValueError
        If the backend is not recognized.
    """
    if config.backend == "whitespace":
        return WhitespaceTokenizer()
    elif config.backend == "spacy":
        return SpacyTokenizer(language=config.language, model_name=config.model_name)
    elif config.backend == "stanza":
        return StanzaTokenizer(language=config.language, model_name=config.model_name)
    else:
        raise ValueError(f"Unknown tokenizer backend: {config.backend}")


# structural typing protocols for spaCy/Stanza (avoids hard imports)
class _SpacyTokenProtocol(Protocol):
    text: str
    whitespace_: str
    idx: int


class _SpacyDocProtocol(Protocol):
    def __iter__(self) -> Iterator[_SpacyTokenProtocol]: ...  # noqa: D105


class _StanzaTokenProtocol(Protocol):
    text: str
    start_char: int
    end_char: int
    misc: str | None


class _StanzaSentenceProtocol(Protocol):
    tokens: list[_StanzaTokenProtocol]


class _StanzaDocProtocol(Protocol):
    sentences: list[_StanzaSentenceProtocol]


class _StanzaPipelineProtocol(Protocol):
    def __call__(self, text: str) -> _StanzaDocProtocol: ...  # noqa: D102
