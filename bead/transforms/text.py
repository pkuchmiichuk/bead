"""Pure text transforms that require no external resources.

These transforms operate on the surface string and ignore the
:class:`TransformContext`.  They are always safe to register
regardless of language.
"""

from __future__ import annotations

import html
import re
from typing import TYPE_CHECKING

from bead.transforms.base import TransformContext

if TYPE_CHECKING:
    from bead.tokenization.config import TokenizerConfig

# markdown / web text patterns (module-level so they compile once)
_MD_IMAGE = re.compile(r"!\[([^\]]*)\]\([^)]*\)")
_MD_LINK = re.compile(r"\[([^\]]*)\]\([^)]*\)")
_MD_EMPHASIS = re.compile(r"(\*\*|__|\*|_|~~)(.+?)\1")
_MD_INLINE_CODE = re.compile(r"`([^`]*)`")
_MD_HEADING = re.compile(r"^\s{0,3}#{1,6}\s*", re.MULTILINE)
_MD_BLOCKQUOTE = re.compile(r"^\s*>+\s?", re.MULTILINE)
_URL = re.compile(r"https?://\S+|www\.\S+")
_REDDIT_DELETED = re.compile(r"\[(?:deleted|removed)\]")
_WHITESPACE = re.compile(r"[^\S\n]+")
_SENTENCE_BOUNDARY = re.compile(r"(?<=[.!?])\s+(?=\S)")


class LowerTransform:
    """Convert text to lowercase.

    Examples
    --------
    >>> LowerTransform()("Hello World", TransformContext())
    'hello world'
    """

    def __call__(self, text: str, context: TransformContext) -> str:
        """Apply ``str.lower`` to *text*."""
        return text.lower()


class UpperTransform:
    """Convert text to uppercase.

    Examples
    --------
    >>> UpperTransform()("Hello World", TransformContext())
    'HELLO WORLD'
    """

    def __call__(self, text: str, context: TransformContext) -> str:
        """Apply ``str.upper`` to *text*."""
        return text.upper()


class CapitalizeTransform:
    """Capitalize the first character, lowercase the rest.

    Examples
    --------
    >>> CapitalizeTransform()("hELLO WORLD", TransformContext())
    'Hello world'
    """

    def __call__(self, text: str, context: TransformContext) -> str:
        """Apply ``str.capitalize`` to *text*."""
        return text.capitalize()


class TitleTransform:
    """Title-case each word.

    Examples
    --------
    >>> TitleTransform()("hello world", TransformContext())
    'Hello World'
    """

    def __call__(self, text: str, context: TransformContext) -> str:
        """Apply ``str.title`` to *text*."""
        return text.title()


class MarkdownStripTransform:
    """Strip common Markdown markup, keeping the human-readable text.

    Removes link/image targets (keeping the visible text), emphasis markers,
    inline code backticks, heading markers, and blockquote markers.

    Examples
    --------
    >>> MarkdownStripTransform()("**bold** and [a link](http://x)", TransformContext())
    'bold and a link'
    """

    def __call__(self, text: str, context: TransformContext) -> str:
        """Strip Markdown markup from *text*."""
        text = _MD_IMAGE.sub(r"\1", text)
        text = _MD_LINK.sub(r"\1", text)
        text = _MD_INLINE_CODE.sub(r"\1", text)
        # apply emphasis stripping repeatedly to handle nested markers
        previous = None
        while previous != text:
            previous = text
            text = _MD_EMPHASIS.sub(r"\2", text)
        text = _MD_HEADING.sub("", text)
        text = _MD_BLOCKQUOTE.sub("", text)
        return text.strip()


class RedditCleanupTransform:
    """Clean Reddit comment text into plain prose.

    Unescapes HTML entities, strips Markdown (reusing
    :class:`MarkdownStripTransform`), removes URLs and ``[deleted]``/
    ``[removed]`` markers, and collapses runs of intra-line whitespace.

    Examples
    --------
    >>> RedditCleanupTransform()("see [here](http://x) &amp; more", TransformContext())
    'see here & more'
    """

    def __init__(self) -> None:
        self._markdown = MarkdownStripTransform()

    def __call__(self, text: str, context: TransformContext) -> str:
        """Clean Reddit markup from *text*."""
        text = html.unescape(text)
        text = self._markdown(text, context)
        text = _URL.sub("", text)
        text = _REDDIT_DELETED.sub("", text)
        text = _WHITESPACE.sub(" ", text)
        return text.strip()


def split_sentences(
    text: str,
    *,
    tokenizer_config: TokenizerConfig | None = None,
) -> tuple[str, ...]:
    """Split *text* into sentences.

    When *tokenizer_config* selects a ``spacy`` or ``stanza`` backend, sentence
    boundaries come from that parser's segmenter. Otherwise a regular-expression
    fallback splits on sentence-final punctuation followed by whitespace.

    Parameters
    ----------
    text : str
        Text to split.
    tokenizer_config : TokenizerConfig | None
        Backend selector. ``None`` or the ``whitespace`` backend uses the
        regex fallback.

    Returns
    -------
    tuple[str, ...]
        The sentences, with surrounding whitespace stripped (empties dropped).
    """
    if tokenizer_config is not None and tokenizer_config.backend != "whitespace":
        from bead.tokenization.parsers import create_parser  # noqa: PLC0415

        parser = create_parser(tokenizer_config)
        return tuple(
            sentence.original_text.strip()
            for sentence in parser(text)
            if sentence.original_text.strip()
        )

    return tuple(
        part.strip() for part in _SENTENCE_BOUNDARY.split(text) if part.strip()
    )
