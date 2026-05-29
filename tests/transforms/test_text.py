"""Tests for text transforms."""

from __future__ import annotations

from bead.tokenization.config import TokenizerConfig
from bead.transforms.base import TransformContext
from bead.transforms.text import (
    CapitalizeTransform,
    LowerTransform,
    MarkdownStripTransform,
    RedditCleanupTransform,
    TitleTransform,
    UpperTransform,
    split_sentences,
)


class TestLowerTransform:
    """Tests for LowerTransform."""

    def test_basic(self) -> None:
        assert LowerTransform()("HELLO", TransformContext()) == "hello"

    def test_mixed_case(self) -> None:
        assert LowerTransform()("HeLLo WoRLd", TransformContext()) == "hello world"

    def test_already_lower(self) -> None:
        assert LowerTransform()("hello", TransformContext()) == "hello"


class TestUpperTransform:
    """Tests for UpperTransform."""

    def test_basic(self) -> None:
        assert UpperTransform()("hello", TransformContext()) == "HELLO"

    def test_already_upper(self) -> None:
        assert UpperTransform()("HELLO", TransformContext()) == "HELLO"


class TestCapitalizeTransform:
    """Tests for CapitalizeTransform."""

    def test_basic(self) -> None:
        assert CapitalizeTransform()("hello world", TransformContext()) == "Hello world"

    def test_all_caps_input(self) -> None:
        assert CapitalizeTransform()("HELLO WORLD", TransformContext()) == "Hello world"


class TestTitleTransform:
    """Tests for TitleTransform."""

    def test_basic(self) -> None:
        assert TitleTransform()("hello world", TransformContext()) == "Hello World"

    def test_already_title(self) -> None:
        assert TitleTransform()("Hello World", TransformContext()) == "Hello World"


class TestMarkdownStripTransform:
    """Tests for MarkdownStripTransform."""

    def test_link(self) -> None:
        out = MarkdownStripTransform()("see [the docs](http://x)", TransformContext())
        assert out == "see the docs"

    def test_emphasis(self) -> None:
        out = MarkdownStripTransform()("**bold** and *italic*", TransformContext())
        assert out == "bold and italic"

    def test_inline_code_and_heading(self) -> None:
        out = MarkdownStripTransform()("# Title `code`", TransformContext())
        assert out == "Title code"

    def test_blockquote(self) -> None:
        out = MarkdownStripTransform()("> quoted text", TransformContext())
        assert out == "quoted text"


class TestRedditCleanupTransform:
    """Tests for RedditCleanupTransform."""

    def test_unescape_and_markdown(self) -> None:
        out = RedditCleanupTransform()(
            "see [here](http://x) &amp; more", TransformContext()
        )
        assert out == "see here & more"

    def test_removes_url_and_deleted(self) -> None:
        out = RedditCleanupTransform()(
            "check https://example.com [deleted]", TransformContext()
        )
        assert out == "check"

    def test_collapses_whitespace(self) -> None:
        out = RedditCleanupTransform()("a    b\tc", TransformContext())
        assert out == "a b c"


class TestSplitSentences:
    """Tests for split_sentences."""

    def test_regex_fallback(self) -> None:
        result = split_sentences("Hello world. How are you? Fine!")
        assert result == ("Hello world.", "How are you?", "Fine!")

    def test_single_sentence(self) -> None:
        assert split_sentences("Just one sentence") == ("Just one sentence",)

    def test_empty(self) -> None:
        assert split_sentences("") == ()

    def test_whitespace_backend_uses_fallback(self) -> None:
        result = split_sentences(
            "One. Two.", tokenizer_config=TokenizerConfig(backend="whitespace")
        )
        assert result == ("One.", "Two.")
