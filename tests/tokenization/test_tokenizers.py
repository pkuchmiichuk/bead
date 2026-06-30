"""Tests for tokenizer implementations."""

from __future__ import annotations

import didactic.api as dx
import pytest

from bead.tokenization.config import TokenizerConfig
from bead.tokenization.tokenizers import (
    DisplayToken,
    TokenizedText,
    WhitespaceTokenizer,
    create_tokenizer,
)


class TestWhitespaceTokenizer:
    """Test WhitespaceTokenizer."""

    def test_simple_sentence(self) -> None:
        """Test tokenizing a simple English sentence."""
        tokenizer = WhitespaceTokenizer()
        result = tokenizer("The cat sat on the mat.")

        assert isinstance(result, TokenizedText)
        assert result.token_texts == ("The", "cat", "sat", "on", "the", "mat.")

    def test_empty_string(self) -> None:
        """Test tokenizing empty string."""
        tokenizer = WhitespaceTokenizer()
        result = tokenizer("")

        assert result.tokens == ()
        assert result.token_texts == ()

    def test_single_word(self) -> None:
        """Test tokenizing single word."""
        tokenizer = WhitespaceTokenizer()
        result = tokenizer("Hello")

        assert result.token_texts == ("Hello",)
        assert result.tokens[0].space_after is False

    def test_space_after_flags(self) -> None:
        """Test space_after flags are correct."""
        tokenizer = WhitespaceTokenizer()
        result = tokenizer("The cat sat.")

        assert result.tokens[0].space_after is True  # "The "
        assert result.tokens[1].space_after is True  # "cat "
        assert result.tokens[2].space_after is False  # "sat." (end)

    def test_multiple_spaces(self) -> None:
        """Test handling of multiple spaces."""
        tokenizer = WhitespaceTokenizer()
        result = tokenizer("The  cat")

        # Whitespace tokenizer treats any whitespace as delimiter
        assert len(result.tokens) == 2

    def test_character_offsets(self) -> None:
        """Test character offsets are correct."""
        tokenizer = WhitespaceTokenizer()
        result = tokenizer("The cat")

        assert result.tokens[0].start_char == 0
        assert result.tokens[0].end_char == 3
        assert result.tokens[1].start_char == 4
        assert result.tokens[1].end_char == 7

    def test_round_trip(self) -> None:
        """Test that render() reproduces the original text."""
        tokenizer = WhitespaceTokenizer()
        text = "The cat sat on the mat."
        result = tokenizer(text)

        assert result.render() == text

    def test_round_trip_trailing_space(self) -> None:
        """Test round trip strips trailing space."""
        tokenizer = WhitespaceTokenizer()
        result = tokenizer("Hello world")

        assert result.render() == "Hello world"

    def test_pre_tokenized(self) -> None:
        """Test with pre-tokenized text (tab-separated)."""
        tokenizer = WhitespaceTokenizer()
        result = tokenizer("word1\tword2\tword3")

        assert len(result.tokens) == 3


class TestDisplayToken:
    """Test DisplayToken model."""

    def test_create(self) -> None:
        """Test creating a DisplayToken."""
        token = DisplayToken(
            text="hello",
            space_after=True,
            start_char=0,
            end_char=5,
        )

        assert token.text == "hello"
        assert token.space_after is True
        assert token.start_char == 0
        assert token.end_char == 5

    def test_default_space_after(self) -> None:
        """Test default space_after is True."""
        token = DisplayToken(text="hello", start_char=0, end_char=5)
        assert token.space_after is True


class TestTokenizedText:
    """Test TokenizedText model."""

    def test_token_texts(self) -> None:
        """Test token_texts property."""
        result = TokenizedText(
            tokens=(
                DisplayToken(text="The", start_char=0, end_char=3),
                DisplayToken(text="cat", start_char=4, end_char=7),
            ),
            original_text="The cat",
        )

        assert result.token_texts == ("The", "cat")

    def test_space_after_flags(self) -> None:
        """Test space_after_flags property."""
        result = TokenizedText(
            tokens=(
                DisplayToken(text="The", space_after=True, start_char=0, end_char=3),
                DisplayToken(text="cat", space_after=False, start_char=4, end_char=7),
            ),
            original_text="The cat",
        )

        assert result.space_after_flags == (True, False)

    def test_render(self) -> None:
        """Test render reconstructs text."""
        result = TokenizedText(
            tokens=(
                DisplayToken(text="The", space_after=True, start_char=0, end_char=3),
                DisplayToken(text="cat", space_after=True, start_char=4, end_char=7),
                DisplayToken(text="sat.", space_after=False, start_char=8, end_char=12),
            ),
            original_text="The cat sat.",
        )

        assert result.render() == "The cat sat."

    def test_render_no_trailing_space(self) -> None:
        """Test render strips trailing spaces."""
        result = TokenizedText(
            tokens=(
                DisplayToken(text="hello", space_after=True, start_char=0, end_char=5),
            ),
            original_text="hello ",
        )

        assert result.render() == "hello"


class TestCreateTokenizer:
    """Test create_tokenizer factory."""

    def test_whitespace_backend(self) -> None:
        """Test creating whitespace tokenizer."""
        config = TokenizerConfig(backend="whitespace")
        tokenizer = create_tokenizer(config)

        result = tokenizer("Hello world")
        assert result.token_texts == ("Hello", "world")

    def test_unknown_backend_raises(self) -> None:
        """Test that unknown backend raises ValueError."""
        with pytest.raises(dx.ValidationError):
            TokenizerConfig.model_validate({"backend": "unknown"})

    def test_spacy_backend_without_install(self) -> None:
        """Test that spaCy backend works or raises ImportError gracefully."""
        config = TokenizerConfig(backend="spacy", language="en")
        tokenizer = create_tokenizer(config)
        # Just test that the factory returns something callable
        assert callable(tokenizer)

    def test_default_config(self) -> None:
        """Test default config uses spacy."""
        config = TokenizerConfig()
        assert config.backend == "spacy"
        assert config.language == "en"
        assert config.model_name is None
