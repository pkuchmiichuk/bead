"""Tests for the transform base abstractions."""

from __future__ import annotations

import pytest

from bead.transforms.base import (
    TransformContext,
    TransformPipeline,
    TransformRegistry,
)
from bead.transforms.text import LowerTransform


class TestTransformContext:
    """Tests for TransformContext."""

    def test_defaults(self) -> None:
        """All fields default to None or empty."""
        ctx = TransformContext()

        assert ctx.language_code is None
        assert ctx.lemma is None
        assert ctx.pos is None
        assert ctx.head_index is None
        assert ctx.tokens == ()
        assert ctx.metadata == {}

    def test_fields_set(self) -> None:
        """Fields can be set via constructor."""
        ctx = TransformContext(
            language_code="eng",
            lemma="run",
            pos="VERB",
            head_index=0,
            tokens=["run", "to"],
            metadata={"key": "value"},
        )

        assert ctx.language_code == "eng"
        assert ctx.lemma == "run"
        assert ctx.pos == "VERB"
        assert ctx.head_index == 0
        assert ctx.tokens == ("run", "to")
        assert ctx.metadata == {"key": "value"}

    def test_frozen(self) -> None:
        """TransformContext is immutable."""
        ctx = TransformContext()

        with pytest.raises(AttributeError):
            ctx.lemma = "walk"


class TestTransformPipeline:
    """Tests for TransformPipeline."""

    def test_empty_pipeline_passes_through(self) -> None:
        """Empty pipeline returns text unchanged."""
        pipe = TransformPipeline()
        ctx = TransformContext()

        assert pipe("hello", ctx) == "hello"

    def test_single_transform(self) -> None:
        """Pipeline with one transform applies it."""
        pipe = TransformPipeline([lambda t, c: t.upper()])
        ctx = TransformContext()

        assert pipe("hello", ctx) == "HELLO"

    def test_chained_transforms(self) -> None:
        """Pipeline applies transforms left to right."""
        pipe = TransformPipeline(
            [
                lambda t, c: t.upper(),
                lambda t, c: t + "!",
            ]
        )
        ctx = TransformContext()

        assert pipe("hello", ctx) == "HELLO!"

    def test_len(self) -> None:
        """Pipeline reports correct length."""
        pipe = TransformPipeline([lambda t, c: t, lambda t, c: t])

        assert len(pipe) == 2

    def test_append(self) -> None:
        """Append adds a transform to the end."""
        pipe = TransformPipeline([lambda t, c: t.upper()])
        pipe.append(lambda t, c: t + "!")
        ctx = TransformContext()

        assert pipe("hi", ctx) == "HI!"

    def test_prepend(self) -> None:
        """Prepend inserts a transform at the start."""
        pipe = TransformPipeline([lambda t, c: t + "!"])
        pipe.prepend(lambda t, c: t.upper())
        ctx = TransformContext()

        assert pipe("hi", ctx) == "HI!"

    def test_repr(self) -> None:
        """Pipeline has an informative repr."""
        pipe = TransformPipeline([LowerTransform()])

        assert "LowerTransform" in repr(pipe)


class TestTransformRegistry:
    """Tests for TransformRegistry."""

    def test_register_and_get(self) -> None:
        """Registered transform can be retrieved."""
        reg = TransformRegistry()
        reg.register("shout", lambda t, c: t.upper())

        assert reg.get("shout")("hello", TransformContext()) == "HELLO"

    def test_case_insensitive(self) -> None:
        """Names are case-insensitive."""
        reg = TransformRegistry()
        reg.register("Shout", lambda t, c: t.upper())

        assert reg.get("shout")("hi", TransformContext()) == "HI"
        assert reg.get("SHOUT")("hi", TransformContext()) == "HI"

    def test_missing_raises_key_error(self) -> None:
        """Getting an unregistered name raises KeyError."""
        reg = TransformRegistry()

        with pytest.raises(KeyError, match="nonexistent"):
            reg.get("nonexistent")

    def test_empty_name_raises_value_error(self) -> None:
        """Empty name is rejected."""
        reg = TransformRegistry()

        with pytest.raises(ValueError, match="non-empty"):
            reg.register("", lambda t, c: t)

    def test_resolve_pipeline(self) -> None:
        """Pipeline is built from a list of names."""
        reg = TransformRegistry()
        reg.register("upper", lambda t, c: t.upper())
        reg.register("exclaim", lambda t, c: t + "!")

        pipe = reg.resolve_pipeline(["upper", "exclaim"])

        assert pipe("hello", TransformContext()) == "HELLO!"

    def test_available(self) -> None:
        """Available returns sorted list of names."""
        reg = TransformRegistry()
        reg.register("beta", lambda t, c: t)
        reg.register("alpha", lambda t, c: t)

        assert reg.available() == ["alpha", "beta"]

    def test_contains(self) -> None:
        """'in' operator checks name existence."""
        reg = TransformRegistry()
        reg.register("foo", lambda t, c: t)

        assert "foo" in reg
        assert "bar" not in reg

    def test_len(self) -> None:
        """Registry reports number of transforms."""
        reg = TransformRegistry()
        reg.register("a", lambda t, c: t)
        reg.register("b", lambda t, c: t)

        assert len(reg) == 2
