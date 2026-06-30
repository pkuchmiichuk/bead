"""Core abstractions for the span text transform system.

Defines the :class:`SpanTextTransform` protocol, :class:`TransformContext`
for passing metadata to transforms, :class:`TransformPipeline` for
composing transforms, and :class:`TransformRegistry` for name-based
lookup.

The transforms operate at the value level (``str -> str`` parameterised
by a ``TransformContext``). Use ``dx.Iso`` or ``dx.Lens`` directly when
the transformation crosses schema boundaries.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Protocol, runtime_checkable

import didactic.api as dx

from bead.data.base import BeadBaseModel, JsonValue


class TransformContext(BeadBaseModel):
    """Metadata available to transforms at resolution time.

    Attributes
    ----------
    language_code : str | None
        ISO 639 code (e.g. ``"eng"``, ``"en"``).
    lemma : str | None
        Lemma of the span head, if known.
    pos : str | None
        Universal POS tag of the span head (e.g. ``"VERB"``).
    head_index : int | None
        Token index of the syntactic head within the span.
    tokens : tuple[str, ...]
        Individual tokens of the span text. Empty when unknown.
    metadata : dict[str, JsonValue]
        Arbitrary extra metadata.
    """

    language_code: str | None = None
    lemma: str | None = None
    pos: str | None = None
    head_index: int | None = None
    tokens: tuple[str, ...] = ()
    metadata: dict[str, JsonValue] = dx.field(default_factory=dict)


@runtime_checkable
class SpanTextTransform(Protocol):
    """Protocol for a single text transform.

    Any callable ``(str, TransformContext) -> str`` satisfies this
    protocol. Implementations may ignore the context when the transform
    is purely textual (e.g. lowercasing).
    """

    def __call__(self, text: str, context: TransformContext, /) -> str:
        """Apply the transform to *text*."""
        ...


class TransformPipeline:
    """An ordered chain of transforms applied left-to-right.

    Examples
    --------
    >>> from bead.transforms.text import LowerTransform, CapitalizeTransform
    >>> ctx = TransformContext()
    >>> pipe = TransformPipeline([LowerTransform(), CapitalizeTransform()])
    >>> pipe("HELLO WORLD", ctx)
    'Hello world'
    """

    def __init__(self, transforms: list[SpanTextTransform] | None = None) -> None:
        self._transforms: list[SpanTextTransform] = list(transforms or [])

    def __call__(self, text: str, context: TransformContext) -> str:
        """Apply each transform in sequence."""
        for transform in self._transforms:
            text = transform(text, context)
        return text

    def __len__(self) -> int:
        """Return the number of transforms in the pipeline."""
        return len(self._transforms)

    def __repr__(self) -> str:
        """Return a debug-friendly representation of the pipeline."""
        names = [type(t).__name__ for t in self._transforms]
        return f"TransformPipeline({names})"

    def append(self, transform: SpanTextTransform) -> None:
        """Append a transform to the end of the pipeline."""
        self._transforms.append(transform)

    def prepend(self, transform: SpanTextTransform) -> None:
        """Insert a transform at the beginning of the pipeline."""
        self._transforms.insert(0, transform)


class TransformRegistry:
    """Name-to-transform mapping with pipeline construction.

    Transforms are registered under short string names (e.g.
    ``"gerund"``, ``"lower"``) and looked up when resolving
    ``[[label|name1|name2]]`` prompt references.

    Examples
    --------
    >>> from bead.transforms.text import LowerTransform
    >>> reg = TransformRegistry()
    >>> reg.register("lower", LowerTransform())
    >>> t = reg.get("lower")
    >>> t("HELLO", TransformContext())
    'hello'
    """

    def __init__(self) -> None:
        self._transforms: dict[str, SpanTextTransform] = {}

    def register(
        self,
        name: str,
        transform: SpanTextTransform | Callable[[str, TransformContext], str],
    ) -> None:
        """Register *transform* under *name* (case-insensitive)."""
        name = name.strip().lower()
        if not name:
            raise ValueError("Transform name must be non-empty")
        self._transforms[name] = transform

    def get(self, name: str) -> SpanTextTransform:
        """Return the transform registered under *name*.

        Raises
        ------
        KeyError
            If no transform with that name exists.
        """
        name = name.strip().lower()
        try:
            return self._transforms[name]
        except KeyError:
            available = sorted(self._transforms)
            raise KeyError(
                f"No transform registered as '{name}'. Available: {available}"
            ) from None

    def resolve_pipeline(self, names: list[str]) -> TransformPipeline:
        """Return a pipeline applying the named transforms left-to-right."""
        return TransformPipeline([self.get(n) for n in names])

    def available(self) -> list[str]:
        """Return the registered transform names, sorted."""
        return sorted(self._transforms)

    def __contains__(self, name: str) -> bool:
        """Return whether *name* is registered."""
        return name.strip().lower() in self._transforms

    def __len__(self) -> int:
        """Return the number of registered transforms."""
        return len(self._transforms)

    def __repr__(self) -> str:
        """Return a debug-friendly representation of the registry."""
        return f"TransformRegistry(transforms={self.available()})"
