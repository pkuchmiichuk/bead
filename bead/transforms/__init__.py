"""Composable text transforms for span reference resolution.

This module provides a generic, extensible transform system that can
modify span text when resolving ``[[label|transform]]`` references in
prompts. Transforms are registered by name in a :class:`TransformRegistry`
and can be chained via the ``|`` pipe syntax in prompt references.

The system is language-agnostic at the protocol level: any callable that
conforms to the :class:`SpanTextTransform` protocol can be registered.
Language-specific behaviour (e.g. morphological inflection via UniMorph)
is provided by concrete implementations that accept a ``language_code``
at construction time.

Integration Points
------------------
- Prompt resolution: bead/deployment/jspsych/trials.py
- Span labeling: bead/items/span_labeling.py
- Morphology backend: bead/resources/adapters/unimorph.py
"""

from bead.transforms.base import (
    SpanTextTransform,
    TransformContext,
    TransformPipeline,
    TransformRegistry,
)
from bead.transforms.morphology import (
    MorphologicalTransform,
    register_morphological_transforms,
)
from bead.transforms.text import (
    CapitalizeTransform,
    LowerTransform,
    MarkdownStripTransform,
    RedditCleanupTransform,
    TitleTransform,
    UpperTransform,
    split_sentences,
)

__all__ = [
    "CapitalizeTransform",
    "LowerTransform",
    "MarkdownStripTransform",
    "MorphologicalTransform",
    "RedditCleanupTransform",
    "SpanTextTransform",
    "TitleTransform",
    "TransformContext",
    "TransformPipeline",
    "TransformRegistry",
    "UpperTransform",
    "create_default_registry",
    "split_sentences",
]


def create_default_registry(
    language_code: str | None = None,
) -> TransformRegistry:
    """Create a registry pre-loaded with the built-in transforms.

    Text transforms (``lower``, ``upper``, ``capitalize``, ``title``,
    ``markdown_strip``, ``reddit_cleanup``) are always registered.  If
    *language_code* is provided, morphological
    transforms (``gerund``, ``past_tense``, ``present_3sg``,
    ``past_participle``, ``infinitive``) are also registered using the
    UniMorph backend.

    Parameters
    ----------
    language_code : str | None
        ISO 639 language code for morphological transforms.  When
        ``None``, only text transforms are registered.

    Returns
    -------
    TransformRegistry
        A ready-to-use registry.
    """
    registry = TransformRegistry()

    # text transforms — always available
    registry.register("lower", LowerTransform())
    registry.register("upper", UpperTransform())
    registry.register("capitalize", CapitalizeTransform())
    registry.register("title", TitleTransform())
    registry.register("markdown_strip", MarkdownStripTransform())
    registry.register("reddit_cleanup", RedditCleanupTransform())

    # morphological transforms — require a language
    if language_code is not None:
        register_morphological_transforms(registry, language_code)

    return registry
