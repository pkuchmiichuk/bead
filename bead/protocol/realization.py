"""Realization strategies: how dependent functions are computed.

A :class:`RealizationStrategy` maps a
:class:`~bead.protocol.anchor.SemanticAnchor` and a
:class:`~bead.protocol.context.ProtocolContext` to a concrete prompt
string. It is the computational content of the dependent function
``Pi(ctx). Question(ctx)``.

Three strategies are provided:

- :class:`TemplateRealization`: a fixed template (the simplest
  strategy and a safe fallback).
- :class:`ContextualTemplateRealization`: rule-based selection from
  ranked template variants.
- :class:`LMRealization`: prompts a language model to paraphrase the
  canonical question for the specific context. Should always be paired
  with a :class:`~bead.protocol.drift.DriftGuard` to validate that the
  paraphrase preserves semantic content.

These classes carry callable fields (predicates, LM clients) so they
are plain frozen Python classes rather than
:class:`~bead.data.base.BeadBaseModel` subclasses; didactic Models do
not accept :class:`~collections.abc.Callable` field types.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Protocol, runtime_checkable

from bead.protocol.anchor import SemanticAnchor
from bead.protocol.context import ContextPredicate, ProtocolContext, always

if TYPE_CHECKING:
    from bead.items.cache import ModelOutputCache


@runtime_checkable
class RealizationStrategy(Protocol):
    """Protocol for question realization.

    A realization strategy is the computational content of the
    dependent function ``Pi(ctx). Question(ctx)``: it produces a
    prompt string for a given anchor-and-context pair.

    Examples
    --------
    A minimal conforming implementation:

    >>> class EchoCanonical:
    ...     def realize(
    ...         self, anchor, context
    ...     ):
    ...         return anchor.canonical_prompt
    >>> isinstance(EchoCanonical(), RealizationStrategy)
    True
    """

    def realize(
        self,
        anchor: SemanticAnchor,
        context: ProtocolContext,
    ) -> str:
        """Produce a prompt string for the given anchor and context.

        Parameters
        ----------
        anchor : SemanticAnchor
            The semantic invariant to preserve.
        context : ProtocolContext
            The context to condition on.

        Returns
        -------
        str
            A prompt string, possibly containing ``[[label]]`` or
            ``[[label|transform]]`` references.
        """
        ...


@dataclass(frozen=True)
class TemplateVariant:
    """A context-conditioned question template.

    Parameters
    ----------
    template : str
        Question template, possibly containing ``[[label]]`` or
        ``[[label|transform]]`` references.
    condition : ContextPredicate, optional
        Returns ``True`` when this variant is appropriate for the
        context. Variants are evaluated in priority order; the first
        match wins. Defaults to :func:`always`.
    priority : int, optional
        Higher-priority variants are tried first. Use this to order
        more-specific variants before less-specific ones. Defaults to
        ``0``.
    description : str, optional
        Human-readable description for experimenters. Defaults to the
        empty string.

    Attributes
    ----------
    template : str
        The template string.
    condition : ContextPredicate
        Variant-applicability predicate.
    priority : int
        Selection priority.
    description : str
        Human-readable description.
    """

    template: str
    condition: ContextPredicate = field(default=always)
    priority: int = 0
    description: str = ""


@dataclass(frozen=True)
class TemplateRealization:
    """Fixed-template realization.

    Always returns the same template string regardless of context. The
    simplest strategy and a safe fallback when context-dependent
    phrasing is not needed.

    Parameters
    ----------
    template : str | None, optional
        Template string. When ``None``, the anchor's canonical prompt
        is used at realization time. Defaults to ``None``.

    Attributes
    ----------
    template : str | None
        The configured template, or ``None`` to defer to the anchor.
    """

    template: str | None = None

    def realize(
        self,
        anchor: SemanticAnchor,
        context: ProtocolContext,  # noqa: ARG002
    ) -> str:
        """Return the configured template or the canonical prompt.

        Parameters
        ----------
        anchor : SemanticAnchor
            The semantic invariant. Its ``canonical_prompt`` is used
            when this strategy was constructed without an explicit
            template.
        context : ProtocolContext
            The annotation context (unused by this strategy but
            required by the :class:`RealizationStrategy` protocol).

        Returns
        -------
        str
            The realized prompt string.
        """
        return self.template if self.template is not None else anchor.canonical_prompt


@dataclass(frozen=True)
class ContextualTemplateRealization:
    """Rule-based selection from ranked template variants.

    Evaluates variant conditions in descending priority order and
    returns the template of the first matching variant. Falls back to
    a configurable fallback template (or the anchor's canonical prompt
    if none is configured) when no variant matches.

    This is the recommended strategy for production use: it gives
    experimenters fine-grained control over how questions adapt to
    context while guaranteeing the output is one of a pre-approved set
    of templates.

    Parameters
    ----------
    variants : tuple[TemplateVariant, ...]
        Candidate templates. They are evaluated in descending priority
        order; ties are broken by registration order.
    fallback : str | None, optional
        Template used when no variant matches. When ``None``, the
        anchor's canonical prompt is used. Defaults to ``None``.

    Attributes
    ----------
    variants : tuple[TemplateVariant, ...]
        The configured variants, sorted by descending priority.
    fallback : str | None
        Fallback template, or ``None`` to defer to the anchor.
    """

    variants: tuple[TemplateVariant, ...]
    fallback: str | None = None

    def __post_init__(self) -> None:
        """Sort variants by descending priority, stable on ties."""
        sorted_variants = tuple(
            sorted(self.variants, key=lambda v: v.priority, reverse=True)
        )
        object.__setattr__(self, "variants", sorted_variants)

    def realize(
        self,
        anchor: SemanticAnchor,
        context: ProtocolContext,
    ) -> str:
        """Return the first matching variant's template, or the fallback.

        Parameters
        ----------
        anchor : SemanticAnchor
            The semantic invariant.
        context : ProtocolContext
            The annotation context tested against each variant's
            condition.

        Returns
        -------
        str
            The template of the highest-priority matching variant, the
            configured fallback if none match, or
            ``anchor.canonical_prompt`` when no fallback is configured.
        """
        for variant in self.variants:
            if variant.condition(context):
                return variant.template
        return self.fallback if self.fallback is not None else anchor.canonical_prompt


@runtime_checkable
class LMClient(Protocol):
    """Protocol for language-model completion.

    Any object with a ``complete`` method matching this signature can
    serve as an LM backend for :class:`LMRealization`. The keyword
    parameters ``temperature`` and ``max_tokens`` are required, since
    :class:`LMRealization` always supplies them.

    Examples
    --------
    A minimal stub for testing:

    >>> class StubClient:
    ...     def complete(
    ...         self, prompt: str, *,
    ...         temperature: float, max_tokens: int,
    ...     ) -> str:
    ...         return "Did the event reach an endpoint?"
    >>> isinstance(StubClient(), LMClient)
    True
    """

    def complete(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        """Generate a completion for the given prompt.

        Parameters
        ----------
        prompt : str
            Full prompt including any system context.
        temperature : float
            Sampling temperature.
        max_tokens : int
            Maximum response length in tokens.

        Returns
        -------
        str
            Generated text.
        """
        ...


_DEFAULT_SYSTEM_PROMPT = (
    "You are helping design annotation questions for a linguistics "
    "experiment. You will be given a sentence, information about a "
    "highlighted target, and a canonical question about a specific "
    "linguistic property.\n\n"
    "Your task: rephrase the canonical question so it is natural, "
    "clear, and easy for a non-linguist to answer, while preserving:\n"
    "1. The same semantic target (the same property is being "
    "measured)\n"
    "2. The same response scale\n"
    "3. References to the highlighted target using [[label]] or "
    "[[label|transform]] syntax wherever they appear in the canonical "
    "question.\n\n"
    "Output ONLY the rephrased question, nothing else."
)
"""Default system prompt for :class:`LMRealization`.

Tuned to preserve the response scale and ``[[label]]`` references
required by structural drift validation.
"""


class LMRealization:
    """LM-based question paraphrasing.

    Prompts a language model to rephrase the canonical question for
    the specific annotation context. The LM receives the sentence,
    target information, and canonical question as context, and
    produces a paraphrase that should be more natural for the specific
    sentence.

    This strategy should always be paired with a
    :class:`~bead.protocol.drift.DriftGuard` to validate that the
    paraphrase preserves semantic content.

    When ``cache`` is supplied (a :class:`~bead.items.cache.ModelOutputCache`),
    realized prompts are stored under the
    ``(model_name, "lm_completion", prompt=full_prompt)`` key. Repeated
    calls with the same anchor-and-context pair avoid redundant LM
    calls. The cache is the single canonical caching surface across
    bead; this class does not maintain its own.

    Parameters
    ----------
    client : LMClient
        Language-model backend.
    model_name : str
        Identifier for the model behind ``client``. Used as the cache
        key prefix.
    cache : ModelOutputCache | None, optional
        Output cache shared with the rest of bead. Pass ``None`` to
        disable caching. Defaults to ``None``.
    system_prompt : str, optional
        System prompt controlling paraphrase behavior. Defaults to
        :data:`_DEFAULT_SYSTEM_PROMPT`.
    temperature : float, optional
        Sampling temperature. Lower values are more conservative.
        Defaults to ``0.3``.
    max_tokens : int, optional
        Maximum response length in tokens. Defaults to ``200``.
    """

    def __init__(
        self,
        client: LMClient,
        *,
        model_name: str,
        cache: ModelOutputCache | None = None,
        system_prompt: str = _DEFAULT_SYSTEM_PROMPT,
        temperature: float = 0.3,
        max_tokens: int = 200,
    ) -> None:
        self._client = client
        self._model_name = model_name
        self._cache = cache
        self._system_prompt = system_prompt
        self._temperature = temperature
        self._max_tokens = max_tokens

    def _build_user_prompt(
        self, anchor: SemanticAnchor, context: ProtocolContext
    ) -> str:
        """Construct the user-facing portion of the LM prompt.

        Parameters
        ----------
        anchor : SemanticAnchor
            The semantic invariant.
        context : ProtocolContext
            The annotation context.

        Returns
        -------
        str
            A multi-line string summarizing the context and the
            canonical question.
        """
        parts: list[str] = [
            f'Sentence: "{context.sentence}"',
            f'Highlighted target: "{context.target_span_text}"',
            f'Target lemma: "{context.target_lemma}"',
        ]

        if context.dependents:
            dep_strs = [
                f'  - {d.head_lemma} ({d.head_upos}): "{d.span_text}"'
                for d in context.dependents
            ]
            parts.append("Dependents:\n" + "\n".join(dep_strs))

        parts.extend(
            [
                f"Semantic target: {anchor.target_property}",
                f"Description: {anchor.description}",
                f'Canonical question: "{anchor.canonical_prompt}"',
                f"Response scale: {list(anchor.response_space.options)}",
            ]
        )

        if anchor.required_span_labels:
            parts.append(
                f"Required span references: {sorted(anchor.required_span_labels)}"
            )

        return "\n".join(parts)

    def realize(
        self,
        anchor: SemanticAnchor,
        context: ProtocolContext,
    ) -> str:
        """Generate a context-adapted question via the LM.

        When a cache was supplied at construction time and a cached
        result exists for the same prompt, the cached value is
        returned without calling the LM.

        Parameters
        ----------
        anchor : SemanticAnchor
            Semantic specification.
        context : ProtocolContext
            Current annotation context.

        Returns
        -------
        str
            LM-generated prompt string. Surrounding quotes and
            whitespace are stripped, and a trailing ``?`` is appended
            when missing.

        Raises
        ------
        RuntimeError
            If the LM backend raises, or if the LM returns an empty
            response.
        """
        full_prompt = (
            f"{self._system_prompt}\n\n"
            f"{self._build_user_prompt(anchor, context)}\n\n"
            f"Rephrased question:"
        )

        if self._cache is not None:
            cached = self._cache.get(
                self._model_name, "lm_completion", prompt=full_prompt
            )
            if isinstance(cached, str):
                return cached

        try:
            raw = self._client.complete(
                full_prompt,
                temperature=self._temperature,
                max_tokens=self._max_tokens,
            )
        except Exception as exc:
            raise RuntimeError(
                f"LM realization failed for anchor {anchor.name!r}: {exc}"
            ) from exc

        cleaned = raw.strip().strip("\"'").strip()
        if not cleaned:
            raise RuntimeError(
                f"LM realization returned an empty response for anchor {anchor.name!r}"
            )
        if not cleaned.endswith("?"):
            cleaned = f"{cleaned}?"

        if self._cache is not None:
            self._cache.set(
                self._model_name,
                "lm_completion",
                cleaned,
                prompt=full_prompt,
            )

        return cleaned
