"""Annotation contexts: dependent indices for question realization.

A :class:`ProtocolContext` gathers everything known about the current
annotation target into a single immutable object. It is the *index*
in the dependent type ``Question(ctx)``: different contexts license
different questions and different phrasings.

The context layer is deliberately domain-neutral. It carries
sentence-level, target-level, and dependent-level information common
to most token- or span-targeted annotation protocols, plus a
JSON-shaped ``metadata`` map (inherited from
:class:`~bead.data.base.BeadBaseModel`) for domain-specific data
that does not fit the standard fields. Domain-specific *predicates
over* the context live in the **predicate registry** documented at
the bottom of this module: callers register named predicates at
import time and refer to them by name from
:class:`~bead.protocol.realization.ContextualTemplateRealization`.
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Self

import didactic.api as dx

from bead.data.base import BeadBaseModel


class ContextItem(BeadBaseModel):
    """Generic per-token-or-span dependent context.

    Captures the structural properties of a single dependent (an
    argument, an adjunct, a related span, ...) of the annotation
    target. Domain-specific scalar attributes live in
    :attr:`attributes`.

    Attributes
    ----------
    node_id : str
        Identifier from the upstream parse or annotation source.
        Defaults to the empty string.
    head_lemma : str
        Lemma of the dependent head. Defaults to the empty string.
    head_form : str
        Surface form of the dependent head. Defaults to the empty
        string.
    head_upos : str
        Universal POS tag of the dependent head. Defaults to the empty
        string.
    head_position : int
        1-based token position of the dependent head. Defaults to
        ``0``.
    span_text : str
        Full surface span text of the dependent. Defaults to the empty
        string.
    span_positions : tuple[int, ...]
        1-based token positions in the dependent span. Defaults to
        the empty tuple.
    is_plural : bool
        Whether the dependent head is morphologically plural. Defaults
        to ``False``.
    attributes : dict[str, float]
        Domain-specific scalar attributes, keyed by attribute name
        (for example ``{"definiteness": 0.7}``). Defaults to the
        empty dict.
    """

    node_id: str = ""
    head_lemma: str = ""
    head_form: str = ""
    head_upos: str = ""
    head_position: int = 0
    span_text: str = ""
    span_positions: tuple[int, ...] = ()
    is_plural: bool = False
    attributes: dict[str, float] = dx.field(default_factory=dict)

    def attribute(self, name: str) -> float | None:
        """Return the value of a named attribute, or ``None`` if absent.

        Parameters
        ----------
        name : str
            Attribute name to look up.

        Returns
        -------
        float | None
            The attribute value, or ``None`` when the attribute is not
            present on this context item.

        Examples
        --------
        >>> item = ContextItem(
        ...     attributes={"change_of_state": 4.2,
        ...                 "instigation": 3.1},
        ... )
        >>> item.attribute("change_of_state")
        4.2
        >>> item.attribute("absent") is None
        True
        """
        return self.attributes.get(name)


class ProtocolContext(BeadBaseModel):
    """Everything known about the current annotation target.

    This is the value that parameterizes the dependent question type.
    Question families inspect the context to decide *which* question
    variant to realize and *how* to phrase it.

    The :meth:`with_response` method threads an annotator response
    into the context, supporting dependent products in which later
    questions condition on earlier answers.

    Attributes
    ----------
    sentence : str
        Full sentence text. Defaults to the empty string.
    tokens : tuple[str, ...]
        Sentence tokens, in order. Defaults to the empty tuple.
    tokens_lemma : tuple[str, ...]
        Token lemmas, in order. Defaults to the empty tuple.
    tokens_upos : tuple[str, ...]
        Universal POS tags, in order. Defaults to the empty tuple.
    target_lemma : str
        Lemma of the annotation target's head. Defaults to the empty
        string.
    target_form : str
        Surface form of the target head. Defaults to the empty string.
    target_upos : str
        UPOS tag of the target head. Defaults to the empty string.
    target_position : int
        1-based token position of the target head. Defaults to ``0``.
    target_span_text : str
        Full surface span text of the target. Defaults to the empty
        string.
    target_span_positions : tuple[int, ...]
        1-based token positions of the target span. Defaults to the
        empty tuple.
    dependents : tuple[ContextItem, ...]
        Structural dependents of the target. Defaults to the empty
        tuple.
    previous_responses : dict[str, str]
        Annotator responses to earlier questions, keyed by anchor
        name. Defaults to the empty dict.
    target_id : str
        Identifier for the annotation target, for traceability.
        Defaults to the empty string.
    source_id : str
        Identifier for the source document or graph. Defaults to the
        empty string.

    See Also
    --------
    register_context_predicate : Register a named predicate over
        :class:`ProtocolContext` instances.
    """

    sentence: str = ""
    tokens: tuple[str, ...] = ()
    tokens_lemma: tuple[str, ...] = ()
    tokens_upos: tuple[str, ...] = ()

    target_lemma: str = ""
    target_form: str = ""
    target_upos: str = ""
    target_position: int = 0
    target_span_text: str = ""
    target_span_positions: tuple[int, ...] = ()

    dependents: tuple[dx.Embed[ContextItem], ...] = ()

    previous_responses: dict[str, str] = dx.field(default_factory=dict)

    target_id: str = ""
    source_id: str = ""

    def with_response(self, question_name: str, response: str) -> Self:
        """Return a new context with one additional response recorded.

        Supports the dependent-product structure: the type of a later
        question can depend on the value (response) of an earlier
        question.

        Parameters
        ----------
        question_name : str
            Name of the anchor whose response is being recorded.
        response : str
            The annotator's response label.

        Returns
        -------
        ProtocolContext
            A new context whose :attr:`previous_responses` includes
            ``{question_name: response}``.

        Examples
        --------
        >>> ctx = ProtocolContext(sentence="Mary built a sandcastle.")
        >>> ctx2 = ctx.with_response("dynamicity", "yes")
        >>> ctx2.previous_responses
        {'dynamicity': 'yes'}
        >>> ctx.previous_responses
        {}
        """
        updated = {**self.previous_responses, question_name: response}
        return self.with_(previous_responses=updated)

    def get_response(self, question_name: str) -> str | None:
        """Return the recorded response for a question, or ``None``.

        Parameters
        ----------
        question_name : str
            The anchor name to look up.

        Returns
        -------
        str | None
            The recorded response label, or ``None`` if no response
            has been threaded for this question.
        """
        return self.previous_responses.get(question_name)


# ---------------------------------------------------------------------------
# Context-predicate registry
# ---------------------------------------------------------------------------
#
# A :data:`ContextPredicate` is a function ``ProtocolContext -> bool``
# used by :class:`~bead.protocol.realization.ContextualTemplateRealization`
# to select among template variants based on context properties.
#
# The registry is intentionally module-level mutable state. It is
# populated at import time by user code and read at realization time.
# Do not mutate it from request-path code: the registry is not
# thread-safe and is not intended to carry per-request state.

ContextPredicate = Callable[[ProtocolContext], bool]
"""Type alias for predicates over :class:`ProtocolContext`."""


_PREDICATES: dict[str, ContextPredicate] = {}


def register_context_predicate(name: str, predicate: ContextPredicate) -> None:
    """Register a named predicate over :class:`ProtocolContext`.

    Callers register their domain-specific predicates at import time.
    The registered predicates are then available by name to
    :class:`~bead.protocol.realization.ContextualTemplateRealization`
    and other realization strategies that select among variants.

    Parameters
    ----------
    name : str
        Unique predicate name. Re-registering an existing name
        overwrites the previous predicate.
    predicate : ContextPredicate
        Callable that returns ``True`` when the context matches.

    Examples
    --------
    >>> def has_plural_dependent(ctx: ProtocolContext) -> bool:
    ...     return any(d.is_plural for d in ctx.dependents)
    >>> register_context_predicate(
    ...     "has_plural_dependent", has_plural_dependent
    ... )
    >>> get_context_predicate("has_plural_dependent") is has_plural_dependent
    True
    """
    _PREDICATES[name] = predicate


def get_context_predicate(name: str) -> ContextPredicate:
    """Look up a registered predicate by name.

    Parameters
    ----------
    name : str
        The predicate name to look up.

    Returns
    -------
    ContextPredicate
        The registered predicate.

    Raises
    ------
    KeyError
        If no predicate with that name is registered.
    """
    try:
        return _PREDICATES[name]
    except KeyError:
        raise KeyError(
            f"No context predicate registered under name {name!r}. "
            f"Registered: {sorted(_PREDICATES)}"
        ) from None


def list_context_predicates() -> tuple[str, ...]:
    """Return the names of all registered context predicates, sorted.

    Returns
    -------
    tuple[str, ...]
        All registered predicate names in sorted order.
    """
    return tuple(sorted(_PREDICATES))


def always(_ctx: ProtocolContext) -> bool:
    """Predicate that matches every context.

    Used as the catch-all condition for fallback template variants and
    the default applicability predicate for question families.

    Parameters
    ----------
    _ctx : ProtocolContext
        Ignored.

    Returns
    -------
    bool
        Always ``True``.
    """
    return True


register_context_predicate("always", always)
