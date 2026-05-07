"""Semantic anchors: the type-level specification of a question.

A :class:`SemanticAnchor` defines *what* a question measures,
independently of how it is phrased. It is the invariant that any
realization must preserve, the *type* in the dependent type
``Question(ctx)``.

The anchor includes:

- a *canonical prompt*: the reference phrasing
- a *response space*: the set of valid responses and their ordering
- *structural constraints*: keywords, span references, and embedding
  bounds that any realization must satisfy
"""

from __future__ import annotations

from typing import Self

import didactic.api as dx

from bead.data.base import BeadBaseModel


class SemanticPoles(BeadBaseModel):
    """Pole labels for an ordered response scale.

    Ordered scales are characterized by their two participant-facing
    endpoint labels (for example ``low="definitely no"`` and
    ``high="definitely yes"``). Unordered scales have no poles and use
    ``None`` in place of an instance of this model.

    Attributes
    ----------
    low : str
        Label of the low end of the scale.
    high : str
        Label of the high end of the scale.

    Examples
    --------
    >>> poles = SemanticPoles(low="definitely no", high="definitely yes")
    >>> poles.as_tuple()
    ('definitely no', 'definitely yes')
    """

    low: str
    high: str

    def as_tuple(self) -> tuple[str, str]:
        """Return ``(low, high)`` as a 2-tuple.

        Returns
        -------
        tuple[str, str]
            The pole labels as a Python tuple.
        """
        return (self.low, self.high)


class ResponseSpace(BeadBaseModel):
    """The space of valid responses for a question.

    Attributes
    ----------
    options : tuple[str, ...]
        Ordered response options.
    is_ordered : bool
        Whether the options form an ordinal scale. Defaults to
        ``True``.
    semantic_poles : SemanticPoles | None
        Pole labels for ordered scales (for example ``low="never"``,
        ``high="always"``). ``None`` for unordered (categorical)
        response spaces. Defaults to ``None``.

    Examples
    --------
    >>> rs = ResponseSpace(
    ...     options=("definitely no", "probably no", "unsure",
    ...              "probably yes", "definitely yes"),
    ...     is_ordered=True,
    ...     semantic_poles=SemanticPoles(
    ...         low="definitely no", high="definitely yes",
    ...     ),
    ... )
    >>> len(rs)
    5
    >>> "probably yes" in rs
    True
    """

    options: tuple[str, ...]
    is_ordered: bool = True
    semantic_poles: dx.Embed[SemanticPoles] | None = None

    def __len__(self) -> int:
        """Return the number of response options."""
        return len(self.options)

    def __contains__(self, item: str) -> bool:
        """Return whether ``item`` is one of the response options.

        Parameters
        ----------
        item : str
            Candidate response label.

        Returns
        -------
        bool
            ``True`` when ``item`` is a registered option.
        """
        return item in self.options


class SemanticAnchor(BeadBaseModel):
    """Type-level specification of what a question measures.

    Any realization of a question must preserve the anchor's semantic
    content. The anchor is the *type*; a realized prompt string is the
    *value*.

    Attributes
    ----------
    name : str
        Short identifier (for example ``"completion"``).
    target_property : str
        The property being measured (for example ``"telicity"``).
    canonical_prompt : str
        Reference phrasing of the question. Serves as both
        documentation and the default template.
    response_space : ResponseSpace
        Valid responses.
    required_span_labels : frozenset[str]
        Span labels that must appear in any realization (for example
        ``frozenset({"situation"})``). Defaults to the empty set.
    required_keywords : frozenset[str]
        Keywords that must appear in any realization to preserve
        semantic content. Used by :class:`StructuralDriftValidator`.
        Defaults to the empty set.
    embedding_center : tuple[float, ...] | None
        Pre-computed embedding of the canonical prompt for drift
        validation via cosine distance. ``None`` means the embedding
        is computed on demand by the validator. Defaults to ``None``.
    max_drift : float
        Maximum allowed cosine distance.
    description : str
        Human-readable description.

    Examples
    --------
    >>> rs = ResponseSpace(
    ...     options=("no", "yes"), is_ordered=False
    ... )
    >>> anchor = SemanticAnchor(
    ...     name="dynamicity",
    ...     target_property="dynamic",
    ...     canonical_prompt="Is anything changing during [[situation]]?",
    ...     response_space=rs,
    ...     required_span_labels=frozenset({"situation"}),
    ...     required_keywords=frozenset({"changing"}),
    ... )
    >>> anchor.name
    'dynamicity'

    See Also
    --------
    bead.protocol.drift.StructuralDriftValidator : Enforces
        ``required_span_labels`` and ``required_keywords``.
    bead.protocol.drift.EmbeddingDriftValidator : Enforces
        ``embedding_center`` and ``max_drift``.
    """

    name: str
    target_property: str
    canonical_prompt: str
    response_space: dx.Embed[ResponseSpace]
    required_span_labels: frozenset[str] = dx.field(default_factory=frozenset)
    required_keywords: frozenset[str] = dx.field(default_factory=frozenset)
    embedding_center: tuple[float, ...] | None = None
    max_drift: float = 0.3
    description: str = ""

    @classmethod
    def from_response_options(
        cls,
        *,
        name: str,
        target_property: str,
        canonical_prompt: str,
        options: tuple[str, ...],
        is_ordered: bool = True,
        semantic_poles: SemanticPoles | None = None,
        required_span_labels: frozenset[str] = frozenset(),
        required_keywords: frozenset[str] = frozenset(),
        embedding_center: tuple[float, ...] | None = None,
        max_drift: float = 0.3,
        description: str = "",
    ) -> Self:
        """Build an anchor from a flat list of response options.

        Convenience constructor for the common case in which a
        :class:`ResponseSpace` is built inline from its options.

        Parameters
        ----------
        name : str
            Short identifier.
        target_property : str
            The property being measured.
        canonical_prompt : str
            Reference phrasing.
        options : tuple[str, ...]
            Ordered response options.
        is_ordered : bool, optional
            Whether the options form an ordinal scale. Defaults to
            ``True``.
        semantic_poles : SemanticPoles | None, optional
            Pole labels for ordered scales. Defaults to ``None``.
        required_span_labels : frozenset[str], optional
            Span labels required in every realization. Defaults to the
            empty set.
        required_keywords : frozenset[str], optional
            Keywords required in every realization. Defaults to the
            empty set.
        embedding_center : tuple[float, ...] | None, optional
            Pre-computed canonical-prompt embedding. Defaults to
            ``None``.
        max_drift : float, optional
            Maximum allowed cosine distance. Defaults to ``0.3``.
        description : str, optional
            Human-readable description. Defaults to the empty string.

        Returns
        -------
        SemanticAnchor
            A new anchor with an inline-constructed response space.

        Examples
        --------
        >>> anchor = SemanticAnchor.from_response_options(
        ...     name="completion",
        ...     target_property="telicity",
        ...     canonical_prompt="Does [[situation]] reach an endpoint?",
        ...     options=("definitely no", "probably no", "unsure",
        ...              "probably yes", "definitely yes"),
        ...     is_ordered=True,
        ...     semantic_poles=SemanticPoles(
        ...         low="definitely no", high="definitely yes"
        ...     ),
        ...     required_span_labels=frozenset({"situation"}),
        ... )
        >>> anchor.response_space.is_ordered
        True
        """
        space = ResponseSpace(
            options=options,
            is_ordered=is_ordered,
            semantic_poles=semantic_poles,
        )
        return cls(
            name=name,
            target_property=target_property,
            canonical_prompt=canonical_prompt,
            response_space=space,
            required_span_labels=required_span_labels,
            required_keywords=required_keywords,
            embedding_center=embedding_center,
            max_drift=max_drift,
            description=description,
        )
