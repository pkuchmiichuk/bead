"""Drift validation: the type-checker for realized prompts.

A :class:`DriftGuard` verifies that a realized prompt still inhabits
the type defined by its :class:`~bead.protocol.anchor.SemanticAnchor`.
Without drift control an LM paraphraser, or even a rule-based
selector, may produce prompts that subtly change what is being
measured.

Three validators are provided:

- :class:`StructuralDriftValidator` checks that required span
  references and keywords appear in the realization and that the
  question is well-formed.
- :class:`EmbeddingDriftValidator` checks that the embedding of the
  realized prompt is within a configured cosine distance of the
  anchor's canonical-prompt embedding.
- :class:`PerplexityDriftValidator` flags realizations whose language-
  model perplexity exceeds a configured ceiling.

These compose under a :class:`DriftGuard`, which runs all configured
validators and aggregates their findings: a realization passes only
when every validator passes.
"""

from __future__ import annotations

import math
from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Protocol, runtime_checkable

from bead.data.base import BeadBaseModel
from bead.labels import find_label_names
from bead.protocol.anchor import SemanticAnchor
from bead.protocol.context import ProtocolContext


@runtime_checkable
class EmbeddingAdapter(Protocol):
    """Structural type for objects that can embed text.

    Conforms to bead :class:`~bead.items.adapters.ModelAdapter` and to
    any other object exposing a ``get_embedding`` method that returns
    a sequence of floats.

    Examples
    --------
    >>> class StubEmbedder:
    ...     def get_embedding(self, text: str) -> Sequence[float]:
    ...         return (1.0, 0.0)
    >>> isinstance(StubEmbedder(), EmbeddingAdapter)
    True
    """

    def get_embedding(self, text: str) -> Sequence[float]:
        """Embed ``text`` to a fixed-length sequence of floats.

        Parameters
        ----------
        text : str
            Text to embed.

        Returns
        -------
        Sequence[float]
            Embedding vector, treated as a flat sequence of floats.
        """
        ...


@runtime_checkable
class PerplexityAdapter(Protocol):
    """Structural type for objects that can score text perplexity.

    Conforms to bead :class:`~bead.items.adapters.ModelAdapter` and to
    any other object exposing a ``compute_perplexity`` method.
    """

    def compute_perplexity(self, text: str) -> float:
        """Compute the perplexity of ``text`` under the backend.

        Parameters
        ----------
        text : str
            Text to score.

        Returns
        -------
        float
            Perplexity in the open interval ``(0, +inf)``.
        """
        ...


class DriftScore(BeadBaseModel):
    """Result of one or more drift validation checks.

    Attributes
    ----------
    passed : bool
        Whether the realization passes the validators that produced
        this score. Defaults to ``True``.
    structural_ok : bool
        Whether structural constraints are satisfied. Defaults to
        ``True``.
    embedding_distance : float | None
        Cosine distance from the canonical-prompt embedding, if an
        embedding validator ran. Defaults to ``None``.
    perplexity : float | None
        Perplexity of the realized prompt under the validating
        language model, if a perplexity validator ran. Defaults to
        ``None``.
    findings : tuple[str, ...]
        Human-readable descriptions of any issues found. Defaults to
        the empty tuple.
    """

    passed: bool = True
    structural_ok: bool = True
    embedding_distance: float | None = None
    perplexity: float | None = None
    findings: tuple[str, ...] = ()


@runtime_checkable
class DriftValidator(Protocol):
    """Protocol for a single drift-validation check.

    Examples
    --------
    A minimal conforming validator:

    >>> class AlwaysPasses:
    ...     def validate(self, realization, anchor, context):
    ...         return DriftScore(passed=True)
    >>> isinstance(AlwaysPasses(), DriftValidator)
    True
    """

    def validate(
        self,
        realization: str,
        anchor: SemanticAnchor,
        context: ProtocolContext,
    ) -> DriftScore:
        """Check the realization against the anchor.

        Parameters
        ----------
        realization : str
            The realized prompt string.
        anchor : SemanticAnchor
            The semantic specification.
        context : ProtocolContext
            The annotation context.

        Returns
        -------
        DriftScore
            Validation result.
        """
        ...


@dataclass(frozen=True)
class StructuralDriftValidator:
    """Validate structural properties of a realized prompt.

    Checks that:

    1. All required span labels appear as ``[[label]]`` references.
    2. Required keywords appear somewhere in the prompt.
    3. The prompt ends with appropriate punctuation.
    4. The prompt is not trivially short.

    Parameters
    ----------
    min_length : int, optional
        Minimum non-whitespace character length for a valid prompt.
        Defaults to ``15``.
    require_question_mark : bool, optional
        Whether the realization must end with ``?``. Defaults to
        ``True``.
    keyword_case_sensitive : bool, optional
        Whether keyword checks are case-sensitive. Defaults to
        ``False``.

    Attributes
    ----------
    min_length : int
        Minimum prompt length.
    require_question_mark : bool
        Whether the trailing ``?`` is required.
    keyword_case_sensitive : bool
        Whether keyword matches are case-sensitive.
    """

    min_length: int = 15
    require_question_mark: bool = True
    keyword_case_sensitive: bool = False

    def validate(
        self,
        realization: str,
        anchor: SemanticAnchor,
        context: ProtocolContext,  # noqa: ARG002
    ) -> DriftScore:
        """Run the structural checks against a realization.

        Parameters
        ----------
        realization : str
            The realized prompt string.
        anchor : SemanticAnchor
            The semantic specification supplying required labels and
            keywords.
        context : ProtocolContext
            The annotation context (unused by this validator but
            required by the :class:`DriftValidator` protocol).

        Returns
        -------
        DriftScore
            Score with ``structural_ok`` set and any failures listed
            in ``findings``.
        """
        findings: list[str] = []
        structural_ok = True

        stripped = realization.strip()

        if len(stripped) < self.min_length:
            findings.append(
                f"Realization too short ({len(stripped)} chars, "
                f"minimum {self.min_length})"
            )
            structural_ok = False

        found_labels = find_label_names(realization)
        for required in anchor.required_span_labels:
            if required not in found_labels:
                findings.append(f"Missing required span reference [[{required}]]")
                structural_ok = False

        check_text = realization if self.keyword_case_sensitive else realization.lower()
        for keyword in anchor.required_keywords:
            check_keyword = keyword if self.keyword_case_sensitive else keyword.lower()
            if check_keyword not in check_text:
                findings.append(f"Missing required keyword: {keyword!r}")
                structural_ok = False

        if self.require_question_mark and not stripped.endswith("?"):
            findings.append("Realization does not end with '?'")
            structural_ok = False

        return DriftScore(
            passed=structural_ok,
            structural_ok=structural_ok,
            findings=tuple(findings),
        )


def _cosine_distance(a: tuple[float, ...], b: tuple[float, ...]) -> float:
    """Compute cosine distance ``1 - cos(a, b)`` between two vectors.

    Parameters
    ----------
    a, b : tuple[float, ...]
        Equal-length vectors.

    Returns
    -------
    float
        Cosine distance in ``[0.0, 2.0]``. Returns ``1.0`` when either
        vector has zero norm (treated as orthogonal).

    Raises
    ------
    ValueError
        If ``a`` and ``b`` have different lengths.
    """
    if len(a) != len(b):
        raise ValueError(f"Vector dimension mismatch: {len(a)} vs {len(b)}")
    dot = sum(ai * bi for ai, bi in zip(a, b, strict=True))
    norm_a = math.sqrt(sum(ai * ai for ai in a))
    norm_b = math.sqrt(sum(bi * bi for bi in b))
    if norm_a == 0.0 or norm_b == 0.0:
        return 1.0
    return 1.0 - dot / (norm_a * norm_b)


class EmbeddingDriftValidator:
    """Validate that a realization is semantically close to the anchor.

    Computes cosine distance between the realization embedding and the
    anchor's canonical-prompt embedding (either pre-computed in the
    anchor or computed on demand from the canonical prompt). The
    realization passes when the distance is at most the configured
    maximum (or the anchor's :attr:`~SemanticAnchor.max_drift`, if no
    explicit maximum is set).

    Embeddings are obtained from any object conforming to the
    :class:`EmbeddingAdapter` Protocol, which includes the bead
    :class:`~bead.items.adapters.ModelAdapter` family.

    Parameters
    ----------
    adapter : EmbeddingAdapter
        Adapter exposing ``get_embedding(text)``. The returned
        sequence is treated as a flat vector and converted to a
        ``tuple[float, ...]``.
    max_distance : float | None, optional
        Override for the anchor's ``max_drift`` value. Defaults to
        ``None`` (use the anchor's own value).

    Attributes
    ----------
    max_distance : float | None
        Configured override, or ``None`` to defer to the anchor.
    """

    def __init__(
        self,
        adapter: EmbeddingAdapter,
        *,
        max_distance: float | None = None,
    ) -> None:
        self._adapter = adapter
        self.max_distance = max_distance

    def _embed(self, text: str) -> tuple[float, ...]:
        """Embed ``text`` via the wrapped adapter and coerce to tuple."""
        emb = self._adapter.get_embedding(text)
        return tuple(float(x) for x in emb)

    def validate(
        self,
        realization: str,
        anchor: SemanticAnchor,
        context: ProtocolContext,  # noqa: ARG002
    ) -> DriftScore:
        """Score the realization by cosine distance from the anchor.

        Parameters
        ----------
        realization : str
            The realized prompt string.
        anchor : SemanticAnchor
            The semantic specification supplying the canonical prompt
            (and optionally a pre-computed embedding center and a
            ``max_drift`` value).
        context : ProtocolContext
            The annotation context (unused by this validator but
            required by the :class:`DriftValidator` protocol).

        Returns
        -------
        DriftScore
            Score with ``embedding_distance`` set; ``passed`` is
            ``True`` iff the distance is within the configured
            maximum.
        """
        if anchor.embedding_center is not None:
            canonical = anchor.embedding_center
        else:
            canonical = self._embed(anchor.canonical_prompt)

        realization_emb = self._embed(realization)
        distance = _cosine_distance(canonical, realization_emb)

        max_dist = (
            self.max_distance if self.max_distance is not None else anchor.max_drift
        )
        passed = distance <= max_dist

        findings: tuple[str, ...] = ()
        if not passed:
            findings = (
                f"Embedding distance {distance:.3f} exceeds maximum {max_dist:.3f}",
            )

        return DriftScore(
            passed=passed,
            embedding_distance=distance,
            findings=findings,
        )


class PerplexityDriftValidator:
    """Validate that a realization has acceptable language-model perplexity.

    Wraps any object conforming to the :class:`PerplexityAdapter`
    Protocol (which includes the bead
    :class:`~bead.items.adapters.ModelAdapter` family). The realization
    passes when its perplexity is at most the configured ceiling.
    Useful for catching ungrammatical or otherwise unnatural
    LM-generated paraphrases that might still pass structural and
    embedding checks.

    Parameters
    ----------
    adapter : PerplexityAdapter
        Adapter exposing ``compute_perplexity(text) -> float``.
    max_perplexity : float
        Maximum allowed perplexity. Realizations with perplexity above
        this value fail.

    Attributes
    ----------
    max_perplexity : float
        The configured perplexity ceiling.
    """

    def __init__(
        self,
        adapter: PerplexityAdapter,
        *,
        max_perplexity: float,
    ) -> None:
        if max_perplexity <= 0.0:
            raise ValueError("max_perplexity must be positive")
        self._adapter = adapter
        self.max_perplexity = max_perplexity

    def validate(
        self,
        realization: str,
        anchor: SemanticAnchor,  # noqa: ARG002
        context: ProtocolContext,  # noqa: ARG002
    ) -> DriftScore:
        """Score the realization by language-model perplexity.

        Parameters
        ----------
        realization : str
            The realized prompt string.
        anchor : SemanticAnchor
            The semantic specification (unused by this validator).
        context : ProtocolContext
            The annotation context (unused by this validator).

        Returns
        -------
        DriftScore
            Score with ``perplexity`` set; ``passed`` is ``True`` iff
            ``perplexity <= max_perplexity``.
        """
        perplexity = float(self._adapter.compute_perplexity(realization))
        passed = perplexity <= self.max_perplexity

        findings: tuple[str, ...] = ()
        if not passed:
            findings = (
                f"Perplexity {perplexity:.2f} exceeds maximum "
                f"{self.max_perplexity:.2f}",
            )

        return DriftScore(
            passed=passed,
            perplexity=perplexity,
            findings=findings,
        )


@dataclass
class DriftGuard:
    """Composite drift validator.

    Runs every configured validator and aggregates their results: the
    aggregate :class:`DriftScore` ``passed`` field is ``True`` only
    when every validator passes. Findings from all validators are
    collected in order. ``embedding_distance`` and ``perplexity`` are
    populated from the last validator that set them.

    Attributes
    ----------
    validators : list[DriftValidator]
        Mutable list of configured validators. Defaults to the empty
        list; calls to :meth:`check` on a guard with no validators
        always pass.
    """

    validators: list[DriftValidator] = field(default_factory=list)

    def add(self, validator: DriftValidator) -> None:
        """Append a validator to the guard.

        Parameters
        ----------
        validator : DriftValidator
            The validator to add.
        """
        self.validators.append(validator)

    def check(
        self,
        realization: str,
        anchor: SemanticAnchor,
        context: ProtocolContext,
    ) -> DriftScore:
        """Run every validator and return an aggregated score.

        Parameters
        ----------
        realization : str
            The realized prompt string.
        anchor : SemanticAnchor
            The semantic specification.
        context : ProtocolContext
            The annotation context.

        Returns
        -------
        DriftScore
            Aggregate score. ``passed`` is ``True`` iff every
            validator passes; ``findings`` concatenates all
            validator-level findings; ``embedding_distance`` and
            ``perplexity`` are taken from the validators that set
            them.
        """
        all_findings: list[str] = []
        all_passed = True
        structural_ok = True
        embedding_distance: float | None = None
        perplexity: float | None = None

        for validator in self.validators:
            score = validator.validate(realization, anchor, context)
            all_findings.extend(score.findings)

            if not score.passed:
                all_passed = False
            if not score.structural_ok:
                structural_ok = False
            if score.embedding_distance is not None:
                embedding_distance = score.embedding_distance
            if score.perplexity is not None:
                perplexity = score.perplexity

        return DriftScore(
            passed=all_passed,
            structural_ok=structural_ok,
            embedding_distance=embedding_distance,
            perplexity=perplexity,
            findings=tuple(all_findings),
        )

    def __len__(self) -> int:
        """Return the number of configured validators."""
        return len(self.validators)
