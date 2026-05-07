"""Question families and annotation protocols.

A :class:`QuestionFamily` is a dependent function
``Pi(ctx : ProtocolContext). Question(ctx)``: for each context it
produces a valid, drift-checked
:class:`~bead.protocol.family.QuestionRealization`.

An :class:`AnnotationProtocol` is the iterated dependent product

    Sigma(a_1 : Q_1(ctx)). Sigma(a_2 : Q_2(ctx, a_1)). ... Q_n(ctx, ...)

a sequence of question families where later families may condition on
the responses to earlier ones. The dependency edges between families
are recorded explicitly in :attr:`QuestionFamily.depends_on`, which
:class:`~bead.protocol.diagnostics.ConditionalObservationValidator`
consults to check the integrity of conditional responses.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, field

import didactic.api as dx

from bead.data.base import BeadBaseModel
from bead.protocol.anchor import SemanticAnchor
from bead.protocol.context import ProtocolContext
from bead.protocol.drift import DriftGuard, DriftScore
from bead.protocol.realization import RealizationStrategy, TemplateRealization

ApplicabilityPredicate = Callable[[ProtocolContext], bool]
"""Type alias for predicates determining when a family applies."""


def _always_applicable(_ctx: ProtocolContext) -> bool:
    """Default applicability: family applies to every context."""
    return True


class QuestionRealization(BeadBaseModel):
    """A realized question paired with its provenance.

    This is the dependent pair ``Sigma(ctx). Question(ctx)``: a
    concrete prompt together with the context that produced it and
    evidence of its validity (a :class:`~bead.protocol.drift.DriftScore`).

    Attributes
    ----------
    prompt : str
        The realized prompt string. May contain ``[[label]]``
        references for downstream rendering.
    anchor : SemanticAnchor
        The semantic specification this question satisfies.
    context : ProtocolContext
        The context that parameterized the realization.
    drift_score : DriftScore | None
        Result of drift validation, if a guard was applied. Defaults
        to ``None``.
    strategy_name : str
        Name of the realization strategy that produced this question.
        Defaults to the empty string.
    """

    prompt: str
    anchor: dx.Embed[SemanticAnchor]
    context: dx.Embed[ProtocolContext]
    drift_score: dx.Embed[DriftScore] | None = None
    strategy_name: str = ""

    @property
    def passed_drift_check(self) -> bool:
        """Whether the realization passed drift validation.

        ``True`` when no drift score is attached (no validation was
        run) or when the attached score's ``passed`` flag is ``True``.
        """
        if self.drift_score is None:
            return True
        return self.drift_score.passed


@dataclass
class QuestionFamily:
    """Dependent function from contexts to realized questions.

    For each :class:`ProtocolContext`, a family produces a
    :class:`QuestionRealization` by:

    1. Checking applicability (is this question relevant for this
       context?).
    2. Invoking the realization strategy to produce a prompt.
    3. Running drift validation, if a guard is configured.
    4. Falling back to the canonical prompt if the realization drifts
       (when ``fallback_on_drift`` is enabled).

    Parameters
    ----------
    anchor : SemanticAnchor
        The semantic type of questions this family produces.
    realization : RealizationStrategy | None, optional
        Strategy producing a prompt for a given context. Defaults to
        an unparameterized :class:`TemplateRealization`, which echoes
        the anchor's canonical prompt.
    drift_guard : DriftGuard | None, optional
        Optional drift validator. Defaults to ``None``.
    condition : ApplicabilityPredicate | None, optional
        When to ask this question. ``None`` (the default) marks the
        family as always applicable; any non-``None`` value sets
        :attr:`is_always_applicable` to ``False``.
    depends_on : tuple[str, ...], optional
        Anchor names whose responses must precede this family in a
        protocol. Read by
        :class:`~bead.protocol.diagnostics.ConditionalObservationValidator`.
        Defaults to the empty tuple.
    fallback_on_drift : bool, optional
        If ``True`` (the default), fall back to the canonical prompt
        when drift validation fails. If ``False``, raise
        :class:`ValueError`.

    Attributes
    ----------
    anchor : SemanticAnchor
        Configured anchor.
    realization : RealizationStrategy
        Configured realization strategy.
    drift_guard : DriftGuard | None
        Configured drift guard.
    condition : ApplicabilityPredicate
        Configured applicability predicate.
    depends_on : tuple[str, ...]
        Names of anchors this family depends on.
    fallback_on_drift : bool
        Whether to fall back on drift failure.
    """

    anchor: SemanticAnchor
    realization: RealizationStrategy = field(default_factory=TemplateRealization)
    drift_guard: DriftGuard | None = None
    condition: ApplicabilityPredicate = field(default=_always_applicable)
    depends_on: tuple[str, ...] = ()
    fallback_on_drift: bool = True
    is_always_applicable: bool = field(init=False)

    def __post_init__(self) -> None:
        """Record whether ``condition`` is the default predicate."""
        self.is_always_applicable = self.condition is _always_applicable

    @property
    def name(self) -> str:
        """Short name from the anchor."""
        return self.anchor.name

    def is_applicable(self, context: ProtocolContext) -> bool:
        """Whether this family should be asked for the given context.

        Parameters
        ----------
        context : ProtocolContext
            Current annotation context.

        Returns
        -------
        bool
            ``True`` when the family applies.
        """
        return self.condition(context)

    def realize(self, context: ProtocolContext) -> QuestionRealization:
        """Produce a question for the given context.

        Parameters
        ----------
        context : ProtocolContext
            Current annotation context.

        Returns
        -------
        QuestionRealization
            The realized question with its drift score and provenance.

        Raises
        ------
        ValueError
            If drift validation fails and :attr:`fallback_on_drift` is
            ``False``.
        """
        prompt = self.realization.realize(self.anchor, context)
        strategy_name = type(self.realization).__name__

        drift_score: DriftScore | None = None
        guard = self.drift_guard
        if guard is not None:
            score = guard.check(prompt, self.anchor, context)
            if not score.passed:
                if self.fallback_on_drift:
                    prompt = self.anchor.canonical_prompt
                    strategy_name = f"{strategy_name}->fallback"
                    score = guard.check(prompt, self.anchor, context)
                else:
                    raise ValueError(
                        f"Drift validation failed for "
                        f"{self.anchor.name!r}: {list(score.findings)}"
                    )
            drift_score = score

        return QuestionRealization(
            prompt=prompt,
            anchor=self.anchor,
            context=context,
            drift_score=drift_score,
            strategy_name=strategy_name,
        )


@dataclass
class AnnotationProtocol:
    """A sequence of question families forming a complete protocol.

    Represents the iterated dependent product

        Sigma(a_1 : Q_1(ctx)). Sigma(a_2 : Q_2(ctx, a_1)). ...

    When realized, the protocol threads annotator responses through
    the context so later families can condition on earlier answers.

    Parameters
    ----------
    families : list[QuestionFamily]
        Families in protocol order.
    name : str, optional
        Descriptive name for the protocol. Defaults to the empty
        string.

    Attributes
    ----------
    families : list[QuestionFamily]
        Families in protocol order.
    name : str
        Descriptive name.

    Raises
    ------
    ValueError
        If two families share the same anchor name (anchor names must
        be unique within a protocol), or if any family's
        :attr:`~QuestionFamily.depends_on` references a family that
        does not appear earlier in the sequence.
    """

    families: list[QuestionFamily]
    name: str = ""

    def __post_init__(self) -> None:
        """Validate uniqueness and forward-only ``depends_on`` edges."""
        seen: set[str] = set()
        for family in self.families:
            if family.name in seen:
                raise ValueError(f"Duplicate anchor name in protocol: {family.name!r}")
            for dep in family.depends_on:
                if dep == family.name:
                    raise ValueError(f"Family {family.name!r} depends on itself")
                if dep not in seen:
                    raise ValueError(
                        f"Family {family.name!r} depends on {dep!r}, "
                        f"which is not earlier in the protocol "
                        f"(known so far: {sorted(seen)})"
                    )
            seen.add(family.name)

    def append(self, family: QuestionFamily) -> None:
        """Add a family to the end of the protocol.

        Parameters
        ----------
        family : QuestionFamily
            The family to append.

        Raises
        ------
        ValueError
            If a family with the same anchor name is already present,
            or if any of its :attr:`~QuestionFamily.depends_on`
            references a family not already in the protocol.
        """
        existing = {f.name for f in self.families}
        if family.name in existing:
            raise ValueError(f"Duplicate anchor name in protocol: {family.name!r}")
        if family.name in family.depends_on:
            raise ValueError(f"Family {family.name!r} depends on itself")
        for dep in family.depends_on:
            if dep not in existing:
                raise ValueError(
                    f"Family {family.name!r} depends on {dep!r}, "
                    f"which is not in the protocol (known: "
                    f"{sorted(existing)})"
                )
        self.families.append(family)

    def family_by_name(self, name: str) -> QuestionFamily:
        """Look up a family by its anchor name.

        Parameters
        ----------
        name : str
            The anchor name to look up.

        Returns
        -------
        QuestionFamily
            The matching family.

        Raises
        ------
        KeyError
            If no family with that name exists in the protocol.
        """
        for family in self.families:
            if family.name == name:
                return family
        raise KeyError(
            f"No family named {name!r} in protocol "
            f"(have: {[f.name for f in self.families]})"
        )

    def realize_all(
        self,
        context: ProtocolContext,
        *,
        responses: dict[str, str] | None = None,
    ) -> list[QuestionRealization]:
        """Realize all applicable families for a context.

        Threads responses through the context as the protocol is
        traversed. When ``responses`` is provided it is injected
        before any family is realized; otherwise, after each family is
        realized, the first option of its response space is used as a
        placeholder so downstream families can be exercised in dry-run
        mode.

        Parameters
        ----------
        context : ProtocolContext
            Base annotation context.
        responses : dict[str, str] | None, optional
            Pre-supplied responses keyed by anchor name. Defaults to
            ``None``.

        Returns
        -------
        list[QuestionRealization]
            Realized questions in protocol order, skipping families
            whose :meth:`QuestionFamily.is_applicable` returns
            ``False`` for the running context.

        Raises
        ------
        ValueError
            If ``responses`` references an anchor not in the protocol.
        """
        if responses:
            unknown = set(responses) - {f.name for f in self.families}
            if unknown:
                raise ValueError(
                    f"Responses reference unknown anchors: {sorted(unknown)}"
                )

        running_ctx = context
        if responses:
            for family in self.families:
                if family.name in responses:
                    running_ctx = running_ctx.with_response(
                        family.name, responses[family.name]
                    )

        results: list[QuestionRealization] = []
        for family in self.families:
            if not family.is_applicable(running_ctx):
                continue

            realization = family.realize(running_ctx)
            results.append(realization)

            if family.anchor.name not in running_ctx.previous_responses:
                options = family.anchor.response_space.options
                if options:
                    running_ctx = running_ctx.with_response(
                        family.anchor.name, options[0]
                    )

        return results

    def __len__(self) -> int:
        """Return the number of families in the protocol."""
        return len(self.families)
