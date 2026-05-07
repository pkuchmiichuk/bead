"""Configuration for the annotation-protocol layer.

Declares :class:`ProtocolConfig` (the top-level stage config that
plugs into :class:`~bead.config.config.BeadConfig`) along with the
declarative specs (:class:`AnchorSpec`, :class:`TemplateVariantSpec`,
:class:`FamilySpec`, :class:`DriftConfig`) that materialize into
runtime :class:`~bead.protocol.SemanticAnchor`,
:class:`~bead.protocol.QuestionFamily`, and
:class:`~bead.protocol.AnnotationProtocol` objects.

Configuration is *declarative*: anchors, drift thresholds, realization
strategies, and protocol composition are written in YAML or TOML, and
:meth:`ProtocolConfig.build` produces the live objects. Runtime-only
parameters (LM clients, embedding adapters, output caches) are passed
to :meth:`build` rather than stored in the config.

Predicates are referenced *by registered name*; callables cannot be
serialized. Register predicates in the
:mod:`~bead.protocol.context` registry at import time, then refer to
them by name from a :class:`FamilySpec` or :class:`TemplateVariantSpec`.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import didactic.api as dx

from bead.data.base import BeadBaseModel
from bead.protocol.anchor import ResponseSpace, SemanticAnchor, SemanticPoles
from bead.protocol.context import get_context_predicate
from bead.protocol.drift import (
    DriftGuard,
    EmbeddingDriftValidator,
    PerplexityDriftValidator,
    StructuralDriftValidator,
)
from bead.protocol.family import AnnotationProtocol, QuestionFamily
from bead.protocol.realization import (
    ContextualTemplateRealization,
    LMClient,
    LMRealization,
    RealizationStrategy,
    TemplateRealization,
    TemplateVariant,
)

if TYPE_CHECKING:
    from bead.items.cache import ModelOutputCache
    from bead.protocol.drift import EmbeddingAdapter, PerplexityAdapter


RealizationKind = Literal["template", "contextual", "lm"]
"""Discriminator for which realization strategy a family uses."""


class TemplateVariantSpec(BeadBaseModel):
    """Declarative form of :class:`TemplateVariant` for config files.

    Attributes
    ----------
    template : str
        Question template, possibly containing ``[[label]]`` references.
    condition_name : str
        Name of a registered context predicate. Looked up via
        :func:`bead.protocol.context.get_context_predicate` at build
        time. Defaults to ``"always"``.
    priority : int
        Higher-priority variants are tried first. Defaults to ``0``.
    description : str
        Human-readable description. Defaults to empty.
    """

    template: str
    condition_name: str = "always"
    priority: int = 0
    description: str = ""

    def build(self) -> TemplateVariant:
        """Build a :class:`TemplateVariant` from this spec.

        Returns
        -------
        TemplateVariant
            Live variant with the named predicate resolved.

        Raises
        ------
        KeyError
            If ``condition_name`` is not registered.
        """
        return TemplateVariant(
            template=self.template,
            condition=get_context_predicate(self.condition_name),
            priority=self.priority,
            description=self.description,
        )


class AnchorSpec(BeadBaseModel):
    """Declarative form of :class:`SemanticAnchor` for config files.

    Pole labels are flattened to two string fields rather than a
    nested :class:`SemanticPoles`; ``build()`` constructs the embedded
    model.

    Attributes
    ----------
    name : str
        Short identifier.
    target_property : str
        The property being measured.
    canonical_prompt : str
        Reference phrasing.
    options : tuple[str, ...]
        Ordered response options.
    is_ordered : bool
        Whether the response space is ordinal. Defaults to ``True``.
    semantic_pole_low : str | None
        Low-pole label, when ordered. Defaults to ``None``.
    semantic_pole_high : str | None
        High-pole label, when ordered. Defaults to ``None``.
    required_span_labels : frozenset[str]
        Span labels every realization must reference. Defaults to
        the empty set.
    required_keywords : frozenset[str]
        Keywords every realization must contain. Defaults to the
        empty set.
    embedding_center : tuple[float, ...] | None
        Pre-computed canonical-prompt embedding. Defaults to ``None``.
    max_drift : float
        Maximum cosine distance for embedding drift. Defaults to
        ``0.3``.
    description : str
        Human-readable description.
    """

    name: str
    target_property: str
    canonical_prompt: str
    options: tuple[str, ...]
    is_ordered: bool = True
    semantic_pole_low: str | None = None
    semantic_pole_high: str | None = None
    required_span_labels: frozenset[str] = dx.field(default_factory=frozenset)
    required_keywords: frozenset[str] = dx.field(default_factory=frozenset)
    embedding_center: tuple[float, ...] | None = None
    max_drift: float = 0.3
    description: str = ""

    def build(self) -> SemanticAnchor:
        """Build a :class:`SemanticAnchor` from this spec.

        Returns
        -------
        SemanticAnchor
            Live anchor.

        Raises
        ------
        ValueError
            If exactly one of ``semantic_pole_low`` and
            ``semantic_pole_high`` is supplied.
        """
        if (self.semantic_pole_low is None) != (self.semantic_pole_high is None):
            raise ValueError(
                f"AnchorSpec {self.name!r} sets only one pole; both "
                f"semantic_pole_low and semantic_pole_high must be set "
                f"or both must be None"
            )
        poles: SemanticPoles | None = None
        if self.semantic_pole_low is not None and self.semantic_pole_high is not None:
            poles = SemanticPoles(
                low=self.semantic_pole_low,
                high=self.semantic_pole_high,
            )
        space = ResponseSpace(
            options=self.options,
            is_ordered=self.is_ordered,
            semantic_poles=poles,
        )
        return SemanticAnchor(
            name=self.name,
            target_property=self.target_property,
            canonical_prompt=self.canonical_prompt,
            response_space=space,
            required_span_labels=self.required_span_labels,
            required_keywords=self.required_keywords,
            embedding_center=self.embedding_center,
            max_drift=self.max_drift,
            description=self.description,
        )


class DriftConfig(BeadBaseModel):
    """Configuration for the drift guard applied to a protocol.

    Every realized prompt runs through one shared
    :class:`~bead.protocol.DriftGuard` configured by this section.

    Attributes
    ----------
    min_length : int
        Minimum non-whitespace length for the structural validator.
        Defaults to ``15``.
    require_question_mark : bool
        Whether a trailing ``?`` is required. Defaults to ``True``.
    keyword_case_sensitive : bool
        Whether structural keyword checks are case-sensitive. Defaults
        to ``False``.
    embedding_max_distance : float | None
        Cosine-distance ceiling for the embedding validator. ``None``
        defers to each anchor's ``max_drift``. Defaults to ``None``.
    enable_embedding : bool
        Whether to add an :class:`EmbeddingDriftValidator`. Requires an
        embedding adapter at build time. Defaults to ``False``.
    enable_perplexity : bool
        Whether to add a :class:`PerplexityDriftValidator`. Requires a
        perplexity adapter at build time. Defaults to ``False``.
    max_perplexity : float
        Perplexity ceiling for the perplexity validator. Defaults to
        ``100.0``.
    """

    min_length: int = 15
    require_question_mark: bool = True
    keyword_case_sensitive: bool = False
    embedding_max_distance: float | None = None
    enable_embedding: bool = False
    enable_perplexity: bool = False
    max_perplexity: float = 100.0

    def build(
        self,
        *,
        embedding_adapter: EmbeddingAdapter | None = None,
        perplexity_adapter: PerplexityAdapter | None = None,
    ) -> DriftGuard:
        """Build a :class:`DriftGuard` with structural + optional checks.

        Parameters
        ----------
        embedding_adapter : EmbeddingAdapter | None, optional
            Required when :attr:`enable_embedding` is ``True``. Defaults
            to ``None``.
        perplexity_adapter : PerplexityAdapter | None, optional
            Required when :attr:`enable_perplexity` is ``True``. Defaults
            to ``None``.

        Returns
        -------
        DriftGuard
            Live composite drift validator.

        Raises
        ------
        ValueError
            If a validator is enabled but its adapter was not supplied.
        """
        guard = DriftGuard()
        guard.add(
            StructuralDriftValidator(
                min_length=self.min_length,
                require_question_mark=self.require_question_mark,
                keyword_case_sensitive=self.keyword_case_sensitive,
            )
        )
        if self.enable_embedding:
            if embedding_adapter is None:
                raise ValueError(
                    "drift.enable_embedding=True but no "
                    "embedding_adapter was supplied to build()"
                )
            guard.add(
                EmbeddingDriftValidator(
                    embedding_adapter,
                    max_distance=self.embedding_max_distance,
                )
            )
        if self.enable_perplexity:
            if perplexity_adapter is None:
                raise ValueError(
                    "drift.enable_perplexity=True but no "
                    "perplexity_adapter was supplied to build()"
                )
            guard.add(
                PerplexityDriftValidator(
                    perplexity_adapter,
                    max_perplexity=self.max_perplexity,
                )
            )
        return guard


class FamilySpec(BeadBaseModel):
    """Declarative form of :class:`QuestionFamily` for config files.

    Attributes
    ----------
    anchor : AnchorSpec
        The anchor declaration. Built into a
        :class:`SemanticAnchor` at build time.
    realization_kind : RealizationKind
        Which realization strategy to use.
    template : str | None
        Used when ``realization_kind="template"``. ``None`` defers to
        the anchor's canonical prompt.
    variants : tuple[TemplateVariantSpec, ...]
        Used when ``realization_kind="contextual"``. Empty tuple is
        invalid for that kind.
    fallback : str | None
        Fallback template used when no variant matches. ``None``
        defers to the anchor's canonical prompt.
    condition_name : str
        Registered predicate name controlling family applicability.
        Defaults to ``"always"``.
    depends_on : tuple[str, ...]
        Names of anchors whose responses must precede this family in
        the protocol. Defaults to the empty tuple.
    fallback_on_drift : bool
        Whether to fall back to the canonical prompt on drift failure.
        Defaults to ``True``.
    """

    anchor: dx.Embed[AnchorSpec]
    realization_kind: RealizationKind = "template"
    template: str | None = None
    variants: tuple[dx.Embed[TemplateVariantSpec], ...] = ()
    fallback: str | None = None
    condition_name: str = "always"
    depends_on: tuple[str, ...] = ()
    fallback_on_drift: bool = True

    def _build_realization(
        self,
        *,
        lm_client: LMClient | None,
        lm_model_name: str,
        cache: ModelOutputCache | None,
        lm_temperature: float,
        lm_max_tokens: int,
    ) -> RealizationStrategy:
        """Construct the realization strategy named by ``realization_kind``."""
        if self.realization_kind == "template":
            return TemplateRealization(template=self.template)
        if self.realization_kind == "contextual":
            if not self.variants:
                raise ValueError(
                    f"FamilySpec {self.anchor.name!r} has "
                    f"realization_kind='contextual' but variants is empty"
                )
            return ContextualTemplateRealization(
                variants=tuple(v.build() for v in self.variants),
                fallback=self.fallback,
            )
        if self.realization_kind == "lm":
            if lm_client is None:
                raise ValueError(
                    f"FamilySpec {self.anchor.name!r} has "
                    f"realization_kind='lm' but no lm_client was "
                    f"supplied to ProtocolConfig.build()"
                )
            return LMRealization(
                lm_client,
                model_name=lm_model_name,
                cache=cache,
                temperature=lm_temperature,
                max_tokens=lm_max_tokens,
            )
        raise ValueError(f"Unknown realization_kind: {self.realization_kind!r}")

    def build(
        self,
        *,
        drift_guard: DriftGuard,
        lm_client: LMClient | None,
        lm_model_name: str,
        cache: ModelOutputCache | None,
        lm_temperature: float,
        lm_max_tokens: int,
    ) -> QuestionFamily:
        """Build a :class:`QuestionFamily` from this spec.

        Parameters
        ----------
        drift_guard : DriftGuard
            Shared drift guard for the protocol.
        lm_client : LMClient | None
            LM backend; required when ``realization_kind == "lm"``.
        lm_model_name : str
            Cache-key prefix for LM realizations.
        cache : ModelOutputCache | None
            Output cache for LM realizations.
        lm_temperature : float
            Sampling temperature for LM realizations.
        lm_max_tokens : int
            Maximum response length for LM realizations.

        Returns
        -------
        QuestionFamily
            Live family.
        """
        return QuestionFamily(
            anchor=self.anchor.build(),
            realization=self._build_realization(
                lm_client=lm_client,
                lm_model_name=lm_model_name,
                cache=cache,
                lm_temperature=lm_temperature,
                lm_max_tokens=lm_max_tokens,
            ),
            drift_guard=drift_guard,
            condition=get_context_predicate(self.condition_name),
            depends_on=self.depends_on,
            fallback_on_drift=self.fallback_on_drift,
        )


def _default_drift() -> DriftConfig:
    return DriftConfig()


class ProtocolConfig(BeadBaseModel):
    """Top-level annotation-protocol stage configuration.

    Plugs into :class:`~bead.config.config.BeadConfig` as the
    ``protocol`` field. Declares the families, drift settings, and
    LM defaults for an annotation protocol that can be loaded from
    YAML or TOML and materialized via :meth:`build`.

    Attributes
    ----------
    name : str
        Descriptive protocol name. Defaults to empty.
    families : tuple[FamilySpec, ...]
        Declarative family specs in protocol order. Defaults to the
        empty tuple.
    drift : DriftConfig
        Drift-guard configuration shared by all families. Defaults to
        a structural-only guard with the standard defaults.
    lm_model_name : str
        Cache-key prefix for LM realizations. Used when any family
        has ``realization_kind="lm"``. Defaults to empty (forces the
        caller to set it explicitly when LM realizations are used).
    lm_temperature : float
        Default sampling temperature for LM realizations. Defaults to
        ``0.3``.
    lm_max_tokens : int
        Default maximum response length for LM realizations. Defaults
        to ``200``.
    """

    name: str = ""
    families: tuple[dx.Embed[FamilySpec], ...] = ()
    drift: dx.Embed[DriftConfig] = dx.field(default_factory=_default_drift)
    lm_model_name: str = ""
    lm_temperature: float = 0.3
    lm_max_tokens: int = 200

    def build(
        self,
        *,
        lm_client: LMClient | None = None,
        cache: ModelOutputCache | None = None,
        embedding_adapter: EmbeddingAdapter | None = None,
        perplexity_adapter: PerplexityAdapter | None = None,
    ) -> AnnotationProtocol:
        """Materialize the configured protocol.

        Parameters
        ----------
        lm_client : LMClient | None, optional
            LM backend, required if any family declares
            ``realization_kind="lm"``. Defaults to ``None``.
        cache : ModelOutputCache | None, optional
            Output cache for LM realizations. Defaults to ``None``.
        embedding_adapter : EmbeddingAdapter | None, optional
            Required when ``drift.enable_embedding=True``. Defaults to
            ``None``.
        perplexity_adapter : PerplexityAdapter | None, optional
            Required when ``drift.enable_perplexity=True``. Defaults to
            ``None``.

        Returns
        -------
        AnnotationProtocol
            Live protocol with every family materialized in declared
            order.

        Raises
        ------
        ValueError
            If a required runtime dependency is not supplied or a
            family declares an unknown realization kind.
        """
        guard = self.drift.build(
            embedding_adapter=embedding_adapter,
            perplexity_adapter=perplexity_adapter,
        )
        families = [
            family_spec.build(
                drift_guard=guard,
                lm_client=lm_client,
                lm_model_name=self.lm_model_name,
                cache=cache,
                lm_temperature=self.lm_temperature,
                lm_max_tokens=self.lm_max_tokens,
            )
            for family_spec in self.families
        ]
        return AnnotationProtocol(families=families, name=self.name)
