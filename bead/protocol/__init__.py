"""Annotation protocol primitives.

This package provides the type-theoretic stack for defining annotation
protocols independent of any specific linguistic domain. The design is
organized around four roles:

- :class:`SemanticAnchor` is the *type* of a question: a declarative
  specification of what is being measured, independent of how the
  question is phrased.
- :class:`ProtocolContext` is the dependent *index*: everything known
  about the current annotation target. Different contexts license
  different questions and different phrasings.
- :class:`RealizationStrategy` is the computational *content* of the
  dependent function ``Pi(ctx). Question(ctx)``: a strategy that maps
  an anchor and a context to a concrete prompt string.
- :class:`DriftGuard` is the *type-checker*: it verifies that a realized
  prompt still inhabits the type defined by its anchor.

On top of these, :class:`QuestionFamily` packages an anchor with a
realization strategy and a drift guard, and :class:`AnnotationProtocol`
sequences families into the iterated dependent product
``Sigma(a_1 : Q_1(ctx)). Sigma(a_2 : Q_2(ctx, a_1)). ...``, threading
responses through the context so later questions can condition on
earlier answers.

The :mod:`~bead.protocol.encoding` and :mod:`~bead.protocol.diagnostics`
submodules add a likelihood-agnostic response-encoding layer and an
immutable diagnostic-record system used by both the protocol layer and
downstream modeling code.
"""

from __future__ import annotations

from bead.protocol.anchor import ResponseSpace, SemanticAnchor
from bead.protocol.context import (
    ContextItem,
    ContextPredicate,
    ProtocolContext,
    always,
    get_context_predicate,
    list_context_predicates,
    register_context_predicate,
)
from bead.protocol.diagnostics import (
    ConditionalObservationValidator,
    DatasetReport,
    DiagnosticLevel,
    DiagnosticRecord,
    RecordLike,
)
from bead.protocol.drift import (
    DriftGuard,
    DriftScore,
    DriftValidator,
    EmbeddingAdapter,
    EmbeddingDriftValidator,
    PerplexityAdapter,
    PerplexityDriftValidator,
    StructuralDriftValidator,
)
from bead.protocol.encoding import ResponseEncoding, ScaleType, encode_response_space
from bead.protocol.family import (
    AnnotationProtocol,
    ApplicabilityPredicate,
    QuestionFamily,
    QuestionRealization,
)
from bead.protocol.items import (
    family_to_item_template,
    protocol_to_item_templates,
    realization_to_item,
    realize_protocol_to_items,
    scale_type_to_task_type,
)
from bead.protocol.realization import (
    ContextualTemplateRealization,
    LMClient,
    LMRealization,
    RealizationStrategy,
    TemplateRealization,
    TemplateVariant,
)

__all__ = [
    "AnnotationProtocol",
    "ApplicabilityPredicate",
    "ConditionalObservationValidator",
    "ContextItem",
    "ContextPredicate",
    "ContextualTemplateRealization",
    "DatasetReport",
    "DiagnosticLevel",
    "DiagnosticRecord",
    "DriftGuard",
    "DriftScore",
    "DriftValidator",
    "EmbeddingAdapter",
    "EmbeddingDriftValidator",
    "LMClient",
    "LMRealization",
    "PerplexityAdapter",
    "PerplexityDriftValidator",
    "ProtocolContext",
    "QuestionFamily",
    "QuestionRealization",
    "RealizationStrategy",
    "RecordLike",
    "ResponseEncoding",
    "ResponseSpace",
    "ScaleType",
    "SemanticAnchor",
    "StructuralDriftValidator",
    "TemplateRealization",
    "TemplateVariant",
    "always",
    "encode_response_space",
    "family_to_item_template",
    "protocol_to_item_templates",
    "realization_to_item",
    "realize_protocol_to_items",
    "scale_type_to_task_type",
    "get_context_predicate",
    "list_context_predicates",
    "register_context_predicate",
]
