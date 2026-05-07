"""End-to-end integration test of the protocol layer.

Builds a three-question protocol with a conditional family, realizes
through every strategy with a stub LM, validates via a composite
:class:`DriftGuard`, runs reliability metrics over simulated
responses, and verifies the resulting :class:`DatasetReport` is
well-formed.
"""

from __future__ import annotations

from collections.abc import Sequence

from bead.evaluation.reliability import (
    AnnotationRecord,
    annotator_reliability,
    low_entropy_annotators,
)
from bead.protocol import (
    AnnotationProtocol,
    ConditionalObservationValidator,
    ContextItem,
    ContextualTemplateRealization,
    DatasetReport,
    DiagnosticLevel,
    DriftGuard,
    EmbeddingDriftValidator,
    LMRealization,
    PerplexityDriftValidator,
    ProtocolContext,
    QuestionFamily,
    ResponseSpace,
    SemanticAnchor,
    StructuralDriftValidator,
    TemplateRealization,
    TemplateVariant,
    encode_response_space,
)
from bead.protocol.anchor import SemanticPoles


class _StubLMClient:
    def __init__(self) -> None:
        self.calls = 0

    def complete(
        self,
        prompt: str,
        *,
        temperature: float,
        max_tokens: int,
    ) -> str:
        del prompt, temperature, max_tokens
        self.calls += 1
        return "Does anything change in [[situation]] that has an endpoint?"


class _StubAdapter:
    def get_embedding(self, text: str) -> Sequence[float]:
        # Two-cluster deterministic embedding: texts containing
        # "endpoint" map to one direction, everything else to the
        # orthogonal direction. The change anchor uses "changing", so
        # its canonical and realizations share the (0, 1, 0) cluster;
        # the completion and uniformity anchors share the (1, 0, 0)
        # cluster via "endpoint" / "moments".
        if "endpoint" in text or "moments" in text:
            return (1.0, 0.0, 0.0)
        return (0.0, 1.0, 0.0)

    def compute_perplexity(self, text: str) -> float:
        del text
        return 25.0


def _build_anchors() -> tuple[SemanticAnchor, SemanticAnchor, SemanticAnchor]:
    binary = ResponseSpace(options=("no", "yes"), is_ordered=False)
    likert = ResponseSpace(
        options=(
            "definitely no",
            "probably no",
            "unsure",
            "probably yes",
            "definitely yes",
        ),
        is_ordered=True,
        semantic_poles=SemanticPoles(
            low="definitely no",
            high="definitely yes",
        ),
    )

    change = SemanticAnchor(
        name="change",
        target_property="dynamicity",
        canonical_prompt="Is anything changing in [[situation]] over time?",
        response_space=binary,
        required_span_labels=frozenset({"situation"}),
        required_keywords=frozenset({"changing"}),
    )
    completion = SemanticAnchor(
        name="completion",
        target_property="telicity",
        canonical_prompt=("Does [[situation]] reach a definite endpoint?"),
        response_space=likert,
        required_span_labels=frozenset({"situation"}),
        required_keywords=frozenset({"endpoint"}),
    )
    uniformity = SemanticAnchor(
        name="uniformity",
        target_property="homogeneity",
        canonical_prompt=(
            "Are different moments of [[situation]] qualitatively similar?"
        ),
        response_space=binary,
        required_span_labels=frozenset({"situation"}),
        required_keywords=frozenset({"moments"}),
    )
    return change, completion, uniformity


def test_protocol_end_to_end() -> None:
    change_anchor, completion_anchor, uniformity_anchor = _build_anchors()

    adapter = _StubAdapter()
    guard = DriftGuard(
        validators=[
            StructuralDriftValidator(),
            EmbeddingDriftValidator(adapter, max_distance=0.5),
            PerplexityDriftValidator(adapter, max_perplexity=100.0),
        ]
    )

    # Family A: contextual templates by target UPOS
    contextual = ContextualTemplateRealization(
        variants=(
            TemplateVariant(
                template=(
                    "Does anything happen during [[situation]] that is changing?"
                ),
                condition=lambda ctx: ctx.target_upos == "VERB",
                priority=10,
            ),
            TemplateVariant(
                template=("Is [[situation]] something that is changing in any way?"),
                priority=0,
            ),
        ),
    )
    family_change = QuestionFamily(
        anchor=change_anchor,
        realization=contextual,
        drift_guard=guard,
    )

    # Family B: LM realization, conditional on change=='yes'
    family_completion = QuestionFamily(
        anchor=completion_anchor,
        realization=LMRealization(_StubLMClient(), model_name="stub-lm"),
        drift_guard=guard,
        condition=lambda ctx: ctx.previous_responses.get("change") == "yes",
        depends_on=("change",),
    )

    # Family C: plain template, conditional on change=='yes'
    family_uniformity = QuestionFamily(
        anchor=uniformity_anchor,
        realization=TemplateRealization(),
        drift_guard=guard,
        condition=lambda ctx: ctx.previous_responses.get("change") == "yes",
        depends_on=("change",),
    )

    protocol = AnnotationProtocol(
        families=[family_change, family_completion, family_uniformity],
        name="aspect-protocol",
    )

    ctx = ProtocolContext(
        sentence="Mary built a sandcastle.",
        target_form="built",
        target_lemma="build",
        target_upos="VERB",
        target_position=2,
        target_span_text="built a sandcastle",
        target_span_positions=(2, 3, 4),
        dependents=(
            ContextItem(
                head_lemma="Mary",
                head_form="Mary",
                head_upos="PROPN",
                head_position=1,
                span_text="Mary",
            ),
            ContextItem(
                head_lemma="sandcastle",
                head_form="sandcastle",
                head_upos="NOUN",
                head_position=4,
                span_text="a sandcastle",
                attributes={"definiteness": 0.0},
            ),
        ),
    )

    # All three fire when 'change' was answered 'yes'.
    realizations = protocol.realize_all(ctx, responses={"change": "yes"})
    assert [r.anchor.name for r in realizations] == [
        "change",
        "completion",
        "uniformity",
    ]
    for r in realizations:
        assert r.passed_drift_check, r.drift_score

    # Only 'change' fires when no responses are pre-supplied (placeholder
    # for change is its first option, "no", which fails both conditions).
    only_change = protocol.realize_all(ctx)
    assert [r.anchor.name for r in only_change] == ["change"]

    # Encoding round-trip
    encoding = encode_response_space("change", change_anchor.response_space)
    assert encoding.is_binary
    encoding2 = encode_response_space("completion", completion_anchor.response_space)
    assert encoding2.is_ordinal
    assert encoding2.n_levels == 5

    # Reliability over simulated responses
    records = [
        AnnotationRecord(
            annotator_id="a1",
            item_id="i1",
            question_name="change",
            response_label="yes",
        ),
        AnnotationRecord(
            annotator_id="a1",
            item_id="i2",
            question_name="change",
            response_label="no",
        ),
        AnnotationRecord(
            annotator_id="a2",
            item_id="i1",
            question_name="change",
            response_label="yes",
        ),
        AnnotationRecord(
            annotator_id="a2",
            item_id="i2",
            question_name="change",
            response_label="yes",
        ),
    ]
    profiles = annotator_reliability(records)
    assert len(profiles) == 2
    flagged = low_entropy_annotators(profiles, threshold=0.5)
    assert flagged == ("a2",)

    # Conditional dependency check: completion has a response for an
    # item that lacks a 'change' response, which should warn.
    cond_records = {
        "completion": [
            AnnotationRecord(
                annotator_id="a1",
                item_id="i_orphan",
                question_name="completion",
                response_label="probably yes",
            ),
        ],
    }
    cond_validator = ConditionalObservationValidator()
    findings = cond_validator.validate(cond_records, protocol)
    assert len(findings) == 1
    assert findings[0].category == "conditional_missing_dependency"

    # Build a final report
    report = (
        DatasetReport(
            n_records_input=len(records),
            n_items=2,
            n_records_encoded=len(records),
        )
        .with_coverage("change", 1.0)
        .extend(findings)
        .add(DiagnosticLevel.INFO, "summary", "end-to-end test complete")
    )
    summary = report.summary()
    assert "2 items" in summary
    assert "warnings" in summary
