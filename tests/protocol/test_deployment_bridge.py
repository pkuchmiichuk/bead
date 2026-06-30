"""Tests for :mod:`bead.protocol.deployment`."""

from __future__ import annotations

from bead.deployment.distribution import ListDistributionStrategy
from bead.deployment.jspsych.config import (
    ChoiceConfig,
    ExperimentConfig,
    InstructionsConfig,
)
from bead.deployment.protocol_trials import protocol_to_jspsych_trials
from bead.protocol import (
    AnnotationProtocol,
    ProtocolContext,
    QuestionFamily,
    ResponseSpace,
    SemanticAnchor,
)


def _binary_anchor() -> SemanticAnchor:
    return SemanticAnchor(
        name="completion",
        target_property="telicity",
        canonical_prompt="Does [[situation]] reach an endpoint?",
        response_space=ResponseSpace(options=("no", "yes"), is_ordered=False),
        required_span_labels=frozenset({"situation"}),
    )


def _experiment_config() -> ExperimentConfig:
    return ExperimentConfig(
        experiment_type="binary_choice",
        title="test",
        description="test",
        instructions=InstructionsConfig.from_text("Click yes or no."),
        distribution_strategy=ListDistributionStrategy(
            strategy_type="random",
        ),
    )


def test_protocol_to_jspsych_trials_emits_one_per_realization() -> None:
    proto = AnnotationProtocol(families=[QuestionFamily(anchor=_binary_anchor())])
    contexts = [
        ProtocolContext(
            sentence=f"Mary built sandcastle {i}.",
            target_position=2,
            target_span_text=f"built sandcastle {i}",
            target_span_positions=(2, 3, 4),
        )
        for i in range(3)
    ]
    trials = protocol_to_jspsych_trials(
        proto,
        contexts,
        experiment_config=_experiment_config(),
        judgment_type="acceptability",
        choice_config=ChoiceConfig(),
    )
    assert len(trials) == 3
    for trial in trials:
        assert "type" in trial or "stimulus" in trial


def test_protocol_to_jspsych_trials_skips_non_applicable_families() -> None:
    second = SemanticAnchor(
        name="follow_up",
        target_property="follow_up",
        canonical_prompt="Did [[situation]] have a follow-up?",
        response_space=ResponseSpace(options=("no", "yes"), is_ordered=False),
        required_span_labels=frozenset({"situation"}),
    )
    proto = AnnotationProtocol(
        families=[
            QuestionFamily(anchor=_binary_anchor()),
            QuestionFamily(
                anchor=second,
                condition=(
                    lambda ctx: ctx.previous_responses.get("completion") == "yes"
                ),
                depends_on=("completion",),
            ),
        ]
    )
    ctx = ProtocolContext(
        sentence="Mary built a sandcastle.",
        target_position=2,
        target_span_text="built a sandcastle",
        target_span_positions=(2, 3, 4),
    )
    # The placeholder threading injects "no" for completion (first option),
    # so follow_up's condition fails and only "completion" fires.
    trials = protocol_to_jspsych_trials(
        proto,
        [ctx],
        experiment_config=_experiment_config(),
        judgment_type="acceptability",
        choice_config=ChoiceConfig(),
    )
    assert len(trials) == 1
