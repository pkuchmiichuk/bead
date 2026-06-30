"""Bridge from the protocol layer to jsPsych deployment.

End-to-end path from a configured :class:`AnnotationProtocol` and a
sequence of :class:`~bead.protocol.ProtocolContext` records to a list
of jsPsych trial objects ready for batch deployment.

This is the canonical bridge to deployment. There is no other way to
materialize a protocol-defined experiment.
"""

from __future__ import annotations

from collections.abc import Iterable

from bead.data.base import JsonValue
from bead.deployment.jspsych.config import (
    ChoiceConfig,
    ExperimentConfig,
    RatingScaleConfig,
)
from bead.deployment.jspsych.trials import create_trial
from bead.items.item_template import ItemTemplate, JudgmentType, PresentationSpec
from bead.protocol.context import ProtocolContext
from bead.protocol.family import AnnotationProtocol
from bead.protocol.items import (
    protocol_to_item_templates,
    realize_protocol_to_items,
)


def protocol_to_jspsych_trials(
    protocol: AnnotationProtocol,
    contexts: Iterable[ProtocolContext],
    *,
    experiment_config: ExperimentConfig,
    judgment_type: JudgmentType,
    presentation_spec: PresentationSpec | None = None,
    rating_config: RatingScaleConfig | None = None,
    choice_config: ChoiceConfig | None = None,
) -> list[dict[str, JsonValue]]:
    """Materialize an entire protocol as a flat list of jsPsych trials.

    Each :class:`ProtocolContext` is realized through every applicable
    :class:`~bead.protocol.QuestionFamily`. Each resulting realization
    is packaged as an :class:`~bead.items.item.Item` bound to the
    family's :class:`ItemTemplate` and turned into a jsPsych trial
    via :func:`bead.deployment.jspsych.trials.create_trial`. Trials
    are returned in
    ``(context_order, family_order)`` order: every realized question
    for the first context comes first, then the second context, and
    so on.

    Parameters
    ----------
    protocol : AnnotationProtocol
        Configured protocol whose families to realize.
    contexts : Iterable[ProtocolContext]
        Contexts to realize, one per annotation target.
    experiment_config : ExperimentConfig
        Shared experiment configuration applied to every trial.
    judgment_type : JudgmentType
        Common judgment type assigned to every per-family
        :class:`ItemTemplate`.
    presentation_spec : PresentationSpec | None, optional
        Common presentation spec across families. Defaults to a fresh
        :class:`PresentationSpec` per template.
    rating_config : RatingScaleConfig | None, optional
        Configuration for rating-scale trials (ordinal task type).
    choice_config : ChoiceConfig | None, optional
        Configuration for choice trials (binary, categorical, or
        forced-choice task types).

    Returns
    -------
    list[dict[str, JsonValue]]
        Flat list of jsPsych trial dicts in ``trial_number`` order.
    """
    templates: dict[str, ItemTemplate] = protocol_to_item_templates(
        protocol,
        judgment_type=judgment_type,
        presentation_spec=presentation_spec,
    )

    template_by_id = {t.id: t for t in templates.values()}

    trials: list[dict[str, JsonValue]] = []
    trial_number = 0
    for ctx in contexts:
        for _realization, item in realize_protocol_to_items(
            protocol,
            ctx,
            judgment_type=judgment_type,
            item_templates=templates,
            presentation_spec=presentation_spec,
        ):
            trials.append(
                create_trial(
                    item=item,
                    template=template_by_id[item.item_template_id],
                    experiment_config=experiment_config,
                    trial_number=trial_number,
                    rating_config=rating_config,
                    choice_config=choice_config,
                )
            )
            trial_number += 1

    return trials
