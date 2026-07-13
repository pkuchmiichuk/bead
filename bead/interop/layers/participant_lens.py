"""Lens between a bead ``Participant`` and its layers representation.

layers has no dedicated participant record. It represents whoever produced data
as an ``agentRef`` (:class:`lairs.records.defs.AgentRef`) whose docstring is
explicit: it "separates the identity of the producer from the interpretive
framework (persona) and the software used (tool)". A participant's study-level
attributes (demographics, session metadata, consent, payment) have their own
native home: the ``featureMap`` on the ``judgmentSet`` that binds one annotator
to one experiment, documented to hold "annotator demographics, session metadata,
completion time, payment info". A ``persona`` is the annotator *role* and
interpretive framework, not a concrete enrolled participant, so it is not the
target here.

This module maps a bead :class:`~bead.participants.models.Participant` onto that
representation:

- :data:`PARTICIPANT_AGENT` is a lossless lens
  ``Participant <-> (agentRef, complement)``: the participant's UUID becomes the
  ``agentRef`` identity and the bead framework identity plus the study fields ride
  in the lens complement so reconstruction is exact.
- :func:`participant_features` renders a participant's study fields as a layers
  ``featureMap`` for attaching to a ``judgmentSet``, the schema's native slot for
  participant-level attributes.
- :func:`agent_ref_of` is the identity half on its own.
"""

from __future__ import annotations

import json
from datetime import datetime

import didactic.api as dx
from lairs.records import defs

from bead.data.base import JsonValue
from bead.interop.layers._convert import (
    apply_identity,
    identity_of,
    j_list,
    j_obj,
    j_str,
    j_str_or_none,
)
from bead.participants.models import Participant


def agent_ref_of(participant: Participant) -> defs.AgentRef:
    """Build the layers ``agentRef`` identity for a participant.

    The internal UUID becomes the ``agentRef`` ``id``; the layers docstring
    blesses ``id`` for "anonymized crowdworker ID, platform username" and similar
    opaque identifiers, which is exactly a bead participant UUID.
    """
    return defs.AgentRef(id=str(participant.id))


def participant_features(participant: Participant) -> defs.FeatureMap | None:
    """Render a participant's study fields as a layers ``featureMap``.

    Collects ``participant_metadata`` (demographics) together with ``study_id``,
    ``session_ids``, ``consent_timestamp``, and ``notes`` into the ``featureMap``
    that a ``judgmentSet`` documents as the home for "annotator demographics,
    session metadata, completion time, payment info". Each value is serialized
    with ``json.dumps`` (the shared feature-map convention), so
    :func:`bead.interop.layers._convert.read_feature_map` decodes it exactly.
    Returns ``None`` when the participant carries no study fields (a faithful
    layers view omits empty optionals).
    """
    entries: dict[str, JsonValue] = dict(participant.participant_metadata)
    if participant.study_id is not None:
        entries["study_id"] = participant.study_id
    if participant.session_ids:
        entries["session_ids"] = participant.session_ids
    if participant.consent_timestamp is not None:
        entries["consent_timestamp"] = participant.consent_timestamp.isoformat()
    if participant.notes is not None:
        entries["notes"] = participant.notes
    if not entries:
        return None
    return defs.FeatureMap(
        entries=tuple(
            defs.Feature(key=key, value=json.dumps(entries[key])) for key in entries
        )
    )


class ParticipantAgentLens(dx.Lens[Participant, defs.AgentRef, JsonValue]):
    """Lossless lens ``Participant <-> (layers agentRef, complement)``.

    The layers ``agentRef`` (:class:`lairs.records.defs.AgentRef`) is the
    canonical identity of a data producer, so a participant projects to the
    ``agentRef`` carrying its UUID. Everything else, the bead framework identity
    and the study fields (demographics, study id, sessions, consent, notes),
    rides in the lens complement, since a standalone ``agentRef`` has no slot for
    study-level attributes (those attach to a ``judgmentSet`` via
    :func:`participant_features`). The GetPut/PutGet laws guarantee an exact
    round-trip.
    """

    def forward(self, participant: Participant) -> tuple[defs.AgentRef, JsonValue]:
        """Project a participant to a layers agent reference and complement."""
        view = agent_ref_of(participant)
        complement: JsonValue = {
            "identity": identity_of(participant),
            "participant_metadata": dict(participant.participant_metadata),
            "study_id": participant.study_id,
            "session_ids": participant.session_ids,
            "consent_timestamp": (
                participant.consent_timestamp.isoformat()
                if participant.consent_timestamp is not None
                else None
            ),
            "notes": participant.notes,
        }
        return view, complement

    def backward(self, view: defs.AgentRef, complement: JsonValue) -> Participant:
        """Reconstruct a participant from its layers agent reference + complement."""
        comp = j_obj(complement)
        consent = comp["consent_timestamp"]
        participant = Participant(
            participant_metadata=j_obj(comp["participant_metadata"]),
            study_id=j_str_or_none(comp["study_id"]),
            session_ids=tuple(j_str(value) for value in j_list(comp["session_ids"])),
            consent_timestamp=(
                datetime.fromisoformat(j_str(consent)) if consent is not None else None
            ),
            notes=j_str_or_none(comp["notes"]),
        )
        return apply_identity(participant, comp["identity"])


PARTICIPANT_AGENT = ParticipantAgentLens()
