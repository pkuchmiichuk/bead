"""Round-trip law tests for the participant / agent-reference lens."""

from __future__ import annotations

from datetime import datetime

from lairs.records import defs

from bead.evaluation.reliability import AnnotationRecord
from bead.interop.layers._convert import read_feature_map
from bead.interop.layers.judgment_lens import (
    judgment_set_to_records,
    records_to_judgment_set,
)
from bead.interop.layers.participant_lens import (
    PARTICIPANT_AGENT,
    agent_ref_of,
    participant_features,
)
from bead.participants.models import Participant


def _record(
    *, annotator_id: str = "ann_1", item_id: str = "item_1"
) -> AnnotationRecord:
    return AnnotationRecord(
        annotator_id=annotator_id,
        item_id=item_id,
        question_name="acceptability",
        response_label="option_a",
    )


def _participant() -> Participant:
    return Participant(
        participant_metadata={"age": 34, "native_speaker": True, "l1": "eng"},
        study_id="megaacceptability",
        session_ids=("session_a", "session_b"),
        consent_timestamp=datetime(2026, 6, 1, 9, 30, 0),
        notes="recruited via prolific",
    )


class TestParticipantAgent:
    """Participant <-> layers agentRef."""

    def test_view_is_agent_ref(self) -> None:
        participant = _participant()
        view, _ = PARTICIPANT_AGENT.forward(participant)
        assert isinstance(view, defs.AgentRef)
        assert view.id == str(participant.id)

    def test_roundtrip_exact(self) -> None:
        participant = _participant()
        view, complement = PARTICIPANT_AGENT.forward(participant)
        assert PARTICIPANT_AGENT.backward(view, complement) == participant

    def test_roundtrip_through_serialization(self) -> None:
        participant = _participant()
        view, complement = PARTICIPANT_AGENT.forward(participant)
        view2 = defs.AgentRef.model_validate_json(view.model_dump_json())
        assert PARTICIPANT_AGENT.backward(view2, complement) == participant

    def test_roundtrip_minimal(self) -> None:
        participant = Participant()
        view, complement = PARTICIPANT_AGENT.forward(participant)
        assert PARTICIPANT_AGENT.backward(view, complement) == participant


class TestParticipantFeatures:
    """A participant's study fields render to a layers featureMap."""

    def test_features_decode_to_study_fields(self) -> None:
        participant = _participant()
        features = participant_features(participant)
        assert features is not None
        decoded = read_feature_map(features)
        assert decoded["age"] == 34
        assert decoded["native_speaker"] is True
        assert decoded["study_id"] == "megaacceptability"
        assert decoded["session_ids"] == ("session_a", "session_b")
        assert decoded["consent_timestamp"] == "2026-06-01T09:30:00"
        assert decoded["notes"] == "recruited via prolific"

    def test_features_none_when_empty(self) -> None:
        assert participant_features(Participant()) is None

    def test_agent_ref_of_carries_uuid(self) -> None:
        participant = _participant()
        assert agent_ref_of(participant).id == str(participant.id)


class TestJudgmentSetWithParticipant:
    """A participant enriches the judgment set's agent and features."""

    def test_participant_populates_agent_and_features(self) -> None:
        participant = _participant()
        records = (
            _record(item_id="item_1"),
            _record(item_id="item_2"),
        )
        view, _ = records_to_judgment_set(records, participant)
        assert view.agent is not None
        assert view.agent.id == str(participant.id)
        assert view.features is not None
        assert read_feature_map(view.features)["study_id"] == "megaacceptability"

    def test_records_still_roundtrip_with_participant(self) -> None:
        records = (
            _record(item_id="item_1"),
            _record(item_id="item_2"),
        )
        view, complement = records_to_judgment_set(records, _participant())
        assert judgment_set_to_records(view, complement) == records
