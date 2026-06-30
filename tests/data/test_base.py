"""Tests for BeadBaseModel."""

from __future__ import annotations

import time

import didactic.api as dx
import pytest

from bead.data.base import BeadBaseModel
from bead.data.identifiers import is_valid_uuid7


class SampleModel(BeadBaseModel):
    """Sample model for base model tests."""

    name: str
    value: int


def test_beadbasemodel_creates_uuid() -> None:
    obj = SampleModel(name="test", value=42)
    assert obj.id is not None
    assert is_valid_uuid7(obj.id)


def test_beadbasemodel_creates_timestamps() -> None:
    obj = SampleModel(name="test", value=42)
    assert obj.created_at is not None
    assert obj.modified_at is not None
    assert obj.created_at.tzinfo is not None
    assert obj.modified_at.tzinfo is not None


def test_beadbasemodel_default_version() -> None:
    obj = SampleModel(name="test", value=42)
    assert obj.version == "1.0.0"


def test_beadbasemodel_default_metadata() -> None:
    obj = SampleModel(name="test", value=42)
    assert obj.metadata == {}


def test_beadbasemodel_touched_returns_new_instance() -> None:
    """``touched`` returns a new instance with a newer ``modified_at``."""
    obj = SampleModel(name="test", value=42)
    time.sleep(0.01)
    refreshed = obj.touched()
    assert refreshed is not obj
    assert refreshed.modified_at > obj.modified_at
    assert refreshed.id == obj.id


def test_beadbasemodel_forbids_extra_fields() -> None:
    with pytest.raises(dx.ValidationError):
        SampleModel.model_validate(
            {"name": "test", "value": 42, "extra_field": "not allowed"}
        )


def test_beadbasemodel_is_frozen() -> None:
    """Frozen Models reject in-place attribute assignment."""
    obj = SampleModel(name="test", value=42)
    with pytest.raises((AttributeError, TypeError)):
        obj.value = 99


def test_beadbasemodel_timestamps_ordered() -> None:
    obj = SampleModel(name="test", value=42)
    assert obj.created_at <= obj.modified_at


def test_beadbasemodel_custom_metadata() -> None:
    metadata = {"key": "value", "number": 42}
    obj = SampleModel(name="test", value=42, metadata=metadata)
    assert obj.metadata == metadata


def test_beadbasemodel_custom_version() -> None:
    obj = SampleModel(name="test", value=42, version="2.0.0")
    assert obj.version == "2.0.0"
