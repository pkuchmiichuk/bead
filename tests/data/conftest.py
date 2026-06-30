"""Pytest fixtures for data module tests."""

from __future__ import annotations

from pathlib import Path
from uuid import uuid4

import pytest

from bead.data.metadata import MetadataTracker
from tests.data.data_helpers import SimpleTestModel


@pytest.fixture
def simple_test_model() -> type[SimpleTestModel]:
    """Provide SimpleTestModel class.

    Returns
    -------
    type[SimpleTestModel]
        Test model class
    """
    return SimpleTestModel


@pytest.fixture
def sample_test_objects() -> list[SimpleTestModel]:
    """Create sample test objects.

    Returns
    -------
    list[SimpleTestModel]
        List of test objects
    """
    return [
        SimpleTestModel(name="test1", value=1),
        SimpleTestModel(name="test2", value=2),
        SimpleTestModel(name="test3", value=3),
    ]


@pytest.fixture
def repository_storage(tmp_path: Path) -> Path:
    """Create empty repository storage path.

    Parameters
    ----------
    tmp_path : Path
        Pytest's tmp_path fixture

    Returns
    -------
    Path
        Path to repository storage file
    """
    return tmp_path / "repository.jsonl"


@pytest.fixture
def sample_metadata() -> MetadataTracker:
    """Create sample metadata tracker.

    Returns
    -------
    MetadataTracker
        Metadata tracker with sample data
    """
    metadata = MetadataTracker()
    metadata = metadata.with_provenance(uuid4(), "Template", "filled_from")
    metadata = metadata.with_processing("fill_template", {"strategy": "exhaustive"})
    return metadata
