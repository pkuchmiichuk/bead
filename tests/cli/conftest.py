"""Test fixtures for CLI tests."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock
from uuid import uuid4

import pytest
from click.testing import CliRunner
from pytest_mock import MockerFixture

from bead.deployment.jatos.api import JATOSClient
from bead.items.adapters.base import ModelAdapter
from bead.items.item import Item, ModelOutput
from bead.lists import ExperimentList
from bead.resources.lexical_item import LexicalItem
from bead.resources.lexicon import Lexicon
from bead.resources.template import Slot, Template
from bead.resources.template_collection import TemplateCollection
from bead.templates.filler import FilledTemplate


@pytest.fixture
def cli_runner() -> CliRunner:
    """Provide Click CLI test runner.

    Returns
    -------
    CliRunner
        Click test runner.
    """
    return CliRunner()


@pytest.fixture
def temp_project_dir(tmp_path: Path) -> Path:
    """Provide temporary project directory.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.

    Returns
    -------
    Path
        Temporary project directory.
    """
    return tmp_path / "test_project"


@pytest.fixture
def mock_config_file(tmp_path: Path) -> Path:
    """Create mock bead.yaml config file.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.

    Returns
    -------
    Path
        Path to mock config file.
    """
    config_file = tmp_path / "bead.yaml"
    config_content = """
profile: test

logging:
  level: DEBUG
  format: "%(message)s"

paths:
  data_dir: .test_data

resources:
  cache_external: true

templates:
  filling_strategy: random
  max_combinations: 100
  random_seed: 42
"""
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def mock_invalid_config_file(tmp_path: Path) -> Path:
    """Create invalid config file for error testing.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.

    Returns
    -------
    Path
        Path to invalid config file.
    """
    config_file = tmp_path / "invalid.yaml"
    # Invalid YAML syntax
    config_content = """
profile: test
logging:
  level: DEBUG
  invalid_indentation
"""
    config_file.write_text(config_content)
    return config_file


@pytest.fixture
def mock_config_with_validation_errors(tmp_path: Path) -> Path:
    """Create config file with validation errors.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.

    Returns
    -------
    Path
        Path to config file with validation errors.
    """
    config_file = tmp_path / "validation_errors.yaml"
    config_content = """
profile: test

templates:
  filling_strategy: mlm
  # Missing mlm_model_name - validation error
  max_combinations: -1  # Invalid negative value

lists:
  n_lists: -1  # Invalid negative value
"""
    config_file.write_text(config_content)
    return config_file


# Pipeline Stage Fixtures


@pytest.fixture
def mock_lexicon_file(tmp_path: Path) -> Path:
    """Create mock lexicon file.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.

    Returns
    -------
    Path
        Path to mock lexicon file.
    """
    lexicon = Lexicon(
        name="test_lexicon",
        language_code="eng",
    )

    # Add items using add() method since items is a dict
    lexicon = lexicon.with_item(
        LexicalItem(
            lemma="run",
            language_code="eng",
            features={"pos": "VERB"},
        )
    )
    lexicon = lexicon.with_item(
        LexicalItem(
            lemma="walk",
            language_code="eng",
            features={"pos": "VERB"},
        )
    )

    lexicon_file = tmp_path / "test_lexicon.jsonl"
    lexicon.to_jsonl(str(lexicon_file))

    return lexicon_file


@pytest.fixture
def mock_template_file(tmp_path: Path) -> Path:
    """Create mock template file.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.

    Returns
    -------
    Path
        Path to mock template file.
    """
    template = Template(
        name="test_template",
        template_string="{subject} {verb} the {object}",
        language_code="eng",
        slots={
            "subject": Slot(name="subject", required=True),
            "verb": Slot(name="verb", required=True),
            "object": Slot(name="object", required=True),
        },
    )

    collection = TemplateCollection(
        name="test_templates",
        description="Test templates",
    )
    collection = collection.with_template(template)

    template_file = tmp_path / "test_templates.jsonl"
    collection.to_jsonl(str(template_file))

    return template_file


@pytest.fixture
def mock_filled_templates_file(tmp_path: Path) -> Path:
    """Create mock filled templates file.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.

    Returns
    -------
    Path
        Path to mock filled templates file.
    """
    # Create lexical items for slot fillers
    subject = LexicalItem(lemma="dog", language_code="eng", features={"pos": "NOUN"})
    verb = LexicalItem(lemma="ran", language_code="eng", features={"pos": "VERB"})
    obj = LexicalItem(lemma="ball", language_code="eng", features={"pos": "NOUN"})

    filled_template = FilledTemplate(
        template_id=str(uuid4()),
        template_name="test_template",
        slot_fillers={
            "subject": subject,
            "verb": verb,
            "object": obj,
        },
        rendered_text="dog ran the ball",
        strategy_name="exhaustive",
    )

    filled_file = tmp_path / "filled_templates.jsonl"
    with open(filled_file, "w") as f:
        f.write(filled_template.model_dump_json() + "\n")

    return filled_file


@pytest.fixture
def mock_items_file(tmp_path: Path) -> Path:
    """Create mock items file.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.

    Returns
    -------
    Path
        Path to mock items file.
    """
    item = Item(
        item_template_id=uuid4(),
        filled_template_refs=[uuid4()],
        rendered_elements={"sentence": "The dog ran quickly"},
        model_outputs=[
            ModelOutput(
                model_name="gpt2",
                model_version="1.0",
                operation="log_probability",
                inputs={"text": "The dog ran quickly"},
                output=-2.5,
                cache_key="abc123",
            )
        ],
    )

    items_file = tmp_path / "items.jsonl"
    with open(items_file, "w") as f:
        f.write(item.model_dump_json() + "\n")

    return items_file


@pytest.fixture
def mock_experiment_lists_file(tmp_path: Path) -> Path:
    """Create mock experiment lists JSONL file.

    Parameters
    ----------
    tmp_path : Path
        Pytest temporary directory.

    Returns
    -------
    Path
        Path to mock experiment lists JSONL file.
    """
    exp_list = ExperimentList(
        name="list_1",
        list_number=1,
        item_refs=[uuid4(), uuid4(), uuid4()],
    )

    lists_file = tmp_path / "lists.jsonl"
    with open(lists_file, "w") as f:
        f.write(exp_list.model_dump_json() + "\n")

    return lists_file


@pytest.fixture
def mock_jatos_api(mocker: MockerFixture) -> MagicMock:
    """Mock JATOS API client.

    Parameters
    ----------
    mocker : pytest_mock.MockerFixture
        Pytest mocker fixture.

    Returns
    -------
    MagicMock
        Mocked JATOS client.
    """
    mock_client = mocker.Mock(spec=JATOSClient)
    mock_client.get_study_info.return_value = {
        "id": 1,
        "title": "Test Study",
        "description": "Test description",
    }
    mock_client.get_results.return_value = [
        {"result_id": 1, "data": {"response": "test"}},
        {"result_id": 2, "data": {"response": "test2"}},
    ]
    return mock_client


@pytest.fixture
def mock_model_adapter(mocker: MockerFixture) -> MagicMock:
    """Mock model adapter.

    Parameters
    ----------
    mocker : pytest_mock.MockerFixture
        Pytest mocker fixture.

    Returns
    -------
    MagicMock
        Mocked model adapter.
    """
    mock_adapter = mocker.Mock(spec=ModelAdapter)
    mock_adapter.predict.return_value = {"score": 0.85}
    return mock_adapter
