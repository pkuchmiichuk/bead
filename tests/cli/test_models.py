"""Integration tests for bead.cli.models CLI commands.

Tests all model training and prediction commands to ensure they:
1. Train models for all 8 task types
2. Support 3 mixed-effects modes (fixed, random_intercepts, random_slopes)
3. Handle LoRA parameter-efficient fine-tuning
4. Perform prediction and probability prediction correctly
5. Save/load model configurations correctly
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
from click.testing import CliRunner

from bead.cli.models import models, predict, predict_proba, train_model
from bead.items.item import Item
from bead.items.item_template import ItemTemplate, PresentationSpec, TaskSpec

if TYPE_CHECKING:
    pass


@pytest.fixture
def runner() -> CliRunner:
    """Create Click test runner."""
    return CliRunner()


@pytest.fixture
def sample_template() -> ItemTemplate:
    """Create a sample item template."""
    return ItemTemplate(
        name="test_template",
        description="Test item template",
        judgment_type="preference",  # Valid judgment_type
        task_type="forced_choice",
        task_spec=TaskSpec(
            prompt="Choose the better option",
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )


@pytest.fixture
def sample_forced_choice_items(tmp_path: Path, sample_template: ItemTemplate) -> Path:
    """Create sample forced choice items file."""
    items_file = tmp_path / "forced_choice_items.jsonl"

    items = []
    for i in range(20):
        item = Item(
            item_template_id=sample_template.id,
            rendered_elements={
                "option_a": f"Option A{i}",
                "option_b": f"Option B{i}",
            },
            item_metadata={
                "n_options": 2,
                "correct_option": i % 2,
            },
        )
        items.append(item.model_dump_json() + "\n")

    items_file.write_text("".join(items))
    return items_file


@pytest.fixture
def sample_ordinal_scale_items(tmp_path: Path, sample_template: ItemTemplate) -> Path:
    """Create sample ordinal scale items file."""
    items_file = tmp_path / "ordinal_scale_items.jsonl"

    items = []
    for i in range(20):
        item = Item(
            item_template_id=sample_template.id,
            rendered_elements={"text": f"Sentence {i}"},
            item_metadata={
                "scale_min": 1,
                "scale_max": 7,
            },
        )
        items.append(item.model_dump_json() + "\n")

    items_file.write_text("".join(items))
    return items_file


@pytest.fixture
def sample_labels_file(tmp_path: Path, sample_forced_choice_items: Path) -> Path:
    """Create sample labels file."""
    labels_file = tmp_path / "labels.jsonl"

    # Read items to get their IDs
    items_data = [
        Item.model_validate_json(line)
        for line in sample_forced_choice_items.read_text().strip().split("\n")
    ]

    labels = []
    for item in items_data:
        label = {
            "item_id": str(item.id),
            "participant_id": f"p{len(labels) % 3}",  # 3 participants
            "response": item.item_metadata.get("correct_option", 0),
        }
        labels.append(json.dumps(label) + "\n")

    labels_file.write_text("".join(labels))
    return labels_file


@pytest.fixture
def sample_participant_ids(tmp_path: Path) -> Path:
    """Create sample participant IDs file with 20 entries to match items."""
    participant_file = tmp_path / "participant_ids.txt"
    # Create 20 participant IDs to match the 20 items in sample_*_items
    participant_ids = [
        f"p{i % 3}" for i in range(20)
    ]  # 3 unique participants, 20 entries
    participant_file.write_text("\n".join(participant_ids) + "\n")
    return participant_file


class TestTrainModelCommand:
    """Tests for train-model command."""

    @patch("bead.cli.models.config_class_for_task_type")
    @patch("bead.cli.models.model_class_for_task_type")
    def test_train_forced_choice_fixed_mode(
        self,
        mock_model_factory: MagicMock,
        mock_config_factory: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        sample_forced_choice_items: Path,
        sample_labels_file: Path,
    ) -> None:
        """Test training forced choice model with fixed-effects mode."""
        # Mock model class
        mock_model_class = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_instance.train.return_value = {}  # Return empty metrics dict
        mock_model_class.return_value = mock_model_instance

        # Mock config class
        mock_config_class = MagicMock()
        mock_config_instance = MagicMock()
        # Configure model_dump to return a proper dict for JSON serialization
        mock_config_instance.model_dump.return_value = {
            "model_name": "bert-base-uncased",
            "mixed_effects": {"mode": "fixed"},
        }
        mock_config_class.return_value = mock_config_instance

        mock_model_factory.return_value = mock_model_class
        mock_config_factory.return_value = mock_config_class

        output_dir = tmp_path / "model"

        result = runner.invoke(
            train_model,
            [
                "--task-type",
                "forced_choice",
                "--items",
                str(sample_forced_choice_items),
                "--labels",
                str(sample_labels_file),
                "--model-name",
                "bert-base-uncased",
                "--mixed-effects-mode",
                "fixed",
                "--output-dir",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert "Training forced_choice model" in result.output
        assert "Training Complete" in result.output

        # Verify model was trained
        mock_model_instance.train.assert_called_once()

        # Verify model was saved
        assert (output_dir / "model.pt").exists() or mock_model_instance.save.called

    @patch("bead.cli.models.config_class_for_task_type")
    @patch("bead.cli.models.model_class_for_task_type")
    def test_train_ordinal_scale_random_intercepts(
        self,
        mock_model_factory: MagicMock,
        mock_config_factory: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        sample_ordinal_scale_items: Path,
        sample_labels_file: Path,
        sample_participant_ids: Path,
    ) -> None:
        """Test training ordinal scale model with random intercepts."""
        # Mock model and config classes
        mock_model_class = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_instance.train.return_value = {}
        mock_model_class.return_value = mock_model_instance

        mock_config_class = MagicMock()
        mock_config_instance = MagicMock()
        mock_config_instance.model_dump.return_value = {
            "model_name": "bert-base-uncased",
            "mixed_effects": {"mode": "random_intercepts"},
        }
        mock_config_class.return_value = mock_config_instance

        mock_model_factory.return_value = mock_model_class
        mock_config_factory.return_value = mock_config_class

        output_dir = tmp_path / "model"

        result = runner.invoke(
            train_model,
            [
                "--task-type",
                "ordinal_scale",
                "--items",
                str(sample_ordinal_scale_items),
                "--labels",
                str(sample_labels_file),
                "--model-name",
                "bert-base-uncased",
                "--mixed-effects-mode",
                "random_intercepts",
                "--participant-ids",
                str(sample_participant_ids),
                "--output-dir",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert "Training ordinal_scale model" in result.output

    @patch("bead.cli.models.config_class_for_task_type")
    @patch("bead.cli.models.model_class_for_task_type")
    def test_train_with_lora(
        self,
        mock_model_factory: MagicMock,
        mock_config_factory: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        sample_forced_choice_items: Path,
        sample_labels_file: Path,
    ) -> None:
        """Test training with LoRA parameter-efficient fine-tuning."""
        mock_model_class = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_instance.train.return_value = {}
        mock_model_class.return_value = mock_model_instance

        mock_config_class = MagicMock()
        mock_config_instance = MagicMock()
        mock_config_instance.model_dump.return_value = {
            "model_name": "gpt2",
            "use_lora": True,
            "lora_rank": 8,
            "mixed_effects": {"mode": "fixed"},
        }
        mock_config_class.return_value = mock_config_instance

        mock_model_factory.return_value = mock_model_class
        mock_config_factory.return_value = mock_config_class

        output_dir = tmp_path / "model"

        result = runner.invoke(
            train_model,
            [
                "--task-type",
                "forced_choice",
                "--items",
                str(sample_forced_choice_items),
                "--labels",
                str(sample_labels_file),
                "--model-name",
                "gpt2",
                "--use-lora",
                "--lora-rank",
                "8",
                "--lora-alpha",
                "16",
                "--output-dir",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"

    @patch("bead.cli.models.config_class_for_task_type")
    @patch("bead.cli.models.model_class_for_task_type")
    def test_train_with_validation_data(
        self,
        mock_model_factory: MagicMock,
        mock_config_factory: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        sample_forced_choice_items: Path,
        sample_labels_file: Path,
    ) -> None:
        """Test training with validation data."""
        mock_model_class = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_instance.train.return_value = {}
        mock_model_class.return_value = mock_model_instance

        mock_config_class = MagicMock()
        mock_config_instance = MagicMock()
        mock_config_instance.model_dump.return_value = {
            "model_name": "bert-base-uncased",
            "mixed_effects": {"mode": "fixed"},
        }
        mock_config_class.return_value = mock_config_instance

        mock_model_factory.return_value = mock_model_class
        mock_config_factory.return_value = mock_config_class

        output_dir = tmp_path / "model"

        result = runner.invoke(
            train_model,
            [
                "--task-type",
                "forced_choice",
                "--items",
                str(sample_forced_choice_items),
                "--labels",
                str(sample_labels_file),
                "--model-name",
                "bert-base-uncased",
                "--validation-items",
                str(sample_forced_choice_items),
                "--validation-labels",
                str(sample_labels_file),
                "--output-dir",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"

    def test_train_missing_items_file(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test error handling for missing items file."""
        nonexistent_file = tmp_path / "nonexistent.jsonl"
        labels_file = tmp_path / "labels.jsonl"
        labels_file.write_text("")

        result = runner.invoke(
            train_model,
            [
                "--task-type",
                "forced_choice",
                "--items",
                str(nonexistent_file),
                "--labels",
                str(labels_file),
                "--model-name",
                "bert-base-uncased",
                "--output-dir",
                str(tmp_path / "model"),
            ],
        )

        assert result.exit_code != 0
        assert "does not exist" in result.output or "not found" in result.output.lower()

    def test_train_invalid_task_type(
        self,
        runner: CliRunner,
        tmp_path: Path,
        sample_forced_choice_items: Path,
        sample_labels_file: Path,
    ) -> None:
        """Test error handling for invalid task type."""
        result = runner.invoke(
            train_model,
            [
                "--task-type",
                "invalid_type",
                "--items",
                str(sample_forced_choice_items),
                "--labels",
                str(sample_labels_file),
                "--model-name",
                "bert-base-uncased",
                "--output-dir",
                str(tmp_path / "model"),
            ],
        )

        assert result.exit_code != 0
        assert "invalid" in result.output.lower() or "choice" in result.output.lower()


class TestPredictCommand:
    """Tests for predict command."""

    @patch("bead.cli.models.config_class_for_task_type")
    @patch("bead.cli.models.model_class_for_task_type")
    def test_predict_basic(
        self,
        mock_model_factory: MagicMock,
        mock_config_factory: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        sample_forced_choice_items: Path,
        sample_participant_ids: Path,
    ) -> None:
        """Test basic prediction."""
        # Create mock model directory with config
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        config_file = model_dir / "config.json"
        config_file.write_text(
            json.dumps(
                {
                    "task_type": "forced_choice",
                    "model_name": "bert-base-uncased",
                    "mixed_effects_mode": "fixed",
                }
            )
        )
        # Create empty model.pt file (will be loaded by mock)
        (model_dir / "model.pt").write_bytes(b"")

        # Mock model class
        mock_model_class = MagicMock()
        mock_model_instance = MagicMock()
        # Return mock predictions with model_dump_json method
        mock_predictions = []
        for i in range(20):
            mock_pred = MagicMock()
            mock_pred.model_dump_json.return_value = json.dumps({"label": i % 2})
            mock_pred.predicted_label = i % 2
            mock_pred.confidence = 0.95
            mock_predictions.append(mock_pred)
        mock_model_instance.predict.return_value = mock_predictions
        mock_model_class.return_value = mock_model_instance

        # Mock config class (not needed for predict, but imported)
        mock_config_class = MagicMock()
        mock_model_factory.return_value = mock_model_class
        mock_config_factory.return_value = mock_config_class

        output_file = tmp_path / "predictions.jsonl"

        result = runner.invoke(
            predict,
            [
                "--model-dir",
                str(model_dir),
                "--items",
                str(sample_forced_choice_items),
                "--participant-ids",
                str(sample_participant_ids),
                "--output",
                str(output_file),
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert "Making Predictions" in result.output
        assert "Saved" in result.output and "predictions" in result.output

        # Verify predictions file was created
        assert output_file.exists()

    @patch("bead.cli.models.config_class_for_task_type")
    @patch("bead.cli.models.model_class_for_task_type")
    def test_predict_missing_config(
        self,
        mock_model_factory: MagicMock,
        mock_config_factory: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        sample_forced_choice_items: Path,
        sample_participant_ids: Path,
    ) -> None:
        """Test error handling for missing config file."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        # No config.json created

        result = runner.invoke(
            predict,
            [
                "--model-dir",
                str(model_dir),
                "--items",
                str(sample_forced_choice_items),
                "--participant-ids",
                str(sample_participant_ids),
                "--output",
                str(tmp_path / "predictions.jsonl"),
            ],
        )

        assert result.exit_code != 0
        assert (
            "config.json" in result.output.lower()
            or "not found" in result.output.lower()
        )


class TestPredictProbaCommand:
    """Tests for predict-proba command."""

    @patch("bead.cli.models.config_class_for_task_type")
    @patch("bead.cli.models.model_class_for_task_type")
    def test_predict_proba_basic(
        self,
        mock_model_factory: MagicMock,
        mock_config_factory: MagicMock,
        runner: CliRunner,
        tmp_path: Path,
        sample_forced_choice_items: Path,
        sample_participant_ids: Path,
    ) -> None:
        """Test basic probability prediction."""
        # Create mock model directory with config
        model_dir = tmp_path / "model"
        model_dir.mkdir()
        config_file = model_dir / "config.json"
        config_file.write_text(
            json.dumps(
                {
                    "task_type": "forced_choice",
                    "model_name": "bert-base-uncased",
                    "mixed_effects_mode": "fixed",
                }
            )
        )
        # Create empty model.pt file (will be loaded by mock)
        (model_dir / "model.pt").write_bytes(b"")

        # Mock model class
        mock_model_class = MagicMock()
        mock_model_instance = MagicMock()
        # Return mock numpy array with tolist method for binary classification
        mock_probs = np.array([[0.7, 0.3]] * 20)
        mock_model_instance.predict_proba.return_value = mock_probs
        mock_model_class.return_value = mock_model_instance

        mock_config_class = MagicMock()
        mock_model_factory.return_value = mock_model_class
        mock_config_factory.return_value = mock_config_class

        output_file = tmp_path / "probabilities.json"

        result = runner.invoke(
            predict_proba,
            [
                "--model-dir",
                str(model_dir),
                "--items",
                str(sample_forced_choice_items),
                "--participant-ids",
                str(sample_participant_ids),
                "--output",
                str(output_file),
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert "Predicting Probabilities" in result.output
        assert "Saved" in result.output and "probability" in result.output

        # Verify output file was created
        assert output_file.exists()

        # Verify JSON format
        with open(output_file) as f:
            data = json.load(f)
            assert isinstance(data, list)


class TestModelsHelp:
    """Tests for models command help."""

    def test_models_help(self, runner: CliRunner) -> None:
        """Test models --help command."""
        result = runner.invoke(models, ["--help"])

        assert result.exit_code == 0
        # Check for command names
        assert "train-model" in result.output
        assert "predict" in result.output
        assert "predict-proba" in result.output

    def test_train_model_help(self, runner: CliRunner) -> None:
        """Test train-model --help command."""
        result = runner.invoke(train_model, ["--help"])

        assert result.exit_code == 0
        # Check for key parameters
        assert "--task-type" in result.output
        assert "--mixed-effects-mode" in result.output
        assert "--use-lora" in result.output or "--lora" in result.output.lower()


class TestAllTaskTypes:
    """Test training for all 8 task types."""

    @pytest.mark.parametrize(
        "task_type",
        [
            "forced_choice",
            "categorical",
            "binary",
            "multi_select",
            "ordinal_scale",
            "magnitude",
            "free_text",
            "cloze",
        ],
    )
    @patch("bead.cli.models.config_class_for_task_type")
    @patch("bead.cli.models.model_class_for_task_type")
    def test_train_all_task_types(
        self,
        mock_model_factory: MagicMock,
        mock_config_factory: MagicMock,
        task_type: str,
        runner: CliRunner,
        tmp_path: Path,
        sample_forced_choice_items: Path,
        sample_labels_file: Path,
    ) -> None:
        """Test training models for all 8 task types."""
        mock_model_class = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_instance.train.return_value = {}
        mock_model_class.return_value = mock_model_instance

        mock_config_class = MagicMock()
        mock_config_instance = MagicMock()
        mock_config_instance.model_dump.return_value = {
            "model_name": "bert-base-uncased",
            "mixed_effects": {"mode": "fixed"},
        }
        mock_config_class.return_value = mock_config_instance

        mock_model_factory.return_value = mock_model_class
        mock_config_factory.return_value = mock_config_class

        output_dir = tmp_path / f"model_{task_type}"

        result = runner.invoke(
            train_model,
            [
                "--task-type",
                task_type,
                "--items",
                str(sample_forced_choice_items),
                "--labels",
                str(sample_labels_file),
                "--model-name",
                "bert-base-uncased",
                "--output-dir",
                str(output_dir),
            ],
        )

        assert result.exit_code == 0, (
            f"Failed for task_type={task_type}: {result.output}"
        )
        assert f"Training {task_type} model" in result.output


class TestAllMixedEffectsModes:
    """Test all 3 mixed-effects modes."""

    @pytest.mark.parametrize(
        "mode",
        ["fixed", "random_intercepts", "random_slopes"],
    )
    @patch("bead.cli.models.config_class_for_task_type")
    @patch("bead.cli.models.model_class_for_task_type")
    def test_train_all_modes(
        self,
        mock_model_factory: MagicMock,
        mock_config_factory: MagicMock,
        mode: str,
        runner: CliRunner,
        tmp_path: Path,
        sample_forced_choice_items: Path,
        sample_labels_file: Path,
        sample_participant_ids: Path,
    ) -> None:
        """Test training with all 3 mixed-effects modes."""
        mock_model_class = MagicMock()
        mock_model_instance = MagicMock()
        mock_model_instance.train.return_value = {}
        mock_model_class.return_value = mock_model_instance

        mock_config_class = MagicMock()
        mock_config_instance = MagicMock()
        mock_config_instance.model_dump.return_value = {
            "model_name": "bert-base-uncased",
            "mixed_effects": {"mode": mode},
        }
        mock_config_class.return_value = mock_config_instance

        mock_model_factory.return_value = mock_model_class
        mock_config_factory.return_value = mock_config_class

        output_dir = tmp_path / f"model_{mode}"

        # Build args - add participant-ids for random effects modes
        args = [
            "--task-type",
            "forced_choice",
            "--items",
            str(sample_forced_choice_items),
            "--labels",
            str(sample_labels_file),
            "--model-name",
            "bert-base-uncased",
            "--mixed-effects-mode",
            mode,
            "--output-dir",
            str(output_dir),
        ]

        # Random effects modes require participant IDs
        if mode in ("random_intercepts", "random_slopes"):
            args.extend(["--participant-ids", str(sample_participant_ids)])

        result = runner.invoke(train_model, args)

        assert result.exit_code == 0, f"Failed for mode={mode}: {result.output}"
