"""Tests for training CLI commands."""

from __future__ import annotations

import json
from pathlib import Path
from unittest.mock import MagicMock, patch
from uuid import uuid4

from click.testing import CliRunner

from bead.cli.training import training
from bead.items.item import Item


def test_show_data_stats_empty_file(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test showing stats for empty file."""
    data_file = tmp_path / "empty.jsonl"
    data_file.write_text("")

    result = cli_runner.invoke(
        training,
        ["show-data-stats", str(data_file)],
    )

    assert result.exit_code == 1
    assert "No data found" in result.output


def test_show_data_stats_valid(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test showing statistics for valid data."""
    data_file = tmp_path / "results.jsonl"

    # Create mock results
    results = [
        {"worker_id": "w1", "data": {"response": "1"}},
        {"worker_id": "w2", "data": {"response": "2"}},
        {"worker_id": "w1", "data": {"response": "3"}},
    ]

    with open(data_file, "w") as f:
        for result in results:
            f.write(json.dumps(result) + "\n")

    result = cli_runner.invoke(
        training,
        ["show-data-stats", str(data_file)],
    )

    assert result.exit_code == 0
    assert "Total Results" in result.output
    assert "3" in result.output
    assert "Unique Workers" in result.output
    assert "2" in result.output


def test_show_data_stats_invalid_json(cli_runner: CliRunner, tmp_path: Path) -> None:
    """Test showing stats for file with invalid JSON."""
    data_file = tmp_path / "invalid.jsonl"
    data_file.write_text("not valid json\n")

    result = cli_runner.invoke(
        training,
        ["show-data-stats", str(data_file)],
    )

    assert result.exit_code == 1
    assert "Invalid JSON" in result.output


def test_training_help(cli_runner: CliRunner) -> None:
    """Test training command help."""
    result = cli_runner.invoke(training, ["--help"])

    assert result.exit_code == 0
    assert "Training commands" in result.output


def test_collect_data_help(cli_runner: CliRunner) -> None:
    """Test collect-data command help."""
    result = cli_runner.invoke(training, ["collect-data", "--help"])

    assert result.exit_code == 0
    assert "Collect judgment data" in result.output


def test_show_data_stats_help(cli_runner: CliRunner) -> None:
    """Test show-data-stats command help."""
    result = cli_runner.invoke(training, ["show-data-stats", "--help"])

    assert result.exit_code == 0
    assert "Show statistics" in result.output


# ============================================================================
# Phase 5.3: Evaluation Commands Tests
# ============================================================================


class TestEvaluateCommand:
    """Tests for evaluate command."""

    def test_evaluate_basic(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test basic model evaluation."""
        # Create mock model directory with config
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        config_file = model_dir / "config.json"
        config_file.write_text(json.dumps({"task_type": "forced_choice"}))

        # Create model weights file
        (model_dir / "model.pth").write_text("fake weights")

        # Create test items
        items_file = tmp_path / "items.jsonl"
        template_id = uuid4()
        test_items = [
            Item(
                id=uuid4(),
                item_template_id=template_id,
                rendered_elements={"option_a": "A", "option_b": "B"},
                item_metadata={"n_options": 2},
            )
            for _ in range(10)
        ]
        with open(items_file, "w") as f:
            for item in test_items:
                f.write(item.model_dump_json() + "\n")

        # Create test labels
        labels_file = tmp_path / "labels.jsonl"
        with open(labels_file, "w") as f:
            for _ in range(10):
                f.write("0\n")

        # Mock model loading and prediction
        with (
            patch("bead.cli.training.model_class_for_task_type") as mock_import,
            patch("bead.cli.training.config_class_for_task_type") as _,
        ):
            mock_model_class = MagicMock()
            mock_model = MagicMock()
            mock_model.predict.return_value = [0] * 10  # Perfect predictions
            mock_model_class.load.return_value = mock_model
            mock_import.return_value = mock_model_class

            result = cli_runner.invoke(
                training,
                [
                    "evaluate",
                    "--model-dir",
                    str(model_dir),
                    "--test-items",
                    str(items_file),
                    "--test-labels",
                    str(labels_file),
                ],
            )

        assert result.exit_code == 0
        assert "Evaluating model" in result.output
        assert "accuracy" in result.output.lower()

    def test_evaluate_with_output(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test evaluate with JSON output."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        config_file = model_dir / "config.json"
        config_file.write_text(json.dumps({"task_type": "binary"}))

        (model_dir / "model.pth").write_text("fake weights")

        items_file = tmp_path / "items.jsonl"
        template_id = uuid4()
        test_items = [
            Item(
                id=uuid4(),
                item_template_id=template_id,
                rendered_elements={"text": "test"},
                item_metadata={},
            )
            for _ in range(5)
        ]
        with open(items_file, "w") as f:
            for item in test_items:
                f.write(item.model_dump_json() + "\n")

        labels_file = tmp_path / "labels.jsonl"
        with open(labels_file, "w") as f:
            for _ in range(5):
                f.write("1\n")

        output_file = tmp_path / "results.json"

        with (
            patch("bead.cli.training.model_class_for_task_type") as mock_import,
            patch("bead.cli.training.config_class_for_task_type") as _,
        ):
            mock_model_class = MagicMock()
            mock_model = MagicMock()
            mock_model.predict.return_value = [1, 1, 1, 0, 1]
            mock_model_class.load.return_value = mock_model
            mock_import.return_value = mock_model_class

            result = cli_runner.invoke(
                training,
                [
                    "evaluate",
                    "--model-dir",
                    str(model_dir),
                    "--test-items",
                    str(items_file),
                    "--test-labels",
                    str(labels_file),
                    "--output",
                    str(output_file),
                ],
            )

        assert result.exit_code == 0
        assert output_file.exists()

        with open(output_file) as f:
            results = json.load(f)
        assert "metrics" in results
        assert "accuracy" in results["metrics"]

    def test_evaluate_missing_config(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test evaluate with missing model config."""
        model_dir = tmp_path / "model"
        model_dir.mkdir()

        items_file = tmp_path / "items.jsonl"
        items_file.write_text("")

        labels_file = tmp_path / "labels.jsonl"
        labels_file.write_text("")

        result = cli_runner.invoke(
            training,
            [
                "evaluate",
                "--model-dir",
                str(model_dir),
                "--test-items",
                str(items_file),
                "--test-labels",
                str(labels_file),
            ],
        )

        assert result.exit_code == 1
        assert "config not found" in result.output.lower()

    def test_evaluate_help(self, cli_runner: CliRunner) -> None:
        """Test evaluate command help."""
        result = cli_runner.invoke(training, ["evaluate", "--help"])

        assert result.exit_code == 0
        assert "Evaluate trained model" in result.output


class TestCrossValidateCommand:
    """Tests for cross-validate command."""

    def test_cross_validate_basic(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test basic cross-validation."""
        # Create items
        items_file = tmp_path / "items.jsonl"
        template_id = uuid4()
        test_items = [
            Item(
                id=uuid4(),
                item_template_id=template_id,
                rendered_elements={"text": f"Item {i}"},
                item_metadata={"scale_min": 1, "scale_max": 7},
            )
            for i in range(20)
        ]
        with open(items_file, "w") as f:
            for item in test_items:
                f.write(item.model_dump_json() + "\n")

        # Create labels
        labels_file = tmp_path / "labels.jsonl"
        with open(labels_file, "w") as f:
            for i in range(20):
                f.write(f"{(i % 7) + 1}\n")

        # Create model config
        config_file = tmp_path / "config.json"
        config_file.write_text(
            json.dumps(
                {
                    "task_type": "ordinal_scale",
                    "model_name": "bert-base-uncased",
                }
            )
        )

        # Mock model training and prediction
        with (
            patch("bead.cli.training.model_class_for_task_type") as mock_import,
            patch("bead.cli.training.config_class_for_task_type") as _,
        ):
            mock_model_class = MagicMock()
            mock_model = MagicMock()

            # Return prediction objects with predicted_class attribute
            def make_predictions(items, **kwargs):
                preds = []
                for _ in range(len(items)):
                    mock_pred = MagicMock()
                    mock_pred.predicted_class = 1
                    preds.append(mock_pred)
                return preds

            mock_model.predict.side_effect = make_predictions
            mock_model_class.return_value = mock_model
            mock_import.return_value = mock_model_class

            result = cli_runner.invoke(
                training,
                [
                    "cross-validate",
                    "--items",
                    str(items_file),
                    "--labels",
                    str(labels_file),
                    "--model-config",
                    str(config_file),
                    "--k-folds",
                    "3",
                ],
            )

        assert result.exit_code == 0
        assert "cross-validation" in result.output.lower()
        assert "Fold" in result.output

    def test_cross_validate_with_stratification(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test cross-validation with stratification."""
        items_file = tmp_path / "items.jsonl"
        template_id = uuid4()
        test_items = [
            Item(
                id=uuid4(),
                item_template_id=template_id,
                rendered_elements={"premise": "P", "hypothesis": "H"},
                item_metadata={"categories": ["A", "B", "C"]},
            )
            for _ in range(15)
        ]
        with open(items_file, "w") as f:
            for item in test_items:
                f.write(item.model_dump_json() + "\n")

        labels_file = tmp_path / "labels.jsonl"
        with open(labels_file, "w") as f:
            for i in range(15):
                f.write(f'"{chr(65 + (i % 3))}"\n')  # "A", "B", "C"

        config_file = tmp_path / "config.json"
        config_file.write_text(
            json.dumps({"task_type": "categorical", "model_name": "bert-base-uncased"})
        )

        output_file = tmp_path / "cv_results.json"

        with (
            patch("bead.cli.training.model_class_for_task_type") as mock_import,
            patch("bead.cli.training.config_class_for_task_type") as _,
        ):
            mock_model_class = MagicMock()
            mock_model = MagicMock()

            # Return prediction objects with predicted_class attribute
            def make_predictions(items, **kwargs):
                preds = []
                for _ in range(len(items)):
                    mock_pred = MagicMock()
                    mock_pred.predicted_class = "A"
                    preds.append(mock_pred)
                return preds

            mock_model.predict.side_effect = make_predictions
            mock_model_class.return_value = mock_model
            mock_import.return_value = mock_model_class

            result = cli_runner.invoke(
                training,
                [
                    "cross-validate",
                    "--items",
                    str(items_file),
                    "--labels",
                    str(labels_file),
                    "--model-config",
                    str(config_file),
                    "--k-folds",
                    "3",
                    "--stratify-by",
                    "label",
                    "--output",
                    str(output_file),
                ],
            )

        assert result.exit_code == 0
        assert output_file.exists()

    def test_cross_validate_help(self, cli_runner: CliRunner) -> None:
        """Test cross-validate command help."""
        result = cli_runner.invoke(training, ["cross-validate", "--help"])

        assert result.exit_code == 0
        assert "K-fold cross-validation" in result.output


class TestLearningCurveCommand:
    """Tests for learning-curve command."""

    def test_learning_curve_basic(self, cli_runner: CliRunner, tmp_path: Path) -> None:
        """Test basic learning curve generation."""
        items_file = tmp_path / "items.jsonl"
        template_id = uuid4()
        test_items = [
            Item(
                id=uuid4(),
                item_template_id=template_id,
                rendered_elements={"option_a": "A", "option_b": "B"},
                item_metadata={"n_options": 2},
            )
            for _ in range(30)
        ]
        with open(items_file, "w") as f:
            for item in test_items:
                f.write(item.model_dump_json() + "\n")

        labels_file = tmp_path / "labels.jsonl"
        with open(labels_file, "w") as f:
            for i in range(30):
                f.write(f"{i % 2}\n")

        config_file = tmp_path / "config.json"
        config_file.write_text(
            json.dumps(
                {"task_type": "forced_choice", "model_name": "bert-base-uncased"}
            )
        )

        with (
            patch("bead.cli.training.model_class_for_task_type") as mock_import,
            patch("bead.cli.training.config_class_for_task_type") as _,
        ):
            mock_model_class = MagicMock()
            mock_model = MagicMock()
            mock_model.predict.side_effect = lambda items, **kwargs: [0] * len(items)
            mock_model_class.return_value = mock_model
            mock_import.return_value = mock_model_class

            result = cli_runner.invoke(
                training,
                [
                    "learning-curve",
                    "--items",
                    str(items_file),
                    "--labels",
                    str(labels_file),
                    "--model-config",
                    str(config_file),
                    "--train-sizes",
                    "0.2,0.5,1.0",
                ],
            )

        assert result.exit_code == 0
        assert "learning curve" in result.output.lower()
        assert "Train Size" in result.output

    def test_learning_curve_with_output(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test learning curve with JSON output."""
        items_file = tmp_path / "items.jsonl"
        template_id = uuid4()
        test_items = [
            Item(
                id=uuid4(),
                item_template_id=template_id,
                rendered_elements={"text": "test"},
                item_metadata={},
            )
            for _ in range(20)
        ]
        with open(items_file, "w") as f:
            for item in test_items:
                f.write(item.model_dump_json() + "\n")

        labels_file = tmp_path / "labels.jsonl"
        with open(labels_file, "w") as f:
            for i in range(20):
                f.write(f"{i % 2}\n")

        config_file = tmp_path / "config.json"
        config_file.write_text(
            json.dumps({"task_type": "binary", "model_name": "bert-base-uncased"})
        )

        output_file = tmp_path / "learning_curve.json"

        with (
            patch("bead.cli.training.model_class_for_task_type") as mock_import,
            patch("bead.cli.training.config_class_for_task_type") as _,
        ):
            mock_model_class = MagicMock()
            mock_model = MagicMock()
            mock_model.predict.side_effect = lambda items, **kwargs: [0] * len(items)
            mock_model_class.return_value = mock_model
            mock_import.return_value = mock_model_class

            result = cli_runner.invoke(
                training,
                [
                    "learning-curve",
                    "--items",
                    str(items_file),
                    "--labels",
                    str(labels_file),
                    "--model-config",
                    str(config_file),
                    "--train-sizes",
                    "0.5,1.0",
                    "--output",
                    str(output_file),
                ],
            )

        assert result.exit_code == 0
        assert output_file.exists()

    def test_learning_curve_help(self, cli_runner: CliRunner) -> None:
        """Test learning-curve command help."""
        result = cli_runner.invoke(training, ["learning-curve", "--help"])

        assert result.exit_code == 0
        assert "Generate learning curve" in result.output


class TestComputeAgreementCommand:
    """Tests for compute-agreement command."""

    def test_compute_agreement_krippendorff(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test agreement computation with Krippendorff's alpha."""
        annotations_file = tmp_path / "annotations.jsonl"

        # Create annotations from 3 raters
        annotations = []
        for rater_id in ["rater1", "rater2", "rater3"]:
            for i in range(10):
                annotations.append({"rater_id": rater_id, "label": i % 3})

        with open(annotations_file, "w") as f:
            for annotation in annotations:
                f.write(json.dumps(annotation) + "\n")

        result = cli_runner.invoke(
            training,
            [
                "compute-agreement",
                "--annotations",
                str(annotations_file),
                "--metric",
                "krippendorff_alpha",
            ],
        )

        assert result.exit_code == 0
        assert "Krippendorff" in result.output
        assert "Inter-Annotator Agreement" in result.output

    def test_compute_agreement_fleiss_kappa(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test agreement computation with Fleiss' kappa."""
        annotations_file = tmp_path / "annotations.jsonl"

        annotations = []
        for rater_id in ["r1", "r2", "r3", "r4"]:
            for i in range(15):
                annotations.append(
                    {"rater_id": rater_id, "label": (i + hash(rater_id)) % 5}
                )

        with open(annotations_file, "w") as f:
            for annotation in annotations:
                f.write(json.dumps(annotation) + "\n")

        result = cli_runner.invoke(
            training,
            [
                "compute-agreement",
                "--annotations",
                str(annotations_file),
                "--metric",
                "fleiss_kappa",
            ],
        )

        assert result.exit_code == 0
        assert "Fleiss" in result.output

    def test_compute_agreement_cohens_kappa(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test agreement computation with Cohen's kappa."""
        annotations_file = tmp_path / "annotations.jsonl"

        # Cohen's kappa requires exactly 2 raters
        annotations = []
        for rater_id in ["rater1", "rater2"]:
            for i in range(20):
                annotations.append({"rater_id": rater_id, "label": i % 4})

        with open(annotations_file, "w") as f:
            for annotation in annotations:
                f.write(json.dumps(annotation) + "\n")

        result = cli_runner.invoke(
            training,
            [
                "compute-agreement",
                "--annotations",
                str(annotations_file),
                "--metric",
                "cohens_kappa",
            ],
        )

        assert result.exit_code == 0
        assert "Cohen" in result.output

    def test_compute_agreement_with_output(
        self, cli_runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test agreement computation with JSON output."""
        annotations_file = tmp_path / "annotations.jsonl"

        annotations = []
        for rater_id in ["r1", "r2"]:  # percentage_agreement requires exactly 2 raters
            for i in range(10):
                annotations.append({"rater_id": rater_id, "label": i % 2})

        with open(annotations_file, "w") as f:
            for annotation in annotations:
                f.write(json.dumps(annotation) + "\n")

        output_file = tmp_path / "agreement.json"

        result = cli_runner.invoke(
            training,
            [
                "compute-agreement",
                "--annotations",
                str(annotations_file),
                "--metric",
                "percentage_agreement",
                "--output",
                str(output_file),
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

        with open(output_file) as f:
            results = json.load(f)
        assert "metric" in results
        assert "score" in results

    def test_compute_agreement_help(self, cli_runner: CliRunner) -> None:
        """Test compute-agreement command help."""
        result = cli_runner.invoke(training, ["compute-agreement", "--help"])

        assert result.exit_code == 0
        assert "inter-annotator agreement" in result.output.lower()
