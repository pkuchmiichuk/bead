"""Integration tests for bead.cli.deployment CLI commands.

Tests all deployment commands to ensure they:
1. Generate valid jsPsych experiments
2. Handle distribution strategies correctly
3. Create proper output directory structure
4. Integrate correctly with core bead.deployment utilities
"""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from click.testing import CliRunner

from bead.cli.deployment import deployment, export_jatos, generate, validate
from bead.cli.deployment_trials import (
    configure_choice,
    configure_rating,
    configure_timing,
    show_config,
)
from bead.cli.deployment_ui import customize, generate_css
from bead.items.item import Item
from bead.items.item_template import (
    ItemTemplate,
    PresentationSpec,
    ScaleBounds,
    ScalePointLabel,
    TaskSpec,
)
from bead.lists import ExperimentList


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
        judgment_type="acceptability",
        task_type="ordinal_scale",
        task_spec=TaskSpec(
            prompt="How natural is this sentence?",
            scale_bounds=ScaleBounds(min=1, max=7),
            scale_labels=(
                ScalePointLabel(point=1, label="Very unnatural"),
                ScalePointLabel(point=7, label="Very natural"),
            ),
        ),
        presentation_spec=PresentationSpec(mode="static"),
    )


@pytest.fixture
def sample_items_file(tmp_path: Path, sample_template: ItemTemplate) -> Path:
    """Create a sample items JSONL file."""
    items_file = tmp_path / "items.jsonl"

    # Create 10 test items
    items = []
    for i in range(10):
        item = Item(
            item_template_id=sample_template.id,
            rendered_elements={"text": f"This is test sentence number {i + 1}."},
            item_metadata={
                "condition": "A" if i % 2 == 0 else "B",
                "item_number": i,
                "scale_min": 1,
                "scale_max": 7,
            },
        )
        items.append(item.model_dump_json() + "\n")

    items_file.write_text("".join(items))
    return items_file


@pytest.fixture
def sample_lists_file(tmp_path: Path, sample_items_file: Path) -> Path:
    """Create a sample lists JSONL file (one list per line)."""
    lists_file = tmp_path / "lists.jsonl"

    # Read items to get their IDs
    items_data = [
        Item.model_validate_json(line)
        for line in sample_items_file.read_text().strip().split("\n")
    ]
    item_ids = [item.id for item in items_data]

    # Create 3 lists and write them all to one file
    lines = []
    for list_num in range(3):
        exp_list = ExperimentList(
            name=f"list_{list_num}",
            list_number=list_num,
        )

        # Add subset of items to each list (different items per list)
        start_idx = list_num * 3
        end_idx = start_idx + 4
        for item_id in item_ids[start_idx:end_idx]:
            exp_list = exp_list.with_item(item_id)
        lines.append(exp_list.model_dump_json())

    lists_file.write_text("\n".join(lines) + "\n")
    return lists_file


# ==================== Generate Command Tests ====================


class TestGenerateCommand:
    """Test deployment generate command."""

    def test_generate_with_balanced_strategy(
        self,
        runner: CliRunner,
        sample_lists_file: Path,
        sample_items_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test generate command with balanced distribution strategy."""
        output_dir = tmp_path / "experiment"

        result = runner.invoke(
            generate,
            [
                str(sample_lists_file),
                str(sample_items_file),
                str(output_dir),
                "--experiment-type",
                "likert_rating",
                "--title",
                "Test Experiment",
                "--distribution-strategy",
                "balanced",
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"

        # Verify directory structure
        assert output_dir.exists()
        assert (output_dir / "index.html").exists()
        assert (output_dir / "css" / "experiment.css").exists()
        assert (output_dir / "js" / "experiment.js").exists()
        assert (output_dir / "js" / "list_distributor.js").exists()
        assert (output_dir / "data" / "config.json").exists()
        assert (output_dir / "data" / "lists.jsonl").exists()
        assert (output_dir / "data" / "items.jsonl").exists()
        assert (output_dir / "data" / "distribution.json").exists()

    def test_generate_with_sequential_strategy(
        self,
        runner: CliRunner,
        sample_lists_file: Path,
        sample_items_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test generate command with sequential distribution strategy."""
        output_dir = tmp_path / "experiment"

        result = runner.invoke(
            generate,
            [
                str(sample_lists_file),
                str(sample_items_file),
                str(output_dir),
                "--experiment-type",
                "likert_rating",
                "--distribution-strategy",
                "sequential",
            ],
        )

        assert result.exit_code == 0

        # Verify distribution config
        dist_config = json.loads(
            (output_dir / "data" / "distribution.json").read_text()
        )
        assert dist_config["strategy_type"] == "sequential"

    def test_generate_with_quota_strategy(
        self,
        runner: CliRunner,
        sample_lists_file: Path,
        sample_items_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test generate command with quota-based distribution strategy."""
        output_dir = tmp_path / "experiment"

        config_json = json.dumps({"participants_per_list": 10, "allow_overflow": False})

        result = runner.invoke(
            generate,
            [
                str(sample_lists_file),
                str(sample_items_file),
                str(output_dir),
                "--experiment-type",
                "likert_rating",
                "--distribution-strategy",
                "quota_based",
                "--distribution-config",
                config_json,
            ],
        )

        assert result.exit_code == 0

        # Verify distribution config
        dist_config = json.loads(
            (output_dir / "data" / "distribution.json").read_text()
        )
        assert dist_config["strategy_type"] == "quota_based"
        assert dist_config["strategy_config"]["participants_per_list"] == 10

    def test_generate_with_debug_mode(
        self,
        runner: CliRunner,
        sample_lists_file: Path,
        sample_items_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test generate command with debug mode enabled."""
        output_dir = tmp_path / "experiment"

        result = runner.invoke(
            generate,
            [
                str(sample_lists_file),
                str(sample_items_file),
                str(output_dir),
                "--experiment-type",
                "likert_rating",
                "--distribution-strategy",
                "balanced",
                "--debug-mode",
                "--debug-list-index",
                "1",
            ],
        )

        assert result.exit_code == 0

        # Verify debug mode config
        dist_config = json.loads(
            (output_dir / "data" / "distribution.json").read_text()
        )
        assert dist_config["debug_mode"] is True
        assert dist_config["debug_list_index"] == 1

    def test_generate_missing_distribution_strategy(
        self,
        runner: CliRunner,
        sample_lists_file: Path,
        sample_items_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test error when distribution strategy is missing."""
        output_dir = tmp_path / "experiment"

        result = runner.invoke(
            generate,
            [
                str(sample_lists_file),
                str(sample_items_file),
                str(output_dir),
                "--experiment-type",
                "likert_rating",
                # Missing --distribution-strategy (required)
            ],
        )

        # Should fail due to missing required option
        assert result.exit_code != 0

    def test_generate_invalid_distribution_config(
        self,
        runner: CliRunner,
        sample_lists_file: Path,
        sample_items_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test error when distribution config JSON is invalid."""
        output_dir = tmp_path / "experiment"

        result = runner.invoke(
            generate,
            [
                str(sample_lists_file),
                str(sample_items_file),
                str(output_dir),
                "--experiment-type",
                "likert_rating",
                "--distribution-strategy",
                "quota_based",
                "--distribution-config",
                "not-valid-json",
            ],
        )

        # Should fail due to invalid JSON
        assert result.exit_code != 0
        assert "Invalid JSON" in result.output


# ==================== Validate Command Tests ====================


class TestValidateCommand:
    """Test deployment validate command."""

    def test_validate_valid_experiment(
        self,
        runner: CliRunner,
        sample_lists_file: Path,
        sample_items_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test validate command on valid experiment directory."""
        output_dir = tmp_path / "experiment"

        # First generate an experiment
        runner.invoke(
            generate,
            [
                str(sample_lists_file),
                str(sample_items_file),
                str(output_dir),
                "--experiment-type",
                "likert_rating",
                "--distribution-strategy",
                "balanced",
            ],
        )

        # Then validate it
        result = runner.invoke(
            validate,
            [str(output_dir)],
        )

        assert result.exit_code == 0, f"Validation failed: {result.output}"

    def test_validate_missing_directory(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test validate command on non-existent directory."""
        nonexistent_dir = tmp_path / "does_not_exist"

        result = runner.invoke(
            validate,
            [str(nonexistent_dir)],
        )

        # Should fail due to missing directory
        assert result.exit_code != 0

    def test_validate_incomplete_structure(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test validate command on incomplete experiment directory."""
        exp_dir = tmp_path / "experiment"
        exp_dir.mkdir()

        # Create only partial structure
        (exp_dir / "index.html").write_text("<html></html>")

        result = runner.invoke(
            validate,
            [str(exp_dir)],
        )

        # Should fail due to missing required files
        assert result.exit_code != 0


# ==================== Export JATOS Command Tests ====================


class TestExportJATOSCommand:
    """Test deployment export-jatos command."""

    def test_export_jatos_basic(
        self,
        runner: CliRunner,
        sample_lists_file: Path,
        sample_items_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test export-jatos command with basic options."""
        output_dir = tmp_path / "experiment"
        jzip_path = tmp_path / "study.jzip"

        # First generate an experiment
        runner.invoke(
            generate,
            [
                str(sample_lists_file),
                str(sample_items_file),
                str(output_dir),
                "--experiment-type",
                "likert_rating",
                "--distribution-strategy",
                "balanced",
            ],
        )

        # Then export to JATOS
        result = runner.invoke(
            export_jatos,
            [
                str(output_dir),
                str(jzip_path),
                "--title",
                "Test Study",
                "--description",
                "Test Description",
            ],
        )

        assert result.exit_code == 0
        assert jzip_path.exists()
        assert jzip_path.suffix == ".jzip"


# ==================== Distribution Strategy Integration Tests ====================


class TestDistributionStrategies:
    """Test all 8 distribution strategies."""

    @pytest.mark.parametrize(
        "strategy",
        [
            "random",
            "sequential",
            "balanced",
            "latin_square",
        ],
    )
    def test_simple_strategies(
        self,
        runner: CliRunner,
        sample_lists_file: Path,
        sample_items_file: Path,
        tmp_path: Path,
        strategy: str,
    ) -> None:
        """Test simple distribution strategies (no config required)."""
        output_dir = tmp_path / f"experiment_{strategy}"

        result = runner.invoke(
            generate,
            [
                str(sample_lists_file),
                str(sample_items_file),
                str(output_dir),
                "--experiment-type",
                "likert_rating",
                "--distribution-strategy",
                strategy,
            ],
        )

        assert result.exit_code == 0, f"Strategy {strategy} failed: {result.output}"

        # Verify distribution config
        dist_config = json.loads(
            (output_dir / "data" / "distribution.json").read_text()
        )
        assert dist_config["strategy_type"] == strategy

    def test_stratified_strategy(
        self,
        runner: CliRunner,
        sample_lists_file: Path,
        sample_items_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test stratified distribution strategy with factors."""
        output_dir = tmp_path / "experiment"

        config_json = json.dumps({"factors": ["condition"]})

        result = runner.invoke(
            generate,
            [
                str(sample_lists_file),
                str(sample_items_file),
                str(output_dir),
                "--experiment-type",
                "likert_rating",
                "--distribution-strategy",
                "stratified",
                "--distribution-config",
                config_json,
            ],
        )

        assert result.exit_code == 0

        # Verify distribution config
        dist_config = json.loads(
            (output_dir / "data" / "distribution.json").read_text()
        )
        assert dist_config["strategy_type"] == "stratified"
        assert dist_config["strategy_config"]["factors"] == ["condition"]

    def test_weighted_random_strategy(
        self,
        runner: CliRunner,
        sample_lists_file: Path,
        sample_items_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test weighted_random distribution strategy."""
        output_dir = tmp_path / "experiment"

        config_json = json.dumps(
            {"weight_expression": "1.0", "normalize_weights": True}
        )

        result = runner.invoke(
            generate,
            [
                str(sample_lists_file),
                str(sample_items_file),
                str(output_dir),
                "--experiment-type",
                "likert_rating",
                "--distribution-strategy",
                "weighted_random",
                "--distribution-config",
                config_json,
            ],
        )

        assert result.exit_code == 0

    def test_metadata_based_strategy(
        self,
        runner: CliRunner,
        sample_lists_file: Path,
        sample_items_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test metadata_based distribution strategy."""
        output_dir = tmp_path / "experiment"

        config_json = json.dumps(
            {
                "filter_expression": "true",
                "rank_expression": "list_metadata.list_number || 0",
                "rank_ascending": True,
            }
        )

        result = runner.invoke(
            generate,
            [
                str(sample_lists_file),
                str(sample_items_file),
                str(output_dir),
                "--experiment-type",
                "likert_rating",
                "--distribution-strategy",
                "metadata_based",
                "--distribution-config",
                config_json,
            ],
        )

        assert result.exit_code == 0


# ==================== Help Command Tests ====================


class TestHelpCommands:
    """Test help output for deployment commands."""

    def test_deployment_help(self, runner: CliRunner) -> None:
        """Test deployment command help."""
        result = runner.invoke(deployment, ["--help"])

        assert result.exit_code == 0
        assert "Deployment commands" in result.output

    def test_generate_help(self, runner: CliRunner) -> None:
        """Test generate command help."""
        result = runner.invoke(deployment, ["generate", "--help"])

        assert result.exit_code == 0
        assert (
            "Generate jsPsych experiment" in result.output
            or "generate" in result.output.lower()
        )

    def test_export_jatos_help(self, runner: CliRunner) -> None:
        """Test export-jatos command help."""
        result = runner.invoke(deployment, ["export-jatos", "--help"])

        assert result.exit_code == 0
        assert "export" in result.output.lower() or "JATOS" in result.output


# ==================== Trials Configuration Tests ====================


class TestTrialsCommands:
    """Test deployment trials configuration commands."""

    def test_configure_rating_basic(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test configure-rating with basic options."""
        output_file = tmp_path / "rating_config.json"

        result = runner.invoke(
            configure_rating,
            [
                str(output_file),
                "--min-value",
                "1",
                "--max-value",
                "7",
                "--min-label",
                "Very unnatural",
                "--max-label",
                "Very natural",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

        # Verify config content
        config = json.loads(output_file.read_text())
        assert config["type"] == "rating_scale"
        assert config["min_value"] == 1
        assert config["max_value"] == 7
        assert config["step"] == 1
        assert config["min_label"] == "Very unnatural"
        assert config["max_label"] == "Very natural"
        assert config["show_numeric_labels"] is False
        assert config["required"] is True

    def test_configure_rating_with_numeric_labels(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test configure-rating with numeric labels enabled."""
        output_file = tmp_path / "rating_config.json"

        result = runner.invoke(
            configure_rating,
            [
                str(output_file),
                "--min-value",
                "1",
                "--max-value",
                "5",
                "--show-numeric-labels",
            ],
        )

        assert result.exit_code == 0

        config = json.loads(output_file.read_text())
        assert config["show_numeric_labels"] is True
        assert config["max_value"] == 5

    def test_configure_rating_invalid_range(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test configure-rating with invalid range (min >= max)."""
        output_file = tmp_path / "rating_config.json"

        result = runner.invoke(
            configure_rating,
            [
                str(output_file),
                "--min-value",
                "7",
                "--max-value",
                "1",
            ],
        )

        assert result.exit_code != 0

    def test_configure_rating_invalid_step(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test configure-rating with invalid step (not divisible)."""
        output_file = tmp_path / "rating_config.json"

        result = runner.invoke(
            configure_rating,
            [
                str(output_file),
                "--min-value",
                "1",
                "--max-value",
                "7",
                "--step",
                "2",  # 6 is not divisible by 2... wait, it is. Let me use 4
            ],
        )

        # This should actually succeed since (7-1) = 6, divisible by 2
        # Let me test with a step that doesn't divide evenly
        result = runner.invoke(
            configure_rating,
            [
                str(output_file),
                "--min-value",
                "1",
                "--max-value",
                "7",
                "--step",
                "4",  # (7-1) = 6, not divisible by 4
            ],
        )

        assert result.exit_code != 0

    def test_configure_choice_basic(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test configure-choice with basic options."""
        output_file = tmp_path / "choice_config.json"

        result = runner.invoke(
            configure_choice,
            [
                str(output_file),
                "--button-html",
                '<button class="custom-btn">%choice%</button>',
                "--randomize-position",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

        config = json.loads(output_file.read_text())
        assert config["type"] == "choice"
        assert config["button_html"] == '<button class="custom-btn">%choice%</button>'
        assert config["enable_keyboard"] is True
        assert config["randomize_position"] is True

    def test_configure_choice_missing_placeholder(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test configure-choice with missing %choice% placeholder."""
        output_file = tmp_path / "choice_config.json"

        result = runner.invoke(
            configure_choice,
            [
                str(output_file),
                "--button-html",
                '<button class="custom-btn">No placeholder</button>',
            ],
        )

        assert result.exit_code != 0
        assert "%choice%" in result.output

    def test_configure_timing_basic(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test configure-timing with basic options."""
        output_file = tmp_path / "timing_config.json"

        result = runner.invoke(
            configure_timing,
            [
                str(output_file),
                "--duration-ms",
                "500",
                "--isi-ms",
                "100",
                "--mask-char",
                "#",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

        config = json.loads(output_file.read_text())
        assert config["type"] == "timing"
        assert config["duration_ms"] == 500
        assert config["isi_ms"] == 100
        assert config["mask_char"] == "#"
        assert config["cumulative"] is False

    def test_configure_timing_cumulative(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test configure-timing with cumulative mode."""
        output_file = tmp_path / "timing_config.json"

        result = runner.invoke(
            configure_timing,
            [
                str(output_file),
                "--isi-ms",
                "50",
                "--cumulative",
            ],
        )

        assert result.exit_code == 0

        config = json.loads(output_file.read_text())
        assert config["cumulative"] is True
        assert "duration_ms" not in config  # Optional parameter

    def test_configure_timing_invalid_duration(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test configure-timing with invalid duration."""
        output_file = tmp_path / "timing_config.json"

        result = runner.invoke(
            configure_timing,
            [
                str(output_file),
                "--duration-ms",
                "-100",  # Negative duration
            ],
        )

        assert result.exit_code != 0

    def test_show_config_single_file(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test show-config with single configuration file."""
        # Create a config file first
        config_file = tmp_path / "test_config.json"
        config = {
            "type": "rating_scale",
            "min_value": 1,
            "max_value": 7,
        }
        config_file.write_text(json.dumps(config))

        result = runner.invoke(
            show_config,
            [str(config_file)],
        )

        assert result.exit_code == 0
        assert "test_config.json" in result.output
        assert "rating_scale" in result.output

    def test_show_config_multiple_files(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test show-config with multiple configuration files."""
        # Create two config files
        config1 = tmp_path / "rating.json"
        config1.write_text(json.dumps({"type": "rating_scale"}))

        config2 = tmp_path / "choice.json"
        config2.write_text(json.dumps({"type": "choice"}))

        result = runner.invoke(
            show_config,
            [str(config1), str(config2)],
        )

        assert result.exit_code == 0
        assert "rating.json" in result.output
        assert "choice.json" in result.output

    def test_show_config_invalid_json(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test show-config with invalid JSON file."""
        config_file = tmp_path / "invalid.json"
        config_file.write_text("not valid json")

        result = runner.invoke(
            show_config,
            [str(config_file)],
        )

        assert result.exit_code != 0
        assert "JSON" in result.output

    def test_trials_help(self, runner: CliRunner) -> None:
        """Test trials command group help."""
        result = runner.invoke(deployment, ["trials", "--help"])

        assert result.exit_code == 0
        assert "Trial configuration" in result.output
        assert "configure-rating" in result.output
        assert "configure-choice" in result.output
        assert "configure-timing" in result.output


# ==================== UI Customization Tests ====================


class TestUICommands:
    """Test deployment UI customization commands."""

    def test_generate_css_basic(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test generate-css with basic options."""
        output_file = tmp_path / "custom.css"

        result = runner.invoke(
            generate_css,
            [
                str(output_file),
                "--theme",
                "dark",
                "--primary-color",
                "#1976D2",
                "--secondary-color",
                "#FF5722",
            ],
        )

        assert result.exit_code == 0
        assert output_file.exists()

        css_content = output_file.read_text()
        assert "#1976D2" in css_content
        assert "#FF5722" in css_content
        assert "dark" in css_content.lower()

    def test_generate_css_light_theme(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test generate-css with light theme."""
        output_file = tmp_path / "light.css"

        result = runner.invoke(
            generate_css,
            [
                str(output_file),
                "--theme",
                "light",
            ],
        )

        assert result.exit_code == 0
        css_content = output_file.read_text()
        assert "light" in css_content.lower() or "background" in css_content.lower()

    def test_generate_css_invalid_color(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test generate-css with invalid hex color."""
        output_file = tmp_path / "custom.css"

        result = runner.invoke(
            generate_css,
            [
                str(output_file),
                "--primary-color",
                "not-a-hex-color",
            ],
        )

        assert result.exit_code != 0
        assert "Invalid" in result.output

    def test_customize_basic(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test customize command with basic options."""
        # Create mock experiment directory
        exp_dir = tmp_path / "experiment"
        exp_dir.mkdir()
        (exp_dir / "css").mkdir()
        index_html = exp_dir / "index.html"
        index_html.write_text("<html><head></head><body></body></html>")

        result = runner.invoke(
            customize,
            [
                str(exp_dir),
                "--theme",
                "dark",
                "--primary-color",
                "#1976D2",
            ],
        )

        assert result.exit_code == 0
        assert (exp_dir / "css" / "experiment.css").exists()

        # Check that index.html was updated
        html_content = index_html.read_text()
        assert "css/experiment.css" in html_content

    def test_customize_with_custom_css(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test customize command with custom CSS file."""
        # Create mock experiment directory
        exp_dir = tmp_path / "experiment"
        exp_dir.mkdir()
        (exp_dir / "css").mkdir()
        (exp_dir / "index.html").write_text("<html><head></head><body></body></html>")

        # Create custom CSS file
        custom_css = tmp_path / "custom.css"
        custom_css.write_text(".my-class { color: red; }")

        result = runner.invoke(
            customize,
            [
                str(exp_dir),
                "--css-file",
                str(custom_css),
                "--output-name",
                "styles.css",
            ],
        )

        assert result.exit_code == 0
        output_css = exp_dir / "css" / "styles.css"
        assert output_css.exists()

        css_content = output_css.read_text()
        assert ".my-class" in css_content
        assert "Custom CSS" in css_content

    def test_customize_invalid_color(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test customize command with invalid color."""
        exp_dir = tmp_path / "experiment"
        exp_dir.mkdir()

        result = runner.invoke(
            customize,
            [
                str(exp_dir),
                "--primary-color",
                "invalid-color",
            ],
        )

        assert result.exit_code != 0
        assert "Invalid" in result.output

    def test_ui_help(self, runner: CliRunner) -> None:
        """Test ui command group help."""
        result = runner.invoke(deployment, ["ui", "--help"])

        assert result.exit_code == 0
        assert "UI customization" in result.output
        assert "generate-css" in result.output
        assert "customize" in result.output


# ==================== Enhanced Validate Command Tests ====================


class TestEnhancedValidateCommand:
    """Test enhanced validate command with new flags."""

    def test_validate_check_distribution(
        self,
        runner: CliRunner,
        sample_lists_file: Path,
        sample_items_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test validate command with --check-distribution flag."""
        output_dir = tmp_path / "experiment"

        # Generate experiment first
        runner.invoke(
            generate,
            [
                str(sample_lists_file),
                str(sample_items_file),
                str(output_dir),
                "--experiment-type",
                "likert_rating",
                "--distribution-strategy",
                "balanced",
            ],
        )

        # Validate with distribution check
        result = runner.invoke(
            validate,
            [str(output_dir), "--check-distribution"],
        )

        assert result.exit_code == 0

    def test_validate_check_data_structure(
        self,
        runner: CliRunner,
        sample_lists_file: Path,
        sample_items_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test validate command with --check-data-structure flag."""
        output_dir = tmp_path / "experiment"

        # Generate experiment first
        runner.invoke(
            generate,
            [
                str(sample_lists_file),
                str(sample_items_file),
                str(output_dir),
                "--experiment-type",
                "likert_rating",
                "--distribution-strategy",
                "balanced",
            ],
        )

        # Validate with data structure check
        result = runner.invoke(
            validate,
            [str(output_dir), "--check-data-structure"],
        )

        assert result.exit_code == 0

    def test_validate_strict_mode(
        self,
        runner: CliRunner,
        sample_lists_file: Path,
        sample_items_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test validate command with --strict flag (all checks)."""
        output_dir = tmp_path / "experiment"

        # Generate experiment first
        runner.invoke(
            generate,
            [
                str(sample_lists_file),
                str(sample_items_file),
                str(output_dir),
                "--experiment-type",
                "likert_rating",
                "--distribution-strategy",
                "balanced",
            ],
        )

        # Validate with strict mode
        result = runner.invoke(
            validate,
            [str(output_dir), "--strict"],
        )

        assert result.exit_code == 0

    def test_validate_check_trials_with_config(
        self,
        runner: CliRunner,
        sample_lists_file: Path,
        sample_items_file: Path,
        tmp_path: Path,
    ) -> None:
        """Test validate command with --check-trials when trial configs exist."""
        output_dir = tmp_path / "experiment"

        # Generate experiment first
        runner.invoke(
            generate,
            [
                str(sample_lists_file),
                str(sample_items_file),
                str(output_dir),
                "--experiment-type",
                "likert_rating",
                "--distribution-strategy",
                "balanced",
            ],
        )

        # Add a trial config file
        config_dir = output_dir / "config"
        config_dir.mkdir(exist_ok=True)
        trial_config = config_dir / "rating.json"
        trial_config.write_text(
            json.dumps(
                {
                    "type": "rating_scale",
                    "min_value": 1,
                    "max_value": 7,
                }
            )
        )

        # Validate with trials check
        result = runner.invoke(
            validate,
            [str(output_dir), "--check-trials"],
        )

        assert result.exit_code == 0

    def test_validate_invalid_distribution_config(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test validate detects invalid distribution configuration."""
        exp_dir = tmp_path / "experiment"
        exp_dir.mkdir()
        data_dir = exp_dir / "data"
        data_dir.mkdir()

        # Create required files
        (exp_dir / "index.html").write_text("<html></html>")
        (data_dir / "config.json").write_text("{}")
        (data_dir / "items.jsonl").write_text("")
        (data_dir / "lists.jsonl").write_text("")

        # Create invalid distribution config
        (data_dir / "distribution.json").write_text(
            json.dumps(
                {
                    "strategy_type": "invalid_strategy",  # Invalid strategy
                }
            )
        )

        result = runner.invoke(
            validate,
            [str(exp_dir), "--check-distribution"],
        )

        # Should fail due to invalid strategy
        assert result.exit_code != 0

    def test_validate_broken_item_references(
        self,
        runner: CliRunner,
        tmp_path: Path,
    ) -> None:
        """Test validate detects broken item references in lists."""
        exp_dir = tmp_path / "experiment"
        exp_dir.mkdir()
        data_dir = exp_dir / "data"
        data_dir.mkdir()

        # Create required files
        (exp_dir / "index.html").write_text("<html></html>")
        (data_dir / "config.json").write_text("{}")
        (data_dir / "distribution.json").write_text(
            json.dumps(
                {
                    "strategy_type": "balanced",
                }
            )
        )

        # Create items file with one item
        (data_dir / "items.jsonl").write_text(
            '{"id": "12345678-1234-5678-1234-567812345678"}\n'
        )

        # Create lists file referencing non-existent item
        (data_dir / "lists.jsonl").write_text(
            '{"id": "11111111-1111-1111-1111-111111111111", '
            '"item_refs": ["nonexistent-uuid"]}\n'
        )

        result = runner.invoke(
            validate,
            [str(exp_dir), "--check-data-structure"],
        )

        # Should fail due to broken reference
        assert result.exit_code != 0
        assert "error" in result.output.lower()
