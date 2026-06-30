"""Tests for JsPsychExperimentGenerator (batch mode)."""

from __future__ import annotations

import json
from pathlib import Path
from uuid import uuid4

import pytest

from bead.data.serialization import read_jsonlines
from bead.deployment.jspsych.config import (
    ChoiceConfig,
    ExperimentConfig,
    RatingScaleConfig,
)
from bead.deployment.jspsych.generator import JsPsychExperimentGenerator
from bead.items.item import Item
from bead.lists import ExperimentList


class TestGeneratorInitialization:
    """Tests for JsPsychExperimentGenerator initialization."""

    def test_basic_initialization(
        self,
        sample_experiment_config: ExperimentConfig,
        tmp_output_dir: Path,
    ) -> None:
        """Test generator initialization."""
        generator = JsPsychExperimentGenerator(
            config=sample_experiment_config,
            output_dir=tmp_output_dir,
        )

        assert generator.config == sample_experiment_config
        assert generator.output_dir == tmp_output_dir
        assert generator.rating_config is not None
        assert generator.choice_config is not None

    def test_initialization_with_custom_configs(
        self,
        sample_experiment_config: ExperimentConfig,
        sample_rating_config: RatingScaleConfig,
        sample_choice_config: ChoiceConfig,
        tmp_output_dir: Path,
    ) -> None:
        """Test initialization with custom rating and choice configs."""
        generator = JsPsychExperimentGenerator(
            config=sample_experiment_config,
            output_dir=tmp_output_dir,
            rating_config=sample_rating_config,
            choice_config=sample_choice_config,
        )

        assert generator.rating_config == sample_rating_config
        assert generator.choice_config == sample_choice_config


class TestDirectoryStructure:
    """Tests for directory structure creation."""

    def test_create_directory_structure(
        self,
        sample_experiment_config: ExperimentConfig,
        tmp_output_dir: Path,
    ) -> None:
        """Test directory structure creation."""
        generator = JsPsychExperimentGenerator(
            config=sample_experiment_config,
            output_dir=tmp_output_dir,
        )

        generator._create_directory_structure()

        assert (tmp_output_dir / "css").exists()
        assert (tmp_output_dir / "js").exists()
        assert (tmp_output_dir / "data").exists()


class TestBatchGeneration:
    """Tests for batch experiment generation."""

    def test_generate_batch_experiment(
        self,
        sample_experiment_config: ExperimentConfig,
        sample_experiment_list: ExperimentList,
        sample_items: dict,
        sample_templates: dict,
        tmp_output_dir: Path,
    ) -> None:
        """Test complete batch experiment generation."""
        generator = JsPsychExperimentGenerator(
            config=sample_experiment_config,
            output_dir=tmp_output_dir,
        )

        output_path = generator.generate(
            lists=[sample_experiment_list],
            items=sample_items,
            templates=sample_templates,
        )

        assert output_path == tmp_output_dir

        # Check core files
        assert (tmp_output_dir / "index.html").exists()
        assert (tmp_output_dir / "css" / "experiment.css").exists()
        assert (tmp_output_dir / "js" / "experiment.js").exists()
        assert (tmp_output_dir / "data" / "config.json").exists()

        # Check batch mode files
        assert (tmp_output_dir / "js" / "list_distributor.js").exists()
        assert (tmp_output_dir / "data" / "lists.jsonl").exists()
        assert (tmp_output_dir / "data" / "items.jsonl").exists()
        assert (tmp_output_dir / "data" / "distribution.json").exists()

    def test_lists_jsonl_content(
        self,
        sample_experiment_config: ExperimentConfig,
        sample_experiment_list: ExperimentList,
        sample_items: dict,
        sample_templates: dict,
        tmp_output_dir: Path,
    ) -> None:
        """Test lists.jsonl file content."""
        generator = JsPsychExperimentGenerator(
            config=sample_experiment_config,
            output_dir=tmp_output_dir,
        )

        generator.generate(
            lists=[sample_experiment_list],
            items=sample_items,
            templates=sample_templates,
        )

        # Read lists.jsonl
        lists_path = tmp_output_dir / "data" / "lists.jsonl"
        loaded_lists = read_jsonlines(lists_path, ExperimentList)

        assert len(loaded_lists) == 1
        assert loaded_lists[0].name == sample_experiment_list.name
        assert loaded_lists[0].id == sample_experiment_list.id

    def test_items_jsonl_content(
        self,
        sample_experiment_config: ExperimentConfig,
        sample_experiment_list: ExperimentList,
        sample_items: dict,
        sample_templates: dict,
        tmp_output_dir: Path,
    ) -> None:
        """Test items.jsonl file content."""
        generator = JsPsychExperimentGenerator(
            config=sample_experiment_config,
            output_dir=tmp_output_dir,
        )

        generator.generate(
            lists=[sample_experiment_list],
            items=sample_items,
            templates=sample_templates,
        )

        # Read items.jsonl
        items_path = tmp_output_dir / "data" / "items.jsonl"
        loaded_items = read_jsonlines(items_path, Item)

        assert len(loaded_items) == len(sample_items)

        # Check all items are present
        loaded_ids = {item.id for item in loaded_items}
        expected_ids = set(sample_items.keys())
        assert loaded_ids == expected_ids

    def test_distribution_json_content(
        self,
        sample_experiment_config: ExperimentConfig,
        sample_experiment_list: ExperimentList,
        sample_items: dict,
        sample_templates: dict,
        tmp_output_dir: Path,
    ) -> None:
        """Test distribution.json file content."""
        generator = JsPsychExperimentGenerator(
            config=sample_experiment_config,
            output_dir=tmp_output_dir,
        )

        generator.generate(
            lists=[sample_experiment_list],
            items=sample_items,
            templates=sample_templates,
        )

        # Read distribution.json
        dist_path = tmp_output_dir / "data" / "distribution.json"
        dist_config = json.loads(dist_path.read_text())

        assert "strategy_type" in dist_config
        assert dist_config["strategy_type"] == "balanced"


class TestValidation:
    """Tests for input validation."""

    def test_empty_lists_raises_error(
        self,
        sample_experiment_config: ExperimentConfig,
        sample_items: dict,
        sample_templates: dict,
        tmp_output_dir: Path,
    ) -> None:
        """Test error when no lists provided."""
        generator = JsPsychExperimentGenerator(
            config=sample_experiment_config,
            output_dir=tmp_output_dir,
        )

        with pytest.raises(ValueError, match="at least one ExperimentList"):
            generator.generate(
                lists=[],
                items=sample_items,
                templates=sample_templates,
            )

    def test_empty_items_raises_error(
        self,
        sample_experiment_config: ExperimentConfig,
        sample_experiment_list: ExperimentList,
        sample_templates: dict,
        tmp_output_dir: Path,
    ) -> None:
        """Test error when no items provided."""
        generator = JsPsychExperimentGenerator(
            config=sample_experiment_config,
            output_dir=tmp_output_dir,
        )

        with pytest.raises(ValueError, match="requires items dictionary"):
            generator.generate(
                lists=[sample_experiment_list],
                items={},
                templates=sample_templates,
            )

    def test_empty_templates_raises_error(
        self,
        sample_experiment_config: ExperimentConfig,
        sample_experiment_list: ExperimentList,
        sample_items: dict,
        tmp_output_dir: Path,
    ) -> None:
        """Test error when no templates provided."""
        generator = JsPsychExperimentGenerator(
            config=sample_experiment_config,
            output_dir=tmp_output_dir,
        )

        with pytest.raises(ValueError, match="requires templates dictionary"):
            generator.generate(
                lists=[sample_experiment_list],
                items=sample_items,
                templates={},
            )

    def test_missing_item_reference_raises_error(
        self,
        sample_experiment_config: ExperimentConfig,
        sample_experiment_list: ExperimentList,
        sample_items: dict,
        sample_templates: dict,
        tmp_output_dir: Path,
    ) -> None:
        """Test error when list references non-existent item."""
        generator = JsPsychExperimentGenerator(
            config=sample_experiment_config,
            output_dir=tmp_output_dir,
        )

        # Add a non-existent item to the list
        sample_experiment_list = sample_experiment_list.with_item(uuid4())
        with pytest.raises(ValueError, match="not found in items"):
            generator.generate(
                lists=[sample_experiment_list],
                items=sample_items,
                templates=sample_templates,
            )

    def test_missing_template_reference_raises_error(
        self,
        sample_experiment_config: ExperimentConfig,
        sample_experiment_list: ExperimentList,
        sample_items: dict,
        sample_templates: dict,
        tmp_output_dir: Path,
    ) -> None:
        """Test error when item references non-existent template."""
        generator = JsPsychExperimentGenerator(
            config=sample_experiment_config,
            output_dir=tmp_output_dir,
        )

        # Create an item with non-existent template
        bad_item_id = uuid4()
        bad_item = Item(
            id=bad_item_id,
            item_template_id=uuid4(),  # Non-existent template
            rendered_elements={"text": "test"},
            item_metadata={},
        )
        sample_items[bad_item_id] = bad_item
        sample_experiment_list = sample_experiment_list.with_item(bad_item_id)
        with pytest.raises(ValueError, match="not found in templates"):
            generator.generate(
                lists=[sample_experiment_list],
                items=sample_items,
                templates=sample_templates,
            )


class TestMultipleLists:
    """Tests for multiple list generation."""

    def test_multiple_lists(
        self,
        sample_experiment_config: ExperimentConfig,
        sample_items: dict,
        sample_templates: dict,
        tmp_output_dir: Path,
    ) -> None:
        """Test generation with multiple lists."""
        generator = JsPsychExperimentGenerator(
            config=sample_experiment_config,
            output_dir=tmp_output_dir,
        )

        # Create 3 lists
        item_ids = list(sample_items.keys())
        lists = []
        for i in range(3):
            exp_list = ExperimentList(name=f"list_{i}", list_number=i)
            # Add first 3 items to each list (simple example)
            for item_id in item_ids[:3]:
                exp_list = exp_list.with_item(item_id)
            lists.append(exp_list)

        generator.generate(
            lists=lists,
            items=sample_items,
            templates=sample_templates,
        )

        # Read lists.jsonl
        lists_path = tmp_output_dir / "data" / "lists.jsonl"
        loaded_lists = read_jsonlines(lists_path, ExperimentList)

        assert len(loaded_lists) == 3
        assert {lst.name for lst in loaded_lists} == {"list_0", "list_1", "list_2"}


class TestListDistributorGeneration:
    """Tests for list_distributor.js generation and content."""

    def test_list_distributor_js_exists(
        self,
        sample_experiment_config: ExperimentConfig,
        sample_experiment_list: ExperimentList,
        sample_items: dict,
        sample_templates: dict,
        tmp_output_dir: Path,
    ) -> None:
        """Test that list_distributor.js is generated."""
        generator = JsPsychExperimentGenerator(
            config=sample_experiment_config,
            output_dir=tmp_output_dir,
        )

        generator.generate(
            lists=[sample_experiment_list],
            items=sample_items,
            templates=sample_templates,
        )

        list_distributor_path = tmp_output_dir / "js" / "list_distributor.js"
        assert list_distributor_path.exists()

    def test_list_distributor_contains_queue_functions(
        self,
        sample_experiment_config: ExperimentConfig,
        sample_experiment_list: ExperimentList,
        sample_items: dict,
        sample_templates: dict,
        tmp_output_dir: Path,
    ) -> None:
        """Test that list_distributor.js contains new queue initialization functions."""
        generator = JsPsychExperimentGenerator(
            config=sample_experiment_config,
            output_dir=tmp_output_dir,
        )

        generator.generate(
            lists=[sample_experiment_list],
            items=sample_items,
            templates=sample_templates,
        )

        list_distributor_path = tmp_output_dir / "js" / "list_distributor.js"
        content = list_distributor_path.read_text()

        # Check for new queue initialization functions
        assert "function initializeRandom" in content
        assert "function initializeSequential" in content
        assert "function initializeLatinSquare" in content
        assert "function initializeQuotaBased" in content
        assert "function shuffleArray" in content

    def test_list_distributor_contains_atomic_functions(
        self,
        sample_experiment_config: ExperimentConfig,
        sample_experiment_list: ExperimentList,
        sample_items: dict,
        sample_templates: dict,
        tmp_output_dir: Path,
    ) -> None:
        """Test that list_distributor.js contains atomic update functions."""
        generator = JsPsychExperimentGenerator(
            config=sample_experiment_config,
            output_dir=tmp_output_dir,
        )

        generator.generate(
            lists=[sample_experiment_list],
            items=sample_items,
            templates=sample_templates,
        )

        list_distributor_path = tmp_output_dir / "js" / "list_distributor.js"
        content = list_distributor_path.read_text()

        # Check for atomic update functions (TypeScript version consolidated into two)
        assert "function updateQueueAtomically" in content
        assert "function updateStatisticsAtomically" in content

    def test_list_distributor_contains_all_strategies(
        self,
        sample_experiment_config: ExperimentConfig,
        sample_experiment_list: ExperimentList,
        sample_items: dict,
        sample_templates: dict,
        tmp_output_dir: Path,
    ) -> None:
        """Test list_distributor.js contains all 8 strategy functions."""
        generator = JsPsychExperimentGenerator(
            config=sample_experiment_config,
            output_dir=tmp_output_dir,
        )

        generator.generate(
            lists=[sample_experiment_list],
            items=sample_items,
            templates=sample_templates,
        )

        list_distributor_path = tmp_output_dir / "js" / "list_distributor.js"
        content = list_distributor_path.read_text()

        # Check for all strategy assignment functions (async versions)
        assert "async function assignRandom" in content
        assert "async function assignSequential" in content
        assert "async function assignBalanced" in content
        assert "async function assignLatinSquare" in content
        assert "async function assignStratified" in content
        assert "async function assignWeightedRandom" in content
        assert "async function assignQuotaBased" in content
        assert "async function assignMetadataBased" in content

    def test_list_distributor_contains_unified_assignment(
        self,
        sample_experiment_config: ExperimentConfig,
        sample_experiment_list: ExperimentList,
        sample_items: dict,
        sample_templates: dict,
        tmp_output_dir: Path,
    ) -> None:
        """Test that list_distributor.js contains unified assignList function."""
        generator = JsPsychExperimentGenerator(
            config=sample_experiment_config,
            output_dir=tmp_output_dir,
        )

        generator.generate(
            lists=[sample_experiment_list],
            items=sample_items,
            templates=sample_templates,
        )

        list_distributor_path = tmp_output_dir / "js" / "list_distributor.js"
        content = list_distributor_path.read_text()

        # Check for unified assignment function
        assert "async function assignList" in content
        # Check it routes to all strategies (double quotes in compiled TS)
        assert 'case "random":' in content
        assert 'case "sequential":' in content
        assert 'case "balanced":' in content
        assert 'case "latin_square":' in content
        assert 'case "stratified":' in content
        assert 'case "weighted_random":' in content
        assert 'case "quota_based":' in content
        assert 'case "metadata_based":' in content
