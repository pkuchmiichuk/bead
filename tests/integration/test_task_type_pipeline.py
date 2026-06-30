"""Integration tests for task-type utilities through full pipeline.

These tests verify that items created by each task-type utility work correctly
through all pipeline stages:
- Stage 3: Items (task-type utilities)
- Stage 4: Lists (partitioning with constraints)
- Stage 5: Deployment (jsPsych/JATOS generation)

Stage 6 (active learning) is skipped as models may not be fully implemented.
"""

from __future__ import annotations

from pathlib import Path

from bead.deployment.distribution import (
    DistributionStrategyType,
    ListDistributionStrategy,
)
from bead.deployment.jatos.exporter import JATOSExporter
from bead.deployment.jspsych.config import ExperimentConfig, InstructionsConfig
from bead.deployment.jspsych.generator import JsPsychExperimentGenerator
from bead.items.binary import create_binary_item
from bead.items.categorical import create_nli_item
from bead.items.cloze import create_simple_cloze_item
from bead.items.forced_choice import create_forced_choice_item
from bead.items.free_text import create_free_text_item
from bead.items.item import Item
from bead.items.item_template import (
    ItemTemplate,
    PresentationSpec,
    ScaleBounds,
    TaskSpec,
)
from bead.items.magnitude import create_magnitude_item
from bead.items.multi_select import create_multi_select_item
from bead.items.ordinal_scale import create_likert_7_item
from bead.lists import ExperimentList
from bead.lists.partitioner import ListPartitioner


class TestForcedChoiceIntegration:
    """Integration tests for forced_choice through pipeline."""

    def test_forced_choice_full_pipeline(self, tmp_path: Path) -> None:
        """Test forced_choice items through stages 3-5."""
        # Stage 3: Create items
        items_list = [
            create_forced_choice_item(
                "The cat sat on the mat.",
                "The dog sat on the mat.",
                metadata={"condition": "A"},
            ),
            create_forced_choice_item(
                "Red apple", "Green apple", "Yellow apple", metadata={"condition": "B"}
            ),
            create_forced_choice_item(
                "Hot coffee", "Cold coffee", metadata={"condition": "A"}
            ),
        ]

        # Verify items created correctly
        assert len(items_list) == 3
        assert all(isinstance(item, Item) for item in items_list)
        assert len(items_list[0].options) == 2
        assert items_list[0].options[0] == "The cat sat on the mat."
        assert items_list[0].options[1] == "The dog sat on the mat."

        # Stage 4: Partition into lists
        partitioner = ListPartitioner(random_seed=42)
        item_uuids = [item.id for item in items_list]
        metadata_dict = {item.id: item.item_metadata for item in items_list}

        lists = partitioner.partition(
            items=item_uuids, n_lists=2, strategy="balanced", metadata=metadata_dict
        )

        assert len(lists) == 2
        assert all(isinstance(lst, ExperimentList) for lst in lists)
        assert len(lists[0].item_refs) + len(lists[1].item_refs) == 3

        # Stage 5: Generate jsPsych deployment
        config = ExperimentConfig(
            experiment_type="forced_choice",
            title="Forced Choice Test",
            description="Test forced choice deployment",
            instructions=InstructionsConfig.from_text("Choose the best option"),
            randomize_trial_order=False,
            distribution_strategy=ListDistributionStrategy(
                strategy_type=DistributionStrategyType.BALANCED
            ),
        )

        generator = JsPsychExperimentGenerator(
            config=config, output_dir=tmp_path / "jspsych"
        )

        # Create items dict and dummy templates
        items_dict = {item.id: item for item in items_list}
        dummy_template = ItemTemplate(
            name="test",
            judgment_type="preference",
            task_type="forced_choice",
            task_spec=TaskSpec(prompt="Choose"),
            presentation_spec=PresentationSpec(mode="static"),
        )
        templates_dict = {dummy_template.id: dummy_template}

        items_list = [
            item.with_(item_template_id=dummy_template.id) for item in items_list
        ]
        items_dict = {item.id: item for item in items_list}

        output_dir = generator.generate(lists, items_dict, templates_dict)

        # Verify jsPsych files created
        assert output_dir.exists()
        assert (output_dir / "index.html").exists()
        assert (output_dir / "css").exists()
        assert (output_dir / "js").exists()
        assert (output_dir / "data").exists()

        # Stage 5b: Export to JATOS
        exporter = JATOSExporter(
            study_title="Forced Choice Study", study_description="Test study"
        )
        jzip_path = tmp_path / "forced_choice.jzip"
        exporter.export(experiment_dir=output_dir, output_path=jzip_path)

        assert jzip_path.exists()
        assert jzip_path.suffix == ".jzip"


class TestMultiSelectIntegration:
    """Integration tests for multi_select through pipeline."""

    def test_multi_select_full_pipeline(self, tmp_path: Path) -> None:
        """Test multi_select items through stages 3-5."""
        # Stage 3: Create items
        items_list = [
            create_multi_select_item(
                "Option A",
                "Option B",
                "Option C",
                min_selections=1,
                max_selections=3,
                metadata={"group": "1"},
            ),
            create_multi_select_item(
                "Red", "Blue", "Green", "Yellow", min_selections=2, max_selections=2
            ),
        ]

        assert len(items_list) == 2
        assert items_list[0].item_metadata["min_selections"] == 1
        assert items_list[0].item_metadata["max_selections"] == 3

        # Stage 4: Partition
        partitioner = ListPartitioner(random_seed=42)
        item_uuids = [item.id for item in items_list]
        metadata_dict = {item.id: item.item_metadata for item in items_list}

        lists = partitioner.partition(
            items=item_uuids, n_lists=1, strategy="random", metadata=metadata_dict
        )

        assert len(lists) == 1
        assert len(lists[0].item_refs) == 2

        # Stage 5: Deploy
        config = ExperimentConfig(
            experiment_type="forced_choice",
            title="Multi-Select Test",
            description="Test",
            instructions=InstructionsConfig.from_text("Select options"),
            distribution_strategy=ListDistributionStrategy(
                strategy_type=DistributionStrategyType.BALANCED
            ),
        )

        generator = JsPsychExperimentGenerator(
            config=config, output_dir=tmp_path / "jspsych"
        )

        items_dict = {item.id: item for item in items_list}
        dummy_template = ItemTemplate(
            name="test",
            judgment_type="comprehension",
            task_type="multi_select",
            task_spec=TaskSpec(prompt="Select"),
            presentation_spec=PresentationSpec(mode="static"),
        )
        templates_dict = {dummy_template.id: dummy_template}

        items_list = [
            item.with_(item_template_id=dummy_template.id) for item in items_list
        ]
        items_dict = {item.id: item for item in items_list}

        output_dir = generator.generate(lists, items_dict, templates_dict)
        assert (output_dir / "index.html").exists()


class TestBinaryIntegration:
    """Integration tests for binary through pipeline."""

    def test_binary_full_pipeline(self, tmp_path: Path) -> None:
        """Test binary items through stages 3-5."""
        # Stage 3: Create items
        items_list = [
            create_binary_item("The cat sat.", prompt="Is this grammatical?"),
            create_binary_item("Cat the sat.", prompt="Is this grammatical?"),
        ]

        assert len(items_list) == 2
        assert "text" in items_list[0].rendered_elements
        assert "prompt" in items_list[0].rendered_elements

        # Stage 4: Partition
        partitioner = ListPartitioner()
        item_uuids = [item.id for item in items_list]

        lists = partitioner.partition(items=item_uuids, n_lists=1, strategy="random")

        assert len(lists) == 1

        # Stage 5: Deploy
        config = ExperimentConfig(
            experiment_type="binary_choice",
            title="Binary Test",
            description="Test",
            instructions=InstructionsConfig.from_text("Answer yes/no"),
            distribution_strategy=ListDistributionStrategy(
                strategy_type=DistributionStrategyType.BALANCED
            ),
        )

        generator = JsPsychExperimentGenerator(
            config=config, output_dir=tmp_path / "jspsych"
        )

        items_dict = {item.id: item for item in items_list}
        dummy_template = ItemTemplate(
            name="test",
            judgment_type="acceptability",
            task_type="binary",
            task_spec=TaskSpec(prompt="Grammatical?"),
            presentation_spec=PresentationSpec(mode="static"),
        )
        templates_dict = {dummy_template.id: dummy_template}

        items_list = [
            item.with_(item_template_id=dummy_template.id) for item in items_list
        ]
        items_dict = {item.id: item for item in items_list}

        output_dir = generator.generate(lists, items_dict, templates_dict)
        assert (output_dir / "index.html").exists()


class TestCategoricalIntegration:
    """Integration tests for categorical through pipeline."""

    def test_categorical_full_pipeline(self, tmp_path: Path) -> None:
        """Test categorical items through stages 3-5."""
        # Stage 3: Create items (using NLI helper)
        items_list = [
            create_nli_item("All dogs bark", "Some dogs bark"),
            create_nli_item("No cats meow", "Some cats meow"),
        ]

        assert len(items_list) == 2
        assert "categories" in items_list[0].item_metadata
        categories = items_list[0].item_metadata["categories"]
        assert isinstance(categories, list | tuple)
        assert len(categories) == 3

        # Stage 4: Partition
        partitioner = ListPartitioner()
        item_uuids = [item.id for item in items_list]

        lists = partitioner.partition(items=item_uuids, n_lists=1, strategy="random")

        # Stage 5: Deploy
        # Note: categorical items use categories in metadata, not item.options
        # Using likert_rating as a deployment test since categorical experiment
        # type is not yet implemented
        config = ExperimentConfig(
            experiment_type="likert_rating",
            title="NLI Test",
            description="Test",
            instructions=InstructionsConfig.from_text("Select relationship"),
            distribution_strategy=ListDistributionStrategy(
                strategy_type=DistributionStrategyType.BALANCED
            ),
        )

        generator = JsPsychExperimentGenerator(
            config=config, output_dir=tmp_path / "jspsych"
        )

        items_dict = {item.id: item for item in items_list}
        dummy_template = ItemTemplate(
            name="test",
            judgment_type="inference",
            task_type="categorical",
            task_spec=TaskSpec(prompt="Relationship?"),
            presentation_spec=PresentationSpec(mode="static"),
        )
        templates_dict = {dummy_template.id: dummy_template}

        items_list = [
            item.with_(item_template_id=dummy_template.id) for item in items_list
        ]
        items_dict = {item.id: item for item in items_list}

        output_dir = generator.generate(lists, items_dict, templates_dict)
        assert (output_dir / "index.html").exists()


class TestOrdinalScaleIntegration:
    """Integration tests for ordinal_scale through pipeline."""

    def test_ordinal_scale_full_pipeline(self, tmp_path: Path) -> None:
        """Test ordinal_scale items through stages 3-5."""
        # Stage 3: Create items
        items_list = [
            create_likert_7_item("How natural is this sentence?"),
            create_likert_7_item("How acceptable is this phrase?"),
        ]

        assert len(items_list) == 2
        assert items_list[0].item_metadata["scale_min"] == 1
        assert items_list[0].item_metadata["scale_max"] == 7

        # Stage 4: Partition
        partitioner = ListPartitioner()
        item_uuids = [item.id for item in items_list]

        lists = partitioner.partition(items=item_uuids, n_lists=1, strategy="random")

        # Stage 5: Deploy
        config = ExperimentConfig(
            experiment_type="likert_rating",
            title="Likert Test",
            description="Test",
            instructions=InstructionsConfig.from_text("Rate sentences"),
            distribution_strategy=ListDistributionStrategy(
                strategy_type=DistributionStrategyType.BALANCED
            ),
        )

        generator = JsPsychExperimentGenerator(
            config=config, output_dir=tmp_path / "jspsych"
        )

        items_dict = {item.id: item for item in items_list}
        dummy_template = ItemTemplate(
            name="test",
            judgment_type="acceptability",
            task_type="ordinal_scale",
            task_spec=TaskSpec(prompt="Rate", scale_bounds=ScaleBounds(min=1, max=7)),
            presentation_spec=PresentationSpec(mode="static"),
        )
        templates_dict = {dummy_template.id: dummy_template}

        items_list = [
            item.with_(item_template_id=dummy_template.id) for item in items_list
        ]
        items_dict = {item.id: item for item in items_list}

        output_dir = generator.generate(lists, items_dict, templates_dict)
        assert (output_dir / "index.html").exists()


class TestMagnitudeIntegration:
    """Integration tests for magnitude through pipeline."""

    def test_magnitude_full_pipeline(self, tmp_path: Path) -> None:
        """Test magnitude items through stages 3-5."""
        # Stage 3: Create items
        items_list = [
            create_magnitude_item(
                "The cat sat on the mat.", unit="ms", prompt="Reading time?"
            ),
            create_magnitude_item(
                "How confident?", bounds=(0, 100), unit="%", prompt="Confidence?"
            ),
        ]

        assert len(items_list) == 2
        assert items_list[0].item_metadata.get("unit") == "ms"

        # Stage 4: Partition
        partitioner = ListPartitioner()
        item_uuids = [item.id for item in items_list]

        lists = partitioner.partition(items=item_uuids, n_lists=1, strategy="random")

        # Stage 5: Deploy
        config = ExperimentConfig(
            experiment_type="slider_rating",
            title="Magnitude Test",
            description="Test",
            instructions=InstructionsConfig.from_text("Enter numeric value"),
            distribution_strategy=ListDistributionStrategy(
                strategy_type=DistributionStrategyType.BALANCED
            ),
        )

        generator = JsPsychExperimentGenerator(
            config=config, output_dir=tmp_path / "jspsych"
        )

        items_dict = {item.id: item for item in items_list}
        dummy_template = ItemTemplate(
            name="test",
            judgment_type="plausibility",
            task_type="magnitude",
            task_spec=TaskSpec(prompt="Value?"),
            presentation_spec=PresentationSpec(mode="static"),
        )
        templates_dict = {dummy_template.id: dummy_template}

        items_list = [
            item.with_(item_template_id=dummy_template.id) for item in items_list
        ]
        items_dict = {item.id: item for item in items_list}

        output_dir = generator.generate(lists, items_dict, templates_dict)
        assert (output_dir / "index.html").exists()


class TestFreeTextIntegration:
    """Integration tests for free_text through pipeline."""

    def test_free_text_full_pipeline(self, tmp_path: Path) -> None:
        """Test free_text items through stages 3-5."""
        # Stage 3: Create items
        items_list = [
            create_free_text_item("The cat chased the mouse.", prompt="Paraphrase:"),
            create_free_text_item(
                "The dog barked.", prompt="What is the subject?", max_length=50
            ),
        ]

        assert len(items_list) == 2
        assert "text" in items_list[0].rendered_elements
        assert "prompt" in items_list[0].rendered_elements

        # Stage 4: Partition
        partitioner = ListPartitioner()
        item_uuids = [item.id for item in items_list]

        lists = partitioner.partition(items=item_uuids, n_lists=1, strategy="random")

        # Stage 5: Deploy
        config = ExperimentConfig(
            experiment_type="likert_rating",
            title="Free Text Test",
            description="Test",
            instructions=InstructionsConfig.from_text("Enter text"),
            distribution_strategy=ListDistributionStrategy(
                strategy_type=DistributionStrategyType.BALANCED
            ),
        )

        generator = JsPsychExperimentGenerator(
            config=config, output_dir=tmp_path / "jspsych"
        )

        items_dict = {item.id: item for item in items_list}
        dummy_template = ItemTemplate(
            name="test",
            judgment_type="comprehension",
            task_type="free_text",
            task_spec=TaskSpec(prompt="Answer:"),
            presentation_spec=PresentationSpec(mode="static"),
        )
        templates_dict = {dummy_template.id: dummy_template}

        items_list = [
            item.with_(item_template_id=dummy_template.id) for item in items_list
        ]
        items_dict = {item.id: item for item in items_list}

        output_dir = generator.generate(lists, items_dict, templates_dict)
        assert (output_dir / "index.html").exists()


class TestClozeIntegration:
    """Integration tests for cloze through pipeline."""

    def test_cloze_full_pipeline(self, tmp_path: Path) -> None:
        """Test cloze items through stages 3-5."""
        # Stage 3: Create items
        items_list = [
            create_simple_cloze_item(
                text="The quick brown fox",
                blank_positions=[1],
                blank_labels=["adjective"],
            ),
            create_simple_cloze_item(
                text="She walks to school",
                blank_positions=[1],
                blank_labels=["verb"],
            ),
        ]

        assert len(items_list) == 2
        assert len(items_list[0].unfilled_slots) == 1
        assert items_list[0].item_metadata["n_unfilled_slots"] == 1

        # Stage 4: Partition
        partitioner = ListPartitioner()
        item_uuids = [item.id for item in items_list]

        lists = partitioner.partition(items=item_uuids, n_lists=1, strategy="random")

        # Stage 5: Deploy
        config = ExperimentConfig(
            experiment_type="likert_rating",
            title="Cloze Test",
            description="Test",
            instructions=InstructionsConfig.from_text("Fill in the blank"),
            distribution_strategy=ListDistributionStrategy(
                strategy_type=DistributionStrategyType.BALANCED
            ),
        )

        generator = JsPsychExperimentGenerator(
            config=config, output_dir=tmp_path / "jspsych"
        )

        items_dict = {item.id: item for item in items_list}
        dummy_template = ItemTemplate(
            name="test",
            judgment_type="comprehension",
            task_type="cloze",
            task_spec=TaskSpec(prompt="Fill blank:"),
            presentation_spec=PresentationSpec(mode="static"),
        )
        templates_dict = {dummy_template.id: dummy_template}

        items_list = [
            item.with_(item_template_id=dummy_template.id) for item in items_list
        ]
        items_dict = {item.id: item for item in items_list}

        output_dir = generator.generate(lists, items_dict, templates_dict)
        assert (output_dir / "index.html").exists()
