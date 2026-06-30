"""Tests for the simulation framework.

This module tests the simulation pipeline functions and integration with bead.simulation.
"""

from __future__ import annotations

import json

# Import from parent directory
import sys
import tempfile
from pathlib import Path
from uuid import uuid4

import pytest

from bead.items.item import Item

sys.path.insert(0, str(Path(__file__).parent.parent))
from simulate_pipeline import (
    get_forced_choice_template,
    load_2afc_pairs,
    run_simulation,
)


class TestGetForcedChoiceTemplate:
    """Test suite for get_forced_choice_template function."""

    def test_returns_valid_template(self) -> None:
        """Test that function returns a forced-choice ItemTemplate from the protocol."""
        template = get_forced_choice_template()

        # Template name comes from the protocol anchor (acceptability)
        assert template.name == "acceptability"
        assert template.task_type == "forced_choice"
        assert template.judgment_type == "acceptability"

    def test_template_has_required_task_spec(self) -> None:
        """Test that template has the prompt declared in config.yaml."""
        template = get_forced_choice_template()

        assert template.task_spec.prompt == "Which sentence sounds more natural?"
        # ``get_forced_choice_template`` enriches the bare protocol
        # template with response-space labels so the simulator can
        # sample from them.
        assert template.task_spec.options == ("first", "second")

    def test_template_presentation_spec(self) -> None:
        """Test that template has static presentation mode."""
        template = get_forced_choice_template()

        assert template.presentation_spec.mode == "static"


class TestLoad2AFCPairs:
    """Test suite for load_2afc_pairs function."""

    def test_load_with_limit(self) -> None:
        """Test loading with limit parameter."""
        # Create temporary JSONL file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i in range(10):
                item = Item(
                    item_template_id=uuid4(),
                    rendered_elements={"option_a": f"A{i}", "option_b": f"B{i}"},
                    item_metadata={"lm_score1": float(i), "lm_score2": 0.0},
                )
                f.write(item.model_dump_json() + "\n")
            temp_path = Path(f.name)

        try:
            # Load with limit
            items = load_2afc_pairs(temp_path, limit=5)
            assert len(items) == 5

            # Load all
            items_all = load_2afc_pairs(temp_path)
            assert len(items_all) == 10
        finally:
            temp_path.unlink()

    def test_load_with_skip(self) -> None:
        """Test loading with skip parameter."""
        # Create temporary JSONL file
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            for i in range(10):
                item = Item(
                    item_template_id=uuid4(),
                    rendered_elements={"option_a": f"A{i}", "option_b": f"B{i}"},
                    item_metadata={"lm_score1": float(i), "lm_score2": 0.0},
                )
                f.write(item.model_dump_json() + "\n")
            temp_path = Path(f.name)

        try:
            # Skip first 3, load next 5
            items = load_2afc_pairs(temp_path, limit=5, skip=3)
            assert len(items) == 5

            # Check that we skipped the first 3
            first_item = items[0]
            assert "A3" in str(first_item.rendered_elements["option_a"])
        finally:
            temp_path.unlink()


class TestRunSimulation:
    """Test suite for run_simulation function."""

    @pytest.fixture
    def temp_items_dir(self):
        """Create temporary directory with sample 2AFC pairs."""
        import tempfile

        with tempfile.TemporaryDirectory() as tmpdir:
            tmpdir_path = Path(tmpdir)

            # Create items subdirectory
            items_dir = tmpdir_path / "items"
            items_dir.mkdir()

            # Create sample 2AFC pairs
            pairs_path = items_dir / "2afc_pairs.jsonl"
            with open(pairs_path, "w") as f:
                for i in range(100):
                    item = Item(
                        item_template_id=uuid4(),
                        rendered_elements={
                            "option_a": f"Option A {i}",
                            "option_b": f"Option B {i}",
                        },
                        item_metadata={
                            "lm_score": float(i % 10),  # Use single lm_score key
                        },
                    )
                    f.write(item.model_dump_json() + "\n")

            # Change to temp directory
            import os

            original_dir = os.getcwd()
            os.chdir(tmpdir_path)

            yield tmpdir_path

            # Restore original directory
            os.chdir(original_dir)

    def test_simulation_completes(self, temp_items_dir) -> None:
        """Test that simulation runs to completion."""
        output_dir = temp_items_dir / "simulation_output"

        results = run_simulation(
            initial_size=20,
            budget_per_iteration=10,
            max_iterations=3,
            temperature=1.0,
            random_state=42,
            output_dir=output_dir,
            max_items=50,
        )

        # Check results structure
        assert "config" in results
        assert "human_agreement" in results
        assert "iterations" in results
        assert "converged" in results
        assert "total_annotations" in results

        # Check iterations
        assert len(results["iterations"]) <= 3
        assert results["total_annotations"] >= 20

    def test_simulation_creates_output_file(self, temp_items_dir) -> None:
        """Test that simulation creates output JSON file."""
        output_dir = temp_items_dir / "simulation_output"

        run_simulation(
            initial_size=20,
            budget_per_iteration=10,
            max_iterations=2,
            random_state=42,
            output_dir=output_dir,
            max_items=50,
        )

        results_path = output_dir / "simulation_results.json"
        assert results_path.exists()

        # Load and validate JSON
        with open(results_path) as f:
            data = json.load(f)
            assert "config" in data
            assert "iterations" in data

    def test_simulation_reproducible(self, temp_items_dir) -> None:
        """Test that simulation is reproducible with seed."""
        output_dir1 = temp_items_dir / "output1"
        output_dir2 = temp_items_dir / "output2"

        results1 = run_simulation(
            initial_size=20,
            budget_per_iteration=10,
            max_iterations=2,
            random_state=42,
            output_dir=output_dir1,
            max_items=50,
        )

        results2 = run_simulation(
            initial_size=20,
            budget_per_iteration=10,
            max_iterations=2,
            random_state=42,
            output_dir=output_dir2,
            max_items=50,
        )

        # Compare final accuracies
        assert (
            results1["iterations"][-1]["test_accuracy"]
            == results2["iterations"][-1]["test_accuracy"]
        )
        assert results1["human_agreement"] == results2["human_agreement"]
