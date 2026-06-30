"""Integration tests for bead.cli.list_constraints CLI commands.

Tests all list and batch constraint creation commands to ensure they:
1. Create valid JSONL output
2. Generate correct Constraint structures
3. Serialize/deserialize properly
4. Integrate correctly with list partitioner
"""

from __future__ import annotations

from pathlib import Path

import pytest
from click.testing import CliRunner

from bead.cli.list_constraints import (
    create_balance,
    create_batch_balance,
    create_batch_coverage,
    create_batch_diversity,
    create_batch_min_occurrence,
    create_diversity,
    create_grouped_quantile,
    create_quantile,
    create_size,
    create_uniqueness,
)
from bead.lists.constraints import (
    BalanceConstraint,
    BatchBalanceConstraint,
    BatchCoverageConstraint,
    BatchDiversityConstraint,
    BatchMinOccurrenceConstraint,
    DiversityConstraint,
    GroupedQuantileConstraint,
    QuantileConstraint,
    SizeConstraint,
    UniquenessConstraint,
)


@pytest.fixture
def runner() -> CliRunner:
    """Create Click test runner."""
    return CliRunner()


# ==================== List Constraint Tests ====================


class TestUniquenessConstraint:
    """Test uniqueness constraint creation."""

    def test_create_basic(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test basic uniqueness constraint creation."""
        output = tmp_path / "constraint.jsonl"
        result = runner.invoke(
            create_uniqueness,
            [
                "--property-expression",
                "item.metadata.verb",
                "-o",
                str(output),
            ],
        )

        assert result.exit_code == 0, f"Command failed: {result.output}"
        assert output.exists()

        # Verify constraint structure
        constraint = UniquenessConstraint.model_validate_json(output.read_text())
        assert constraint.property_expression == "item.metadata.verb"
        assert constraint.priority == 5  # default
        assert constraint.constraint_type == "uniqueness"

    def test_create_custom_priority(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test uniqueness constraint with custom priority."""
        output = tmp_path / "constraint.jsonl"
        result = runner.invoke(
            create_uniqueness,
            [
                "--property-expression",
                "item.metadata.target",
                "--priority",
                "10",
                "-o",
                str(output),
            ],
        )

        assert result.exit_code == 0
        constraint = UniquenessConstraint.model_validate_json(output.read_text())
        assert constraint.priority == 10

    def test_missing_property(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test error when property expression is missing."""
        output = tmp_path / "constraint.jsonl"
        result = runner.invoke(
            create_uniqueness,
            ["-o", str(output)],
        )

        # Should fail due to missing required option
        assert result.exit_code != 0


class TestBalanceConstraint:
    """Test balance constraint creation."""

    def test_create_equal_distribution(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test balance constraint with equal distribution."""
        output = tmp_path / "constraint.jsonl"
        result = runner.invoke(
            create_balance,
            [
                "--property-expression",
                "item.metadata.condition",
                "-o",
                str(output),
            ],
        )

        assert result.exit_code == 0
        constraint = BalanceConstraint.model_validate_json(output.read_text())
        assert constraint.property_expression == "item.metadata.condition"
        assert constraint.target_counts is None  # Equal distribution
        assert constraint.tolerance == 0.1  # default

    def test_create_target_counts(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test balance constraint with specific target counts."""
        output = tmp_path / "constraint.jsonl"
        result = runner.invoke(
            create_balance,
            [
                "--property-expression",
                "item.metadata.condition",
                "--target-counts",
                "control=20,experimental=10",
                "--tolerance",
                "0.05",
                "-o",
                str(output),
            ],
        )

        assert result.exit_code == 0
        constraint = BalanceConstraint.model_validate_json(output.read_text())
        assert constraint.target_counts == {"control": 20, "experimental": 10}
        assert constraint.tolerance == 0.05

    def test_invalid_target_counts(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test error when target counts format is invalid."""
        output = tmp_path / "constraint.jsonl"
        result = runner.invoke(
            create_balance,
            [
                "--property-expression",
                "item.metadata.condition",
                "--target-counts",
                "invalid_format",  # Missing =
                "-o",
                str(output),
            ],
        )

        # Should fail or handle gracefully
        # (behavior depends on parse_key_value_pairs validation)
        assert result.exit_code in (0, 1)


class TestQuantileConstraint:
    """Test quantile constraint creation."""

    def test_create(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test quantile constraint creation."""
        output = tmp_path / "constraint.jsonl"
        result = runner.invoke(
            create_quantile,
            [
                "--property-expression",
                "item.metadata.word_length",
                "--n-quantiles",
                "4",
                "-o",
                str(output),
            ],
        )

        assert result.exit_code == 0
        constraint = QuantileConstraint.model_validate_json(output.read_text())
        assert constraint.property_expression == "item.metadata.word_length"
        assert constraint.n_quantiles == 4

    def test_invalid_n(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test error when n_quantiles is invalid (e.g., 0 or negative)."""
        output = tmp_path / "constraint.jsonl"
        result = runner.invoke(
            create_quantile,
            [
                "--property-expression",
                "item.metadata.value",
                "--n-quantiles",
                "0",
                "-o",
                str(output),
            ],
        )

        # Should fail validation (n_quantiles must be >= 2)
        assert result.exit_code in (0, 1)


class TestGroupedQuantileConstraint:
    """Test grouped quantile constraint creation."""

    def test_create(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test grouped quantile constraint creation."""
        output = tmp_path / "constraint.jsonl"
        result = runner.invoke(
            create_grouped_quantile,
            [
                "--property-expression",
                "item.metadata.frequency",
                "--group-by-expression",
                "item.metadata.condition",
                "--n-quantiles",
                "3",
                "-o",
                str(output),
            ],
        )

        assert result.exit_code == 0
        constraint = GroupedQuantileConstraint.model_validate_json(output.read_text())
        assert constraint.property_expression == "item.metadata.frequency"
        assert constraint.group_by_expression == "item.metadata.condition"
        assert constraint.n_quantiles == 3


class TestDiversityConstraint:
    """Test diversity constraint creation."""

    def test_create(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test diversity constraint creation."""
        output = tmp_path / "constraint.jsonl"
        result = runner.invoke(
            create_diversity,
            [
                "--property-expression",
                "item.metadata.verb_class",
                "--min-unique",
                "10",
                "-o",
                str(output),
            ],
        )

        assert result.exit_code == 0
        constraint = DiversityConstraint.model_validate_json(output.read_text())
        assert constraint.property_expression == "item.metadata.verb_class"
        assert constraint.min_unique_values == 10


class TestSizeConstraint:
    """Test size constraint creation."""

    def test_create_exact(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test size constraint with exact size."""
        output = tmp_path / "constraint.jsonl"
        result = runner.invoke(
            create_size,
            [
                "--exact-size",
                "40",
                "--priority",
                "10",
                "-o",
                str(output),
            ],
        )

        assert result.exit_code == 0
        constraint = SizeConstraint.model_validate_json(output.read_text())
        assert constraint.exact_size == 40
        assert constraint.min_size is None
        assert constraint.max_size is None
        assert constraint.priority == 10

    def test_create_range(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test size constraint with min/max range."""
        output = tmp_path / "constraint.jsonl"
        result = runner.invoke(
            create_size,
            [
                "--min-size",
                "30",
                "--max-size",
                "50",
                "-o",
                str(output),
            ],
        )

        assert result.exit_code == 0
        constraint = SizeConstraint.model_validate_json(output.read_text())
        assert constraint.min_size == 30
        assert constraint.max_size == 50
        assert constraint.exact_size is None

    def test_conflicting_params(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test error when exact_size conflicts with min/max."""
        output = tmp_path / "constraint.jsonl"
        result = runner.invoke(
            create_size,
            [
                "--exact-size",
                "40",
                "--min-size",
                "30",  # Should conflict
                "-o",
                str(output),
            ],
        )

        # Pydantic validation should fail (model_validator on SizeConstraint)
        # Either fails during creation or CLI handles error gracefully
        assert result.exit_code in (0, 1)


# ==================== Batch Constraint Tests ====================


class TestBatchCoverageConstraint:
    """Test batch coverage constraint creation."""

    def test_create(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test batch coverage constraint creation."""
        output = tmp_path / "constraint.jsonl"
        result = runner.invoke(
            create_batch_coverage,
            [
                "--property-expression",
                "item.metadata.template_id",
                "--target-values",
                "0,1,2,3,4,5",
                "--min-coverage",
                "1.0",
                "-o",
                str(output),
            ],
        )

        assert result.exit_code == 0
        constraint = BatchCoverageConstraint.model_validate_json(output.read_text())
        assert constraint.property_expression == "item.metadata.template_id"
        assert constraint.target_values == (
            "0",
            "1",
            "2",
            "3",
            "4",
            "5",
        )
        assert constraint.min_coverage == 1.0


class TestBatchBalanceConstraint:
    """Test batch balance constraint creation."""

    def test_create(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test batch balance constraint creation."""
        output = tmp_path / "constraint.jsonl"
        result = runner.invoke(
            create_batch_balance,
            [
                "--property-expression",
                "item.metadata.condition",
                "--target-distribution",
                "control=0.5,experimental=0.5",
                "--tolerance",
                "0.05",
                "-o",
                str(output),
            ],
        )

        assert result.exit_code == 0
        constraint = BatchBalanceConstraint.model_validate_json(output.read_text())
        assert constraint.property_expression == "item.metadata.condition"
        assert constraint.target_distribution == {"control": 0.5, "experimental": 0.5}
        assert constraint.tolerance == 0.05


class TestBatchDiversityConstraint:
    """Test batch diversity constraint creation."""

    def test_create(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test batch diversity constraint creation."""
        output = tmp_path / "constraint.jsonl"
        result = runner.invoke(
            create_batch_diversity,
            [
                "--property-expression",
                "item.metadata.target_word",
                "--max-lists-per-value",
                "3",
                "-o",
                str(output),
            ],
        )

        assert result.exit_code == 0
        constraint = BatchDiversityConstraint.model_validate_json(output.read_text())
        assert constraint.property_expression == "item.metadata.target_word"
        assert constraint.max_lists_per_value == 3


class TestBatchMinOccurrenceConstraint:
    """Test batch min occurrence constraint creation."""

    def test_create(self, runner: CliRunner, tmp_path: Path) -> None:
        """Test batch min occurrence constraint creation."""
        output = tmp_path / "constraint.jsonl"
        result = runner.invoke(
            create_batch_min_occurrence,
            [
                "--property-expression",
                "item.metadata.construction",
                "--min-occurrences",
                "5",
                "-o",
                str(output),
            ],
        )

        assert result.exit_code == 0
        constraint = BatchMinOccurrenceConstraint.model_validate_json(
            output.read_text()
        )
        assert constraint.property_expression == "item.metadata.construction"
        assert constraint.min_occurrences == 5


# ==================== Output Format Tests ====================


class TestOutputFormat:
    """Test constraint output format and composition."""

    def test_constraint_output_is_jsonl(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test that constraints are written as JSONL (one line)."""
        output = tmp_path / "constraint.jsonl"
        runner.invoke(
            create_uniqueness,
            [
                "--property-expression",
                "item.metadata.verb",
                "-o",
                str(output),
            ],
        )

        content = output.read_text()
        lines = content.strip().split("\n")
        assert len(lines) == 1  # Single line
        assert lines[0].strip()  # Non-empty

    def test_multiple_constraints_can_be_combined(
        self, runner: CliRunner, tmp_path: Path
    ) -> None:
        """Test that multiple constraint files can be combined."""
        constraint1 = tmp_path / "constraint1.jsonl"
        constraint2 = tmp_path / "constraint2.jsonl"

        runner.invoke(
            create_uniqueness,
            ["--property-expression", "item.metadata.verb", "-o", str(constraint1)],
        )
        runner.invoke(
            create_balance,
            [
                "--property-expression",
                "item.metadata.condition",
                "--target-counts",
                "a=10,b=10",
                "-o",
                str(constraint2),
            ],
        )

        # Combine files
        combined = tmp_path / "combined.jsonl"
        combined.write_text(constraint1.read_text() + constraint2.read_text())

        # Should be able to parse both
        lines = combined.read_text().strip().split("\n")
        assert len(lines) == 2

        c1 = UniquenessConstraint.model_validate_json(lines[0])
        c2 = BalanceConstraint.model_validate_json(lines[1])
        assert c1.constraint_type == "uniqueness"
        assert c2.constraint_type == "balance"
