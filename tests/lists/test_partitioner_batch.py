"""Tests for partitioner with batch constraints."""

from __future__ import annotations

from uuid import uuid4

from bead.lists.constraints import (
    BatchBalanceConstraint,
    BatchCoverageConstraint,
    BatchDiversityConstraint,
    BatchMinOccurrenceConstraint,
)
from bead.lists.partitioner import ListPartitioner


class TestPartitionWithBatchConstraints:
    """Tests for partition_with_batch_constraints method."""

    def test_partition_with_no_batch_constraints(self) -> None:
        """Test partitioning with no batch constraints returns normal partitioning."""
        partitioner = ListPartitioner(random_seed=42)
        items = [uuid4() for _ in range(100)]
        metadata = {uid: {"template_id": i % 10} for i, uid in enumerate(items)}

        lists = partitioner.partition_with_batch_constraints(
            items=items,
            n_lists=5,
            batch_constraints=None,
            metadata=metadata,
        )

        assert len(lists) == 5
        # Check total items match
        total_items = sum(len(lst.item_refs) for lst in lists)
        assert total_items == 100

    def test_partition_with_coverage_constraint(self) -> None:
        """Test partitioning satisfies coverage constraint."""
        partitioner = ListPartitioner(random_seed=42)
        items = [uuid4() for _ in range(100)]
        # Create items with template_ids 0-9, but sparse coverage
        metadata = {uid: {"template_id": i % 10} for i, uid in enumerate(items)}

        constraint = BatchCoverageConstraint(
            constraint_type="coverage",
            property_expression="item['template_id']",
            target_values=list(range(10)),
            min_coverage=1.0,
        )

        lists = partitioner.partition_with_batch_constraints(
            items=items,
            n_lists=4,
            batch_constraints=[constraint],
            metadata=metadata,
        )

        # Check all templates are covered
        all_templates = set()
        for lst in lists:
            for item_id in lst.item_refs:
                all_templates.add(metadata[item_id]["template_id"])

        assert len(all_templates) == 10

    def test_partition_with_balance_constraint(self) -> None:
        """Test partitioning satisfies balance constraint."""
        partitioner = ListPartitioner(random_seed=42)
        items = [uuid4() for _ in range(100)]
        # Create 50/50 split
        metadata = {
            uid: {"pair_type": "same" if i < 50 else "different"}
            for i, uid in enumerate(items)
        }

        constraint = BatchBalanceConstraint(
            constraint_type="balance",
            property_expression="item['pair_type']",
            target_distribution={"same": 0.5, "different": 0.5},
            tolerance=0.05,
        )

        lists = partitioner.partition_with_batch_constraints(
            items=items,
            n_lists=4,
            batch_constraints=[constraint],
            metadata=metadata,
            tolerance=0.05,
        )

        # Check distribution
        counts = {"same": 0, "different": 0}
        for lst in lists:
            for item_id in lst.item_refs:
                counts[metadata[item_id]["pair_type"]] += 1

        total = sum(counts.values())
        assert abs(counts["same"] / total - 0.5) <= 0.05
        assert abs(counts["different"] / total - 0.5) <= 0.05

    def test_partition_with_diversity_constraint(self) -> None:
        """Test partitioning attempts to satisfy diversity constraint."""
        partitioner = ListPartitioner(random_seed=42)
        items = [uuid4() for _ in range(80)]
        # Create 8 verbs, each appearing 10 times
        metadata = {uid: {"verb": f"verb_{i % 8}"} for i, uid in enumerate(items)}

        constraint = BatchDiversityConstraint(
            constraint_type="diversity",
            property_expression="item['verb']",
            max_lists_per_value=4,  # Max half the lists
        )

        lists = partitioner.partition_with_batch_constraints(
            items=items,
            n_lists=8,
            batch_constraints=[constraint],
            metadata=metadata,
            max_iterations=500,
        )

        # Check diversity constraint is being evaluated (score should be computed)
        score = partitioner._compute_batch_diversity_score(lists, constraint, metadata)
        # Score should be reasonable (algorithm attempts to improve it)
        assert 0.0 <= score <= 1.0

        # Check total items match
        total_items = sum(len(lst.item_refs) for lst in lists)
        assert total_items == 80

    def test_partition_with_min_occurrence_constraint(self) -> None:
        """Test partitioning satisfies min occurrence constraint."""
        partitioner = ListPartitioner(random_seed=42)
        items = [uuid4() for _ in range(120)]
        # Create 3 quantiles, 40 items each
        metadata = {uid: {"quantile": i % 3} for i, uid in enumerate(items)}

        constraint = BatchMinOccurrenceConstraint(
            constraint_type="min_occurrence",
            property_expression="item['quantile']",
            min_occurrences=30,  # At least 30 of each quantile
        )

        lists = partitioner.partition_with_batch_constraints(
            items=items,
            n_lists=4,
            batch_constraints=[constraint],
            metadata=metadata,
        )

        # Count occurrences of each quantile
        quantile_counts = {0: 0, 1: 0, 2: 0}
        for lst in lists:
            for item_id in lst.item_refs:
                quantile_counts[metadata[item_id]["quantile"]] += 1

        for _quantile, count in quantile_counts.items():
            assert count >= 30

    def test_partition_with_multiple_batch_constraints(self) -> None:
        """Test partitioning with multiple batch constraints."""
        partitioner = ListPartitioner(random_seed=42)
        items = [uuid4() for _ in range(100)]
        metadata = {
            uid: {
                "template_id": i % 10,
                "pair_type": "same" if i < 50 else "different",
            }
            for i, uid in enumerate(items)
        }

        coverage = BatchCoverageConstraint(
            constraint_type="coverage",
            property_expression="item['template_id']",
            target_values=list(range(10)),
            min_coverage=1.0,
        )
        balance = BatchBalanceConstraint(
            constraint_type="balance",
            property_expression="item['pair_type']",
            target_distribution={"same": 0.5, "different": 0.5},
            tolerance=0.1,
        )

        lists = partitioner.partition_with_batch_constraints(
            items=items,
            n_lists=4,
            batch_constraints=[coverage, balance],
            metadata=metadata,
            max_iterations=500,
        )

        # Check coverage
        all_templates = set()
        for lst in lists:
            for item_id in lst.item_refs:
                all_templates.add(metadata[item_id]["template_id"])
        assert len(all_templates) == 10

        # Check balance
        counts = {"same": 0, "different": 0}
        for lst in lists:
            for item_id in lst.item_refs:
                counts[metadata[item_id]["pair_type"]] += 1
        total = sum(counts.values())
        assert abs(counts["same"] / total - 0.5) <= 0.1

    def test_partition_with_max_iterations(self) -> None:
        """Test max_iterations parameter is respected."""
        partitioner = ListPartitioner(random_seed=42)
        items = [uuid4() for _ in range(100)]
        metadata = {uid: {"template_id": i % 10} for i, uid in enumerate(items)}

        constraint = BatchCoverageConstraint(
            constraint_type="coverage",
            property_expression="item['template_id']",
            target_values=list(range(10)),
            min_coverage=1.0,
        )

        lists = partitioner.partition_with_batch_constraints(
            items=items,
            n_lists=4,
            batch_constraints=[constraint],
            metadata=metadata,
            max_iterations=10,  # Very low limit
        )

        # Should still return valid lists even if not converged
        assert len(lists) == 4
        total_items = sum(len(lst.item_refs) for lst in lists)
        assert total_items == 100

    def test_partition_with_tolerance_parameter(self) -> None:
        """Test tolerance parameter affects constraint satisfaction."""
        partitioner = ListPartitioner(random_seed=42)
        items = [uuid4() for _ in range(100)]
        metadata = {
            uid: {"pair_type": "same" if i < 50 else "different"}
            for i, uid in enumerate(items)
        }

        constraint = BatchBalanceConstraint(
            constraint_type="balance",
            property_expression="item['pair_type']",
            target_distribution={"same": 0.5, "different": 0.5},
            tolerance=0.1,
        )

        lists = partitioner.partition_with_batch_constraints(
            items=items,
            n_lists=4,
            batch_constraints=[constraint],
            metadata=metadata,
            tolerance=0.2,  # Looser tolerance
        )

        assert len(lists) == 4

    def test_partition_empty_items(self) -> None:
        """Test partitioning with empty items list."""
        partitioner = ListPartitioner(random_seed=42)
        items: list = []
        metadata: dict = {}

        constraint = BatchCoverageConstraint(
            constraint_type="coverage",
            property_expression="item['template_id']",
            target_values=list(range(10)),
        )

        lists = partitioner.partition_with_batch_constraints(
            items=items,
            n_lists=4,
            batch_constraints=[constraint],
            metadata=metadata,
        )

        assert len(lists) == 4
        for lst in lists:
            assert len(lst.item_refs) == 0

    def test_partition_single_list(self) -> None:
        """Test partitioning with n_lists=1."""
        partitioner = ListPartitioner(random_seed=42)
        items = [uuid4() for _ in range(50)]
        metadata = {uid: {"template_id": i % 5} for i, uid in enumerate(items)}

        constraint = BatchCoverageConstraint(
            constraint_type="coverage",
            property_expression="item['template_id']",
            target_values=list(range(5)),
        )

        lists = partitioner.partition_with_batch_constraints(
            items=items,
            n_lists=1,
            batch_constraints=[constraint],
            metadata=metadata,
        )

        assert len(lists) == 1
        assert len(lists[0].item_refs) == 50

    def test_partition_with_strategy_parameter(self) -> None:
        """Test different partitioning strategies work with batch constraints."""
        partitioner = ListPartitioner(random_seed=42)
        items = [uuid4() for _ in range(100)]
        metadata = {uid: {"template_id": i % 10} for i, uid in enumerate(items)}

        constraint = BatchCoverageConstraint(
            constraint_type="coverage",
            property_expression="item['template_id']",
            target_values=list(range(10)),
        )

        for strategy in ["balanced", "random"]:
            lists = partitioner.partition_with_batch_constraints(
                items=items,
                n_lists=4,
                batch_constraints=[constraint],
                metadata=metadata,
                strategy=strategy,
            )
            assert len(lists) == 4


class TestComputeBatchConstraintScore:
    """Tests for _compute_batch_constraint_score method."""

    def test_compute_coverage_score_full_coverage(self) -> None:
        """Test coverage score with full coverage."""
        partitioner = ListPartitioner(random_seed=42)
        items = [uuid4() for _ in range(100)]
        metadata = {uid: {"template_id": i % 10} for i, uid in enumerate(items)}

        lists = partitioner.partition(
            items=items, n_lists=4, constraints=[], metadata=metadata
        )

        constraint = BatchCoverageConstraint(
            constraint_type="coverage",
            property_expression="item['template_id']",
            target_values=list(range(10)),
        )

        score = partitioner._compute_batch_coverage_score(lists, constraint, metadata)
        assert score == 1.0

    def test_compute_coverage_score_partial_coverage(self) -> None:
        """Test coverage score with partial coverage."""
        partitioner = ListPartitioner(random_seed=42)
        items = [uuid4() for _ in range(50)]
        # Only covers 0-4 (5 out of 10)
        metadata = {uid: {"template_id": i % 5} for i, uid in enumerate(items)}

        lists = partitioner.partition(
            items=items, n_lists=4, constraints=[], metadata=metadata
        )

        constraint = BatchCoverageConstraint(
            constraint_type="coverage",
            property_expression="item['template_id']",
            target_values=list(range(10)),
        )

        score = partitioner._compute_batch_coverage_score(lists, constraint, metadata)
        assert score == 0.5  # 5 out of 10

    def test_compute_balance_score_perfect(self) -> None:
        """Test balance score with perfect balance."""
        partitioner = ListPartitioner(random_seed=42)
        items = [uuid4() for _ in range(100)]
        metadata = {
            uid: {"pair_type": "same" if i < 50 else "different"}
            for i, uid in enumerate(items)
        }

        lists = partitioner.partition(
            items=items, n_lists=4, constraints=[], metadata=metadata
        )

        constraint = BatchBalanceConstraint(
            constraint_type="balance",
            property_expression="item['pair_type']",
            target_distribution={"same": 0.5, "different": 0.5},
        )

        score = partitioner._compute_batch_balance_score(lists, constraint, metadata)
        assert score >= 0.95  # Should be very close to 1.0

    def test_compute_diversity_score_satisfied(self) -> None:
        """Test diversity score computation."""
        partitioner = ListPartitioner(random_seed=42)
        items = [uuid4() for _ in range(40)]
        # 4 lists, 2 verbs
        metadata = {uid: {"verb": f"verb_{i % 2}"} for i, uid in enumerate(items)}

        lists = partitioner.partition(
            items=items, n_lists=4, constraints=[], metadata=metadata
        )

        constraint = BatchDiversityConstraint(
            constraint_type="diversity",
            property_expression="item['verb']",
            max_lists_per_value=3,
        )

        score = partitioner._compute_batch_diversity_score(lists, constraint, metadata)
        # Score should be in valid range
        assert 0.0 <= score <= 1.0

    def test_compute_min_occurrence_score_satisfied(self) -> None:
        """Test min occurrence score when constraint is satisfied."""
        partitioner = ListPartitioner(random_seed=42)
        items = [uuid4() for _ in range(120)]
        # 3 quantiles, 40 each
        metadata = {uid: {"quantile": i % 3} for i, uid in enumerate(items)}

        lists = partitioner.partition(
            items=items, n_lists=4, constraints=[], metadata=metadata
        )

        constraint = BatchMinOccurrenceConstraint(
            constraint_type="min_occurrence",
            property_expression="item['quantile']",
            min_occurrences=30,
        )

        score = partitioner._compute_batch_min_occurrence_score(
            lists, constraint, metadata
        )
        # Score should be at least 1.0 (40/30 = 1.33, clipped to 1.0)
        assert score == 1.0

    def test_compute_min_occurrence_score_below_minimum(self) -> None:
        """Test min occurrence score when below minimum."""
        partitioner = ListPartitioner(random_seed=42)
        items = [uuid4() for _ in range(60)]
        # 3 quantiles, 20 each
        metadata = {uid: {"quantile": i % 3} for i, uid in enumerate(items)}

        lists = partitioner.partition(
            items=items, n_lists=4, constraints=[], metadata=metadata
        )

        constraint = BatchMinOccurrenceConstraint(
            constraint_type="min_occurrence",
            property_expression="item['quantile']",
            min_occurrences=30,
        )

        score = partitioner._compute_batch_min_occurrence_score(
            lists, constraint, metadata
        )
        # Score should be 20/30 = 0.667
        assert abs(score - 0.667) < 0.01


class TestImproveBatchConstraint:
    """Tests for _improve_batch_constraint method."""

    def test_improve_coverage_constraint(self) -> None:
        """Test improving coverage constraint through swaps."""
        partitioner = ListPartitioner(random_seed=42)
        items = [uuid4() for _ in range(100)]
        # Skewed distribution
        metadata = {uid: {"template_id": i % 5} for i, uid in enumerate(items)}

        lists = partitioner.partition(
            items=items, n_lists=4, constraints=[], metadata=metadata
        )

        constraint = BatchCoverageConstraint(
            constraint_type="coverage",
            property_expression="item['template_id']",
            target_values=list(range(10)),
        )

        # Try to improve (should succeed or fail gracefully)
        improved = partitioner._improve_batch_constraint(
            lists=lists,
            constraint=constraint,
            list_constraints=[],
            batch_constraints=[constraint],
            metadata=metadata,
            n_attempts=50,
        )

        # Just check it doesn't crash
        assert isinstance(improved, bool)

    def test_improve_with_insufficient_lists(self) -> None:
        """Test improve returns False with insufficient lists."""
        partitioner = ListPartitioner(random_seed=42)
        items = [uuid4() for _ in range(50)]
        metadata = {uid: {"template_id": i % 5} for i, uid in enumerate(items)}

        lists = partitioner.partition(
            items=items, n_lists=1, constraints=[], metadata=metadata
        )

        constraint = BatchCoverageConstraint(
            constraint_type="coverage",
            property_expression="item['template_id']",
            target_values=list(range(10)),
        )

        improved = partitioner._improve_batch_constraint(
            lists=lists,
            constraint=constraint,
            list_constraints=[],
            batch_constraints=[constraint],
            metadata=metadata,
        )

        assert improved is False
