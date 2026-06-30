"""List partitioning for experimental item distribution.

This module provides the ListPartitioner class for partitioning items into
balanced experimental lists. Implements three strategies: random, balanced,
and stratified. Uses stand-off annotation (works with UUIDs only).
"""

from __future__ import annotations

from collections import Counter, defaultdict
from typing import Any
from uuid import UUID

import numpy as np

from bead.dsl.evaluator import DSLEvaluator
from bead.lists.balancer import QuantileBalancer
from bead.lists.constraints import (
    BalanceConstraint,
    BatchBalanceConstraint,
    BatchConstraint,
    BatchCoverageConstraint,
    BatchDiversityConstraint,
    BatchMinOccurrenceConstraint,
    ListConstraint,
    QuantileConstraint,
    SizeConstraint,
    UniquenessConstraint,
)
from bead.lists.experiment_list import ExperimentList, MetadataValue
from bead.resources.constraints import ContextValue

# Type aliases for clarity
type ItemMetadata = dict[str, Any]  # Arbitrary item properties
type MetadataDict = dict[UUID, ItemMetadata]  # Metadata indexed by UUID
# BalanceMetrics is the structural value type that
# ``ExperimentList.balance_metrics`` accepts. The alias is the same shape as
# the field declaration so that ``with_(balance_metrics=...)`` type-checks
# without dict-invariance noise.
type BalanceMetrics = dict[str, "MetadataValue"]


class ListPartitioner:
    """Partitions items into balanced experimental lists.

    Uses stand-off annotation: only stores UUIDs, not full item objects.
    Requires item metadata dict for constraint checking and balancing.

    Implements three partitioning strategies:
    - Random: Simple round-robin after shuffling
    - Balanced: Greedy algorithm to minimize constraint violations
    - Stratified: Quantile-based stratification with balanced distribution

    Parameters
    ----------
    random_seed : int | None, default=None
        Random seed for reproducibility.

    Attributes
    ----------
    random_seed : int | None
        Random seed for reproducibility.

    Examples
    --------
    >>> from uuid import uuid4
    >>> partitioner = ListPartitioner(random_seed=42)
    >>> items = [uuid4() for _ in range(100)]
    >>> metadata = {uid: {"property": i} for i, uid in enumerate(items)}
    >>> lists = partitioner.partition(items, n_lists=5, metadata=metadata)
    >>> len(lists)
    5
    """

    def __init__(self, random_seed: int | None = None) -> None:
        self.random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)
        self.dsl_evaluator = DSLEvaluator()

    def partition(
        self,
        items: list[UUID],
        n_lists: int,
        constraints: list[ListConstraint] | None = None,
        strategy: str = "balanced",
        metadata: MetadataDict | None = None,
    ) -> list[ExperimentList]:
        """Partition items into lists.

        Parameters
        ----------
        items : list[UUID]
            Item UUIDs to partition.
        n_lists : int
            Number of lists to create.
        constraints : list[ListConstraint] | None, default=None
            Constraints to satisfy.
        strategy : str, default="balanced"
            Partitioning strategy ("balanced", "random", "stratified").
        metadata : dict[UUID, dict[str, Any]] | None, default=None
            Metadata for each item UUID. Required for constraint checking.

        Returns
        -------
        list[ExperimentList]
            The partitioned lists.

        Raises
        ------
        ValueError
            If strategy is unknown or n_lists < 1.
        """
        if n_lists < 1:
            raise ValueError(f"n_lists must be >= 1, got {n_lists}")

        constraints = constraints or []
        metadata = metadata or {}

        # Select partitioning method based on strategy
        match strategy:
            case "balanced":
                return self._partition_balanced(items, n_lists, constraints, metadata)
            case "random":
                return self._partition_random(items, n_lists, constraints, metadata)
            case "stratified":
                return self._partition_stratified(items, n_lists, constraints, metadata)
            case _:
                raise ValueError(f"Unknown strategy: {strategy}")

    def _partition_random(
        self,
        items: list[UUID],
        n_lists: int,
        constraints: list[ListConstraint],
        metadata: MetadataDict,
    ) -> list[ExperimentList]:
        """Partition items randomly.

        Parameters
        ----------
        items : list[UUID]
            Items to partition.
        n_lists : int
            Number of lists.
        constraints : list[ListConstraint]
            Constraints to attach to lists.
        metadata : dict[UUID, dict[str, Any]]
            Item metadata.

        Returns
        -------
        list[ExperimentList]
            Partitioned lists.
        """
        # Initialize lists
        lists = [
            ExperimentList(
                name=f"list_{i}",
                list_number=i,
                list_constraints=tuple(constraints),
            )
            for i in range(n_lists)
        ]

        # Shuffle and distribute round robin
        items_shuffled = np.array(items)
        self._rng.shuffle(items_shuffled)

        for i, item_id in enumerate(items_shuffled):
            list_idx = i % n_lists
            lists[list_idx] = lists[list_idx].with_item(item_id)

        return [
            exp_list.with_(
                balance_metrics=self._compute_balance_metrics(
                    exp_list, constraints, metadata
                )
            )
            for exp_list in lists
        ]

    def _partition_balanced(
        self,
        items: list[UUID],
        n_lists: int,
        constraints: list[ListConstraint],
        metadata: MetadataDict,
    ) -> list[ExperimentList]:
        """Partition items with balanced distribution.

        Uses greedy algorithm to distribute items to minimize imbalance.

        Parameters
        ----------
        items : list[UUID]
            Items to partition.
        n_lists : int
            Number of lists.
        constraints : list[ListConstraint]
            Constraints to satisfy.
        metadata : dict[UUID, dict[str, Any]]
            Item metadata.

        Returns
        -------
        list[ExperimentList]
            Partitioned lists.
        """
        # Initialize lists
        lists = [
            ExperimentList(
                name=f"list_{i}",
                list_number=i,
                list_constraints=tuple(constraints),
            )
            for i in range(n_lists)
        ]

        # Shuffle items
        items_shuffled = np.array(items)
        self._rng.shuffle(items_shuffled)

        # For each item, assign to list that best maintains balance
        for item_id in items_shuffled:
            best_idx = self._find_best_list_index(item_id, lists, constraints, metadata)
            lists[best_idx] = lists[best_idx].with_item(item_id)

        return [
            exp_list.with_(
                balance_metrics=self._compute_balance_metrics(
                    exp_list, constraints, metadata
                )
            )
            for exp_list in lists
        ]

    def _partition_stratified(
        self,
        items: list[UUID],
        n_lists: int,
        constraints: list[ListConstraint],
        metadata: MetadataDict,
    ) -> list[ExperimentList]:
        """Partition items with stratification.

        Creates strata based on quantile constraints and distributes
        items from each stratum across lists.

        Parameters
        ----------
        items : list[UUID]
            Items to partition.
        n_lists : int
            Number of lists.
        constraints : list[ListConstraint]
            Constraints to satisfy (must include quantile constraints).
        metadata : dict[UUID, dict[str, Any]]
            Item metadata.

        Returns
        -------
        list[ExperimentList]
            Partitioned lists.
        """
        # Find quantile constraints
        quantile_constraints = [
            c for c in constraints if isinstance(c, QuantileConstraint)
        ]

        if not quantile_constraints:
            # Fall back to balanced
            return self._partition_balanced(items, n_lists, constraints, metadata)

        # Use first quantile constraint for stratification
        qc = quantile_constraints[0]

        # Create balancer
        balancer = QuantileBalancer(
            n_quantiles=qc.n_quantiles, random_seed=self.random_seed
        )

        # Create value function
        def value_func(item_id: UUID) -> float:
            return float(
                self._extract_property_value(
                    item_id, qc.property_expression, qc.context, metadata
                )
            )

        # Balance items across lists
        balanced_lists = balancer.balance(
            items, value_func, n_lists, qc.items_per_quantile
        )

        # Convert to ExperimentList objects
        lists: list[ExperimentList] = []
        for i, item_ids in enumerate(balanced_lists):
            exp_list = ExperimentList(
                name=f"list_{i}",
                list_number=i,
                list_constraints=tuple(constraints),
                item_refs=tuple(item_ids),
            )
            exp_list = exp_list.with_(
                balance_metrics=self._compute_balance_metrics(
                    exp_list, constraints, metadata
                )
            )
            lists.append(exp_list)

        return lists

    def _find_best_list_index(
        self,
        item_id: UUID,
        lists: list[ExperimentList],
        constraints: list[ListConstraint],
        metadata: MetadataDict,
    ) -> int:
        """Return the index of the list that best maintains balance after the add.

        Scores each list by ``(violations_after_adding, current_size)`` and
        picks the lowest.
        """
        scores: list[tuple[int, int]] = []
        for exp_list in lists:
            hypothetical = exp_list.with_item(item_id)
            violations = self._count_violations(hypothetical, constraints, metadata)
            scores.append((violations, len(exp_list.item_refs)))

        return int(np.argmin([s[0] * 1000 + s[1] for s in scores]))

    def _count_violations(
        self,
        exp_list: ExperimentList,
        constraints: list[ListConstraint],
        metadata: MetadataDict,
    ) -> int:
        """Count constraint violations for a list.

        Violations are weighted by constraint priority. Higher priority
        constraints contribute more to the total violation score.

        Parameters
        ----------
        exp_list : ExperimentList
            The list to check.
        constraints : list[ListConstraint]
            Constraints to check.
        metadata : MetadataDict
            Item metadata.

        Returns
        -------
        int
            Weighted violation score (sum of priorities of violated constraints).
        """
        violations = 0

        for constraint in constraints:
            is_violated = False

            if isinstance(constraint, UniquenessConstraint):
                if not self._check_uniqueness(exp_list, constraint, metadata):
                    is_violated = True
            elif isinstance(constraint, BalanceConstraint):
                if not self._check_balance(exp_list, constraint, metadata):
                    is_violated = True
            elif isinstance(constraint, SizeConstraint):
                if not self._check_size(exp_list, constraint):
                    is_violated = True

            if is_violated:
                priority = constraint.priority
                assert isinstance(priority, int)
                violations += priority

        return violations

    def _check_uniqueness(
        self,
        exp_list: ExperimentList,
        constraint: UniquenessConstraint,
        metadata: MetadataDict,
    ) -> bool:
        """Check uniqueness constraint.

        Parameters
        ----------
        exp_list : ExperimentList
            List to check.
        constraint : UniquenessConstraint
            Uniqueness constraint.
        metadata : dict[UUID, dict[str, Any]]
            Item metadata.

        Returns
        -------
        bool
            True if constraint is satisfied.
        """
        # Get values for property
        values: list[Any] = []
        for item_id in exp_list.item_refs:
            value = self._extract_property_value(
                item_id, constraint.property_expression, constraint.context, metadata
            )
            values.append(value)

        # Check for duplicates
        if constraint.allow_null:
            values = [v for v in values if v is not None]

        return bool(len(values) == len(set(values)))

    def _check_balance(
        self,
        exp_list: ExperimentList,
        constraint: BalanceConstraint,
        metadata: MetadataDict,
    ) -> bool:
        """Check balance constraint.

        Parameters
        ----------
        exp_list : ExperimentList
            List to check.
        constraint : BalanceConstraint
            Balance constraint.
        metadata : dict[UUID, dict[str, Any]]
            Item metadata.

        Returns
        -------
        bool
            True if constraint is satisfied.
        """
        # Get values for property
        values: list[Any] = []
        for item_id in exp_list.item_refs:
            value = self._extract_property_value(
                item_id, constraint.property_expression, constraint.context, metadata
            )
            values.append(value)

        # Count occurrences
        counts = Counter(values)

        # Check against target counts if specified
        if constraint.target_counts is not None:
            for category, target_count in constraint.target_counts.items():
                actual_count = counts.get(category, 0)
                deviation = abs(actual_count - target_count) / max(target_count, 1)
                if deviation > constraint.tolerance:
                    return False
            return True

        # Otherwise check for balanced distribution
        if len(counts) == 0:
            return True

        count_values = list(counts.values())
        mean_count = np.mean(count_values)
        max_deviation = max(
            abs(c - mean_count) / max(mean_count, 1) for c in count_values
        )

        return bool(max_deviation <= constraint.tolerance)

    def _check_size(self, exp_list: ExperimentList, constraint: SizeConstraint) -> bool:
        """Check size constraint.

        Parameters
        ----------
        exp_list : ExperimentList
            List to check.
        constraint : SizeConstraint
            Size constraint.

        Returns
        -------
        bool
            True if constraint is satisfied.
        """
        size = len(exp_list.item_refs)

        if constraint.exact_size is not None:
            return size == constraint.exact_size

        if constraint.min_size is not None and size < constraint.min_size:
            return False

        if constraint.max_size is not None and size > constraint.max_size:
            return False

        return True

    def _extract_property_value(
        self,
        item_id: UUID,
        property_expression: str,
        context: dict[str, ContextValue] | None,
        metadata: MetadataDict,
    ) -> Any:
        """Extract property value using DSL expression.

        Parameters
        ----------
        item_id : UUID
            Item UUID.
        property_expression : str
            DSL expression using dict access syntax (e.g., "item['lm_prob']",
            "variance([item['val1'], item['val2']])"). The 'item' variable
            refers to the metadata dict for this item.
        context : dict[str, ContextValue] | None
            Additional context variables for evaluation.
        metadata : dict[UUID, dict[str, Any]]
            Metadata dict mapping item UUIDs to their metadata.

        Returns
        -------
        Any
            Evaluated property value.

        Raises
        ------
        KeyError
            If item_id not in metadata.

        Notes
        -----
        Since ListPartitioner uses stand-off annotation (UUIDs only, not full
        Item objects), the 'item' variable in property expressions refers to
        the item's metadata dict, not a full Item object. Use dict access
        syntax: item['key'] rather than item.key.
        """
        if item_id not in metadata:
            raise KeyError(f"Item {item_id} not found in metadata")

        # Build evaluation context with item metadata directly
        # The metadata dict IS the item for property expression purposes
        eval_context: dict[str, Any] = {"item": metadata[item_id]}
        if context:
            eval_context.update(context)

        return self.dsl_evaluator.evaluate(property_expression, eval_context)

    def _compute_balance_metrics(
        self,
        exp_list: ExperimentList,
        constraints: list[ListConstraint],
        metadata: MetadataDict,
    ) -> BalanceMetrics:
        """Compute balance metrics for a list.

        Parameters
        ----------
        exp_list : ExperimentList
            The list.
        constraints : list[ListConstraint]
            Constraints to compute metrics for.
        metadata : dict[UUID, dict[str, Any]]
            Item metadata.

        Returns
        -------
        dict[str, Any]
            Balance metrics.
        """
        metrics: dict[str, Any] = {}

        # Compute metrics for each constraint
        for constraint in constraints:
            if isinstance(constraint, QuantileConstraint):
                metrics[f"quantile_{constraint.property_expression}"] = (
                    self._compute_quantile_distribution(exp_list, constraint, metadata)
                )
            elif isinstance(constraint, BalanceConstraint):
                metrics[f"balance_{constraint.property_expression}"] = (
                    self._compute_category_distribution(exp_list, constraint, metadata)
                )

        # Overall size
        metrics["size"] = len(exp_list.item_refs)

        return metrics

    def _compute_quantile_distribution(
        self,
        exp_list: ExperimentList,
        constraint: QuantileConstraint,
        metadata: MetadataDict,
    ) -> dict[str, float | list[float]]:
        """Compute distribution across quantiles.

        Parameters
        ----------
        exp_list : ExperimentList
            The list.
        constraint : QuantileConstraint
            Quantile constraint.
        metadata : dict[UUID, dict[str, Any]]
            Item metadata.

        Returns
        -------
        dict[str, Any]
            Distribution metrics.
        """
        if not exp_list.item_refs:
            return {
                "mean": 0.0,
                "std": 0.0,
                "min": 0.0,
                "max": 0.0,
                "quantiles": [],
            }

        values = [
            float(
                self._extract_property_value(
                    item_id,
                    constraint.property_expression,
                    constraint.context,
                    metadata,
                )
            )
            for item_id in exp_list.item_refs
        ]

        # percentile with list input returns array
        percentiles: np.ndarray[Any, np.dtype[np.floating[Any]]] = np.percentile(
            values, [25, 50, 75]
        )
        # min/max with array input returns scalar
        min_val: np.floating[Any] = np.min(values)
        max_val: np.floating[Any] = np.max(values)

        return {
            "mean": float(np.mean(values)),
            "std": float(np.std(values)),
            "min": float(min_val),
            "max": float(max_val),
            "quantiles": [float(q) for q in percentiles],
        }

    def _compute_category_distribution(
        self,
        exp_list: ExperimentList,
        constraint: BalanceConstraint,
        metadata: MetadataDict,
    ) -> dict[str, dict[str, int] | int | tuple[Any, int] | None]:
        """Compute distribution across categories.

        Parameters
        ----------
        exp_list : ExperimentList
            The list.
        constraint : BalanceConstraint
            Balance constraint.
        metadata : dict[UUID, dict[str, Any]]
            Item metadata.

        Returns
        -------
        dict[str, Any]
            Distribution metrics.
        """
        if not exp_list.item_refs:
            return {"counts": {}, "n_categories": 0, "most_common": None}

        values = [
            self._extract_property_value(
                item_id, constraint.property_expression, constraint.context, metadata
            )
            for item_id in exp_list.item_refs
        ]
        counts = Counter(values)

        return {
            "counts": dict(counts),
            "n_categories": len(counts),
            "most_common": counts.most_common(1)[0] if counts else None,
        }

    # ========================================================================
    # Batch Constraint Methods
    # ========================================================================

    def partition_with_batch_constraints(
        self,
        items: list[UUID],
        n_lists: int,
        list_constraints: list[ListConstraint] | None = None,
        batch_constraints: list[BatchConstraint] | None = None,
        strategy: str = "balanced",
        metadata: MetadataDict | None = None,
        max_iterations: int = 1000,
        tolerance: float = 0.05,
    ) -> list[ExperimentList]:
        """Partition items with batch-level constraints.

        Creates initial partition using standard partitioning, then iteratively
        refines to satisfy batch constraints through item swaps between lists.

        Parameters
        ----------
        items : list[UUID]
            Item UUIDs to partition.
        n_lists : int
            Number of lists to create.
        list_constraints : list[ListConstraint] | None, default=None
            Per-list constraints to satisfy.
        batch_constraints : list[BatchConstraint] | None, default=None
            Batch-level constraints to satisfy.
        strategy : str, default="balanced"
            Initial partitioning strategy ("balanced", "random", "stratified").
        metadata : dict[UUID, dict[str, Any]] | None, default=None
            Metadata for each item UUID.
        max_iterations : int, default=1000
            Maximum refinement iterations.
        tolerance : float, default=0.05
            Tolerance for batch constraint satisfaction (score >= 1.0 - tolerance).

        Returns
        -------
        list[ExperimentList]
            Partitioned lists satisfying both list and batch constraints.

        Examples
        --------
        >>> from bead.lists.constraints import BatchCoverageConstraint
        >>> partitioner = ListPartitioner(random_seed=42)
        >>> constraint = BatchCoverageConstraint(
        ...     property_expression="item['template_id']",
        ...     target_values=list(range(26)),
        ...     min_coverage=1.0
        ... )
        >>> lists = partitioner.partition_with_batch_constraints(
        ...     items=item_uids,
        ...     n_lists=8,
        ...     batch_constraints=[constraint],
        ...     metadata=metadata_dict,
        ...     max_iterations=500
        ... )
        """
        # Initial partitioning with list constraints
        lists = self.partition(
            items=items,
            n_lists=n_lists,
            constraints=list_constraints,
            strategy=strategy,
            metadata=metadata,
        )

        # If no batch constraints, return immediately
        if not batch_constraints:
            return lists

        metadata = metadata or {}

        # Iterative refinement loop
        for _ in range(max_iterations):
            # Check all batch constraints
            all_satisfied = True
            min_score = 1.0
            worst_constraint = None

            for constraint in batch_constraints:
                score = self._compute_batch_constraint_score(
                    lists, constraint, metadata
                )
                if score < (1.0 - tolerance):
                    all_satisfied = False
                if score < min_score:
                    min_score = score
                    worst_constraint = constraint

            # If all satisfied, we're done
            if all_satisfied:
                break

            # Try to improve worst constraint
            if worst_constraint is not None:
                improved = self._improve_batch_constraint(
                    lists,
                    worst_constraint,
                    list_constraints or [],
                    batch_constraints,
                    metadata,
                )
                if not improved:
                    # No improvement possible, stop
                    break

        return lists

    def _improve_batch_constraint(
        self,
        lists: list[ExperimentList],
        constraint: BatchConstraint,
        list_constraints: list[ListConstraint],
        batch_constraints: list[BatchConstraint],
        metadata: MetadataDict,
        n_attempts: int = 100,
    ) -> bool:
        """Attempt to improve batch constraint through item swaps.

        Parameters
        ----------
        lists : list[ExperimentList]
            Current lists.
        constraint : BatchConstraint
            Constraint to improve.
        list_constraints : list[ListConstraint]
            Per-list constraints that must remain satisfied.
        batch_constraints : list[BatchConstraint]
            All batch constraints to check.
        metadata : MetadataDict
            Item metadata.
        n_attempts : int, default=100
            Number of swap attempts.

        Returns
        -------
        bool
            True if improvement was made.
        """
        current_score = self._compute_batch_constraint_score(
            lists, constraint, metadata
        )

        for _ in range(n_attempts):
            # Select two random lists
            if len(lists) < 2:
                return False

            # integers returns int or array depending on size parameter
            list_idx_a = int(self._rng.integers(0, len(lists)))
            list_idx_b = int(self._rng.integers(0, len(lists)))

            if list_idx_a == list_idx_b:
                continue

            list_a = lists[list_idx_a]
            list_b = lists[list_idx_b]

            # Select random items from each list
            if len(list_a.item_refs) == 0 or len(list_b.item_refs) == 0:
                continue

            item_idx_a = int(self._rng.integers(0, len(list_a.item_refs)))
            item_idx_b = int(self._rng.integers(0, len(list_b.item_refs)))

            item_a = list_a.item_refs[item_idx_a]
            item_b = list_b.item_refs[item_idx_b]

            swapped_a = list_a.with_(
                item_refs=tuple(
                    item_b if i == item_idx_a else ref
                    for i, ref in enumerate(list_a.item_refs)
                )
            )
            swapped_b = list_b.with_(
                item_refs=tuple(
                    item_a if i == item_idx_b else ref
                    for i, ref in enumerate(list_b.item_refs)
                )
            )

            hypothetical = list(lists)
            hypothetical[list_idx_a] = swapped_a
            hypothetical[list_idx_b] = swapped_b

            new_score = self._compute_batch_constraint_score(
                hypothetical, constraint, metadata
            )

            list_a_valid = (
                self._count_violations(swapped_a, list_constraints, metadata) == 0
            )
            list_b_valid = (
                self._count_violations(swapped_b, list_constraints, metadata) == 0
            )

            if new_score > current_score and list_a_valid and list_b_valid:
                lists[list_idx_a] = swapped_a
                lists[list_idx_b] = swapped_b
                return True

        return False

    def _compute_batch_constraint_score(
        self,
        lists: list[ExperimentList],
        constraint: BatchConstraint,
        metadata: MetadataDict,
    ) -> float:
        """Compute satisfaction score for batch constraint.

        Parameters
        ----------
        lists : list[ExperimentList]
            All lists in the batch.
        constraint : BatchConstraint
            Batch constraint to check.
        metadata : MetadataDict
            Item metadata.

        Returns
        -------
        float
            Satisfaction score in [0, 1].
        """
        if isinstance(constraint, BatchCoverageConstraint):
            return self._compute_batch_coverage_score(lists, constraint, metadata)
        if isinstance(constraint, BatchBalanceConstraint):
            return self._compute_batch_balance_score(lists, constraint, metadata)
        if isinstance(constraint, BatchDiversityConstraint):
            return self._compute_batch_diversity_score(lists, constraint, metadata)
        assert isinstance(constraint, BatchMinOccurrenceConstraint)
        return self._compute_batch_min_occurrence_score(lists, constraint, metadata)

    def _compute_batch_coverage_score(
        self,
        lists: list[ExperimentList],
        constraint: BatchCoverageConstraint,
        metadata: MetadataDict,
    ) -> float:
        """Compute coverage score across all lists.

        Parameters
        ----------
        lists : list[ExperimentList]
            All lists in the batch.
        constraint : BatchCoverageConstraint
            Coverage constraint.
        metadata : MetadataDict
            Item metadata.

        Returns
        -------
        float
            Coverage ratio (observed_values / target_values).
        """
        # Collect all observed values across all lists
        observed_values: set[int | float | str | bool] = set()
        for exp_list in lists:
            for item_id in exp_list.item_refs:
                try:
                    value = self._extract_property_value(
                        item_id,
                        constraint.property_expression,
                        constraint.context,
                        metadata,
                    )
                    observed_values.add(value)
                except Exception:
                    continue

        # Compute coverage
        if constraint.target_values is None:
            return 1.0

        if len(constraint.target_values) == 0:
            return 1.0

        target_set: set[int | float | str | bool] = set(constraint.target_values)
        coverage = len(observed_values & target_set) / len(target_set)
        return float(coverage)

    def _compute_batch_balance_score(
        self,
        lists: list[ExperimentList],
        constraint: BatchBalanceConstraint,
        metadata: MetadataDict,
    ) -> float:
        """Compute balance score across all lists.

        Parameters
        ----------
        lists : list[ExperimentList]
            All lists in the batch.
        constraint : BatchBalanceConstraint
            Balance constraint.
        metadata : MetadataDict
            Item metadata.

        Returns
        -------
        float
            Score in [0, 1] based on deviation from target distribution.
        """
        # Count occurrences across all lists
        counts: Counter[str] = Counter()
        total = 0

        for exp_list in lists:
            for item_id in exp_list.item_refs:
                try:
                    value = self._extract_property_value(
                        item_id,
                        constraint.property_expression,
                        constraint.context,
                        metadata,
                    )
                    counts[value] += 1
                    total += 1
                except Exception:
                    continue

        if total == 0:
            return 1.0

        # Compute actual distribution
        actual_dist = {k: v / total for k, v in counts.items()}

        # Compute max deviation from target
        max_deviation = 0.0
        for value, target_prob in constraint.target_distribution.items():
            actual_prob = actual_dist.get(value, 0.0)
            deviation = abs(actual_prob - target_prob)
            max_deviation = max(max_deviation, deviation)

        # Score decreases with deviation
        score = max(0.0, 1.0 - max_deviation)
        return float(score)

    def _compute_batch_diversity_score(
        self,
        lists: list[ExperimentList],
        constraint: BatchDiversityConstraint,
        metadata: MetadataDict,
    ) -> float:
        """Compute diversity score across all lists.

        Parameters
        ----------
        lists : list[ExperimentList]
            All lists in the batch.
        constraint : BatchDiversityConstraint
            Diversity constraint.
        metadata : MetadataDict
            Item metadata.

        Returns
        -------
        float
            Score in [0, 1]. 1.0 if constraint satisfied, decreases with violations.
        """
        # Track which lists contain each value
        value_to_lists: defaultdict[str | int | float, set[int]] = defaultdict(set)

        for list_idx, exp_list in enumerate(lists):
            for item_id in exp_list.item_refs:
                try:
                    value = self._extract_property_value(
                        item_id,
                        constraint.property_expression,
                        constraint.context,
                        metadata,
                    )
                    value_to_lists[value].add(list_idx)
                except Exception:
                    continue

        if not value_to_lists:
            return 1.0

        # Compute violations
        violations = 0
        total_values = len(value_to_lists)

        for _value, list_indices in value_to_lists.items():
            if len(list_indices) > constraint.max_lists_per_value:
                violations += 1

        # Score = 1.0 when no violations, decreases linearly
        score = 1.0 - (violations / max(total_values, 1))
        return float(max(0.0, score))

    def _compute_batch_min_occurrence_score(
        self,
        lists: list[ExperimentList],
        constraint: BatchMinOccurrenceConstraint,
        metadata: MetadataDict,
    ) -> float:
        """Compute minimum occurrence score across all lists.

        Parameters
        ----------
        lists : list[ExperimentList]
            All lists in the batch.
        constraint : BatchMinOccurrenceConstraint
            Minimum occurrence constraint.
        metadata : MetadataDict
            Item metadata.

        Returns
        -------
        float
            Score in [0, 1] based on minimum count ratio.
        """
        # Count occurrences of each value
        counts: Counter[str] = Counter()

        for exp_list in lists:
            for item_id in exp_list.item_refs:
                try:
                    value = self._extract_property_value(
                        item_id,
                        constraint.property_expression,
                        constraint.context,
                        metadata,
                    )
                    counts[value] += 1
                except Exception:
                    continue

        if not counts:
            return 1.0

        # Score = min(count / target) across all values
        min_ratio = float("inf")
        for _value, count in counts.items():
            ratio = count / constraint.min_occurrences
            min_ratio = min(min_ratio, ratio)

        # Clip to [0, 1]
        score = min(1.0, max(0.0, min_ratio))
        return float(score)
