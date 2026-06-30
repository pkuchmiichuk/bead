"""List configuration models."""

from __future__ import annotations

from typing import Literal

import didactic.api as dx


class BatchConstraintConfig(dx.Model):
    """Configuration for batch-level constraints.

    Attributes
    ----------
    type : Literal["coverage", "balance", "diversity", "min_occurrence"]
        Type of batch constraint.
    property_expression : str
        Expression to extract the property (e.g. ``"item['template_id']"``).
    target_values : tuple[str | int | float, ...] | None
        Target values for coverage constraint.
    min_coverage : float
        Minimum coverage fraction (0.0-1.0).
    target_distribution : dict[str, float] | None
        Target distribution for balance constraint.
    tolerance : float
        Tolerance for balance constraint (0.0-1.0).
    max_lists_per_value : int | None
        Maximum lists per value for diversity constraint.
    min_occurrences : int | None
        Minimum occurrences per value for min_occurrence constraint.
    priority : int
        Constraint priority (higher = more important).
    """

    type: Literal["coverage", "balance", "diversity", "min_occurrence"]
    property_expression: str
    target_values: tuple[str | int | float, ...] | None = None
    min_coverage: float = 1.0
    target_distribution: dict[str, float] | None = None
    tolerance: float = 0.1
    max_lists_per_value: int | None = None
    min_occurrences: int | None = None
    priority: int = 1

    @dx.validates("property_expression")
    def _check_property_expression(self, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("property_expression must be non-empty")
        return value.strip()


def validate_batch_constraint_config(config: BatchConstraintConfig) -> None:
    """Raise ``ValueError`` if *config*'s required fields for its type are absent."""
    if config.type == "balance" and config.target_distribution is None:
        raise ValueError("target_distribution required for balance constraint")
    if config.type == "diversity" and config.max_lists_per_value is None:
        raise ValueError("max_lists_per_value required for diversity constraint")
    if config.type == "min_occurrence" and config.min_occurrences is None:
        raise ValueError("min_occurrences required for min_occurrence constraint")


class ListConfig(dx.Model):
    """Configuration for list partitioning.

    Attributes
    ----------
    partitioning_strategy : str
        Strategy name.
    num_lists : int
        Number of lists to create.
    items_per_list : int | None
        Items per list.
    balance_by : tuple[str, ...]
        Fields to balance on.
    ensure_uniqueness : bool
        Whether items must be unique across lists.
    random_seed : int | None
        Random seed for reproducibility.
    batch_constraints : tuple[BatchConstraintConfig, ...] | None
        Batch-level constraints to apply across all lists.
    """

    partitioning_strategy: str = "balanced"
    num_lists: int = 1
    items_per_list: int | None = None
    balance_by: tuple[str, ...] = ()
    ensure_uniqueness: bool = True
    random_seed: int | None = None
    batch_constraints: tuple[dx.Embed[BatchConstraintConfig], ...] | None = None

    @dx.validates("items_per_list")
    def _check_items_per_list(self, value: int | None) -> int | None:
        if value is not None and value <= 0:
            raise ValueError(f"items_per_list must be positive, got {value}")
        return value
