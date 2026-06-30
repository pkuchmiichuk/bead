"""List distribution configuration and strategies for batch experiments.

Models for configuring list distribution strategies in JATOS batch
experiments. Supports eight distribution strategies (random, sequential,
balanced, latin square, stratified, weighted random, quota-based, and
metadata-based) for assigning participants to experiment lists.
"""

from __future__ import annotations

from enum import StrEnum

import didactic.api as dx

from bead.data.base import BeadBaseModel, JsonValue


class DistributionStrategyType(StrEnum):
    """Named distribution strategies for list assignment."""

    RANDOM = "random"
    SEQUENTIAL = "sequential"
    BALANCED = "balanced"
    LATIN_SQUARE = "latin_square"
    STRATIFIED = "stratified"
    WEIGHTED_RANDOM = "weighted_random"
    QUOTA_BASED = "quota_based"
    METADATA_BASED = "metadata_based"


class QuotaConfig(BeadBaseModel):
    """Configuration for quota-based assignment.

    Attributes
    ----------
    participants_per_list : int
        Target participants per list (> 0).
    allow_overflow : bool
        Allow continued assignment after quota is reached.
    """

    participants_per_list: int
    allow_overflow: bool = False


class WeightedRandomConfig(BeadBaseModel):
    """Configuration for weighted random assignment.

    Attributes
    ----------
    weight_expression : str
        JavaScript expression evaluated with ``list_metadata`` in scope.
    normalize_weights : bool
        Whether to normalize weights to sum to 1.0.
    """

    weight_expression: str
    normalize_weights: bool = True

    @dx.validates("weight_expression")
    def _check_weight_expression(self, value: str) -> str:
        if not value or not value.strip():
            raise ValueError(
                "weight_expression must be non-empty. "
                "Provide a JavaScript expression like 'list_metadata.priority || 1.0'."
            )
        return value.strip()


class LatinSquareConfig(BeadBaseModel):
    """Configuration for Latin square counterbalancing.

    Attributes
    ----------
    balanced : bool
        Use a balanced Latin square (Bradley's 1958 algorithm).
    """

    balanced: bool = True


class MetadataBasedConfig(BeadBaseModel):
    """Configuration for metadata-based assignment.

    Attributes
    ----------
    filter_expression : str | None
        JavaScript boolean expression filtering eligible lists.
    rank_expression : str | None
        JavaScript expression ranking lists.
    rank_ascending : bool
        Sort ascending when using ``rank_expression``.
    """

    filter_expression: str | None = None
    rank_expression: str | None = None
    rank_ascending: bool = True


def validate_metadata_based_config(config: MetadataBasedConfig) -> None:
    """Raise ``ValueError`` if neither expression is supplied."""
    if config.filter_expression is None and config.rank_expression is None:
        raise ValueError(
            "MetadataBasedConfig requires at least one of 'filter_expression' "
            "or 'rank_expression'. Got neither."
        )


class StratifiedConfig(BeadBaseModel):
    """Configuration for stratified assignment.

    Attributes
    ----------
    factors : tuple[str, ...]
        Metadata keys used as stratification factors. Must be non-empty
        and distinct.
    """

    factors: tuple[str, ...]

    @dx.validates("factors")
    def _check_factors(self, value: tuple[str, ...]) -> tuple[str, ...]:
        if not value:
            raise ValueError(
                "StratifiedConfig requires at least one factor in 'factors'."
            )
        if len(value) != len(set(value)):
            duplicates = [x for x in value if value.count(x) > 1]
            raise ValueError(
                f"StratifiedConfig 'factors' contains duplicates: {duplicates}."
            )
        return value


class ListDistributionStrategy(BeadBaseModel):
    """Configuration for list distribution in batch experiments.

    Attributes
    ----------
    strategy_type : DistributionStrategyType
        Type of distribution strategy.
    strategy_config : dict[str, JsonValue]
        Strategy-specific configuration parameters.
    max_participants : int | None
        Maximum total participants across all lists.
    error_on_exhaustion : bool
        Raise an error when ``max_participants`` is reached.
    debug_mode : bool
        Always assign the same list (development aid).
    debug_list_index : int
        List index to use in debug mode (>= 0).
    """

    strategy_type: DistributionStrategyType = DistributionStrategyType.BALANCED
    strategy_config: dict[str, JsonValue] = dx.field(default_factory=dict)
    max_participants: int | None = None
    error_on_exhaustion: bool = True
    debug_mode: bool = False
    debug_list_index: int = 0


def validate_list_distribution_strategy(
    strategy: ListDistributionStrategy,
) -> None:
    """Raise ``ValueError`` if *strategy*'s config doesn't match its type."""
    config = strategy.strategy_config
    keys = list(config.keys())

    if strategy.strategy_type == DistributionStrategyType.QUOTA_BASED:
        if "participants_per_list" not in config:
            raise ValueError(
                f"QuotaConfig requires 'participants_per_list'. Got keys: {keys}."
            )
        ppl = config["participants_per_list"]
        if not isinstance(ppl, int) or ppl <= 0:
            raise ValueError(
                f"'participants_per_list' must be positive int. "
                f"Got: {ppl} ({type(ppl).__name__})."
            )
    elif strategy.strategy_type == DistributionStrategyType.WEIGHTED_RANDOM:
        if "weight_expression" not in config:
            raise ValueError(
                f"WeightedRandomConfig requires 'weight_expression'. Got keys: {keys}."
            )
        expr = config["weight_expression"]
        if not isinstance(expr, str) or not expr.strip():
            raise ValueError(
                f"'weight_expression' must be a non-empty string. "
                f"Got: {expr!r} ({type(expr).__name__})."
            )
    elif strategy.strategy_type == DistributionStrategyType.METADATA_BASED:
        has_filter = bool(config.get("filter_expression"))
        has_rank = bool(config.get("rank_expression"))
        if not has_filter and not has_rank:
            raise ValueError(
                f"MetadataBasedConfig requires 'filter_expression' "
                f"or 'rank_expression'. Got keys: {keys}."
            )
    elif strategy.strategy_type == DistributionStrategyType.STRATIFIED:
        if "factors" not in config:
            raise ValueError(f"StratifiedConfig requires 'factors'. Got keys: {keys}.")
        factors = config["factors"]
        if not isinstance(factors, (list, tuple)) or not factors:
            raise ValueError(
                f"StratifiedConfig 'factors' must be a non-empty list of strings. "
                f"Got: {factors!r}."
            )

    if strategy.debug_list_index < 0:
        raise ValueError(
            f"debug_list_index must be >= 0. Got: {strategy.debug_list_index}."
        )
