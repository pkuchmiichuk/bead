"""Quantile balancing for experimental list partitioning.

This module provides the QuantileBalancer class for ensuring uniform distribution
of items across quantiles of a numeric property. Uses NumPy for efficient
quantile computation and maintains stand-off annotation pattern (works with UUIDs).
"""

from __future__ import annotations

from collections.abc import Callable
from typing import Any
from uuid import UUID

import numpy as np


class QuantileBalancer:
    """Ensures uniform distribution of items across quantiles.

    Used by stratified partitioning strategy to create balanced distribution
    of numeric properties (e.g., LM probabilities, word frequencies).

    Works with UUIDs only (stand-off annotation). Requires value_func callable
    to extract numeric values from items via their UUIDs.

    Parameters
    ----------
    n_quantiles : int, default=5
        Number of quantiles to create (must be >= 2).
    random_seed : int | None, default=None
        Random seed for reproducibility. If None, uses non-deterministic RNG.

    Attributes
    ----------
    n_quantiles : int
        Number of quantiles to create.
    random_seed : int | None
        Random seed for reproducibility.

    Examples
    --------
    >>> from uuid import uuid4
    >>> import numpy as np
    >>> balancer = QuantileBalancer(n_quantiles=5, random_seed=42)
    >>> # Create items with known values
    >>> items = [uuid4() for _ in range(100)]
    >>> values = {item: float(i) for i, item in enumerate(items)}
    >>> value_func = lambda uid: values[uid]
    >>> # Balance across 4 lists, 5 items per quantile per list
    >>> lists = balancer.balance(items, value_func, n_lists=4,
    ...                          items_per_quantile_per_list=5)
    >>> len(lists)
    4
    """

    def __init__(self, n_quantiles: int = 5, random_seed: int | None = None) -> None:
        if n_quantiles < 2:
            raise ValueError(f"n_quantiles must be >= 2, got {n_quantiles}")

        self.n_quantiles = n_quantiles
        self.random_seed = random_seed
        self._rng = np.random.default_rng(random_seed)

    def balance(
        self,
        item_ids: list[UUID],
        value_func: Callable[[UUID], float],
        n_lists: int,
        items_per_quantile_per_list: int,
    ) -> list[list[UUID]]:
        """Balance items across lists and quantiles.

        Distributes items uniformly across quantiles and lists to ensure
        balanced representation of the numeric property across all lists.

        Parameters
        ----------
        item_ids : list[UUID]
            UUIDs of items to balance.
        value_func : Callable[[UUID], float]
            Function to extract numeric value from item UUID.
        n_lists : int
            Number of lists to create.
        items_per_quantile_per_list : int
            Target number of items per quantile per list.

        Returns
        -------
        list[list[UUID]]
            Balanced lists of item UUIDs.

        Raises
        ------
        ValueError
            If n_lists < 1 or items_per_quantile_per_list < 1.

        Examples
        --------
        >>> from uuid import uuid4
        >>> balancer = QuantileBalancer(n_quantiles=5, random_seed=42)
        >>> items = [uuid4() for _ in range(100)]
        >>> values = {item: float(i) for i, item in enumerate(items)}
        >>> lists = balancer.balance(items, lambda uid: values[uid], 4, 5)
        >>> all(len(lst) == 25 for lst in lists)  # 5 quantiles * 5 items
        True

        Notes
        -----
        - Items are assigned to quantiles using np.percentile and np.digitize
        - Within each quantile, items are shuffled before distribution
        - If insufficient items exist in a quantile, fewer items are assigned
        """
        if n_lists < 1:
            raise ValueError(f"n_lists must be >= 1, got {n_lists}")
        if items_per_quantile_per_list < 1:
            raise ValueError(
                f"items_per_quantile_per_list must be >= 1, "
                f"got {items_per_quantile_per_list}"
            )

        # create quantile-based strata and distribute across lists
        strata = self._create_strata(item_ids, value_func)
        return self._distribute(
            strata, self.n_quantiles, n_lists, items_per_quantile_per_list
        )

    def balance_by_cell(
        self,
        item_ids: list[UUID],
        cell_func: Callable[[UUID], int],
        n_cells: int,
        n_lists: int,
        items_per_cell_per_list: int,
    ) -> list[list[UUID]]:
        """Balance items across lists by an arbitrary precomputed stratum id.

        Generalizes :meth:`balance` from one-dimensional quantiles to the cells
        of an N-dimensional stratification grid. Each item's stratum is its grid
        cell id; items in each cell are spread uniformly across lists.

        Parameters
        ----------
        item_ids : list[UUID]
            UUIDs of items to balance.
        cell_func : Callable[[UUID], int]
            Function returning the grid cell id (0 to n_cells-1) for an item.
        n_cells : int
            Number of grid cells.
        n_lists : int
            Number of lists to create.
        items_per_cell_per_list : int
            Target number of items per cell per list.

        Returns
        -------
        list[list[UUID]]
            Balanced lists of item UUIDs.

        Raises
        ------
        ValueError
            If n_cells < 1, n_lists < 1, or items_per_cell_per_list < 1.
        """
        if n_cells < 1:
            raise ValueError(f"n_cells must be >= 1, got {n_cells}")
        if n_lists < 1:
            raise ValueError(f"n_lists must be >= 1, got {n_lists}")
        if items_per_cell_per_list < 1:
            raise ValueError(
                f"items_per_cell_per_list must be >= 1, got {items_per_cell_per_list}"
            )

        strata: dict[int, list[UUID]] = {c: [] for c in range(n_cells)}
        for item_id in item_ids:
            strata[cell_func(item_id)].append(item_id)

        return self._distribute(strata, n_cells, n_lists, items_per_cell_per_list)

    def _distribute(
        self,
        strata: dict[int, list[UUID]],
        n_strata: int,
        n_lists: int,
        items_per_stratum_per_list: int,
    ) -> list[list[UUID]]:
        """Distribute each stratum's items uniformly across lists.

        Within each stratum, items are shuffled and sliced contiguously per
        list. If a stratum holds fewer items than ``n_lists *
        items_per_stratum_per_list``, the later lists receive fewer items.
        """
        lists: list[list[UUID]] = [[] for _ in range(n_lists)]

        for stratum in range(n_strata):
            stratum_items = strata.get(stratum, [])
            stratum_array = np.array(stratum_items, dtype=object)
            self._rng.shuffle(stratum_array)

            for list_idx in range(n_lists):
                start_idx = list_idx * items_per_stratum_per_list
                end_idx = start_idx + items_per_stratum_per_list
                lists[list_idx].extend(stratum_array[start_idx:end_idx].tolist())

        return lists

    def compute_balance_score(
        self, item_ids: list[UUID], value_func: Callable[[UUID], float]
    ) -> float:
        """Compute balance score for items.

        Score is 1.0 for perfect balance (uniform distribution across quantiles),
        lower for imbalanced distributions. Score is based on deviation from
        expected uniform distribution.

        Parameters
        ----------
        item_ids : list[UUID]
            UUIDs of items to score.
        value_func : Callable[[UUID], float]
            Function to extract numeric values.

        Returns
        -------
        float
            Balance score (0.0-1.0, higher is better).

        Examples
        --------
        >>> from uuid import uuid4
        >>> balancer = QuantileBalancer(n_quantiles=5)
        >>> # Uniformly distributed values
        >>> items = [uuid4() for _ in range(100)]
        >>> values = {item: float(i) for i, item in enumerate(items)}
        >>> score = balancer.compute_balance_score(items, lambda uid: values[uid])
        >>> score > 0.9  # Should be close to 1.0
        True

        Notes
        -----
        - Returns 0.0 for empty item lists
        - Uses mean absolute deviation from expected uniform count
        """
        if not item_ids:
            return 0.0

        # compute values
        values: np.ndarray[tuple[int, ...], np.dtype[np.floating[Any]]] = np.array(
            [value_func(item_id) for item_id in item_ids]
        )

        # create expected quantile bins
        expected_quantiles: np.ndarray[tuple[int], np.dtype[np.floating[Any]]] = (
            np.linspace(0, 100, self.n_quantiles + 1)
        )
        # percentile with array input returns array
        expected_bins: np.ndarray[Any, np.dtype[np.floating[Any]]] = np.percentile(
            values, expected_quantiles
        )

        # count items in each quantile; digitize returns array of integers
        quantile_assignments: np.ndarray[Any, np.dtype[np.intp]] = (
            np.digitize(values, expected_bins) - 1
        )
        quantile_assignments = np.clip(quantile_assignments, 0, self.n_quantiles - 1)

        quantile_counts = np.bincount(quantile_assignments, minlength=self.n_quantiles)

        # compute uniformity score
        expected_count = len(item_ids) / self.n_quantiles
        deviations = np.abs(quantile_counts - expected_count)
        score = 1.0 - (np.mean(deviations) / expected_count)

        return float(max(0.0, score))

    def _create_strata(
        self, item_ids: list[UUID], value_func: Callable[[UUID], float]
    ) -> dict[int, list[UUID]]:
        """Create quantile-based strata from items.

        Parameters
        ----------
        item_ids : list[UUID]
            UUIDs of items to stratify.
        value_func : Callable[[UUID], float]
            Function to extract numeric values.

        Returns
        -------
        dict[int, list[UUID]]
            Dictionary mapping quantile index (0 to n_quantiles-1) to list
            of item UUIDs in that quantile.

        Notes
        -----
        - Uses np.percentile to compute quantile boundaries
        - Uses np.digitize to assign items to quantiles
        - Edge cases are handled by clipping to valid quantile range
        """
        # extract values
        values: np.ndarray[tuple[int, ...], np.dtype[np.floating[Any]]] = np.array(
            [value_func(item_id) for item_id in item_ids]
        )

        # compute quantile bins
        quantiles: np.ndarray[tuple[int], np.dtype[np.floating[Any]]] = np.linspace(
            0, 100, self.n_quantiles + 1
        )
        bins: np.ndarray[Any, np.dtype[np.floating[Any]]] = np.percentile(
            values, quantiles
        )

        # assign items to quantiles
        quantile_assignments: np.ndarray[Any, np.dtype[np.intp]] = (
            np.digitize(values, bins) - 1
        )
        quantile_assignments = np.clip(quantile_assignments, 0, self.n_quantiles - 1)

        # group items by quantile
        strata: dict[int, list[UUID]] = {q: [] for q in range(self.n_quantiles)}
        for item_id, q in zip(item_ids, quantile_assignments, strict=False):
            strata[q].append(item_id)

        return strata
