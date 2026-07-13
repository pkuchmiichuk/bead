"""Stratification utilities for binning items into quantile and grid strata.

This module assigns items to bins along one or more dimensions. Each dimension
declares a binning strategy via a :class:`~bead.lists.constraints.BinningSpec`
(quantile, equal-width, threshold, standard-deviation, or categorical). The
Cartesian product of the per-dimension bins forms a grid; an item's grid cell
is the tuple of its per-dimension bin indices, optionally flattened to a single
integer id for stand-off storage and balancing.

The one-dimensional :func:`assign_quantiles` helper is the quantile
specialization of the grid engine, kept for ergonomics.
"""

from __future__ import annotations

from collections import defaultdict
from collections.abc import Callable, Hashable, Sequence
from typing import Any, Protocol
from uuid import UUID

import numpy as np

from bead.items.item import MetadataValue
from bead.lists.constraints import (
    BinningSpec,
    CategoricalBinning,
    EqualWidthBinning,
    QuantileBinning,
    StdDevBinning,
    ThresholdBinning,
)

# ---------------------------------------------------------------------------
# fitted binners (behavior; not didactic models, since they carry numpy state)
# ---------------------------------------------------------------------------


class _Binner(Protocol):
    """A fitted binning function mapping a single value to a bin index."""

    n_bins: int

    def bin(self, value: float | str) -> int: ...


class _QuantileBinner:
    """Equal-frequency binner using empirical quantile cut points."""

    def __init__(
        self,
        cut_points: np.ndarray[Any, np.dtype[np.floating[Any]]],
        n_bins: int,
    ) -> None:
        self._cut = cut_points
        self.n_bins = n_bins

    def bin(self, value: float | str) -> int:
        idx = int(np.searchsorted(self._cut, float(value)))
        return min(max(idx, 0), self.n_bins - 1)


class _EqualWidthBinner:
    """Equal-interval binner over a fixed range."""

    def __init__(self, lo: float, width: float, n_bins: int) -> None:
        self._lo = lo
        self._width = width
        self.n_bins = n_bins

    def bin(self, value: float | str) -> int:
        if self._width <= 0.0:
            return 0
        idx = int((float(value) - self._lo) // self._width)
        return min(max(idx, 0), self.n_bins - 1)


class _ThresholdBinner:
    """Binner delimited by a strictly increasing array of cut points."""

    def __init__(self, edges: Sequence[float]) -> None:
        self._edges = np.asarray(edges, dtype=float)
        self.n_bins = len(edges) + 1

    def bin(self, value: float | str) -> int:
        idx = int(np.searchsorted(self._edges, float(value), side="right"))
        return min(max(idx, 0), self.n_bins - 1)


class _CategoricalBinner:
    """Binner mapping each distinct discrete value to its own bin."""

    def __init__(
        self, index: dict[str, int], other_bin: int | None, n_bins: int
    ) -> None:
        self._index = index
        self._other_bin = other_bin
        self.n_bins = n_bins

    def bin(self, value: float | str) -> int:
        key = str(value)
        if key in self._index:
            return self._index[key]
        if self._other_bin is not None:
            return self._other_bin
        raise ValueError(f"value {key!r} is outside the declared categories")


def _fit_binner(spec: BinningSpec, values: Sequence[float | str]) -> _Binner:
    """Fit a binner for one dimension from its spec and the dimension's values."""
    if isinstance(spec, QuantileBinning):
        numeric = np.asarray([float(v) for v in values], dtype=float)
        edges = np.quantile(numeric, np.linspace(0.0, 1.0, spec.n_quantiles + 1))
        return _QuantileBinner(edges[1:], spec.n_quantiles)
    if isinstance(spec, EqualWidthBinning):
        numeric = np.asarray([float(v) for v in values], dtype=float)
        lo = spec.range_min if spec.range_min is not None else float(np.min(numeric))
        hi = spec.range_max if spec.range_max is not None else float(np.max(numeric))
        width = (hi - lo) / spec.n_bins
        return _EqualWidthBinner(lo, width, spec.n_bins)
    if isinstance(spec, ThresholdBinning):
        return _ThresholdBinner(spec.edges)
    if isinstance(spec, StdDevBinning):
        numeric = np.asarray([float(v) for v in values], dtype=float)
        mean = float(np.mean(numeric))
        sd = float(np.std(numeric))
        return _ThresholdBinner([mean + k * sd for k in spec.k_values])
    assert isinstance(spec, CategoricalBinning)
    if spec.categories is not None:
        declared = {category: i for i, category in enumerate(spec.categories)}
        other = len(spec.categories) if spec.include_other else None
        n_bins = len(spec.categories) + (1 if spec.include_other else 0)
        return _CategoricalBinner(declared, other, n_bins)
    discovered = sorted({str(v) for v in values})
    index = {category: i for i, category in enumerate(discovered)}
    return _CategoricalBinner(index, None, len(discovered))


def _fit_binners[T](
    items: list[T],
    getters: Sequence[Callable[[T], float | str]],
    binnings: Sequence[BinningSpec],
) -> list[_Binner]:
    """Fit one binner per dimension over all items."""
    binners: list[_Binner] = []
    for getter, spec in zip(getters, binnings, strict=True):
        binners.append(_fit_binner(spec, [getter(item) for item in items]))
    return binners


# ---------------------------------------------------------------------------
# grid stratification engine
# ---------------------------------------------------------------------------


def assign_grid_cells[T](
    items: list[T],
    getters: Sequence[Callable[[T], float | str]],
    binnings: Sequence[BinningSpec],
    stratify_by: Callable[[T], Hashable] | None = None,
) -> dict[T, tuple[int, ...]]:
    """Assign each item to an N-dimensional grid cell.

    Each dimension bins one value via its strategy; the item's cell is the
    tuple of per-dimension bin indices. When ``stratify_by`` is given,
    continuous bins are computed within each group while categorical bin counts
    stay global so the grid shape is consistent across groups.

    Parameters
    ----------
    items : list[T]
        Items to assign.
    getters : Sequence[Callable[[T], float | str]]
        One value accessor per dimension. Numeric strategies coerce to float;
        the categorical strategy coerces to str.
    binnings : Sequence[BinningSpec]
        One binning strategy per dimension; same length as ``getters``.
    stratify_by : Callable[[T], Hashable] | None
        Optional grouping function; continuous bins are fit within each group.

    Returns
    -------
    dict[T, tuple[int, ...]]
        Mapping from each item to its per-dimension bin-index tuple.

    Raises
    ------
    ValueError
        If ``items`` is empty, no dimensions are given, or ``getters`` and
        ``binnings`` differ in length.
    """
    if not items:
        raise ValueError("items list cannot be empty")
    if len(getters) != len(binnings):
        raise ValueError("getters and binnings must have the same length")
    if len(binnings) < 1:
        raise ValueError("at least one dimension is required")

    global_binners = _fit_binners(items, getters, binnings)

    if stratify_by is None:
        return {
            item: tuple(
                binner.bin(getter(item))
                for binner, getter in zip(global_binners, getters, strict=True)
            )
            for item in items
        }

    groups: dict[Hashable, list[T]] = defaultdict(list)
    for item in items:
        groups[stratify_by(item)].append(item)

    result: dict[T, tuple[int, ...]] = {}
    for group_items in groups.values():
        group_binners: list[_Binner] = []
        for dim, (getter, spec) in enumerate(zip(getters, binnings, strict=True)):
            if isinstance(spec, CategoricalBinning):
                group_binners.append(global_binners[dim])
            else:
                group_binners.append(
                    _fit_binner(spec, [getter(item) for item in group_items])
                )
        for item in group_items:
            result[item] = tuple(
                binner.bin(getter(item))
                for binner, getter in zip(group_binners, getters, strict=True)
            )
    return result


def grid_shape[T](
    items: list[T],
    getters: Sequence[Callable[[T], float | str]],
    binnings: Sequence[BinningSpec],
) -> tuple[int, ...]:
    """Return the per-dimension bin counts of the grid.

    The shape is the cardinality of each dimension fit over all items, used to
    flatten grid cells to single integer ids.
    """
    if not items:
        raise ValueError("items list cannot be empty")
    return tuple(binner.n_bins for binner in _fit_binners(items, getters, binnings))


def flatten_cell(coords: Sequence[int], shape: Sequence[int]) -> int:
    """Flatten a per-dimension bin-index tuple to a row-major integer id."""
    if len(coords) != len(shape):
        raise ValueError("coords and shape must have the same length")
    cell_id = 0
    for coord, size in zip(coords, shape, strict=True):
        cell_id = cell_id * size + coord
    return cell_id


def unflatten_cell(cell_id: int, shape: Sequence[int]) -> tuple[int, ...]:
    """Invert :func:`flatten_cell`, recovering the per-dimension bin indices."""
    coords: list[int] = []
    for size in reversed(shape):
        coords.append(cell_id % size)
        cell_id //= size
    return tuple(reversed(coords))


def assign_grid_cells_by_uuid(
    item_ids: list[UUID],
    item_metadata: dict[UUID, dict[str, MetadataValue]],
    property_keys: Sequence[str],
    binnings: Sequence[BinningSpec],
    stratify_by_key: str | None = None,
) -> dict[UUID, int]:
    """Assign UUID-keyed items to flattened grid cells via metadata lookup.

    Convenience wrapper over :func:`assign_grid_cells` for the stand-off
    annotation pattern. Each dimension reads one metadata key; the returned
    cell id is the row-major flattening of the per-dimension bin indices.

    Parameters
    ----------
    item_ids : list[UUID]
        Item UUIDs.
    item_metadata : dict[UUID, dict[str, MetadataValue]]
        Metadata dict mapping UUIDs to their metadata dicts.
    property_keys : Sequence[str]
        One metadata key per dimension; same length as ``binnings``.
    binnings : Sequence[BinningSpec]
        One binning strategy per dimension.
    stratify_by_key : str | None
        Optional metadata key for stratification.

    Returns
    -------
    dict[UUID, int]
        Mapping from each UUID to its flattened grid-cell id.

    Raises
    ------
    ValueError
        If ``property_keys`` and ``binnings`` differ in length, a property key
        is missing from some item, or the stratification key is missing.
    KeyError
        If any UUID is absent from ``item_metadata``.
    """
    if len(property_keys) != len(binnings):
        raise ValueError("property_keys and binnings must have the same length")

    for uid in item_ids:
        if uid not in item_metadata:
            raise KeyError(f"UUID {uid} not found in item_metadata")
        for key in property_keys:
            if key not in item_metadata[uid]:
                raise ValueError(
                    f"Property '{key}' not found in metadata for UUID {uid}"
                )
    if stratify_by_key is not None and any(
        stratify_by_key not in item_metadata[uid] for uid in item_ids
    ):
        raise ValueError(
            f"Stratification key '{stratify_by_key}' not found in all items"
        )

    def make_getter(key: str) -> Callable[[UUID], float | str]:
        def getter(uid: UUID) -> float | str:
            value = item_metadata[uid][key]
            return value  # type: ignore[return-value]

        return getter

    getters = [make_getter(key) for key in property_keys]

    stratify_func: Callable[[UUID], Hashable] | None
    if stratify_by_key is not None:
        strat_key = stratify_by_key

        def stratify_getter(uid: UUID) -> Hashable:
            return item_metadata[uid][strat_key]  # type: ignore[return-value]

        stratify_func = stratify_getter
    else:
        stratify_func = None

    shape = grid_shape(item_ids, getters, binnings)
    coords = assign_grid_cells(item_ids, getters, binnings, stratify_func)
    return {uid: flatten_cell(coords[uid], shape) for uid in item_ids}


# ---------------------------------------------------------------------------
# one-dimensional quantile helpers (quantile specialization of the grid engine)
# ---------------------------------------------------------------------------


def assign_quantiles[T](
    items: list[T],
    property_getter: Callable[[T], float],
    n_quantiles: int = 10,
    stratify_by: Callable[[T], Hashable] | None = None,
) -> dict[T, int]:
    """Assign quantile bins to items based on a numeric property.

    Divides items into ``n_quantiles`` equal-frequency bins based on the
    distribution of a numeric property. Optionally stratifies by a grouping
    variable, computing separate quantiles for each group. This is the
    one-dimensional quantile case of :func:`assign_grid_cells`.

    Parameters
    ----------
    items : list[T]
        List of items to assign to quantile bins.
    property_getter : Callable[[T], float]
        Function that extracts a numeric value from each item.
    n_quantiles : int
        Number of quantile bins (default: 10 for deciles). Must be >= 2.
    stratify_by : Callable[[T], Hashable] | None
        Optional grouping function; quantiles are computed per group.

    Returns
    -------
    dict[T, int]
        Dictionary mapping each item to its quantile bin (0 to n_quantiles-1).

    Raises
    ------
    ValueError
        If n_quantiles < 2 or items list is empty.

    Examples
    --------
    >>> items = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]
    >>> result = assign_quantiles(
    ...     items,
    ...     property_getter=lambda x: x,
    ...     n_quantiles=4,
    ... )
    >>> result[1]
    0
    >>> result[10]
    3
    """
    if not items:
        raise ValueError("items list cannot be empty")
    if n_quantiles < 2:
        raise ValueError(f"n_quantiles must be >= 2, got {n_quantiles}")

    binning = QuantileBinning(binning="quantile", n_quantiles=n_quantiles)
    coords = assign_grid_cells(items, [property_getter], [binning], stratify_by)
    return {item: cell[0] for item, cell in coords.items()}


def assign_quantiles_by_uuid(
    item_ids: list[UUID],
    item_metadata: dict[UUID, dict[str, MetadataValue]],
    property_key: str,
    n_quantiles: int = 10,
    stratify_by_key: str | None = None,
) -> dict[UUID, int]:
    """Assign quantile bins to items by UUID with metadata lookup.

    Convenience function for the common pattern of working with UUIDs and
    metadata dictionaries (stand-off annotation pattern).

    Parameters
    ----------
    item_ids : list[UUID]
        List of item UUIDs.
    item_metadata : dict[UUID, dict[str, MetadataValue]]
        Metadata dictionary mapping UUIDs to their metadata dicts.
    property_key : str
        Key in item_metadata[uuid] dict to use for quantile computation.
    n_quantiles : int
        Number of quantile bins (default: 10).
    stratify_by_key : str | None
        Optional key in metadata dict to use for stratification.

    Returns
    -------
    dict[UUID, int]
        Dictionary mapping each UUID to its quantile bin (0 to n_quantiles-1).

    Raises
    ------
    ValueError
        If property_key missing from any item's metadata.
    KeyError
        If any UUID not found in item_metadata.
    """
    # validate that all items have the property
    for uid in item_ids:
        if uid not in item_metadata:
            raise KeyError(f"UUID {uid} not found in item_metadata")
        if property_key not in item_metadata[uid]:
            raise ValueError(
                f"Property '{property_key}' not found in metadata for UUID {uid}"
            )

    # create property getter
    def property_getter(uid: UUID) -> float:
        value = item_metadata[uid][property_key]
        return float(value)  # type: ignore[arg-type]

    # create stratification getter if needed
    stratify_func: Callable[[UUID], Hashable] | None
    if stratify_by_key:
        if any(stratify_by_key not in item_metadata[uid] for uid in item_ids):
            raise ValueError(
                f"Stratification key '{stratify_by_key}' not found in all items"
            )

        strat_key = stratify_by_key

        def stratify_getter(uid: UUID) -> Hashable:
            return item_metadata[uid][strat_key]  # type: ignore[return-value]

        stratify_func = stratify_getter
    else:
        stratify_func = None

    return assign_quantiles(item_ids, property_getter, n_quantiles, stratify_func)
