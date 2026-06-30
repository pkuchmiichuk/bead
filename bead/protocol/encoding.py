"""Response-space encodings for probabilistic modeling.

Bridges the annotation-side :class:`~bead.protocol.anchor.ResponseSpace`
representation and a model-ready description of a response scale,
providing the index-to-label mapping and scale-type metadata that
downstream modeling code needs.

This module supports three response scale types:

- *Binary*: two unordered options (for example
  ``("change", "no_change")``). Naturally modeled via Bernoulli
  likelihoods.
- *Ordinal*: ordered options on a Likert-like scale (for example a
  five-point Likert scale from ``"definitely no"`` to ``"definitely
  yes"``). Naturally modeled via cumulative-link (ordered logistic)
  likelihoods.
- *Nominal*: unordered multi-option (for example a categorical choice
  among unordered alternatives). Naturally modeled via softmax
  categorical likelihoods.

The encoding itself is likelihood-agnostic. It does *not* select a
likelihood family; downstream modeling code (for example
:mod:`bead.active_learning.models`) chooses the appropriate model
class based on the scale type.
"""

from __future__ import annotations

from typing import Self

import didactic.api as dx

from bead.data.base import BeadBaseModel
from bead.protocol.anchor import ResponseSpace, ScaleType, SemanticPoles

__all__ = [
    "ResponseEncoding",
    "ScaleType",
    "encode_response_space",
]


class ResponseEncoding(BeadBaseModel):
    """Encoding of a response space for probabilistic modeling.

    Bridges the annotation-side :class:`ResponseSpace` and a
    modeling-side representation, providing the index-to-label
    mapping and scale-type metadata needed by both systems.

    Attributes
    ----------
    name : str
        Identifier for this encoding (typically the anchor name, for
        example ``"completion"``).
    n_levels : int
        Number of response categories. Must equal ``len(labels)``.
    scale_type : ScaleType
        Whether the scale is binary, ordinal, or nominal.
    labels : tuple[str, ...]
        Human-readable labels for each index, in order.
    semantic_poles : SemanticPoles | None
        The two participant-facing endpoints of the scale, if ordered
        (for example ``SemanticPoles(low="definitely no",
        high="definitely yes")``). Defaults to ``None``.

    Examples
    --------
    >>> enc = ResponseEncoding(
    ...     name="completion",
    ...     n_levels=5,
    ...     scale_type=ScaleType.ORDINAL,
    ...     labels=("definitely no", "probably no", "unsure",
    ...             "probably yes", "definitely yes"),
    ...     semantic_poles=SemanticPoles(
    ...         low="definitely no", high="definitely yes"
    ...     ),
    ... )
    >>> enc.label_to_index("probably yes")
    3
    >>> enc.index_to_label(0)
    'definitely no'
    >>> enc.is_ordinal
    True

    See Also
    --------
    encode_response_space : Build an encoding from a
        :class:`ResponseSpace`.
    """

    name: str
    n_levels: int
    scale_type: ScaleType
    labels: tuple[str, ...]
    semantic_poles: dx.Embed[SemanticPoles] | None = None

    @dx.model_validator(mode="after")
    def _check_levels_match_labels(self) -> Self:
        """Enforce ``n_levels == len(labels)`` and label uniqueness."""
        if self.n_levels != len(self.labels):
            raise ValueError(
                f"n_levels ({self.n_levels}) does not match "
                f"len(labels) ({len(self.labels)}) for encoding "
                f"{self.name!r}"
            )
        if len(set(self.labels)) != len(self.labels):
            raise ValueError(
                f"Duplicate labels in encoding {self.name!r}: {self.labels}"
            )
        if self.scale_type == ScaleType.BINARY and self.n_levels != 2:
            raise ValueError(
                f"BINARY scale must have exactly 2 levels, got "
                f"{self.n_levels} in encoding {self.name!r}"
            )
        if self.scale_type == ScaleType.FORCED_CHOICE and self.n_levels < 2:
            raise ValueError(
                f"FORCED_CHOICE scale must have at least 2 levels, "
                f"got {self.n_levels} in encoding {self.name!r}"
            )
        return self

    @property
    def is_ordinal(self) -> bool:
        """Whether the response scale is ordered."""
        return self.scale_type == ScaleType.ORDINAL

    @property
    def is_binary(self) -> bool:
        """Whether the response scale is binary."""
        return self.scale_type == ScaleType.BINARY

    @property
    def is_nominal(self) -> bool:
        """Whether the response scale is unordered multi-option."""
        return self.scale_type == ScaleType.NOMINAL

    @property
    def is_forced_choice(self) -> bool:
        """Whether the response scale uses positional forced-choice labels."""
        return self.scale_type == ScaleType.FORCED_CHOICE

    def label_to_index(self, label: str) -> int:
        """Convert a response label to its integer index.

        Parameters
        ----------
        label : str
            The response label string.

        Returns
        -------
        int
            The 0-based index of the label.

        Raises
        ------
        ValueError
            If the label is not in the encoding.
        """
        try:
            return self.labels.index(label)
        except ValueError:
            raise ValueError(
                f"Label {label!r} not found in encoding {self.name!r}. "
                f"Valid labels: {self.labels}"
            ) from None

    def index_to_label(self, index: int) -> str:
        """Convert an integer index to its response label.

        Parameters
        ----------
        index : int
            The 0-based index.

        Returns
        -------
        str
            The response label at that index.

        Raises
        ------
        IndexError
            If the index is out of range for this encoding.
        """
        if index < 0 or index >= len(self.labels):
            raise IndexError(
                f"Index {index} out of range for encoding {self.name!r} "
                f"with {len(self.labels)} levels."
            )
        return self.labels[index]


def _classify_scale(response_space: ResponseSpace) -> ScaleType:
    """Determine the :class:`ScaleType` of a response space.

    A two-option, unordered space is classified as binary; otherwise an
    ordered space is ordinal and an unordered space is nominal.

    Parameters
    ----------
    response_space : ResponseSpace
        The response space to classify.

    Returns
    -------
    ScaleType
        The classified scale type.
    """
    if len(response_space.options) == 2 and not response_space.is_ordered:
        return ScaleType.BINARY
    if response_space.is_ordered:
        return ScaleType.ORDINAL
    return ScaleType.NOMINAL


def encode_response_space(
    name: str,
    response_space: ResponseSpace,
    *,
    scale_type: ScaleType | None = None,
) -> ResponseEncoding:
    """Build a :class:`ResponseEncoding` from a :class:`ResponseSpace`.

    This is the primary bridge from the protocol layer to the modeling
    layer. The resulting encoding shares its labels with the response
    space and inherits the space's ordering as a :class:`ScaleType`,
    unless ``scale_type`` is set to override the inferred kind.

    Parameters
    ----------
    name : str
        Name for the encoding (typically the anchor name, for example
        ``"completion"``).
    response_space : ResponseSpace
        The response space to encode.
    scale_type : ScaleType | None, optional
        Override the kind inferred from the response space. Required
        when declaring a forced-choice encoding, since forced-choice
        and binary share the "two unordered options" shape but are
        modeled differently.

    Returns
    -------
    ResponseEncoding
        The modeling-side encoding.

    Examples
    --------
    >>> rs = ResponseSpace(
    ...     options=("no", "yes"), is_ordered=False
    ... )
    >>> enc = encode_response_space("dynamicity", rs)
    >>> enc.scale_type
    <ScaleType.BINARY: 'binary'>
    >>> enc.is_binary
    True

    >>> rs = ResponseSpace(
    ...     options=("first", "second"), is_ordered=False
    ... )
    >>> enc = encode_response_space(
    ...     "acceptability", rs, scale_type=ScaleType.FORCED_CHOICE
    ... )
    >>> enc.is_forced_choice
    True
    """
    if scale_type is not None:
        resolved = scale_type
    elif response_space.scale_type is not None:
        resolved = response_space.scale_type
    else:
        resolved = _classify_scale(response_space)
    return ResponseEncoding(
        name=name,
        n_levels=len(response_space.options),
        scale_type=resolved,
        labels=response_space.options,
        semantic_poles=response_space.semantic_poles,
    )
