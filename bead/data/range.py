"""Generic numeric range model with validation.

Provides ``Range[T]`` for representing validated numeric ranges with bounds
checking, containment testing, and value clamping. Backed by didactic's
auto-parameterised generics, so ``Range[int](min=1, max=7)`` and
``Range[float](min=0.0, max=1.0)`` both work.
"""

from __future__ import annotations

from typing import Generic, TypeVar

import didactic.api as dx

T = TypeVar("T", int, float)


class Range(dx.Model, Generic[T]):
    """A validated numeric range with inclusive bounds.

    Attributes
    ----------
    min : T
        Minimum value (inclusive).
    max : T
        Maximum value (inclusive).

    Examples
    --------
    >>> scale = Range[int](min=1, max=7)
    >>> scale.contains(4)
    True
    >>> scale.contains(0)
    False
    >>> scale.clamp(10)
    7

    >>> probability = Range[float](min=0.0, max=1.0)
    >>> probability.contains(0.5)
    True
    >>> probability.clamp(-0.1)
    0.0
    """

    min: T
    max: T
    __axioms__ = (dx.axiom("max > min", message="min must be strictly less than max"),)

    def contains(self, value: T) -> bool:
        """Return whether *value* lies in ``[min, max]``."""
        return self.min <= value <= self.max

    def clamp(self, value: T) -> T:
        """Clamp *value* into ``[min, max]``."""
        return max(self.min, min(self.max, value))
