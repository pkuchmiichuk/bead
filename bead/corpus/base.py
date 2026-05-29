"""Corpus source protocol.

A ``CorpusSource`` is anything that streams ``CorpusRecord``s. It is modeled as
a runtime-checkable ``Protocol`` (behavior, not data) rather than a didactic
model, mirroring the transform protocols elsewhere in bead.
"""

from __future__ import annotations

from collections.abc import Iterator
from typing import Protocol, runtime_checkable

from bead.corpus.records import CorpusRecord


@runtime_checkable
class CorpusSource(Protocol):
    """A streaming source of corpus records.

    Attributes
    ----------
    source_name : str
        Identifier stamped onto every record's ``source_name``.
    """

    source_name: str

    def __iter__(self) -> Iterator[CorpusRecord]:
        """Iterate the records of the source."""
        ...
