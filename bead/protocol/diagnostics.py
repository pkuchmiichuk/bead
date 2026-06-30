"""Dataset diagnostics and quality reporting for annotation protocols.

Provides :class:`DatasetReport`, a structured immutable summary of
quality issues discovered during dataset preparation, and
:class:`ConditionalObservationValidator`, which checks that responses
to conditional questions respect the protocol's
:attr:`~bead.protocol.family.QuestionFamily.depends_on` graph.

Diagnostic findings are immutable :class:`DiagnosticRecord` instances
collected in order of discovery. The :meth:`DatasetReport.summary`
method produces a human-readable overview suitable for logging.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from enum import StrEnum
from typing import Protocol, Self, runtime_checkable

import didactic.api as dx

from bead.data.base import BeadBaseModel
from bead.protocol.family import AnnotationProtocol


class DiagnosticLevel(StrEnum):
    """Severity of a diagnostic finding.

    Attributes
    ----------
    INFO : str
        Informational message. Wire value: ``"info"``.
    WARNING : str
        Warning that does not prevent dataset use. Wire value:
        ``"warning"``.
    ERROR : str
        Error that may invalidate downstream analysis. Wire value:
        ``"error"``.
    """

    INFO = "info"
    WARNING = "warning"
    ERROR = "error"


class DiagnosticRecord(BeadBaseModel):
    """A single diagnostic finding.

    Attributes
    ----------
    level : DiagnosticLevel
        Severity of the finding.
    category : str
        Short category tag (for example ``"missing_embedding"`` or
        ``"unrecognized_label"``).
    message : str
        Human-readable description.
    item_id : str | None
        The item this finding pertains to, if applicable. Defaults to
        ``None``.
    question_name : str | None
        The anchor name this finding pertains to, if applicable.
        Defaults to ``None``.
    """

    level: DiagnosticLevel
    category: str
    message: str
    item_id: str | None = None
    question_name: str | None = None


class DatasetReport(BeadBaseModel):
    """Immutable structured report of dataset-preparation quality.

    Mutating methods (:meth:`add`, :meth:`with_coverage`,
    :meth:`with_missing_embedding`) follow the bead convention of
    returning a new instance via ``.with_(...)``; the original is
    unchanged.

    Attributes
    ----------
    n_records_input : int
        Total number of input records received. Defaults to ``0``.
    n_items : int
        Number of unique item ids. Defaults to ``0``.
    n_records_encoded : int
        Number of records successfully encoded. Defaults to ``0``.
    n_records_dropped : int
        Number of records dropped. Defaults to ``0``.
    coverage : dict[str, float]
        Per-question response-coverage rate (fraction of items with a
        valid response). Defaults to the empty dict.
    findings : tuple[DiagnosticRecord, ...]
        All diagnostic findings, in order of discovery. Defaults to
        the empty tuple.
    items_missing_embeddings : tuple[str, ...]
        Item ids that had no embedding provided. Defaults to the empty
        tuple.
    """

    n_records_input: int = 0
    n_items: int = 0
    n_records_encoded: int = 0
    n_records_dropped: int = 0
    coverage: dict[str, float] = dx.field(default_factory=dict)
    findings: tuple[dx.Embed[DiagnosticRecord], ...] = ()
    items_missing_embeddings: tuple[str, ...] = ()

    def add(
        self,
        level: DiagnosticLevel,
        category: str,
        message: str,
        *,
        item_id: str | None = None,
        question_name: str | None = None,
    ) -> Self:
        """Return a new report with one additional finding appended.

        Parameters
        ----------
        level : DiagnosticLevel
            Severity.
        category : str
            Category tag.
        message : str
            Description.
        item_id : str | None, optional
            Related item id. Defaults to ``None``.
        question_name : str | None, optional
            Related anchor name. Defaults to ``None``.

        Returns
        -------
        DatasetReport
            New report with the finding added.
        """
        record = DiagnosticRecord(
            level=level,
            category=category,
            message=message,
            item_id=item_id,
            question_name=question_name,
        )
        return self.with_(findings=(*self.findings, record))

    def extend(self, records: Sequence[DiagnosticRecord]) -> Self:
        """Return a new report with multiple findings appended.

        Parameters
        ----------
        records : Sequence[DiagnosticRecord]
            Findings to append.

        Returns
        -------
        DatasetReport
            New report with the findings added.
        """
        return self.with_(findings=(*self.findings, *records))

    def with_coverage(self, question_name: str, rate: float) -> Self:
        """Return a new report with one coverage entry set.

        Parameters
        ----------
        question_name : str
            Anchor name.
        rate : float
            Coverage rate in ``[0.0, 1.0]``.

        Returns
        -------
        DatasetReport
            New report with the entry set or replaced.
        """
        new_coverage = dict(self.coverage)
        new_coverage[question_name] = rate
        return self.with_(coverage=new_coverage)

    def with_missing_embedding(self, item_id: str) -> Self:
        """Return a new report flagging one item as missing an embedding.

        If ``item_id`` is already flagged the report is returned
        unchanged (the missing-embedding list is a set semantically).

        Parameters
        ----------
        item_id : str
            The item id that lacked an embedding.

        Returns
        -------
        DatasetReport
            New report with the item recorded.
        """
        if item_id in self.items_missing_embeddings:
            return self
        return self.with_(
            items_missing_embeddings=(*self.items_missing_embeddings, item_id)
        )

    @property
    def has_warnings(self) -> bool:
        """Whether any warning-level findings exist."""
        return any(f.level == DiagnosticLevel.WARNING for f in self.findings)

    @property
    def has_errors(self) -> bool:
        """Whether any error-level findings exist."""
        return any(f.level == DiagnosticLevel.ERROR for f in self.findings)

    @property
    def warnings(self) -> tuple[DiagnosticRecord, ...]:
        """All warning-level findings, in discovery order."""
        return tuple(f for f in self.findings if f.level == DiagnosticLevel.WARNING)

    @property
    def errors(self) -> tuple[DiagnosticRecord, ...]:
        """All error-level findings, in discovery order."""
        return tuple(f for f in self.findings if f.level == DiagnosticLevel.ERROR)

    def by_category(self, category: str) -> tuple[DiagnosticRecord, ...]:
        """Filter findings by category tag.

        Parameters
        ----------
        category : str
            Category tag to filter on.

        Returns
        -------
        tuple[DiagnosticRecord, ...]
            Matching findings, in discovery order.
        """
        return tuple(f for f in self.findings if f.category == category)

    def summary(self) -> str:
        """Produce a human-readable multi-line summary.

        Returns
        -------
        str
            A summary string suitable for logging.
        """
        lines = [
            f"DatasetReport: {self.n_items} items, {self.n_records_input} records",
            f"  encoded: {self.n_records_encoded}, dropped: {self.n_records_dropped}",
        ]

        if self.items_missing_embeddings:
            lines.append(
                f"  items missing embeddings: {len(self.items_missing_embeddings)}"
            )

        if self.coverage:
            lines.append("  coverage:")
            for name, rate in sorted(self.coverage.items()):
                lines.append(f"    {name}: {rate:.1%}")

        n_warn = len(self.warnings)
        n_err = len(self.errors)
        if n_warn or n_err:
            lines.append(f"  warnings: {n_warn}, errors: {n_err}")

        return "\n".join(lines)


@runtime_checkable
class RecordLike(Protocol):
    """Structural type for records consumed by the validator.

    Any object with the three attributes below conforms. The bead
    :class:`~bead.evaluation.reliability.AnnotationRecord` is a
    canonical example.

    Attributes
    ----------
    item_id : str
        Identifier of the annotation item.
    response_label : str
        Annotator's response label.
    question_name : str
        Anchor name of the question being answered.
    """

    item_id: str
    response_label: str
    question_name: str


@dataclass(frozen=True)
class ConditionalObservationValidator:
    """Verify that conditional responses respect protocol dependencies.

    For every family in a protocol with non-empty
    :attr:`~bead.protocol.family.QuestionFamily.depends_on`, the
    validator checks two things:

    1. *Dependency presence*: each item with a response on the
       conditional question must also have a response on every
       upstream question.
    2. *Dependency value* (optional): when ``conditioning_values`` is
       supplied for the conditional anchor, the upstream response
       must be one of the allowed labels.

    Findings are emitted as :class:`DiagnosticRecord` instances at the
    :attr:`DiagnosticLevel.WARNING` level.

    Parameters
    ----------
    conditioning_values : Mapping[str, set[str]] | None, optional
        Per-conditional-anchor mapping from upstream label set to
        validity. When omitted the validator only checks dependency
        presence. Defaults to ``None``.

    Attributes
    ----------
    conditioning_values : Mapping[str, set[str]]
        Conditioning-value table (immutable view).
    """

    conditioning_values: Mapping[str, set[str]] = field(default_factory=dict)

    def validate(
        self,
        records_by_question: Mapping[str, Sequence[RecordLike]],
        protocol: AnnotationProtocol,
    ) -> tuple[DiagnosticRecord, ...]:
        """Check conditional-observation consistency for a protocol.

        Parameters
        ----------
        records_by_question : Mapping[str, Sequence[record-like]]
            Records grouped by anchor name. Each record must expose
            ``item_id``, ``response_label``, and ``question_name``
            attributes.
        protocol : AnnotationProtocol
            The protocol whose dependency edges drive the validation.

        Returns
        -------
        tuple[DiagnosticRecord, ...]
            Warning-level findings for any inconsistencies detected.
        """
        findings: list[DiagnosticRecord] = []

        response_lookup: dict[str, dict[str, str]] = {}
        for q_name, records in records_by_question.items():
            lookup: dict[str, str] = {}
            for rec in records:
                lookup[rec.item_id] = rec.response_label
            response_lookup[q_name] = lookup

        for family in protocol.families:
            if not family.depends_on:
                continue

            obs_responses = response_lookup.get(family.name, {})

            for item_id in obs_responses:
                for dep_name in family.depends_on:
                    dep_responses = response_lookup.get(dep_name, {})

                    if item_id not in dep_responses:
                        findings.append(
                            DiagnosticRecord(
                                level=DiagnosticLevel.WARNING,
                                category="conditional_missing_dependency",
                                message=(
                                    f"conditional observation "
                                    f"{family.name!r} has response for "
                                    f"item {item_id!r} but conditioning "
                                    f"observation {dep_name!r} has no "
                                    f"response"
                                ),
                                item_id=item_id,
                                question_name=family.name,
                            )
                        )
                        continue

                    if family.name in self.conditioning_values:
                        valid_vals = self.conditioning_values[family.name]
                        dep_label = dep_responses[item_id]
                        if dep_label not in valid_vals:
                            findings.append(
                                DiagnosticRecord(
                                    level=DiagnosticLevel.WARNING,
                                    category="conditional_inapplicable",
                                    message=(
                                        f"conditional observation "
                                        f"{family.name!r} has response for "
                                        f"item {item_id!r} but conditioning "
                                        f"observation {dep_name!r} has value "
                                        f"{dep_label!r} (expected one of "
                                        f"{sorted(valid_vals)})"
                                    ),
                                    item_id=item_id,
                                    question_name=family.name,
                                )
                            )

        return tuple(findings)
