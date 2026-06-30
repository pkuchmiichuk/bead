"""Validation utilities for data integrity checks.

Provides functions beyond didactic's built-in validation, including
JSONLines-file validation, UUID-reference validation, and provenance-chain
validation.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Self
from uuid import UUID

import didactic.api as dx

from bead.data.metadata import MetadataTracker


class ValidationReport(dx.Model):
    """A frozen report of validation results.

    Attributes
    ----------
    valid : bool
        Overall validation status. Set to ``False`` once any error is added.
    errors : tuple[str, ...]
        Error messages.
    warnings : tuple[str, ...]
        Warning messages.
    object_count : int
        Number of objects validated.

    Examples
    --------
    >>> report = ValidationReport(valid=True)
    >>> report = report.add_error("Invalid field")
    >>> report.valid
    False
    >>> report.has_errors()
    True
    >>> len(report.errors)
    1
    """

    valid: bool
    errors: tuple[str, ...] = ()
    warnings: tuple[str, ...] = ()
    object_count: int = 0

    def add_error(self, message: str) -> Self:
        """Return a new report with *message* appended and ``valid=False``."""
        return self.with_(errors=(*self.errors, message), valid=False)

    def add_warning(self, message: str) -> Self:
        """Return a new report with *message* appended to ``warnings``."""
        return self.with_(warnings=(*self.warnings, message))

    def has_errors(self) -> bool:
        """Return whether the report contains any errors."""
        return len(self.errors) > 0

    def has_warnings(self) -> bool:
        """Return whether the report contains any warnings."""
        return len(self.warnings) > 0


def validate_jsonlines_file(
    path: Path, model_class: type[dx.Model], strict: bool = True
) -> ValidationReport:
    """Validate every line of *path* against *model_class*.

    Parameters
    ----------
    path
        Path to the JSONLines file.
    model_class
        didactic Model class to validate against.
    strict
        If ``True``, return on the first error.

    Returns
    -------
    ValidationReport
        Report containing the collected errors and the count of validated
        records.
    """
    report = ValidationReport(valid=True)
    if not path.exists():
        return report.add_error(f"File not found: {path}")

    try:
        with path.open("r", encoding="utf-8") as f:
            for line_num, line in enumerate(f, start=1):
                line = line.strip()
                if not line:
                    continue
                try:
                    model_class.model_validate_json(line)
                    report = report.with_(object_count=report.object_count + 1)
                except dx.ValidationError as e:
                    report = report.add_error(f"Line {line_num}: {e}")
                    if strict:
                        return report
                except (ValueError, TypeError) as e:
                    report = report.add_error(f"Line {line_num}: parse error - {e}")
                    if strict:
                        return report
    except OSError as e:
        report = report.add_error(f"Failed to read file: {e}")

    return report


def validate_uuid_references(
    objects: Sequence[dx.Model], reference_pool: Mapping[UUID, dx.Model]
) -> ValidationReport:
    """Verify every UUID-typed field in *objects* points into *reference_pool*.

    Supports single ``UUID`` fields and tuple/list-of-UUID fields. The
    object's own ``id`` attribute is excluded from the check.
    """
    report = ValidationReport(valid=True, object_count=len(objects))

    for obj in objects:
        specs = getattr(type(obj), "__field_specs__", None)
        if not specs:
            continue

        for field_name in specs:
            if field_name == "id":
                continue
            try:
                field_value = getattr(obj, field_name)
            except AttributeError:
                continue

            if isinstance(field_value, (list, tuple)):
                items: tuple[object, ...] = tuple(field_value)
                for item in items:
                    if not isinstance(item, UUID):
                        continue
                    if item not in reference_pool:
                        obj_id = getattr(obj, "id", "unknown")
                        report = report.add_error(
                            f"Object {obj_id}: field '{field_name}' "
                            f"references missing UUID {item}"
                        )
            elif isinstance(field_value, UUID) and field_value not in reference_pool:
                obj_id = getattr(obj, "id", "unknown")
                report = report.add_error(
                    f"Object {obj_id}: field '{field_name}' references "
                    f"missing UUID {field_value}"
                )

    return report


def validate_provenance_chain(
    metadata: MetadataTracker, repository: Mapping[UUID, dx.Model]
) -> ValidationReport:
    """Validate every parent reference in *metadata*'s provenance chain."""
    report = ValidationReport(valid=True, object_count=len(metadata.provenance))

    for record in metadata.provenance:
        if record.parent_id not in repository:
            report = report.add_error(
                f"Provenance record references missing parent: {record.parent_id}"
            )
            continue
        parent_obj = repository[record.parent_id]
        actual_type = type(parent_obj).__name__
        if record.parent_type != actual_type:
            report = report.add_error(
                f"Provenance record for {record.parent_id}: "
                f"expected type '{record.parent_type}', got '{actual_type}'"
            )

    return report
