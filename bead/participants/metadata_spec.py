"""Metadata specification for participant attributes.

``FieldSpec`` defines the constraints and display properties for a single
participant metadata field. ``ParticipantMetadataSpec`` is the schema for
the full set of participant fields.
"""

from __future__ import annotations

from typing import TYPE_CHECKING, Literal

import didactic.api as dx

from bead.data.base import BeadBaseModel
from bead.data.range import Range

if TYPE_CHECKING:
    from bead.deployment.jspsych.config import DemographicsConfig


class FieldSpec(BeadBaseModel):
    """Specification for a single metadata field.

    Attributes
    ----------
    name : str
        Field name. Must be a valid Python identifier.
    field_type : Literal["int", "float", "str", "bool"]
        Data type for the field.
    required : bool
        Whether the field must be supplied.
    allowed_values : tuple[str | int | float | bool, ...] | None
        Exhaustive list of allowed values (categorical fields). ``None``
        means any value of the correct type is accepted.
    range : Range[float] | None
        Numeric range constraint (numeric fields). ``Range[int]`` may be
        passed since ``int`` is a subtype of ``float`` for the purpose
        of bound checking.
    label : str | None
        Display label for forms. ``None`` defaults to ``name``
        title-cased with underscores replaced by spaces.
    description : str | None
        Help text for the field.
    """

    name: str
    field_type: Literal["int", "float", "str", "bool"]
    required: bool = False
    allowed_values: tuple[str | int | float | bool, ...] | None = None
    range: dx.Embed[Range[float]] | None = None
    label: str | None = None
    description: str | None = None

    @dx.validates("name")
    def _check_name(self, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Field name cannot be empty")
        stripped = value.strip()
        if not stripped.isidentifier():
            raise ValueError(f"Field name must be valid Python identifier: {stripped}")
        return stripped

    def validate_value(self, value: str | int | float | bool | None) -> bool:
        """Return whether *value* satisfies this spec's constraints."""
        if value is None:
            return not self.required

        expected_type: type | tuple[type, ...]
        if self.field_type == "int":
            expected_type = int
        elif self.field_type == "float":
            expected_type = (int, float)
        elif self.field_type == "str":
            expected_type = str
        else:
            expected_type = bool

        if not isinstance(value, expected_type):
            return False

        if self.allowed_values is not None and value not in self.allowed_values:
            return False

        if self.range is not None and isinstance(value, int | float):
            if not self.range.contains(float(value)):
                return False

        return True

    def get_display_label(self) -> str:
        """Return the display label, falling back to a title-cased name."""
        if self.label:
            return self.label
        return self.name.replace("_", " ").title()


def validate_field_spec(spec: FieldSpec) -> None:
    """Raise ``ValueError`` if *spec*'s constraints contradict its type.

    Validates that ``range`` is only used with numeric types and that every
    value in ``allowed_values`` matches ``field_type``.
    """
    if spec.range is not None and spec.field_type not in ("int", "float"):
        raise ValueError(
            f"range constraint only valid for numeric types, not {spec.field_type}"
        )

    if spec.allowed_values is None:
        return

    expected_type: type | tuple[type, ...]
    if spec.field_type == "int":
        expected_type = int
    elif spec.field_type == "float":
        expected_type = (int, float)
    elif spec.field_type == "str":
        expected_type = str
    else:
        expected_type = bool

    for val in spec.allowed_values:
        if not isinstance(val, expected_type):
            raise ValueError(
                f"allowed_values item {val!r} does not match "
                f"field_type {spec.field_type}"
            )


class ParticipantMetadataSpec(BeadBaseModel):
    """Schema for participant metadata.

    Attributes
    ----------
    name : str
        Spec name (e.g. ``"prolific_demographics"``).
    version : str
        Spec version.
    fields : tuple[FieldSpec, ...]
        Field specifications.
    """

    name: str
    version: str = "1.0.0"
    fields: tuple[dx.Embed[FieldSpec], ...] = ()

    @dx.validates("name")
    def _check_name(self, value: str) -> str:
        if not value or not value.strip():
            raise ValueError("Spec name cannot be empty")
        return value.strip()

    @dx.validates("fields")
    def _check_unique_field_names(
        self, value: tuple[FieldSpec, ...]
    ) -> tuple[FieldSpec, ...]:
        names = [f.name for f in value]
        if len(names) != len(set(names)):
            duplicates = {n for n in names if names.count(n) > 1}
            raise ValueError(f"Duplicate field names: {duplicates}")
        return value

    def get_field(self, name: str) -> FieldSpec | None:
        """Return the field spec named *name*, or ``None``."""
        for field in self.fields:
            if field.name == name:
                return field
        return None

    def get_required_fields(self) -> tuple[FieldSpec, ...]:
        """Return the required field specs."""
        return tuple(f for f in self.fields if f.required)

    def validate_metadata(
        self, metadata: dict[str, str | int | float | bool | None]
    ) -> tuple[bool, list[str]]:
        """Validate *metadata* against this spec.

        Returns ``(is_valid, error_messages)``.
        """
        errors: list[str] = []

        for field in self.get_required_fields():
            if field.name not in metadata or metadata[field.name] is None:
                errors.append(f"Missing required field: {field.name}")

        for key, value in metadata.items():
            field_spec = self.get_field(key)
            if field_spec is None:
                continue
            if not field_spec.validate_value(value):
                range_str = ""
                if field_spec.range is not None:
                    range_str = (
                        f", range=[{field_spec.range.min}, {field_spec.range.max}]"
                    )
                allowed_str = ""
                if field_spec.allowed_values is not None:
                    allowed_str = f", allowed={list(field_spec.allowed_values)}"
                errors.append(
                    f"Invalid value for {key}: {value!r} "
                    f"(expected {field_spec.field_type}{range_str}{allowed_str})"
                )

        return len(errors) == 0, errors

    def to_demographics_config(self) -> DemographicsConfig:
        """Render the spec as a ``DemographicsConfig`` for jsPsych deployment."""
        from bead.deployment.jspsych.config import (  # noqa: PLC0415
            DemographicsConfig,
            DemographicsFieldConfig,
        )

        fields: list[DemographicsFieldConfig] = []
        for field in self.fields:
            form_field_type: Literal["text", "number", "dropdown", "radio", "checkbox"]
            if field.field_type in ("int", "float"):
                form_field_type = "number"
            elif field.field_type == "bool":
                form_field_type = "checkbox"
            elif field.allowed_values is not None:
                form_field_type = "dropdown"
            else:
                form_field_type = "text"

            options: tuple[str, ...] | None = None
            if field.allowed_values is not None:
                options = tuple(str(v) for v in field.allowed_values)

            fields.append(
                DemographicsFieldConfig(
                    name=field.name,
                    field_type=form_field_type,
                    label=field.get_display_label(),
                    required=field.required,
                    options=options,
                    range=field.range,
                    help_text=field.description,
                )
            )

        return DemographicsConfig(
            enabled=True,
            title="Participant Information",
            fields=tuple(fields),
        )
